"""Composer-driven detail — THE GLUE.

The part-composer (trained on real BuildingNet part layouts) decides WHICH architectural
elements a building gets and how they're arranged — glazing density, roof type, dome /
towers / steps — from the building's massing (class + footprint aspect/slenderness/fill).
`scene/sdf_detail` then INSTANTIATES those elements as solid SDF primitives.

So element placement is LEARNED from real buildings (not hand-sampled), realizing the
"AI understands a set of doors/windows/roofs and composes a sensible building" idea:

    massing (class, footprint, height)
        -> PartComposer.sample_layout  (p(parts | massing), learned)
        -> layout_to_decisions         (glazing, roof_shape, dome, n_towers, steps)
        -> sdf_detail: facade windows + DOOR + roof + landmarks
        -> a coherent building made of understood elements, adapted to the input.

Composer ckpt: outputs/part_composer/part_composer.pth (scripts/train_part_composer.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from models.networks.recipe_param_diffusion import ConditionalDenoiser, GaussianDiffusion
from scene import sdf_detail as det

CKPT = REPO / "outputs/part_composer/part_composer.pth"
CLASSES = ["COMMERCIAL", "PUBLIC", "RELIGIOUS", "RESIDENTIAL"]
_CACHE = {}


class PartComposer:
    """Loads the trained part-composition diffusion and samples a part layout for a massing."""

    def __init__(self, ckpt=CKPT, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(ckpt, map_location=self.device, weights_only=False)
        self.den = ConditionalDenoiser(cond_dim=ck["cond_dim"], n_params=ck["n_params"],
                                       hidden=ck["hidden"], depth=ck["depth"]).to(self.device)
        self.den.load_state_dict(ck["model"]); self.den.eval()
        self.diff = GaussianDiffusion(ck["timesteps"], device=self.device)
        self.cmean = np.asarray(ck["cmean"], np.float64); self.cstd = np.asarray(ck["cstd"], np.float64)
        self.lmean = np.asarray(ck["lmean"], np.float64); self.lstd = np.asarray(ck["lstd"], np.float64)
        self.cont = list(ck["cont_cond"]); self.lay_names = list(ck["layout_names"])
        self.classes = list(ck["classes"]); self.cond_dim = int(ck["cond_dim"])
        self.n_params = int(ck["n_params"])

    def _cond(self, building_class, footprint, height):
        poly = np.asarray(footprint, np.float64)
        ex, ez = float(np.ptp(poly[:, 0])), float(np.ptp(poly[:, 1]))
        bbox_area = max(ex * ez, 1e-6)
        aspect = max(ex, ez) / max(min(ex, ez), 1e-3)
        slender = float(height) / max(np.sqrt(bbox_area), 1e-3)
        x, z = poly[:, 0], poly[:, 1]
        parea = 0.5 * abs(np.dot(x, np.roll(z, 1)) - np.dot(z, np.roll(x, 1)))
        fill = float(np.clip(parea / bbox_area, 0.0, 1.0))
        cls = building_class.upper()
        ci = self.classes.index(cls) if cls in self.classes else 3
        c = np.zeros(self.cond_dim, np.float32); c[ci] = 1.0
        c[self.cont] = ((np.array([aspect, slender, fill]) - self.cmean) / self.cstd).astype(np.float32)
        return torch.tensor(c[None], device=self.device)

    @torch.no_grad()
    def sample_layout(self, building_class, footprint, height, seed=None, steps=50, eta=1.0) -> dict:
        if seed is not None:
            torch.manual_seed(int(seed))
        c = self._cond(building_class, footprint, height)
        x = self.diff.ddim_sample(self.den, c, n_params=self.n_params, steps=steps, eta=eta)
        raw = x.cpu().numpy()[0] * self.lstd + self.lmean
        return {n: float(v) for n, v in zip(self.lay_names, raw)}


def get_composer(device=None) -> PartComposer:
    key = device or "default"
    if key not in _CACHE:
        _CACHE[key] = PartComposer(device=device)
    return _CACHE[key]


def layout_to_decisions(layout: dict, footprint) -> dict:
    """Map the composer's 14-dim layout -> concrete sdf_detail decisions."""
    ext = np.ptp(np.asarray(footprint, np.float64), 0)
    elong = max(ext) / max(min(ext), 1e-3)
    glazing = float(np.clip(layout["glazing"], 0.02, 0.55))
    # roof_flat = mean |roof-normal.y|: ~1 flat, lower = pitched
    if layout["roof_flat"] < 0.55:
        roof_shape = "gabled" if elong > 1.4 else "hipped"
    else:
        roof_shape = "flat"
    return {
        "glazing": glazing,
        "roof_shape": roof_shape,
        "dome": layout["has_dome"] > 0.5,
        "dome_r": float(np.clip(layout["dome_r"], 0.15, 0.5)),
        "n_towers": int(np.clip(round(layout["n_towers"] * 4), 0, 4)),
        "tower_h_ratio": float(np.clip(1.0 + layout["tower_h"] * 0.9, 1.1, 1.95)),
        "steps": layout["has_steps"] > 0.5,
    }


def _glaze_to_target(p: det.DetailParams, target: float) -> det.DetailParams:
    """Scale window area so facade glazing matches the composer's predicted ratio."""
    cov = (p.win_w * p.win_h) / max(p.win_spacing * p.floor_h, 1e-6)
    s = float(np.clip((target / max(cov, 1e-4)) ** 0.5, 0.45, 1.9))
    p.win_w = float(np.clip(p.win_w * s, det.DETAIL_LO[2], det.DETAIL_HI[2]))
    p.win_h = float(np.clip(p.win_h * s, det.DETAIL_LO[1], det.DETAIL_HI[1]))
    return p


def compose_detail(base_sdf, footprint, height, building_class, style="modern",
                   seed=None, composer=None, door=True, add_ops=None):
    """Run the composer for this massing, then instantiate the chosen elements onto base_sdf.
    `add_ops`: user-ADDED primitives (EditOp dicts, mode='add'), WORLD-meter frame matching
    base_sdf/footprint/height (Y from the building's ground) — each gets its OWN detail via
    det.add_element_detail (classified tower/balcony/etc), layered on AFTER the whole-building
    treatment so a bare CSG-added box doesn't stay bare (2026-07-02 gap fix).
    Returns (composed_sdf, layout_dict, decisions_dict)."""
    composer = composer or get_composer()
    layout = composer.sample_layout(building_class, footprint, height, seed=seed)
    dec = layout_to_decisions(layout, footprint)

    rng = np.random.default_rng(0 if seed is None else int(seed))
    p = det.vector_to_params(det.sample_detail_vector(style, rng))
    p = _glaze_to_target(p, dec["glazing"])
    # expose the FINAL effective facade params (post-glaze) — detailizer v2 conditions on
    # these so coarse->fine becomes deterministic given the layout (composer decides,
    # detailizer renders)
    dec["detail_vec"] = [float(getattr(p, f)) for f in det.DETAIL_FIELDS]

    sdf = det.add_facade_detail(base_sdf, footprint, height, p)
    if door:
        sdf = det.add_door(sdf, footprint, height)
    sdf = det.apply_roof_shape(sdf, footprint, height, dec["roof_shape"])
    if dec["dome"] or dec["n_towers"] or dec["steps"]:
        sdf = det.add_landmarks(sdf, footprint, height, dome=dec["dome"],
                                n_towers=dec["n_towers"], tower_h_ratio=dec["tower_h_ratio"],
                                steps=dec["steps"])
    if add_ops:
        body_bbox = det._bbox(footprint, height)
        for i, op in enumerate(add_ops):
            op_seed = None if seed is None else int(seed) + 1000 + i
            sdf = det.add_element_detail(sdf, op, body_bbox, style=style, seed=op_seed)
    return sdf, layout, dec
