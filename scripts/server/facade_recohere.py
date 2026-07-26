"""GENERATIVE facade re-coherence (the "different approach", 2026-06-14).

Instead of re-cohering a set of free boxes (the PartSetRefiner — unstructured, no alignment
guarantee → confetti), the correction lives in FACADE-PROGRAM space: the trained generative
detail head (`outputs/detail_generator/detail_gen.pth`) samples a 12-dim `DetailParams`
program, and `scene.sdf_detail.add_facade_detail` renders it by IQ domain-repetition →
**every sample is aligned-by-construction** (perfect window rows). Generativity lives where
every point in the space is already coherent architecture, so "generative correction" reads
as "interesting architecture" instead of noise.

Correction = SDEdit on the PROGRAM: fit the current facade to its param vector, partially
noise it (strength), and denoise conditioned on style → a *nearby but coherent* facade
program. strength→0 keeps the current facade; strength→1 is a fresh coherent sample.
Different seeds → different style-appropriate facades (the "interesting" part).

This reuses the proven B+.6 diffusion stack wholesale; nothing new is trained.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from models.networks.recipe_param_diffusion import ConditionalDenoiser, GaussianDiffusion
from models.networks import recipe_param_space as ps
from scene import sdf_detail as det

CK = REPO / "outputs/detail_generator/detail_gen.pth"
_HEAD = None


def load_head(device="cuda"):
    """Lazy-load (denoiser, diffusion, meta). Cached."""
    global _HEAD
    if _HEAD is None:
        ck = torch.load(CK, map_location=device, weights_only=False)
        den = ConditionalDenoiser(cond_dim=len(ps.STYLES), n_params=ck["n_dim"],
                                  hidden=ck["hidden"], depth=ck["depth"]).to(device)
        den.load_state_dict(ck["model"]); den.eval()
        diff = GaussianDiffusion(ck["timesteps"], device=device)
        _HEAD = (den, diff, ck)
    return _HEAD


def _style_cond(style, B, device):
    si = ps.STYLE_TO_IDX.get(style, ps.STYLE_TO_IDX.get("modern", 0))
    c = torch.zeros(B, len(ps.STYLES), device=device); c[:, si] = 1.0
    return c


@torch.no_grad()
def _sdedit_vec(x0_raw, style, strength, device, steps=40, eta=1.0, guidance=1.0):
    """SDEdit on the normalized param vector: q_sample to t0=strength*T, denoise to 0
    conditioned on `style`. x0_raw: (B,12) raw DetailParams vectors. Returns (B,12) raw."""
    den, diff, ck = load_head(device)
    mean = torch.as_tensor(ck["mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(ck["std"], dtype=torch.float32, device=device)
    x0 = (torch.as_tensor(x0_raw, dtype=torch.float32, device=device) - mean) / std
    B = x0.shape[0]
    cond = _style_cond(style, B, device)
    T = diff.T
    t0 = int(np.clip(strength, 1e-3, 1.0) * (T - 1))
    x = diff.q_sample(x0, torch.full((B,), t0, device=device, dtype=torch.long),
                      torch.randn_like(x0))
    ts = torch.linspace(t0, 0, steps, device=device).round().long()
    for i in range(steps):
        t_cur = int(ts[i].item())
        t_b = torch.full((B,), t_cur, device=device, dtype=torch.long)
        if guidance != 1.0:
            eps_c = den(x, t_b, cond, torch.zeros(B, dtype=torch.bool, device=device))
            eps_u = den(x, t_b, cond, torch.ones(B, dtype=torch.bool, device=device))
            eps = eps_u + guidance * (eps_c - eps_u)
        else:
            eps = den(x, t_b, cond, None)
        acp_t = diff.alphas_cumprod[t_cur]
        x0_pred = ((x - (1 - acp_t).sqrt() * eps) / acp_t.sqrt()).clamp(-5.0, 5.0)
        if i + 1 < steps:
            t_next = int(ts[i + 1].item())
            acp_n = diff.alphas_cumprod[t_next]
            sigma = eta * ((1 - acp_n) / (1 - acp_t)).sqrt() * (1 - acp_t / acp_n).sqrt()
            dir_xt = (1 - acp_n - sigma ** 2).clamp_min(0).sqrt() * eps
            x = acp_n.sqrt() * x0_pred + dir_xt + sigma * torch.randn_like(x)
        else:
            x = x0_pred
    raw = (x * std + mean).cpu().numpy()
    return np.clip(raw, det.DETAIL_LO, det.DETAIL_HI).astype(np.float32)


def params_to_vec(p: "det.DetailParams") -> np.ndarray:
    return np.array([getattr(p, f) for f in det.DETAIL_FIELDS], np.float32)


def recohere_facade(cur, style="modern", strength=0.6, seed=None, device="cuda",
                    building_class=None, steps=40, guidance=1.0):
    """GENERATIVE facade correction. `cur` = current DetailParams | 12-vec | None (None →
    a coherent style sample, i.e. strength forced to 1.0). Returns a corrected DetailParams
    whose render is coherent-by-construction. Different seeds → different facades."""
    if seed is not None:
        torch.manual_seed(int(seed)); np.random.seed(int(seed) & 0xFFFFFFFF)
    if cur is None:                                   # nothing to start from → fresh sample
        cur = det.sample_detail_vector(style, np.random.default_rng(seed))
        strength = 1.0
    vec = params_to_vec(cur) if isinstance(cur, det.DetailParams) else np.asarray(cur, np.float32)
    out = _sdedit_vec(vec[None], style, strength, device, steps=steps, guidance=guidance)[0]
    p = det.vector_to_params(out)
    if building_class:                                # class glazing prior (ground floor etc.)
        p = det.ground_glazing(p, building_class)
    return p


# crude "incoherent facade" generator for the demo: ignore the per-style structure and draw
# params uniformly across the full valid range → ugly/off proportions (the corrector's input).
def broken_facade_vec(seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(det.DETAIL_LO, det.DETAIL_HI).astype(np.float32)
