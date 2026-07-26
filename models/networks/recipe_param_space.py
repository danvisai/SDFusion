"""Shared recipe parameter-space utilities for Option B+ generative heads.

This module is the common ground between:
  - B+.5  deterministic param-prediction head   (scripts/train_recipe_param_head.py)
  - B+.6  recipe-parameter diffusion             (future)

It owns the canonical contract between *conditioning* (footprint polygon, height,
class, style — the symbolic OSM inputs) and the per-style *recipe parameter vector*
consumed by `models/networks/diff_recipe.py`.

Design choices (kept deliberately simple so they survive into B+.6):
  - Styles are indexed in the SAME order as DIFF_RECIPE_REGISTRY, which matches the
    `style_id` baked into the B+.4 synthetic extraction (modern=0 ... public_civic=7).
  - All 8 styles have different param counts (3..12). We pad to MAX_PARAMS=12 and carry
    a per-style boolean validity mask so a single tensor can hold any style and loss
    only counts the dims that actually exist for that style.
  - Parameters span wildly different scales (ratios ~0.04 alongside counts ~43), and the
    B+.7 optimizer pushed some to extremes. We standardise per (style, dim) with a z-score
    whose stats are fit from the training data and saved alongside the checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Tuple

import numpy as np

from models.networks.diff_recipe import DIFF_RECIPE_REGISTRY

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

# Registry insertion order == canonical style index == B+.4 synthetic style_id.
STYLES: List[str] = list(DIFF_RECIPE_REGISTRY.keys())
STYLE_TO_IDX: Dict[str, int] = {s: i for i, s in enumerate(STYLES)}
IDX_TO_STYLE: Dict[int, str] = {i: s for i, s in enumerate(STYLES)}
N_STYLES: int = len(STYLES)

# Per-style parameter dimensionality, read straight from the registry (n_params).
STYLE_DIMS: Dict[str, int] = {s: DIFF_RECIPE_REGISTRY[s][2] for s in STYLES}
MAX_PARAMS: int = max(STYLE_DIMS.values())  # 12 (victorian)

# Sanity: the synthetic extraction assumed modern=0, victorian=2, mediterranean=5.
assert STYLE_TO_IDX["modern"] == 0 and STYLE_TO_IDX["victorian"] == 2 \
    and STYLE_TO_IDX["mediterranean"] == 5, "style index drifted from B+.4 contract"


def pad_params(params: np.ndarray, style: str) -> np.ndarray:
    """(n,) raw params for `style` -> (MAX_PARAMS,) zero-padded."""
    n = STYLE_DIMS[style]
    out = np.zeros(MAX_PARAMS, dtype=np.float32)
    out[:n] = np.asarray(params, dtype=np.float32)[:n]
    return out


def param_mask(style: str) -> np.ndarray:
    """(MAX_PARAMS,) bool mask — True for dims that exist in `style`."""
    m = np.zeros(MAX_PARAMS, dtype=bool)
    m[: STYLE_DIMS[style]] = True
    return m


def unpad_params(padded: np.ndarray, style: str) -> np.ndarray:
    """(MAX_PARAMS,) -> (n,) trimmed back to the style's real dimensionality."""
    return np.asarray(padded, dtype=np.float32)[: STYLE_DIMS[style]]


# ---------------------------------------------------------------------------
# BuildingNet top-level classes (from asset-id prefix)
# ---------------------------------------------------------------------------

# The 4 top-level prefixes present in the B+.7 fits.
CLASSES: List[str] = ["COMMERCIAL", "PUBLIC", "RELIGIOUS", "RESIDENTIAL"]
CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CLASSES)}
N_CLASSES: int = len(CLASSES)


def class_of(asset_id: str) -> int:
    """Map a BuildingNet asset id (e.g. 'RESIDENTIALhouse_mesh0798') to a class index.

    Unknown prefixes fall back to RESIDENTIAL (the dominant class) so the featurizer
    never crashes on an out-of-vocabulary id.
    """
    m = re.match(r"^([A-Z_]+?)(?=[a-z])", asset_id)
    prefix = m.group(1) if m else asset_id.split("_")[0]
    return CLASS_TO_IDX.get(prefix, CLASS_TO_IDX["RESIDENTIAL"])


# ---------------------------------------------------------------------------
# Conditioning featurizer
# ---------------------------------------------------------------------------

# Footprint polygons from the fitter are always 16 verts with the 16th == 1st
# (a closed ring). We drop the closing duplicate -> 15 unique points.
POLY_VERTS: int = 15
_POLY_FEATS = POLY_VERTS * 2          # 30  normalized outline coords
# SCALE-INVARIANT scalars only. Real BuildingNet fits live in normalized Frame-N
# (~unit), but the synthetic recipe-aug data lives in world METERS (5-35m). Absolute
# w/d/area/height would split the two domains into disjoint feature clusters. Since the
# recipe params are themselves ratios applied to the polygon+height passed to the recipe
# at forward time, conditioning only needs *shape and proportion*, which is frame-free.
_SCALAR_FEATS = 4                     # aspect, fill_ratio, compactness, slenderness
COND_DIM: int = _POLY_FEATS + _SCALAR_FEATS + N_CLASSES + N_STYLES  # 30+4+4+8 = 46

# Index ranges within the conditioning vector.
_POLY_SLICE = slice(0, _POLY_FEATS)
_SCALAR_SLICE = slice(_POLY_FEATS, _POLY_FEATS + _SCALAR_FEATS)
_ONEHOT_START = _POLY_FEATS + _SCALAR_FEATS  # class + style one-hots live here onward
# Continuous dims (everything before the one-hots) get z-scored; one-hots stay 0/1.
CONTINUOUS_DIM: int = _ONEHOT_START  # 34
# Index of the `slenderness` (height/sqrt(area)) scalar within the cond vector. It is the
# only height-bearing feature; the B+.6h height-generation experiment zeros it in the
# conditioning and instead GENERATES it (so the model produces height -> diversity).
SLENDERNESS_FEAT_IDX: int = _POLY_FEATS + 3  # 33


def _resample_closed_polygon(poly: np.ndarray, n: int) -> np.ndarray:
    """Resample a (P,2) polygon outline to `n` arc-length-even points (open ring)."""
    poly = np.asarray(poly, dtype=np.float64)
    # Drop a trailing closing vertex if present.
    if len(poly) >= 2 and np.allclose(poly[0], poly[-1]):
        poly = poly[:-1]
    if len(poly) == n:
        return poly
    # Cumulative arc length around the closed loop.
    closed = np.vstack([poly, poly[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1] if cum[-1] > 1e-9 else 1.0
    targets = np.linspace(0.0, total, n, endpoint=False)
    out = np.empty((n, 2), dtype=np.float64)
    for i, t in enumerate(targets):
        j = int(np.searchsorted(cum, t, side="right") - 1)
        j = min(max(j, 0), len(seg) - 1)
        denom = seg[j] if seg[j] > 1e-9 else 1.0
        frac = (t - cum[j]) / denom
        out[i] = closed[j] + frac * (closed[j + 1] - closed[j])
    return out


def _polygon_scalars(poly: np.ndarray) -> Tuple[float, float, float, float, float]:
    """Return (width, depth, aspect, area, perimeter) for a polygon outline."""
    poly = np.asarray(poly, dtype=np.float64)
    if len(poly) >= 2 and np.allclose(poly[0], poly[-1]):
        poly = poly[:-1]
    x, z = poly[:, 0], poly[:, 1]
    w = float(x.max() - x.min())
    d = float(z.max() - z.min())
    aspect = max(w, d) / max(min(w, d), 1e-6)
    # Shoelace area (absolute).
    area = 0.5 * abs(float(np.sum(x * np.roll(z, -1) - np.roll(x, -1) * z)))
    closed = np.vstack([poly, poly[:1]])
    perim = float(np.sum(np.linalg.norm(np.diff(closed, axis=0), axis=1)))
    return w, d, aspect, area, perim


def raw_conditioning(polygon: np.ndarray, height: float,
                     class_idx: int, style_idx: int) -> np.ndarray:
    """Build the un-standardised (COND_DIM,) conditioning vector for one building.

    Layout: [ normalized-outline (30) | aspect, fill_ratio, compactness, slenderness (4)
              | class one-hot (4) | style one-hot (8) ].

    Everything here is scale-invariant so a normalized-Frame-N real fit and a
    world-meter synthetic sample with the same *shape and proportions* map to the
    same vector:
      - outline      : centred on centroid, scaled to unit max-extent
      - aspect       : max(w,d)/min(w,d)
      - fill_ratio   : area / (w*d)            (how rectangular; in (0,1])
      - compactness  : perimeter / sqrt(area)  (4 for a square, larger for ornate)
      - slenderness  : height / sqrt(area)     (ties height to footprint scale)
    """
    poly = np.asarray(polygon, dtype=np.float64)
    rs = _resample_closed_polygon(poly, POLY_VERTS)
    centroid = rs.mean(axis=0)
    centred = rs - centroid
    scale = float(np.abs(centred).max())
    norm_outline = (centred / scale if scale > 1e-9 else centred).reshape(-1)  # (30,)

    w, d, aspect, area, perim = _polygon_scalars(poly)
    sqrt_area = float(np.sqrt(max(area, 1e-9)))
    fill_ratio = area / max(w * d, 1e-9)
    compactness = perim / sqrt_area
    slenderness = float(height) / sqrt_area
    scalars = np.array([aspect, fill_ratio, compactness, slenderness], dtype=np.float64)

    class_oh = np.zeros(N_CLASSES, dtype=np.float64)
    class_oh[class_idx] = 1.0
    style_oh = np.zeros(N_STYLES, dtype=np.float64)
    style_oh[style_idx] = 1.0

    return np.concatenate([norm_outline, scalars, class_oh, style_oh]).astype(np.float32)


# ---------------------------------------------------------------------------
# Standardisers (saved alongside the checkpoint)
# ---------------------------------------------------------------------------

@dataclass
class FeatureScaler:
    """Z-scores the continuous part of the conditioning vector; leaves one-hots alone."""
    mean: np.ndarray  # (CONTINUOUS_DIM,)
    std: np.ndarray   # (CONTINUOUS_DIM,)

    @classmethod
    def fit(cls, cond: np.ndarray, std_floor: float = 1e-3) -> "FeatureScaler":
        cont = cond[:, :CONTINUOUS_DIM]
        mean = cont.mean(axis=0)
        std = np.maximum(cont.std(axis=0), std_floor)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, cond: np.ndarray) -> np.ndarray:
        out = np.asarray(cond, dtype=np.float32).copy()
        out[:, :CONTINUOUS_DIM] = (out[:, :CONTINUOUS_DIM] - self.mean) / self.std
        return out


@dataclass
class ParamNormalizer:
    """Per-(style, dim) z-score for the padded param target.

    mean/std have shape (N_STYLES, MAX_PARAMS). Invalid (padded) dims get mean=0,
    std=1 so they map to 0 and are masked out of the loss anyway.
    """
    mean: np.ndarray  # (N_STYLES, MAX_PARAMS)
    std: np.ndarray   # (N_STYLES, MAX_PARAMS)

    @classmethod
    def fit(cls, padded: np.ndarray, style_idx: np.ndarray,
            std_floor: float = 1e-3) -> "ParamNormalizer":
        mean = np.zeros((N_STYLES, MAX_PARAMS), dtype=np.float32)
        std = np.ones((N_STYLES, MAX_PARAMS), dtype=np.float32)
        for s in range(N_STYLES):
            sel = style_idx == s
            if not np.any(sel):
                continue
            n = STYLE_DIMS[IDX_TO_STYLE[s]]
            block = padded[sel][:, :n]
            mean[s, :n] = block.mean(axis=0)
            std[s, :n] = np.maximum(block.std(axis=0), std_floor)
        return cls(mean=mean, std=std)

    def transform(self, padded: np.ndarray, style_idx: np.ndarray) -> np.ndarray:
        return ((padded - self.mean[style_idx]) / self.std[style_idx]).astype(np.float32)

    def inverse(self, normed: np.ndarray, style_idx: np.ndarray) -> np.ndarray:
        return (normed * self.std[style_idx] + self.mean[style_idx]).astype(np.float32)


def fit_param_normalizer_with_jitter(padded: np.ndarray, style_idx: np.ndarray,
                                     jitter_frac: float = 0.1,
                                     jitter_abs_floor: float = 0.05,
                                     zero_var_thresh: float = 1e-4,
                                     std_floor: float = 1e-3):
    """Like ParamNormalizer.fit, but prepares for training-time jitter of the
    zero-variance styles (victorian/industrial/mediterranean/public_civic and any
    other constant (style,dim)).

    The B+.4 synthetic params are identical across all samples for several styles, so
    those (style, dim) cells are delta distributions a diffusion model can only memorise
    as a point. We give them a controlled spread:

      - flagged cell := valid dim whose data std < zero_var_thresh
      - ref_scale    := max(jitter_frac * |mean|, jitter_abs_floor)
      - normalizer.std[flagged] := ref_scale   (so the cell's data, all == mean, maps to
        ~0 in normalized space, and adding UNIT Gaussian noise in normalized space is
        exactly a `ref_scale` raw-space jitter)

    Returns (ParamNormalizer, jitter_mask) where jitter_mask is (N_STYLES, MAX_PARAMS)
    float {0,1} marking the cells to add unit normalized noise to during training.
    Non-flagged dims keep the ordinary z-score and get no jitter.
    """
    mean = np.zeros((N_STYLES, MAX_PARAMS), dtype=np.float32)
    std = np.ones((N_STYLES, MAX_PARAMS), dtype=np.float32)
    jitter_mask = np.zeros((N_STYLES, MAX_PARAMS), dtype=np.float32)
    for s in range(N_STYLES):
        sel = style_idx == s
        if not np.any(sel):
            continue
        n = STYLE_DIMS[IDX_TO_STYLE[s]]
        block = padded[sel][:, :n]
        m = block.mean(axis=0)
        sd = block.std(axis=0)
        mean[s, :n] = m
        std[s, :n] = np.maximum(sd, std_floor)
        if jitter_frac > 0:
            flagged = sd < zero_var_thresh
            ref = np.maximum(jitter_frac * np.abs(m), jitter_abs_floor)
            std[s, :n] = np.where(flagged, ref.astype(np.float32), std[s, :n])
            jitter_mask[s, :n] = flagged.astype(np.float32)
    return ParamNormalizer(mean=mean, std=std), jitter_mask


def save_scalers(path: str, feat: FeatureScaler, pnorm: ParamNormalizer) -> None:
    np.savez(path,
             feat_mean=feat.mean, feat_std=feat.std,
             param_mean=pnorm.mean, param_std=pnorm.std)


def load_scalers(path: str) -> Tuple[FeatureScaler, ParamNormalizer]:
    d = np.load(path)
    return (FeatureScaler(mean=d["feat_mean"], std=d["feat_std"]),
            ParamNormalizer(mean=d["param_mean"], std=d["param_std"]))
