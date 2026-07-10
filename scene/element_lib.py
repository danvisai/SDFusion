"""Element library access for the `element` EditOp kind (Phase R3 of
GENERATIVE_MAKE_IT_ARCHITECTURE_BUILD_SPEC_2026-07-08).

The library (data/element_library_v1, built by scripts/foundations/build_element_library.py)
is 3,204 real BuildingNet component instances as 48^3 SDF crops, normalized to [-1,1] with
aspect PRESERVED (max half-extent = 1; the element's true per-axis footprint inside the
crop is `crop_half` <= 1, recovered from meta ext_rel ratios).

`element_sdf(lib_id, half_target, device)` returns an origin-centered SDF callable that
STRETCHES the element to exactly fill a box of the given half-extents (the placed
primitive's box), trilinearly sampling the crop; values are rescaled by the smallest axis
scale (the standard conservative correction under non-uniform scaling) and points beyond
the crop get the clamped value plus the world-space overshoot distance, so CSG composition
stays sane away from the element.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
LIB = REPO / "data/element_library_v1"

_meta = None
_crops = None            # np.memmap (N,48,48,48) float16
_cache = {}              # (lib_id, device) -> (tensor crop, tensor crop_half)


def meta():
    global _meta
    if _meta is None:
        with open(LIB / "meta.json") as f:
            _meta = json.load(f)
    return _meta


def n_elements():
    return len(meta())


def _crops_mm():
    global _crops
    if _crops is None:
        _crops = np.load(LIB / "elements_f16.npy", mmap_mode="r")
    return _crops


def crop_half(lib_id):
    """The element's per-axis half-extent in crop units (max axis = 1.0)."""
    e = np.asarray(meta()[lib_id]["ext_rel"], np.float32)
    return e / max(float(e.max()), 1e-6)


def get_element(lib_id: int, device="cpu"):
    key = (int(lib_id), str(device))
    if key not in _cache:
        crop = torch.as_tensor(np.asarray(_crops_mm()[lib_id], np.float32), device=device)
        ch = torch.as_tensor(crop_half(lib_id), device=device)
        _cache[key] = (crop, ch)
    return _cache[key]


def element_sdf(lib_id: int, half_target, device="cpu"):
    """Origin-centered SDF of library element `lib_id` stretched to fill a box of
    `half_target` (3,) half-extents. Rotation/translation are applied by the caller
    (scene/sdf_edit._primitive), like every other primitive."""
    crop, ch = get_element(lib_id, device)
    vol = crop[None, None]                                     # (1,1,D=z,H=y,W=x)
    ht = torch.as_tensor(np.asarray(half_target, np.float32), device=device)
    axis_scale = ht / ch.clamp(min=1e-6)                       # world units per crop unit
    val_scale = float(axis_scale.min().clamp(min=1e-6))

    def f(p: torch.Tensor) -> torch.Tensor:
        q = p / ht.clamp(min=1e-6) * ch                        # element fills the box
        qc = q.clamp(-1.0, 1.0)
        g = qc.view(1, 1, 1, -1, 3)                            # grid_sample wants (x,y,z)
        v = F.grid_sample(vol, g, mode="bilinear", align_corners=True,
                          padding_mode="border").view(-1) * val_scale
        over = ((q - qc) * axis_scale).norm(dim=-1)            # world-space overshoot
        return v + over
    return f
