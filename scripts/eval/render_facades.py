"""Neutral-shader facade rendering for the FID harness (ticket 05).

Renders a cube-frame `[-1,1]³` SDF grid to geometry-only, normal-shaded images via the existing SDF
sphere-tracer (`scripts/appearance/gbuffer_neural_render.sphere_trace`). Identical camera + shader for
the real, monolith, and decomposition arms — representation parity: every arm is an SDF grid traced
the same way, so apparent detail differences come from geometry, not appearance (ADR 0002/0004). No
texture, no lighting variability; cameras are a deterministic fixed orbit.

Real BuildingNet SDFs load from `resolution_64/<id>/ori_sample_grid.h5` (`pc_sdf_sample` is the 64³
field flattened, 262144 = 64³) — a LOWER voxel density than the locked WORKING_RES=96 (ADR 0004).
`load_buildingnet_sdf` resamples to WORKING_RES by default so the real arm is never compared to the
monolith/decomposition arms at unequal sampling density (mixed-resolution comparisons are prohibited).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "appearance"))  # for gbuffer_neural_render.sphere_trace

DEFAULT_FOV = 38.0
WORKING_RES = 96  # ADR 0004: locked shared working resolution for every arm


def orbit_cameras(n_views=6, elev_deg=12.0, radius=2.6, fov=DEFAULT_FOV, look=(0.0, -0.05, 0.0)):
    """Deterministic ring of camera poses (azimuth evenly spaced) around the building."""
    e = math.radians(elev_deg)
    cams = []
    for k in range(n_views):
        az = 2.0 * math.pi * k / n_views
        cp = (radius * math.cos(e) * math.sin(az),
              radius * math.sin(e),
              radius * math.cos(e) * math.cos(az))
        cams.append(dict(cam_pos=cp, look=tuple(look), fov=fov))
    return cams


def render_sdf_neutral(grid, cameras=None, res=256, device="cuda"):
    """List of (res,res,3) uint8 normal-shaded renders, one per camera. Deterministic."""
    from gbuffer_neural_render import sphere_trace
    if cameras is None:
        cameras = orbit_cameras()
    imgs = []
    for cam in cameras:
        _depth, nrm, _mask = sphere_trace(grid, res=res, cam_pos=cam["cam_pos"],
                                          look=cam["look"], fov=cam["fov"], device=device)
        imgs.append((np.asarray(nrm) * 255.0).clip(0, 255).astype(np.uint8))
    return imgs


def resample_sdf_grid(grid, out_res, device="cuda"):
    """Trilinearly resample a cube-frame `[-1,1]³` SDF grid to `out_res³`.

    `align_corners=True` to match `sphere_trace`'s own `grid_sample` convention (voxel 0 <-> cube
    corner -1, voxel res-1 <-> +1), so resampling and tracing agree on where a voxel sits.
    """
    import torch
    import torch.nn.functional as F
    if grid.shape[0] == out_res:
        return grid
    g = torch.as_tensor(grid, dtype=torch.float32, device=device)[None, None]
    out = F.interpolate(g, size=(out_res, out_res, out_res), mode="trilinear", align_corners=True)
    return out[0, 0].cpu().numpy()


def load_buildingnet_sdf(building_id, native_res=64, working_res=WORKING_RES, device="cuda"):
    """Load a real BuildingNet building's native `native_res³` SDF and resample it to
    `working_res³` (ADR 0004's locked shared resolution) via `resample_sdf_grid`. Pass
    `working_res=native_res` to get the raw, un-resampled 64³ field."""
    import h5py
    p = REPO / "data/BuildingNet_dataset_v0_1/resolution_64" / building_id / "ori_sample_grid.h5"
    with h5py.File(p, "r") as f:
        g = np.asarray(f["pc_sdf_sample"]).reshape(native_res, native_res, native_res).astype(np.float32)
    return resample_sdf_grid(g, working_res, device=device)
