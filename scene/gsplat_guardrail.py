"""Footprint/SDF guardrail at inference time for placed Gaussians.

Drop Gaussians whose (x, z) world position lies outside the polygon footprint.
Optionally clamp y to the building's height extent.

This is the inference-time form of the SDF/footprint guardrail. The training-
time form (a differentiable footprint BCE loss) lives in
models/networks/gsplat_to_voxel.py for Stage 3.
"""
from __future__ import annotations
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageDraw

from scene.gsplat_common import GaussianSet


def rasterize_polygon_xz(
    polygon_xz: np.ndarray, resolution: int = 256, dilate_px: int = 0,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Rasterize a world-space XZ polygon to a binary mask.

    Returns:
        mask : (H, W) uint8 with 1 inside polygon, 0 outside.
        bbox : (x_min, z_min, x_max, z_max) in world coords.
    """
    poly = np.asarray(polygon_xz, dtype=np.float64)
    x_min, z_min = poly.min(axis=0)
    x_max, z_max = poly.max(axis=0)
    pw = max(x_max - x_min, 1e-6)
    pd = max(z_max - z_min, 1e-6)
    H = W = int(resolution)
    # Map each polygon vertex to (col, row).
    cols = (poly[:, 0] - x_min) / pw * (W - 1)
    rows = (poly[:, 1] - z_min) / pd * (H - 1)
    img = Image.new("L", (W, H), 0)
    ImageDraw.Draw(img).polygon(
        list(zip(cols.tolist(), rows.tolist())), outline=1, fill=1,
    )
    mask = np.asarray(img, dtype=np.uint8)
    if dilate_px > 0:
        # Simple max-pool dilation; avoids a scipy dependency.
        k = dilate_px
        m = mask.astype(np.uint8)
        out = m.copy()
        for _ in range(k):
            shifted = np.zeros_like(m)
            shifted[1:, :] = np.maximum(shifted[1:, :], m[:-1, :])
            shifted[:-1, :] = np.maximum(shifted[:-1, :], m[1:, :])
            shifted[:, 1:] = np.maximum(shifted[:, 1:], m[:, :-1])
            shifted[:, :-1] = np.maximum(shifted[:, :-1], m[:, 1:])
            out = np.maximum(out, shifted)
            m = out
        mask = out
    return mask, (float(x_min), float(z_min), float(x_max), float(z_max))


def cull_outside_footprint(
    g: GaussianSet,
    polygon_xz: np.ndarray,
    target_height: Optional[float] = None,
    ground_y: float = 0.0,
    mask_resolution: int = 256,
    dilate_px: int = 1,
    y_margin: float = 0.05,
) -> GaussianSet:
    """Drop Gaussians whose (x, z) projects outside the polygon mask, and
    (optionally) whose y is outside [ground_y - y_margin, ground_y + target_height + y_margin].

    `dilate_px` adds a 1-pixel safety dilation so Gaussians right on the
    boundary are kept. `y_margin` is a fractional margin of the target height
    (e.g. 0.05 = 5%) added above and below before clamping.
    """
    mask, (x_min, z_min, x_max, z_max) = rasterize_polygon_xz(
        polygon_xz, mask_resolution, dilate_px,
    )
    H, W = mask.shape
    pw = max(x_max - x_min, 1e-6)
    pd = max(z_max - z_min, 1e-6)

    means_np = g.means.detach().cpu().numpy()
    x = means_np[:, 0]
    z = means_np[:, 2]
    cols = np.clip(((x - x_min) / pw * (W - 1)).round().astype(np.int64), 0, W - 1)
    rows = np.clip(((z - z_min) / pd * (H - 1)).round().astype(np.int64), 0, H - 1)
    inside_xz = mask[rows, cols] > 0  # (N,)

    keep = inside_xz
    if target_height is not None:
        y = means_np[:, 1]
        margin = y_margin * target_height
        keep_y = (y >= ground_y - margin) & (y <= ground_y + target_height + margin)
        keep = keep & keep_y

    keep_t = torch.from_numpy(keep).to(g.means.device)
    return g.filter(keep_t)
