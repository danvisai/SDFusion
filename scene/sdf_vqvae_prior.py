"""Apply the frozen BuildingNet VQVAE as a neural prior over a procedural SDF.

Pipeline:
    procedural SDF callable
      -> sample 64^3 in Frame N (centered, isotropic max-half-extent = 1)
      -> clamp to [-trunc, +trunc] (matches VQVAE training)
      -> VQVAE encode (quantized) -> decode
      -> upsample back if requested
      -> rescale Frame-N units back to world meters
      -> marching cubes on the world-frame grid

The VQVAE was trained on real BuildingNet SDFs at 64^3, T=0.2, in a centered,
unit-sphere-equivalent frame. Encoding a primitive-blocky procedural SDF and
decoding through the frozen codebook smooths/regularizes the field while
preserving the gross footprint shape.

Frame conventions match scene/sdf_primitives.py:
    +y up, +x and +z horizontal, SDF grid index order (D=z, H=y, W=x).
"""
from __future__ import annotations
from types import SimpleNamespace
from typing import Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

from models.model_utils import load_vqvae
from scene.sdf_primitives import grid_to_mesh


VQVAE_CONF = "configs/vqvae_bnet.yaml"
VQVAE_CKPT = "logs_building/2025-05-19T19-58-28-vqvae-building-all-res64-LR1e-4-T0.2-release/ckpt/vqvae_steps-latest.pth"
TRUNC = 0.2  # matches VQVAE training (T0.2 in ckpt name)


def load_buildingnet_vqvae(device: str = "cuda"):
    """Load the frozen BuildingNet VQVAE from the canonical paths."""
    vq_conf = OmegaConf.load(VQVAE_CONF)
    opt = SimpleNamespace(device=torch.device(device))
    return load_vqvae(vq_conf, vq_ckpt=VQVAE_CKPT, opt=opt)


def _frame_n_world_bbox(polygon_xz: np.ndarray, target_height: float,
                        pad: float = 0.10) -> Tuple[float, np.ndarray, Tuple]:
    """Return (scene_half_extent, world_center [3,], world_bbox [6,]) for a
    cube-shaped sampling region that fully contains the building footprint
    plus its roof (target_height * 2.5 vertical headroom).
    """
    poly = np.asarray(polygon_xz, dtype=np.float64)
    x_min, z_min = poly.min(axis=0)
    x_max, z_max = poly.max(axis=0)
    cx = (x_min + x_max) / 2.0
    cz = (z_min + z_max) / 2.0
    w_pad = pad * max(x_max - x_min, z_max - z_min, 1.0)
    half_xz = max(x_max - x_min, z_max - z_min) / 2.0 + w_pad
    half_y = target_height * 1.25 + pad * target_height
    scene_half = float(max(half_xz, half_y))
    center_y = scene_half  # so the building (y in [0, height]) sits inside a [0, 2*scene_half] vertical band
    world_center = np.array([cx, center_y, cz], dtype=np.float64)
    bbox = (
        float(cx - scene_half), 0.0, float(cz - scene_half),
        float(cx + scene_half), float(2.0 * scene_half), float(cz + scene_half),
    )
    return scene_half, world_center, bbox


def sample_sdf_in_frame_n(sdf_fn, polygon_xz, target_height, res: int = 64,
                          device: str = "cuda", chunk: int = 1 << 18):
    """Evaluate `sdf_fn(world_points)` on a (res, res, res) grid covering a
    cube around the building, in Frame-N normalized values.

    Returns:
        sdf_n_grid : Tensor[res, res, res]   (indexed [z, y, x])
        scene_half : float                   world half-extent of the cube
        world_bbox : tuple of 6 floats       (x0, y0, z0, x1, y1, z1) in world meters
    """
    scene_half, world_center, world_bbox = _frame_n_world_bbox(polygon_xz, target_height)
    fn = torch.linspace(-1.0, 1.0, res, device=device, dtype=torch.float32)
    Z, Y, X = torch.meshgrid(fn, fn, fn, indexing="ij")  # (res, res, res) each
    pts_n = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)
    center_t = torch.tensor(world_center, device=device, dtype=torch.float32)
    pts_world = pts_n * scene_half + center_t  # (res^3, 3)
    out_world = torch.empty(pts_world.shape[0], device=device)
    for i in range(0, pts_world.shape[0], chunk):
        out_world[i:i + chunk] = sdf_fn(pts_world[i:i + chunk])
    sdf_world = out_world.reshape(res, res, res)
    sdf_n = sdf_world / scene_half
    return sdf_n, scene_half, world_bbox


def apply_vqvae_prior_to_sdf(sdf_fn, polygon_xz, target_height, vqvae,
                             res: int = 64, trunc: float = TRUNC,
                             device: str = "cuda"):
    """Run procedural SDF through frozen VQVAE encode+decode.

    Returns (sdf_world_grid: Tensor[D,H,W], world_bbox: tuple, scene_half: float).
    """
    sdf_n, scene_half, world_bbox = sample_sdf_in_frame_n(
        sdf_fn, polygon_xz, target_height, res=res, device=device,
    )
    x_in = sdf_n.clamp(-trunc, trunc).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    with torch.no_grad():
        quant, _, _ = vqvae.encode(x_in)
        sdf_n_out = vqvae.decode(quant).squeeze(0).squeeze(0)
    sdf_world_out = sdf_n_out * scene_half
    return sdf_world_out, world_bbox, scene_half


def procedural_to_mesh_via_vqvae(sdf_fn, polygon_xz, target_height, vqvae,
                                 res: int = 64, trunc: float = TRUNC,
                                 device: str = "cuda"):
    """Convenience: SDF -> VQVAE prior -> marching cubes -> trimesh.Trimesh."""
    sdf_world, world_bbox, _ = apply_vqvae_prior_to_sdf(
        sdf_fn, polygon_xz, target_height, vqvae, res=res, trunc=trunc, device=device,
    )
    return grid_to_mesh(sdf_world, world_bbox), world_bbox
