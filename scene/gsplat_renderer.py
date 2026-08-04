"""Render a GaussianSet via gsplat in a pytorch3d-compatible camera convention.

Two entry points:
    render_gsplat_view(g, R_p3d, T_p3d, fov_deg, image_size) -> PIL.Image
        pytorch3d-style camera (R, T from look_at_view_transform). Used as a
        drop-in for existing pytorch3d renders in osm_pipeline_map_choices.py.

    render_gsplat_topdown(g, image_size, world_bbox, bg) -> PIL.Image
        Top-down ortho-style render for the per-tile output map. Cast a
        large-FoV perspective from far above, looking straight down. Cheap
        and matches how the existing top-down map sheets work.

Camera convention:
    pytorch3d : +x LEFT, +y up, +z into screen.
    OpenCV / gsplat : +x right, +y down, +z forward (into screen).
    Conversion : left-multiply col-vec viewmat by diag(-1, -1, 1, 1).
"""
from __future__ import annotations
import math
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

from scene.gsplat_common import GaussianSet


def pytorch3d_RT_to_gsplat(
    R_p3d: torch.Tensor, T_p3d: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Convert pytorch3d (R, T) to gsplat-compatible world-to-camera matrices.

    pytorch3d uses row-vector convention internally with axes (+x left, +y up,
    +z into screen). gsplat expects column-vector OpenCV (+x right, +y down,
    +z into screen). Steps:
        1) row-vec -> col-vec by transposing the rotation.
        2) flip x and y via left-multiplied diag(-1, -1, 1, 1).
    """
    R_p3d = R_p3d.to(device)
    T_p3d = T_p3d.to(device)
    V = R_p3d.shape[0]
    viewmat = torch.eye(4, device=device).unsqueeze(0).repeat(V, 1, 1)
    viewmat[:, :3, :3] = R_p3d.transpose(-1, -2)
    viewmat[:, :3, 3] = T_p3d
    flip = torch.diag(
        torch.tensor([-1.0, -1.0, 1.0, 1.0], device=device)
    ).unsqueeze(0)
    return flip @ viewmat


def perspective_K(width: int, height: int, fov_deg: float, device) -> torch.Tensor:
    fy = height / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    fx = fy  # assume square pixels with W ~ H
    cx, cy = width / 2.0, height / 2.0
    K = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        device=device, dtype=torch.float32,
    )
    return K.unsqueeze(0)  # (1, 3, 3)


def _rasterize(
    g: GaussianSet,
    viewmats: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    bg: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    from gsplat import rasterization

    device = g.means.device
    bg_t = torch.tensor(bg, device=device, dtype=torch.float32)
    with torch.no_grad():
        out, _alpha, _meta = rasterization(
            means=g.means,
            quats=g.activated_quats(),
            scales=g.activated_scales(),
            opacities=g.activated_opac(),
            colors=g.activated_colors(),
            viewmats=viewmats,
            Ks=K,
            width=width,
            height=height,
            packed=False,
            render_mode="RGB",
            backgrounds=bg_t.unsqueeze(0).expand(viewmats.shape[0], 3).contiguous(),
        )
    img = out[0].clamp(0, 1).cpu().numpy()
    return (img * 255).astype(np.uint8)


def render_gsplat_view(
    g: GaussianSet,
    R_p3d: torch.Tensor,
    T_p3d: torch.Tensor,
    fov_deg: float = 30.0,
    image_size: int = 512,
    bg: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Image.Image:
    """Render at the pytorch3d-style camera given by (R, T)."""
    device = g.means.device
    viewmats = pytorch3d_RT_to_gsplat(R_p3d, T_p3d, device)  # (1, 4, 4)
    K = perspective_K(image_size, image_size, fov_deg, device)
    rgb = _rasterize(g, viewmats, K, image_size, image_size, bg)
    return Image.fromarray(rgb, "RGB")


def render_gsplat_topdown(
    g: GaussianSet,
    image_size: int = 1024,
    world_bbox: Optional[Tuple[float, float, float, float]] = None,
    margin: float = 0.05,
    cam_height: Optional[float] = None,
    fov_deg: float = 30.0,
    bg: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Image.Image:
    """Look straight down at the (XZ-extent of the) scene.

    Args:
        world_bbox : (x_min, z_min, x_max, z_max). If None, computed from g.means.
        margin     : fractional padding around scene bbox.
        cam_height : world Y at which to place the camera. If None, set so
                     the scene's XZ extent fits the FoV.
    """
    device = g.means.device
    if world_bbox is None:
        mins = g.means.amin(dim=0).cpu().numpy()
        maxs = g.means.amax(dim=0).cpu().numpy()
        x_min, x_max = float(mins[0]), float(maxs[0])
        z_min, z_max = float(mins[2]), float(maxs[2])
    else:
        x_min, z_min, x_max, z_max = world_bbox
    cx = (x_min + x_max) / 2.0
    cz = (z_min + z_max) / 2.0
    half = max(x_max - x_min, z_max - z_min) / 2.0 * (1.0 + margin)
    if cam_height is None:
        # Distance such that half = dist * tan(fov/2).
        cam_height = half / max(math.tan(math.radians(fov_deg) / 2.0), 1e-6)
    # World-to-camera for an OpenCV camera at (cx, cam_height, cz) looking down.
    # Camera +z forward = -world_y; +y down = +world_z; +x right = +world_x.
    R = torch.tensor(
        [[1.0, 0.0, 0.0],   # cam_x = world_x
         [0.0, 0.0, 1.0],   # cam_y = world_z (down for ortho top-down → world_z is fine)
         [0.0, -1.0, 0.0]], # cam_z = -world_y (forward into the ground)
        device=device, dtype=torch.float32,
    )
    cam_pos = torch.tensor([cx, cam_height, cz], device=device, dtype=torch.float32)
    T = -R @ cam_pos
    viewmat = torch.eye(4, device=device).unsqueeze(0)
    viewmat[0, :3, :3] = R
    viewmat[0, :3, 3] = T
    K = perspective_K(image_size, image_size, fov_deg, device)
    rgb = _rasterize(g, viewmat, K, image_size, image_size, bg)
    return Image.fromarray(rgb, "RGB")
