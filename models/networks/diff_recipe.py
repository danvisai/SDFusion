"""Differentiable building recipe forward pass.

Phase 1 of Option B+ (recipe-parameter diffusion) — verifies that we can
re-express scene/sdf_recipes.py in a fully differentiable form where the
recipe parameters (sizes, positions, blend factors, etc.) are tensors with
requires_grad. The output is an SDF that flows gradients all the way back
to the parameter vector.

This module mirrors the operations in scene/sdf_recipes.py:recipe_modern
but exposes them as a torch.nn.Module taking a fixed-length parameter
vector. All decisions that are sampled randomly in the procedural recipe
are made explicit, parameterized choices here.

Coordinate convention matches scene/sdf_primitives.py:
    x = east-west, z = north-south, y = up.
SDF sign convention: negative inside, positive outside.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


# =====================================================================
# Differentiable primitives — parameters are torch.Tensor, gradients flow
# =====================================================================

def d_box(p: torch.Tensor, center: torch.Tensor, half_extents: torch.Tensor) -> torch.Tensor:
    """Axis-aligned box SDF.

    p: (Q, 3) query points.
    center: (3,) tensor.
    half_extents: (3,) tensor — positive.
    """
    q = (p - center).abs() - half_extents
    outside = q.clamp_min(0.0).pow(2).sum(dim=-1).clamp_min(1e-12).sqrt()
    inside = q.max(dim=-1).values.clamp_max(0.0)
    return outside + inside


def d_sphere(p: torch.Tensor, center: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
    """Sphere SDF. center: (3,), radius: scalar tensor."""
    return torch.linalg.vector_norm(p - center, dim=-1) - radius


def d_cylinder_y(p: torch.Tensor, center: torch.Tensor,
                 radius: torch.Tensor, height: torch.Tensor) -> torch.Tensor:
    """Vertical (Y-axis) cylinder SDF. center: (3,), radius+height: scalars."""
    local = p - center
    half_h = height * 0.5
    d_xz = torch.linalg.norm(local[..., [0, 2]], dim=-1) - radius
    d_y = local[..., 1].abs() - half_h
    outside = torch.stack([d_xz.clamp_min(0.0), d_y.clamp_min(0.0)], dim=-1)
    outside_n = outside.pow(2).sum(dim=-1).clamp_min(1e-12).sqrt()
    inside = torch.maximum(d_xz, d_y).clamp_max(0.0)
    return outside_n + inside


def d_cone_y(p: torch.Tensor, apex: torch.Tensor,
             angle_deg: torch.Tensor, height: torch.Tensor) -> torch.Tensor:
    """Cone pointing +y with apex at `apex`. Base at apex.y - height.
    angle_deg: half-angle in degrees (tensor for differentiability).
    """
    ang = angle_deg * (math.pi / 180.0)
    sin_a = torch.sin(ang)
    cos_a = torch.cos(ang)
    # Shift apex to origin, point +y.
    q = p - apex
    q_xz = torch.linalg.norm(q[..., [0, 2]], dim=-1)
    # IQ cone: d1 = lateral slope, d2 = base cap
    d1 = q_xz * cos_a + q[..., 1] * sin_a
    d2 = -q[..., 1] - height
    return torch.maximum(d1, d2)


def d_polygon_prism(p: torch.Tensor, poly_xz: torch.Tensor,
                    y_min: torch.Tensor, y_max: torch.Tensor) -> torch.Tensor:
    """Polygon prism SDF, extrude poly along +y between y_min and y_max.

    poly_xz: (P, 2) ordered CCW. Differentiable in vertex positions.
    y_min, y_max: scalar tensors.
    p: (Q, 3) queries.
    """
    Q = p.shape[0]
    P = poly_xz.shape[0]
    pl = poly_xz                                       # (P, 2)
    pln = torch.roll(pl, -1, dims=0)
    edges = pln - pl                                   # (P, 2)
    elen2 = (edges * edges).sum(dim=-1).clamp_min(1e-12)

    # 2D XZ projection
    p_xz = p[..., [0, 2]]                              # (Q, 2)
    # vec from each polygon vertex to each query
    w = p_xz.unsqueeze(1) - pl.unsqueeze(0)            # (Q, P, 2)
    t = (w * edges.unsqueeze(0)).sum(dim=-1) / elen2.unsqueeze(0)
    t = t.clamp(0.0, 1.0)
    b = w - edges.unsqueeze(0) * t.unsqueeze(-1)       # (Q, P, 2)
    d2 = (b * b).sum(dim=-1)                           # (Q, P)
    d2_min = d2.min(dim=-1).values                     # (Q,)

    # Winding-number sign via Inigo Quilez's polygon SDF trick.
    v_i = pl.unsqueeze(0)                              # (1, P, 2)
    v_j = pln.unsqueeze(0)
    cond1 = p_xz[:, 1].unsqueeze(-1) >= v_i[..., 1]
    cond2 = p_xz[:, 1].unsqueeze(-1) < v_j[..., 1]
    cross_z = edges[..., 0].unsqueeze(0) * w[..., 1] - edges[..., 1].unsqueeze(0) * w[..., 0]
    cond3 = cross_z > 0
    all_t = cond1 & cond2 & cond3
    all_f = (~cond1) & (~cond2) & (~cond3)
    flips = (all_t | all_f).to(p.dtype)
    flips_sum = flips.sum(dim=-1) % 2
    signs = torch.where(flips_sum < 0.5,
                        torch.ones((), dtype=p.dtype, device=p.device),
                        -torch.ones((), dtype=p.dtype, device=p.device))
    d_xz = signs * (d2_min + 1e-12).sqrt()

    # Y direction.
    y = p[..., 1]
    half_h = (y_max - y_min) * 0.5
    y_center = (y_max + y_min) * 0.5
    d_y = (y - y_center).abs() - half_h

    outside = torch.stack([d_xz.clamp_min(0.0), d_y.clamp_min(0.0)], dim=-1).pow(2).sum(dim=-1).clamp_min(1e-12).sqrt()
    inside = torch.maximum(d_xz, d_y).clamp_max(0.0)
    return outside + inside


def d_gable_roof(p: torch.Tensor, center_xz: torch.Tensor, width: torch.Tensor,
                 depth: torch.Tensor, base_y: torch.Tensor, roof_height: torch.Tensor) -> torch.Tensor:
    """Gable roof SDF. Ridge runs along X axis at z = center_z, y = base_y + roof_height.
    Box-shaped cap with two sloping planes.

    center_xz: (2,) — XZ center of the roof footprint.
    width, depth: scalar tensors — XZ extent.
    base_y: scalar — bottom of roof (typically = building height).
    roof_height: scalar — ridge height above base_y.
    """
    cx, cz = center_xz[0], center_xz[1]
    hw = width * 0.5
    hd = depth * 0.5
    # Slope normal components: angle is atan2(roof_height, hd).
    # nz = sin(angle) = roof_height / sqrt(roof_height^2 + hd^2), ny = cos(angle).
    diag = torch.sqrt(roof_height * roof_height + hd * hd).clamp_min(1e-6)
    ny = hd / diag
    nz = roof_height / diag
    py = p[..., 1] - base_y
    pz = p[..., 2] - cz
    slope1 = py * ny + (-(pz - hd)) * nz       # ≥0 outside top-front slope
    slope2 = py * ny + ((pz + hd)) * nz        # using (-(pz - (-hd))) = pz + hd; wait let me redo
    # Above original: plane1 anchored at (cx, base_y, cz + hd) with normal (0, ny, -nz):
    #   dot(p - p0, n) = (p.y - base_y)*ny + (p.z - (cz+hd))*(-nz)
    #                  = py*ny - (pz - hd)*nz
    # plane2 anchored at (cz - hd) with normal (0, ny, +nz):
    #   = py*ny + (pz - (-hd))*nz = py*ny + (pz + hd)*nz
    slope1 = py * ny - (pz - hd) * nz
    slope2 = py * ny + (pz + hd) * nz
    # cap to box of dimensions (width, roof_height, depth) centered at
    # (cx, base_y + roof_height/2, cz).
    cap_center = torch.stack([cx, base_y + roof_height * 0.5, cz], dim=0)
    cap_he = torch.stack([hw, roof_height * 0.5, hd], dim=0)
    cap = d_box(p, cap_center, cap_he)
    roof = torch.maximum(torch.maximum(slope1, slope2), cap)
    return roof


def d_hip_roof(p: torch.Tensor, center_xz: torch.Tensor, width: torch.Tensor,
               depth: torch.Tensor, base_y: torch.Tensor, roof_height: torch.Tensor) -> torch.Tensor:
    """Hip (pyramidal) roof SDF — apex at center, sloping down to all 4 edges."""
    cx, cz = center_xz[0], center_xz[1]
    hw = width * 0.5
    hd = depth * 0.5
    diag_x = torch.sqrt(roof_height * roof_height + hw * hw).clamp_min(1e-6)
    ny_x = hw / diag_x
    nx_x = roof_height / diag_x
    diag_z = torch.sqrt(roof_height * roof_height + hd * hd).clamp_min(1e-6)
    ny_z = hd / diag_z
    nz_z = roof_height / diag_z
    py = p[..., 1] - base_y
    px = p[..., 0] - cx
    pz = p[..., 2] - cz
    s1 = py * ny_x + (px - hw) * nx_x
    s2 = py * ny_x - (px + hw) * nx_x
    s3 = py * ny_z + (pz - hd) * nz_z
    s4 = py * ny_z - (pz + hd) * nz_z
    cap_center = torch.stack([cx, base_y + roof_height * 0.5, cz], dim=0)
    cap_he = torch.stack([hw, roof_height * 0.5, hd], dim=0)
    cap = d_box(p, cap_center, cap_he)
    return torch.maximum(torch.maximum(torch.maximum(torch.maximum(s1, s2), s3), s4), cap)


def expand_polygon_from_centroid(poly_xz: torch.Tensor, amount: torch.Tensor) -> torch.Tensor:
    """Differentiable polygon expansion (away from centroid by `amount`)."""
    centroid = poly_xz.mean(dim=0, keepdim=True)
    direction = poly_xz - centroid
    dir_norm = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return poly_xz + dir_norm * amount


def d_union(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.minimum(a, b)


def d_subtract(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.maximum(a, -b)


def d_smooth_union(a: torch.Tensor, b: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Quadratic polynomial smooth-min (Inigo Quilez baseline). k = blend radius.

    Kept for backwards compatibility. Prefer `d_smooth_union_quartic` (below)
    for new code — it has better gradient behavior near the blend region.
    """
    k_safe = k.clamp_min(1e-3)
    h = (k_safe - (a - b).abs()).clamp_min(0.0) / k_safe
    return torch.minimum(a, b) - h * h * k_safe * 0.25


def d_smooth_union_quartic(a: torch.Tensor, b: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """C2 quartic polynomial smin from Inigo Quilez (iquilezles.org/articles/smin/).

    Quartic falls off more gracefully than the quadratic version near the blend
    region — fewer gradient discontinuities at the boundary where one SDF
    transitions from dominating to blending. Same signature.

    Reference: 'A study on smooth-minimums' (IQ, 2024). Section "polynomial 4".
    """
    k_safe = k.clamp_min(1e-3) * 4.0  # rescale to match qty.smooth_union's blend radius
    h = (k_safe - (a - b).abs()).clamp_min(0.0) / k_safe   # in [0, 1]
    return torch.minimum(a, b) - h * h * h * k_safe * (1.0 / 6.0)


def d_smooth_union_circular(a: torch.Tensor, b: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Circular smin (IQ 2024). Constant curvature blend region — most uniform
    gradient magnitudes of any common smin variant. Slightly more expensive."""
    k_safe = k.clamp_min(1e-3)
    h = torch.maximum(k_safe - (a - b).abs(), torch.zeros_like(a))
    return torch.minimum(a, b) - (k_safe + h - torch.sqrt(k_safe * k_safe + h * h)) * 0.5


def soft_blend(a: torch.Tensor, b: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Soft union via differentiable 'gating': returns ~a if alpha->0, ~union(a,b) if alpha->1.

    Used to make primitive occupancy differentiable. alpha in [0, 1].
    """
    union = torch.minimum(a, b)
    # Linear interpolate. When alpha=0, return a; alpha=1, return union(a,b).
    return a * (1.0 - alpha) + union * alpha


# =====================================================================
# Smooth clamps — replace hard max()/min() with differentiable versions
# =====================================================================

def soft_max(x: torch.Tensor, floor_value: float, beta: float = 10.0) -> torch.Tensor:
    """Differentiable approximation of max(floor_value, x) — useful when we
    want a lower bound that doesn't kill gradient when x < floor_value.

    softplus version: floor + softplus(beta * (x - floor)) / beta.
    Matches max() in the limit beta -> infinity; has finite, smooth gradient
    everywhere for finite beta. beta=10 gives ~ 1% deviation near the elbow.

    Why this matters: the hard max() in the procedural recipe kills gradients
    on the parameters that scale below the floor — Phase 1 test showed
    params[0] (PARAPET_H_SCALE) and params[1] (PARAPET_INNER_SHRINK) had zero
    gradient because `max(0.4, height * params[0])` clipped to 0.4. With
    soft_max, gradient flows even when parameters fall below the floor.
    """
    if isinstance(floor_value, (int, float)):
        floor_t = torch.full_like(x, float(floor_value))
    else:
        floor_t = floor_value
    return floor_t + torch.nn.functional.softplus(beta * (x - floor_t)) / beta


# =====================================================================
# Polygon utility — differentiable polygon shrink (inset toward centroid)
# =====================================================================

def shrink_polygon_to_centroid(poly_xz: torch.Tensor, amount: torch.Tensor) -> torch.Tensor:
    """Move each vertex toward the polygon centroid by `amount` (scalar tensor).

    This is a cheap, differentiable approximation to a true inset. For convex
    polygons it's nearly identical to a Minkowski erosion when amount is small.
    poly_xz: (P, 2) tensor. Returns (P, 2).
    """
    centroid = poly_xz.mean(dim=0, keepdim=True)        # (1, 2)
    direction = centroid - poly_xz                       # (P, 2)
    dir_norm = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return poly_xz + dir_norm * amount


# =====================================================================
# DiffRecipeModern — full differentiable modern building recipe
# =====================================================================

@dataclass
class ModernParamLayout:
    """Names + indices of the 9-dim parameter vector for the modern recipe."""
    PARAPET_H_SCALE: int = 0       # multiplier on height for parapet height (default 0.05)
    PARAPET_INNER_SHRINK: int = 1  # ratio of min(w,d) for inner cutout inset (default 0.04)
    MECH_ACTIVE_LOGIT: int = 2     # sigmoid -> mech blending weight in [0,1] (default ~0.6)
    MECH_W_RATIO: int = 3          # mech box width as ratio of min(w,d) (default 0.18)
    MECH_H_RATIO: int = 4          # mech box height as ratio of building height (default 0.07)
    MECH_OFF_X: int = 5            # mech x offset in normalized [-0.5, 0.5] (default 0.0)
    MECH_OFF_Z: int = 6            # mech z offset in normalized [-0.5, 0.5] (default 0.0)
    MECH_Y_LIFT_RATIO: int = 7     # mech y lift above roof as ratio of parapet (default 0.4)
    PARAPET_INNER_H_EXTRA: int = 8 # extra height for inner cutout above parapet (default 0.2)


MODERN_N_PARAMS = 9


def modern_default_params(device: str = "cpu") -> torch.Tensor:
    """Hand-set defaults that closely mirror scene.sdf_recipes.recipe_modern
    with mech_active = on, mech_off = centered."""
    return torch.tensor([
        0.05,    # PARAPET_H_SCALE
        0.04,    # PARAPET_INNER_SHRINK
        2.0,     # MECH_ACTIVE_LOGIT (sigmoid(2.0) ~= 0.88 ON)
        0.18,    # MECH_W_RATIO
        0.07,    # MECH_H_RATIO
        0.0,     # MECH_OFF_X
        0.0,     # MECH_OFF_Z
        0.4,     # MECH_Y_LIFT_RATIO
        0.2,     # PARAPET_INNER_H_EXTRA
    ], dtype=torch.float32, device=device)


class DiffRecipeModern(nn.Module):
    """Differentiable forward of the 'modern' building style.

    Inputs (all torch tensors):
        params: (N_PARAMS,) parameter vector — what a generative head would predict.
        polygon_xz: (P, 2) building footprint polygon (CCW).
        height: scalar — building height.
        query_points: (Q, 3) where to evaluate the SDF.

    Returns:
        (Q,) SDF values.

    Differentiability: gradients flow from output SDF -> params, polygon_xz,
    height, and query_points.
    """

    N_PARAMS = MODERN_N_PARAMS
    L = ModernParamLayout()

    def forward(self,
                params: torch.Tensor,
                polygon_xz: torch.Tensor,
                height: torch.Tensor,
                query_points: torch.Tensor) -> torch.Tensor:
        L = self.L
        # Polygon stats
        p_min = polygon_xz.min(dim=0).values            # (2,)
        p_max = polygon_xz.max(dim=0).values
        center = (p_min + p_max) * 0.5
        cx, cz = center[0], center[1]
        w = (p_max[0] - p_min[0]).clamp_min(1e-3)
        d = (p_max[1] - p_min[1]).clamp_min(1e-3)
        min_wd = torch.minimum(w, d)

        # 1) Body: prism from y=0 to y=height
        y_zero = torch.zeros((), dtype=height.dtype, device=height.device)
        sdf_body = d_polygon_prism(query_points, polygon_xz, y_zero, height)

        # 2) Parapet ring on top
        # Floors are PROPORTIONAL to building extent so recipes work at any scale
        # (was hardcoded 0.4m / 0.15m — fine for ~10m buildings, broken at norm scale).
        parapet_h = soft_max(height * params[L.PARAPET_H_SCALE], height * 0.005)
        sdf_parapet = d_polygon_prism(query_points, polygon_xz,
                                      height, height + parapet_h)

        # inner cutout — polygon inset by shrink_amount
        shrink_amount = soft_max(min_wd * params[L.PARAPET_INNER_SHRINK], min_wd * 0.005)
        inner_poly = shrink_polygon_to_centroid(polygon_xz, shrink_amount)
        # No clamp_min: negative values just give degenerate (zero-height) prism,
        # which contributes no extra geometry. Gradient flows everywhere.
        inner_h_extra = params[L.PARAPET_INNER_H_EXTRA]
        sdf_inner = d_polygon_prism(
            query_points, inner_poly,
            height - 0.1, height + parapet_h + inner_h_extra,
        )
        sdf_parapet_ring = d_subtract(sdf_parapet, sdf_inner)

        out = d_union(sdf_body, sdf_parapet_ring)

        # 3) Optional mechanical box on the roof
        # Drop clamp_min: a negative MECH_W_RATIO gives a degenerate (empty) box
        # whose SDF is positive everywhere — no contribution to union. The
        # gradient still flows so the optimizer can recover when params go neg.
        mech_alpha = torch.sigmoid(params[L.MECH_ACTIVE_LOGIT])
        mech_w = min_wd * params[L.MECH_W_RATIO]
        mech_d = mech_w  # square
        mech_h = soft_max(height * params[L.MECH_H_RATIO], height * 0.005)
        off_x = params[L.MECH_OFF_X] * w * 0.35
        off_z = params[L.MECH_OFF_Z] * d * 0.35
        mech_y = height + mech_h * 0.5 + parapet_h * params[L.MECH_Y_LIFT_RATIO]

        mech_center = torch.stack([cx + off_x, mech_y, cz + off_z], dim=0)
        mech_he = torch.stack([mech_w * 0.5, mech_h * 0.5, mech_d * 0.5], dim=0)
        sdf_mech = d_box(query_points, mech_center, mech_he)

        # Differentiable soft union: alpha gates how much mech contributes.
        out = soft_blend(out, sdf_mech, mech_alpha)

        return out


# =====================================================================
# Convenience: sample on a (D, H, W) grid matching preprocess/create_sdf.py
# =====================================================================

def make_grid_points(resolution: int, bbox: Tuple[float, float, float, float, float, float],
                     device: str) -> torch.Tensor:
    """Return (resolution^3, 3) query points in the order (D=z, H=y, W=x).

    bbox: (x0, y0, z0, x1, y1, z1).
    """
    x0, y0, z0, x1, y1, z1 = bbox
    xs = torch.linspace(x0, x1, resolution, device=device)
    ys = torch.linspace(y0, y1, resolution, device=device)
    zs = torch.linspace(z0, z1, resolution, device=device)
    Z, Y, X = torch.meshgrid(zs, ys, xs, indexing="ij")
    return torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)


def grid_from_flat(sdf_flat: torch.Tensor, resolution: int) -> torch.Tensor:
    """Reshape (resolution^3,) -> (D, H, W) = (resolution, resolution, resolution)."""
    return sdf_flat.reshape(resolution, resolution, resolution)


# =====================================================================
# DiffRecipeColonial — gable roof + optional chimney
# =====================================================================

@dataclass
class ColonialParamLayout:
    ROOF_H_RATIO: int = 0          # roof height as ratio of min(w, d) (default 0.45)
    CHIMNEY_ACTIVE_LOGIT: int = 1  # sigmoid -> chimney active (default ~0.7)
    CHIMNEY_W_RATIO: int = 2       # chimney width as ratio of min(w,d) (default 0.07)
    CHIMNEY_H_FRAC: int = 3        # chimney height as fraction of roof_h (default 0.85)
    CHIMNEY_OFF_X: int = 4         # chimney x offset in [-0.5, 0.5] * w (default 0.0)


COLONIAL_N_PARAMS = 5


def colonial_default_params(device: str = "cpu") -> torch.Tensor:
    return torch.tensor([0.45, 1.0, 0.07, 0.85, 0.0], dtype=torch.float32, device=device)


class DiffRecipeColonial(nn.Module):
    N_PARAMS = COLONIAL_N_PARAMS
    L = ColonialParamLayout()

    def forward(self, params, polygon_xz, height, query_points):
        L = self.L
        p_min = polygon_xz.min(dim=0).values
        p_max = polygon_xz.max(dim=0).values
        center_xz = torch.stack([(p_min[0] + p_max[0]) * 0.5,
                                 (p_min[1] + p_max[1]) * 0.5], dim=0)
        w = (p_max[0] - p_min[0]).clamp_min(1e-3)
        d = (p_max[1] - p_min[1]).clamp_min(1e-3)
        min_wd = torch.minimum(w, d)

        y_zero = torch.zeros_like(height)
        sdf_body = d_polygon_prism(query_points, polygon_xz, y_zero, height)

        roof_h = min_wd * params[L.ROOF_H_RATIO]
        # Use longer axis as ridge.
        long_axis_x = w >= d
        if long_axis_x:
            roof = d_gable_roof(query_points, center_xz, w, d, height, roof_h)
        else:
            roof = d_gable_roof(query_points, center_xz, d, w, height, roof_h)
        out = d_union(sdf_body, roof)

        # Chimney (optional via sigmoid gate)
        ch_alpha = torch.sigmoid(params[L.CHIMNEY_ACTIVE_LOGIT])
        ch_w = soft_max(min_wd * params[L.CHIMNEY_W_RATIO], min_wd * 0.005)
        ch_h = roof_h * params[L.CHIMNEY_H_FRAC] + 0.6
        ch_off_x = params[L.CHIMNEY_OFF_X] * w * 0.25
        ch_center = torch.stack([
            center_xz[0] + ch_off_x,
            height + ch_h * 0.5 + roof_h * 0.4,
            center_xz[1],
        ], dim=0)
        ch_he = torch.stack([ch_w, ch_h * 0.5, ch_w], dim=0)
        sdf_chimney = d_box(query_points, ch_center, ch_he)
        return soft_blend(out, sdf_chimney, ch_alpha)


# =====================================================================
# DiffRecipeVictorian — hip roof + tower + cone spire + bay window
# =====================================================================

@dataclass
class VictorianParamLayout:
    ROOF_H_RATIO: int = 0
    TOWER_R_RATIO: int = 1
    TOWER_H_FRAC: int = 2          # tower height as fraction of body height
    TOWER_POS_X_RATIO: int = 3     # offset in [-0.5,0.5]*w
    TOWER_POS_Z_RATIO: int = 4
    SPIRE_H_FRAC: int = 5
    SPIRE_ANGLE_DEG: int = 6
    BAY_W_RATIO: int = 7
    BAY_H_FRAC: int = 8
    BAY_D_RATIO: int = 9
    BAY_OFF_X_RATIO: int = 10
    BAY_BLEND_K: int = 11


VICTORIAN_N_PARAMS = 12


def victorian_default_params(device: str = "cpu") -> torch.Tensor:
    return torch.tensor([0.40, 0.16, 0.40, 0.15, 0.15, 0.80, 20.0,
                         0.20, 0.55, 0.12, -0.25, 0.4],
                        dtype=torch.float32, device=device)


class DiffRecipeVictorian(nn.Module):
    N_PARAMS = VICTORIAN_N_PARAMS
    L = VictorianParamLayout()

    def forward(self, params, polygon_xz, height, query_points):
        L = self.L
        p_min = polygon_xz.min(dim=0).values
        p_max = polygon_xz.max(dim=0).values
        center_xz = torch.stack([(p_min[0] + p_max[0]) * 0.5,
                                 (p_min[1] + p_max[1]) * 0.5], dim=0)
        w = (p_max[0] - p_min[0]).clamp_min(1e-3)
        d = (p_max[1] - p_min[1]).clamp_min(1e-3)
        min_wd = torch.minimum(w, d)
        y_zero = torch.zeros_like(height)

        body = d_polygon_prism(query_points, polygon_xz, y_zero, height)
        roof_h = min_wd * params[L.ROOF_H_RATIO]
        roof = d_hip_roof(query_points, center_xz, w, d, height, roof_h)
        out = d_union(body, roof)

        # Tower
        tower_r = min_wd * params[L.TOWER_R_RATIO]
        tower_h = height * params[L.TOWER_H_FRAC]
        tower_pos = torch.stack([
            center_xz[0] + w * params[L.TOWER_POS_X_RATIO],
            height + roof_h + tower_h * 0.5,
            center_xz[1] + d * params[L.TOWER_POS_Z_RATIO],
        ], dim=0)
        tower = d_cylinder_y(query_points, tower_pos, tower_r, tower_h)
        out = d_union(out, tower)

        # Spire
        spire_h = tower_h * params[L.SPIRE_H_FRAC]
        apex = torch.stack([
            tower_pos[0],
            height + roof_h + tower_h + spire_h,
            tower_pos[2],
        ], dim=0)
        spire = d_cone_y(query_points, apex, params[L.SPIRE_ANGLE_DEG], spire_h)
        out = d_union(out, spire)

        # Bay window (smooth-union)
        bay_w = min_wd * params[L.BAY_W_RATIO]
        bay_h = height * params[L.BAY_H_FRAC]
        bay_d = min_wd * params[L.BAY_D_RATIO]
        bay_center = torch.stack([
            center_xz[0] + w * params[L.BAY_OFF_X_RATIO],
            bay_h * 0.5,
            center_xz[1] + d * 0.5 + bay_d * 0.4,
        ], dim=0)
        bay_he = torch.stack([bay_w * 0.5, bay_h * 0.5, bay_d * 0.5], dim=0)
        bay = d_box(query_points, bay_center, bay_he)
        k_blend = soft_max(params[L.BAY_BLEND_K], min_wd * 0.005)
        out = d_smooth_union_quartic(out, bay, k_blend)
        return out


# =====================================================================
# DiffRecipeIndustrial — flat slab + eaves + vent stack
# =====================================================================

@dataclass
class IndustrialParamLayout:
    SLAB_H: int = 0                 # roof slab thickness (default 0.3)
    EAVES_EXPAND_RATIO: int = 1     # eaves expansion ratio of min_wd (default 0.03)
    EAVES_H: int = 2                # eaves thickness (default 0.18)
    STACK_R_RATIO: int = 3          # stack radius / min(w,d) (default 0.05)
    STACK_H_FRAC: int = 4           # stack height / body height (default 0.20)
    STACK_OFF_X: int = 5            # offset ratio in [-0.5,0.5]*w (default 0.18)
    STACK_OFF_Z: int = 6


INDUSTRIAL_N_PARAMS = 7


def industrial_default_params(device: str = "cpu") -> torch.Tensor:
    return torch.tensor([0.3, 0.03, 0.18, 0.05, 0.20, 0.18, 0.05],
                        dtype=torch.float32, device=device)


class DiffRecipeIndustrial(nn.Module):
    N_PARAMS = INDUSTRIAL_N_PARAMS
    L = IndustrialParamLayout()

    def forward(self, params, polygon_xz, height, query_points):
        L = self.L
        p_min = polygon_xz.min(dim=0).values
        p_max = polygon_xz.max(dim=0).values
        center_xz = torch.stack([(p_min[0] + p_max[0]) * 0.5,
                                 (p_min[1] + p_max[1]) * 0.5], dim=0)
        w = (p_max[0] - p_min[0]).clamp_min(1e-3)
        d = (p_max[1] - p_min[1]).clamp_min(1e-3)
        min_wd = torch.minimum(w, d)
        y_zero = torch.zeros_like(height)
        body = d_polygon_prism(query_points, polygon_xz, y_zero, height)

        slab_h = soft_max(params[L.SLAB_H], height * 0.005)
        slab = d_polygon_prism(query_points, polygon_xz, height, height + slab_h)

        eaves_amount = min_wd * params[L.EAVES_EXPAND_RATIO]
        eaves_poly = expand_polygon_from_centroid(polygon_xz, eaves_amount)
        eaves_h = soft_max(params[L.EAVES_H], height * 0.005)
        eaves = d_polygon_prism(query_points, eaves_poly, height - 0.05, height - 0.05 + eaves_h)

        stack_r = soft_max(min_wd * params[L.STACK_R_RATIO], min_wd * 0.005)
        stack_h = height * params[L.STACK_H_FRAC]
        stack_pos = torch.stack([
            center_xz[0] + w * params[L.STACK_OFF_X],
            height + stack_h * 0.5 + slab_h + 0.3,
            center_xz[1] + d * params[L.STACK_OFF_Z],
        ], dim=0)
        stack = d_cylinder_y(query_points, stack_pos, stack_r, stack_h)
        return d_union(d_union(body, d_union(slab, eaves)), stack)


# =====================================================================
# DiffRecipeCraftsman — low hip + eaves + optional porch
# =====================================================================

@dataclass
class CraftsmanParamLayout:
    ROOF_H_RATIO: int = 0           # roof height / min(w,d) (default 0.20)
    EAVES_EXPAND_RATIO: int = 1     # default 0.03
    PORCH_ACTIVE_LOGIT: int = 2     # sigmoid (default ~0.5)
    PORCH_W_RATIO: int = 3          # porch width / w (default 0.55)
    PORCH_D_RATIO: int = 4          # porch depth / min(w,d) (default 0.20)
    PORCH_H_FRAC: int = 5           # porch height / body height (default 0.03)


CRAFTSMAN_N_PARAMS = 6


def craftsman_default_params(device: str = "cpu") -> torch.Tensor:
    return torch.tensor([0.20, 0.03, 0.0, 0.55, 0.20, 0.03],
                        dtype=torch.float32, device=device)


class DiffRecipeCraftsman(nn.Module):
    N_PARAMS = CRAFTSMAN_N_PARAMS
    L = CraftsmanParamLayout()

    def forward(self, params, polygon_xz, height, query_points):
        L = self.L
        p_min = polygon_xz.min(dim=0).values
        p_max = polygon_xz.max(dim=0).values
        center_xz = torch.stack([(p_min[0] + p_max[0]) * 0.5,
                                 (p_min[1] + p_max[1]) * 0.5], dim=0)
        w = (p_max[0] - p_min[0]).clamp_min(1e-3)
        d = (p_max[1] - p_min[1]).clamp_min(1e-3)
        min_wd = torch.minimum(w, d)
        y_zero = torch.zeros_like(height)
        body = d_polygon_prism(query_points, polygon_xz, y_zero, height)

        roof_h = min_wd * params[L.ROOF_H_RATIO]
        eaves_amount = min_wd * params[L.EAVES_EXPAND_RATIO]
        eaves_poly = expand_polygon_from_centroid(polygon_xz, eaves_amount)
        eaves_body = d_polygon_prism(query_points, eaves_poly, height - 0.05, height - 0.05 + 0.20)
        roof = d_hip_roof(query_points, center_xz, w * 1.02, d * 1.02, height, roof_h)
        out = d_union(body, d_union(eaves_body, roof))

        porch_alpha = torch.sigmoid(params[L.PORCH_ACTIVE_LOGIT])
        porch_w = w * params[L.PORCH_W_RATIO]
        porch_d = soft_max(min_wd * params[L.PORCH_D_RATIO], min_wd * 0.005)
        porch_h = soft_max(height * params[L.PORCH_H_FRAC], height * 0.005)
        porch_center = torch.stack([
            center_xz[0],
            porch_h * 0.5,
            center_xz[1] + d * 0.5 + porch_d * 0.5,
        ], dim=0)
        porch_he = torch.stack([porch_w * 0.5, porch_h * 0.5, porch_d * 0.5], dim=0)
        porch = d_box(query_points, porch_center, porch_he)
        return soft_blend(out, porch, porch_alpha)


# =====================================================================
# DiffRecipeMediterranean — very low hip + edge band
# =====================================================================

@dataclass
class MediterraneanParamLayout:
    ROOF_H_RATIO: int = 0           # default 0.14
    EAVES_EXPAND_RATIO: int = 1     # default 0.04
    EDGE_BAND_H: int = 2            # default 0.25


MEDITERRANEAN_N_PARAMS = 3


def mediterranean_default_params(device: str = "cpu") -> torch.Tensor:
    return torch.tensor([0.14, 0.04, 0.25], dtype=torch.float32, device=device)


class DiffRecipeMediterranean(nn.Module):
    N_PARAMS = MEDITERRANEAN_N_PARAMS
    L = MediterraneanParamLayout()

    def forward(self, params, polygon_xz, height, query_points):
        L = self.L
        p_min = polygon_xz.min(dim=0).values
        p_max = polygon_xz.max(dim=0).values
        center_xz = torch.stack([(p_min[0] + p_max[0]) * 0.5,
                                 (p_min[1] + p_max[1]) * 0.5], dim=0)
        w = (p_max[0] - p_min[0]).clamp_min(1e-3)
        d = (p_max[1] - p_min[1]).clamp_min(1e-3)
        min_wd = torch.minimum(w, d)
        y_zero = torch.zeros_like(height)
        body = d_polygon_prism(query_points, polygon_xz, y_zero, height)
        roof_h = min_wd * params[L.ROOF_H_RATIO]
        roof = d_hip_roof(query_points, center_xz, w * 1.04, d * 1.04, height, roof_h)
        eaves_amount = min_wd * params[L.EAVES_EXPAND_RATIO]
        eaves_poly = expand_polygon_from_centroid(polygon_xz, eaves_amount)
        band_h = soft_max(params[L.EDGE_BAND_H], height * 0.005)
        edge_band = d_polygon_prism(query_points, eaves_poly, height, height + band_h)
        return d_union(body, d_union(edge_band, roof))


# =====================================================================
# DiffRecipeContemporary — offset upper volume (smooth-union)
# =====================================================================

@dataclass
class ContemporaryParamLayout:
    UPPER_H_RATIO: int = 0          # upper height / body height (default 0.45)
    UPPER_W_RATIO: int = 1          # upper width / body w (default 0.65)
    UPPER_D_RATIO: int = 2          # default 0.7
    UPPER_OFF_X: int = 3            # offset in [-0.5,0.5]*w (default 0.0)
    UPPER_OFF_Z: int = 4
    BLEND_K: int = 5                # smooth-union blend radius (default 0.4)


CONTEMPORARY_N_PARAMS = 6


def contemporary_default_params(device: str = "cpu") -> torch.Tensor:
    return torch.tensor([0.45, 0.65, 0.7, 0.0, 0.0, 0.4],
                        dtype=torch.float32, device=device)


class DiffRecipeContemporary(nn.Module):
    N_PARAMS = CONTEMPORARY_N_PARAMS
    L = ContemporaryParamLayout()

    def forward(self, params, polygon_xz, height, query_points):
        L = self.L
        p_min = polygon_xz.min(dim=0).values
        p_max = polygon_xz.max(dim=0).values
        center_xz = torch.stack([(p_min[0] + p_max[0]) * 0.5,
                                 (p_min[1] + p_max[1]) * 0.5], dim=0)
        w = (p_max[0] - p_min[0]).clamp_min(1e-3)
        d = (p_max[1] - p_min[1]).clamp_min(1e-3)
        min_wd = torch.minimum(w, d)
        y_zero = torch.zeros_like(height)
        body = d_polygon_prism(query_points, polygon_xz, y_zero, height)
        upper_h = height * params[L.UPPER_H_RATIO]
        upper_w = w * params[L.UPPER_W_RATIO]
        upper_d = d * params[L.UPPER_D_RATIO]
        upper_center = torch.stack([
            center_xz[0] + w * params[L.UPPER_OFF_X],
            height + upper_h * 0.5,
            center_xz[1] + d * params[L.UPPER_OFF_Z],
        ], dim=0)
        upper_he = torch.stack([upper_w * 0.5, upper_h * 0.5, upper_d * 0.5], dim=0)
        upper = d_box(query_points, upper_center, upper_he)
        k = soft_max(params[L.BLEND_K], min_wd * 0.005)
        return d_smooth_union_quartic(body, upper, k)


# =====================================================================
# DiffRecipePublicCivic — dome + drum + flanking volumes
# =====================================================================

@dataclass
class PublicCivicParamLayout:
    DOME_R_RATIO: int = 0           # dome radius / min(w,d) (default 0.28)
    DOME_Y_OFFSET_FRAC: int = 1     # dome center y offset above height as fraction of dome_r (default 0.55)
    DRUM_H_FRAC: int = 2            # drum height / dome_r (default 0.6)
    DRUM_R_FRAC: int = 3            # drum radius / dome_r (default 0.95)
    FLANK_W_RATIO: int = 4          # flank width / w (default 0.18)
    FLANK_H_FRAC: int = 5           # flank height / body height (default 0.25)
    FLANK_D_RATIO: int = 6          # flank depth / d (default 0.5)
    FLANK_OFF_X_RATIO: int = 7      # flank x offset / w (default 0.35)


PUBLIC_CIVIC_N_PARAMS = 8


def public_civic_default_params(device: str = "cpu") -> torch.Tensor:
    return torch.tensor([0.28, 0.55, 0.6, 0.95, 0.18, 0.25, 0.5, 0.35],
                        dtype=torch.float32, device=device)


class DiffRecipePublicCivic(nn.Module):
    N_PARAMS = PUBLIC_CIVIC_N_PARAMS
    L = PublicCivicParamLayout()

    def forward(self, params, polygon_xz, height, query_points):
        L = self.L
        p_min = polygon_xz.min(dim=0).values
        p_max = polygon_xz.max(dim=0).values
        center_xz = torch.stack([(p_min[0] + p_max[0]) * 0.5,
                                 (p_min[1] + p_max[1]) * 0.5], dim=0)
        w = (p_max[0] - p_min[0]).clamp_min(1e-3)
        d = (p_max[1] - p_min[1]).clamp_min(1e-3)
        min_wd = torch.minimum(w, d)
        y_zero = torch.zeros_like(height)
        body = d_polygon_prism(query_points, polygon_xz, y_zero, height)

        dome_r = min_wd * params[L.DOME_R_RATIO]
        dome_center = torch.stack([
            center_xz[0],
            height + dome_r * params[L.DOME_Y_OFFSET_FRAC],
            center_xz[1],
        ], dim=0)
        dome = d_sphere(query_points, dome_center, dome_r)

        drum_h = dome_r * params[L.DRUM_H_FRAC]
        drum_r = dome_r * params[L.DRUM_R_FRAC]
        drum_center = torch.stack([
            center_xz[0],
            height + drum_h * 0.5,
            center_xz[1],
        ], dim=0)
        drum = d_cylinder_y(query_points, drum_center, drum_r, drum_h)

        out = d_union(body, d_union(drum, dome))

        flank_w = w * params[L.FLANK_W_RATIO]
        flank_h = height * params[L.FLANK_H_FRAC]
        flank_d = d * params[L.FLANK_D_RATIO]
        off_x = w * params[L.FLANK_OFF_X_RATIO]
        left_center = torch.stack([
            center_xz[0] - off_x,
            flank_h * 0.5,
            center_xz[1],
        ], dim=0)
        right_center = torch.stack([
            center_xz[0] + off_x,
            flank_h * 0.5,
            center_xz[1],
        ], dim=0)
        flank_he = torch.stack([flank_w * 0.5, flank_h * 0.5, flank_d * 0.5], dim=0)
        left = d_box(query_points, left_center, flank_he)
        right = d_box(query_points, right_center, flank_he)
        return d_union(out, d_union(left, right))


# =====================================================================
# Style registry — uniform API for all 8 styles
# =====================================================================

DIFF_RECIPE_REGISTRY = {
    "modern":         (DiffRecipeModern,        modern_default_params,        MODERN_N_PARAMS),
    "colonial":       (DiffRecipeColonial,      colonial_default_params,      COLONIAL_N_PARAMS),
    "victorian":      (DiffRecipeVictorian,     victorian_default_params,     VICTORIAN_N_PARAMS),
    "industrial":     (DiffRecipeIndustrial,    industrial_default_params,    INDUSTRIAL_N_PARAMS),
    "craftsman":      (DiffRecipeCraftsman,     craftsman_default_params,     CRAFTSMAN_N_PARAMS),
    "mediterranean":  (DiffRecipeMediterranean, mediterranean_default_params, MEDITERRANEAN_N_PARAMS),
    "contemporary":   (DiffRecipeContemporary,  contemporary_default_params,  CONTEMPORARY_N_PARAMS),
    "public_civic":   (DiffRecipePublicCivic,   public_civic_default_params,  PUBLIC_CIVIC_N_PARAMS),
}


def build_diff_recipe(style: str):
    """Factory: returns (module instance, default_params_fn, n_params)."""
    if style not in DIFF_RECIPE_REGISTRY:
        raise KeyError(f"Unknown style '{style}'. Known: {list(DIFF_RECIPE_REGISTRY)}")
    cls, defaults, n = DIFF_RECIPE_REGISTRY[style]
    return cls(), defaults, n


def bbox_for_polygon(polygon_xz: np.ndarray, height: float,
                     pad: float = 0.5) -> Tuple[float, float, float, float, float, float]:
    """Square-pad bbox around polygon + height. Mirrors scene.sdf_primitives.polygon_bbox_with_pad."""
    poly = np.asarray(polygon_xz)
    x0, z0 = poly.min(axis=0) - pad
    x1, z1 = poly.max(axis=0) + pad
    # Square in XZ + Y starting at 0.
    span_x = x1 - x0
    span_z = z1 - z0
    side = max(span_x, span_z, height + pad)
    cx = (x0 + x1) * 0.5
    cz = (z0 + z1) * 0.5
    half = side * 0.5
    return (cx - half, -pad, cz - half, cx + half, height + pad, cz + half)
