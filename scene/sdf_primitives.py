"""Torch-native SDF primitives, boolean ops, and transforms.

An SDF is just a callable f(p: Tensor[Q, 3]) -> Tensor[Q] returning the signed
distance to a surface (negative inside, positive outside). Composition is by
higher-order functions; no class hierarchy.

Coordinate convention (matches scene/run_demo.py + preprocess/create_sdf.py):
    x = world east-west, z = world north-south, y = up.

References:
    Inigo Quilez "distance functions": https://iquilezles.org/articles/distfunctions/
    IQ "distance functions 2d":         https://iquilezles.org/articles/distfunctions2d/
"""
from __future__ import annotations
import math
from typing import Callable, Iterable, List, Sequence

import numpy as np
import torch

SDF = Callable[[torch.Tensor], torch.Tensor]  # (Q, 3) -> (Q,)


# --- 3D primitives -----------------------------------------------------------

def sdf_box(half_extents: Sequence[float]) -> SDF:
    """Axis-aligned box centered at origin with given half-extents (3,)."""
    he = torch.tensor(half_extents, dtype=torch.float32)

    def f(p: torch.Tensor) -> torch.Tensor:
        q = p.abs() - he.to(p.device)
        return torch.linalg.norm(q.clamp_min(0.0), dim=-1) + q.max(dim=-1).values.clamp_max(0.0)

    return f


def sdf_rounded_box(half_extents: Sequence[float], radius: float) -> SDF:
    """Box with rounded corners (radius shrinks the box, then offsets the iso-surface out)."""
    inner = sdf_box([h - radius for h in half_extents])

    def f(p: torch.Tensor) -> torch.Tensor:
        return inner(p) - radius

    return f


def sdf_sphere(radius: float) -> SDF:
    def f(p: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(p, dim=-1) - radius

    return f


def sdf_cylinder_y(radius: float, height: float) -> SDF:
    """Cylinder centered at origin, axis along +y, total height = height."""
    half_h = height / 2.0

    def f(p: torch.Tensor) -> torch.Tensor:
        d_xz = torch.linalg.norm(p[..., [0, 2]], dim=-1) - radius
        d_y = p[..., 1].abs() - half_h
        outside = torch.stack([d_xz.clamp_min(0.0), d_y.clamp_min(0.0)], dim=-1)
        inside = torch.maximum(d_xz, d_y).clamp_max(0.0)
        return torch.linalg.norm(outside, dim=-1) + inside

    return f


def sdf_cone_y(angle_deg: float, height: float) -> SDF:
    """Cone pointing +y, apex at (0, height, 0), base at y=0, base radius = height*tan(angle)."""
    # IQ-style cone SDF.
    ang = math.radians(angle_deg)
    c = (math.sin(ang), math.cos(ang))

    def f(p: torch.Tensor) -> torch.Tensor:
        # Shift so apex at origin pointing +y.
        q = p.clone()
        q[..., 1] = q[..., 1] - height
        r_xz = torch.linalg.norm(q[..., [0, 2]], dim=-1)
        d1 = r_xz * c[1] + q[..., 1] * c[0]
        d2 = -q[..., 1] - height  # base cap (y = -height in shifted frame)
        return torch.maximum(d1, d2)

    return f


def sdf_polygon_2d(polygon_xz: np.ndarray) -> SDF:
    """Signed distance to a polygon in the XZ plane, ignoring y. Negative inside.

    polygon_xz : (P, 2) np.ndarray of polygon vertices, CCW for outward normal.
                 Works for convex AND concave polygons (IQ winding-based sdf).

    Split out of `sdf_polygon_prism` because a roof needs the *distance to the walls* on its own:
    the height a hip roof falls to is a function of how far a point is from the footprint boundary,
    which is this field and cannot be written as a combination of half-spaces (an inward-facing
    plane per wall is right until the footprint turns a reflex corner, where the nearest wall is a
    vertex rather than an edge).
    """
    poly = torch.tensor(np.asarray(polygon_xz, dtype=np.float32))
    poly_next = torch.roll(poly, -1, dims=0)
    edges = poly_next - poly  # (P, 2)
    edge_lens2 = (edges * edges).sum(dim=-1).clamp_min(1e-12)  # (P,)

    def f(p: torch.Tensor) -> torch.Tensor:
        device = p.device
        pl = poly.to(device)            # (P, 2)
        pln = poly_next.to(device)
        e = edges.to(device)            # (P, 2)
        elen2 = edge_lens2.to(device)   # (P,)

        # 2D point in XZ frame.
        p_xz = p[..., [0, 2]]  # (Q, 2)
        # For each query, distance to each edge segment.
        # w_i = p_xz - v_i  for all queries vs all polygon verts.
        w = p_xz.unsqueeze(1) - pl.unsqueeze(0)  # (Q, P, 2)
        t = (w * e.unsqueeze(0)).sum(dim=-1) / elen2.unsqueeze(0)  # (Q, P)
        t = t.clamp(0.0, 1.0)
        b = w - e.unsqueeze(0) * t.unsqueeze(-1)  # (Q, P, 2)
        d2 = (b * b).sum(dim=-1)  # (Q, P)
        d2_min = d2.min(dim=-1).values  # (Q,)

        # Winding-number sign (Inigo Quilez).
        # For each edge i->j: cross product sign and y-band membership.
        v_i = pl.unsqueeze(0)        # (1, P, 2)
        v_j = pln.unsqueeze(0)       # (1, P, 2)
        cond1 = p_xz[:, 1].unsqueeze(-1) >= v_i[..., 1]
        cond2 = p_xz[:, 1].unsqueeze(-1) < v_j[..., 1]
        cross_z = e[..., 0].unsqueeze(0) * w[..., 1] - e[..., 1].unsqueeze(0) * w[..., 0]
        cond3 = cross_z > 0
        all_t = cond1 & cond2 & cond3
        all_f = (~cond1) & (~cond2) & (~cond3)
        flips = (all_t | all_f).to(p.dtype)  # (Q, P) -- flip when crossing
        signs = torch.where(flips.sum(dim=-1) % 2 < 0.5, torch.tensor(1.0, device=device), torch.tensor(-1.0, device=device))
        return signs * torch.sqrt(d2_min + 1e-12)

    return f


def sdf_polygon_prism(polygon_xz: np.ndarray, height: float) -> SDF:
    """Extrude a 2D polygon in XZ along +y from y=0 to y=height.

    polygon_xz : (P, 2) np.ndarray of polygon vertices, CCW for outward normal.
                 Works for convex AND concave polygons (IQ winding-based sdf).
    """
    outline = sdf_polygon_2d(polygon_xz)
    half_h = height / 2.0
    y_offset = height / 2.0  # so the prism is y in [0, height]

    def f(p: torch.Tensor) -> torch.Tensor:
        d_xz = outline(p)

        # Y direction.
        y_local = p[..., 1] - y_offset
        d_y = y_local.abs() - half_h

        # Combine 2D XZ + Y.
        outside = torch.stack([d_xz.clamp_min(0.0), d_y.clamp_min(0.0)], dim=-1)
        inside = torch.maximum(d_xz, d_y).clamp_max(0.0)
        return torch.linalg.norm(outside, dim=-1) + inside

    return f


def sdf_plane_halfspace(normal: Sequence[float], offset: float) -> SDF:
    """The solid half-space ``{p : dot(n, p) + offset <= 0}``.

    `normal` is normalised here, so the result is an exact SDF (unit gradient) whatever scale the
    caller's coefficients came in at. This is the missing half of a layer program: `Ramp` is a
    plane cutting a polygonal region, and `CutRoof` is several of them (one per eave).
    """
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(n))
    if norm < 1e-12:
        raise ValueError("plane normal must be non-zero")
    n32 = torch.tensor((n / norm).astype(np.float32))
    d = float(offset) / norm

    def f(p: torch.Tensor) -> torch.Tensor:
        return (p * n32.to(p.device)).sum(dim=-1) + d
    return f


def sdf_gable_roof(width: float, depth: float, height: float, roof_height: float,
                   center_xz: tuple = (0.0, 0.0)) -> SDF:
    """Box with a two-sloped (gable) roof on top, ridge along the X axis.

    Building base: y in [0, height], xz extent (width, depth) centered at center_xz.
    Roof: y in [height, height + roof_height], ridge runs along X at z = center_z.
    """
    hw, hd = width / 2.0, depth / 2.0
    cx, cz = center_xz
    body_he = (hw, height / 2.0, hd)
    body_off = (cx, height / 2.0, cz)
    body = sdf_translate(sdf_box(body_he), body_off)

    # Gable = two half-planes intersected with a slab.
    # Each slope: plane through (cx, height, cz +/- hd) with normal sloping inward+up.
    # SDF of a slanted half-space: dot(p - p0, normal) (positive outside).
    # Slope angle from horizontal:
    angle = math.atan2(roof_height, hd)
    nx, ny, nz = 0.0, math.cos(angle), math.sin(angle)
    # plane 1 normal: (0, ny, -nz) anchored at (cx, height, cz + hd)
    # plane 2 normal: (0, ny, +nz) anchored at (cx, height, cz - hd)
    def slope1(p: torch.Tensor) -> torch.Tensor:
        return (p[..., 1] - height) * ny + (-(p[..., 2] - (cz + hd))) * nz - 0.0
    def slope2(p: torch.Tensor) -> torch.Tensor:
        return (p[..., 1] - height) * ny + ((p[..., 2] - (cz - hd))) * nz - 0.0
    # Roof prism = the intersection of slopes (gable wedge), capped to the building's xz extent.
    # Capped by box of dimensions (width, roof_height, depth) centered at (cx, height + roof_height/2, cz).
    cap = sdf_translate(sdf_box((hw, roof_height / 2.0, hd)),
                        (cx, height + roof_height / 2.0, cz))
    def roof(p: torch.Tensor) -> torch.Tensor:
        return torch.maximum(torch.maximum(slope1(p), slope2(p)), cap(p))

    return sdf_union(body, roof)


def sdf_hip_roof(width: float, depth: float, height: float, roof_height: float,
                 center_xz: tuple = (0.0, 0.0)) -> SDF:
    """Body + pyramidal hip roof (apex at center, sloping down to all four edges)."""
    hw, hd = width / 2.0, depth / 2.0
    cx, cz = center_xz
    body = sdf_translate(sdf_box((hw, height / 2.0, hd)), (cx, height / 2.0, cz))
    # Pyramid via intersection of 4 slanted half-spaces, capped to a box.
    angle_x = math.atan2(roof_height, hw)
    angle_z = math.atan2(roof_height, hd)
    nx_x, ny_x = math.sin(angle_x), math.cos(angle_x)
    nz_z, ny_z = math.sin(angle_z), math.cos(angle_z)
    cap = sdf_translate(sdf_box((hw, roof_height / 2.0, hd)),
                        (cx, height + roof_height / 2.0, cz))

    def s1(p: torch.Tensor) -> torch.Tensor:
        return (p[..., 1] - height) * ny_x + (p[..., 0] - (cx + hw)) * nx_x

    def s2(p: torch.Tensor) -> torch.Tensor:
        return (p[..., 1] - height) * ny_x + (-(p[..., 0] - (cx - hw))) * nx_x

    def s3(p: torch.Tensor) -> torch.Tensor:
        return (p[..., 1] - height) * ny_z + (p[..., 2] - (cz + hd)) * nz_z

    def s4(p: torch.Tensor) -> torch.Tensor:
        return (p[..., 1] - height) * ny_z + (-(p[..., 2] - (cz - hd))) * nz_z

    def roof(p: torch.Tensor) -> torch.Tensor:
        d = torch.maximum(s1(p), s2(p))
        d = torch.maximum(d, s3(p))
        d = torch.maximum(d, s4(p))
        return torch.maximum(d, cap(p))

    return sdf_union(body, roof)


# --- boolean operations ------------------------------------------------------

def sdf_union(*sdfs: SDF) -> SDF:
    def f(p: torch.Tensor) -> torch.Tensor:
        d = sdfs[0](p)
        for s in sdfs[1:]:
            d = torch.minimum(d, s(p))
        return d

    return f


def sdf_intersect(*sdfs: SDF) -> SDF:
    def f(p: torch.Tensor) -> torch.Tensor:
        d = sdfs[0](p)
        for s in sdfs[1:]:
            d = torch.maximum(d, s(p))
        return d

    return f


def sdf_subtract(a: SDF, b: SDF) -> SDF:
    def f(p: torch.Tensor) -> torch.Tensor:
        return torch.maximum(a(p), -b(p))

    return f


def sdf_smooth_union(a: SDF, b: SDF, k: float) -> SDF:
    """Polynomial smooth-min (Inigo Quilez). k controls blend radius."""
    def f(p: torch.Tensor) -> torch.Tensor:
        da, db = a(p), b(p)
        h = (k - (da - db).abs()).clamp_min(0.0) / k
        return torch.minimum(da, db) - h * h * k * 0.25

    return f


def sdf_smooth_subtract(a: SDF, b: SDF, k: float) -> SDF:
    def f(p: torch.Tensor) -> torch.Tensor:
        da, db = a(p), b(p)
        h = (k - (da + db).abs()).clamp_min(0.0) / k
        return torch.maximum(da, -db) + h * h * k * 0.25

    return f


# --- transforms --------------------------------------------------------------

def sdf_translate(sdf: SDF, offset: Sequence[float]) -> SDF:
    off = torch.tensor(offset, dtype=torch.float32)

    def f(p: torch.Tensor) -> torch.Tensor:
        return sdf(p - off.to(p.device))

    return f


def sdf_scale(sdf: SDF, factor: float) -> SDF:
    """Uniform scale: f'(p) = s * f(p/s). Preserves Lipschitz constant."""
    s = float(factor)

    def f(p: torch.Tensor) -> torch.Tensor:
        return s * sdf(p / s)

    return f


def sdf_rotate_y(sdf: SDF, angle_deg: float) -> SDF:
    """Rotate about the +y axis (turn the building horizontally)."""
    a = math.radians(angle_deg)
    cos_a, sin_a = math.cos(a), math.sin(a)
    R = torch.tensor([[cos_a, 0.0, -sin_a], [0.0, 1.0, 0.0], [sin_a, 0.0, cos_a]],
                     dtype=torch.float32)

    def f(p: torch.Tensor) -> torch.Tensor:
        Rt = R.T.to(p.device)
        return sdf(p @ Rt.T)  # rotate inverse on the query point

    return f


# --- sampling + marching cubes ----------------------------------------------

def sample_grid(sdf: SDF, resolution: int, bbox: tuple, device: str = "cuda",
                chunk: int = 1 << 20) -> torch.Tensor:
    """Sample sdf on a (resolution^3) grid covering bbox = (x0,y0,z0,x1,y1,z1).

    Returns Tensor[resolution, resolution, resolution] = SDF values, indexed as
    grid[z, y, x] (matches preprocess/create_sdf.py: (D=z, H=y, W=x)).
    """
    x0, y0, z0, x1, y1, z1 = bbox
    xs = torch.linspace(x0, x1, resolution, device=device)
    ys = torch.linspace(y0, y1, resolution, device=device)
    zs = torch.linspace(z0, z1, resolution, device=device)
    Z, Y, X = torch.meshgrid(zs, ys, xs, indexing="ij")  # (D, H, W) each
    pts = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)
    out = torch.empty(pts.shape[0], device=device)
    with torch.no_grad():
        for i in range(0, pts.shape[0], chunk):
            out[i:i + chunk] = sdf(pts[i:i + chunk])
    return out.reshape(resolution, resolution, resolution)  # (D, H, W)


def grid_to_mesh(grid: torch.Tensor, bbox: tuple, iso: float = 0.0):
    """Marching cubes on a (D, H, W) SDF grid. Returns trimesh.Trimesh in world
    coordinates spanning bbox.
    """
    from skimage import measure
    import trimesh as _trimesh

    g = grid.detach().cpu().numpy().astype(np.float32)
    D, H, W = g.shape
    x0, y0, z0, x1, y1, z1 = bbox
    spacing = ((z1 - z0) / max(D - 1, 1),
               (y1 - y0) / max(H - 1, 1),
               (x1 - x0) / max(W - 1, 1))
    try:
        verts, faces, _normals, _vals = measure.marching_cubes(g, level=iso, spacing=spacing)
    except (ValueError, RuntimeError):
        return None
    # verts come out as (z, y, x). Translate to bbox origin and reorder to (x, y, z).
    verts[:, 0] += z0
    verts[:, 1] += y0
    verts[:, 2] += x0
    verts = verts[:, [2, 1, 0]]
    return _trimesh.Trimesh(vertices=verts, faces=faces, process=False)


# --- utilities ---------------------------------------------------------------

def polygon_bbox_with_pad(polygon_xz: np.ndarray, target_height: float, pad: float) -> tuple:
    """Return (x0, y0, z0, x1, y1, z1) bbox for sampling, padded around the
    polygon's XZ extent and the building height."""
    poly = np.asarray(polygon_xz, dtype=np.float32)
    x_min, z_min = poly.min(axis=0).tolist()
    x_max, z_max = poly.max(axis=0).tolist()
    pw = x_max - x_min
    pd = z_max - z_min
    p_xz = float(pad) * max(pw, pd)
    p_y = float(pad) * target_height
    return (
        x_min - p_xz, -p_y, z_min - p_xz,
        x_max + p_xz, target_height + p_y, z_max + p_xz,
    )
