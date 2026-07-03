"""Layer 2.5a — procedural geometric weathering (cracks / edge wear / surface erosion).

Ages a building by displacing its FINAL detailed SDF grid (post composer/CSG, pre marching
cubes). Pure procedure — needs NO data, stays inside the crisp-geometry doctrine (the surface
is still rendered from an SDF we constructed; nothing is synthesized by a net), and is
deterministic per (seed, intensity), so a building's weathering is part of its symbolic state
and survives rebuild/export exactly.

Three meters-calibrated effects, all limited to a band around the surface:
  1. EDGE WEAR   — convex edges/corners chip first. On an SDF the Laplacian ≈ mean curvature
                   at the surface (positive on convex features), so eroding proportionally to
                   clip(∇²sdf, 0, ·) rounds exactly the sharp exposed features, modulated by
                   noise so wear is uneven.
  2. SURFACE EROSION — fBm noise displacement, biased positive (material loss -> pitted
                   stone), stronger near the ground (splash/wear zone).
  3. CRACKS      — Worley (cellular) F2-F1 ridges carve thin channels near cell borders,
                   gated by a large-scale noise patch mask so cracking is patchy, and applied
                   only within `crack_depth` of the surface so the core is never hollowed.

Grid convention matches the pipeline: (D=z, H=y, W=x), SDF values in METERS, per-axis voxel
sizes in meters (the sample bboxes are anisotropic). Total erosion is clipped so moderate
intensities age a building without destroying composed detail (windows are 1-2 m; max wear
at intensity 1.0 is ~0.35 m).
"""
from __future__ import annotations

import numpy as np


def _fbm(shape, voxel, wavelengths_m, rng):
    """Multi-octave value noise on an anisotropic grid: seeded coarse lattices trilinearly
    upsampled to `shape` (exact, via torch interpolate). Returned ~N(0,1), clipped to [-2,2]."""
    import torch
    import torch.nn.functional as F
    out = np.zeros(shape, np.float32)
    total = 0.0
    for i, wl in enumerate(wavelengths_m):
        amp = 0.5 ** i
        coarse = [max(2, int(round(shape[a] * voxel[a] / wl)) + 1) for a in range(3)]
        c = rng.standard_normal(coarse).astype(np.float32)
        t = F.interpolate(torch.from_numpy(c)[None, None], size=tuple(shape),
                          mode="trilinear", align_corners=True)[0, 0].numpy()
        out += amp * t
        total += amp
    out /= total
    sd = float(out.std())
    if sd > 1e-6:
        out /= sd
    return np.clip(out, -2.0, 2.0)


def _worley_ridge(shape, voxel, cell_m, rng):
    """F2-F1 of a jittered cellular lattice (meters). Small near cell borders -> crack lines.
    `voxel` is AXIS-ordered (vz, vy, vx) to match the (D=z, H=y, W=x) grid layout."""
    D, H, W = shape
    vz, vy, vx = voxel
    # lattice sized to the volume, padded one cell each side
    nz = int(np.ceil(D * vz / cell_m)) + 2
    ny = int(np.ceil(H * vy / cell_m)) + 2
    nx = int(np.ceil(W * vx / cell_m)) + 2
    pts = rng.random((nz, ny, nx, 3)).astype(np.float32)          # jitter inside each cell
    zz, yy, xx = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing="ij")
    # voxel positions in CELL units (continuous), shifted +1 for the pad
    cz = (zz * vz / cell_m + 1.0).astype(np.float32)
    cy = (yy * vy / cell_m + 1.0).astype(np.float32)
    cx = (xx * vx / cell_m + 1.0).astype(np.float32)
    bz, by, bx = np.floor(cz).astype(np.int32), np.floor(cy).astype(np.int32), np.floor(cx).astype(np.int32)
    f1 = np.full(shape, np.inf, np.float32)
    f2 = np.full(shape, np.inf, np.float32)
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                iz = np.clip(bz + dz, 0, nz - 1)
                iy = np.clip(by + dy, 0, ny - 1)
                ix = np.clip(bx + dx, 0, nx - 1)
                p = pts[iz, iy, ix]                                # (D,H,W,3) point-in-cell
                d = np.sqrt((iz + p[..., 0] - cz) ** 2
                            + (iy + p[..., 1] - cy) ** 2
                            + (ix + p[..., 2] - cx) ** 2) * cell_m
                closer = d < f1
                f2 = np.where(closer, f1, np.minimum(f2, d))
                f1 = np.where(closer, d, f1)
    return f2 - f1


def weather_grid(grid, voxel_m, seed=0, intensity=0.5, y0_m=0.0,
                 edge_amp=0.16, noise_amp=0.11, crack_depth=0.32, cell_m=2.2):
    """Weather a (D=z,H=y,W=x) SDF grid (METERS). voxel_m=(vx,vy,vz) meters/voxel;
    y0_m = world y of grid index 0 along H (ground bias). Returns a new float32 grid.

    intensity in [0,1]: 0 = untouched, ~0.3 = lightly aged stone, 1.0 = heavily worn ruin
    (max total erosion ~0.35 m — composed windows/doors survive)."""
    from scipy.ndimage import laplace
    g = np.asarray(grid, np.float32)
    if intensity <= 0:
        return g.copy()
    intensity = float(min(intensity, 1.0))
    shape = g.shape
    vx, vy, vz = (float(v) for v in voxel_m)
    voxel = (vz, vy, vx)                                          # axis order of (D,H,W)
    mvox = (vx + vy + vz) / 3.0
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)

    # surface band: effects fade out |band| away from the surface
    band = max(2.0 * mvox, 0.45)
    band_w = np.clip(1.0 - np.abs(g) / band, 0.0, 1.0)

    # ground bias: wear concentrates low on the walls (splash/contact zone)
    y_m = y0_m + np.arange(shape[1], dtype=np.float32) * vy
    ground_w = (1.0 + 0.9 * np.exp(-np.clip(y_m, 0, None) / 2.5))[None, :, None]

    # 1. edge wear — SDF Laplacian ~ curvature * vox^2; positive at convex edges
    curv = laplace(g) / (mvox * mvox)
    k_max = 1.0 / (2.0 * mvox)                                   # sharpest representable edge
    edge_noise = 0.35 + 0.65 * (0.5 + 0.5 * _fbm(shape, voxel, (1.6, 0.8), rng))
    wear = (intensity * edge_amp) * np.clip(curv, 0.0, k_max) / k_max * edge_noise

    # 2. surface erosion — fBm biased toward material loss, boosted near the ground
    n = _fbm(shape, voxel, (2.4, 1.2, 0.6), rng)
    erosion = (intensity * noise_amp) * (0.65 * n + 0.35)

    # 3. cracks — carve where the Worley ridge is thin, in large-scale patches only,
    #    and never deeper into the body than crack_depth from the surface
    depth = intensity * crack_depth
    width = max(0.6 * mvox, 0.05)
    ridge = _worley_ridge(shape, voxel, cell_m, rng)
    channel = np.clip((width - ridge) / width, 0.0, 1.0)
    patch = np.clip((_fbm(shape, voxel, (5.0,), rng) + 0.4) / 0.8, 0.0, 1.0)
    not_deep = np.clip(1.0 + g / (depth + mvox), 0.0, 1.0)       # 0 deep inside, 1 at surface
    cracks = depth * channel * patch * not_deep

    offset = (wear + np.maximum(erosion, 0.0) * ground_w + cracks) * band_w \
        + np.minimum(erosion, 0.0) * 0.35 * band_w               # slight accretion -> pitted
    offset = np.clip(offset, -0.08, 0.35 * intensity + 1e-6)
    return (g + offset).astype(np.float32)
