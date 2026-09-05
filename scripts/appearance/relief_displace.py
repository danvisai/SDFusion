"""Measure where a surface's geometry sits on the spatial-frequency scale, in metres.

CONTEXT.md defines massing as "low-spatial-frequency geometry above `s*`" and detail as
"high-spatial-frequency geometry below `s*`". That is a band split, so this measures a band split.
Written for #136 and kept because nothing else in the repo can do it: `measure_scale_spectrum.py`
scores per-semantic-label instance sizes on labelled BuildingNet part meshes and needs
`outputs/part_labels_full/`, and `surface_roughness` is a grid scalar with no length in it at all
(map #34 already recorded 2 scalar metrics failing to separate form).

⚠️ **Read the history before reusing the mesh helpers.** This module was written to argue that
detail should be carried on a mesh because "a voxel grid cannot represent sub-`s*` geometry". That
premise is FALSE for this corpus and the claim is withdrawn. Measured over the 714 pinned buildings
in `data/real_massing_v1/real.h5`, using each building's own `height_m` and its occupied extent
(every building normalises to a max extent of 60 voxels, so the pitch is isotropic):

    64^3  -> 0.204 m/voxel (median) -> resolves 0.41 m     |  s* is 1.0 m
    96^3  -> 0.136 m/voxel          -> resolves 0.27 m     |  only 3.9% of buildings
    128^3 -> 0.102 m/voxel          -> resolves 0.20 m     |  cannot reach s* at 64^3

The grid is far below `s*` already. `remesh_isotropic` and `displace_along_normals` are kept
because they are correct and tested, not because the route they were built for survived; a facade
at 0.15 m edges costs ~215k faces per building against ~200 for the massing itself.

Contents:

  * `radial_psd`, `peak_wavelength`   -- the spectrum, in metres.
  * `sub_s_star_fraction`             -- share of variance below `s*`. ⚠️ A SHARE, not a quantity.
  * `carries_feature`                 -- did the surface carry the feature it was asked for, or
                                         alias it into another band? Never read the share alone.
  * `edge_length_for`                 -- Nyquist sizing, verified rather than assumed.
  * `remesh_isotropic`                -- coarse-to-fine, per RADmesh (arXiv 2608.17182).
  * `displace_along_normals`          -- inward-only, so spill cannot be created by construction.
  * `sample_displacement`             -- achieved displacement of a planar wall, from the geometry.
"""
from __future__ import annotations

import numpy as np

S_STAR_M = 1.0                      # ADR 0004: 1.0 m = 5 voxels @96^3. Fixed a priori.
COARSE_TO_FINE = (1.7, 1.3, 1.0)    # RADmesh's local-growth schedule, as multipliers of target


def vertex_normals(verts, faces):
    """Area-weighted vertex normals. Same convention as texture_bake.vertex_normals."""
    v = np.asarray(verts, np.float64)
    f = np.asarray(faces, np.int64)
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, f[:, k], fn)                      # area-weighted: fn is not normalized
    n = np.linalg.norm(vn, axis=1, keepdims=True)
    return vn / np.maximum(n, 1e-12)


def remesh_isotropic(verts, faces, target_len, schedule=COARSE_TO_FINE, iterations=3):
    """Coarse-to-fine isotropic remesh to an absolute target edge length.

    `target_len` is in the mesh's own units and is the FINAL target; `schedule` multiplies it, so
    (1.7, 1.3, 1.0) runs three passes ending at `target_len`. Returns (verts, faces).

    Sizing rule: to resolve a feature of size `d` you need edges <= d/2, so pass
    `target_len = d / 2`. Picking it by eye is how you end up with a mesh that cannot carry the
    displacement you are about to apply.
    """
    import pymeshlab
    v = np.asarray(verts, np.float64)
    f = np.asarray(faces, np.int32)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(v, f))
    for mult in schedule:
        ms.meshing_isotropic_explicit_remeshing(
            targetlen=pymeshlab.PureValue(float(target_len) * float(mult)),
            iterations=int(iterations))
    m = ms.current_mesh()
    return np.asarray(m.vertex_matrix(), np.float64), np.asarray(m.face_matrix(), np.int64)


def displace_along_normals(verts, faces, height, amplitude, inward_only=True):
    """Move each vertex along its normal by `height` (per-vertex, any scale) * `amplitude`.

    `height` is normalized to [0, 1] first, so `amplitude` is the achieved peak-to-trough depth in
    the mesh's units and is the only knob that matters for the `s*` comparison.

    `inward_only` (the default) maps the normalized height to [-amplitude, 0] instead of
    [-amplitude/2, +amplitude/2], so no vertex ever moves OUTWARD. Outward motion is spill, spill is
    charged against the allowance, and carving inward keeps footprint fidelity true by construction
    rather than by hoping the projection stayed inside.
    """
    v = np.asarray(verts, np.float64)
    h = np.asarray(height, np.float64).ravel()
    if h.shape[0] != v.shape[0]:
        raise ValueError(f"height has {h.shape[0]} values for {v.shape[0]} vertices")
    lo, hi = float(h.min()), float(h.max())
    hn = np.zeros_like(h) if hi - lo < 1e-12 else (h - lo) / (hi - lo)
    offset = -amplitude * hn if inward_only else amplitude * (hn - 0.5)
    return v + vertex_normals(v, faces) * offset[:, None]


def radial_psd(field, extent_m):
    """Radially averaged power spectrum of a 2-D scalar field.

    `extent_m` is the field's physical width (== height; a square patch is assumed). Returns
    (wavelength_m, power) sorted by DECREASING wavelength, with the DC term dropped -- a constant
    offset is not a feature at any scale, and leaving it in swamps every real band.
    """
    a = np.asarray(field, np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"expected a square 2-D field, got {a.shape}")
    n = a.shape[0]
    a = a - a.mean()
    p = np.abs(np.fft.fftshift(np.fft.fft2(a))) ** 2
    c = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
    rb = r.astype(int)
    keep = (rb > 0) & (rb <= c)                          # drop DC, drop the unsampled corners
    power = np.bincount(rb[keep], weights=p[keep], minlength=c + 1)[1:c + 1]
    cycles = np.arange(1, c + 1)                         # cycles across the whole patch
    return extent_m / cycles, power


def sub_s_star_fraction(field, extent_m, s_star=S_STAR_M):
    """Share of a displacement field's variance sitting at wavelengths BELOW `s*`.

    HIGHER IS BETTER for the detail claim: 1.0 means every bit of the added geometry is
    high-spatial-frequency, i.e. detail by CONTEXT.md's definition; 0.0 means it is all massing.

    🔑🔑 **It is a SHARE of variance, so it CANNOT compare two fields' absolute detail content.**
    A field carrying almost nothing still reports where its own noise sits. Measured: Marigold
    normals on a generated facade scored 0.304 against Depth-Anything's 0.095 -- which reads as a
    3x win and is not one. The normal map was uniform (mean `n_z` 0.975, a flat wall) and the
    implied physical relief was **0.1 mm**. To compare two fields, convert both to a physical
    quantity and compare absolute magnitude in the band; to ask where ONE field's energy sits,
    this is the right number. Read it beside `carries_feature` in either case.

    Returns 0.0 for a flat field (nothing was added, so nothing is detail).
    """
    wav, power = radial_psd(field, extent_m)
    total = power.sum()
    if total <= 0:
        return 0.0
    return float(power[wav < s_star].sum() / total)


def sample_displacement(verts_before, verts_after, plane_origin, plane_u, plane_v,
                        extent_m, res=256):
    """Resample the ACHIEVED displacement of a planar wall onto a regular grid.

    Displacement is along the normal, so every vertex keeps its position in the wall plane: project
    the pre-displacement vertices onto (plane_u, plane_v) to get their (u, v), take the signed
    normal offset each one actually moved, and scatter-interpolate onto a res x res grid.

    This is the honest input to `sub_s_star_fraction`: it measures what the MESH carries, not what
    the height map asked for. A mesh too coarse to represent the pattern will show the loss here.
    """
    from scipy.interpolate import griddata
    b = np.asarray(verts_before, np.float64)
    a = np.asarray(verts_after, np.float64)
    u = np.asarray(plane_u, np.float64) / np.linalg.norm(plane_u)
    v = np.asarray(plane_v, np.float64) / np.linalg.norm(plane_v)
    nrm = np.cross(u, v)
    rel = b - np.asarray(plane_origin, np.float64)
    uu, vv = rel @ u, rel @ v
    d = (a - b) @ nrm
    g = np.linspace(0.0, extent_m, res)
    gu, gv = np.meshgrid(g, g)
    out = griddata(np.column_stack([uu, vv]), d, (gu, gv), method="linear", fill_value=0.0)
    return np.nan_to_num(out, nan=0.0)


def peak_wavelength(field, extent_m):
    """Wavelength (m) carrying the most variance in `field`. No direction -- it is a location."""
    wav, power = radial_psd(field, extent_m)
    if power.sum() <= 0:
        return float("nan")
    return float(wav[int(np.argmax(power))])


def carries_feature(field, extent_m, requested_m, tol=0.25):
    """Did the surface carry the feature it was ASKED for, or alias it into another band?

    🔑 `sub_s_star_fraction` alone cannot answer this, and will happily reward a mesh that failed.
    Measured: an 8 m wall remeshed to 0.316 m edges (a 96^3 marching-cubes surface on a 30 m
    building) and asked for a 0.30 m pattern scores **0.404 sub-`s*`** -- which reads like partial
    success -- while its energy actually peaks at **4.0 m**. The pattern is gone; the score is
    aliasing noise scattered across bands. Asked for 0.15 m it scores 0.491 and peaks at 1.6 m.

    So the gate is a PAIR, and the pair is read together or not at all: the sub-`s*` fraction says
    how much high-frequency energy there is, and this says whether that energy is the requested
    feature. Same lesson as #127 (a scalar hid a mound) and #131 (a median passed while 89.5% of
    buildings grew a visible fin).

    Returns True when the achieved peak is within `tol` (relative) of `requested_m`.
    """
    peak = peak_wavelength(field, extent_m)
    if not np.isfinite(peak):
        return False
    return abs(peak - requested_m) / requested_m <= tol


def edge_length_for(feature_m):
    """Target edge length needed to carry a feature of size `feature_m`, from Nyquist.

    Verified rather than assumed: at 0.125 m edges a 0.30 m pattern lands at 0.296 m (carried) and
    a 0.15 m pattern lands at 0.242 m (aliased). The boundary sits where this rule puts it.
    """
    return float(feature_m) / 2.0
