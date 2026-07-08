"""Geometry cleanup — remove the noise placed primitives / snaps / CSG leave behind.

Two levels (research-backed: SDF floating-debris removal + mesh connected-component +
vertex-weld cleanup):
  - cleanup_sdf_grid: drop small DISCONNECTED occupied blobs in a (D,H,W) SDF *before*
    meshing, so floaters never become geometry (the main noise source).
  - cleanup_mesh: split into connected components, keep the large ones, weld vertices, drop
    degenerate / duplicate faces, fix normals — kills slivers + leftover fragments.

Both are conservative (keep anything above a fraction of the largest part) so legitimate
detached pieces (a placed wing/annex) survive while specks are removed.
"""
from __future__ import annotations

import numpy as np


def cleanup_sdf_grid(grid, keep_frac=0.04, trunc=None):
    """grid (D,H,W) SDF (inside<0). Remove occupied components smaller than keep_frac of the
    LARGEST; their voxels are pushed positive (outside) so they vanish at iso=0. Returns a
    cleaned copy. No-op if scipy missing or single component."""
    try:
        from scipy.ndimage import label
    except Exception:
        return grid
    g = np.asarray(grid, np.float32).copy()
    occ = g <= 0
    if not occ.any():
        return g
    lab, n = label(occ)
    if n <= 1:
        return g
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    big = sizes.max()
    fill = trunc if trunc is not None else float(max(np.abs(g).max(), 0.2))
    for i in range(1, n + 1):
        if sizes[i] < keep_frac * big:
            g[lab == i] = fill                       # push the speck outside the surface
    return g


def cleanup_mesh(mesh, keep_frac=0.03, weld=True, smooth_iters=0):
    """trimesh -> cleaned trimesh: keep connected components >= keep_frac of the largest
    (by face count), weld near-duplicate vertices, drop degenerate/duplicate faces, fix
    normals. Optional light Taubin. Returns the input unchanged on any failure."""
    import trimesh
    try:
        if mesh is None or len(mesh.faces) == 0:
            return mesh
        comps = mesh.split(only_watertight=False)
        if len(comps) > 1:
            fc = np.array([len(c.faces) for c in comps])
            keep = [c for c, n in zip(comps, fc) if n >= keep_frac * fc.max()]
            mesh = trimesh.util.concatenate(keep) if keep else comps[int(fc.argmax())]
        if weld:
            mesh.merge_vertices()
            mesh.update_faces(mesh.unique_faces())
            mesh.update_faces(mesh.nondegenerate_faces())
            mesh.remove_unreferenced_vertices()
        if smooth_iters > 0:
            trimesh.smoothing.filter_taubin(mesh, iterations=int(smooth_iters))
        try:
            mesh.fix_normals()
        except Exception:
            pass
    except Exception as ex:
        print(f"[cleanup_mesh] skipped ({ex})")
    return mesh
