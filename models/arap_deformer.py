"""
ARAP-based mesh deformer guided by a corrected SDF.

Given:
  - retrieved_mesh   trimesh.Trimesh of the retrieved BuildingNet OBJ (hollow, full detail)
  - corrected_sdf    np.ndarray (D, H, W) in (z, y, x) layout, Frame N coords [-1, 1]^3
                     (output of VQVAE.decode_no_quant of the corrected latent)

Compute target positions for "anchor" vertices on the retrieved mesh by
walking each anchor along grad(SDF) toward the corrected iso-surface, then
solve As-Rigid-As-Possible deformation (libigl) to morph the rest of the mesh
coherently while preserving local rigidity (i.e. architectural detail).

Usage:
    from models.arap_deformer import arap_deform
    deformed = arap_deform(retrieved_mesh, corrected_sdf,
                           anchor_threshold=2/64, max_displacement=0.3,
                           n_iters=10)
"""
from __future__ import annotations
from typing import Tuple

import numpy as np
import scipy.ndimage as ndi
import trimesh


# ---------- SDF helpers (Frame N, voxels at [-1, 1]^3) ------------------------

def sdf_at_points(sdf: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Trilinear-interpolate a (D,H,W) SDF (Frame N, axes z,y,x) at world pts (N,3 xyz).

    pts are in Frame N: [-1, 1]^3. The SDF voxel grid is also in Frame N
    (created by preprocess/create_sdf.py with expand_rate=1.3, so it lives in
    [-1.3, 1.3]^3 — but the BuildingNet meshes only occupy ~[-1, 1]^3).
    """
    D, H, W = sdf.shape
    # Map xyz [-1, 1] -> voxel index ranges (z, y, x).
    # Index along the first axis (z) maps to pts[:, 2] (x,y,z layout in pts).
    # We map [-1, 1] -> [0, dim-1].
    xs = (pts[:, 0] + 1.0) * 0.5 * (W - 1)
    ys = (pts[:, 1] + 1.0) * 0.5 * (H - 1)
    zs = (pts[:, 2] + 1.0) * 0.5 * (D - 1)
    coords = np.stack([zs, ys, xs], axis=0)  # ndimage uses axis-major
    return ndi.map_coordinates(sdf, coords, order=1, mode="nearest")


def sdf_grad_at_points(sdf: np.ndarray, pts: np.ndarray, eps: float = 1.0 / 64) -> np.ndarray:
    """Finite-difference gradient of the SDF at given points. Returns (N, 3) in xyz order."""
    grads = np.zeros_like(pts)
    for axis in range(3):
        offset = np.zeros(3); offset[axis] = eps
        gp = sdf_at_points(sdf, pts + offset)
        gm = sdf_at_points(sdf, pts - offset)
        grads[:, axis] = (gp - gm) / (2 * eps)
    return grads


def project_to_iso(sdf: np.ndarray, pts: np.ndarray, n_iters: int = 5,
                   step_clip: float = 0.5) -> np.ndarray:
    """Iterate a few Newton-style steps along the SDF gradient to push points
    onto the iso-surface. step = -s(p) * grad / |grad|^2. Clip step magnitude
    per iteration."""
    p = pts.copy()
    for _ in range(n_iters):
        s = sdf_at_points(sdf, p)
        g = sdf_grad_at_points(sdf, p)
        norm2 = (g * g).sum(axis=1) + 1e-9
        step = -(s[:, None] * g) / norm2[:, None]
        step_norm = np.linalg.norm(step, axis=1, keepdims=True)
        too_big = step_norm > step_clip
        step = np.where(too_big, step * (step_clip / np.maximum(step_norm, 1e-9)), step)
        p = p + step
    return p


# ---------- Anchor selection -------------------------------------------------

def select_anchors(verts: np.ndarray, sdf: np.ndarray,
                   threshold: float = 2.0 / 64,
                   max_anchors: int = 5000) -> np.ndarray:
    """Pick vertex indices whose |sdf| is within `threshold` of the iso-surface.
    These are the vertices we'll constrain to land on the corrected iso-surface.
    Subsamples to `max_anchors` for ARAP efficiency on large meshes."""
    s = np.abs(sdf_at_points(sdf, verts))
    near = np.where(s < threshold)[0]
    # If too few anchors are near, widen until we get something workable.
    width = threshold
    while len(near) < 100 and width < 0.4:
        width *= 1.5
        near = np.where(s < width)[0]
    if len(near) > max_anchors:
        # Subsample uniformly
        rng = np.random.default_rng(0)
        near = rng.choice(near, size=max_anchors, replace=False)
        near.sort()
    return near.astype(np.int64)


def normalize_mesh_to_frame_N(mesh: trimesh.Trimesh,
                              target_extent: float = 2.0) -> Tuple[trimesh.Trimesh, np.ndarray, float]:
    """Center the mesh and scale so the longest axis is `target_extent` (default
    2.0 -> mesh fits in [-1, 1]^3). Returns the (in-place modified) mesh, the
    centroid, and the scale used. Required so the SDF (in Frame N) and the
    mesh share coordinates."""
    v = np.asarray(mesh.vertices, dtype=np.float64).copy()
    centroid = (v.max(0) + v.min(0)) / 2
    v = v - centroid
    extent = float(np.abs(v).max())
    scale = target_extent / 2 / max(extent, 1e-9)
    v = v * scale
    mesh.vertices = v
    return mesh, centroid, scale


# ---------- Main ARAP deformation -------------------------------------------

def _clean_for_arap(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Merge face-local OBJ vertices and drop degenerate/unreferenced geometry."""
    mesh.merge_vertices()
    try:
        mesh.update_faces(mesh.nondegenerate_faces())
    except Exception:
        pass
    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass
    return mesh


def _solve_component_arap(component: trimesh.Trimesh,
                          corrected_sdf: np.ndarray,
                          anchor_threshold: float,
                          max_displacement: float,
                          n_iters: int,
                          max_anchors: int,
                          project_iters: int) -> tuple[trimesh.Trimesh, bool]:
    """Run ARAP on one connected component. Returns the input unchanged on failure."""
    import igl

    if len(component.vertices) < 4 or len(component.faces) < 4:
        return component, False

    V = np.ascontiguousarray(component.vertices, dtype=np.float64)
    F = np.ascontiguousarray(component.faces, dtype=np.int32)

    anchor_idx = select_anchors(V, corrected_sdf, threshold=anchor_threshold,
                                max_anchors=max_anchors)
    if len(anchor_idx) < 10:
        return component, False

    anchor_verts = V[anchor_idx]
    target_pos = project_to_iso(corrected_sdf, anchor_verts, n_iters=project_iters)

    displ = target_pos - anchor_verts
    displ_norm = np.linalg.norm(displ, axis=1)
    keep = displ_norm < max_displacement
    if keep.sum() < 10:
        keep = displ_norm < (max_displacement * 2)
    if keep.sum() < 10:
        return component, False

    anchor_idx = np.ascontiguousarray(anchor_idx[keep], dtype=np.int32)
    target_pos = np.ascontiguousarray(target_pos[keep], dtype=np.float64)

    arap_data = igl.ARAPData()
    arap_data.max_iter = n_iters
    try:
        igl.arap_precomputation(V, F, 3, anchor_idx, arap_data)
        V_new = igl.arap_solve(target_pos, arap_data, V)
    except Exception:
        return component, False

    out = component.copy()
    out.vertices = V_new
    return out, True

def arap_deform(retrieved_mesh: trimesh.Trimesh,
                corrected_sdf: np.ndarray,
                anchor_threshold: float = 2.0 / 64,
                max_displacement: float = 0.3,
                n_iters: int = 10,
                max_anchors: int = 5000,
                project_iters: int = 5,
                normalize: bool = True) -> trimesh.Trimesh:
    """Deform the retrieved mesh so its near-surface vertices land on the
    corrected SDF iso-surface, using ARAP to keep the rest coherent.

    Parameters
    ----------
    retrieved_mesh : trimesh.Trimesh
        The retrieved BuildingNet OBJ. NOT modified in-place if normalize=True.
    corrected_sdf : (D, H, W) float
        Output of VQVAE.decode_no_quant of the corrected latent, in Frame N.
    anchor_threshold : float
        Vertices whose |sdf| < threshold are anchored.
    max_displacement : float
        Cap on per-anchor target displacement (Frame N units). Anchors whose
        target moves further than this are dropped.
    n_iters : int
        ARAP outer iterations.
    max_anchors : int
        Cap on number of anchors to keep ARAP fast on large meshes.
    project_iters : int
        Newton-style projection steps to land anchors on the iso-surface.
    normalize : bool
        If True, work on a Frame-N-normalized copy of the mesh and restore the
        original frame at the end. Default True (the SDF is in Frame N, so
        if the input mesh isn't already normalized, alignment is broken).

    Returns
    -------
    deformed : trimesh.Trimesh
        Same topology as input, with vertices displaced. If normalize=True,
        the deformed mesh is returned in the original input frame.
    """
    # 1. Operate on a copy in Frame N. ARAP requires connected vertex topology
    #    (shared verts between adjacent faces), so merge co-located vertices.
    mesh = retrieved_mesh.copy()
    mesh = _clean_for_arap(mesh)

    if normalize:
        _, centroid, scale = normalize_mesh_to_frame_N(mesh)
    else:
        centroid, scale = np.zeros(3), 1.0

    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int32)

    if len(V) < 4 or len(F) < 4:
        return retrieved_mesh

    # BuildingNet OBJs often store face-local vertices, so even after merging they
    # can remain heavily disconnected. libigl ARAP fails if unconstrained
    # disconnected components are present, so solve component-wise and leave
    # components with too few anchors unchanged.
    components = list(mesh.split(only_watertight=False))
    if not components:
        components = [mesh]

    solved = []
    solved_any = False
    for component in components:
        deformed, ok = _solve_component_arap(
            component,
            corrected_sdf,
            anchor_threshold=anchor_threshold,
            max_displacement=max_displacement,
            n_iters=n_iters,
            max_anchors=max(50, max_anchors // max(len(components), 1)),
            project_iters=project_iters,
        )
        solved.append(deformed)
        solved_any = solved_any or ok

    if solved_any:
        mesh = trimesh.util.concatenate(tuple(solved))

    # 3. Restore original frame
    if normalize:
        mesh.vertices = mesh.vertices / scale + centroid

    return mesh
