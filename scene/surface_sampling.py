"""Seam 2 (spec #68): turn a building mesh into the point streams a vecset encoder consumes.

A query-based encoder does not read a grid -- it reads points on the surface, each with a normal. This
module is the one place that conversion happens, kept pure and CPU-only so it is testable without a
model or a GPU.

Two streams, matching the vecset recipe:
  * **coarse** -- uniform surface coverage
  * **sharp**  -- concentrated on high-dihedral-angle edges, because uniform sampling alone is the
    documented cause of lost sharp detail in this family, and LoD2 massing is mostly flat faces meeting
    at hard edges

**Outward normals are enforced here, deliberately.** A reflection anywhere upstream (the Frame-N y/z
swap is one) silently inverts winding, and an encoder fed inside-out normals degrades without erroring
-- it cost a full round of measurements before it was caught. Signed-distance paths never notice, because
fast-winding-number signing is orientation-agnostic, so this cannot be delegated to them.

**There are TWO mesh frames in this project and they are not interchangeable** -- see `to_array_frame`.
Mixing them cost both A2 training runs, so the conversion lives here as one named function rather than as
an axis-reversal open-coded at each call site.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

SHARP_DEG = 25.0


def to_array_frame(verts: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Frame-N mesh -> the ARRAY frame the encode/decode/eval path uses. Returns (verts, faces).

    Two frames exist and both are legitimate:

    * **Frame-N** -- what `ingest_surfaces.to_frame_n` writes and what `building_to_sdf` normalises to.
      The recovered LoD2 surface corpus is in this frame.
    * **the array frame** -- the stored 64^3 SDF is indexed ``[z, y, x]``, so a mesh extracted from it
      (marching cubes + `verts_to_world`) has its coordinate components in that same ``(z, y, x)`` order.
      `grid_points` queries in this order too, so `decode_grid` returns a field directly comparable to the
      stored SDF. **This is the frame the generator, the codec and the evaluation harness all speak.**

    They differ by an x<->z swap, i.e. reversing the coordinate columns.

    ⚠️ That swap is a **reflection** (determinant -1), so it reverses face orientation. Winding is flipped
    here to compensate; `ensure_outward` in the samplers is the backstop, not the fix.

    Why this exists: `precompute_vecset_latents.py` encoded blockouts in the array frame and real
    surfaces in Frame-N, so aligned-pair training learned "blockout -> the TRANSPOSED building". The
    latents were sound and sat on the codec ceiling; only their orientation was wrong. Keeping the
    conversion in one named place is the cheap half of not repeating that.
    """
    v = np.ascontiguousarray(np.asarray(verts, np.float64)[:, ::-1])
    f = np.ascontiguousarray(np.asarray(faces)[:, ::-1])
    return v, f


def ensure_outward(mesh):
    """Return `mesh` wound so face normals point outward, flipping it if the volume is negative.

    Cheap, and the one guard that stops an upstream reflection from poisoning every encoding.
    """
    if mesh.volume < 0:
        mesh = mesh.copy()
        mesh.invert()
    return mesh


def sample_uniform(mesh, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Uniform surface points with outward normals -> (n, 6) as [x, y, z, nx, ny, nz].

    ⚠️ `rng` is honoured, and that is load-bearing (#88). This function previously accepted `rng` and
    **ignored** it, calling `sample_surface` with no seed -- which draws from numpy's GLOBAL generator.
    Since the coarse stream is the bulk of what the encoder sees, every latent cached before this fix
    is a function of global interpreter state at write time, and reseeding the codec could not have
    helped because the codec's generator was never in charge. `seed=` takes a Generator and advances
    it in place, so a caller holding one gets a reproducible stream that no unrelated numpy call can
    perturb.
    """
    import trimesh
    rng = rng if rng is not None else np.random.default_rng(0)
    mesh = ensure_outward(mesh)
    pts, fid = trimesh.sample.sample_surface(mesh, n, seed=rng)
    return np.concatenate([pts, mesh.face_normals[fid]], axis=1).astype(np.float32)


def sample_sharp(mesh, n: int, rng: Optional[np.random.Generator] = None,
                 deg: float = SHARP_DEG) -> np.ndarray:
    """Points along sharp edges -> (n, 6), normal = mean of the two adjacent face normals.

    Edges are selected by dihedral angle, so no external modelling application is required, and are
    sampled in proportion to length so long edges are represented fairly. A mesh with no edge above the
    threshold falls back to uniform coverage rather than raising -- a smooth shape should still encode.
    """
    rng = rng or np.random.default_rng(0)
    mesh = ensure_outward(mesh)
    ang = mesh.face_adjacency_angles
    keep = ang > np.deg2rad(deg)
    if not keep.any():
        return sample_uniform(mesh, n, rng)

    e = mesh.face_adjacency_edges[keep]
    fn = mesh.face_normals[mesh.face_adjacency[keep]]
    en = fn.mean(axis=1)
    en /= np.linalg.norm(en, axis=1, keepdims=True).clip(1e-9)

    a, b = mesh.vertices[e[:, 0]], mesh.vertices[e[:, 1]]
    w = np.linalg.norm(b - a, axis=1)
    if w.sum() <= 0:
        return sample_uniform(mesh, n, rng)
    idx = rng.choice(len(e), size=n, p=w / w.sum())
    t = rng.random((n, 1))
    return np.concatenate([a[idx] * (1 - t) + b[idx] * t, en[idx]], axis=1).astype(np.float32)


def sample_streams(mesh, n_coarse: int = 8192, n_sharp: int = 8192,
                   rng: Optional[np.random.Generator] = None,
                   deg: float = SHARP_DEG) -> Tuple[np.ndarray, np.ndarray]:
    """The pair a vecset encoder wants: (coarse, sharp), each (n, 6) with outward normals."""
    rng = rng or np.random.default_rng(0)
    return sample_uniform(mesh, n_coarse, rng), sample_sharp(mesh, n_sharp, rng, deg)
