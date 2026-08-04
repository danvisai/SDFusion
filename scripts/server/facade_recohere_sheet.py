"""Output sheet for GENERATIVE facade re-coherence (the different approach).

Story:
  Row 1  base massing → a BROKEN/incoherent facade (input) → 3 generative corrections
         (same input, different seeds) = coherent + VARIED ("interesting architecture").
  Row 2  one generative correction per style = style-conditioned, and aligned-by-construction
         no matter what's sampled.

  env -u LD_PRELOAD -u LD_LIBRARY_PATH PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
    /tmp/sdfusion_venv/bin/python scripts/server/facade_recohere_sheet.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from recipe_inference import RecipeInferenceEngine
from scene.sdf_edit import recipe_base_sdf
from scene.sdf_primitives import sample_grid, grid_to_mesh
from scene import sdf_detail as det
import facade_recohere as fr

OUT = REPO / "outputs/facade_recohere"
RES = 100


def render(ax, mesh, title, color="bone"):
    if mesh is None or len(mesh.faces) == 0:
        ax.set_title(title + " (empty)", fontsize=7); ax.set_axis_off(); return
    V, F = mesh.vertices, mesh.faces; tris = V[F]
    fy = tris[:, :, 1].mean(1)
    c = getattr(plt.cm, color)(0.25 + 0.6 * (fy - fy.min()) / (np.ptp(fy) + 1e-9))
    ax.add_collection3d(Poly3DCollection(tris[:, :, [0, 2, 1]], facecolors=c,
                                         edgecolors="k", linewidths=0.04))
    x, z, y = V[:, 0], V[:, 2], V[:, 1]
    ax.set_xlim(x.min(), x.max()); ax.set_ylim(z.min(), z.max()); ax.set_zlim(0, max(y.max(), 1))
    try: ax.set_box_aspect((np.ptp(x), np.ptp(z), max(np.ptp(y), 1)))
    except Exception: pass
    ax.view_init(elev=14, azim=-52); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title(title, fontsize=7)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    eng = RecipeInferenceEngine()
    fp = [[-7, -10], [7, -10], [7, 10], [-7, 10]]; H = 20.0
    poly = np.asarray(fp); pad = 1.5
    bbox = (poly[:, 0].min()-pad, 0.0, poly[:, 1].min()-pad,
            poly[:, 0].max()+pad, H+1.5, poly[:, 1].max()+pad)

    def mesh_of(style, vec_or_p):
        params = eng.sample_params(fp, H, "COMMERCIAL", style, seed=1)
        base = recipe_base_sdf(style, params, fp, H, device=dev)
        if vec_or_p is None:
            sdf = base
        else:
            p = vec_or_p if isinstance(vec_or_p, det.DetailParams) else det.vector_to_params(vec_or_p)
            sdf = det.add_facade_detail(base, fp, H, p)
        return grid_to_mesh(sample_grid(sdf, RES, bbox, device=dev), bbox, 0.0)

    t0 = time.time()
    broken = fr.broken_facade_vec(seed=3)            # the incoherent input facade

    # Row 1 — the KNOB: broken input, then increasing correction strength (one seed) →
    # faithful tweak ... fresh coherent. Shows generative correction is a continuous knob.
    row1 = [(mesh_of("modern", broken), "BROKEN facade (input)", "Oranges")]
    for s in (0.3, 0.5, 0.7, 1.0):
        p = fr.recohere_facade(broken, style="modern", strength=s, seed=11, device=dev)
        row1.append((mesh_of("modern", p), f"corrected · strength {s:.1f}", "bone"))

    # Row 2 — generative VARIETY + style conditioning: fresh coherent samples per style
    # (every one aligned-by-construction no matter what's sampled).
    styles = ["modern", "victorian", "industrial", "colonial", "public_civic"]
    row2 = [(mesh_of(s, fr.recohere_facade(None, style=s, seed=31 + j, device=dev)),
             f"{s}", "bone") for j, s in enumerate(styles)]

    ncol = 5
    fig = plt.figure(figsize=(3.0 * ncol, 6.6))
    for i, (m, t, c) in enumerate(row1):
        render(fig.add_subplot(2, ncol, i + 1, projection="3d"), m, t, c)
    for j, (m, t, c) in enumerate(row2):
        render(fig.add_subplot(2, ncol, ncol + 1 + j, projection="3d"), m, t, c)
    fig.suptitle("Generative facade re-coherence — TOP: correction strength knob "
                 "(broken→fresh coherent); BOTTOM: generated per-style facades "
                 "(aligned-by-construction)", fontsize=12)
    fig.tight_layout()
    path = OUT / "facade_recohere_sheet.png"
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"[save] {path}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
