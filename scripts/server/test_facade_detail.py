"""Validate procedural facade detail: boxy recipe building -> windowed/banded building.

Renders base | +detail for several footprints, and confirms the detail keeps the SDF a
valid solid (interior not opened).
"""

from __future__ import annotations

import sys
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
from scene.sdf_detail import add_facade_detail, DetailParams
from scene.sdf_primitives import sample_grid, grid_to_mesh


def render(ax, mesh, title):
    if mesh is None or len(mesh.faces) == 0:
        ax.set_title(title + "\n(empty)", fontsize=8); return
    V, F = mesh.vertices, mesh.faces; tris = V[F]
    fy = tris[:, :, 1].mean(1); c = plt.cm.bone(0.25 + 0.6 * (fy - fy.min()) / (np.ptp(fy) + 1e-9))
    ax.add_collection3d(Poly3DCollection(tris[:, :, [0, 2, 1]], facecolors=c,
                                         edgecolors="k", linewidths=0.05))
    x, z, y = V[:, 0], V[:, 2], V[:, 1]
    ax.set_xlim(x.min(), x.max()); ax.set_ylim(z.min(), z.max()); ax.set_zlim(0, max(y.max(), 1))
    try: ax.set_box_aspect((np.ptp(x), np.ptp(z), max(np.ptp(y), 1)))
    except Exception: pass
    ax.view_init(elev=14, azim=-52); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title(title, fontsize=8)


def main():
    out = REPO / "outputs/facade_detail"; out.mkdir(parents=True, exist_ok=True)
    eng = RecipeInferenceEngine(); dev = eng.device

    cases = [
        ("modern office", "modern", "COMMERCIAL", [[-7, -11], [7, -11], [7, 11], [-7, 11]], 22.0),
        ("modern house", "modern", "RESIDENTIAL", [[-6, -8], [6, -8], [6, 8], [-6, 8]], 9.0),
        ("contemporary", "contemporary", "COMMERCIAL", [[-8, -8], [8, -8], [8, 8], [-8, 8]], 16.0),
        ("public_civic", "public_civic", "PUBLIC", [[-10, -7], [10, -7], [10, 7], [-10, 7]], 14.0),
    ]
    fig = plt.figure(figsize=(8, 4 * len(cases)))
    for r, (label, style, cls, fp, H) in enumerate(cases):
        params = eng.sample_params(fp, H, cls, style, seed=1)
        base = recipe_base_sdf(style, params, fp, H, device=dev)
        detailed = add_facade_detail(base, fp, H, DetailParams())
        poly = np.asarray(fp); pad = 1.5
        bbox = (poly[:, 0].min() - pad, 0.0, poly[:, 1].min() - pad,
                poly[:, 0].max() + pad, H + 1.5, poly[:, 1].max() + pad)
        R = 112
        gm = grid_to_mesh(sample_grid(base, R, bbox, device=dev), bbox, 0.0)
        dm = grid_to_mesh(sample_grid(detailed, R, bbox, device=dev), bbox, 0.0)
        wt = dm.is_watertight if dm else False
        print(f"  {label:16s} base {len(gm.vertices):6d}v -> detailed {len(dm.vertices):6d}v "
              f"faces {len(dm.faces):6d} watertight={wt}")
        for c, (m, t) in enumerate([(gm, f"{label}\nrecipe base (boxy)"),
                                    (dm, f"{label}\n+ procedural detail")]):
            ax = fig.add_subplot(len(cases), 2, r * 2 + c + 1, projection="3d")
            render(ax, m, t)
    fig.suptitle("Procedural facade detail: windows + floor bands + cornice + plinth", fontsize=12)
    fig.tight_layout(); fig.savefig(out / "facade_detail.png", dpi=115); plt.close(fig)
    print(f"[save] {out/'facade_detail.png'}")


if __name__ == "__main__":
    main()
