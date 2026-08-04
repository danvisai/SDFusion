"""Render a grid of the clean procedural training corpus (data/recipe_augmentation_v1).

Rows = the 8 recipe styles, cols = random samples. Marching-cubes each stored 64^3 SDF
and renders it, so you can see exactly what the SDEdit prior is learning from. CPU-only
(does NOT touch the training GPU).
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from scene.sdf_primitives import grid_to_mesh

CORPUS = REPO / "data/recipe_augmentation_v1"
BBOX_N = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
N_COLS = 5
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/corpus_preview.png"

styles = sorted(p.stem for p in CORPUS.glob("*.h5"))
rng = np.random.default_rng(seed)
fig = plt.figure(figsize=(3.0 * N_COLS, 2.9 * len(styles)))

for r, st in enumerate(styles):
    with h5py.File(CORPUS / f"{st}.h5", "r") as f:
        n = f["sdf"].shape[0]
        idxs = sorted(rng.choice(n, size=N_COLS, replace=False).tolist())
        for c, i in enumerate(idxs):
            sdf = torch.from_numpy(f["sdf"][i].astype(np.float32))
            cls = f["class_label"][i].decode() if "class_label" in f else "?"
            h_m = float(f["height_m"][i]) if "height_m" in f else 0.0
            ax = fig.add_subplot(len(styles), N_COLS, r * N_COLS + c + 1, projection="3d")
            m = grid_to_mesh(sdf, BBOX_N, iso=0.0)
            if m is not None:
                v, fc = np.asarray(m.vertices), np.asarray(m.faces)
                ax.plot_trisurf(v[:, 0], v[:, 2], fc, v[:, 1], color="#cdb892",
                                edgecolor="none", linewidth=0, antialiased=True, shade=True)
                lim = [v.min(), v.max()]
                ax.set_xlim(lim); ax.set_ylim(lim); ax.set_zlim(lim)
            ax.view_init(elev=20, azim=-58)
            ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
            if c == 0:
                ax.text2D(-0.15, 0.5, st, transform=ax.transAxes, fontsize=13,
                          rotation=90, va="center", weight="bold")
            ax.set_title(f"{cls} · {h_m:.0f}m", fontsize=8)

fig.suptitle("Clean procedural training corpus — data/recipe_augmentation_v1 (8 styles × 50k)",
             fontsize=14, y=0.995)
fig.tight_layout(rect=(0.01, 0, 1, 0.985))
fig.savefig(out, dpi=85)
print(f"[corpus_preview] {len(styles)} styles -> {out}")
