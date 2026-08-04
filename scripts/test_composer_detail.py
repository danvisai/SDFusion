"""Test the composer->detail glue: a boxy massing per class -> the part-composer decides
elements -> sdf_detail instantiates them. Renders massing vs composed and prints the
composer's per-class decisions (religious should get dome/towers, residential steps, etc.).

CPU-only (does not touch the training GPU).
  env -u LD_PRELOAD -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="" \
    ./sdfusion/bin/python scripts/test_composer_detail.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scene.sdf_primitives import sdf_box, sdf_translate, sample_grid, grid_to_mesh
from scene.composer_detail import PartComposer, compose_detail

OUT = REPO / "outputs/composer_detail_preview"; OUT.mkdir(parents=True, exist_ok=True)

# (class, footprint w x d (m), height m, recipe style for facade character)
CASES = [
    ("RESIDENTIAL", 11, 8, 8.5, "victorian"),
    ("COMMERCIAL", 20, 16, 32, "modern"),
    ("PUBLIC", 28, 22, 17, "public_civic"),
    ("RELIGIOUS", 16, 15, 15, "colonial"),
]


def rect(w, d):
    return [[-w / 2, -d / 2], [w / 2, -d / 2], [w / 2, d / 2], [-w / 2, d / 2]]


def box_massing(footprint, height):
    p = np.asarray(footprint)
    cx, cz = p[:, 0].mean(), p[:, 1].mean()
    wx = (p[:, 0].max() - p[:, 0].min()) / 2
    wz = (p[:, 1].max() - p[:, 1].min()) / 2
    return sdf_translate(sdf_box((wx, height / 2, wz)), (cx, height / 2, cz)), (wx, wz, cx, cz)


def to_mesh(sdf, footprint, height, head_mult, res=128):
    p = np.asarray(footprint)
    x0, z0, x1, z1 = p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()
    pad = 0.12 * max(x1 - x0, z1 - z0) + 1.0
    bbox = (x0 - pad, 0.0, z0 - pad, x1 + pad, height * head_mult, z1 + pad)
    g = sample_grid(sdf, res, bbox, device="cpu")
    return grid_to_mesh(g, bbox, iso=0.0)


def panel(ax, mesh, title):
    if mesh is not None:
        v, fc = np.asarray(mesh.vertices), np.asarray(mesh.faces)
        ax.plot_trisurf(v[:, 0], v[:, 2], fc, v[:, 1], color="#cdb892",
                        edgecolor="none", shade=True)
        lim = [v.min(), v.max()]; ax.set_xlim(lim); ax.set_ylim(lim); ax.set_zlim(lim)
    ax.view_init(elev=18, azim=-55); ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
    ax.set_title(title, fontsize=9)


def main():
    comp = PartComposer(device="cpu")
    fig = plt.figure(figsize=(4 * 2, 3.6 * len(CASES)))
    print("\n[composer->detail] per-class decisions:")
    for r, (cls, w, d, h, style) in enumerate(CASES):
        fp = rect(w, d)
        base, _ = box_massing(fp, h)
        sdf, layout, dec = compose_detail(base, fp, h, cls, style=style, seed=r)
        head = 1.9 if dec["n_towers"] else (1.5 if dec["roof_shape"] != "flat" or dec["dome"] else 1.25)
        m_base = to_mesh(base, fp, h, 1.1)
        m_comp = to_mesh(sdf, fp, h, head)
        print(f"  {cls:12s} glaze={dec['glazing']:.2f} roof={dec['roof_shape']:7s} "
              f"dome={int(dec['dome'])} towers={dec['n_towers']} steps={int(dec['steps'])}")
        panel(fig.add_subplot(len(CASES), 2, r * 2 + 1, projection="3d"), m_base, f"{cls} — massing")
        panel(fig.add_subplot(len(CASES), 2, r * 2 + 2, projection="3d"),
              m_comp, f"{cls} — composed (roof={dec['roof_shape']}, dome={int(dec['dome'])}, tow={dec['n_towers']})")
    fig.suptitle("Composer-driven detail: massing -> learned element placement -> instantiated", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(OUT / "composer_detail.png", dpi=85)
    print(f"\n[saved] {OUT / 'composer_detail.png'}")


if __name__ == "__main__":
    main()
