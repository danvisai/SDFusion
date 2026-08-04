"""HOW IT LOOKS WIRED — exterior facade-grammar coherent-add (mesh).

  ① base building with its facade GRID (floors x bays, exterior-by-construction)
  ② user drops a MOLDY primitive (crude protruding blob)
  ③ it snaps to the nearest EXTERIOR cell -> a grid-aligned window. It physically cannot go
     inside (every cell is on a wall plane) -> fixes the old interior-pull / holes.

  env -u LD_PRELOAD -u LD_LIBRARY_PATH PYTHONPATH=. \
    /tmp/sdfusion_venv/bin/python scripts/foundations/facade_grammar_sheet.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts" / "foundations"))
sys.path.insert(0, str(REPO / "scripts" / "server"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from coherent_add_wired_sheet import box, compose, render, R  # reuse mesh helpers
import facade_grammar as fg

OUT = REPO / "outputs/part_set_refiner"
# coarse, BIG windows so individual cells are clearly visible
GP = dict(floor_pitch=0.46, bay_spacing=0.34, win_h=0.17, win_w=0.14, margin=0.17, depth=0.11)
_EDGES = [(0,1),(0,2),(1,3),(2,3),(4,5),(4,6),(5,7),(6,7),(0,4),(1,5),(2,6),(3,7)]


def c2v(c):
    return (np.asarray(c, float) + 1) / 2 * (R - 1)          # cube [-1,1] -> voxel [0,R-1]


def mark_box(ax, center, half, color="red", lw=2.6):
    c, hv = c2v(center), np.asarray(half, float) * (R - 1) / 2
    lo, hi = c - hv * 1.25, c + hv * 1.25
    p = [[lo[0],lo[1],lo[2]],[hi[0],lo[1],lo[2]],[lo[0],hi[1],lo[2]],[hi[0],hi[1],lo[2]],
         [lo[0],lo[1],hi[2]],[hi[0],lo[1],hi[2]],[lo[0],hi[1],hi[2]],[hi[0],hi[1],hi[2]]]
    for a, b in _EDGES:
        seg = np.array([p[a], p[b]])[:, [0, 2, 1]]
        ax.add_collection3d(Line3DCollection([seg], colors=color, linewidths=lw))


def arrow(ax, p0, p1, color="tab:orange"):
    a, b = c2v(p0)[[0, 2, 1]], c2v(p1)[[0, 2, 1]]
    ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color=color, lw=2.0, ls="--")
    ax.scatter([a[0]], [a[1]], [a[2]], color=color, s=24)


def main():
    block = ([-0.55, -0.95, -0.35], [0.55, 0.7, 0.35])
    grid = [(np.asarray(o["center"]), np.asarray(o["size"]))
            for o in fg.full_facade_ops(block, GP)]
    moldies = [(np.array([0.85, 0.45, 0.18], np.float32), np.array([0.13, 0.12, 0.15], np.float32)),
               (np.array([0.30, -0.6, 0.9], np.float32), np.array([0.12, 0.10, 0.16], np.float32)),
               (np.array([-0.9, 0.28, -0.2], np.float32), np.array([0.14, 0.16, 0.13], np.float32))]

    base = compose(block, grid, [])
    fig = plt.figure(figsize=(3.0 * 3, 2.9 * len(moldies)))
    for r_, (mc, mh) in enumerate(moldies):
        op = fg.coherent_add(block, mc, mh, mode="subtract", params=GP)
        sc = np.asarray(op["center"]); snap = (sc, np.asarray(op["size"]))
        moldy = compose(block, grid, [(mc, mh)])
        coh = compose(block, grid + [snap], [])
        render(fig.add_subplot(len(moldies), 3, 3 * r_ + 1, projection="3d"), base,
               "① base + facade grid" if r_ == 0 else "")
        ax2 = fig.add_subplot(len(moldies), 3, 3 * r_ + 2, projection="3d")
        render(ax2, moldy, "② user drops a MOLDY blob" if r_ == 0 else "")
        mark_box(ax2, mc, mh, "tab:orange")                  # ring the dropped blob
        ax3 = fig.add_subplot(len(moldies), 3, 3 * r_ + 3, projection="3d")
        render(ax3, coh, f"③ snapped here → {op['det']}" if r_ == 0 else "")
        mark_box(ax3, sc, np.asarray(op["size"]), "red")     # RING the snapped cell
        arrow(ax3, mc, sc)                                   # dropped -> snapped
    fig.suptitle("Exterior facade-grammar coherent-add: dropped blob (orange) → snapped to the "
                 "nearest EXTERIOR cell (red) = grid-aligned window", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "facade_grammar_wired_sheet.png", dpi=120); plt.close(fig)
    print(f"[save] {OUT/'facade_grammar_wired_sheet.png'}")


if __name__ == "__main__":
    main()
