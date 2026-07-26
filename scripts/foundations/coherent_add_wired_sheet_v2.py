"""Same scenario as coherent_add_wired_sheet.py, but routed through the NEW production function
layout_detail.integrate_new_part() (encoding + resnap_ops_to_surface) instead of a direct
model.refine() + a bespoke snap_to_wall — validates the actual code path before wiring it
further into the server.

  env -u LD_PRELOAD -u LD_LIBRARY_PATH PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
    ./sdfusion/bin/python scripts/foundations/coherent_add_wired_sheet_v2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts" / "foundations"))
sys.path.insert(0, str(REPO / "scripts" / "server"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.measure import marching_cubes

from train_coherent_refiner import box_sdf_64
import layout_detail as ld

OUT = REPO / "outputs/part_set_refiner"
R = 96
_g = np.linspace(-1, 1, R, dtype=np.float32)
GX, GY, GZ = np.meshgrid(_g, _g, _g, indexing="ij")


def box(lo, hi):
    lo, hi = np.asarray(lo), np.asarray(hi)
    c, h = (lo + hi) / 2, (hi - lo) / 2 + 1e-4
    q = np.stack([np.abs(GX - c[0]) - h[0], np.abs(GY - c[1]) - h[1], np.abs(GZ - c[2]) - h[2]], 0)
    return (np.sqrt((np.clip(q, 0, None) ** 2).sum(0)) + np.minimum(q.max(0), 0)).astype(np.float32)


def compose(block, recesses, blobs):
    v = box(*block)
    for c, hw in recesses:
        v = np.maximum(v, -box(np.asarray(c) - hw, np.asarray(c) + hw))
    for c, hw in blobs:
        v = np.minimum(v, box(np.asarray(c) - hw, np.asarray(c) + hw))
    return v


def render(ax, vol, title):
    try:
        v, f, *_ = marching_cubes(vol, level=0.0)
    except Exception:
        ax.set_axis_off(); ax.set_title(title, fontsize=7); return
    tris = v[f][:, :, [0, 2, 1]]
    fy = tris[:, :, 2].mean(1)
    col = plt.cm.bone(0.3 + 0.55 * (fy - fy.min()) / (np.ptp(fy) + 1e-9))
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    ax.add_collection3d(Poly3DCollection(tris, facecolors=col, edgecolors="k", linewidths=0.03))
    for s in (ax.set_xlim, ax.set_ylim): s(0, R)
    ax.set_zlim(0, R); ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=12, azim=-60)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.set_title(title, fontsize=8)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    block = ([-0.55, -0.95, -0.35], [0.55, 0.7, 0.35])
    DEEP, WW, WH = 0.16, 0.07, 0.10
    rows = [-0.55, -0.12, 0.32]
    grid, ops = [], []
    for ry in rows:
        for cx in (-0.3, 0.0, 0.3):
            for c, h in [([cx, ry, 0.35], [WW, WH, DEEP]), ([cx, ry, -0.35], [WW, WH, DEEP])]:
                grid.append((np.array(c, np.float32), np.array(h, np.float32)))
                ops.append(dict(kind="box", center=c, size=h, mode="subtract", det="window",
                                grp=f"w{len(ops)}"))
        for cz in (-0.18, 0.18):
            for c, h in [([0.55, ry, cz], [DEEP, WH, WW]), ([-0.55, ry, cz], [DEEP, WH, WW])]:
                grid.append((np.array(c, np.float32), np.array(h, np.float32)))
                ops.append(dict(kind="box", center=c, size=h, mode="subtract", det="window",
                                grp=f"w{len(ops)}"))
    win_recess = grid
    sdf_vol = box_sdf_64(*block)

    moldies = [np.array([0.85, 0.45, 0.18], np.float32),
               np.array([0.3, -0.6, 0.85], np.float32),
               np.array([-0.85, 0.3, -0.2], np.float32)]

    fig = plt.figure(figsize=(3.0 * 3, 2.9 * len(moldies)))
    for r_, mc in enumerate(moldies):
        mhw = np.array([0.13, 0.12, 0.15], np.float32)
        new_op = dict(kind="box", center=mc.tolist(), size=mhw.tolist(), mode="add",
                      det="window", grp="gNew")
        out_ops, used = ld.integrate_new_part(sdf_vol, ops, new_op, building_class="RESIDENTIAL",
                                              device=dev, strength=0.4, neighbor_k=8)
        new_final = next(o for o in out_ops if o.get("grp") == "gNew")
        rc = np.asarray(new_final["center"], np.float32)
        rhalf = np.asarray(new_final["size"], np.float32)
        print(f"[{r_}] used={used}  moldy={mc.round(2)} -> integrated={rc.round(3)} half={rhalf.round(3)}")

        base = compose(block, win_recess, [])
        moldy = compose(block, win_recess, [(mc, mhw)])
        coh = compose(block, win_recess + [(rc, rhalf)], [])
        render(fig.add_subplot(len(moldies), 3, 3 * r_ + 1, projection="3d"), base,
               "① base building (window grid)" if r_ == 0 else "")
        render(fig.add_subplot(len(moldies), 3, 3 * r_ + 2, projection="3d"), moldy,
               "② user drops a MOLDY blob" if r_ == 0 else "")
        render(fig.add_subplot(len(moldies), 3, 3 * r_ + 3, projection="3d"), coh,
               "③ integrate_new_part -> coherent window" if r_ == 0 else "")
    fig.suptitle("integrate_new_part() [layout_detail.py, production code path]: moldy blob -> "
                 "integrated window", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "coherent_add_wired_sheet_v2.png", dpi=120); plt.close(fig)
    print(f"[save] {OUT/'coherent_add_wired_sheet_v2.png'}")


if __name__ == "__main__":
    main()
