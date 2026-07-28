"""HOW IT LOOKS ONCE WIRED — mesh render of the coherent-add-primitive flow.

Shows the actual building, not the abstract part-set boxes:
  (a) base building with its window grid (the 'other pieces')
  (b) + a MOLDY primitive the user drops (crude protruding blob)
  (c) the CoherentPartRefiner integrates it -> instantiated procedurally (snap-to-wall + carve)
      = a flush window in the grid. This is what /interpret_mass would return.

  env -u LD_PRELOAD -u LD_LIBRARY_PATH PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
    /tmp/sdfusion_venv/bin/python scripts/foundations/coherent_add_wired_sheet.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts" / "foundations"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

from make_part_edit_pairs import encode, SLOTS
from models.networks.part_set_refiner import CoherentPartRefiner, N_TYPES
from train_coherent_refiner import box_sdf_64

OUT = REPO / "outputs/part_set_refiner"
R = 96
_g = np.linspace(-1, 1, R, dtype=np.float32)
GX, GY, GZ = np.meshgrid(_g, _g, _g, indexing="ij")          # vol indexed [x,y,z]


def box(lo, hi):
    lo, hi = np.asarray(lo), np.asarray(hi)
    c, h = (lo + hi) / 2, (hi - lo) / 2 + 1e-4
    q = np.stack([np.abs(GX - c[0]) - h[0], np.abs(GY - c[1]) - h[1], np.abs(GZ - c[2]) - h[2]], 0)
    return (np.sqrt((np.clip(q, 0, None) ** 2).sum(0)) + np.minimum(q.max(0), 0)).astype(np.float32)


def compose(block, recesses, blobs):
    v = box(*block)
    for c, hw in recesses:                                   # carve window recesses (subtract)
        v = np.maximum(v, -box(np.asarray(c) - hw, np.asarray(c) + hw))
    for c, hw in blobs:                                      # protruding moldy blob (union)
        v = np.minimum(v, box(np.asarray(c) - hw, np.asarray(c) + hw))
    return v


def render(ax, vol, title):
    try:
        v, f, *_ = marching_cubes(vol, level=0.0)
    except Exception:
        ax.set_axis_off(); ax.set_title(title, fontsize=7); return
    tris = v[f][:, :, [0, 2, 1]]                              # (x,y,z)->plot (x,z,y)
    fy = tris[:, :, 2].mean(1)
    col = plt.cm.bone(0.3 + 0.55 * (fy - fy.min()) / (np.ptp(fy) + 1e-9))
    ax.add_collection3d(Poly3DCollection(tris, facecolors=col, edgecolors="k", linewidths=0.03))
    for s in (ax.set_xlim, ax.set_ylim): s(0, R)
    ax.set_zlim(0, R); ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=12, azim=-60)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.set_title(title, fontsize=8)


def snap_to_wall(c, deep=0.16, ww=0.07, wh=0.10):
    """snap a part centre to the nearest of the 4 walls + return a window half-extent oriented to
    that wall (what the wired pipeline's snap-to-surface + instantiation does)."""
    c = np.array(c, np.float32)
    cand = [(0, 0.55), (0, -0.55), (2, 0.35), (2, -0.35)]    # (axis, wall coord)
    ax, val = min(cand, key=lambda av: abs(c[av[0]] - av[1]))
    c[ax] = val
    half = np.array([deep if ax == 0 else ww, wh, deep if ax == 2 else ww], np.float32)
    return c, half


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = CoherentPartRefiner(device=dev)
    model.net.load_state_dict(torch.load(OUT / "coherent_refiner.pth", map_location=dev)["net"])
    model.net.eval()

    block = ([-0.55, -0.95, -0.35], [0.55, 0.7, 0.35])
    DEEP, WW, WH = 0.16, 0.07, 0.10                           # recess depth / half-width / half-height
    rows = [-0.55, -0.12, 0.32]
    # window grid on ALL FOUR vertical walls so windows are visible from any camera angle.
    grid = []   # (center, half-extent) each
    for ry in rows:
        for cx in (-0.3, 0.0, 0.3):
            grid.append((np.array([cx, ry, 0.35], np.float32), np.array([WW, WH, DEEP], np.float32)))
            grid.append((np.array([cx, ry, -0.35], np.float32), np.array([WW, WH, DEEP], np.float32)))
        for cz in (-0.18, 0.18):
            grid.append((np.array([0.55, ry, cz], np.float32), np.array([DEEP, WH, WW], np.float32)))
            grid.append((np.array([-0.55, ry, cz], np.float32), np.array([DEEP, WH, WW], np.float32)))
    parts = [(0, np.concatenate([c, h])) for c, h in grid]
    win_recess = [(c, h) for c, h in grid]

    moldies = [np.array([0.85, 0.45, 0.18], np.float32),     # off the right wall
               np.array([0.3, -0.6, 0.85], np.float32),      # floating off the front
               np.array([-0.85, 0.3, -0.2], np.float32)]     # off the left wall

    fig = plt.figure(figsize=(3.0 * 3, 2.9 * len(moldies)))
    for r_, mc in enumerate(moldies):
        mhw = np.array([0.13, 0.12, 0.15], np.float32)       # a fat crude blob
        # refiner: grid windows + the moldy piece (marked)
        pin = parts + [(0, np.concatenate([mc, mhw]))]
        x = torch.from_numpy(encode(pin)[None]).float().to(dev)
        mk = torch.zeros(1, SLOTS, device=dev); slot = min(len(parts), SLOTS - 1); mk[0, slot] = 1.0
        sdf = torch.from_numpy(box_sdf_64(*block)).view(1, 1, 64, 64, 64).clamp(-0.2, 0.2).to(dev)
        out = model.refine(x, sdf, mk, torch.tensor([3], device=dev),
                           strength=0.4, steps=16, neighbor_k=8)[0, slot].cpu().numpy()
        rc, rhalf = snap_to_wall(out[N_TYPES:N_TYPES + 3], DEEP, WW, WH)   # wired: snap + carve

        base = compose(block, win_recess, [])
        moldy = compose(block, win_recess, [(mc, mhw)])
        coh = compose(block, win_recess + [(rc, rhalf)], [])
        render(fig.add_subplot(len(moldies), 3, 3 * r_ + 1, projection="3d"), base,
               "① base building (window grid)" if r_ == 0 else "")
        render(fig.add_subplot(len(moldies), 3, 3 * r_ + 2, projection="3d"), moldy,
               "② user drops a MOLDY blob" if r_ == 0 else "")
        render(fig.add_subplot(len(moldies), 3, 3 * r_ + 3, projection="3d"), coh,
               "③ refiner → coherent window" if r_ == 0 else "")
    fig.suptitle("Coherent add-primitive, WIRED (mesh): moldy blob → integrated as a flush window "
                 "in the grid", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "coherent_add_wired_sheet.png", dpi=120); plt.close(fig)
    print(f"[save] {OUT/'coherent_add_wired_sheet.png'}")


if __name__ == "__main__":
    main()
