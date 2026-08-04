"""Extract per-building PART LAYOUTS from the BuildingNet labels — the training data for
the part-composition model (sculpt -> AI composes a sensible building from labeled parts).

For each labeled building, in NORMALIZED building coords (y in [0,1] by height; xz centered
and scaled by footprint half-extent), compute a structured part descriptor:

  cond (input/massing):  class one-hot(4), footprint aspect, slenderness(h/sqrt area),
                         footprint fill (area / bbox)
  layout (to generate):  glazing, window_y_profile[5] (where windows band vertically),
                         roof_flat, roof_y, has_dome, dome_y, dome_r,
                         n_towers(0..4), tower_h, has_steps

Parts (ids confirmed by geometry/visual): window=2 wall=1 roof=4 dome=22 tower=7 stairs=17.
Output: outputs/part_layouts/layouts.npz (cond, layout, meta).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
BN = REPO / "data/BuildingNet_dataset_v0_1"
PL = BN / "model_data/point_cloud/point_labels"
PC = BN / "POINT_CLOUDS"
OUT = REPO / "outputs/part_layouts"
CLASSES = ["COMMERCIAL", "PUBLIC", "RELIGIOUS", "RESIDENTIAL"]
WINDOW, WALL, ROOF, DOME, TOWER, STAIRS = 2, 1, 4, 22, 7, 17


def load_xyz_n(path, stride):
    L = open(path).read().splitlines(); hi = L.index("end_header") + 1
    a = np.fromstring(" ".join(L[hi:hi + 100000][::stride]), sep=" ").reshape(-1, 9)
    return a[:, :3], a[:, 3:6]


def n_clusters_xz(pts_xz, min_pts=30, cell=0.25):
    """Rough count of distinct vertical elements (towers) by binning xz into cells."""
    if len(pts_xz) < min_pts:
        return 0
    keys = set((round(x / cell), round(z / cell)) for x, z in pts_xz)
    # merge into connected components on the grid
    keys = list(keys); seen = set(); comp = 0
    ks = set(keys)
    for k in keys:
        if k in seen:
            continue
        comp += 1; stack = [k]
        while stack:
            cx, cz = stack.pop()
            if (cx, cz) in seen or (cx, cz) not in ks:
                continue
            seen.add((cx, cz))
            stack += [(cx+1, cz), (cx-1, cz), (cx, cz+1), (cx, cz-1)]
    return min(comp, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1849)
    ap.add_argument("--stride", type=int, default=4)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    conds, layouts, meta = [], [], []
    files = sorted(os.listdir(PL))[:args.n]
    for fi, fn in enumerate(files):
        aid = fn.replace("_label.json", ""); pf = PC / f"{aid}.ply"
        if not pf.exists():
            continue
        try:
            labs = np.fromiter(json.load(open(PL / fn)).values(), dtype=np.int32)[::args.stride]
            xyz, nrm = load_xyz_n(pf, args.stride)
            if len(labs) != len(xyz):
                continue
        except Exception:
            continue
        win, wall, roof, dome, tow, stair = (labs == WINDOW, labs == WALL, labs == ROOF,
                                             labs == DOME, labs == TOWER, labs == STAIRS)
        if win.sum() + wall.sum() < 40:
            continue
        y = xyz[:, 1]; yf = (y - y.min()) / (np.ptp(y) + 1e-9)
        xz = xyz[:, [0, 2]]; cen = xz.mean(0); ext = np.ptp(xz, 0)
        scale = max(ext.max() / 2, 1e-3); xzn = (xz - cen) / scale
        area = ext[0] * ext[1]; aspect = max(ext) / max(min(ext), 1e-3)
        slender = np.ptp(y) / max(np.sqrt(area), 1e-3)
        fill = float(((np.abs(xzn) < 1.0).all(1)).mean())

        cls = re.match(r"^([A-Z]+)", aid).group(1)
        cls_oh = [int(cls == c) for c in CLASSES]
        cond = cls_oh + [aspect, slender, fill]

        glazing = win.sum() / max(win.sum() + wall.sum(), 1)
        wy = np.histogram(yf[win], bins=5, range=(0, 1))[0] if win.sum() > 5 else np.zeros(5)
        wy = (wy / max(wy.sum(), 1)).tolist()
        roof_flat = float(np.abs(nrm[roof, 1]).mean()) if roof.sum() > 10 else 0.85
        roof_y = float(yf[roof].mean()) if roof.sum() > 10 else 0.85
        has_dome = int(dome.sum() > 0.005 * len(labs))
        dome_y = float(yf[dome].mean()) if has_dome else 0.6
        dome_r = float(np.abs(xzn[dome]).mean()) if has_dome else 0.25
        n_tow = n_clusters_xz(xz[tow]) if tow.sum() > 30 else 0
        tow_h = float(yf[tow].max()) if tow.sum() > 5 else 0.0
        has_steps = int(stair.sum() > 0.03 * len(labs))

        layout = [glazing] + wy + [roof_flat, roof_y, has_dome, dome_y, dome_r,
                                   n_tow / 4.0, tow_h, has_steps]
        conds.append(cond); layouts.append(layout); meta.append({"id": aid, "class": cls})
        if (fi + 1) % 400 == 0:
            print(f"  {fi+1}/{len(files)}")

    cond = np.array(conds, np.float32); layout = np.array(layouts, np.float32)
    np.savez(OUT / "layouts.npz", cond=cond, layout=layout,
             cond_names=["COMMERCIAL", "PUBLIC", "RELIGIOUS", "RESIDENTIAL", "aspect", "slender", "fill"],
             layout_names=["glazing", "wy0", "wy1", "wy2", "wy3", "wy4", "roof_flat", "roof_y",
                           "has_dome", "dome_y", "dome_r", "n_towers", "tower_h", "has_steps"],
             meta=json.dumps(meta))
    print(f"[extracted] {len(cond)} buildings | cond_dim={cond.shape[1]} layout_dim={layout.shape[1]}")
    # quick per-class summary of a few part fields
    for c in range(4):
        sel = cond[:, c] == 1
        if sel.sum() == 0:
            continue
        L = layout[sel]
        print(f"  {CLASSES[c]:12s} n={int(sel.sum()):4d} glazing={L[:,0].mean():.2f} "
              f"roof_flat={L[:,6].mean():.2f} dome={L[:,8].mean():.2f} "
              f"towers={4*L[:,11].mean():.2f} steps={L[:,13].mean():.2f}")
    print(f"[save] {OUT/'layouts.npz'}")


if __name__ == "__main__":
    main()
