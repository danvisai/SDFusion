"""Extract REAL architectural-detail statistics from the local BuildingNet part labels,
to ground scene/sdf_detail.STYLE_DETAIL_PRIORS in data instead of hand guesses.

Uses the per-point labels (id 2=window, 1=wall, 4=roof — identified by geometry, see
memory/project_buildingnet_labels_local.md) aligned to the colored point clouds. Per
top-level class (COMMERCIAL/PUBLIC/RELIGIOUS/RESIDENTIAL) it computes:
  - glazing_ratio    = window_pts / (window+wall) pts        -> window size/density
  - roof_flatness    = mean |ny| of roof pts (1=flat)        -> flat vs pitched
  - window_y_frac    = mean/std vertical position of windows -> floor band placement
  - n_floor_est      = building height / 3.2 m (Frame-N rescaled by typical extent)

Output: outputs/buildingnet_detail_stats/stats.json + printed table.

  env -u LD_PRELOAD -u LD_LIBRARY_PATH PYTHONPATH=. CUDA_VISIBLE_DEVICES="" \
    ./sdfusion/bin/python scripts/extract_buildingnet_detail_stats.py --n 400
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
BN = REPO / "data/BuildingNet_dataset_v0_1"
PL = BN / "model_data/point_cloud/point_labels"
PC = BN / "POINT_CLOUDS"
OUT = REPO / "outputs/buildingnet_detail_stats"

WINDOW, WALL, ROOF = 2, 1, 4


def top_class(aid):
    m = re.match(r"^([A-Z]+)", aid)
    return m.group(1) if m else "OTHER"


def load_ply(path, stride):
    """Return subsampled (xyz, normals) from an ascii BuildingNet point cloud."""
    with open(path) as f:
        lines = f.read().splitlines()
    hi = lines.index("end_header") + 1
    body = lines[hi:hi + 100000][::stride]
    a = np.fromstring(" ".join(body), sep=" ").reshape(-1, 9)
    return a[:, :3], a[:, 3:6]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    files = sorted(os.listdir(PL))
    rng = np.random.RandomState(args.seed); rng.shuffle(files)
    by_cls = defaultdict(lambda: defaultdict(list))
    done = 0
    for fn in files:
        if done >= args.n:
            break
        aid = fn.replace("_label.json", "")
        pf = PC / f"{aid}.ply"
        if not pf.exists():
            continue
        try:
            labs = np.fromiter(json.load(open(PL / fn)).values(), dtype=np.int32)
            xyz, nrm = load_ply(pf, args.stride)
            labs = labs[::args.stride]
            if len(labs) != len(xyz):
                continue
        except Exception:
            continue
        y = xyz[:, 1]; yf = (y - y.min()) / (np.ptp(y) + 1e-9)
        win = labs == WINDOW; wall = labs == WALL; roof = labs == ROOF
        nwin, nwall, nroof = int(win.sum()), int(wall.sum()), int(roof.sum())
        if nwin + nwall < 50:
            continue
        c = by_cls[top_class(aid)]
        c["glazing"].append(nwin / max(nwin + nwall, 1))
        if nroof > 20:
            c["roof_flat"].append(float(np.abs(nrm[roof, 1]).mean()))
        if nwin > 20:
            c["win_yf"].append(float(yf[win].mean()))
            c["win_yspread"].append(float(yf[win].std()))
        # building slenderness proxy: y-extent / mean horizontal extent
        ext = np.ptp(xyz, axis=0)
        c["slender"].append(float(ext[1] / max((ext[0] + ext[2]) / 2, 1e-6)))
        done += 1

    print(f"[extracted from {done} labeled BuildingNet buildings]  (window=2, wall=1, roof=4)\n")
    print(f"{'class':12s} {'n':>4} {'glazing':>16} {'roof_flat(1=flat)':>18} "
          f"{'win_y':>12} {'slender':>12}")
    summary = {}
    for cls in sorted(by_cls):
        c = by_cls[cls]

        def ms(k):
            a = np.array(c[k]) if c[k] else np.array([np.nan])
            return float(np.nanmean(a)), float(np.nanstd(a))
        g, gs = ms("glazing"); rf, rfs = ms("roof_flat"); wy, wys = ms("win_yf"); sl, sls = ms("slender")
        n = len(c["glazing"])
        pitched = 100 * np.mean(np.array(c["roof_flat"]) < 0.6) if c["roof_flat"] else 0
        print(f"{cls:12s} {n:>4} {g:.3f}±{gs:.3f}     {rf:.2f}±{rfs:.2f}  "
              f"(pitched~{pitched:.0f}%) {wy:.2f}±{wys:.2f}  {sl:.2f}±{sls:.2f}")
        summary[cls] = {"n": n, "glazing_ratio": [g, gs], "roof_flatness": [rf, rfs],
                        "pct_pitched": float(pitched), "window_y_frac": [wy, wys],
                        "slenderness": [sl, sls]}
    json.dump(summary, open(OUT / "stats.json", "w"), indent=2)
    print(f"\n[save] {OUT/'stats.json'}")


if __name__ == "__main__":
    main()
