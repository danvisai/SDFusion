"""Render per-label point-cluster montages so unknown BuildingNet part ids can be IDENTIFIED
visually (detail-plan step 1; companion to scripts/identify_buildingnet_labels.py stats).

For each label id present in >=min_presence of buildings: pick the 6 buildings with the highest
share of that label, scatter the building (gray) + the label's points (red), 6 panels per id
-> outputs/part_labels_full/id<k>_montage.png. CPU-only (safe alongside GPU training).
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
BN = REPO / "data/BuildingNet_dataset_v0_1"
PL = BN / "model_data/point_cloud/point_labels"
PC = BN / "POINT_CLOUDS"
OUT = REPO / "outputs/part_labels_full"

KNOWN = {0: "undetermined", 1: "wall", 2: "window", 4: "roof", 7: "tower",
         17: "stairs", 22: "dome"}


def load_ply_xyz(path, stride=4):
    with open(path) as f:
        lines = f.read().splitlines()
    hi = lines.index("end_header") + 1
    a = np.fromstring(" ".join(lines[hi:hi + 100000][::stride]), sep=" ").reshape(-1, 9)
    return a[:, :3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", type=int, default=500, help="buildings to scan for label shares")
    ap.add_argument("--min_presence", type=float, default=0.05)
    ap.add_argument("--per_id", type=int, default=6)
    ap.add_argument("--stride", type=int, default=4)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    files = sorted(os.listdir(PL))
    np.random.RandomState(0).shuffle(files)
    files = [f for f in files if (PC / (f.replace("_label.json", "") + ".ply")).exists()][:args.scan]

    # pass 1: per-label share per building
    share = defaultdict(list)        # id -> [(share, fname)]
    present = defaultdict(int)
    for k, fn in enumerate(files):
        try:
            labs = np.fromiter(json.load(open(PL / fn)).values(), dtype=np.int32)
        except Exception:
            continue
        u, c = np.unique(labs, return_counts=True)
        for i, n in zip(u, c):
            share[int(i)].append((n / len(labs), fn))
            present[int(i)] += 1
        if k % 100 == 0:
            print(f"scan {k}/{len(files)}", flush=True)

    ids = sorted(i for i, p in present.items()
                 if p / len(files) >= args.min_presence and i != 0)
    print("ids >= presence threshold:", [(i, f"{present[i]/len(files):.0%}") for i in ids])

    # pass 2: montage per id
    for i in ids:
        top = sorted(share[i], reverse=True)[:args.per_id]
        fig, axes = plt.subplots(1, len(top), figsize=(3.2 * len(top), 3.4),
                                 subplot_kw={"projection": "3d"})
        if len(top) == 1:
            axes = [axes]
        for ax, (sh, fn) in zip(axes, top):
            aid = fn.replace("_label.json", "")
            try:
                labs = np.fromiter(json.load(open(PL / fn)).values(), dtype=np.int32)[::args.stride]
                xyz = load_ply_xyz(PC / f"{aid}.ply", args.stride)
                if len(labs) != len(xyz):
                    continue
            except Exception:
                continue
            m = labs == i
            bg = xyz[~m][::6]
            ax.scatter(bg[:, 0], bg[:, 2], bg[:, 1], s=0.3, c="#b9b9b9", alpha=0.35, linewidths=0)
            ax.scatter(xyz[m, 0], xyz[m, 2], xyz[m, 1], s=0.8, c="red", linewidths=0)
            ax.set_title(f"{aid[:26]}\nshare={sh:.1%}", fontsize=6)
            ax.set_axis_off()
            ax.view_init(elev=14, azim=-55)
            ax.set_box_aspect((1, 1, 1))
        name = KNOWN.get(i, "?")
        fig.suptitle(f"label id {i}  ({name})  presence={present[i]/len(files):.0%}", fontsize=10)
        p = OUT / f"id{i:02d}_montage.png"
        plt.tight_layout()
        plt.savefig(p, dpi=110)
        plt.close()
        print("wrote", p, flush=True)


if __name__ == "__main__":
    main()
