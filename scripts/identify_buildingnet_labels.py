"""Identify ALL 31 BuildingNet part labels (no name-map shipped) via geometric +
class-distribution signatures, so we know what architectural detail is available
(domes, stairs, towers, columns, balconies, chimneys, parapets ...).

Per label id, over a sample of labeled buildings, computes:
  presence%   - fraction of buildings containing it
  pts%        - mean share of points when present
  height      - mean normalized Y (0 base, 1 top)
  horiz       - mean |ny| (1 = flat/horizontal surface, 0 = vertical)
  vspread     - vertical spread (tall thin element -> large)
  curve       - normal-direction spread (dome/curved -> high; flat/wall -> low)
  top class   - which building class its points concentrate in
Then prints a best-guess name (matched to the paper's 31 classes by signature).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
BN = REPO / "data/BuildingNet_dataset_v0_1"
PL = BN / "model_data/point_cloud/point_labels"
PC = BN / "POINT_CLOUDS"
CLASSES = ["COMMERCIAL", "PUBLIC", "RELIGIOUS", "RESIDENTIAL"]

# Confirmed by geometry earlier; the rest are best-guesses this script tests.
KNOWN = {0: "undetermined", 1: "wall", 2: "window", 4: "roof"}


def top_class(aid):
    m = re.match(r"^([A-Z]+)", aid)
    return m.group(1) if m else "OTHER"


def load_ply(path, stride):
    with open(path) as f:
        lines = f.read().splitlines()
    hi = lines.index("end_header") + 1
    a = np.fromstring(" ".join(lines[hi:hi + 100000][::stride]), sep=" ").reshape(-1, 9)
    return a[:, :3], a[:, 3:6]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--stride", type=int, default=6)
    args = ap.parse_args()

    files = sorted(os.listdir(PL))
    np.random.RandomState(0).shuffle(files)
    S = defaultdict(lambda: {"present": 0, "pts": [], "h": [], "horiz": [],
                             "vspread": [], "curve": [], "cls": defaultdict(int)})
    nb = 0
    for fn in files:
        if nb >= args.n:
            break
        aid = fn.replace("_label.json", ""); pf = PC / f"{aid}.ply"
        if not pf.exists():
            continue
        try:
            labs = np.fromiter(json.load(open(PL / fn)).values(), dtype=np.int32)[::args.stride]
            xyz, nrm = load_ply(pf, args.stride)
            if len(labs) != len(xyz):
                continue
        except Exception:
            continue
        cls = top_class(aid)
        y = xyz[:, 1]; yf = (y - y.min()) / (np.ptp(y) + 1e-9)
        for i in np.unique(labs):
            m = labs == i; s = S[int(i)]
            s["present"] += 1; s["pts"].append(m.mean())
            s["h"].append(yf[m].mean()); s["horiz"].append(np.abs(nrm[m, 1]).mean())
            s["vspread"].append(yf[m].std())
            s["curve"].append(float(nrm[m].std(axis=0).mean()))  # normal-direction spread
            s["cls"][cls] += int(m.sum())
        nb += 1

    def guess(i, h, horiz, vspread, curve, dom):
        if i in KNOWN:
            return KNOWN[i]
        # heuristic match to the 31 named classes by signature
        if h < 0.12 and horiz > 0.8:
            return "floor/ground"
        if h > 0.55 and horiz > 0.7:
            return "roof-ish"
        if h > 0.45 and curve > 0.45 and dom in ("RELIGIOUS", "PUBLIC"):
            return "DOME?"
        if h > 0.5 and horiz < 0.3 and vspread > 0.22:
            return "TOWER?"
        if h < 0.35 and horiz < 0.25 and vspread > 0.18:
            return "STAIRS/door?"
        if 0.3 < h < 0.6 and horiz < 0.3 and vspread > 0.2:
            return "COLUMN?"
        if horiz > 0.6 and 0.2 < h < 0.6:
            return "BALCONY/parapet?"
        if h > 0.6 and float(np.mean(S[i]["pts"])) < 0.03:
            return "chimney/dormer?"
        return "?"

    print(f"[{nb} buildings]  id : present% : pts% : height : horiz : vspread : curve : topclass : guess")
    rows = []
    for i in sorted(S):
        s = S[i]
        pr = 100 * s["present"] / nb
        pts = 100 * np.mean(s["pts"]); h = np.mean(s["h"]); ho = np.mean(s["horiz"])
        vs = np.mean(s["vspread"]); cv = np.mean(s["curve"])
        dom = max(s["cls"], key=s["cls"].get) if s["cls"] else "-"
        g = guess(i, h, ho, vs, cv, dom)
        print(f"  {i:2d} : {pr:5.0f} : {pts:4.1f} : {h:.2f} : {ho:.2f} : {vs:.2f} : {cv:.2f} : {dom:11s} : {g}")
        rows.append({"id": i, "present_pct": pr, "pts_pct": float(pts), "height": float(h),
                     "horiz": float(ho), "vspread": float(vs), "curve": float(cv),
                     "top_class": dom, "guess": g})
    out = REPO / "outputs/buildingnet_detail_stats/label_signatures.json"
    json.dump(rows, open(out, "w"), indent=2)
    print(f"\n[save] {out}")


if __name__ == "__main__":
    main()
