"""Extract per-building PART INSTANCES from BuildingNet labels (detail-plan step 2 fuel).

For every building and every adopted part type, cluster the labeled points into instances
(voxel connected components) and emit (type, centroid, bbox extents, n_points) per instance —
the OmniPart/SPLICE-style variable-length part-bbox sequence, in Frame-N-compatible
normalization (center = bbox center, scale = max-extent/2 -> coords in [-1, 1]).

Out: outputs/part_layouts_full/part_instances.npz
     rows  (N, 9): [building_idx, type_id, cx, cy, cz, ex, ey, ez, n_points]
     names (B,)  : building asset ids       types: see TYPES below.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy import ndimage

REPO = Path(__file__).resolve().parents[2]
BN = REPO / "data/BuildingNet_dataset_v0_1"
PL = BN / "model_data/point_cloud/point_labels"
PC = BN / "POINT_CLOUDS"
OUT = REPO / "outputs/part_layouts_full"

# Adopted vocabulary (outputs/part_labels_full/label_names.json, 2026-06-10).
TYPES = {2: "window", 4: "roof", 6: "door", 7: "tower", 12: "column",
         14: "balcony", 16: "balcony_upper", 17: "stairs", 22: "dome", 15: "chimney_cand"}
GRID = 48           # clustering voxel resolution
MIN_PTS = 25        # per instance, at stride
STRIDE = 2


def load_ply_xyz(path, stride):
    with open(path) as f:
        lines = f.read().splitlines()
    hi = lines.index("end_header") + 1
    a = np.fromstring(" ".join(lines[hi:hi + 100000][::stride]), sep=" ").reshape(-1, 9)
    return a[:, :3]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(os.listdir(PL))
    rows, names = [], []
    for bi, fn in enumerate(files):
        aid = fn.replace("_label.json", "")
        pf = PC / f"{aid}.ply"
        if not pf.exists():
            continue
        try:
            labs = np.fromiter(json.load(open(PL / fn)).values(), dtype=np.int32)[::STRIDE]
            xyz = load_ply_xyz(pf, STRIDE)
            if len(labs) != len(xyz):
                continue
        except Exception:
            continue
        lo, hi = xyz.min(0), xyz.max(0)
        c = (lo + hi) / 2
        s = float((hi - lo).max()) / 2 + 1e-9
        pn = (xyz - c) / s                                   # Frame-N-style [-1,1]
        names.append(aid)
        b = len(names) - 1
        for tid in TYPES:
            m = labs == tid
            if m.sum() < MIN_PTS:
                continue
            p = pn[m]
            vox = np.clip(((p + 1) * 0.5 * (GRID - 1)).astype(int), 0, GRID - 1)
            occ = np.zeros((GRID, GRID, GRID), bool)
            occ[vox[:, 0], vox[:, 1], vox[:, 2]] = True
            comp, nc = ndimage.label(occ, structure=np.ones((3, 3, 3)))
            cid = comp[vox[:, 0], vox[:, 1], vox[:, 2]]
            for k in range(1, nc + 1):
                q = p[cid == k]
                if len(q) < MIN_PTS:
                    continue
                qlo, qhi = q.min(0), q.max(0)
                cc, ee = (qlo + qhi) / 2, np.maximum((qhi - qlo) / 2, 1e-3)
                rows.append([b, tid, *cc.tolist(), *ee.tolist(), len(q)])
        if bi % 200 == 0:
            print(f"{bi}/{len(files)}  rows={len(rows)}", flush=True)

    R = np.asarray(rows, np.float32)
    np.savez_compressed(OUT / "part_instances.npz", rows=R, names=np.asarray(names),
                        types=json.dumps(TYPES))
    print(f"[done] {len(names)} buildings, {len(R)} part instances")
    for tid, nm in TYPES.items():
        n = int((R[:, 1] == tid).sum())
        nb = len(np.unique(R[R[:, 1] == tid, 0]))
        print(f"  {nm:14s} inst={n:6d}  buildings={nb}")


if __name__ == "__main__":
    main()
