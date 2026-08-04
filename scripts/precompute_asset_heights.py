"""Precompute Frame-N bounding box dimensions for each BuildingNet asset.

For Stage 3 conditioning we need to know the *normalized* (Frame-N) extents of
each asset — most importantly Y (the building's vertical extent), but also X
and Z for downstream sanity checks. We read the cached 64^3 SDF at
    data/BuildingNet_dataset_v0_1/resolution_64/<id>/ori_sample_grid.h5
and compute the bbox of occupied (sdf <= 0) voxels.

Frame-N convention (see ~/.claude/.../memory/project_sdfusion_axes.md):
    Voxel layout (D=z, H=y, W=x), Y up. Each voxel covers 2/63 Frame-N units
    (linspace(-1, 1, 64) has 63 intervals across 64 samples).

Output: outputs/stage3_metadata/asset_dimensions.csv with columns:
    id, split, n_occupied, x_min_n, x_max_n, y_min_n, y_max_n, z_min_n, z_max_n,
    x_extent_n, y_extent_n, z_extent_n
All ..._n columns are in Frame-N units (range roughly [-1, 1]).

CPU-bound, ~minutes wall-clock for the full 1849-id corpus.

Usage:
    env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \\
        scripts/precompute_asset_heights.py
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


SDF_RES = 64
# linspace(-1, 1, 64) -> step = 2/63
VOXEL_SIZE_N = 2.0 / (SDF_RES - 1)


def load_split(splits_dir: Path, phase: str) -> list[str]:
    p = splits_dir / f"{phase}_split.txt"
    if not p.exists():
        return []
    with open(p) as f:
        return [ln.strip() for ln in f if ln.strip()]


def compute_bbox_n(sdf_h5_path: Path) -> tuple[int, np.ndarray, np.ndarray]:
    """Read SDF, return (n_occupied, lo_n[3], hi_n[3]) in Frame-N axes (x, y, z)."""
    with h5py.File(sdf_h5_path, "r") as f:
        sdf_flat = f["pc_sdf_sample"][:].astype(np.float32).reshape(-1)
    sdf = sdf_flat.reshape(SDF_RES, SDF_RES, SDF_RES)  # (D=z, H=y, W=x)
    occupied = sdf <= 0.0  # iso=0 surface and interior
    n_occ = int(occupied.sum())
    if n_occ == 0:
        return 0, np.full(3, np.nan), np.full(3, np.nan)
    zs, ys, xs = np.nonzero(occupied)
    lo_idx = np.array([xs.min(), ys.min(), zs.min()], dtype=np.float32)
    hi_idx = np.array([xs.max(), ys.max(), zs.max()], dtype=np.float32)
    # Voxel index 0 -> -1, voxel index 63 -> +1 in Frame N.
    lo_n = -1.0 + lo_idx * VOXEL_SIZE_N
    hi_n = -1.0 + hi_idx * VOXEL_SIZE_N
    return n_occ, lo_n, hi_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/BuildingNet_dataset_v0_1")
    ap.add_argument("--out", default="outputs/stage3_metadata/asset_dimensions.csv")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    splits_dir = data_root / "splits"
    res_dir = data_root / "resolution_64"

    rows = []
    by_split: dict[str, str] = {}
    for phase in ("train", "val", "test"):
        for ident in load_split(splits_dir, phase):
            by_split[ident] = phase
    ids = sorted(by_split.keys())
    if args.limit > 0:
        ids = ids[:args.limit]
    print(f"[heights] processing {len(ids)} ids")

    t0 = time.time()
    n_done, n_missing, n_empty = 0, 0, 0
    for i, ident in enumerate(ids):
        h5_path = res_dir / ident / "ori_sample_grid.h5"
        if not h5_path.exists():
            n_missing += 1
            continue
        try:
            n_occ, lo_n, hi_n = compute_bbox_n(h5_path)
        except Exception as exc:
            n_missing += 1
            print(f"  [!] {ident}: {type(exc).__name__}: {exc}")
            continue
        if n_occ == 0:
            n_empty += 1
            continue
        ext = hi_n - lo_n
        rows.append({
            "id": ident,
            "split": by_split[ident],
            "n_occupied": n_occ,
            "x_min_n": float(lo_n[0]), "x_max_n": float(hi_n[0]),
            "y_min_n": float(lo_n[1]), "y_max_n": float(hi_n[1]),
            "z_min_n": float(lo_n[2]), "z_max_n": float(hi_n[2]),
            "x_extent_n": float(ext[0]),
            "y_extent_n": float(ext[1]),
            "z_extent_n": float(ext[2]),
        })
        n_done += 1
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(ids)}] done={n_done} missing={n_missing} empty={n_empty}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    elapsed = time.time() - t0
    print(f"[heights] wrote {len(rows)} rows to {out_path} in {elapsed:.1f}s "
          f"(done={n_done} missing={n_missing} empty={n_empty})")

    if rows:
        y = np.array([r["y_extent_n"] for r in rows])
        x = np.array([r["x_extent_n"] for r in rows])
        z = np.array([r["z_extent_n"] for r in rows])
        print("[heights] Frame-N extent stats:")
        for name, arr in (("y", y), ("x", x), ("z", z)):
            print(f"  {name}: min={arr.min():.3f} p10={np.percentile(arr, 10):.3f} "
                  f"med={np.median(arr):.3f} p90={np.percentile(arr, 90):.3f} max={arr.max():.3f}")


if __name__ == "__main__":
    main()
