"""
Drop ids whose SDF has insufficient inside-voxels (too sparse a marching-cubes
mesh -> blank/fragmented training render).

Threshold: inside% >= 0.20 (medium filter from the empirical spectrum check).

Backups originals as <split>_split.txt.prefilter.bak (only on first run), then
overwrites the split files with the filtered list. h5 files on disk are NOT
touched, so this is reversible by restoring the .prefilter.bak files.

Run from repo root:
    python scripts/filter_low_inside_splits.py --dry_run
    python scripts/filter_low_inside_splits.py
"""
import argparse
import os
import shutil
import sys

import h5py
import numpy as np


def inside_pct(h5p):
    with h5py.File(h5p, "r") as f:
        sdf = f["pc_sdf_sample"][:].reshape(64, 64, 64)
    return float((sdf <= 0).mean()) * 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/BuildingNet_dataset_v0_1")
    ap.add_argument("--threshold", type=float, default=0.20,
                    help="minimum inside%% to keep")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    splits_dir = os.path.join(args.data_root, "splits")
    res_dir = os.path.join(args.data_root, "resolution_64")

    totals_kept = totals_dropped = 0
    for split in ("train", "val", "test"):
        sp = os.path.join(splits_dir, f"{split}_split.txt")
        if not os.path.exists(sp):
            continue
        with open(sp) as f:
            ids = [ln.strip() for ln in f if ln.strip()]
        kept, dropped = [], []
        for mid in ids:
            h5p = os.path.join(res_dir, mid, "ori_sample_grid.h5")
            if not os.path.exists(h5p):
                dropped.append((mid, float("nan"))); continue
            try:
                pct = inside_pct(h5p)
            except Exception as e:
                dropped.append((mid, float("nan"))); continue
            if pct >= args.threshold:
                kept.append(mid)
            else:
                dropped.append((mid, pct))
        print(f"[{split:5s}] kept {len(kept):4d} / {len(ids):4d}, dropped {len(dropped)} (threshold {args.threshold}%)")
        if dropped[:5]:
            print(f"    sample dropped: " + ", ".join(f"{m}({p:.2f}%)" for m, p in dropped[:5]))
        totals_kept += len(kept); totals_dropped += len(dropped)

        if not args.dry_run:
            bak = sp + ".prefilter.bak"
            if not os.path.exists(bak):
                shutil.copyfile(sp, bak)
            with open(sp, "w") as f:
                f.write("\n".join(kept) + "\n")

    print()
    print(f"TOTAL kept    : {totals_kept}")
    print(f"TOTAL dropped : {totals_dropped}")
    if args.dry_run:
        print("(DRY RUN — no files written)")


if __name__ == "__main__":
    main()
