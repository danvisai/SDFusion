"""
Rewrite the 'footprint' field in every BuildingNet SDF h5 with a TRUE
top-down silhouette computed directly from the SDF voxel grid.

Background
----------
The original `bake_footprint()` in preprocess/create_sdf.py projects every
mesh face onto the XZ plane, including roofs, eaves, and ceilings. For most
buildings this produces a near-uniform white image (mean 81% occupied across
30 random samples; 13% of samples are 100% white). It also re-normalizes
each mesh's XZ extent to [0,1]^2, destroying scale information.

This script bypasses the mesh entirely. The SDF already lives in the
normalized (centered, unit-sphere-scaled) coordinate frame, so a silhouette
of `(sdf <= 0).any(axis=Y_AXIS)` is the correct top-down outline.

It:
  - rewrites h5['footprint'] (1, D, D) uint8 in every resolution_<R>/<id>/ori_sample_grid.h5
  - regenerates data/.../footprints_png/<split>/<id>.png from the new footprint
  - prints summary stats (old vs new occupancy)

Run from repo root:
    python scripts/recompute_footprints_from_sdf.py --dry_run    # preview
    python scripts/recompute_footprints_from_sdf.py              # commit
"""
import argparse
import os
import sys

import h5py
import numpy as np
from PIL import Image


def silhouette_from_sdf(sdf_flat, res, axis=1, iso=0.0):
    """
    sdf_flat: 1-D or 2-D array (N,) or (N,1) of length res**3,
              stored in (z, y, x) layout (matches preprocess/create_sdf.py:154).
    axis    : which axis of the (D,D,D) tensor to collapse. Y is up in the
              source meshes -> axis=1 in the (z, y, x) layout.
    iso     : SDF level for inside/outside split. 0.0 is the surface.
    """
    sdf_3d = np.asarray(sdf_flat).reshape(res, res, res)
    return ((sdf_3d <= iso).any(axis=axis)).astype(np.uint8)


def rewrite_footprint(h5p, res, axis, iso):
    """Open h5 in r+ mode, rewrite footprint, return (old_occ%, new_occ%, new_fp)."""
    with h5py.File(h5p, "r+") as f:
        sdf = f["pc_sdf_sample"][:]
        new_fp = silhouette_from_sdf(sdf, res=res, axis=axis, iso=iso)  # (D, D)
        new_fp_3d = new_fp[None, ...]  # (1, D, D)

        old_occ = float("nan")
        if "footprint" in f:
            old_occ = float(f["footprint"][()].mean()) * 100.0
            del f["footprint"]
        f.create_dataset("footprint", data=new_fp_3d,
                         compression="gzip", compression_opts=4)
        new_occ = float(new_fp.mean()) * 100.0
    return old_occ, new_occ, new_fp


def regen_png(model_id, fp, png_root, split):
    out_dir = os.path.join(png_root, split)
    os.makedirs(out_dir, exist_ok=True)
    img = (fp * 255).astype(np.uint8)
    Image.fromarray(img, mode="L").save(
        os.path.join(out_dir, f"{model_id}.png"), optimize=True)


def load_id_to_split(splits_dir):
    id2split = {}
    for split in ("train", "val", "test"):
        p = os.path.join(splits_dir, f"{split}_split.txt")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for line in f:
                mid = line.strip()
                if mid:
                    id2split[mid] = split
    return id2split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/BuildingNet_dataset_v0_1")
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--axis", type=int, default=1,
                    help="vertical axis of the (D,D,D) tensor stored in h5 "
                         "(create_sdf.py uses (z,y,x) layout, Y up = axis 1)")
    ap.add_argument("--iso", type=float, default=0.0,
                    help="SDF level for inside/outside split (0 = surface)")
    ap.add_argument("--dry_run", action="store_true",
                    help="report stats without writing")
    args = ap.parse_args()

    res_dir = os.path.join(args.data_root, f"resolution_{args.res}")
    splits_dir = os.path.join(args.data_root, "splits")
    png_root = os.path.join(args.data_root, "footprints_png")

    id2split = load_id_to_split(splits_dir)
    if not id2split:
        sys.exit(f"No split files found under {splits_dir}")

    all_dirs = sorted(d for d in os.listdir(res_dir)
                      if os.path.isdir(os.path.join(res_dir, d)))
    print(f"[*] {len(all_dirs)} model dirs in {res_dir}")
    print(f"[*] {len(id2split)} ids across train/val/test splits")
    if args.dry_run:
        print("[*] DRY RUN — no files will be written")

    old_occ_list, new_occ_list = [], []
    n_proc = n_no_split = n_skip = 0
    for mid in all_dirs:
        h5p = os.path.join(res_dir, mid, "ori_sample_grid.h5")
        if not os.path.exists(h5p):
            n_skip += 1
            continue

        if args.dry_run:
            with h5py.File(h5p, "r") as f:
                sdf = f["pc_sdf_sample"][:]
                old_fp = f["footprint"][()] if "footprint" in f else None
            new_fp = silhouette_from_sdf(sdf, res=args.res,
                                         axis=args.axis, iso=args.iso)
            old_occ = (float(old_fp.mean()) * 100.0) if old_fp is not None else float("nan")
            new_occ = float(new_fp.mean()) * 100.0
        else:
            old_occ, new_occ, new_fp = rewrite_footprint(
                h5p, res=args.res, axis=args.axis, iso=args.iso)
            split = id2split.get(mid)
            if split is None:
                n_no_split += 1
            else:
                regen_png(mid, new_fp, png_root, split)

        old_occ_list.append(old_occ)
        new_occ_list.append(new_occ)
        n_proc += 1
        if n_proc % 250 == 0:
            print(f"  processed {n_proc:5d}/{len(all_dirs)}")

    old_arr = np.array(old_occ_list)
    new_arr = np.array(new_occ_list)
    print()
    print("=" * 70)
    print(f"  processed                  : {n_proc}")
    print(f"  skipped (no h5)            : {n_skip}")
    print(f"  not in any split (no PNG)  : {n_no_split}")
    print()
    print(f"  OLD footprint occupancy %  : mean={np.nanmean(old_arr):6.2f} "
          f"median={np.nanmedian(old_arr):6.2f}  full-white={(old_arr == 100).sum()}")
    print(f"  NEW footprint occupancy %  : mean={new_arr.mean():6.2f} "
          f"median={np.median(new_arr):6.2f}  full-white={(new_arr == 100).sum()}")
    print(f"  NEW empty (0%) cases       : {(new_arr == 0).sum()}  "
          f"(check these — likely bad SDFs)")
    print("=" * 70)


if __name__ == "__main__":
    main()
