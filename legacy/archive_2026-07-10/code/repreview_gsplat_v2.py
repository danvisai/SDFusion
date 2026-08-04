"""
Regenerate preview sheets for v2 Gaussian Splats.
Loads each PLY, computes a robust bounding box, auto-fits the camera distance,
renders four viewpoints (azimuths 0, 90, 180, 270), and tiles them 2x2.

Usage:
    env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
      scripts/repreview_gsplat_v2.py \
      --gsplat_dir data/BuildingNet_dataset_v0_1/gaussian_splats_v2
"""
import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from scene.gsplat_common import load_inria_ply
from scene.gsplat_renderer import render_gsplat_view


def compute_robust_bounds(g, opac_threshold=0.05, q_low=0.005, q_high=0.995):
    mns = g.means
    if g.raw_opac is not None and opac_threshold > 0:
        keep = torch.sigmoid(g.raw_opac) > opac_threshold
        if keep.sum() > max(64, int(0.05 * g.means.shape[0])):
            mns = g.means[keep]
    lo = torch.quantile(mns, q_low, dim=0)
    hi = torch.quantile(mns, q_high, dim=0)
    center = (lo + hi) / 2.0
    extent = hi - lo
    radius = float(torch.max(extent).item() / 2.0)
    return center.cpu().tolist(), radius


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gsplat_dir",
        default="data/BuildingNet_dataset_v0_1/gaussian_splats_v2",
        help="Directory containing baked 3DGS PLY files.",
    )
    ap.add_argument(
        "--out_dir",
        help="Output directory for previews. Defaults to same as --gsplat_dir.",
    )
    ap.add_argument("--image_size", type=int, default=384, help="Size of each quadrant render.")
    ap.add_argument("--fov_deg", type=float, default=30.0, help="Camera field of view.")
    ap.add_argument("--elev", type=float, default=20.0, help="Camera elevation.")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of processed assets.")
    ap.add_argument("--overwrite", action="store_true", help="Redo preview even if it exists.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    gsplat_dir = Path(args.gsplat_dir)
    if not gsplat_dir.is_dir():
        raise FileNotFoundError(f"gsplat_dir does not exist: {gsplat_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else gsplat_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ply_files = sorted(list(gsplat_dir.glob("*.ply")))
    print(f"[*] Found {len(ply_files)} PLY files in {gsplat_dir}")
    if args.limit > 0:
        ply_files = ply_files[:args.limit]
        print(f"[*] Limited to first {len(ply_files)} files")

    from pytorch3d.renderer import look_at_view_transform

    n_ok = n_skip = n_fail = 0
    t_start = time.time()

    for idx, ply_path in enumerate(ply_files):
        stem = ply_path.stem
        out_png = out_dir / f"{stem}_preview.png"

        if out_png.exists() and not args.overwrite:
            n_skip += 1
            continue

        print(f"[{idx+1}/{len(ply_files)}] Processing {stem}...", end="", flush=True)
        t0 = time.time()
        try:
            g = load_inria_ply(str(ply_path), device=args.device)
            center, radius = compute_robust_bounds(g)

            # Auto-fit camera distance based on bounding sphere radius
            dist = radius / math.tan(math.radians(args.fov_deg) / 2.0) * 1.25
            dist = max(dist, 1.0) # sanity floor

            # Render 4 angles
            azims = [0.0, 90.0, 180.0, 270.0]
            renders = []
            for az in azims:
                R, T = look_at_view_transform(
                    dist=dist,
                    elev=args.elev,
                    azim=az,
                    at=(center,),
                )
                img = render_gsplat_view(
                    g, R, T, fov_deg=args.fov_deg, image_size=args.image_size
                )
                renders.append(img)

            # Assemble into 2x2 grid
            grid = Image.new("RGB", (args.image_size * 2, args.image_size * 2), "white")
            grid.paste(renders[0], (0, 0))
            grid.paste(renders[1], (args.image_size, 0))
            grid.paste(renders[2], (0, args.image_size))
            grid.paste(renders[3], (args.image_size, args.image_size))

            grid.save(out_png, optimize=True)
            elapsed = time.time() - t0
            print(f" OK ({elapsed:.1f}s, G={g.n})")
            n_ok += 1
        except Exception as e:
            print(f" FAIL: {e}")
            n_fail += 1

    print("\n" + "=" * 50)
    print(f"  Processed : {n_ok}")
    print(f"  Skipped   : {n_skip}")
    print(f"  Failed    : {n_fail}")
    print(f"  Total Time: {time.time() - t_start:.1f}s")
    print("" + "=" * 50)


if __name__ == "__main__":
    main()
