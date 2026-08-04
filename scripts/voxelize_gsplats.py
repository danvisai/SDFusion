"""Voxelize v2 baked 3DGS PLYs into fixed-shape grids for Stage 3b training.

Stage 3b learns an SDF -> Gaussians lifter from paired (SDF, GaussianSet)
training data. The raw Gaussians are variable-N (521k-3.8M per asset), which
is awkward as a regression target. We bin them into a 32^3 Frame-N grid with
K=8 fixed slots per cell, keeping the highest-importance Gaussians per cell.

Importance score per Gaussian:
    importance = sigmoid(raw_opac) * exp(raw_scales).max(-1)

(opacity-weighted maximum activated scale — larger and more opaque Gaussians
contribute more to renders, so they survive cell-level top-K).

Per-cell slot tensor (14 floats):
    [0:3]   raw mean offset (within cell, in Frame-N units)
    [3:6]   raw_scales (pre-exp log-scales)
    [6:10]  raw_quats (un-normalized, w,x,y,z)
    [10]    raw_opac (pre-sigmoid)
    [11:14] sh_dc (degree-0 SH RGB coefficient)

Frame-N bbox per asset: use 0.5%-99.5% percentile bbox (opacity-filtered) so
densification outliers don't blow up the grid. Recenter/rescale means to fit
[-1, 1]^3 before binning. The bbox scale is saved so Stage 3b can reproject.

Output: data/BuildingNet_dataset_v0_1/gsplat_voxelized_32k8/<id>.npz
    slots     (32, 32, 32, 8, 14) float32   pad slots are zeros
    occ_count (32, 32, 32)        uint8     number of slots in [0, 8] used
    bbox      (2, 3)              float32   percentile (lo, hi) used for renorm
    n_total   ()                  int32     pre-binning Gaussian count
    n_kept    ()                  int32     post-binning (sum of occ_count)

Usage:
    env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \\
        scripts/voxelize_gsplats.py \\
        --gsplat_dir data/BuildingNet_dataset_v0_1/gaussian_splats_v2 \\
        --out_dir data/BuildingNet_dataset_v0_1/gsplat_voxelized_32k8 \\
        --workers 12 --grid_res 32 --k_slots 8

Smoke:
    ... --limit 4 --workers 0
"""
from __future__ import annotations
import argparse
import multiprocessing as mp
import os
import sys
import time
from glob import glob
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _voxelize_one(args):
    """Worker: load PLY, voxelize, save .npz. Returns (id, n_kept) on success."""
    ply_path, out_dir, grid_res, k_slots, q_low, q_high, opac_thresh = args
    # Clamp intra-op threading so N spawn workers don't oversubscribe the CPU
    # (see scripts/generate_recipe_augmentation.py:_generate_one).
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    from scene.gsplat_common import load_inria_ply

    out_path = Path(out_dir) / (Path(ply_path).stem + ".npz")
    if out_path.exists():
        return Path(ply_path).stem, -1  # signal "skipped"

    g = load_inria_ply(ply_path, device="cpu")
    means = g.means.numpy().astype(np.float32)
    raw_scales = g.raw_scales.numpy().astype(np.float32)
    raw_quats = g.raw_quats.numpy().astype(np.float32)
    raw_opac = g.raw_opac.numpy().astype(np.float32)
    sh_dc = g.sh_dc.numpy().astype(np.float32)
    n_total = means.shape[0]

    # 1) Opacity filter for bbox computation (outliers from densification often
    #    have tiny activated opacity).
    act_opac = 1.0 / (1.0 + np.exp(-raw_opac))
    if (act_opac > opac_thresh).sum() > max(64, int(0.05 * n_total)):
        bbox_means = means[act_opac > opac_thresh]
    else:
        bbox_means = means
    lo = np.quantile(bbox_means, q_low, axis=0).astype(np.float32)
    hi = np.quantile(bbox_means, q_high, axis=0).astype(np.float32)
    extent = np.maximum(hi - lo, 1e-6)

    # 2) Map means -> [0, 1]^3 grid coords, then to integer cell indices [0, R).
    norm = (means - lo) / extent  # (N, 3)
    cells = np.floor(norm * grid_res).astype(np.int32)
    # Clip out-of-bbox Gaussians (the 0.5%/99.5% tails) to nearest cell.
    cells = np.clip(cells, 0, grid_res - 1)
    cell_idx = (cells[:, 0] * grid_res * grid_res
                + cells[:, 1] * grid_res
                + cells[:, 2])  # (N,) in [0, R^3)

    # 3) Per-Gaussian importance.
    importance = act_opac * np.exp(raw_scales).max(axis=-1)  # (N,)

    # 4) For each non-empty cell, take top-K by importance — vectorized.
    #    Sort all Gaussians by (cell_idx primary, -importance secondary). Then
    #    per-cell rank = position - first_position_in_same_cell. Keep only
    #    rank < K. Pure numpy; ~30x faster than the per-Gaussian Python loop
    #    on 3M+ Gaussian assets.
    combined = np.lexsort(keys=(-importance, cell_idx))  # (N,) sorted indices
    sorted_cells = cell_idx[combined]
    n = combined.shape[0]
    group_change = np.empty(n, dtype=bool)
    group_change[0] = True
    if n > 1:
        group_change[1:] = sorted_cells[1:] != sorted_cells[:-1]
    # First-position-of-current-group for every index (running maximum).
    first_pos = np.where(group_change, np.arange(n, dtype=np.int64), 0)
    np.maximum.accumulate(first_pos, out=first_pos)
    rank_in_group = np.arange(n, dtype=np.int64) - first_pos
    keep_mask = rank_in_group < k_slots
    sel_orig = combined[keep_mask]                # original Gaussian indices
    sel_rank = rank_in_group[keep_mask].astype(np.int64)
    sel_cell = sorted_cells[keep_mask]
    sel_d = (sel_cell // (grid_res * grid_res)).astype(np.int64)
    sel_h = ((sel_cell // grid_res) % grid_res).astype(np.int64)
    sel_w = (sel_cell % grid_res).astype(np.int64)

    slots = np.zeros((grid_res, grid_res, grid_res, k_slots, 14), dtype=np.float32)
    occ_count = np.zeros((grid_res, grid_res, grid_res), dtype=np.uint8)
    # Per-cell center in Frame-N units (for offset computation).
    cell_size = (extent / grid_res).astype(np.float32)
    sel_cell_center = lo + (np.stack([sel_d, sel_h, sel_w], axis=-1).astype(np.float32) + 0.5) * cell_size
    sel_offset = means[sel_orig] - sel_cell_center

    slots[sel_d, sel_h, sel_w, sel_rank, 0:3]   = sel_offset
    slots[sel_d, sel_h, sel_w, sel_rank, 3:6]   = raw_scales[sel_orig]
    slots[sel_d, sel_h, sel_w, sel_rank, 6:10]  = raw_quats[sel_orig]
    slots[sel_d, sel_h, sel_w, sel_rank, 10]    = raw_opac[sel_orig]
    slots[sel_d, sel_h, sel_w, sel_rank, 11:14] = sh_dc[sel_orig]

    # occ_count[d,h,w] = max rank+1 of kept Gaussians in that cell.
    # Use np.maximum.at for unordered accumulation.
    np.maximum.at(occ_count, (sel_d, sel_h, sel_w), (sel_rank + 1).astype(np.uint8))

    n_kept = int(occ_count.sum())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        slots=slots,
        occ_count=occ_count,
        bbox=np.stack([lo, hi], axis=0),
        n_total=np.int32(n_total),
        n_kept=np.int32(n_kept),
    )
    return Path(ply_path).stem, n_kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gsplat_dir", required=True,
                    help="Source dir of *.ply files.")
    ap.add_argument("--out_dir", required=True,
                    help="Destination dir for .npz files.")
    ap.add_argument("--grid_res", type=int, default=32)
    ap.add_argument("--k_slots", type=int, default=8)
    ap.add_argument("--q_low", type=float, default=0.005,
                    help="Lower percentile for bbox computation.")
    ap.add_argument("--q_high", type=float, default=0.995,
                    help="Upper percentile for bbox computation.")
    ap.add_argument("--opac_thresh", type=float, default=0.05,
                    help="Opacity floor for bbox-relevant Gaussians.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12,
                    help="Process pool size. 0 = single process (debug).")
    args = ap.parse_args()

    plys = sorted(glob(os.path.join(args.gsplat_dir, "*.ply")))
    if args.limit > 0:
        plys = plys[:args.limit]
    print(f"[voxelize] plys={len(plys)} grid={args.grid_res}^3 k={args.k_slots} workers={args.workers}")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    tasks = [
        (p, args.out_dir, args.grid_res, args.k_slots,
         args.q_low, args.q_high, args.opac_thresh)
        for p in plys
    ]

    t0 = time.time()
    n_done, n_skipped, n_failed = 0, 0, 0
    kept_totals: list[int] = []

    if args.workers <= 0:
        for arg in tasks:
            try:
                ident, n_kept = _voxelize_one(arg)
                if n_kept < 0:
                    n_skipped += 1
                else:
                    n_done += 1
                    kept_totals.append(n_kept)
            except Exception as exc:
                n_failed += 1
                print(f"  [!] {arg[0]}: {type(exc).__name__}: {exc}")
            if (n_done + n_skipped + n_failed) % 50 == 0:
                elapsed = time.time() - t0
                rate = (n_done + n_failed) / max(elapsed, 1e-6)
                print(f"  done={n_done} skipped={n_skipped} failed={n_failed} "
                      f"rate={rate:.2f}/s")
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers) as pool:
            for k, result in enumerate(pool.imap_unordered(_voxelize_one, tasks,
                                                           chunksize=2)):
                try:
                    ident, n_kept = result
                    if n_kept < 0:
                        n_skipped += 1
                    else:
                        n_done += 1
                        kept_totals.append(n_kept)
                except Exception as exc:
                    n_failed += 1
                    print(f"  [!] {type(exc).__name__}: {exc}")
                if (k + 1) % 100 == 0:
                    elapsed = time.time() - t0
                    rate = (k + 1) / max(elapsed, 1e-6)
                    eta = (len(tasks) - (k + 1)) / max(rate, 1e-6)
                    print(f"  [{k+1}/{len(tasks)}] done={n_done} skipped={n_skipped} "
                          f"failed={n_failed} rate={rate:.2f}/s eta={eta/60:.1f}min")

    elapsed_min = (time.time() - t0) / 60
    print(f"[voxelize] done={n_done} skipped={n_skipped} failed={n_failed} "
          f"in {elapsed_min:.1f} min")
    if kept_totals:
        a = np.array(kept_totals)
        print(f"[voxelize] n_kept per asset: min={a.min()} p10={np.percentile(a, 10):.0f} "
              f"med={np.median(a):.0f} p90={np.percentile(a, 90):.0f} max={a.max()} "
              f"(grid={args.grid_res**3} cells * {args.k_slots} = {args.grid_res**3 * args.k_slots} max)")


if __name__ == "__main__":
    main()
