"""Recover conditioning for the B+.4 synthetic recipe-aug samples, so they can be
folded into the B+.5 / B+.6 training set alongside the real B+.7 fits.

The synthetic h5 files store the SDF + footprint + height + seed, but NOT the input
polygon (it was sampled procedurally from the seed). We replay the exact same rng stream
that `generate_recipe_augmentation.py` used:

    rng = np.random.default_rng(seed)
    polygon, kind = sample_polygon(style, rng)        # consumes rng first
    height_m       = sample_height(class_label, rng)  # consumes rng second

This is a *different* rng stream from the recipe-internal one B+.4 replayed for the
params (both seeded by the same `seed`), so params and conditioning stay row-aligned.

Verification: the recovered `height_m` is checked against the value stored in the h5 for
every sample. Because `sample_height` is drawn AFTER `sample_polygon` from the same
stream, an exact height match implies the polygon draw was replayed correctly too. A
handful of samples additionally get a full procedural footprint-IoU check.

Conditioning is built with the SCALE-INVARIANT featurizer in recipe_param_space, so these
world-meter samples share a feature space with the normalized-Frame-N real fits.

Output: outputs/recipe_param_dataset/synthetic_cond.npz
  cond (N,COND_DIM) padded (N,12) mask (N,12) style_idx (N,) class_idx (N,)
  seed (N,) height_m (N,) shape_id (N,)

Usage:
  env -u LD_PRELOAD -u LD_LIBRARY_PATH PYTHONPATH=. CUDA_VISIBLE_DEVICES="" \
    ./sdfusion/bin/python scripts/recover_synthetic_conditioning.py --verify_n 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from models.networks import recipe_param_space as ps  # noqa: E402
import generate_recipe_augmentation as gen            # noqa: E402

EXTRACTED_DIR = REPO / "data/recipe_augmentation_v1/extracted_params"
H5_DIR = REPO / "data/recipe_augmentation_v1"
OUT = REPO / "outputs/recipe_param_dataset/synthetic_cond.npz"


def recover_polygon_height(style: str, seed: int):
    """Replay the worker rng stream: polygon first, then height."""
    rng = np.random.default_rng(int(seed))
    polygon, kind = gen.sample_polygon(style, rng)
    class_label = gen.CLASS_FOR_STYLE[style]
    height_m = gen.sample_height(class_label, rng)
    return polygon, float(height_m), kind, class_label


def verify_footprint_iou(style: str, polygon, height_m, seed) -> float:
    """Reproduce the procedural footprint and IoU it against the stored h5 footprint.

    Mirrors generate_recipe_augmentation._generate_one exactly.
    """
    import torch  # noqa: F401  (build_styled_sdf is torch-native)
    from scene.sdf_primitives import polygon_bbox_with_pad, sample_grid
    from scene.sdf_recipes import build_styled_sdf

    sdf_fn = build_styled_sdf(style, polygon, height_m, seed=int(seed))
    bbox = polygon_bbox_with_pad(polygon, height_m * 2.5, pad=0.10)
    grid = sample_grid(sdf_fn, 64, bbox, device="cpu")
    return (grid.numpy() <= 0.0).any(axis=1).astype(np.uint8)  # (D, W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--styles", nargs="*", default=list(ps.STYLES))
    ap.add_argument("--n_per_style", type=int, default=0,
                    help="cap samples per style (0 = all)")
    ap.add_argument("--verify_n", type=int, default=6,
                    help="per-style samples to footprint-IoU verify against the h5")
    ap.add_argument("--height_tol", type=float, default=1e-4)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    all_cond, all_pad, all_mask = [], [], []
    all_style, all_class, all_seed, all_h, all_shape = [], [], [], [], []

    for style in args.styles:
        ex_path = EXTRACTED_DIR / f"{style}_params.npz"
        if not ex_path.exists():
            print(f"  [SKIP] {style}: no extracted params at {ex_path}")
            continue
        ex = np.load(ex_path)
        params, seeds = ex["params"], ex["seed"]
        n = len(seeds) if args.n_per_style <= 0 else min(args.n_per_style, len(seeds))

        # Stored heights/footprints for verification (cheap: small / lazy datasets).
        h5_path = H5_DIR / f"{style}.h5"
        with h5py.File(h5_path, "r") as f:
            h5_heights = f["height_m"][:n]

        si = ps.STYLE_TO_IDX[style]
        max_h_err = 0.0
        cond_s = np.empty((n, ps.COND_DIM), dtype=np.float32)
        pad_s = np.empty((n, ps.MAX_PARAMS), dtype=np.float32)
        rec_heights = np.empty(n, dtype=np.float32)
        shapes = np.empty(n, dtype=np.int32)
        for i in range(n):
            poly, h_m, kind, class_label = recover_polygon_height(style, seeds[i])
            ci = ps.class_of(class_label)
            cond_s[i] = ps.raw_conditioning(poly, h_m, ci, si)
            pad_s[i] = ps.pad_params(params[i], style)
            rec_heights[i] = h_m
            shapes[i] = gen.SHAPE_IDS[kind]
            max_h_err = max(max_h_err, abs(h_m - float(h5_heights[i])))

        ok_height = max_h_err <= args.height_tol
        flag = "OK " if ok_height else "!! "
        print(f"  {flag}{style:14s} n={n:5d}  height replay max|err|={max_h_err:.2e}  "
              f"class={class_label}")
        if not ok_height:
            raise SystemExit(f"height replay mismatch for {style} "
                             f"(max err {max_h_err:.3e} > tol {args.height_tol}); "
                             f"rng stream drifted — do NOT trust the recovered polygons")

        # Footprint-IoU spot check (a few samples) — proves polygon recovery end-to-end.
        if args.verify_n > 0:
            ious = []
            with h5py.File(h5_path, "r") as f:
                for i in range(min(args.verify_n, n)):
                    poly, h_m, kind, _ = recover_polygon_height(style, seeds[i])
                    rec_fp = verify_footprint_iou(style, poly, h_m, seeds[i])
                    gt_fp = f["footprint"][i].astype(bool)
                    inter = (rec_fp.astype(bool) & gt_fp).sum()
                    union = (rec_fp.astype(bool) | gt_fp).sum()
                    ious.append(inter / union if union else 0.0)
            print(f"       footprint reproduce IoU (n={len(ious)}): "
                  f"mean={np.mean(ious):.4f} min={np.min(ious):.4f}")

        all_cond.append(cond_s); all_pad.append(pad_s)
        all_mask.append(np.broadcast_to(ps.param_mask(style).astype(np.float32),
                                        (n, ps.MAX_PARAMS)).copy())
        all_style.append(np.full(n, si, dtype=np.int64))
        all_class.append(np.array([ps.class_of(gen.CLASS_FOR_STYLE[style])] * n, dtype=np.int64))
        all_seed.append(seeds[:n].astype(np.int64))
        all_h.append(rec_heights)
        all_shape.append(shapes)

    cond = np.concatenate(all_cond); padded = np.concatenate(all_pad)
    mask = np.concatenate(all_mask); style_idx = np.concatenate(all_style)
    class_idx = np.concatenate(all_class); seed = np.concatenate(all_seed)
    height_m = np.concatenate(all_h); shape_id = np.concatenate(all_shape)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, cond=cond, padded=padded, mask=mask,
                        style_idx=style_idx, class_idx=class_idx,
                        seed=seed, height_m=height_m, shape_id=shape_id,
                        cond_dim=ps.COND_DIM)
    print(f"\n[save] {args.out}  N={len(cond)}  cond_dim={ps.COND_DIM}")
    for s in np.unique(style_idx):
        print(f"    {ps.IDX_TO_STYLE[s]:14s} n={int((style_idx==s).sum()):5d}")


if __name__ == "__main__":
    main()
