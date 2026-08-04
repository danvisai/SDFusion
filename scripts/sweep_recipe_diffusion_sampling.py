"""B+.6 sampling sweep — find the guidance/eta sweet spot for the trained denoiser.

Sampling-only (no retraining). For a grid of (classifier-free guidance, DDIM eta) it
measures, on the same held-out real val split:
  - quality  = mean sampled-param footprint IoU vs GT (higher = more faithful)
  - diversity = mean raw-param std over K samples for one conditioning per style
                (higher = more varied generations)

guidance ↑ pulls samples toward the conditioning (quality ↑, diversity ↓);
eta ↑ injects sampling noise (diversity ↑). The goal is the inference default for the
deployment service: high IoU while keeping useful diversity.

Usage:
  env -u LD_PRELOAD -u LD_LIBRARY_PATH PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
    ./sdfusion/bin/python scripts/sweep_recipe_diffusion_sampling.py --iou_n 48
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from models.networks import recipe_param_space as ps                # noqa: E402
from models.networks.recipe_param_diffusion import (                # noqa: E402
    ConditionalDenoiser, GaussianDiffusion)
import train_recipe_param_head as b5                                # noqa: E402
import train_recipe_param_diffusion as b6                           # noqa: E402


def diversity_at(denoiser, diff, feat, pnorm, pool_cond, pool_sidx, device,
                 k, steps, eta, guidance):
    vals = []
    for s in np.unique(pool_sidx):
        idx = int(np.where(pool_sidx == s)[0][0])
        cond_raw = np.repeat(pool_cond[idx:idx + 1], k, axis=0)
        raw = b6.sample_params(denoiser, diff, feat, cond_raw, np.full(k, s),
                               pnorm, device, steps, eta, guidance)
        n = ps.STYLE_DIMS[ps.IDX_TO_STYLE[s]]
        vals.append(float(raw[:, :n].std(axis=0).mean()))
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", type=Path, default=REPO / "outputs/recipe_param_diffusion_b6")
    ap.add_argument("--fits", type=Path, default=REPO / "outputs/fit_recipes_buildingnet/best_params.npz")
    ap.add_argument("--sdf_dir", type=Path, default=REPO / "data/BuildingNet_dataset_v0_1/resolution_64")
    ap.add_argument("--synthetic", type=Path, default=REPO / "outputs/recipe_param_dataset/synthetic_cond.npz")
    ap.add_argument("--guidance", type=float, nargs="*", default=[1.0, 1.5, 2.0, 3.0])
    ap.add_argument("--eta", type=float, nargs="*", default=[0.0, 1.0])
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--iou_n", type=int, default=48)
    ap.add_argument("--diversity_k", type=int, default=12)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = args.device

    # Reload trained denoiser + scalers.
    ck = torch.load(args.ckpt_dir / "denoiser.pth", map_location=device)
    a = ck["args"]
    den = ConditionalDenoiser(hidden=a["hidden"], depth=a["depth"]).to(device)
    den.load_state_dict(ck["model"]); den.eval()
    diff = GaussianDiffusion(ck["timesteps"], device=device)
    feat, pnorm = ps.load_scalers(args.ckpt_dir / "scalers.npz")

    # Same real val split as training; one conditioning/style pool for diversity.
    data = b5.load_dataset(args.fits)
    N = len(data["meta"])
    perm = np.random.RandomState(args.seed).permutation(N)
    val_idx = perm[:max(1, int(N * args.val_frac))]
    syn = b5.load_synthetic(args.synthetic, 200, args.seed)  # small, just for diversity probes

    print(f"[sweep] guidance={args.guidance} eta={args.eta} | iou_n={args.iou_n} "
          f"k={args.diversity_k} steps={args.steps}")
    rows = []
    for g in args.guidance:
        for e in args.eta:
            iou_rows = b6.diffusion_iou_eval(den, diff, feat, pnorm, data, list(val_idx),
                                             args.sdf_dir, device, args.iou_n, args.steps, e, g)
            iou = float(np.mean([r["iou_pred"] for r in iou_rows])) if iou_rows else 0.0
            fit = float(np.mean([r["iou_fitted"] for r in iou_rows])) if iou_rows else 0.0
            div = diversity_at(den, diff, feat, pnorm, syn["cond"], syn["style_idx"],
                               device, args.diversity_k, args.steps, e, g)
            rows.append({"guidance": g, "eta": e, "iou": iou,
                         "retention": iou / max(fit, 1e-6), "diversity": div})
            print(f"  guidance={g:<4} eta={e:<4} | IoU={iou:.3f} "
                  f"(ret {iou/max(fit,1e-6):.1%}) | diversity={div:.4f}")

    # Save + pick.
    out_csv = args.ckpt_dir / "sampling_sweep.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    best_q = max(rows, key=lambda r: r["iou"])
    # Balanced: best IoU among settings keeping >=60% of the max diversity.
    div_max = max(r["diversity"] for r in rows)
    bal_pool = [r for r in rows if r["diversity"] >= 0.6 * div_max] or rows
    best_bal = max(bal_pool, key=lambda r: r["iou"])
    print(f"\n[best quality ] guidance={best_q['guidance']} eta={best_q['eta']} "
          f"IoU={best_q['iou']:.3f} diversity={best_q['diversity']:.4f}")
    print(f"[best balanced] guidance={best_bal['guidance']} eta={best_bal['eta']} "
          f"IoU={best_bal['iou']:.3f} diversity={best_bal['diversity']:.4f}")
    with open(args.ckpt_dir / "sampling_sweep.json", "w") as f:
        json.dump({"rows": rows, "best_quality": best_q, "best_balanced": best_bal}, f, indent=2)
    print(f"[save] {out_csv}")


if __name__ == "__main__":
    main()
