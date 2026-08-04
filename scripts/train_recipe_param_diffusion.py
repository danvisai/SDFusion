"""B+.6 — train the recipe-parameter diffusion model (the truly generative head).

Replaces the B+.5 deterministic MLP with a conditional diffusion over the recipe-param
space: it learns p(recipe_params | conditioning) so sampling yields *diverse* buildings
consistent with (footprint proportions, class, style). Reuses the B+.5 param-space,
scale-invariant featurizer, normalizer, combined real+synthetic dataset, and footprint-
IoU evaluation harness wholesale.

Jitter: the B+.4 synthetic params are identical across all samples for victorian /
industrial / mediterranean / public_civic (and a few constant dims elsewhere). Those
delta distributions are widened with a controlled jitter (see
recipe_param_space.fit_param_normalizer_with_jitter) so the diffusion can learn to
*generate* variation for them instead of memorising a point.

Usage (XALT stripped, in-repo python):
  env -u LD_PRELOAD -u LD_LIBRARY_PATH PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
    ./sdfusion/bin/python scripts/train_recipe_param_diffusion.py \
      --synthetic outputs/recipe_param_dataset/synthetic_cond.npz \
      --synth_cap_per_style 2000 --real_repeat 8 --epochs 3000 --iou_eval
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from models.networks import recipe_param_space as ps          # noqa: E402
from models.networks.recipe_param_diffusion import (          # noqa: E402
    ConditionalDenoiser, GaussianDiffusion)
import train_recipe_param_head as b5                          # noqa: E402  (data + IoU helpers)

DEFAULT_FITS = REPO / "outputs/fit_recipes_buildingnet/best_params.npz"
DEFAULT_SDF_DIR = REPO / "data/BuildingNet_dataset_v0_1/resolution_64"
DEFAULT_SYNTH = REPO / "outputs/recipe_param_dataset/synthetic_cond.npz"
DEFAULT_OUT = REPO / "outputs/recipe_param_diffusion_b6"


# ---------------------------------------------------------------------------
# Sampling-based evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_params(denoiser, diff, feat, cond_raw, style_idx, pnorm, device,
                  steps, eta, guidance):
    """cond_raw (B,COND_DIM) un-standardised -> sampled RAW padded params (B,MAX_PARAMS)."""
    cond = torch.tensor(feat.transform(cond_raw), device=device)
    x0 = diff.ddim_sample(denoiser, cond, steps=steps, eta=eta, guidance=guidance)
    return pnorm.inverse(x0.cpu().numpy(), style_idx)


def diffusion_iou_eval(denoiser, diff, feat, pnorm, data, idxs, sdf_dir, device,
                       max_n, steps, eta, guidance):
    """Footprint IoU of *sampled* params on held-out real assets vs the fitted ceiling."""
    sub = list(idxs[:max_n])
    sidx = data["style_idx"][sub]
    pred_raw_all = sample_params(denoiser, diff, feat, data["cond"][sub], sidx,
                                 pnorm, device, steps, eta, guidance)
    rows = []
    for j, i in enumerate(sub):
        m = data["meta"][i]
        gt_fp = b5._gt_footprint(sdf_dir, m["id"])
        if gt_fp is None:
            continue
        pred_raw = ps.unpad_params(pred_raw_all[j], m["style"])
        fitted_raw = ps.unpad_params(data["padded"][i], m["style"])
        pred_fp = b5.recipe_footprint(m["style"], pred_raw, m["polygon"], m["bbox"], device)
        fit_fp = b5.recipe_footprint(m["style"], fitted_raw, m["polygon"], m["bbox"], device)
        rows.append({"id": m["id"], "style": m["style"],
                     "iou_pred": b5._iou(pred_fp, gt_fp),
                     "iou_fitted": b5._iou(fit_fp, gt_fp), "ceil_iou": m["ceil_iou"],
                     "_gt": gt_fp, "_pred": pred_fp, "_fit": fit_fp})
    return rows


def param_diversity(denoiser, diff, feat, pnorm, pool_cond, pool_sidx, device,
                    k, steps, eta):
    """For one conditioning per style, sample k times and report raw-param spread.

    This is the direct jitter check: previously zero-variance styles should now show
    nonzero std (the diffusion learned a distribution, not a point).
    """
    out = {}
    for s in np.unique(pool_sidx):
        idx = int(np.where(pool_sidx == s)[0][0])
        cond_raw = np.repeat(pool_cond[idx:idx + 1], k, axis=0)
        raw = sample_params(denoiser, diff, feat, cond_raw, np.full(k, s),
                            pnorm, device, steps, eta, guidance=1.0)
        n = ps.STYLE_DIMS[ps.IDX_TO_STYLE[s]]
        out[ps.IDX_TO_STYLE[s]] = float(raw[:, :n].std(axis=0).mean())
    return out


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", type=Path, default=DEFAULT_FITS)
    ap.add_argument("--sdf_dir", type=Path, default=DEFAULT_SDF_DIR)
    ap.add_argument("--synthetic", type=Path, default=DEFAULT_SYNTH)
    ap.add_argument("--synth_cap_per_style", type=int, default=2000)
    ap.add_argument("--real_repeat", type=int, default=8)
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--p_uncond", type=float, default=0.1, help="cond-dropout prob (CFG)")
    ap.add_argument("--ema_decay", type=float, default=0.0,
                    help="EMA decay on denoiser weights for eval+save (0 = off; try 0.999)")
    # jitter
    ap.add_argument("--jitter_frac", type=float, default=0.1)
    ap.add_argument("--jitter_abs_floor", type=float, default=0.05)
    ap.add_argument("--jitter_strength", type=float, default=1.0,
                    help="normalized-space jitter std on flagged dims (0 disables)")
    # sampling / eval
    ap.add_argument("--sample_steps", type=int, default=50)
    ap.add_argument("--eta", type=float, default=1.0, help=">0 stochastic/diverse sampling")
    ap.add_argument("--guidance", type=float, default=1.0, help=">1 classifier-free guidance")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iou_eval", action="store_true")
    ap.add_argument("--iou_n", type=int, default=64)
    ap.add_argument("--diversity_k", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data: real split + combined pool (mirrors B+.5) -------------------
    data = b5.load_dataset(args.fits)
    N = len(data["meta"])
    perm = np.random.RandomState(args.seed).permutation(N)
    n_val = max(1, int(N * args.val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    pool_cond = np.repeat(data["cond"][train_idx], args.real_repeat, axis=0)
    pool_pad = np.repeat(data["padded"][train_idx], args.real_repeat, axis=0)
    pool_mask = np.repeat(data["mask"][train_idx], args.real_repeat, axis=0)
    pool_sidx = np.repeat(data["style_idx"][train_idx], args.real_repeat, axis=0)
    n_synth = 0
    if args.synthetic:
        syn = b5.load_synthetic(args.synthetic, args.synth_cap_per_style, args.seed)
        pool_cond = np.concatenate([pool_cond, syn["cond"]])
        pool_pad = np.concatenate([pool_pad, syn["padded"]])
        pool_mask = np.concatenate([pool_mask, syn["mask"]])
        pool_sidx = np.concatenate([pool_sidx, syn["style_idx"]])
        n_synth = len(syn["cond"])
    print(f"[data] pool={len(pool_cond)} (real_train {len(train_idx)}x{args.real_repeat} "
          f"+ synth {n_synth}) | real_val={n_val} | cond_dim={ps.COND_DIM}")

    # ---- scalers + jitter --------------------------------------------------
    feat = ps.FeatureScaler.fit(pool_cond)
    pnorm, jitter_mask = ps.fit_param_normalizer_with_jitter(
        pool_pad, pool_sidx, jitter_frac=args.jitter_frac,
        jitter_abs_floor=args.jitter_abs_floor)
    jdims = {ps.IDX_TO_STYLE[s]: int(jitter_mask[s].sum()) for s in np.unique(pool_sidx)}
    print(f"[jitter] frac={args.jitter_frac} strength={args.jitter_strength} | "
          f"flagged dims/style={jdims}")

    x0 = torch.tensor(pnorm.transform(pool_pad, pool_sidx), device=device)
    cond = torch.tensor(feat.transform(pool_cond), device=device)
    mask = torch.tensor(pool_mask, device=device)
    jit = torch.tensor(jitter_mask[pool_sidx] * args.jitter_strength, device=device)
    n_train = len(x0)

    # ---- model -------------------------------------------------------------
    denoiser = ConditionalDenoiser(hidden=args.hidden, depth=args.depth).to(device)
    diff = GaussianDiffusion(args.timesteps, device=device)
    opt = torch.optim.Adam(denoiser.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # EMA shadow weights (used for eval + save when enabled) — standard diffusion practice.
    ema = None
    if args.ema_decay > 0:
        ema = {k: v.detach().clone() for k, v in denoiser.state_dict().items()}

    history = []
    for epoch in range(1, args.epochs + 1):
        denoiser.train()
        order = torch.randperm(n_train, device=device)
        ep_loss = 0.0
        for b in range(0, n_train, args.batch_size):
            bi = order[b:b + args.batch_size]
            x0_j = x0[bi] + jit[bi] * torch.randn_like(x0[bi])  # per-step jitter
            opt.zero_grad()
            loss = diff.p_losses(denoiser, x0_j, cond[bi], mask[bi], args.p_uncond)
            loss.backward()
            opt.step()
            if ema is not None:
                with torch.no_grad():
                    for k, v in denoiser.state_dict().items():
                        if v.dtype.is_floating_point:
                            ema[k].mul_(args.ema_decay).add_(v, alpha=1 - args.ema_decay)
                        else:
                            ema[k].copy_(v)
            ep_loss += loss.item() * len(bi)
        ep_loss /= n_train
        if epoch % max(1, args.epochs // 25) == 0 or epoch == 1:
            history.append({"epoch": epoch, "loss": ep_loss})
            print(f"  epoch {epoch:5d} | denoise_loss {ep_loss:.4f}")

    # Swap in EMA weights for all eval + the saved checkpoint.
    if ema is not None:
        denoiser.load_state_dict(ema)
        print(f"[ema] using EMA weights (decay {args.ema_decay}) for eval + save")

    # ---- param-space diversity (jitter check) ------------------------------
    denoiser.eval()
    div = param_diversity(denoiser, diff, feat, pnorm, pool_cond, pool_sidx, device,
                          args.diversity_k, args.sample_steps, args.eta)
    print(f"[diversity] raw-param std over {args.diversity_k} samples (one cond/style):")
    for s, v in div.items():
        print(f"    {s:14s} std={v:.4f}")

    # ---- save --------------------------------------------------------------
    ckpt = args.out_dir / "denoiser.pth"
    torch.save({"model": denoiser.state_dict(),
                "args": {k: (str(v) if isinstance(v, Path) else v)
                         for k, v in vars(args).items()},
                "cond_dim": ps.COND_DIM, "n_params": ps.MAX_PARAMS,
                "timesteps": args.timesteps}, ckpt)
    ps.save_scalers(args.out_dir / "scalers.npz", feat, pnorm)
    np.savez(args.out_dir / "jitter.npz", jitter_mask=jitter_mask,
             jitter_strength=args.jitter_strength)
    print(f"[save] {ckpt}")

    metrics = {"n_pool": int(n_train), "n_val": int(n_val), "n_synth": int(n_synth),
               "history": history, "diversity": div,
               "jitter": {"frac": args.jitter_frac, "strength": args.jitter_strength}}

    # ---- footprint-IoU of sampled params on real val -----------------------
    if args.iou_eval:
        rows = diffusion_iou_eval(denoiser, diff, feat, pnorm, data, list(val_idx),
                                  args.sdf_dir, device, args.iou_n,
                                  args.sample_steps, args.eta, args.guidance)
        if rows:
            ip = float(np.mean([r["iou_pred"] for r in rows]))
            ifit = float(np.mean([r["iou_fitted"] for r in rows]))
            ceil = float(np.mean([r["ceil_iou"] for r in rows]))
            print(f"[IoU on {len(rows)} val assets] sampled={ip:.3f} | "
                  f"fitted-recipe={ifit:.3f} | B+.7 ceiling={ceil:.3f} | "
                  f"retention={ip/max(ifit,1e-6):.1%}")
            metrics["iou"] = {"n": len(rows), "sampled": ip,
                              "fitted_recipe": ifit, "ceiling": ceil}
            b5.render_sheet(rows, args.out_dir / "diffusion_iou_sheet.png",
                            title="B+.6 diffusion: GT vs fitted-param vs SAMPLED-param footprints")
            print(f"[save] {args.out_dir/'diffusion_iou_sheet.png'}")

    with open(args.out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[save] {args.out_dir/'metrics.json'}")


if __name__ == "__main__":
    main()
