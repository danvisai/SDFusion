"""B+.5 — train the deterministic recipe-parameter prediction head.

Purpose: an *overfit/sanity baseline* before the B+.6 diffusion model. It answers two
questions:
  1. Is the (conditioning -> recipe_params) mapping learnable at all? (Can a small MLP
     overfit the 1556 real B+.7 fits?)
  2. Do the predicted params actually reproduce the building? (footprint IoU of the
     predicted params vs the GT, compared against the fitted-param ceiling.)

Data: outputs/fit_recipes_buildingnet/best_params.npz (1556 grounded real fits, each
carrying style + params + polygon + bbox + the fitted IoU ceiling). Class comes from the
asset-id prefix; height = bbox[4]-bbox[1] (same as the fitter used).

Usage (XALT stripped, in-repo python):
  env -u LD_PRELOAD -u LD_LIBRARY_PATH PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
    ./sdfusion/bin/python scripts/train_recipe_param_head.py --iou_eval

  # pure overfit memorisation check (train on everything, no val split):
  ... scripts/train_recipe_param_head.py --overfit --epochs 4000
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch

from models.networks.diff_recipe import build_diff_recipe
from models.networks.recipe_param_head import RecipeParamHead, masked_param_loss
from models.networks import recipe_param_space as ps

REPO = Path(__file__).resolve().parents[1]
DEFAULT_FITS = REPO / "outputs/fit_recipes_buildingnet/best_params.npz"
DEFAULT_SDF_DIR = REPO / "data/BuildingNet_dataset_v0_1/resolution_64"
DEFAULT_OUT = REPO / "outputs/recipe_param_head_b5"
DEFAULT_SYNTH = REPO / "outputs/recipe_param_dataset/synthetic_cond.npz"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_dataset(fits_path: Path):
    """Build arrays from the B+.7 fits.

    Returns dict of arrays + a parallel list of per-asset meta (id, style, polygon,
    bbox, ceiling IoU) used by the IoU evaluation / sheet.
    """
    d = np.load(fits_path, allow_pickle=True)
    fits = d["fits"].item()

    cond, padded, mask, style_idx, ceil_iou = [], [], [], [], []
    meta = []
    for aid, v in fits.items():
        style = v["style"]
        params = np.asarray(v["params"], dtype=np.float32)
        poly = np.asarray(v["polygon"], dtype=np.float32)
        bbox = np.asarray(v["bbox"], dtype=np.float32)
        height = float(bbox[4] - bbox[1])
        ci = ps.class_of(aid)
        si = ps.STYLE_TO_IDX[style]

        cond.append(ps.raw_conditioning(poly, height, ci, si))
        padded.append(ps.pad_params(params, style))
        mask.append(ps.param_mask(style).astype(np.float32))
        style_idx.append(si)
        ceil_iou.append(float(v["iou"]))
        meta.append({"id": aid, "style": style, "polygon": poly,
                     "bbox": bbox, "height": height, "ceil_iou": float(v["iou"])})

    return {
        "cond": np.stack(cond).astype(np.float32),
        "padded": np.stack(padded).astype(np.float32),
        "mask": np.stack(mask).astype(np.float32),
        "style_idx": np.asarray(style_idx, dtype=np.int64),
        "ceil_iou": np.asarray(ceil_iou, dtype=np.float32),
        "meta": meta,
    }


def load_synthetic(npz_path: Path, cap_per_style: int = 0, seed: int = 0):
    """Load the recovered synthetic conditioning (scripts/recover_synthetic_conditioning).

    Optionally subsample `cap_per_style` rows per style so the 50k synthetic set doesn't
    drown the 1.5k grounded real fits. Returns cond/padded/mask/style_idx arrays.
    """
    d = np.load(npz_path)
    if int(d["cond_dim"]) != ps.COND_DIM:
        raise SystemExit(f"synthetic cond_dim {int(d['cond_dim'])} != current "
                         f"COND_DIM {ps.COND_DIM} — re-run recover_synthetic_conditioning.py")
    cond, padded, mask, sidx = d["cond"], d["padded"], d["mask"], d["style_idx"]
    if cap_per_style and cap_per_style > 0:
        rng = np.random.RandomState(seed)
        keep = []
        for s in np.unique(sidx):
            rows = np.where(sidx == s)[0]
            if len(rows) > cap_per_style:
                rows = rng.choice(rows, cap_per_style, replace=False)
            keep.append(rows)
        keep = np.concatenate(keep)
        cond, padded, mask, sidx = cond[keep], padded[keep], mask[keep], sidx[keep]
    return {"cond": cond.astype(np.float32), "padded": padded.astype(np.float32),
            "mask": mask.astype(np.float32), "style_idx": sidx.astype(np.int64)}


# ---------------------------------------------------------------------------
# IoU evaluation (geometric metric — the one that actually matters)
# ---------------------------------------------------------------------------

def _gt_footprint(sdf_dir: Path, asset_id: str):
    h5_path = sdf_dir / asset_id / "ori_sample_grid.h5"
    if not h5_path.exists():
        return None
    with h5py.File(h5_path, "r") as f:
        return f["footprint"][0].astype(np.uint8)  # (D, W)


def _query_grid(bbox, device):
    x0, y0, z0, x1, y1, z1 = [float(v) for v in bbox]
    xs = torch.linspace(x0, x1, 64, device=device)
    ys = torch.linspace(y0, y1, 64, device=device)
    zs = torch.linspace(z0, z1, 64, device=device)
    Z, Y, X = torch.meshgrid(zs, ys, xs, indexing="ij")
    grid = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)
    grid[:, 1] -= y0  # recipe y=0 maps to GT y_min (matches the fitter)
    return grid


@torch.no_grad()
def recipe_footprint(style, params_raw, polygon, bbox, device):
    """Run a DiffRecipe forward and return its top-down (D,W) silhouette bool mask."""
    module, _, _ = build_diff_recipe(style)
    module = module.to(device)
    p = torch.tensor(params_raw, dtype=torch.float32, device=device)
    poly = torch.tensor(polygon, dtype=torch.float32, device=device)
    height = torch.tensor(float(bbox[4] - bbox[1]), dtype=torch.float32, device=device)
    grid = _query_grid(bbox, device)
    pred = module(p, poly, height, grid).reshape(64, 64, 64)
    return (pred <= 0).any(dim=1).cpu().numpy()  # (D, W)


def _iou(a, b):
    a = a.astype(bool); b = b.astype(bool)
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 0.0


def iou_eval(model, feat, pnorm, data, idxs, sdf_dir, device, max_n):
    """For held-out assets: predicted-param IoU vs GT, and fitted-param ceiling IoU."""
    rows = []
    sub = idxs[:max_n]
    for i in sub:
        m = data["meta"][i]
        gt_fp = _gt_footprint(sdf_dir, m["id"])
        if gt_fp is None:
            continue
        cond = feat.transform(data["cond"][i:i + 1])
        with torch.no_grad():
            pred_norm = model(torch.tensor(cond, device=device)).cpu().numpy()
        si = np.array([data["style_idx"][i]])
        pred_raw = ps.unpad_params(pnorm.inverse(pred_norm, si)[0], m["style"])
        fitted_raw = ps.unpad_params(data["padded"][i], m["style"])

        pred_fp = recipe_footprint(m["style"], pred_raw, m["polygon"], m["bbox"], device)
        fit_fp = recipe_footprint(m["style"], fitted_raw, m["polygon"], m["bbox"], device)
        rows.append({
            "id": m["id"], "style": m["style"],
            "iou_pred": _iou(pred_fp, gt_fp),
            "iou_fitted": _iou(fit_fp, gt_fp),
            "ceil_iou": m["ceil_iou"],
            "_gt": gt_fp, "_pred": pred_fp, "_fit": fit_fp,
        })
    return rows


def render_sheet(rows, out_path, n=12,
                 title="B+.5 head: GT vs fitted-param vs predicted-param footprints"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda r: r["iou_pred"])
    if len(rows) > n:  # span the range: worst, middle, best
        sel = sorted(set(np.linspace(0, len(rows) - 1, n, dtype=int)))
        rows = [rows[i] for i in sel]
    fig, axes = plt.subplots(len(rows), 3, figsize=(6, 2 * len(rows)))
    if len(rows) == 1:
        axes = axes[None, :]
    cols = ["GT footprint", "fitted param", "predicted param"]
    for r, ax_row in zip(rows, axes):
        for ax, key, title in zip(ax_row, ["_gt", "_fit", "_pred"], cols):
            ax.imshow(r[key], origin="lower", cmap="gray_r")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(title, fontsize=7)
        ax_row[0].set_ylabel(f"{r['id'][:18]}\n{r['style']}", fontsize=6)
        ax_row[1].set_title(f"fitted IoU {r['iou_fitted']:.2f}", fontsize=7)
        ax_row[2].set_title(f"pred IoU {r['iou_pred']:.2f}", fontsize=7)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def per_style_raw_mse(model, feat, pnorm, cond, padded, mask, style_idx, device):
    """Raw-space param MSE per style (interpretable, in the recipes' native units)."""
    with torch.no_grad():
        pred_norm = model(torch.tensor(feat.transform(cond), device=device)).cpu().numpy()
    pred_raw = pnorm.inverse(pred_norm, style_idx)
    out = {}
    for s in np.unique(style_idx):
        sel = style_idx == s
        n = ps.STYLE_DIMS[ps.IDX_TO_STYLE[s]]
        diff = (pred_raw[sel][:, :n] - padded[sel][:, :n]) ** 2
        out[ps.IDX_TO_STYLE[s]] = {"n": int(sel.sum()), "raw_mse": float(diff.mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", type=Path, default=DEFAULT_FITS)
    ap.add_argument("--sdf_dir", type=Path, default=DEFAULT_SDF_DIR)
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--overfit", action="store_true",
                    help="train on ALL data, no val split (pure memorisation check)")
    ap.add_argument("--synthetic", type=Path, default=None,
                    help="fold in recovered synthetic conditioning (e.g. %s)" % DEFAULT_SYNTH)
    ap.add_argument("--synth_cap_per_style", type=int, default=0,
                    help="subsample synthetic to N rows/style (0 = all 6250)")
    ap.add_argument("--real_repeat", type=int, default=1,
                    help="replicate real-train rows in the pool to upweight grounded data")
    ap.add_argument("--iou_eval", action="store_true",
                    help="evaluate footprint IoU of predicted params + render sheet")
    ap.add_argument("--iou_n", type=int, default=64)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = load_dataset(args.fits)
    N = len(data["meta"])
    print(f"[data] {N} fits | cond_dim={ps.COND_DIM} | "
          f"styles={ {ps.IDX_TO_STYLE[s]: int((data['style_idx']==s).sum()) for s in np.unique(data['style_idx'])} }")

    # Split REAL into train/val. Validation stays REAL-ONLY so footprint IoU is always
    # measured against grounded BuildingNet GT and the B+.7 ceiling.
    perm = np.random.RandomState(args.seed).permutation(N)
    n_val = 0 if args.overfit else max(1, int(N * args.val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    # Assemble the TRAIN pool: real-train (optionally repeated) + optional synthetic.
    pool_cond = np.repeat(data["cond"][train_idx], args.real_repeat, axis=0)
    pool_pad = np.repeat(data["padded"][train_idx], args.real_repeat, axis=0)
    pool_mask = np.repeat(data["mask"][train_idx], args.real_repeat, axis=0)
    pool_sidx = np.repeat(data["style_idx"][train_idx], args.real_repeat, axis=0)
    n_real_pool = len(pool_cond)
    n_synth = 0
    if args.synthetic:
        syn = load_synthetic(args.synthetic, args.synth_cap_per_style, args.seed)
        pool_cond = np.concatenate([pool_cond, syn["cond"]])
        pool_pad = np.concatenate([pool_pad, syn["padded"]])
        pool_mask = np.concatenate([pool_mask, syn["mask"]])
        pool_sidx = np.concatenate([pool_sidx, syn["style_idx"]])
        n_synth = len(syn["cond"])
    print(f"[split] real_train={len(train_idx)}x{args.real_repeat}={n_real_pool} "
          f"+ synth={n_synth} -> pool={len(pool_cond)} | real_val={len(val_idx)} "
          f"(overfit={args.overfit})")

    # Fit scalers on the TRAIN pool.
    feat = ps.FeatureScaler.fit(pool_cond)
    pnorm = ps.ParamNormalizer.fit(pool_pad, pool_sidx)

    def to_dev(c_arr, p_arr, m_arr, s_arr):
        c = torch.tensor(feat.transform(c_arr), device=device)
        t = torch.tensor(pnorm.transform(p_arr, s_arr), device=device)
        m = torch.tensor(m_arr, device=device)
        return c, t, m

    tr_c, tr_t, tr_m = to_dev(pool_cond, pool_pad, pool_mask, pool_sidx)
    if n_val:
        va_c, va_t, va_m = to_dev(data["cond"][val_idx], data["padded"][val_idx],
                                  data["mask"][val_idx], data["style_idx"][val_idx])

    model = RecipeParamHead(hidden=args.hidden, depth=args.depth, dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_train = len(pool_cond)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(n_train, device=device)
        ep_loss = 0.0
        for b in range(0, n_train, args.batch_size):
            bi = order[b:b + args.batch_size]
            opt.zero_grad()
            loss = masked_param_loss(model(tr_c[bi]), tr_t[bi], tr_m[bi])
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(bi)
        ep_loss /= n_train

        if epoch % max(1, args.epochs // 30) == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                tr_mse = masked_param_loss(model(tr_c), tr_t, tr_m).item()
                va_mse = (masked_param_loss(model(va_c), va_t, va_m).item()
                          if n_val else float("nan"))
            history.append({"epoch": epoch, "train_norm_mse": tr_mse, "val_norm_mse": va_mse})
            print(f"  epoch {epoch:5d} | train_norm_mse {tr_mse:.4f} | val_norm_mse {va_mse:.4f}")

    # Per-style raw-space MSE (interpretable units).
    model.eval()
    raw_train = per_style_raw_mse(model, feat, pnorm, pool_cond, pool_pad, pool_mask,
                                  pool_sidx, device)
    print("[per-style raw MSE | TRAIN pool]")
    for s, r in raw_train.items():
        print(f"    {s:14s} n={r['n']:4d} raw_mse={r['raw_mse']:.4f}")
    raw_val = {}
    if n_val:
        raw_val = per_style_raw_mse(model, feat, pnorm, data["cond"][val_idx],
                                    data["padded"][val_idx], data["mask"][val_idx],
                                    data["style_idx"][val_idx], device)
        print("[per-style raw MSE | VAL]")
        for s, r in raw_val.items():
            print(f"    {s:14s} n={r['n']:4d} raw_mse={r['raw_mse']:.4f}")

    # Save checkpoint + scalers.
    ckpt_path = args.out_dir / "head.pth"
    torch.save({"model": model.state_dict(),
                "args": {k: (str(v) if isinstance(v, Path) else v)
                         for k, v in vars(args).items()},
                "cond_dim": ps.COND_DIM, "n_params": ps.MAX_PARAMS}, ckpt_path)
    ps.save_scalers(args.out_dir / "scalers.npz", feat, pnorm)
    print(f"[save] {ckpt_path}")

    metrics = {"n_real": N, "n_real_train": len(train_idx), "real_repeat": args.real_repeat,
               "n_synth": int(n_synth), "n_pool": int(len(pool_cond)), "n_val": int(n_val),
               "synthetic": str(args.synthetic) if args.synthetic else None,
               "history": history, "raw_mse_train": raw_train, "raw_mse_val": raw_val}

    # IoU evaluation — the geometric success metric.
    if args.iou_eval:
        eval_idx = (val_idx if n_val else train_idx)
        rows = iou_eval(model, feat, pnorm, data, list(eval_idx),
                        args.sdf_dir, device, args.iou_n)
        if rows:
            iou_pred = np.mean([r["iou_pred"] for r in rows])
            iou_fit = np.mean([r["iou_fitted"] for r in rows])
            ceil = np.mean([r["ceil_iou"] for r in rows])
            print(f"[IoU on {len(rows)} {'val' if n_val else 'train'} assets] "
                  f"predicted={iou_pred:.3f} | fitted-recipe={iou_fit:.3f} | "
                  f"B+.7 ceiling={ceil:.3f} | retention={iou_pred/max(iou_fit,1e-6):.2%}")
            metrics["iou"] = {"n": len(rows), "predicted": float(iou_pred),
                              "fitted_recipe": float(iou_fit), "ceiling": float(ceil)}
            sheet = args.out_dir / "head_iou_sheet.png"
            render_sheet(rows, sheet)
            print(f"[save] {sheet}")

    with open(args.out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[save] {args.out_dir/'metrics.json'}")


if __name__ == "__main__":
    main()
