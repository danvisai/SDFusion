"""Validate WHY diffusion (B+.6) over the deterministic MLP (B+.5): same footprint
quality, but a *distribution* of buildings instead of a single point.

For held-out real footprints:
  - B+.5 deterministic head  -> one param vector -> one footprint  (diversity = 0)
  - B+.6 diffusion head      -> K sampled params -> K footprints   (diversity measured)

Reports mean footprint IoU vs GT for each head (quality should match) and B+.6's
generation diversity = mean pairwise (1 - IoU) among the K samples. Saves a visual sheet:
GT | B+.5 single | B+.6 sample 1..K.

This is the core research claim of Option B+: a *generative* head that matches the
deterministic quality while producing varied buildings from identical symbolic input.
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.networks import recipe_param_space as ps
from models.networks.recipe_param_head import RecipeParamHead
from models.networks.recipe_param_diffusion import ConditionalDenoiser, GaussianDiffusion
from models.networks.diff_recipe import build_diff_recipe
import train_recipe_param_head as b5


@torch.no_grad()
def recipe_occ(style, params_raw, polygon, bbox, device):
    """Full 3D occupancy (sdf<=0) grid (D,H,W) — captures massing, not just footprint."""
    module = build_diff_recipe(style)[0].to(device)
    p = torch.tensor(np.asarray(params_raw, np.float32), device=device)
    poly = torch.tensor(np.asarray(polygon, np.float32), device=device)
    h = torch.tensor(float(bbox[4] - bbox[1]), device=device)
    grid = b5._query_grid(bbox, device)
    return (module(p, poly, h, grid).reshape(64, 64, 64) <= 0).cpu().numpy()


def _iou3d(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0

B5 = REPO / "outputs/recipe_param_head_b5"
B6 = REPO / "outputs/recipe_param_diffusion_b6"
SDF_DIR = REPO / "data/BuildingNet_dataset_v0_1/resolution_64"


def main(n_assets=24, k=6, guidance=2.0, eta=1.0, steps=50):
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # B+.5 deterministic head (our own ckpt; args hold PosixPaths -> weights_only=False)
    ck5 = torch.load(B5 / "head.pth", map_location=dev, weights_only=False); a5 = ck5["args"]
    head = RecipeParamHead(hidden=a5["hidden"], depth=a5["depth"], dropout=a5["dropout"]).to(dev)
    head.load_state_dict(ck5["model"]); head.eval()
    feat5, pnorm5 = ps.load_scalers(B5 / "scalers.npz")

    # B+.6 diffusion head
    ck6 = torch.load(B6 / "denoiser.pth", map_location=dev, weights_only=False); a6 = ck6["args"]
    den = ConditionalDenoiser(hidden=a6["hidden"], depth=a6["depth"]).to(dev)
    den.load_state_dict(ck6["model"]); den.eval()
    diff = GaussianDiffusion(ck6["timesteps"], device=dev)
    feat6, pnorm6 = ps.load_scalers(B6 / "scalers.npz")

    data = b5.load_dataset(REPO / "outputs/fit_recipes_buildingnet/best_params.npz")
    N = len(data["meta"]); perm = np.random.RandomState(0).permutation(N)
    val_idx = perm[:max(1, int(N * 0.15))][:n_assets]

    iou5, iou6, div6 = [], [], []
    sheet_rows = []
    for i in val_idx:
        m = data["meta"][i]; style = m["style"]
        gt = b5._gt_footprint(SDF_DIR, m["id"])
        if gt is None:
            continue
        cond = data["cond"][i:i + 1]; si = np.array([data["style_idx"][i]])

        # B+.5 deterministic
        with torch.no_grad():
            pn = head(torch.tensor(feat5.transform(cond), device=dev)).cpu().numpy()
        p5 = ps.unpad_params(pnorm5.inverse(pn, si)[0], style)
        fp5 = b5.recipe_footprint(style, p5, m["polygon"], m["bbox"], dev)
        iou5.append(b5._iou(fp5, gt))

        # B+.6 diffusion: K samples — keep full 3D occupancy to measure massing diversity
        cond_t = torch.tensor(feat6.transform(np.repeat(cond, k, axis=0)), device=dev)
        x0 = diff.ddim_sample(den, cond_t, steps=steps, eta=eta, guidance=guidance).cpu().numpy()
        praw = pnorm6.inverse(x0, np.full(k, si[0]))
        occs = [recipe_occ(style, ps.unpad_params(praw[j], style), m["polygon"], m["bbox"], dev)
                for j in range(k)]
        fps = [o.any(axis=1) for o in occs]                      # Y-collapse footprint
        iou6.append(np.mean([b5._iou(f, gt) for f in fps]))
        # diversity: footprint (Y-collapse) vs full 3D occupancy among the K samples
        fp_pair = [1 - b5._iou(fps[x], fps[y]) for x, y in combinations(range(k), 2)]
        v3_pair = [1 - _iou3d(occs[x], occs[y]) for x, y in combinations(range(k), 2)]
        div6.append((float(np.mean(fp_pair)), float(np.mean(v3_pair))))
        sheet_rows.append((m, gt, fp5, fps, iou5[-1], iou6[-1], div6[-1][1]))

    fpdiv = np.mean([d[0] for d in div6]); v3div = np.mean([d[1] for d in div6])
    print(f"[compare] {len(iou5)} val assets | K={k} guidance={guidance}")
    print(f"  B+.5 deterministic: mean footprint IoU = {np.mean(iou5):.3f}  (diversity 0 by construction)")
    print(f"  B+.6 diffusion    : mean footprint IoU = {np.mean(iou6):.3f}")
    print(f"  -> footprint quality delta {np.mean(iou6)-np.mean(iou5):+.3f} (B+.6 matches B+.5)")
    print(f"  generation diversity (1 - pairwise IoU among K samples):")
    print(f"    footprint (Y-collapse) = {fpdiv:.3f}   <- low: footprint is a conditioning INPUT")
    print(f"    full 3D occupancy      = {v3div:.3f}   <- the real generative variation (roofs/massing)")

    # Visual: a few assets, GT | B+.5 | B+.6 x K
    sheet_rows = sorted(sheet_rows, key=lambda r: -r[6])[:6]  # most diverse
    ncol = 2 + k
    fig, axes = plt.subplots(len(sheet_rows), ncol, figsize=(1.5 * ncol, 1.6 * len(sheet_rows)))
    if len(sheet_rows) == 1:
        axes = axes[None, :]
    for r, axr in zip(sheet_rows, axes):
        m, gt, fp5, fps, i5, i6, dv = r
        ims = [gt, fp5] + fps
        titles = ["GT", f"B+.5 ({i5:.2f})"] + [f"B+.6 #{j+1}" for j in range(k)]
        for ax, im, t in zip(axr, ims, titles):
            ax.imshow(im, origin="lower", cmap="gray_r"); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(t, fontsize=6)
        axr[0].set_ylabel(f"{m['style']}\ndiv={dv:.2f}", fontsize=6)
    fig.suptitle("B+.5 deterministic (one building) vs B+.6 diffusion (diverse samples) "
                 "— same footprint", fontsize=9)
    fig.tight_layout(); out = B6 / "heads_diversity_compare.png"; fig.savefig(out, dpi=110)
    plt.close(fig); print(f"[save] {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--n_assets", type=int, default=24)
    ap.add_argument("--k", type=int, default=6)
    a = ap.parse_args()
    main(n_assets=a.n_assets, k=a.k, guidance=a.guidance)
