"""B+.6h experiment — test whether GENERATING height unlocks the missing diversity.

The diversity-ceiling finding (memory/project_b6_diversity_ceiling.md): B+.6 per-building
generation diversity is ~0 because footprint AND height are conditioning INPUTS and the
(cond->params) mapping is ~one-to-one. Height is a ~60x stronger diversity lever than the
recipe params, but the model can't touch it.

This experiment moves `slenderness` (= height/sqrt(area), the scale-free height proxy,
already a column of the conditioning vector at SLENDERNESS_FEAT_IDX) from INPUT to a
GENERATED target dim:
  - conditioning  = footprint shape + class + style  (slenderness column zeroed)
  - generation    = [recipe_params (12), slenderness (1)]  -> 13-dim diffusion
At inference, height = generated_slenderness * sqrt(footprint_area).

The synthetic data has real slenderness variance per class (heights sampled from class
priors), so p(slenderness | shape,class,style) is genuinely one-to-many -> the model has
diversity to learn. We then measure 3D-occupancy diversity with GENERATED heights and
compare to the ~0.008 baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "server"))

from models.networks import recipe_param_space as ps
from models.networks.diff_recipe import build_diff_recipe
from models.networks.recipe_param_diffusion import ConditionalDenoiser, GaussianDiffusion
import train_recipe_param_head as b5

GEN_DIM = ps.MAX_PARAMS + 1          # 12 params + slenderness
SLEN = ps.MAX_PARAMS                  # index of slenderness in the gen vector
SDF_DIR = REPO / "data/BuildingNet_dataset_v0_1/resolution_64"


@torch.no_grad()
def occ_fixed(style, params, poly, bbox_fixed, height, dev):
    mod = build_diff_recipe(style)[0].to(dev)
    p = torch.tensor(np.asarray(params, np.float32), device=dev)
    pt = torch.tensor(np.asarray(poly, np.float32), device=dev)
    h = torch.tensor(float(height), device=dev)
    return (mod(p, pt, h, b5._query_grid(bbox_fixed, dev)).reshape(64, 64, 64) <= 0).cpu().numpy()


def poly_area(poly):
    poly = np.asarray(poly, np.float64)
    if len(poly) > 1 and np.allclose(poly[0], poly[-1]):
        poly = poly[:-1]
    x, z = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.sum(x * np.roll(z, -1) - np.roll(x, -1) * z)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--synth_cap_per_style", type=int, default=2000)
    ap.add_argument("--real_repeat", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--p_uncond", type=float, default=0.1)
    ap.add_argument("--sample_steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n_eval", type=int, default=24)
    ap.add_argument("--out_dir", type=Path, default=REPO / "outputs/recipe_diffusion_genheight")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(0); np.random.seed(0)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = b5.load_dataset(REPO / "outputs/fit_recipes_buildingnet/best_params.npz")
    N = len(data["meta"]); perm = np.random.RandomState(0).permutation(N)
    val_idx, tr_idx = perm[:max(1, int(N * 0.15))], perm[max(1, int(N * 0.15)):]

    cond = np.repeat(data["cond"][tr_idx], args.real_repeat, axis=0)
    pad = np.repeat(data["padded"][tr_idx], args.real_repeat, axis=0)
    msk = np.repeat(data["mask"][tr_idx], args.real_repeat, axis=0)
    sidx = np.repeat(data["style_idx"][tr_idx], args.real_repeat, axis=0)
    syn = b5.load_synthetic(REPO / "outputs/recipe_param_dataset/synthetic_cond.npz",
                            args.synth_cap_per_style, 0)
    cond = np.concatenate([cond, syn["cond"]]); pad = np.concatenate([pad, syn["padded"]])
    msk = np.concatenate([msk, syn["mask"]]); sidx = np.concatenate([sidx, syn["style_idx"]])
    print(f"[data] pool={len(cond)} | GEN_DIM={GEN_DIM} (12 params + slenderness)")

    # slenderness: pull from the cond column, then ZERO it in the conditioning input.
    slender = cond[:, ps.SLENDERNESS_FEAT_IDX].copy()
    cond[:, ps.SLENDERNESS_FEAT_IDX] = 0.0

    feat = ps.FeatureScaler.fit(cond)
    pnorm, _ = ps.fit_param_normalizer_with_jitter(pad, sidx, jitter_frac=0.1)
    s_mean, s_std = float(slender.mean()), float(max(slender.std(), 1e-3))

    # 13-dim normalized target: [normalized params (12), normalized slenderness (1)]
    x_params = pnorm.transform(pad, sidx)
    x_slen = ((slender - s_mean) / s_std)[:, None]
    x0 = torch.tensor(np.concatenate([x_params, x_slen], axis=1), device=dev)
    cond_t = torch.tensor(feat.transform(cond), device=dev)
    mask_t = torch.tensor(np.concatenate([msk, np.ones((len(msk), 1), np.float32)], axis=1), device=dev)

    den = ConditionalDenoiser(cond_dim=ps.COND_DIM, n_params=GEN_DIM,
                              hidden=args.hidden, depth=args.depth).to(dev)
    diff = GaussianDiffusion(args.timesteps, device=dev)
    opt = torch.optim.Adam(den.parameters(), lr=args.lr)
    n = len(x0)
    for ep in range(1, args.epochs + 1):
        den.train(); order = torch.randperm(n, device=dev); tot = 0.0
        for b in range(0, n, args.batch_size):
            bi = order[b:b + args.batch_size]
            opt.zero_grad()
            loss = diff.p_losses(den, x0[bi], cond_t[bi], mask_t[bi], args.p_uncond)
            loss.backward(); opt.step(); tot += loss.item() * len(bi)
        if ep % max(1, args.epochs // 12) == 0 or ep == 1:
            print(f"  epoch {ep:5d} | loss {tot/n:.4f}")

    # ---- eval: footprint IoU (generated height) + 3D-occupancy diversity ----
    den.eval()
    base_iou, gen_iou, gen_div, slen_std = [], [], [], []
    for i in val_idx[:args.n_eval]:
        m = data["meta"][i]; style = m["style"]
        gt = b5._gt_footprint(SDF_DIR, m["id"])
        if gt is None:
            continue
        c = data["cond"][i:i+1].copy(); c[:, ps.SLENDERNESS_FEAT_IDX] = 0.0
        ct = torch.tensor(feat.transform(np.repeat(c, args.k, axis=0)), device=dev)
        with torch.no_grad():
            g = diff.ddim_sample(den, ct, n_params=GEN_DIM, steps=args.sample_steps,
                                 eta=args.eta, guidance=args.guidance).cpu().numpy()
        praw = pnorm.inverse(g[:, :ps.MAX_PARAMS], np.full(args.k, ps.STYLE_TO_IDX[style]))
        slen_g = g[:, SLEN] * s_std + s_mean
        area = poly_area(m["polygon"]); sa = np.sqrt(max(area, 1e-9))
        bbox = np.asarray(m["bbox"], np.float32)
        bbf = bbox.copy(); bbf[1] = 0.0; bbf[4] = max(slen_g.max() * sa, 1e-3) * 1.3
        occs, fps = [], []
        for j in range(args.k):
            h = float(slen_g[j] * sa)
            o = occ_fixed(style, ps.unpad_params(praw[j], style), m["polygon"], bbf, h, dev)
            occs.append(o); fps.append(o.any(axis=1))
        gen_iou.append(np.mean([b5._iou(f, gt) for f in fps]))
        slen_std.append(float(slen_g.std()))
        v3 = [1 - (float((occs[a] & occs[b]).sum()) / max(float((occs[a] | occs[b]).sum()), 1))
              for a, b in combinations(range(args.k), 2)]
        gen_div.append(float(np.mean(v3)))

    res = {"n": len(gen_div), "footprint_iou": float(np.mean(gen_iou)),
           "gen_3d_diversity": float(np.mean(gen_div)),
           "baseline_3d_diversity": 0.008,
           "mean_slenderness_std": float(np.mean(slen_std))}
    print(f"\n[B+.6h height-generation] n={res['n']} k={args.k} guidance={args.guidance}")
    print(f"  footprint IoU (generated height) = {res['footprint_iou']:.3f}")
    print(f"  3D-occupancy diversity           = {res['gen_3d_diversity']:.3f}  "
          f"(baseline height-as-input ~0.008)")
    print(f"  generated slenderness std/sample = {res['mean_slenderness_std']:.3f}")
    torch.save({"model": den.state_dict(), "hidden": args.hidden, "depth": args.depth,
                "gen_dim": GEN_DIM, "s_mean": s_mean, "s_std": s_std,
                "timesteps": args.timesteps}, args.out_dir / "denoiser.pth")
    ps.save_scalers(args.out_dir / "scalers.npz", feat, pnorm)
    json.dump(res, open(args.out_dir / "metrics.json", "w"), indent=2)
    print(f"[save] {args.out_dir}")


if __name__ == "__main__":
    main()
