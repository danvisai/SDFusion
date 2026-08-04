"""Train the PART-COMPOSITION model on real BuildingNet part layouts.

Learns p(part_layout | massing) from ~1800 real labeled buildings: given a building's
massing (class, footprint aspect, slenderness, fill), it generates a sensible part layout
(window bands + glazing, roof type, dome/tower/steps placement) the way real buildings
compose parts. At sculpt time the model reads the sculpt's massing and emits the parts ->
instantiated as clean primitives (scene/sdf_detail) -> a coherent building made of REAL
labeled parts.

This is the label-trained generative core (vs the procedural-target refine UNet). Reuses
the B+.6 conditional diffusion machinery.

  env -u LD_PRELOAD -u LD_LIBRARY_PATH PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
    ./sdfusion/bin/python scripts/train_part_composer.py --epochs 4000
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

from models.networks.recipe_param_diffusion import ConditionalDenoiser, GaussianDiffusion

LAY = REPO / "outputs/part_layouts/layouts.npz"
OUT = REPO / "outputs/part_composer"
CLASSES = ["COMMERCIAL", "PUBLIC", "RELIGIOUS", "RESIDENTIAL"]
CONT_COND = [4, 5, 6]  # aspect, slender, fill (one-hots are 0-3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0); np.random.seed(0)

    d = np.load(LAY, allow_pickle=True)
    cond = d["cond"].astype(np.float32).copy(); layout = d["layout"].astype(np.float32)
    lay_names = list(d["layout_names"])
    print(f"[data] {len(cond)} buildings | cond_dim={cond.shape[1]} layout_dim={layout.shape[1]}")

    # standardize continuous cond dims + all layout dims
    cmean = cond[:, CONT_COND].mean(0); cstd = cond[:, CONT_COND].std(0) + 1e-6
    cond[:, CONT_COND] = (cond[:, CONT_COND] - cmean) / cstd
    lmean = layout.mean(0); lstd = layout.std(0) + 1e-6
    x0 = torch.tensor((layout - lmean) / lstd, device=dev)
    c = torch.tensor(cond, device=dev)
    mask = torch.ones_like(x0)
    cond_dim, n_params = cond.shape[1], layout.shape[1]

    den = ConditionalDenoiser(cond_dim=cond_dim, n_params=n_params, hidden=args.hidden,
                              depth=args.depth).to(dev)
    diff = GaussianDiffusion(1000, device=dev)
    opt = torch.optim.Adam(den.parameters(), lr=args.lr)
    n = len(x0)
    for ep in range(1, args.epochs + 1):
        den.train(); order = torch.randperm(n, device=dev); tot = 0.0
        for b in range(0, n, args.bs):
            bi = order[b:b + args.bs]
            opt.zero_grad()
            loss = diff.p_losses(den, x0[bi], c[bi], mask[bi], p_uncond=0.1)
            loss.backward(); opt.step(); tot += loss.item() * len(bi)
        if ep % max(1, args.epochs // 12) == 0 or ep == 1:
            print(f"  epoch {ep:5d} | loss {tot/n:.4f}")

    # ---- validate: sample per class, check the part composition matches real ----
    den.eval()
    def cond_for(cls_idx, aspect=1.2, slender=1.0, fill=0.85, k=300):
        v = np.zeros((k, cond_dim), np.float32); v[:, cls_idx] = 1.0
        v[:, CONT_COND] = (np.array([aspect, slender, fill]) - cmean) / cstd
        return torch.tensor(v, device=dev)

    def decode(x):
        raw = x.cpu().numpy() * lstd + lmean
        return raw

    print("\n[part composer] sampled vs REAL per-class part composition:")
    print(f"  {'class':12s} {'glazing':>16} {'dome P':>14} {'towers':>14} {'flat-roof':>12}")
    summ = {}
    for ci, cls in enumerate(CLASSES):
        with torch.no_grad():
            s = decode(diff.ddim_sample(den, cond_for(ci), n_params=n_params, steps=50, eta=1.0))
        sel = (d["cond"][:, ci] == 1)
        real = d["layout"][sel]
        gi, di, ni, ri = lay_names.index("glazing"), lay_names.index("has_dome"), lay_names.index("n_towers"), lay_names.index("roof_flat")
        print(f"  {cls:12s} samp {s[:,gi].mean():.2f}/real {real[:,gi].mean():.2f}   "
              f"{np.clip(s[:,di],0,1).mean():.2f}/{real[:,di].mean():.2f}   "
              f"{4*np.clip(s[:,ni],0,1).mean():.2f}/{4*real[:,ni].mean():.2f}   "
              f"{s[:,ri].mean():.2f}/{real[:,ri].mean():.2f}")
        summ[cls] = {"glazing": float(s[:, gi].mean()), "dome_p": float(np.clip(s[:, di], 0, 1).mean()),
                     "towers": float(4 * np.clip(s[:, ni], 0, 1).mean())}

    torch.save({"model": den.state_dict(), "hidden": args.hidden, "depth": args.depth,
                "cond_dim": cond_dim, "n_params": n_params, "timesteps": 1000,
                "cmean": cmean, "cstd": cstd, "lmean": lmean, "lstd": lstd,
                "cont_cond": CONT_COND, "layout_names": lay_names, "classes": CLASSES},
               OUT / "part_composer.pth")
    json.dump(summ, open(OUT / "summary.json", "w"), indent=2)
    print(f"[save] {OUT/'part_composer.pth'}")


if __name__ == "__main__":
    main()
