"""Prototype: a GENERATIVE facade-detail head.

Keeps the building base PROCEDURAL (reliable), but makes the *detail/style* layer
generative: a small conditional diffusion that samples the 12-dim DetailParams vector
(window pattern, cornice, plinth ...) conditioned on style. Because the per-style prior in
scene/sdf_detail.py is one-to-many BY DESIGN, the model learns a genuine distribution ->
different seeds give different, style-appropriate facades on the same base (SDF-sculpt-like
variety). This is where the trained model finally drives the *visible* output.

Trains on samples drawn from the per-style prior (we have no GT detail labels; the prior is
the design input). Reuses the B+.6 diffusion machinery.

  env -u LD_PRELOAD -u LD_LIBRARY_PATH PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
    ./sdfusion/bin/python scripts/train_detail_generator.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from models.networks.recipe_param_diffusion import ConditionalDenoiser, GaussianDiffusion
from models.networks import recipe_param_space as ps
from scene import sdf_detail as det

OUT = REPO / "outputs/detail_generator"
NDIM = len(det.DETAIL_FIELDS)         # 12
STYLES = ps.STYLES                     # 8


def make_data(n_per_style=4000, seed=0):
    rng = np.random.default_rng(seed)
    vecs, sidx = [], []
    for si, s in enumerate(STYLES):
        for _ in range(n_per_style):
            vecs.append(det.sample_detail_vector(s, rng)); sidx.append(si)
    return np.stack(vecs).astype(np.float32), np.array(sidx, np.int64)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)

    vecs, sidx = make_data()
    mean, std = vecs.mean(0), vecs.std(0) + 1e-6
    x0 = torch.tensor((vecs - mean) / std, device=dev)
    cond = torch.zeros(len(sidx), STYLES.__len__(), device=dev)
    cond[torch.arange(len(sidx)), torch.tensor(sidx, device=dev)] = 1.0
    mask = torch.ones(len(sidx), NDIM, device=dev)
    print(f"[data] {len(vecs)} detail vectors | NDIM={NDIM} cond=style one-hot({len(STYLES)})")

    den = ConditionalDenoiser(cond_dim=len(STYLES), n_params=NDIM, hidden=128, depth=3).to(dev)
    diff = GaussianDiffusion(1000, device=dev)
    opt = torch.optim.Adam(den.parameters(), lr=2e-4)
    n = len(x0); bs = 1024
    for ep in range(1, 2501):
        den.train(); order = torch.randperm(n, device=dev); tot = 0.0
        for b in range(0, n, bs):
            bi = order[b:b + bs]
            opt.zero_grad()
            loss = diff.p_losses(den, x0[bi], cond[bi], mask[bi], p_uncond=0.0)
            loss.backward(); opt.step(); tot += loss.item() * len(bi)
        if ep % 250 == 0 or ep == 1:
            print(f"  epoch {ep:5d} | loss {tot/n:.4f}")

    # ---- validate: per-style sampled diversity + that style structure is learned ----
    den.eval()
    print("\n[generative detail] per-style: sampled mean/std of key knobs (K=200):")
    K = 200
    summary = {}
    for si, s in enumerate(STYLES):
        c = torch.zeros(K, len(STYLES), device=dev); c[:, si] = 1.0
        with torch.no_grad():
            g = diff.ddim_sample(den, c, n_params=NDIM, steps=50, eta=1.0).cpu().numpy()
        raw = g * std + mean
        raw = np.clip(raw, det.DETAIL_LO, det.DETAIL_HI)
        wi = det.DETAIL_FIELDS.index("win_w"); sp = det.DETAIL_FIELDS.index("win_spacing")
        co = det.DETAIL_FIELDS.index("cornice_protrude")
        print(f"    {s:14s} win_w={raw[:,wi].mean():.2f}±{raw[:,wi].std():.2f} "
              f"spacing={raw[:,sp].mean():.2f}±{raw[:,sp].std():.2f} "
              f"cornice={raw[:,co].mean():.2f}±{raw[:,co].std():.2f}")
        summary[s] = {"win_w": float(raw[:, wi].mean()), "cornice": float(raw[:, co].mean()),
                      "within_style_std_mean": float(raw.std(0).mean())}
    div = np.mean([v["within_style_std_mean"] for v in summary.values()])
    print(f"\n  mean within-style sampled std (diversity) = {div:.3f}  (recipe-param diversity was ~0)")

    torch.save({"model": den.state_dict(), "mean": mean, "std": std,
                "n_dim": NDIM, "styles": STYLES, "hidden": 128, "depth": 3,
                "timesteps": 1000}, OUT / "detail_gen.pth")
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    print(f"[save] {OUT/'detail_gen.pth'}")


if __name__ == "__main__":
    main()
