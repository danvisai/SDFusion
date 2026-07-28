"""Learned detailizer v1 — style-conditioned coarse->detailed SDF refiner (DECOR-GAN lineage).

The "stylize a massing into a building" model (research session 2026-06-11): takes the 64^3
massing SDF any path produces (recipe / SDEdit prior / sculpt snap) + style/class ids, and
outputs a 96^3 detailed-building SDF (windows/door/roof/landmarks) — the learned counterpart
of the procedural ② detail layer it is trained on.

v1 = regression UNet (L1, surface-weighted) + FiLM style/class conditioning at every block
(style genuinely determines facade detail here, unlike the massing prior where it was
redundant). --gan 1 adds a 3D patch discriminator (hinge) for crisper multimodal detail.
Coarse inputs are DEGRADED on the fly (gaussian smooth + noise) so the model tolerates the
soft massing the SDEdit prior emits at deploy time.

Run (GPU, ~1h for 8k iters):
  env -u LD_PRELOAD -u LD_LIBRARY_PATH PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
    ./sdfusion/bin/python scripts/foundations/train_detailizer.py \
      --pairs data/detail_pairs_v1/pairs.h5 --iters 8000
Outputs: outputs/detailizer_v1/{detailizer.pth, val_<iter>.png, loss.csv}
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

R_C, R_F = 64, 96
N_STYLE, N_CLASS = 8, 4


# ---------------------------------------------------------------- model
class FiLM(nn.Module):
    def __init__(self, ch, emb):
        super().__init__()
        self.f = nn.Linear(emb, ch * 2)

    def forward(self, x, e):
        g, b = self.f(e)[:, :, None, None, None].chunk(2, dim=1)
        return x * (1 + g) + b


class Block(nn.Module):
    def __init__(self, cin, cout, emb):
        super().__init__()
        self.c1 = nn.Conv3d(cin, cout, 3, padding=1)
        self.c2 = nn.Conv3d(cout, cout, 3, padding=1)
        self.n1 = nn.GroupNorm(8, cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.film = FiLM(cout, emb)

    def forward(self, x, e):
        x = F.silu(self.n1(self.c1(x)))
        x = self.film(x, e)
        return F.silu(self.n2(self.c2(x)))


class DetailizerUNet(nn.Module):
    """coarse (1,96,96,96 upsampled) + style/class (+ layout cond) -> detailed SDF residual.

    cond_dim>0 = v2: condition on the composer's decisions + final facade params, making
    the mapping deterministic so regression can be sharp (composer decides, this renders)."""

    def __init__(self, ch=(32, 64, 128, 256), emb=64, cond_dim=0):
        super().__init__()
        self.cond_dim = cond_dim
        self.emb_s = nn.Embedding(N_STYLE, emb)
        self.emb_c = nn.Embedding(N_CLASS, emb)
        if cond_dim:
            self.cond_mlp = nn.Sequential(nn.Linear(cond_dim, emb), nn.SiLU(),
                                          nn.Linear(emb, emb))
        self.enc = nn.ModuleList()
        cin = 1
        for c in ch:
            self.enc.append(Block(cin, c, emb))
            cin = c
        self.mid = Block(ch[-1], ch[-1], emb)
        self.dec = nn.ModuleList()
        for i, c in enumerate(reversed(ch)):
            skip = c
            cout = ch[len(ch) - 2 - i] if i < len(ch) - 1 else ch[0]
            self.dec.append(Block(c + skip, cout, emb))
        self.out = nn.Conv3d(ch[0], 1, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, style, cls, cond=None):
        e = self.emb_s(style) + self.emb_c(cls)
        if self.cond_dim:
            e = e + self.cond_mlp(cond)
        skips = []
        h = x
        for i, blk in enumerate(self.enc):
            h = blk(h, e)
            skips.append(h)
            if i < len(self.enc) - 1:
                h = F.avg_pool3d(h, 2)
        h = self.mid(h, e)
        for i, blk in enumerate(self.dec):
            h = blk(torch.cat([h, skips[-1 - i]], 1), e)
            if i < len(self.dec) - 1:
                h = F.interpolate(h, size=skips[-2 - i].shape[2:], mode="trilinear",
                                  align_corners=False)
        return x + self.out(h)          # residual on the upsampled coarse


class PatchD(nn.Module):
    """3D patch discriminator on normalized SDF, style-conditioned (projection)."""

    def __init__(self, ch=(32, 64, 128), emb=64):
        super().__init__()
        layers, cin = [], 1
        for c in ch:
            layers += [nn.Conv3d(cin, c, 4, stride=2, padding=1),
                       nn.GroupNorm(8, c), nn.LeakyReLU(0.2)]
            cin = c
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv3d(cin, 1, 3, padding=1)
        self.proj = nn.Embedding(N_STYLE, cin)

    def forward(self, x, style):
        h = self.body(x)
        out = self.head(h)
        e = self.proj(style)[:, :, None, None, None]
        out = out + (h * e).sum(1, keepdim=True) / np.sqrt(h.shape[1])
        return out


# ---------------------------------------------------------------- data
class Pairs:
    def __init__(self, path, val_every=30):
        with h5py.File(path, "r") as h:
            n = int(h.attrs.get("n_valid", h["coarse"].shape[0]))
            self.trunc = float(h.attrs.get("trunc", 2.0))
            print(f"[data] preloading {n} pairs from {path} ...", flush=True)
            self.coarse = h["coarse"][:n]
            self.fine = h["fine"][:n]
            self.style = h["style_id"][:n].astype(np.int64)
            self.cls = h["class_id"][:n].astype(np.int64)
            self.cond = h["cond"][:n].astype(np.float32) if "cond" in h else None
            self.cond_dim = 0 if self.cond is None else self.cond.shape[1]
        idx = np.arange(n)
        self.val_idx = idx[::val_every][:8]
        self.train_idx = np.setdiff1d(idx, self.val_idx)
        print(f"[data] {len(self.train_idx)} train / {len(self.val_idx)} val")

    def batch(self, rng, bs, device, degrade=True):
        ids = rng.choice(self.train_idx, bs)
        c = torch.from_numpy(self.coarse[np.sort(ids)].astype(np.float32)) / self.trunc
        f_ = torch.from_numpy(self.fine[np.sort(ids)].astype(np.float32)) / self.trunc
        s = torch.from_numpy(self.style[np.sort(ids)])
        k = torch.from_numpy(self.cls[np.sort(ids)])
        cond = (torch.from_numpy(self.cond[np.sort(ids)]).to(device)
                if self.cond is not None else None)
        c, f_ = c[:, None].to(device), f_[:, None].to(device)
        if degrade:   # mimic the soft SDEdit-prior massing this model sees at deploy
            # (v2: sigma capped at 1.1 — 1.5 erased roof identity, both v1 models
            # learned to flatten pyramids)
            if rng.random() < 0.6:
                sig = rng.uniform(0.4, 1.1)
                kk = int(2 * round(2 * sig) + 1)
                g = torch.arange(kk, device=device, dtype=torch.float32) - kk // 2
                g = torch.exp(-g ** 2 / (2 * sig ** 2)); g /= g.sum()
                for dim in range(3):
                    shape = [1, 1, 1, 1, 1]; shape[2 + dim] = kk
                    c = F.conv3d(c, g.view(shape), padding=[kk // 2 if d == dim else 0
                                                            for d in range(3)])
            c = c + torch.randn_like(c) * rng.uniform(0.0, 0.015)
        return c, f_, s.to(device), k.to(device), cond


def up(c):
    return F.interpolate(c, size=(R_F,) * 3, mode="trilinear", align_corners=False)


# ---------------------------------------------------------------- viz
def montage(model, data, device, path, n=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage import measure
    sty_names = ["modern", "colonial", "victorian", "industrial", "craftsman",
                 "mediterranean", "contemporary", "public_civic"]

    def draw(ax, g, title):
        ax.set_title(title, fontsize=7)
        if (g <= 0).sum() > 8:
            try:
                v, fc, _, _ = measure.marching_cubes(g.astype(np.float32), level=0.0)
                v = v[:, [2, 1, 0]]
                ax.plot_trisurf(v[:, 0], v[:, 2], fc, v[:, 1], color="#cdb892",
                                edgecolor="none", antialiased=True, shade=True)
                lo, hi = v.min(), v.max()
                ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_zlim(lo, hi)
            except Exception:
                pass
        ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(elev=20, azim=-60)

    model.eval()
    ids = data.val_idx[:n]
    fig, axes = plt.subplots(3, n, figsize=(2.3 * n, 7.2), subplot_kw={"projection": "3d"})
    l1s = []
    with torch.no_grad():
        for j, k in enumerate(ids):
            c = torch.from_numpy(data.coarse[k].astype(np.float32))[None, None].to(device) / data.trunc
            gt = data.fine[k].astype(np.float32) / data.trunc
            s = torch.tensor([data.style[k]], device=device)
            cl = torch.tensor([data.cls[k]], device=device)
            cd = (torch.from_numpy(data.cond[k:k + 1]).to(device)
                  if data.cond is not None and model.cond_dim else None)
            pred = model(up(c), s, cl, cd)[0, 0].cpu().numpy()
            l1s.append(float(np.abs(pred - gt).mean()))
            draw(axes[0, j], c[0, 0].cpu().numpy(), f"coarse · {sty_names[data.style[k]]}")
            draw(axes[1, j], pred, f"PRED l1={l1s[-1]:.4f}")
            draw(axes[2, j], gt, "GT detailed")
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)
    model.train()
    return float(np.mean(l1s))


# ---------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=os.path.join(REPO, "data/detail_pairs_v1/pairs.h5"))
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--gan", type=int, default=0)
    ap.add_argument("--gan_w", type=float, default=0.05)
    ap.add_argument("--out", default=os.path.join(REPO, "outputs/detailizer_v1"))
    ap.add_argument("--val_every", type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)

    data = Pairs(args.pairs)
    G = DetailizerUNet(cond_dim=data.cond_dim).to(device)
    print(f"[model] cond_dim={data.cond_dim}" + (" (v2 layout-conditioned)" if data.cond_dim else ""))
    optG = torch.optim.AdamW(G.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optG, args.iters)
    if args.gan:
        D = PatchD().to(device)
        optD = torch.optim.AdamW(D.parameters(), lr=args.lr, weight_decay=1e-4)

    logp = os.path.join(args.out, "loss.csv")
    logf = open(logp, "a", newline="")
    logw = csv.writer(logf)
    if os.path.getsize(logp) == 0:
        logw.writerow(["iter", "l1", "gan_g", "gan_d", "val_l1", "secs"])

    t0 = time.time()
    for it in range(1, args.iters + 1):
        c, f_, s, k, cond = data.batch(rng, args.bs, device)
        pred = G(up(c), s, k, cond)
        band = (f_.abs() < 0.25).float()
        l1 = ((pred - f_).abs() * (1 + 4 * band)).mean()
        gl = torch.tensor(0.0, device=device)
        if args.gan:
            gl = -D(pred, s).mean()
        loss = l1 + args.gan_w * gl
        optG.zero_grad(set_to_none=True)
        loss.backward()
        optG.step(); sched.step()

        dl = torch.tensor(0.0)
        if args.gan:
            d_real = D(f_, s); d_fake = D(pred.detach(), s)
            dl = F.relu(1 - d_real).mean() + F.relu(1 + d_fake).mean()
            optD.zero_grad(set_to_none=True)
            dl.backward()
            optD.step()

        if it % 100 == 0:
            print(f"  it {it:5d}  l1={l1.item():.4f} gan_g={gl.item():.3f} "
                  f"gan_d={dl.item():.3f}  {(time.time()-t0)/it:.2f}s/it", flush=True)
        if it % args.val_every == 0 or it == args.iters:
            png = os.path.join(args.out, f"val_{it:06d}.png")
            vl1 = montage(G, data, device, png)
            logw.writerow([it, round(l1.item(), 5), round(gl.item(), 4),
                           round(float(dl), 4), round(vl1, 5),
                           round(time.time() - t0, 1)])
            logf.flush()
            torch.save({"G": G.state_dict(), "iter": it, "args": vars(args),
                        "cond_dim": data.cond_dim},
                       os.path.join(args.out, "detailizer.pth"))
            print(f"  [val] it {it}  val_l1={vl1:.4f}  -> {png}", flush=True)

    logf.close()
    print(f"[detailizer] done in {(time.time()-t0)/60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()
