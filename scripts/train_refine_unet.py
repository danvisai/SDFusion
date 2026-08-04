"""Train the amortized learned refine (models/networks/refine_unet.RefineUNet3D).

Synthesizes (crude -> good) SDF pairs and trains the UNet to map a blocky/perturbed sculpt
into a coherent detailed building. "Good" = recipe base + data-grounded facade detail +
class-occurrence landmarks (our notion of good architecture, since BuildingNet GT SDFs are
too broken to supervise on). "Crude" = the good building coarsened (blocky) + a few random
box add/subtract edits (user-sculpt-like).

After training the model has LEARNED rough->good and applies it in ONE forward pass.

  env -u LD_PRELOAD -u LD_LIBRARY_PATH PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
    ./sdfusion/bin/python scripts/train_refine_unet.py --n 1200 --epochs 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from models.networks.diff_recipe import build_diff_recipe
from models.networks.refine_unet import RefineUNet3D, surface_weighted_l1
from scene.sdf_edit import recipe_base_sdf
from scene import sdf_detail as det
from scene.sdf_primitives import sample_grid, grid_to_mesh

OUT = REPO / "outputs/refine_unet"
STYLES = ["modern", "victorian", "industrial", "public_civic", "colonial", "contemporary"]
CLASSES = ["RESIDENTIAL", "COMMERCIAL", "PUBLIC", "RELIGIOUS"]
CLIP = 0.25


def rand_footprint(rng):
    w = rng.uniform(8, 20); d = rng.uniform(8, 20)
    if rng.random() < 0.25:  # L-shape
        a, b = w, d; t1 = rng.uniform(0.4, 0.6) * b; t2 = rng.uniform(0.4, 0.6) * a
        p = np.array([[0, 0], [a, 0], [a, t1], [t2, t1], [t2, b], [0, b]], np.float32)
        return p - p.mean(0)
    return np.array([[-w/2, -d/2], [w/2, -d/2], [w/2, d/2], [-w/2, d/2]], np.float32)


def good_sdf(style, cls, rng, R, dev):
    fp = rand_footprint(rng)
    H = float(np.clip(max(np.ptp(fp[:, 0]), np.ptp(fp[:, 1])) * rng.uniform(0.5, 1.6), 5, 40))
    _, default_fn, _ = build_diff_recipe(style)
    params = default_fn(dev).cpu().numpy()
    sdf = recipe_base_sdf(style, params, fp, H, device=dev)
    sdf = det.add_facade_detail(sdf, fp, H, det.ground_glazing(
        det.vector_to_params(det.sample_detail_vector(style, rng)), cls))
    lm = det.sample_landmarks(cls, rng)
    head = H * (1.9 if lm["n_towers"] else 1.4)
    if lm["dome"] or lm["n_towers"] or lm["steps"]:
        sdf = det.add_landmarks(sdf, fp, H, dome=lm["dome"], n_towers=lm["n_towers"], steps=lm["steps"])
    x0, z0 = fp[:, 0].min(), fp[:, 1].min(); x1, z1 = fp[:, 0].max(), fp[:, 1].max()
    px, pz = (x1 - x0) * 0.12 + 1, (z1 - z0) * 0.12 + 1
    bbox = (x0 - px, 0.0, z0 - pz, x1 + px, head, z1 + pz)
    g = sample_grid(sdf, R, bbox, device=dev)
    return g.clamp(-CLIP, CLIP)


def box_sdf_grid(R, rng, dev):
    """Analytic box SDF on the [-1,1]^3 normalized grid (for crude perturbations)."""
    lin = torch.linspace(-1, 1, R, device=dev)
    Z, Y, X = torch.meshgrid(lin, lin, lin, indexing="ij")
    c = torch.tensor(rng.uniform(-0.5, 0.5, 3), device=dev, dtype=torch.float32)
    he = torch.tensor(rng.uniform(0.12, 0.32, 3), device=dev, dtype=torch.float32)
    q = torch.stack([X - c[0], Y - c[1], Z - c[2]], -1).abs() - he
    return (torch.clamp(q, min=0).norm(dim=-1) + torch.clamp(q.amax(-1), max=0)) * CLIP * 3


def make_crude(good, rng, dev):
    R = good.shape[-1]
    c = int(rng.choice([2, 3, 4]))
    g = good[None, None]
    down = F.avg_pool3d(g, c, ceil_mode=True)
    crude = F.interpolate(down, size=(R, R, R), mode="nearest")[0, 0]
    for _ in range(int(rng.integers(0, 3))):
        b = box_sdf_grid(R, rng, dev) * (CLIP / (CLIP * 3))
        if rng.random() < 0.6:
            crude = torch.minimum(crude, b)
        else:
            crude = torch.maximum(crude, -b)
    return crude.clamp(-CLIP, CLIP)


def synthesize(n, R, dev, seed=0):
    cache = OUT / f"pairs_n{n}_R{R}.pt"
    if cache.exists():
        d = torch.load(cache); return d["crude"], d["good"]
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    crude, good = [], []
    for i in range(n):
        style = STYLES[rng.integers(len(STYLES))]; cls = CLASSES[rng.integers(len(CLASSES))]
        try:
            g = good_sdf(style, cls, rng, R, dev)
        except Exception:
            continue
        crude.append(make_crude(g, rng, dev).cpu()); good.append(g.cpu())
        if (i + 1) % 200 == 0:
            print(f"  synth {i+1}/{n}")
    C = torch.stack(crude)[:, None]; G = torch.stack(good)[:, None]
    torch.save({"crude": C, "good": G}, cache)
    return C, G


def render(ax, mesh, title):
    if mesh is None or len(mesh.faces) == 0:
        ax.set_title(title + "\n(empty)", fontsize=7); return
    V, F_ = mesh.vertices, mesh.faces; tris = V[F_]
    fy = tris[:, :, 1].mean(1); col = plt.cm.bone(0.25 + 0.6 * (fy - fy.min()) / (np.ptp(fy) + 1e-9))
    ax.add_collection3d(Poly3DCollection(tris[:, :, [0, 2, 1]], facecolors=col, edgecolors="k", linewidths=0.04))
    x, z, y = V[:, 0], V[:, 2], V[:, 1]
    ax.set_xlim(x.min(), x.max()); ax.set_ylim(z.min(), z.max()); ax.set_zlim(y.min(), y.max())
    try: ax.set_box_aspect((np.ptp(x), np.ptp(z), max(np.ptp(y), 1)))
    except Exception: pass
    ax.view_init(elev=16, azim=-55); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title(title, fontsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--R", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--base", type=int, default=24)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device
    OUT.mkdir(parents=True, exist_ok=True)

    print("[synth] building (crude->good) pairs ...")
    C, G = synthesize(args.n, args.R, dev)
    n = len(C); nv = max(8, n // 10)
    Ctr, Gtr = C[nv:].to(dev), G[nv:].to(dev)
    Cva, Gva = C[:nv].to(dev), G[:nv].to(dev)
    print(f"[data] train {len(Ctr)} val {len(Cva)} | R={args.R}")

    net = RefineUNet3D(base=args.base, delta_scale=2 * CLIP).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=2e-4)
    nt = len(Ctr)
    for ep in range(1, args.epochs + 1):
        net.train(); order = torch.randperm(nt, device=dev); tot = 0.0
        for b in range(0, nt, args.bs):
            bi = order[b:b + args.bs]
            opt.zero_grad()
            pred = net(Ctr[bi])
            loss = surface_weighted_l1(pred, Gtr[bi], band=0.08)
            loss.backward(); opt.step(); tot += loss.item() * len(bi)
        if ep % max(1, args.epochs // 15) == 0 or ep == 1:
            net.eval()
            with torch.no_grad():
                vl = surface_weighted_l1(net(Cva), Gva, band=0.08).item()
                # baseline: crude vs good (how much the UNet improves over the input)
                bl = surface_weighted_l1(Cva, Gva, band=0.08).item()
            print(f"  epoch {ep:4d} | train {tot/nt:.4f} | val {vl:.4f} (crude baseline {bl:.4f})")

    torch.save({"model": net.state_dict(), "base": args.base, "R": args.R,
                "delta_scale": 2 * CLIP, "clip": CLIP}, OUT / "refine_unet.pth")
    print(f"[save] {OUT/'refine_unet.pth'}")

    # render crude | UNet-refined | good target for a few val samples
    net.eval()
    bb = (-1, -1, -1, 1, 1, 1)
    k = 5
    with torch.no_grad():
        ref = net(Cva[:k]).cpu()
    fig = plt.figure(figsize=(9, 3 * k))
    for r in range(k):
        for cidx, (vol, t) in enumerate([(Cva[r, 0].cpu(), "crude sculpt"),
                                         (ref[r, 0], "UNet refined (learned)"),
                                         (Gva[r, 0].cpu(), "good target")]):
            ax = fig.add_subplot(k, 3, r * 3 + cidx + 1, projection="3d")
            m = grid_to_mesh(vol, bb, 0.0)
            render(ax, m, t)
    fig.suptitle("Amortized learned refine: crude sculpt -> coherent building (one forward pass)", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "refine_unet_eval.png", dpi=110); plt.close(fig)
    print(f"[save] {OUT/'refine_unet_eval.png'}")


if __name__ == "__main__":
    main()
