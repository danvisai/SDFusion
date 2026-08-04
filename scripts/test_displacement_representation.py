"""Green-light test for less-boxy generation: can recipe_base + displacement capture
REAL BuildingNet detail the recipe alone can't?

For each asset: base = its B+.7 recipe fit (IoU~0.66 to GT, boxy). Fit a per-building
displacement field so base + d matches the GT SDF in the surface band. Measure how much
the displacement reduces the surface-band L1 and lifts the footprint IoU vs base alone.

If base+displacement reconstructs real buildings well, the decomposition is a good target
for a GENERATIVE displacement model (the path to non-boxy generation). If it stalls, the
recipe base is too poor a starting point and we rethink.

  env -u LD_PRELOAD -u LD_LIBRARY_PATH PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
    ./sdfusion/bin/python scripts/test_displacement_representation.py --n 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from models.networks.diff_recipe import build_diff_recipe
from models.networks import recipe_param_space as ps
from models.networks.displacement_field import fit_displacement, normalizer
from scene.sdf_primitives import grid_to_mesh

SDF_DIR = REPO / "data/BuildingNet_dataset_v0_1/resolution_64"
FITS = REPO / "outputs/fit_recipes_buildingnet/best_params.npz"
OUT = REPO / "outputs/displacement_repr"


def load_gt(aid):
    p = SDF_DIR / aid / "ori_sample_grid.h5"
    if not p.exists():
        return None
    with h5py.File(p, "r") as f:
        sdf = f["pc_sdf_sample"][:].reshape(64, 64, 64).astype(np.float32)
        fp = f["footprint"][0].astype(np.uint8)
        params = f["sdf_params"][:].astype(np.float32)
    return sdf, fp, params


def query_grid(bbox, device):
    x0, y0, z0, x1, y1, z1 = [float(v) for v in bbox]
    xs = torch.linspace(x0, x1, 64, device=device)
    ys = torch.linspace(y0, y1, 64, device=device)
    zs = torch.linspace(z0, z1, 64, device=device)
    Z, Y, X = torch.meshgrid(zs, ys, xs, indexing="ij")
    g = torch.stack([X, Y, Z], -1).reshape(-1, 3)
    g[:, 1] -= y0
    return g


def fp_iou(vals, gt_fp):
    occ = (vals.reshape(64, 64, 64) <= 0).any(dim=1).cpu().numpy()
    gt = gt_fp > 0
    u = (occ | gt).sum()
    return float((occ & gt).sum() / u) if u else 0.0


def render(ax, mesh, title):
    if mesh is None or len(mesh.faces) == 0:
        ax.set_title(title + "\n(empty)", fontsize=7); return
    V, F = mesh.vertices, mesh.faces; tris = V[F]
    fy = tris[:, :, 1].mean(1); c = plt.cm.viridis(0.15 + 0.7 * (fy - fy.min()) / (np.ptp(fy) + 1e-9))
    ax.add_collection3d(Poly3DCollection(tris[:, :, [0, 2, 1]], facecolors=c, edgecolors="none"))
    x, z, y = V[:, 0], V[:, 2], V[:, 1]
    ax.set_xlim(x.min(), x.max()); ax.set_ylim(z.min(), z.max()); ax.set_zlim(0, max(y.max(), 1))
    try: ax.set_box_aspect((np.ptp(x), np.ptp(z), max(np.ptp(y), 1)))
    except Exception: pass
    ax.view_init(elev=20, azim=-55); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title(title, fontsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--band", type=float, default=0.08)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device
    OUT.mkdir(parents=True, exist_ok=True)

    fits = np.load(FITS, allow_pickle=True)["fits"].item()
    # mix of qualities: sort by fit IoU, take spread
    items = sorted(fits.items(), key=lambda kv: kv[1]["iou"])
    idx = np.linspace(0, len(items) - 1, args.n, dtype=int)
    chosen = [items[i] for i in idx]

    rows, sheet = [], []
    print(f"{'asset':32s} {'style':12s} {'band L1 base':>12} {'+disp':>8} {'fpIoU base':>11} {'+disp':>8}")
    for aid, v in chosen:
        gt = load_gt(aid)
        if gt is None:
            continue
        sdf_np, fp_np, sp = gt
        style = v["style"]; params = np.asarray(v["params"], np.float32)
        poly = torch.tensor(np.asarray(v["polygon"], np.float32), device=dev)
        bbox = sp
        height = float(bbox[4] - bbox[1])
        grid = query_grid(bbox, dev)
        module = build_diff_recipe(style)[0].to(dev)
        with torch.no_grad():
            base = module(torch.tensor(params, device=dev), poly,
                          torch.tensor(height, device=dev), grid)
        gtv = torch.tensor(sdf_np.reshape(-1), device=dev)
        band = (gtv.abs() < args.band).float()
        bn = band.sum().clamp_min(1.0)
        l1_base = float(((base - gtv).abs() * band).sum() / bn)

        # fit displacement: base + d -> GT in band
        norm = normalizer(tuple(float(x) for x in [bbox[0], 0.0, bbox[2], bbox[3], bbox[4] - bbox[1], bbox[5]]))
        # use the shifted grid coords for the field (consistent with base eval)
        gshift = grid
        resid_max = float((gtv - base).abs().max())
        out_scale = float(min(max(resid_max * 1.05, 1.0), 6.0))
        field = fit_displacement(base, gtv, norm(gshift), steps=args.steps, device=dev,
                                 n_freq=8, hidden=192, band=args.band * 1.5, reg=0.01,
                                 out_scale=out_scale)
        with torch.no_grad():
            disp = base + field(norm(gshift))
        l1_disp = float(((disp - gtv).abs() * band).sum() / bn)
        iou_b = fp_iou(base, fp_np); iou_d = fp_iou(disp, fp_np)
        print(f"{aid[:32]:32s} {style:12s} {l1_base:12.4f} {l1_disp:8.4f} {iou_b:11.3f} {iou_d:8.3f}")
        rows.append((l1_base, l1_disp, iou_b, iou_d))
        if len(sheet) < 5:
            sheet.append((aid, sdf_np, base, disp, bbox, height))

    rows = np.array(rows)
    print("\n[summary] n=%d" % len(rows))
    print(f"  surface-band L1:  base {rows[:,0].mean():.4f} -> base+disp {rows[:,1].mean():.4f} "
          f"({100*(1-rows[:,1].mean()/rows[:,0].mean()):.0f}% reduction)")
    print(f"  footprint IoU:    base {rows[:,2].mean():.3f} -> base+disp {rows[:,3].mean():.3f}")

    # sheet: GT | base | base+disp
    fig = plt.figure(figsize=(9, 3 * len(sheet)))
    for r, (aid, sdf_np, base, disp, bbox, h) in enumerate(sheet):
        bb = (bbox[0], 0.0, bbox[2], bbox[3], bbox[4] - bbox[1], bbox[5])
        gt_mesh = grid_to_mesh(torch.tensor(sdf_np), bb, 0.0)
        bm = grid_to_mesh(base.reshape(64, 64, 64), bb, 0.0)
        dm = grid_to_mesh(disp.reshape(64, 64, 64), bb, 0.0)
        for c, (m, t) in enumerate([(gt_mesh, "GT"), (bm, "recipe base"), (dm, "base+displacement")]):
            ax = fig.add_subplot(len(sheet), 3, r * 3 + c + 1, projection="3d")
            render(ax, m, f"{aid[:16]}\n{t}")
    fig.suptitle("Can recipe-base + displacement represent REAL BuildingNet detail?", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "displacement_repr.png", dpi=100); plt.close(fig)
    print(f"[save] {OUT/'displacement_repr.png'}")


if __name__ == "__main__":
    main()
