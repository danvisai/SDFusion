"""Test SDEdit on the 3D BAG prior checkpoint — the REAL signal (vs full-gen speckle).

Pulls a real 3D BAG building, applies a crude tower edit, runs sdedit at several strengths,
and renders before/edited/sdedit so we can see whether the *sculpting* output is clean
(partial-noise regime) even though full-noise generation is speckly.

A/B: CFG (zero-cond unconditional branch, uc_scale) vs AUTOGUIDANCE (guide the strong 30k
model with a weaker 10k checkpoint of itself — Karras NeurIPS'24). Autoguidance sidesteps the
untrained unconditional branch (training-audit gap #3) with no retraining.

CPU works but is slow; use --device cuda for the real quality signal (GPU is the right place).
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from datasets.bag3d_dataset import Bag3dDataset
from models.stage3a_model import Stage3aModel
from scene.sdf_primitives import grid_to_mesh

BBOX = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0); TRUNC = 0.2
CKPT_DIR = REPO / "logs_building/2026-06-05T15-02-24-bag3d-prior-fast/ckpt"


def tower_edit(sdf, dev):
    s = sdf[0, 0].clone()
    g = torch.linspace(-1, 1, 64, device=dev); Z, Y, X = torch.meshgrid(g, g, g, indexing="ij")
    qz = (Z - 0.45).abs() - 0.13; qy = (Y - 0.1).abs() - 0.95; qx = (X - 0.45).abs() - 0.13
    q = torch.stack([qz, qy, qx], 0)
    box = torch.linalg.vector_norm(q.clamp(min=0), dim=0) + q.max(0).values.clamp(max=0)
    return torch.minimum(s, box).clamp(-TRUNC, TRUNC)[None, None]


def mesh(sdf):
    return grid_to_mesh(sdf.detach().cpu()[0, 0], BBOX, iso=0.0)


def load_model(ckpt, dev):
    opt = SimpleNamespace(isTrain=False, device=dev, df_cfg=str(REPO/"configs/stage3a_sdf_diffusion.yaml"),
                          vq_cfg=str(REPO/"configs/vqvae_bnet.yaml"),
                          vq_ckpt=str(REPO/"logs_building/2025-05-19T19-58-28-vqvae-building-all-res64-LR1e-4-T0.2-release/ckpt/vqvae_steps-latest.pth"),
                          ckpt=str(ckpt), ddim_steps=50, debug="0", gpu_ids=[0] if dev=="cuda" else [],
                          ckpt_dir="/tmp", latent_size_HW=(16,16), latent_size_D=16)
    print(f"[load] {Path(ckpt).name} on {dev}")
    m = Stage3aModel(); m.initialize(opt)
    return m


def add_panel(fig, ncol, row, col, title, sdf):
    ax = fig.add_subplot(2, ncol, row * ncol + col + 1, projection="3d")
    ax.set_title(title, fontsize=9); ax.set_axis_off()
    if sdf is None:
        return
    mm = mesh(sdf)
    if mm is not None:
        v, fc = np.asarray(mm.vertices), np.asarray(mm.faces)
        ax.plot_trisurf(v[:,0], v[:,2], fc, v[:,1], color="#b9c4cf", edgecolor="none", shade=True)
        lim = [v.min(), v.max()]; ax.set_xlim(lim); ax.set_ylim(lim); ax.set_zlim(lim)
    ax.view_init(elev=20, azim=-58); ax.set_box_aspect((1,1,1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(CKPT_DIR / "stage3a_steps-30000.pth"), help="strong (main) model")
    ap.add_argument("--guide_ckpt", default=str(CKPT_DIR / "stage3a_steps-10000.pth"), help="weak model for autoguidance")
    ap.add_argument("--auto_scale", type=float, default=2.0, help="autoguidance weight w")
    ap.add_argument("--uc_scale", type=float, default=2.0, help="CFG scale for the baseline row")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sample", type=int, default=5)
    ap.add_argument("--strengths", default="0.3,0.5,0.7")
    ap.add_argument("--out", default="outputs/sdedit_bag3d/sdedit_bag3d_autoguidance.png")
    ap.add_argument("--bag3d_h5", default="data/real_massing_v1/real.h5", help="input building corpus")
    args = ap.parse_args()
    dev = args.device

    ds = Bag3dDataset(); ds.initialize(SimpleNamespace(bag3d_h5=args.bag3d_h5, trunc_thres=TRUNC, augment=False), "train")
    it = ds[args.sample]
    sdf0 = it["sdf"].view(1, 1, 64, 64, 64).to(dev)
    data = {"sdf": sdf0, "fp": it["fp"].view(1, 1, 64, 64).to(dev),
            "class_id": it["class_id"].view(1).to(dev), "style_id": it["style_id"].view(1).to(dev),
            "height": it["height"].view(1).to(dev)}
    print(f"[bag3d sdedit] sample {args.sample}  occ={float((sdf0<=0).float().mean()):.3f}")

    main_m = load_model(args.ckpt, dev)
    guide_m = load_model(args.guide_ckpt, dev)

    sdf_edit = tower_edit(sdf0, dev)
    edit_data = dict(data); edit_data["sdf"] = sdf_edit
    strengths = [float(x) for x in args.strengths.split(",")]
    ncol = 2 + len(strengths)
    fig = plt.figure(figsize=(3.2 * ncol, 6.8))

    # Reference column (row 0).
    add_panel(fig, ncol, 0, 0, "before (real)", sdf0)
    add_panel(fig, ncol, 0, 1, "edited (+tower)", sdf_edit)
    add_panel(fig, ncol, 1, 0, "", None)
    add_panel(fig, ncol, 1, 1, "", None)

    for j, s in enumerate(strengths):
        torch.manual_seed(0)
        out_cfg = main_m.sdedit(edit_data, strength=s, ddim_steps=50, uc_scale=args.uc_scale)
        occ_cfg = float((out_cfg <= 0).float().mean())
        torch.manual_seed(0)
        out_ag = main_m.sdedit(edit_data, strength=s, ddim_steps=50,
                               guide_model=guide_m, auto_scale=args.auto_scale)
        occ_ag = float((out_ag <= 0).float().mean())
        print(f"  s={s:.2f}  CFG(uc={args.uc_scale}) occ={occ_cfg:.3f}   "
              f"AutoG(w={args.auto_scale}) occ={occ_ag:.3f}")
        add_panel(fig, ncol, 0, 2 + j, f"CFG s={s:.1f}", out_cfg)
        add_panel(fig, ncol, 1, 2 + j, f"AutoG s={s:.1f}", out_ag)

    out_p = REPO / args.out; out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle(f"SDEdit on 3D BAG prior — row0: CFG (uc={args.uc_scale})   "
                 f"row1: Autoguidance (10k→30k, w={args.auto_scale})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out_p, dpi=88)
    print(f"[saved] {out_p}")


if __name__ == "__main__":
    main()
