"""Plateau check — does training the SDEdit prior LONGER help?

Runs SDEdit with the SAME building + SAME crude edit + SAME seed across several checkpoints
(15k / 20k / 30k), varying ONLY the checkpoint. If the outputs look the same -> training has
plateaued, stop and move to wiring the composer/detail (the real quality lever). If later
checkpoints are visibly sharper -> there's headroom, keep training.

  env -u LD_PRELOAD -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0 \
    ./sdfusion/bin/python scripts/sdedit_plateau_check.py
"""
from __future__ import annotations
import argparse, sys, gc
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from datasets.bag3d_dataset import Bag3dDataset
from models.stage3a_model import Stage3aModel
from scene.sdf_primitives import grid_to_mesh

BBOX = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0); TRUNC = 0.2
CK = REPO / "logs_building/2026-06-05T15-02-24-bag3d-prior-fast/ckpt"


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
    m = Stage3aModel(); m.initialize(opt); return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--sample", type=int, default=5)
    ap.add_argument("--ckpts", default="15000,20000,latest")
    ap.add_argument("--out", default="outputs/sdedit_plateau/plateau.png")
    args = ap.parse_args()
    dev = args.device

    ds = Bag3dDataset(); ds.initialize(SimpleNamespace(bag3d_h5="/dev/shm/bag3d_fast.h5", trunc_thres=TRUNC, augment=False), "train")
    it = ds[args.sample]
    sdf0 = it["sdf"].view(1,1,64,64,64).to(dev)
    base = {"sdf": sdf0, "fp": it["fp"].view(1,1,64,64).to(dev), "class_id": it["class_id"].view(1).to(dev),
            "style_id": it["style_id"].view(1).to(dev), "height": it["height"].view(1).to(dev)}
    sdf_edit = tower_edit(sdf0, dev)
    edit_data = dict(base); edit_data["sdf"] = sdf_edit

    labels = args.ckpts.split(",")
    panels = [("before (real)", mesh(sdf0)), ("edited (+tower)", mesh(sdf_edit))]
    print(f"[plateau] sample {args.sample}, strength {args.strength}")
    for lab in labels:
        ckpt = CK / f"stage3a_steps-{lab}.pth"
        if not ckpt.exists():
            print(f"  {lab}: MISSING ({ckpt.name})"); continue
        m = load_model(ckpt, dev)
        torch.manual_seed(0)
        out = m.sdedit(edit_data, strength=args.strength, ddim_steps=50, uc_scale=2.0)
        mm = mesh(out); nv = 0 if mm is None else len(mm.vertices)
        occ = float((out <= 0).float().mean())
        print(f"  ckpt {lab:8s}  occ={occ:.3f}  verts={nv}")
        panels.append((f"sdedit @ {lab}", mm))
        del m; gc.collect()
        if dev == "cuda": torch.cuda.empty_cache()

    fig = plt.figure(figsize=(3.2*len(panels), 3.5))
    for i,(t,mm) in enumerate(panels):
        ax = fig.add_subplot(1,len(panels),i+1,projection="3d")
        if mm is not None:
            v,fc=np.asarray(mm.vertices),np.asarray(mm.faces)
            ax.plot_trisurf(v[:,0],v[:,2],fc,v[:,1],color="#b9c4cf",edgecolor="none",shade=True)
            lim=[v.min(),v.max()];ax.set_xlim(lim);ax.set_ylim(lim);ax.set_zlim(lim)
        ax.view_init(elev=20,azim=-58);ax.set_box_aspect((1,1,1));ax.set_axis_off();ax.set_title(t,fontsize=9)
    out_p = REPO/args.out; out_p.parent.mkdir(parents=True,exist_ok=True)
    fig.suptitle(f"Plateau check — SDEdit (strength {args.strength}) across checkpoints. Same input/seed.", fontsize=12)
    fig.tight_layout(rect=(0,0,1,0.92)); fig.savefig(out_p,dpi=88)
    print(f"[saved] {out_p}\n  -> if 15k/20k/30k look the same: PLATEAUED (stop training, wire composer).")


if __name__ == "__main__":
    main()
