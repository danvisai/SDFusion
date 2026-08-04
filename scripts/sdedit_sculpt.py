"""SDEdit sculpt test — project a crude SDF edit onto the learned building manifold.

Route A from the AI-sculpting plan (see memory project_ai_sculpting_research): the
Stage-3a conditional latent diffusion is a generative *prior* over buildings. SDEdit
makes it a sculptor: encode the user's crude edit -> add `strength` worth of noise ->
denoise with the conditional prior -> the edit's massing survives but the result snaps
to a plausible building. This validates the mechanism end-to-end:

    clean corpus building  ->  union a crude tower/wing  ->  sdedit(strength)  ->  coherent

Outputs OBJ meshes for: before, crude edit, and sdedit at several strengths (+ full
generation for reference) into --out_dir, plus an occupancy report.

Run:
  env -u LD_PRELOAD -u LD_LIBRARY_PATH PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
    ./sdfusion/bin/python scripts/sdedit_sculpt.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from datasets.stage3a_dataset import Stage3aDataset
from models.stage3a_model import Stage3aModel
from scene.sdf_primitives import grid_to_mesh

BBOX_N = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)   # Frame-N voxel grid
TRUNC = 0.2


def _grid_coords(device):
    """(D,H,W,3) coords in Frame-N, axis order (z=D, y=H, x=W)."""
    g = torch.linspace(-1.0, 1.0, 64, device=device)
    Z, Y, X = torch.meshgrid(g, g, g, indexing="ij")
    return Z, Y, X


def _box_sdf(center, half, device):
    """Analytic box SDF on the 64^3 Frame-N grid. center/half are (z,y,x)."""
    Z, Y, X = _grid_coords(device)
    qz = (Z - center[0]).abs() - half[0]
    qy = (Y - center[1]).abs() - half[1]
    qx = (X - center[2]).abs() - half[2]
    q = torch.stack([qz, qy, qx], 0)
    outside = torch.linalg.vector_norm(q.clamp(min=0.0), dim=0)
    inside = q.max(0).values.clamp(max=0.0)
    return outside + inside


def apply_edit(sdf, kind, device):
    """Crude primitive sculpt on a (1,1,64,64,64) SDF. Returns edited copy (clamped)."""
    s = sdf[0, 0].clone()
    if kind == "tower":      # tall thin mass at a back corner, sticking above the roof
        box = _box_sdf(center=(0.45, 0.1, 0.45), half=(0.13, 0.95, 0.13), device=device)
        out = torch.minimum(s, box)            # union (add mass)
    elif kind == "wing":     # a side wing extending the footprint
        box = _box_sdf(center=(0.0, -0.25, 0.6), half=(0.35, 0.6, 0.22), device=device)
        out = torch.minimum(s, box)
    elif kind == "dent":     # carve a notch out of the upper-front
        box = _box_sdf(center=(-0.5, 0.4, 0.0), half=(0.3, 0.45, 0.3), device=device)
        out = torch.maximum(s, -box)           # subtract mass
    else:
        raise ValueError(kind)
    return out.clamp(-TRUNC, TRUNC)[None, None]


def occ(sdf):
    return float((sdf[0, 0] <= 0).float().mean().item())


def export(sdf, path):
    m = grid_to_mesh(sdf.detach().cpu()[0, 0], BBOX_N, iso=0.0)
    if m is None:
        print(f"  [!] no surface -> {path}")
        return 0
    m.export(path)
    return len(m.vertices)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage3a_ckpt", default=str(REPO / "logs_building/continue-2026-05-29T22-24-26-stage3a-bs32-resume2-from-iter9k/ckpt/stage3a_steps-latest.pth"))
    ap.add_argument("--stage3a_cfg", default=str(REPO / "configs/stage3a_sdf_diffusion.yaml"))
    ap.add_argument("--vq_cfg", default=str(REPO / "configs/vqvae_bnet.yaml"))
    ap.add_argument("--vq_ckpt", default=str(REPO / "logs_building/2025-05-19T19-58-28-vqvae-building-all-res64-LR1e-4-T0.2-release/ckpt/vqvae_steps-latest.pth"))
    ap.add_argument("--style", default="modern")
    ap.add_argument("--sample", type=int, default=0, help="recipe sample index")
    ap.add_argument("--edit", default="tower", choices=["tower", "wing", "dent"])
    ap.add_argument("--strengths", default="0.0,0.3,0.5,0.7")
    ap.add_argument("--ddim_steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default="/tmp/sdedit_test")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    # --- 1) pull a clean corpus building (conditioning EXACTLY as trained) -----
    ds_opt = SimpleNamespace(
        dataroot=str(REPO / "data"),
        recipe_aug_root=str(REPO / "data/recipe_augmentation_v1"),
        heights_csv=str(REPO / "outputs/stage3_metadata/asset_dimensions.csv"),
        recipe_aug_ratio=1.0, trunc_thres=TRUNC, augment=False, seed=0,
        recipe_styles=[args.style],
    )
    ds = Stage3aDataset(); ds.initialize(ds_opt, phase="train", res=64)
    item = ds[args.sample % len(ds)]
    dev = args.device
    sdf0 = item["sdf"].view(1, 1, 64, 64, 64).to(dev)
    data = {
        "sdf": sdf0,
        "fp": item["fp"].view(1, 1, 64, 64).to(dev),
        "class_id": item["class_id"].view(1).to(dev),
        "style_id": item["style_id"].view(1).to(dev),
        "height": item["height"].view(1).to(dev),
    }
    print(f"[sdedit] style={args.style} class_id={int(item['class_id'])} "
          f"height={float(item['height']):.3f} src={item['source']}  occ(before)={occ(sdf0):.4f}")

    # --- 2) crude edit --------------------------------------------------------
    sdf_edit = apply_edit(sdf0, args.edit, dev)
    print(f"[sdedit] edit={args.edit}  occ(edited)={occ(sdf_edit):.4f}")

    # --- 3) load the prior ----------------------------------------------------
    m_opt = SimpleNamespace(
        isTrain=False, device=dev, df_cfg=args.stage3a_cfg, vq_cfg=args.vq_cfg,
        vq_ckpt=args.vq_ckpt, ckpt=args.stage3a_ckpt, ddim_steps=args.ddim_steps,
        debug="0", gpu_ids=[0], ckpt_dir="/tmp",
        latent_size_HW=(16, 16), latent_size_D=16,
    )
    print(f"[sdedit] loading prior {Path(args.stage3a_ckpt).parent.parent.name}")
    model = Stage3aModel(); model.initialize(m_opt)

    # --- 4) before / edited meshes -------------------------------------------
    print(f"  before  V={export(sdf0, str(out / 'a_before.obj'))}")
    print(f"  edited  V={export(sdf_edit, str(out / 'b_edited.obj'))}  (crude sculpt)")

    # --- 5) SDEdit at several strengths --------------------------------------
    edit_data = dict(data); edit_data["sdf"] = sdf_edit
    for s in [float(x) for x in args.strengths.split(",")]:
        torch.manual_seed(0)
        sdf_out = model.sdedit(edit_data, strength=s, ddim_steps=args.ddim_steps,
                               uc_scale=args.guidance)
        v = export(sdf_out, str(out / f"c_sdedit_s{s:.2f}.obj"))
        print(f"  sdedit  strength={s:.2f}  occ={occ(sdf_out):.4f}  V={v}")

    # --- 6) full generation (reference: prior alone, no edit) ----------------
    torch.manual_seed(0)
    sdf_gen = model.inference(data, ddim_steps=args.ddim_steps, uc_scale=args.guidance)
    print(f"  fullgen (ref) occ={occ(sdf_gen):.4f}  V={export(sdf_gen, str(out / 'd_fullgen.obj'))}")
    print(f"[sdedit] meshes -> {out}")


if __name__ == "__main__":
    main()
