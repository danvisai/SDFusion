"""Inference path audit for Stage 3a.

Hypothesis: training loss is dropping but inference produces noise. This script
isolates *which* step in the inference pipeline is broken by comparing 5 outputs
on a single train-set example:

  (A) GT SDF                                     — reference
  (B) VQVAE round-trip of GT SDF                 — decode ceiling
  (C) Single-step denoise from t=50 (low noise)  — does model learn near-clean signal?
  (D) Single-step denoise from t=500 (mid noise) — does model learn medium signal?
  (E) Full DDIM (100 steps) inference            — what the visualizer renders

If (B) is clean, the VQVAE is fine.
If (C) is clean but (E) is noise → DDIM loop bug.
If (C) is also noise → model never learned to denoise (conditioning bug).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

REPO = Path("/scratch/gilbreth/dsimhadr/GenerativeTowns/SDFusion")
sys.path.insert(0, str(REPO))

from datasets.stage3a_dataset import Stage3aDataset
from models.stage3a_model import Stage3aModel
from options.train_options import TrainOptions
from utils.util_3d import init_mesh_renderer, render_sdf
from utils.util import tensor2im


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vq_ckpt", default="logs_building/2025-05-19T19-58-28-vqvae-building-all-res64-LR1e-4-T0.2-release/ckpt/vqvae_steps-latest.pth")
    ap.add_argument("--vq_cfg", default="configs/vqvae_bnet.yaml")
    ap.add_argument("--df_cfg", default="configs/stage3a_sdf_diffusion.yaml")
    ap.add_argument("--out_dir", default="outputs/audit_stage3a_2026_05_29")
    ap.add_argument("--n_samples", type=int, default=4, help="Examples to audit.")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def build_opt(args):
    """Build an argparse-like Namespace matching what Stage3aModel expects."""
    # Reuse TrainOptions's parser by mocking sys.argv.
    sys.argv = [
        "audit.py",
        "--name", "audit",
        "--logs_dir", "logs_building",
        "--gpu_ids", "0",
        "--lr", "1e-4",
        "--batch_size", str(args.n_samples),
        "--model", "stage3a",
        "--df_cfg", args.df_cfg,
        "--vq_cfg", args.vq_cfg,
        "--vq_ckpt", args.vq_ckpt,
        "--ckpt", args.ckpt,
        "--dataset_mode", "stage3a",
        "--cat", "all",
        "--res", "64",
        "--trunc_thres", "0.2",
        "--recipe_aug_ratio", "0.7",
        "--display_freq", "4000",
        "--print_freq", "50",
        "--save_latest_freq", "2000",
        "--save_steps_freq", "10000",
        "--total_iters", "10",  # tiny — we don't train
        "--augment",
        "--use_adamw_cosine",
        "--warmup_steps", "1",
        "--cosine_total_steps", "10",
        "--dataroot", "data",
        "--debug", "0",
        "--continue_train",  # so isTrain=True and ckpt loads with opt
    ]
    opt = TrainOptions().parse_and_setup()
    return opt


def make_dataset(opt, n_samples: int):
    ds = Stage3aDataset()
    ds.initialize(opt, phase="train", cat="all", res=64)
    # Pull n_samples deterministically
    items = []
    for i in range(n_samples):
        items.append(ds[i])
    # Stack — only keep keys the model uses (skip strings like ids)
    needed = {"sdf", "fp", "class_id", "style_id", "height"}
    batch = {}
    for k in items[0].keys():
        if k not in needed:
            continue
        vs = [it[k] for it in items]
        if torch.is_tensor(vs[0]):
            batch[k] = torch.stack(vs, dim=0)
        else:
            batch[k] = torch.tensor(vs)
    return batch


@torch.no_grad()
def main():
    args = parse_args()
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("[*] building opt + model...")
    opt = build_opt(args)
    model = Stage3aModel()
    model.initialize(opt)
    model.switch_eval()

    device = model.device
    print(f"[*] device={device}, scale_factor={model.scale_factor}")

    print("[*] loading train batch...")
    batch = make_dataset(opt, args.n_samples)
    model.set_input(batch, max_sample=args.n_samples)

    # ----- (A) GT SDF -----
    gt_sdf = model.x                              # (B,1,64,64,64)

    # ----- (B) VQVAE round-trip -----
    z_target = model.vqvae(gt_sdf, forward_no_quant=True, encode_only=True)
    sdf_roundtrip = model.vqvae.decode_no_quant(z_target)

    # Build conditioning once (matches inference)
    _, _, D, H, W = z_target.shape
    fp3d = model._build_fp3d_for(D, H, W) * model.fp3d_concat_scale
    ctx = model._build_global_context()
    cond = {"c_concat": [fp3d], "c_crossattn": [ctx]}

    # ----- (C) Single-step denoise from t=50 -----
    z_scaled = z_target * model.scale_factor       # scaled latent (training space)
    B = z_scaled.shape[0]
    for t_val, label in [(50, "C_t050"), (500, "D_t500"), (900, "single_t900")]:
        t = torch.full((B,), t_val, device=device, dtype=torch.long)
        noise = torch.randn_like(z_scaled)
        x_noisy = model.q_sample(z_scaled, t, noise)
        eps_pred = model.apply_model(x_noisy, t, cond)
        x0_lat_scaled = model._predict_x0_from_eps(x_noisy, t, eps_pred)
        x0_lat = x0_lat_scaled / model.scale_factor
        sdf = model.vqvae.decode_no_quant(x0_lat)
        # save
        np.save(out_dir / f"{label}_sdf.npy", sdf.detach().cpu().numpy())
        render(model, sdf, out_dir / f"{label}.png")
        # quick stats
        same_sign = ((sdf <= 0) == (gt_sdf <= 0)).float().mean().item()
        l1 = (sdf - gt_sdf).abs().mean().item()
        print(f"  t={t_val:4d}  sign_match={same_sign:.3f}  L1={l1:.3f}")

    # ----- (E) Full DDIM inference -----
    print("[*] running full DDIM inference (100 steps)...")
    sdf_ddim = model.inference(batch, max_sample=args.n_samples)
    np.save(out_dir / "E_ddim_sdf.npy", sdf_ddim.detach().cpu().numpy())
    render(model, sdf_ddim, out_dir / "E_ddim.png")
    sm_ddim = ((sdf_ddim <= 0) == (gt_sdf <= 0)).float().mean().item()
    l1_ddim = (sdf_ddim - gt_sdf).abs().mean().item()
    print(f"  DDIM   sign_match={sm_ddim:.3f}  L1={l1_ddim:.3f}")

    # ----- (A) (B) render -----
    np.save(out_dir / "A_gt_sdf.npy", gt_sdf.detach().cpu().numpy())
    np.save(out_dir / "B_roundtrip_sdf.npy", sdf_roundtrip.detach().cpu().numpy())
    render(model, gt_sdf, out_dir / "A_gt.png")
    render(model, sdf_roundtrip, out_dir / "B_vqvae_roundtrip.png")
    sm_rt = ((sdf_roundtrip <= 0) == (gt_sdf <= 0)).float().mean().item()
    l1_rt = (sdf_roundtrip - gt_sdf).abs().mean().item()
    print(f"  B_rt   sign_match={sm_rt:.3f}  L1={l1_rt:.3f}")

    # Build the comparison sheet
    panels = [
        ("A GT", "A_gt.png"),
        ("B VQVAE round-trip", "B_vqvae_roundtrip.png"),
        ("C 1-step t=50", "C_t050.png"),
        ("D 1-step t=500", "D_t500.png"),
        ("E DDIM 100-step", "E_ddim.png"),
    ]
    cells = []
    cell_w, cell_h = 256, 256
    for _, name in panels:
        img = Image.open(out_dir / name).convert("RGB")
        cells.append(img.resize((cell_w, cell_h), Image.LANCZOS))

    sheet_w = len(panels) * (cell_w + 8) + 8
    sheet_h = cell_h + 32
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    for i, (label, _) in enumerate(panels):
        x = 8 + i * (cell_w + 8)
        draw.text((x + 4, 4), label, fill="black", font=font)
        sheet.paste(cells[i], (x, 24))
    sheet.save(out_dir / "audit_sheet.png")
    print(f"[*] wrote {out_dir / 'audit_sheet.png'}")


def render(model, sdf, out_path: Path):
    """Render the first sample of the batch and save as PNG."""
    img_t = render_sdf(model.renderer, sdf[:1])
    arr = tensor2im(img_t.data)
    Image.fromarray(arr).save(out_path)


if __name__ == "__main__":
    main()
