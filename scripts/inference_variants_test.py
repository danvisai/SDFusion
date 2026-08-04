"""Test multiple inference strategies on the iter-9000 ckpt.

Hypothesis: the SDFusion DDIM sampler has a subtle bug. We tested above and
found single-step from noisy-GT at t in {50, 500, 900} gives sign_match >= 0.89.
DDIM 100-step gives sign_match = 0.64 — DDIM is broken.

This script tests four inference variants from a TRUE noise start (no GT seed):

  V1  Single-step from pure noise at t=999.
  V2  Single-step from pure noise at t=500.
  V3  Custom DDIM (50 steps, eta=0) using model.apply_model directly,
      bypassing the SDFusion DDIMSampler class.
  V4  Custom DDIM (50 steps, eta=0) WITH classifier-free guidance scale 2.0
      (uc_scale=2) — see if guidance helps.

For each variant: report sign_match + L1 vs GT, render single PNG, then build
a comparison sheet.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPO = Path("/scratch/gilbreth/dsimhadr/GenerativeTowns/SDFusion")
sys.path.insert(0, str(REPO))

from datasets.stage3a_dataset import Stage3aDataset
from models.stage3a_model import Stage3aModel
from options.train_options import TrainOptions
from utils.util_3d import render_sdf
from utils.util import tensor2im


def build_opt(ckpt: str):
    sys.argv = [
        "test.py",
        "--name", "test_inference",
        "--logs_dir", "logs_building",
        "--gpu_ids", "0", "--lr", "1e-4", "--batch_size", "4",
        "--model", "stage3a",
        "--df_cfg", "configs/stage3a_sdf_diffusion.yaml",
        "--vq_cfg", "configs/vqvae_bnet.yaml",
        "--vq_ckpt", "logs_building/2025-05-19T19-58-28-vqvae-building-all-res64-LR1e-4-T0.2-release/ckpt/vqvae_steps-latest.pth",
        "--ckpt", ckpt,
        "--dataset_mode", "stage3a", "--cat", "all", "--res", "64",
        "--trunc_thres", "0.2", "--recipe_aug_ratio", "0.7",
        "--display_freq", "4000", "--print_freq", "50",
        "--save_latest_freq", "2000", "--save_steps_freq", "10000",
        "--total_iters", "10",
        "--augment", "--use_adamw_cosine",
        "--warmup_steps", "1", "--cosine_total_steps", "10",
        "--dataroot", "data", "--debug", "0",
        "--continue_train",
    ]
    return TrainOptions().parse_and_setup()


def make_batch(opt, n: int):
    ds = Stage3aDataset(); ds.initialize(opt, phase="train", cat="all", res=64)
    items = [ds[i] for i in range(n)]
    needed = {"sdf", "fp", "class_id", "style_id", "height"}
    batch = {}
    for k in items[0]:
        if k not in needed:
            continue
        vs = [it[k] for it in items]
        if torch.is_tensor(vs[0]):
            batch[k] = torch.stack(vs, dim=0)
    return batch


@torch.no_grad()
def v_single_step(model, cond, t_val: int, shape):
    """Single-step prediction from PURE NOISE at timestep t_val."""
    B = shape[0]
    device = model.device
    x_T = torch.randn(*shape, device=device)
    t = torch.full((B,), t_val, device=device, dtype=torch.long)
    eps = model.apply_model(x_T, t, cond)
    x0_scaled = model._predict_x0_from_eps(x_T, t, eps)
    x0_lat = x0_scaled / model.scale_factor
    return model.vqvae.decode_no_quant(x0_lat)


@torch.no_grad()
def v_custom_ddim(model, cond, n_steps: int, shape, uc_scale: float = 1.0):
    """Bypass SDFusion DDIMSampler — do DDIM by hand using model.apply_model.

    Standard DDIM with eta=0:
      pred_x0 = (x_t - sqrt(1-a_t) * eps) / sqrt(a_t)
      x_prev  = sqrt(a_prev) * pred_x0 + sqrt(1 - a_prev) * eps
    """
    B = shape[0]
    device = model.device
    x = torch.randn(*shape, device=device)

    T = model.num_timesteps
    # Uniformly spaced DDIM steps from T-1 down to 0.
    steps = np.linspace(0, T - 1, n_steps, dtype=int)[::-1]  # high -> low
    a_cum = model.alphas_cumprod                              # (T,)

    # Build unconditional cond for CFG, if needed.
    if uc_scale != 1.0:
        uc = {
            "c_concat":  [torch.zeros_like(cond["c_concat"][0])],
            "c_crossattn": [torch.zeros_like(cond["c_crossattn"][0])],
        }

    for i, t_val in enumerate(steps):
        t = torch.full((B,), int(t_val), device=device, dtype=torch.long)
        if uc_scale == 1.0:
            eps = model.apply_model(x, t, cond)
        else:
            # CFG: stack uc + cond on batch, run once, split, combine
            x_in = torch.cat([x, x], dim=0)
            t_in = torch.cat([t, t], dim=0)
            c_in = {k: [torch.cat([uc[k][0], cond[k][0]], dim=0)] for k in cond}
            eps_pair = model.apply_model(x_in, t_in, c_in)
            eps_uc, eps_c = eps_pair.chunk(2, dim=0)
            eps = eps_uc + uc_scale * (eps_c - eps_uc)

        a_t = a_cum[int(t_val)].view(1, 1, 1, 1, 1)
        # prev alpha = a_cum at next (lower) t in the schedule
        t_next = int(steps[i + 1]) if i + 1 < len(steps) else 0
        a_prev = a_cum[t_next].view(1, 1, 1, 1, 1) if t_next > 0 else torch.tensor(1.0, device=device).view(1, 1, 1, 1, 1)

        pred_x0 = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt().clamp_min(1e-8)
        x = a_prev.sqrt() * pred_x0 + (1 - a_prev).clamp_min(0).sqrt() * eps

    x0_lat = pred_x0 / model.scale_factor
    return model.vqvae.decode_no_quant(x0_lat)


def stats(sdf, gt):
    sm = ((sdf <= 0) == (gt <= 0)).float().mean().item()
    l1 = (sdf - gt).abs().mean().item()
    return sm, l1


def render_save(model, sdf, out: Path):
    img = render_sdf(model.renderer, sdf[:1])
    arr = tensor2im(img.data)
    Image.fromarray(arr).save(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out_dir", default="outputs/inference_variants_2026_05_29")
    args = ap.parse_args()

    torch.manual_seed(42)
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    opt = build_opt(args.ckpt)
    model = Stage3aModel(); model.initialize(opt)
    model.switch_eval()

    batch = make_batch(opt, args.n)
    model.set_input(batch, max_sample=args.n)
    gt = model.x

    # Build conditioning matching training/inference path
    _, _, D, H, W = model.vqvae(gt, forward_no_quant=True, encode_only=True).shape
    fp3d = model._build_fp3d_for(D, H, W) * model.fp3d_concat_scale
    ctx = model._build_global_context()
    cond = {"c_concat": [fp3d], "c_crossattn": [ctx]}
    shape = (args.n, model.vq_conf.model.params.embed_dim, D, H, W)

    print(f"[*] device={model.device}, scale_factor={model.scale_factor}")
    print(f"[*] running 6 inference variants from PURE NOISE")

    variants = []

    sdf = v_single_step(model, cond, 999, shape)
    sm, l1 = stats(sdf, gt); print(f"  V1  single-step t=999            sign={sm:.3f}  L1={l1:.3f}")
    render_save(model, sdf, out_dir / "V1_single_t999.png"); variants.append(("V1 1step t=999", "V1_single_t999.png", sm))

    sdf = v_single_step(model, cond, 500, shape)
    sm, l1 = stats(sdf, gt); print(f"  V2  single-step t=500            sign={sm:.3f}  L1={l1:.3f}")
    render_save(model, sdf, out_dir / "V2_single_t500.png"); variants.append(("V2 1step t=500", "V2_single_t500.png", sm))

    sdf = v_single_step(model, cond, 50, shape)
    sm, l1 = stats(sdf, gt); print(f"  V3  single-step t=50             sign={sm:.3f}  L1={l1:.3f}")
    render_save(model, sdf, out_dir / "V3_single_t050.png"); variants.append(("V3 1step t=50", "V3_single_t050.png", sm))

    sdf = v_custom_ddim(model, cond, 50, shape, uc_scale=1.0)
    sm, l1 = stats(sdf, gt); print(f"  V4  custom DDIM 50 steps uc=1.0  sign={sm:.3f}  L1={l1:.3f}")
    render_save(model, sdf, out_dir / "V4_custom_ddim_50_uc1.png"); variants.append(("V4 cstm DDIM50 uc=1", "V4_custom_ddim_50_uc1.png", sm))

    sdf = v_custom_ddim(model, cond, 100, shape, uc_scale=1.0)
    sm, l1 = stats(sdf, gt); print(f"  V5  custom DDIM 100 steps uc=1.0 sign={sm:.3f}  L1={l1:.3f}")
    render_save(model, sdf, out_dir / "V5_custom_ddim_100_uc1.png"); variants.append(("V5 cstm DDIM100 uc=1", "V5_custom_ddim_100_uc1.png", sm))

    sdf = v_custom_ddim(model, cond, 50, shape, uc_scale=2.0)
    sm, l1 = stats(sdf, gt); print(f"  V6  custom DDIM 50 steps uc=2.0  sign={sm:.3f}  L1={l1:.3f}")
    render_save(model, sdf, out_dir / "V6_custom_ddim_50_uc2.png"); variants.append(("V6 cstm DDIM50 uc=2", "V6_custom_ddim_50_uc2.png", sm))

    # GT + B for reference
    render_save(model, gt, out_dir / "A_gt.png")
    z = model.vqvae(gt, forward_no_quant=True, encode_only=True)
    sdf_rt = model.vqvae.decode_no_quant(z)
    render_save(model, sdf_rt, out_dir / "B_roundtrip.png")
    sm, l1 = stats(sdf_rt, gt); print(f"  ref VQVAE round-trip            sign={sm:.3f}  L1={l1:.3f}")

    # Sheet
    panels = [("A GT", "A_gt.png"), ("B VQ ceiling", "B_roundtrip.png")] + [(l, p) for l, p, _ in variants]
    cw, ch = 224, 224
    cells = []
    for _, name in panels:
        im = Image.open(out_dir / name).convert("RGB").resize((cw, ch), Image.LANCZOS)
        cells.append(im)
    sheet_w = len(panels) * (cw + 6) + 6
    sheet_h = ch + 28
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 11)
    except OSError:
        font = ImageFont.load_default()
    for i, ((label, _), c) in enumerate(zip(panels, cells)):
        x = 6 + i * (cw + 6)
        draw.text((x + 2, 2), label, fill="black", font=font)
        sheet.paste(c, (x, 22))
    sheet.save(out_dir / "variants_sheet.png")
    print(f"[*] wrote {out_dir / 'variants_sheet.png'}")


if __name__ == "__main__":
    main()
