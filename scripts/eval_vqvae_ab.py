"""A/B compare v1 / v2-final / v2-best VQVAE checkpoints on a small held-out
val set. Reports per-asset and summary IoU + L1 + footprint IoU, plus saves a
visual contact sheet (GT vs each recon).

Usage:
    env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \\
        scripts/eval_vqvae_ab.py \\
        --num_assets 16 \\
        --out_dir outputs/vqvae_ab_diagnostic
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from models.networks.vqvae_networks.network import VQVAE
from utils.util_3d import init_mesh_renderer, render_sdf
from utils.util import tensor2im


CHECKPOINTS = [
    {
        "label": "v1",
        "cfg":  "configs/vqvae_bnet.yaml",
        "ckpt": "logs_building/2025-05-19T19-58-28-vqvae-building-all-res64-LR1e-4-T0.2-release/ckpt/vqvae_steps-latest.pth",
    },
    {
        "label": "v2-final",
        "cfg":  "configs/vqvae_bnet_v2.yaml",
        "ckpt": "logs_building/2026-05-21T20-04-05-vqvae-building-all-res64-LR1e-4-T0.3-v2-aux-aug-cosine/ckpt/vqvae_steps-latest.pth",
    },
    {
        "label": "v2-best",
        "cfg":  "configs/vqvae_bnet_v2.yaml",
        "ckpt": "logs_building/2026-05-21T20-04-05-vqvae-building-all-res64-LR1e-4-T0.3-v2-aux-aug-cosine/ckpt/vqvae_epoch-best.pth",
    },
]


def build_vqvae(cfg_path: str, ckpt_path: str, device: torch.device) -> VQVAE:
    cfg = OmegaConf.load(cfg_path)
    mp = cfg.model.params
    net = VQVAE(mp.ddconfig, mp.n_embed, mp.embed_dim)
    state = torch.load(ckpt_path, map_location="cpu")
    sd = state["vqvae"] if "vqvae" in state else state
    net.load_state_dict(sd)
    net.eval()
    net.to(device)
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def load_sdf(h5_path: Path, trunc: float = 0.0) -> torch.Tensor:
    with h5py.File(h5_path, "r") as f:
        sdf = f["pc_sdf_sample"][:].astype(np.float32)
    t = torch.from_numpy(sdf).view(1, 1, 64, 64, 64)
    if trunc > 0.0:
        t = torch.clamp(t, -trunc, trunc)
    return t


def iou_iso0(a: torch.Tensor, b: torch.Tensor) -> float:
    pa = (a <= 0)
    pb = (b <= 0)
    inter = (pa & pb).sum().float()
    union = (pa | pb).sum().float()
    if union.item() == 0:
        return 0.0
    return float((inter / union).cpu())


def fp_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    # (B, C, D, H, W); H=Y axis (dim 3)
    pa = (a <= 0).any(dim=3)
    pb = (b <= 0).any(dim=3)
    inter = (pa & pb).sum().float()
    union = (pa | pb).sum().float()
    if union.item() == 0:
        return 0.0
    return float((inter / union).cpu())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_split", default="data/BuildingNet_dataset_v0_1/splits/val_split.txt")
    ap.add_argument("--sdf_root", default="data/BuildingNet_dataset_v0_1/resolution_64")
    ap.add_argument("--num_assets", type=int, default=16)
    ap.add_argument("--trunc_eval", type=float, default=0.2,
                    help="Truncation applied to GT for fair comparison. v1 trained at 0.2.")
    ap.add_argument("--out_dir", default="outputs/vqvae_ab_diagnostic")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pick deterministic random subset of val.
    with open(args.val_split) as f:
        val_ids = [ln.strip() for ln in f if ln.strip()]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(val_ids)
    val_ids = val_ids[: args.num_assets]
    print(f"[eval] {len(val_ids)} val assets, trunc_eval={args.trunc_eval}")

    # Load all 3 VQVAEs.
    nets = {}
    for c in CHECKPOINTS:
        if not Path(c["ckpt"]).exists():
            print(f"[eval] SKIP {c['label']}: {c['ckpt']} not found")
            continue
        print(f"[eval] loading {c['label']} from {c['ckpt']}")
        nets[c["label"]] = build_vqvae(c["cfg"], c["ckpt"], device)

    if not nets:
        sys.exit("No checkpoints loaded.")

    # Renderer for visual sheet.
    renderer = init_mesh_renderer(image_size=160, dist=1.7, elev=20, azim=20, device=device)

    rows = []
    visual_panels = []  # list of (label, PIL.Image) per row
    per_label_metrics: dict[str, dict[str, list[float]]] = {l: {"iou": [], "fp_iou": [], "l1": []} for l in nets}

    for idx, mid in enumerate(val_ids):
        sdf_path = Path(args.sdf_root) / mid / "ori_sample_grid.h5"
        if not sdf_path.exists():
            print(f"[eval]   skip {mid}: no SDF")
            continue
        gt = load_sdf(sdf_path, trunc=args.trunc_eval).to(device)
        gt_render = render_sdf(renderer, gt)[0]  # (3, H, W) tensor
        gt_img = Image.fromarray(tensor2im(gt_render.unsqueeze(0)))

        row = {"id": mid}
        row_imgs = [("GT", gt_img)]

        for label, net in nets.items():
            z = net(gt, forward_no_quant=True, encode_only=True)
            recon = net.decode_no_quant(z)
            l1 = float(F.l1_loss(recon, gt).cpu())
            i0 = iou_iso0(gt, recon)
            f_iou = fp_iou(gt, recon)
            row[f"l1_{label}"] = l1
            row[f"iou_{label}"] = i0
            row[f"fp_iou_{label}"] = f_iou
            per_label_metrics[label]["iou"].append(i0)
            per_label_metrics[label]["fp_iou"].append(f_iou)
            per_label_metrics[label]["l1"].append(l1)

            recon_render = render_sdf(renderer, recon)[0]
            recon_img = Image.fromarray(tensor2im(recon_render.unsqueeze(0)))
            row_imgs.append((label, recon_img))

        rows.append(row)
        visual_panels.append((mid, row_imgs))
        print(f"  [{idx+1}/{len(val_ids)}] {mid[:36]:36s} "
              + "  ".join(f"{label}: IoU={row[f'iou_{label}']:.2f} L1={row[f'l1_{label}']:.3f}"
                          for label in nets))

    # Write per-asset CSV.
    if rows:
        fieldnames = list(rows[0].keys())
        with open(out_dir / "per_asset.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"[eval] per-asset CSV -> {out_dir / 'per_asset.csv'}")

    # Summary.
    print("\n" + "=" * 70)
    print("SUMMARY (mean ± std over %d assets)" % len(rows))
    print("=" * 70)
    for label, m in per_label_metrics.items():
        if not m["iou"]:
            continue
        iou_a = np.array(m["iou"])
        fp_a = np.array(m["fp_iou"])
        l1_a = np.array(m["l1"])
        print(f"  {label:10s}  IoU = {iou_a.mean():.3f} ± {iou_a.std():.3f}"
              f"   fpIoU = {fp_a.mean():.3f} ± {fp_a.std():.3f}"
              f"   L1 = {l1_a.mean():.4f} ± {l1_a.std():.4f}")
    print("=" * 70)

    # Save summary too.
    with open(out_dir / "summary.txt", "w") as f:
        for label, m in per_label_metrics.items():
            if not m["iou"]:
                continue
            iou_a = np.array(m["iou"])
            fp_a = np.array(m["fp_iou"])
            l1_a = np.array(m["l1"])
            f.write(f"{label}\tIoU={iou_a.mean():.3f}±{iou_a.std():.3f}\t"
                    f"fpIoU={fp_a.mean():.3f}±{fp_a.std():.3f}\t"
                    f"L1={l1_a.mean():.4f}±{l1_a.std():.4f}\n")

    # Visual sheet: rows = val assets, cols = (GT, v1, v2-final, v2-best).
    if visual_panels:
        col_labels = [t[0] for t in visual_panels[0][1]]
        n_cols = len(col_labels)
        n_rows = len(visual_panels)
        cell_w = 160
        cell_h = 160
        margin = 8
        header_h = 24
        label_w = 200
        W = label_w + n_cols * (cell_w + margin) + margin
        H = header_h + n_rows * (cell_h + margin) + margin
        sheet = Image.new("RGB", (W, H), "white")
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(sheet)
        try:
            font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 12)
        except Exception:
            font = ImageFont.load_default()
        # Column headers
        for ci, cl in enumerate(col_labels):
            x = label_w + ci * (cell_w + margin) + margin
            draw.text((x, 4), cl, fill="black", font=font)
        for ri, (mid, panels) in enumerate(visual_panels):
            y = header_h + ri * (cell_h + margin) + margin
            draw.text((4, y + cell_h // 2 - 6), mid[:24], fill="black", font=font)
            for ci, (_, img) in enumerate(panels):
                x = label_w + ci * (cell_w + margin) + margin
                if img.size != (cell_w, cell_h):
                    img = img.resize((cell_w, cell_h))
                sheet.paste(img, (x, y))
        sheet.save(out_dir / "visual_sheet.png")
        print(f"[eval] visual sheet -> {out_dir / 'visual_sheet.png'}")


if __name__ == "__main__":
    main()
