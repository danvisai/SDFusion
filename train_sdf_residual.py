from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.correction_pair_dataset import CorrectionPairDataset
from models.networks.sdf_residual_net import SDFResidualUNet


def sdf_sign_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    a_in = a <= 0
    b_in = b <= 0
    inter = (a_in & b_in).sum().float()
    union = (a_in | b_in).sum().float()
    if union.item() == 0:
        return 0.0
    return float((inter / union).detach().cpu())


def sdf_fp_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    """Hard top-down footprint IoU. Collapses H (axis dim=3 of (B,C,D,H,W))."""
    a_fp = (a <= 0).any(dim=3)
    b_fp = (b <= 0).any(dim=3)
    inter = (a_fp & b_fp).sum().float()
    union = (a_fp | b_fp).sum().float()
    if union.item() == 0:
        return 0.0
    return float((inter / union).detach().cpu())


def soft_inside(sdf: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.sigmoid(-sdf / max(tau, 1e-6))


def surface_band_smooth_l1(
    corrected: torch.Tensor, target: torch.Tensor, sigma: float, beta: float = 0.1
) -> torch.Tensor:
    """SmoothL1 on (corrected, target) weighted toward voxels near the target iso-surface."""
    band = torch.exp(-target.abs() / max(sigma, 1e-6))
    per_voxel = F.smooth_l1_loss(corrected, target, reduction="none", beta=beta)
    return (band * per_voxel).sum() / band.sum().clamp_min(1e-8)


def soft_sign_bce(corrected: torch.Tensor, target: torch.Tensor, tau: float) -> torch.Tensor:
    p = soft_inside(corrected, tau).clamp(1e-6, 1.0 - 1e-6)
    t = (target <= 0).float()
    return F.binary_cross_entropy(p, t)


def soft_footprint_bce(corrected: torch.Tensor, target: torch.Tensor, tau: float) -> torch.Tensor:
    """BCE on top-down silhouette projections; soft 'any' via amax over H (dim=3)."""
    p = soft_inside(corrected, tau).amax(dim=3).clamp(1e-6, 1.0 - 1e-6)
    t = (target <= 0).any(dim=3).float()
    return F.binary_cross_entropy(p, t)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    keys = [
        "residual_l1", "corrected_l1", "source_l1",
        "corrected_iou", "source_iou",
        "corrected_fp_iou", "source_fp_iou",
    ]
    accum: dict[str, list[float]] = {k: [] for k in keys}
    for batch in loader:
        x = batch["input"].to(device)
        source = batch["source_sdf"].to(device)
        target = batch["target_sdf"].to(device)
        residual = batch["residual_sdf"].to(device)
        pred = model(x)
        corrected = source + pred
        accum["residual_l1"].append(float(F.l1_loss(pred, residual).detach().cpu()))
        accum["corrected_l1"].append(float(F.l1_loss(corrected, target).detach().cpu()))
        accum["source_l1"].append(float(F.l1_loss(source, target).detach().cpu()))
        accum["corrected_iou"].append(sdf_sign_iou(corrected, target))
        accum["source_iou"].append(sdf_sign_iou(source, target))
        accum["corrected_fp_iou"].append(sdf_fp_iou(corrected, target))
        accum["source_fp_iou"].append(sdf_fp_iou(source, target))
    return {k: float(np.mean(v)) if v else 0.0 for k, v in accum.items()}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair_root", default="data/BuildingNet_dataset_v0_1/correction_pairs")
    ap.add_argument("--out_dir", default="Logs_GT/sdf_residual")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base_channels", type=int, default=16)
    ap.add_argument("--residual_clip", type=float, default=1.0)
    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--max_val_samples", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=111)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Composite loss weights. Set any to 0.0 to disable that term.
    ap.add_argument("--w_residual_l1", type=float, default=0.5,
                    help="SmoothL1 on (pred, target_residual). Original objective.")
    ap.add_argument("--w_band_l1", type=float, default=1.0,
                    help="Surface-band-weighted SmoothL1 on (corrected, target_sdf).")
    ap.add_argument("--w_sign_bce", type=float, default=1.0,
                    help="Soft sign BCE on corrected vs target hard occupancy.")
    ap.add_argument("--w_fp_bce", type=float, default=0.5,
                    help="Soft top-down footprint projection BCE.")
    ap.add_argument("--band_sigma", type=float, default=0.1,
                    help="band weight sigma in exp(-|target_sdf|/sigma).")
    ap.add_argument("--sign_tau", type=float, default=0.05,
                    help="sigmoid(-sdf/tau) temperature for sign / footprint BCE.")
    ap.add_argument("--augment", action="store_true",
                    help="Enable 90° Y-rotation + X/Z flip augmentation on training pairs.")
    return ap.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))

    train_ds = CorrectionPairDataset(
        args.pair_root, "train",
        max_samples=args.max_train_samples,
        residual_clip=args.residual_clip,
        augment=args.augment,
    )
    val_ds = CorrectionPairDataset(
        args.pair_root, "val",
        max_samples=args.max_val_samples,
        residual_clip=args.residual_clip,
        augment=False,
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device(args.device)
    model = SDFResidualUNet(
        in_channels=2,
        base_channels=args.base_channels,
        residual_clip=args.residual_clip,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = float("inf")
    best_iou = -1.0
    best_geom = -float("inf")
    component_keys = ["residual_l1", "band_l1", "sign_bce", "fp_bce"]
    log_path = out_dir / "loss_log.txt"
    with log_path.open("w") as log:
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            comp_sums = {k: 0.0 for k in component_keys}
            comp_counts = {k: 0 for k in component_keys}
            pbar = tqdm(train_dl, desc=f"epoch {epoch}/{args.epochs}")
            for batch in pbar:
                x = batch["input"].to(device)
                source = batch["source_sdf"].to(device)
                target = batch["target_sdf"].to(device)
                target_residual = batch["residual_sdf"].to(device)
                pred = model(x)
                corrected = source + pred

                total = pred.new_zeros(())
                if args.w_residual_l1 > 0:
                    t = F.smooth_l1_loss(pred, target_residual, beta=0.1)
                    total = total + args.w_residual_l1 * t
                    comp_sums["residual_l1"] += float(t.detach().cpu())
                    comp_counts["residual_l1"] += 1
                if args.w_band_l1 > 0:
                    t = surface_band_smooth_l1(
                        corrected, target, sigma=args.band_sigma, beta=0.1
                    )
                    total = total + args.w_band_l1 * t
                    comp_sums["band_l1"] += float(t.detach().cpu())
                    comp_counts["band_l1"] += 1
                if args.w_sign_bce > 0:
                    t = soft_sign_bce(corrected, target, tau=args.sign_tau)
                    total = total + args.w_sign_bce * t
                    comp_sums["sign_bce"] += float(t.detach().cpu())
                    comp_counts["sign_bce"] += 1
                if args.w_fp_bce > 0:
                    t = soft_footprint_bce(corrected, target, tau=args.sign_tau)
                    total = total + args.w_fp_bce * t
                    comp_sums["fp_bce"] += float(t.detach().cpu())
                    comp_counts["fp_bce"] += 1

                opt.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                losses.append(float(total.detach().cpu()))
                pbar.set_postfix(loss=f"{np.mean(losses):.4f}")

            metrics = evaluate(model, val_dl, device)
            train_loss = float(np.mean(losses)) if losses else 0.0
            comp_avg = {
                k: comp_sums[k] / max(comp_counts[k], 1) for k in component_keys
            }
            geom_score = (
                metrics["corrected_iou"]
                + metrics["corrected_fp_iou"]
                - 0.5 * metrics["corrected_l1"]
            )
            line = (
                f"epoch {epoch:03d} train_loss {train_loss:.6f} "
                f"residual_l1 {comp_avg['residual_l1']:.4f} "
                f"band_l1 {comp_avg['band_l1']:.4f} "
                f"sign_bce {comp_avg['sign_bce']:.4f} "
                f"fp_bce {comp_avg['fp_bce']:.4f} "
                f"val_corrected_l1 {metrics['corrected_l1']:.6f} "
                f"val_source_l1 {metrics['source_l1']:.6f} "
                f"val_corrected_iou {metrics['corrected_iou']:.4f} "
                f"val_source_iou {metrics['source_iou']:.4f} "
                f"val_corrected_fp_iou {metrics['corrected_fp_iou']:.4f} "
                f"val_source_fp_iou {metrics['source_fp_iou']:.4f} "
                f"geom_score {geom_score:.4f}"
            )
            print(line)
            log.write(line + "\n")
            log.flush()

            ckpt = {
                "model": model.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "metrics": metrics,
                "geom_score": geom_score,
            }
            torch.save(ckpt, out_dir / "ckpt_latest.pth")
            if metrics["corrected_l1"] < best_val:
                best_val = metrics["corrected_l1"]
                torch.save(ckpt, out_dir / "ckpt_best.pth")
            if metrics["corrected_iou"] > best_iou:
                best_iou = metrics["corrected_iou"]
                torch.save(ckpt, out_dir / "ckpt_best_iou.pth")
            if geom_score > best_geom:
                best_geom = geom_score
                torch.save(ckpt, out_dir / "ckpt_best_geom.pth")

    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
