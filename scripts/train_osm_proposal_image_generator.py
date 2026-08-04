"""Train a tiny footprint-conditioned proposal image generator.

This is a deliberately lightweight baseline, not a final diffusion model. It
learns a direct mapping from footprint/class/height features to the successful
retrieved-render conditioning images already used by Hunyuan.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset


CLASS_COUNT = 5


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--val_jsonl", required=True)
    ap.add_argument("--out_dir", default="outputs/osm_proposal_image_generator_v1")
    ap.add_argument("--image_size", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base_channels", type=int, default=32)
    ap.add_argument("--w_l1", type=float, default=1.0)
    ap.add_argument("--w_grad", type=float, default=0.25)
    ap.add_argument("--w_ssim", type=float, default=0.0)
    ap.add_argument("--w_lap", type=float, default=0.0)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260511)
    ap.add_argument("--resume_ckpt", help="Resume model weights from a previous checkpoint.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_mask(path: Path, size: int) -> torch.Tensor:
    arr = np.load(path).astype(np.float32)
    t = torch.from_numpy(arr)[None, None]
    t = F.interpolate(t, size=(size, size), mode="nearest")[0]
    return t


def load_image(path: Path, size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def scalar_map(value: float, size: int) -> torch.Tensor:
    return torch.full((1, size, size), float(value), dtype=torch.float32)


def normalized_features(row: dict) -> tuple[float, float, float]:
    height = min(float(row.get("height_m", 0.0)) / 40.0, 2.0)
    area = min(math.log1p(float(row.get("area_m2", 0.0))) / math.log1p(8000.0), 1.5)
    aspect = min(math.log(max(float(row.get("bbox_aspect", 1.0)), 1e-6)) / math.log(5.0), 1.5)
    return height, area, aspect


class ProposalImageDataset(Dataset):
    def __init__(self, jsonl: Path, image_size: int):
        self.rows = read_jsonl(jsonl)
        self.image_size = image_size
        if not self.rows:
            raise ValueError(f"No rows in {jsonl}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[idx]
        size = self.image_size
        mask = load_mask(Path(row["footprint_mask_npy"]), size)
        height, area, aspect = normalized_features(row)
        class_maps = torch.zeros((CLASS_COUNT, size, size), dtype=torch.float32)
        class_maps[int(row.get("class_id", 0)) % CLASS_COUNT].fill_(1.0)
        cond = torch.cat([
            mask,
            scalar_map(height, size),
            scalar_map(area, size),
            scalar_map(aspect, size),
            class_maps,
        ], dim=0)
        target = load_image(Path(row["target_image_png"]), size)
        return {
            "cond": cond,
            "target": target,
            "osm_id": row["osm_id"],
        }


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProposalUNet(nn.Module):
    def __init__(self, in_ch: int = 9, base: int = 32):
        super().__init__()
        self.e1 = ConvBlock(in_ch, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.mid = ConvBlock(base * 4, base * 4)
        self.d2 = ConvBlock(base * 6, base * 2)
        self.d1 = ConvBlock(base * 3, base)
        self.out = nn.Sequential(
            nn.Conv2d(base, base, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base, 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(F.avg_pool2d(e1, 2))
        e3 = self.e3(F.avg_pool2d(e2, 2))
        mid = self.mid(e3)
        u2 = F.interpolate(mid, scale_factor=2, mode="bilinear", align_corners=False)
        d2 = self.d2(torch.cat([u2, e2], dim=1))
        u1 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False)
        d1 = self.d1(torch.cat([u1, e1], dim=1))
        return self.out(d1)


def image_grad_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    targ_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    targ_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, targ_dx) + F.l1_loss(pred_dy, targ_dy)


def ssim_loss(pred: torch.Tensor, target: torch.Tensor, window: int = 7) -> torch.Tensor:
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    pad = window // 2
    mu_x = F.avg_pool2d(pred, window, stride=1, padding=pad)
    mu_y = F.avg_pool2d(target, window, stride=1, padding=pad)
    sigma_x = F.avg_pool2d(pred * pred, window, stride=1, padding=pad) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, window, stride=1, padding=pad) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, window, stride=1, padding=pad) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2) + 1e-8
    )
    return (1.0 - ssim.clamp(-1, 1)).mean()


def laplacian_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=pred.device,
        dtype=pred.dtype,
    ).view(1, 1, 3, 3)
    kernel = kernel.repeat(pred.shape[1], 1, 1, 1)
    pred_lap = F.conv2d(pred, kernel, padding=1, groups=pred.shape[1])
    targ_lap = F.conv2d(target, kernel, padding=1, groups=target.shape[1])
    return F.l1_loss(pred_lap, targ_lap)


def save_sheet(path: Path, cond: torch.Tensor, pred: torch.Tensor, target: torch.Tensor, max_rows: int = 8) -> None:
    cond = cond.detach().cpu()
    pred = pred.detach().cpu().clamp(0, 1)
    target = target.detach().cpu().clamp(0, 1)
    n = min(max_rows, pred.shape[0])
    size = pred.shape[-1]
    sheet = Image.new("RGB", (3 * size, n * size), "white")
    for i in range(n):
        mask = cond[i, 0].numpy()
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), "L").convert("RGB")
        pred_img = Image.fromarray((pred[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8), "RGB")
        targ_img = Image.fromarray((target[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8), "RGB")
        sheet.paste(mask_img, (0, i * size))
        sheet.paste(pred_img, (size, i * size))
        sheet.paste(targ_img, (2 * size, i * size))
    sheet.save(path, optimize=True)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, out_dir: Path, epoch: int) -> dict[str, float]:
    model.eval()
    total_l1 = 0.0
    total_grad = 0.0
    total_ssim = 0.0
    total_lap = 0.0
    count = 0
    first_batch = None
    for batch in loader:
        cond = batch["cond"].to(device)
        target = batch["target"].to(device)
        pred = model(cond)
        l1 = F.l1_loss(pred, target, reduction="sum").item()
        grad = image_grad_l1(pred, target).item() * cond.shape[0]
        ssim = ssim_loss(pred, target).item() * cond.shape[0]
        lap = laplacian_loss(pred, target).item() * cond.shape[0]
        total_l1 += l1
        total_grad += grad
        total_ssim += ssim
        total_lap += lap
        count += int(np.prod(target.shape))
        if first_batch is None:
            first_batch = (cond, pred, target)
    if first_batch is not None:
        save_sheet(out_dir / f"val_preview_epoch_{epoch:03d}.png", *first_batch)
    return {
        "val_l1_pixel": total_l1 / max(count, 1),
        "val_grad_l1": total_grad / max(len(loader.dataset), 1),
        "val_ssim_loss": total_ssim / max(len(loader.dataset), 1),
        "val_lap_l1": total_lap / max(len(loader.dataset), 1),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    train_ds = ProposalImageDataset(Path(args.train_jsonl), args.image_size)
    val_ds = ProposalImageDataset(Path(args.val_jsonl), args.image_size)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = ProposalUNet(base=args.base_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    metrics_rows = []
    best = float("inf")
    start_epoch = 1
    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best = float(ckpt.get("metrics", {}).get("val_l1_pixel", best))
        print(
            f"[proposal-train] resumed {args.resume_ckpt} "
            f"from epoch {start_epoch - 1} best={best:.6f}",
            flush=True,
        )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for batch in train_dl:
            cond = batch["cond"].to(device)
            target = batch["target"].to(device)
            pred = model(cond)
            l1 = F.l1_loss(pred, target)
            grad = image_grad_l1(pred, target)
            ssim = ssim_loss(pred, target) if args.w_ssim else pred.new_tensor(0.0)
            lap = laplacian_loss(pred, target) if args.w_lap else pred.new_tensor(0.0)
            loss = args.w_l1 * l1 + args.w_grad * grad + args.w_ssim * ssim + args.w_lap * lap
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss.item()) * cond.shape[0]
            seen += cond.shape[0]

        val = evaluate(model, val_dl, device, out_dir, epoch)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(seen, 1),
            **val,
        }
        metrics_rows.append(row)
        print(json.dumps(row), flush=True)
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "args": vars(args),
            "metrics": row,
            "condition_channels": ["footprint", "height", "area", "aspect", "class_0", "class_1", "class_2", "class_3", "class_4"],
        }
        torch.save(ckpt, out_dir / "ckpt_latest.pth")
        if row["val_l1_pixel"] < best:
            best = row["val_l1_pixel"]
            torch.save(ckpt, out_dir / "ckpt_best.pth")

    metrics_path = out_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)

    summary = {
        "train_count": len(train_ds),
        "val_count": len(val_ds),
        "start_epoch": start_epoch,
        "last_epoch": args.epochs,
        "best_val_l1_pixel": best,
        "ckpt_best": str(out_dir / "ckpt_best.pth"),
        "metrics_csv": str(metrics_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
