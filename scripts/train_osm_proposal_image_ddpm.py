"""Train a small conditional DDPM for OSM proposal images.

This is the first diffusion version of:

    footprint mask + class + height/context -> Hunyuan conditioning image

It is intentionally compact so it can be trained on the current tiny corpus and
used as a baseline before moving to latent diffusion.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.train_osm_proposal_image_generator import ProposalImageDataset


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--val_jsonl", required=True)
    ap.add_argument("--out_dir", default="outputs/osm_proposal_image_ddpm_v1")
    ap.add_argument("--image_size", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base_channels", type=int, default=48)
    ap.add_argument("--timesteps", type=int, default=200)
    ap.add_argument("--sample_steps", type=int, default=50)
    ap.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddim")
    ap.add_argument("--sample_every", type=int, default=10)
    ap.add_argument("--recon_every", type=int, default=0)
    ap.add_argument("--recon_timesteps", default="25,50,100,150")
    ap.add_argument("--resume_ckpt", default="")
    ap.add_argument("--max_train_examples", type=int, default=0)
    ap.add_argument("--max_val_examples", type=int, default=0)
    ap.add_argument("--ema_decay", type=float, default=0.995)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260511)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def trim_dataset(ds: ProposalImageDataset, max_examples: int) -> ProposalImageDataset:
    if max_examples > 0:
        ds.rows = ds.rows[:max_examples]
    return ds


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_ch: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time = nn.Sequential(nn.SiLU(), nn.Linear(time_ch, out_ch))
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class ConditionalDDPMUNet(nn.Module):
    def __init__(self, cond_ch: int = 9, base: int = 48, time_ch: int = 192):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_ch, time_ch),
            nn.SiLU(),
            nn.Linear(time_ch, time_ch),
        )
        in_ch = 3 + cond_ch
        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)
        self.e1 = ResBlock(base, base, time_ch)
        self.e2 = ResBlock(base, base * 2, time_ch)
        self.e3 = ResBlock(base * 2, base * 4, time_ch)
        self.mid1 = ResBlock(base * 4, base * 4, time_ch)
        self.mid2 = ResBlock(base * 4, base * 4, time_ch)
        self.d2 = ResBlock(base * 6, base * 2, time_ch)
        self.d1 = ResBlock(base * 3, base, time_ch)
        self.out = nn.Sequential(
            nn.GroupNorm(8, base),
            nn.SiLU(),
            nn.Conv2d(base, 3, 3, padding=1),
        )
        self.time_ch = time_ch

    def forward(self, x: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(sinusoidal_embedding(t, self.time_ch))
        x = torch.cat([x, cond], dim=1)
        x0 = self.in_conv(x)
        e1 = self.e1(x0, t_emb)
        e2 = self.e2(F.avg_pool2d(e1, 2), t_emb)
        e3 = self.e3(F.avg_pool2d(e2, 2), t_emb)
        mid = self.mid2(self.mid1(e3, t_emb), t_emb)
        u2 = F.interpolate(mid, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.d2(torch.cat([u2, e2], dim=1), t_emb)
        u1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.d1(torch.cat([u1, e1], dim=1), t_emb)
        return self.out(d1)


class DiffusionSchedule:
    def __init__(self, timesteps: int, device: torch.device):
        beta = torch.linspace(1e-4, 0.02, timesteps, device=device)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.timesteps = timesteps
        self.beta = beta
        self.alpha = alpha
        self.alpha_bar = alpha_bar
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)
        self.sqrt_recip_alpha = torch.sqrt(1.0 / alpha)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        a = self.sqrt_alpha_bar[t][:, None, None, None]
        b = self.sqrt_one_minus_alpha_bar[t][:, None, None, None]
        return a * x0 + b * noise


def to_model_space(img: torch.Tensor) -> torch.Tensor:
    return img * 2.0 - 1.0


def from_model_space(img: torch.Tensor) -> torch.Tensor:
    return ((img + 1.0) * 0.5).clamp(0, 1)


@torch.no_grad()
def sample_ddpm(
    model: nn.Module,
    cond: torch.Tensor,
    schedule: DiffusionSchedule,
    sample_steps: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    b, _, h, w = cond.shape
    x = torch.randn((b, 3, h, w), device=device)
    indices = torch.linspace(schedule.timesteps - 1, 0, sample_steps, device=device).long().unique_consecutive()
    for t_value in indices:
        t = torch.full((b,), int(t_value.item()), device=device, dtype=torch.long)
        eps = model(x, cond, t)
        beta_t = schedule.beta[t][:, None, None, None]
        sqrt_one_minus = schedule.sqrt_one_minus_alpha_bar[t][:, None, None, None]
        sqrt_recip_alpha = schedule.sqrt_recip_alpha[t][:, None, None, None]
        mean = sqrt_recip_alpha * (x - beta_t * eps / sqrt_one_minus)
        if int(t_value.item()) > 0:
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(beta_t) * noise
        else:
            x = mean
    return from_model_space(x)


def timestep_schedule(max_t: int, sample_steps: int, device: torch.device) -> torch.Tensor:
    steps = min(sample_steps, max_t + 1)
    return torch.linspace(max_t, 0, steps, device=device).long().unique_consecutive()


def predict_x0_from_eps(x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor, schedule: DiffusionSchedule) -> torch.Tensor:
    sqrt_alpha_bar = schedule.sqrt_alpha_bar[t][:, None, None, None]
    sqrt_one_minus = schedule.sqrt_one_minus_alpha_bar[t][:, None, None, None]
    return (x_t - sqrt_one_minus * eps) / sqrt_alpha_bar


@torch.no_grad()
def sample_ddim(
    model: nn.Module,
    cond: torch.Tensor,
    schedule: DiffusionSchedule,
    sample_steps: int,
    device: torch.device,
    start_x: torch.Tensor | None = None,
    start_t: int | None = None,
) -> torch.Tensor:
    model.eval()
    b, _, h, w = cond.shape
    if start_x is None:
        x = torch.randn((b, 3, h, w), device=device)
        max_t = schedule.timesteps - 1 if start_t is None else start_t
    else:
        x = start_x
        max_t = schedule.timesteps - 1 if start_t is None else start_t

    indices = timestep_schedule(max_t, sample_steps, device)
    for i, t_value in enumerate(indices):
        t_int = int(t_value.item())
        t = torch.full((b,), t_int, device=device, dtype=torch.long)
        eps = model(x, cond, t)
        pred_x0 = predict_x0_from_eps(x, t, eps, schedule).clamp(-1.0, 1.0)

        if i == len(indices) - 1:
            x = pred_x0
            continue

        prev_t_int = int(indices[i + 1].item())
        alpha_prev = schedule.alpha_bar[prev_t_int]
        x = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1.0 - alpha_prev) * eps
    return from_model_space(x)


@torch.no_grad()
def sample_images(
    model: nn.Module,
    cond: torch.Tensor,
    schedule: DiffusionSchedule,
    sample_steps: int,
    sampler: str,
    device: torch.device,
) -> torch.Tensor:
    if sampler == "ddpm":
        return sample_ddpm(model, cond, schedule, sample_steps, device)
    return sample_ddim(model, cond, schedule, sample_steps, device)


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.mul_(decay).add_(p, alpha=1.0 - decay)
    for ema_b, b in zip(ema_model.buffers(), model.buffers()):
        ema_b.copy_(b)


def save_sample_sheet(path: Path, cond: torch.Tensor, sample: torch.Tensor, target: torch.Tensor, max_rows: int = 8) -> None:
    cond = cond.detach().cpu()
    sample = sample.detach().cpu().clamp(0, 1)
    target = target.detach().cpu().clamp(0, 1)
    n = min(max_rows, sample.shape[0])
    size = sample.shape[-1]
    sheet = Image.new("RGB", (3 * size, n * size), "white")
    for i in range(n):
        mask = cond[i, 0].numpy()
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), "L").convert("RGB")
        sample_img = Image.fromarray((sample[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8), "RGB")
        target_img = Image.fromarray((target[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8), "RGB")
        sheet.paste(mask_img, (0, i * size))
        sheet.paste(sample_img, (size, i * size))
        sheet.paste(target_img, (2 * size, i * size))
    sheet.save(path, optimize=True)


def save_recon_sheet(
    path: Path,
    cond: torch.Tensor,
    target: torch.Tensor,
    noisy: torch.Tensor,
    recon: torch.Tensor,
    max_rows: int = 8,
) -> None:
    cond = cond.detach().cpu()
    target = target.detach().cpu().clamp(0, 1)
    noisy = noisy.detach().cpu().clamp(0, 1)
    recon = recon.detach().cpu().clamp(0, 1)
    n = min(max_rows, target.shape[0])
    size = target.shape[-1]
    sheet = Image.new("RGB", (4 * size, n * size), "white")
    for i in range(n):
        mask = cond[i, 0].numpy()
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), "L").convert("RGB")
        target_img = Image.fromarray((target[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8), "RGB")
        noisy_img = Image.fromarray((noisy[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8), "RGB")
        recon_img = Image.fromarray((recon[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8), "RGB")
        sheet.paste(mask_img, (0, i * size))
        sheet.paste(target_img, (size, i * size))
        sheet.paste(noisy_img, (2 * size, i * size))
        sheet.paste(recon_img, (3 * size, i * size))
    sheet.save(path, optimize=True)


@torch.no_grad()
def save_reconstruction_diagnostics(
    out_dir: Path,
    epoch: int,
    model: nn.Module,
    cond: torch.Tensor,
    target: torch.Tensor,
    schedule: DiffusionSchedule,
    sample_steps: int,
    recon_timesteps: list[int],
    device: torch.device,
) -> None:
    target_model = to_model_space(target.to(device))
    for t_int in recon_timesteps:
        t_int = max(0, min(t_int, schedule.timesteps - 1))
        t = torch.full((target_model.shape[0],), t_int, device=device, dtype=torch.long)
        noise = torch.randn_like(target_model)
        noisy_model = schedule.q_sample(target_model, t, noise)
        recon = sample_ddim(
            model,
            cond,
            schedule,
            min(sample_steps, t_int + 1),
            device,
            start_x=noisy_model,
            start_t=t_int,
        )
        save_recon_sheet(
            out_dir / f"recon_t{t_int:03d}_epoch_{epoch:03d}.png",
            cond,
            target,
            from_model_space(noisy_model),
            recon,
        )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, schedule: DiffusionSchedule, device: torch.device) -> float:
    model.eval()
    total = 0.0
    seen = 0
    for batch in loader:
        cond = batch["cond"].to(device)
        target = to_model_space(batch["target"].to(device))
        t = torch.randint(0, schedule.timesteps, (target.shape[0],), device=device)
        noise = torch.randn_like(target)
        noisy = schedule.q_sample(target, t, noise)
        pred = model(noisy, cond, t)
        loss = F.mse_loss(pred, noise, reduction="sum")
        total += float(loss.item())
        seen += int(np.prod(noise.shape))
    return total / max(seen, 1)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    train_ds = trim_dataset(ProposalImageDataset(Path(args.train_jsonl), args.image_size), args.max_train_examples)
    val_ds = trim_dataset(ProposalImageDataset(Path(args.val_jsonl), args.image_size), args.max_val_examples)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = ConditionalDDPMUNet(base=args.base_channels).to(device)
    ema_model = copy.deepcopy(model).to(device).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    schedule = DiffusionSchedule(args.timesteps, device)
    metrics_rows = []
    best = float("inf")
    start_epoch = 1
    recon_timesteps = [int(x.strip()) for x in args.recon_timesteps.split(",") if x.strip()]

    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "ema_model" in ckpt:
            ema_model.load_state_dict(ckpt["ema_model"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        metrics = ckpt.get("metrics", {})
        if "val_noise_mse" in metrics:
            best = float(metrics["val_noise_mse"])
        print(json.dumps({"resume_ckpt": args.resume_ckpt, "start_epoch": start_epoch, "best": best}), flush=True)

    fixed = next(iter(val_dl))
    fixed_cond = fixed["cond"].to(device)
    fixed_target = fixed["target"].to(device)

    if start_epoch > args.epochs:
        sample = sample_images(ema_model, fixed_cond, schedule, args.sample_steps, args.sampler, device)
        save_sample_sheet(out_dir / f"sample_{args.sampler}_resume_epoch_{start_epoch - 1:03d}.png", fixed_cond, sample, fixed_target)
        if args.recon_every:
            save_reconstruction_diagnostics(
                out_dir,
                start_epoch - 1,
                ema_model,
                fixed_cond,
                fixed_target,
                schedule,
                args.sample_steps,
                recon_timesteps,
                device,
            )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for batch in train_dl:
            cond = batch["cond"].to(device)
            target = to_model_space(batch["target"].to(device))
            t = torch.randint(0, args.timesteps, (target.shape[0],), device=device)
            noise = torch.randn_like(target)
            noisy = schedule.q_sample(target, t, noise)
            pred = model(noisy, cond, t)
            loss = F.mse_loss(pred, noise)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            update_ema(ema_model, model, args.ema_decay)
            loss_sum += float(loss.item()) * target.shape[0]
            seen += target.shape[0]

        val_noise_mse = evaluate(ema_model, val_dl, schedule, device)
        row = {
            "epoch": epoch,
            "train_noise_mse": loss_sum / max(seen, 1),
            "val_noise_mse": val_noise_mse,
            "ema_decay": args.ema_decay,
        }
        metrics_rows.append(row)
        print(json.dumps(row), flush=True)
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "ema_model": ema_model.state_dict(),
            "optimizer": opt.state_dict(),
            "args": vars(args),
            "metrics": row,
            "condition_channels": ["footprint", "height", "area", "aspect", "class_0", "class_1", "class_2", "class_3", "class_4"],
        }
        torch.save(ckpt, out_dir / "ckpt_latest.pth")
        if val_noise_mse < best:
            best = val_noise_mse
            torch.save(ckpt, out_dir / "ckpt_best.pth")
        if epoch % args.sample_every == 0 or epoch == args.epochs:
            sample = sample_images(ema_model, fixed_cond, schedule, args.sample_steps, args.sampler, device)
            save_sample_sheet(out_dir / f"sample_{args.sampler}_epoch_{epoch:03d}.png", fixed_cond, sample, fixed_target)
        if args.recon_every and (epoch % args.recon_every == 0 or epoch == args.epochs):
            save_reconstruction_diagnostics(
                out_dir,
                epoch,
                ema_model,
                fixed_cond,
                fixed_target,
                schedule,
                args.sample_steps,
                recon_timesteps,
                device,
            )

    metrics_path = out_dir / "metrics.csv"
    if metrics_rows:
        with metrics_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
            writer.writeheader()
            writer.writerows(metrics_rows)
    summary = {
        "train_count": len(train_ds),
        "val_count": len(val_ds),
        "best_val_noise_mse": best,
        "ckpt_best": str(out_dir / "ckpt_best.pth"),
        "metrics_csv": str(metrics_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
