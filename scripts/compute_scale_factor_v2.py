"""Thin wrapper around scripts/compute_scale_factor.py that targets the VQVAE-v2
ckpt and (optionally) patches `configs/stage3a_sdf_diffusion.yaml:scale_factor`
in-place. Run this once the v2 retrain finishes, before kicking off Stage 3a.

The underlying logic (encode the training split with the VQVAE, take std of all
latent values, scale_factor = 1 / std) is identical to the v1 script; we only
add (a) v2-specific defaults and (b) the YAML write-back step.

Usage:

    env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \\
        scripts/compute_scale_factor_v2.py \\
        --vq_ckpt logs_building/<v2-run>/ckpt/vqvae_steps-latest.pth \\
        --write_yaml

`--write_yaml` patches the `model.params.scale_factor` field in the Stage 3a
config (default: configs/stage3a_sdf_diffusion.yaml). Without it, the script
just prints the recommended value.
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Reuse the v1 logic.
from scripts.compute_scale_factor import (
    SDFOnlyDataset, build_vqvae,
)


def _patch_yaml_scale_factor(yaml_path: Path, value: float) -> None:
    """OmegaConf-roundtrip the YAML, write back with the new scale_factor.

    OmegaConf doesn't preserve comments — for a config we only just generated,
    that's fine. If you care about preserving comments, edit the file by hand.
    """
    if not yaml_path.exists():
        raise SystemExit(f"YAML not found: {yaml_path}")
    cfg = OmegaConf.load(yaml_path)
    old = float(cfg.model.params.scale_factor) if "scale_factor" in cfg.model.params else None
    cfg.model.params.scale_factor = float(value)
    with open(yaml_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    print(f"[patch] {yaml_path}: scale_factor {old} -> {value:.6f}")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vq_cfg", default="configs/vqvae_bnet_v2.yaml")
    ap.add_argument("--vq_ckpt", required=True)
    ap.add_argument("--dataroot", default="data")
    ap.add_argument("--split", default="train")
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--trunc_thres", type=float, default=0.3,
                    help="Match VQVAE-v2 training: 0.3 (v1 used 0.2).")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_batches", type=int, default=-1,
                    help="-1 = encode the whole split (recommended for the v2 measurement).")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--write_yaml", action="store_true",
                    help="Patch configs/stage3a_sdf_diffusion.yaml's model.params.scale_factor "
                         "with 1/std.")
    ap.add_argument("--target_yaml", default="configs/stage3a_sdf_diffusion.yaml")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"[*] device={args.device}")
    print(f"[*] loading VQVAE from {args.vq_ckpt}")
    vqvae = build_vqvae(args.vq_cfg, args.vq_ckpt, args.device)

    ds = SDFOnlyDataset(args.dataroot, split=args.split,
                        res=args.res, trunc_thres=args.trunc_thres)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )
    print(f"[*] dataset size = {len(ds)}")

    # Welford-style running stats (identical to scripts/compute_scale_factor.py).
    n = 0
    mean = 0.0
    M2 = 0.0
    s_min, s_max = float("inf"), float("-inf")

    n_done = 0
    for batch in dl:
        batch = batch.to(args.device, non_blocking=True)
        z = vqvae(batch, forward_no_quant=True, encode_only=True)
        z_flat = z.detach().float().reshape(-1)
        b = z_flat.numel()
        b_mean = z_flat.mean().item()
        b_var = z_flat.var(unbiased=False).item()
        delta = b_mean - mean
        new_n = n + b
        mean += delta * b / new_n
        M2 += b_var * b + delta * delta * n * b / new_n
        n = new_n
        s_min = min(s_min, z_flat.min().item())
        s_max = max(s_max, z_flat.max().item())
        n_done += 1
        if n_done % 16 == 0 or n_done == 1:
            cur_std = (M2 / max(n - 1, 1)) ** 0.5
            print(f"  batch {n_done}: latents seen={n:,}  running mean={mean:+.4f}  std={cur_std:.4f}")
        if args.num_batches > 0 and n_done >= args.num_batches:
            break

    std = (M2 / max(n - 1, 1)) ** 0.5
    scale_factor = 1.0 / std if std > 0 else float("nan")
    print()
    print("=" * 60)
    print(f"  latents seen     : {n:,}")
    print(f"  latent mean      : {mean:+.6f}")
    print(f"  latent std       : {std:.6f}")
    print(f"  latent min/max   : {s_min:+.4f} / {s_max:+.4f}")
    print(f"  recommended      : scale_factor = {scale_factor:.6f}")
    print("=" * 60)

    if args.write_yaml:
        _patch_yaml_scale_factor(Path(args.target_yaml), scale_factor)
    else:
        print(f"To patch the config now, rerun with --write_yaml "
              f"(target: {args.target_yaml})")


if __name__ == "__main__":
    main()
