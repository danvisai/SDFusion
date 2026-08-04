"""
Estimate the latent rescaling factor for SDFusion's latent diffusion stage.

Stable-Diffusion-style LDMs train on `z = encode(x) * scale_factor` so that
the diffusion noise schedule (calibrated for unit-variance signal) matches
the latent's actual variance. The default value 0.18215 in the YAML was
derived for SD's 8x VAE on natural images and is wrong for this 3D VQVAE.

This script:
  1. loads the pretrained VQVAE from configs/vqvae_bnet.yaml + a checkpoint,
  2. encodes a batch (or many batches) of training SDFs,
  3. computes std over all latent values,
  4. prints the scale_factor = 1 / std that you should write into
     configs/sdfusion-img2shape.yaml under model.params.scale_factor.

Run from repo root:
  python scripts/compute_scale_factor.py \
      --vq_cfg  configs/vqvae_bnet.yaml \
      --vq_ckpt logs_building/2025-05-19T19-58-28-vqvae-building-all-res64-LR1e-4-T0.2-release/ckpt/vqvae_steps-latest.pth \
      --dataroot data \
      --num_batches 32 --batch_size 8 --trunc_thres 0.2
"""
import argparse
import os
import sys

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from models.networks.vqvae_networks.network import VQVAE


class SDFOnlyDataset(Dataset):
    """Minimal training-set iterator that yields only the SDF tensor.

    Mirrors how BuildingNetDataset reads from data/BuildingNet_dataset_v0_1/.
    """

    def __init__(self, dataroot, split="train", res=64, trunc_thres=0.2):
        self.res = res
        self.trunc_thres = trunc_thres
        split_file = os.path.join(
            dataroot, "BuildingNet_dataset_v0_1", "splits", f"{split}_split.txt"
        )
        sdf_dir = os.path.join(dataroot, "BuildingNet_dataset_v0_1", f"resolution_{res}")
        with open(split_file) as f:
            ids = [l.strip() for l in f if l.strip()]
        self.paths = [os.path.join(sdf_dir, i, "ori_sample_grid.h5") for i in ids]
        self.paths = [p for p in self.paths if os.path.exists(p)]
        if not self.paths:
            raise FileNotFoundError(
                f"No SDF h5 files found under {sdf_dir} for split={split}"
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with h5py.File(self.paths[idx], "r") as h5_f:
            sdf = h5_f["pc_sdf_sample"][:].astype(np.float32)
        sdf = torch.from_numpy(sdf).view(1, self.res, self.res, self.res)
        if self.trunc_thres != 0.0:
            sdf = torch.clamp(sdf, -self.trunc_thres, self.trunc_thres)
        return sdf


def build_vqvae(vq_cfg_path, vq_ckpt, device):
    vq_conf = OmegaConf.load(vq_cfg_path)
    mp = vq_conf.model.params
    vqvae = VQVAE(mp.ddconfig, mp.n_embed, mp.embed_dim)
    state = torch.load(vq_ckpt, map_location="cpu")
    vqvae.load_state_dict(state["vqvae"] if "vqvae" in state else state)
    vqvae.to(device).eval()
    for p in vqvae.parameters():
        p.requires_grad_(False)
    return vqvae


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vq_cfg", default="configs/vqvae_bnet.yaml")
    ap.add_argument("--vq_ckpt", required=True)
    ap.add_argument("--dataroot", default="data")
    ap.add_argument("--split", default="train")
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--trunc_thres", type=float, default=0.2)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_batches", type=int, default=32,
                    help="set to -1 to encode the whole split")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"[*] device = {args.device}")
    print(f"[*] loading VQVAE from {args.vq_ckpt}")
    vqvae = build_vqvae(args.vq_cfg, args.vq_ckpt, args.device)

    ds = SDFOnlyDataset(args.dataroot, split=args.split,
                        res=args.res, trunc_thres=args.trunc_thres)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers)
    print(f"[*] dataset size = {len(ds)}")

    # Welford-style running stats over all latent values across all batches.
    n = 0
    mean = 0.0
    M2 = 0.0
    sample_min, sample_max = float("inf"), float("-inf")

    n_done = 0
    for batch in dl:
        batch = batch.to(args.device, non_blocking=True)
        # encode_only=True returns the pre-quant latent (the same input the
        # diffusion model is trained against in forward()).
        z = vqvae(batch, forward_no_quant=True, encode_only=True)
        z_flat = z.detach().float().reshape(-1)

        # Update running mean/var using Chan's parallel algorithm.
        b = z_flat.numel()
        b_mean = z_flat.mean().item()
        b_var = z_flat.var(unbiased=False).item()
        delta = b_mean - mean
        new_n = n + b
        mean += delta * b / new_n
        M2 += b_var * b + delta * delta * n * b / new_n
        n = new_n
        sample_min = min(sample_min, z_flat.min().item())
        sample_max = max(sample_max, z_flat.max().item())

        n_done += 1
        if n_done % 8 == 0 or n_done == 1:
            cur_std = (M2 / max(n - 1, 1)) ** 0.5
            print(f"  batch {n_done} | latents seen={n:,} "
                  f"running mean={mean:+.4f} std={cur_std:.4f}")
        if args.num_batches > 0 and n_done >= args.num_batches:
            break

    std = (M2 / max(n - 1, 1)) ** 0.5
    scale_factor = 1.0 / std if std > 0 else float("nan")

    print()
    print("=" * 60)
    print(f"  latents seen     : {n:,}")
    print(f"  latent mean      : {mean:+.6f}")
    print(f"  latent std       : {std:.6f}")
    print(f"  latent min/max   : {sample_min:+.4f} / {sample_max:+.4f}")
    print(f"  recommended      : scale_factor = {scale_factor:.6f}")
    print("=" * 60)
    print("Set this in configs/sdfusion-img2shape.yaml under model.params.scale_factor")


if __name__ == "__main__":
    main()
