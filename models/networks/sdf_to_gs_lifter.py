"""SDF -> per-voxel Gaussian slots lifter network for Stage 3b.

Input:
    sdf:        (B, 1, 64, 64, 64)  Frame-N SDF (from Stage 3a at inference, or
                                    GT during training).
    fp3d:       (B, 1, 32, 32, 32)  spatial footprint condition (already broadcast).
    ctx:        (B, ctx_dim)        global conditioning (class/style/height embeddings).

Output:
    slots:      (B, 32, 32, 32, K=8, 14) float   per-cell Gaussian attribute slots.
    occ_logits: (B, K=8, 32, 32, 32)      float  logits over each slot's occupancy.

Per-slot attribute layout (14 floats per slot), matching scripts/voxelize_gsplats.py:
    [0:3]   raw mean offset within cell (Frame-N units)
    [3:6]   raw log-scales (pre-exp)
    [6:10]  raw quaternion (un-normalized, w,x,y,z)
    [10]    raw opacity (pre-sigmoid)
    [11:14] SH degree-0 RGB coefficient (raw, no SH_C0 scaling applied)

Architecture: a small 3D UNet operating at 32^3 resolution. The SDF is
downsampled 64 -> 32 via avg-pooling at the input. Class/style/height
conditioning are injected as channel-wise FiLM (gamma/beta from a linear
projection of `ctx`) at each level of the encoder.

This is patterned on models/networks/sdf_residual_net.py:SDFResidualUNet —
same skeleton but with more output channels and explicit FiLM modulation.
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv3d(c_in: int, c_out: int, k: int = 3, s: int = 1, p: int = 1) -> nn.Conv3d:
    return nn.Conv3d(c_in, c_out, kernel_size=k, stride=s, padding=p)


class FiLMBlock3D(nn.Module):
    """Two conv3d residual block with channel-wise (gamma, beta) modulation.

    gamma, beta are produced from `ctx` via a single Linear and broadcast over
    spatial dims. Group-norm before modulation is standard for diffusion-style
    nets but we keep it lightweight here since the lifter is feed-forward.
    """
    def __init__(self, channels: int, ctx_dim: int, groups: int = 8):
        super().__init__()
        self.conv1 = _conv3d(channels, channels)
        self.conv2 = _conv3d(channels, channels)
        self.gn1 = nn.GroupNorm(groups, channels)
        self.gn2 = nn.GroupNorm(groups, channels)
        self.film = nn.Linear(ctx_dim, channels * 2)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        # FiLM gamma/beta from global context.
        gb = self.film(ctx)
        gamma, beta = gb.chunk(2, dim=1)
        gamma = gamma.view(gamma.size(0), -1, 1, 1, 1)
        beta = beta.view(beta.size(0), -1, 1, 1, 1)
        h = self.gn1(x)
        h = h * (1 + gamma) + beta
        h = F.silu(h)
        h = self.conv1(h)
        h = self.gn2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return x + h


class SDFToGSLifter(nn.Module):
    """3D UNet: (SDF, fp3d, ctx) -> Gaussian slot tensor + per-slot occupancy.

    Args:
        in_channels:    number of input volume channels = 2 (sdf concat fp3d).
        base_channels:  width of the network at the input level (32).
        ctx_dim:        size of the global conditioning vector.
        k_slots:        per-cell Gaussian slot count (must match the
                        voxelization in scripts/voxelize_gsplats.py).
        attr_dim:       per-slot attribute dimension (14).
        grid_res:       output spatial resolution (32; matches voxelize grid).
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 32,
        ctx_dim: int = 256,
        k_slots: int = 8,
        attr_dim: int = 14,
        grid_res: int = 32,
    ):
        super().__init__()
        self.k_slots = int(k_slots)
        self.attr_dim = int(attr_dim)
        self.grid_res = int(grid_res)

        c0 = base_channels
        c1 = base_channels * 2
        c2 = base_channels * 4

        # Stem: avg-pool 64 -> 32 happens upstream of this module
        # (in Stage3bModel.forward); we receive a 32^3 input.
        self.stem = _conv3d(in_channels, c0)

        # Encoder.
        self.enc0 = FiLMBlock3D(c0, ctx_dim)
        self.down0 = _conv3d(c0, c1, s=2)  # 32 -> 16
        self.enc1 = FiLMBlock3D(c1, ctx_dim)
        self.down1 = _conv3d(c1, c2, s=2)  # 16 -> 8

        # Bottleneck.
        self.bottleneck1 = FiLMBlock3D(c2, ctx_dim)
        self.bottleneck2 = FiLMBlock3D(c2, ctx_dim)

        # Decoder.
        self.up1 = nn.ConvTranspose3d(c2, c1, kernel_size=4, stride=2, padding=1)  # 8 -> 16
        self.dec1 = FiLMBlock3D(c1, ctx_dim)
        self.up0 = nn.ConvTranspose3d(c1, c0, kernel_size=4, stride=2, padding=1)  # 16 -> 32
        self.dec0 = FiLMBlock3D(c0, ctx_dim)

        # Heads.
        # slot head: (K*14) channels per cell. We split into mean/scale/quat/opac/sh
        # at training time (loss-wise) but the network just emits the flat tensor.
        self.slot_head = _conv3d(c0, self.k_slots * self.attr_dim, k=1, p=0)
        # occupancy head: K channels (per-slot probability).
        self.occ_head = _conv3d(c0, self.k_slots, k=1, p=0)

    def forward(self, vol: torch.Tensor, ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            vol: (B, in_channels, 32, 32, 32)   downsampled SDF concat fp3d.
            ctx: (B, ctx_dim)                   global conditioning vector.

        Returns:
            slots:      (B, 32, 32, 32, K, 14)  float, raw (no activation).
            occ_logits: (B, K, 32, 32, 32)      float, logits (apply sigmoid for prob).
        """
        h0 = self.stem(vol)               # (B, c0, 32, 32, 32)
        h0 = self.enc0(h0, ctx)
        h1 = self.down0(h0)               # (B, c1, 16, 16, 16)
        h1 = self.enc1(h1, ctx)
        h2 = self.down1(h1)               # (B, c2, 8, 8, 8)
        h2 = self.bottleneck1(h2, ctx)
        h2 = self.bottleneck2(h2, ctx)

        u1 = self.up1(h2) + h1            # (B, c1, 16, 16, 16)  skip
        u1 = self.dec1(u1, ctx)
        u0 = self.up0(u1) + h0            # (B, c0, 32, 32, 32)  skip
        u0 = self.dec0(u0, ctx)

        slot_flat = self.slot_head(u0)    # (B, K*14, 32, 32, 32)
        occ_logits = self.occ_head(u0)    # (B, K, 32, 32, 32)

        # Reshape slot tensor to (B, 32, 32, 32, K, 14) for loss / decoding.
        B = slot_flat.shape[0]
        D = H = W = self.grid_res
        slots = slot_flat.reshape(B, self.k_slots, self.attr_dim, D, H, W)
        slots = slots.permute(0, 3, 4, 5, 1, 2).contiguous()  # (B, D, H, W, K, A)
        return slots, occ_logits


def unpack_slots_to_gaussians(
    slots: torch.Tensor,           # (B, 32, 32, 32, K, 14)
    occ_logits: torch.Tensor,      # (B, K, 32, 32, 32)
    bbox: torch.Tensor,            # (B, 2, 3)  (lo, hi) Frame-N bbox per asset
    occ_threshold: float = 0.5,
) -> list[dict]:
    """Convert lifter output to a list of per-batch GaussianSet attribute dicts.

    Returns a list of dicts (one per batch sample) with keys matching
    scene.gsplat_common.GaussianSet, in raw (pre-activation) form:
        means, raw_scales, raw_quats, raw_opac, sh_dc

    Used at inference time from scripts/stage3_generate.py.
    """
    B, D, H, W, K, A = slots.shape
    out_list = []
    for b in range(B):
        # Per-slot occupancy probabilities -> bool mask.
        occ_b = torch.sigmoid(occ_logits[b])           # (K, D, H, W)
        # Reorder to (D, H, W, K) to align with slot layout.
        occ_b = occ_b.permute(1, 2, 3, 0)              # (D, H, W, K)
        keep = occ_b > occ_threshold                   # (D, H, W, K) bool

        if keep.sum() == 0:
            # Degenerate: no occupied slots predicted. Emit a single null Gaussian
            # at the bbox center to avoid an empty PLY.
            lo, hi = bbox[b, 0], bbox[b, 1]
            center = (lo + hi) / 2
            out_list.append({
                "means":      center.unsqueeze(0),
                "raw_scales": torch.full((1, 3), -6.0, device=slots.device),
                "raw_quats":  torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=slots.device),
                "raw_opac":   torch.full((1,), -10.0, device=slots.device),
                "sh_dc":      torch.zeros(1, 3, device=slots.device),
            })
            continue

        # Cell coordinates of kept slots.
        d, h, w, k = torch.where(keep)
        # Per-slot attribute tensor for kept positions.
        attrs = slots[b, d, h, w, k]                   # (N_kept, 14)

        # Map cell index + in-cell offset to absolute Frame-N positions.
        lo, hi = bbox[b, 0], bbox[b, 1]                 # (3,) each
        extent = (hi - lo).clamp_min(1e-6)
        cell_size = extent / float(D)
        cells = torch.stack([d, h, w], dim=-1).float() # (N_kept, 3)
        cell_centers = lo + (cells + 0.5) * cell_size  # (N_kept, 3)

        mean_offset = attrs[:, 0:3]
        means = cell_centers + mean_offset             # (N_kept, 3)

        out_list.append({
            "means":      means,
            "raw_scales": attrs[:, 3:6],
            "raw_quats":  attrs[:, 6:10],
            "raw_opac":   attrs[:, 10],
            "sh_dc":      attrs[:, 11:14],
        })

    return out_list
