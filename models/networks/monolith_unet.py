"""Ticket 11: a self-contained conditional 3D UNet for the monolith baseline.

C2's monolith is "one SDF net trained from scratch on real (coarse-massing -> BuildingNet-
detail) pairs" (CONTEXT.md). It needs to be a genuine *generator* (evaluated distributionally
via facade FID, CONTEXT.md "Detail fidelity"), not a deterministic regressor that would just
blur toward the training-set mean and hand C2 an easy, unfair win -- so this is a noise-
prediction UNet for a coarse-SDF-conditioned Gaussian diffusion process (`monolith_diffusion.py`
owns the diffusion math; this module is architecture only).

Deliberately NOT built on `models/stage3a_model.py`/`models/base_model.py`'s `create_model`
production dispatch: that framework is sized and wired for the DEPLOYED prior (947M params,
VQVAE latent space, footprint/class/style/era conditioning, EMA, autoguidance, warm-start
checkpoint surgery) serving the live sculptor. The monolith is a research baseline compared
once per experiment run, at ADR 0004's shared 96^3 *raw voxel* resolution (not the 64^3 VQVAE
latent grid, which would need a new 96^3 VQVAE trained first -- out of this ticket's scope and
unnecessary: operating on raw SDF grids trivially satisfies "every arm at the same shared
resolution", ADR 0004's own constraint). Conditioning is channel-concatenation of the coarse
SDF (the SR3/Palette conditional-diffusion pattern), timestep conditioning is FiLM
(scale/shift) after GroupNorm, matching the architecture family this codebase already uses
elsewhere for lighter 3D nets (`models/networks/sdf_residual_net.py`'s ConvBlock3d: same
avg_pool3d-down / trilinear-up / skip-concat backbone, extended here with a timestep input).
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard transformer/DDPM sinusoidal timestep embedding, (N,) -> (N, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=timesteps.device) / max(half, 1))
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:  # odd width: pad one zero column rather than silently truncating a frequency
        emb = F.pad(emb, (0, 1))
    return emb


class TimestepMLP(nn.Module):
    """Embeds a timestep to a `dim*4`-wide conditioning vector every FiLM block reads."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim * 4))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(sinusoidal_embedding(t, self.dim))


class FiLMConvBlock3d(nn.Module):
    """Residual 3D conv block, timestep-conditioned via FiLM (scale/shift) after the first
    norm -- the standard DDPM UNet conditioning mechanism."""

    def __init__(self, in_ch: int, out_ch: int, temb_dim: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.film = nn.Linear(temb_dim, out_ch * 2)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.SiLU()
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        scale, shift = self.film(temb).chunk(2, dim=-1)
        h = h * (1 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class MonolithUNet(nn.Module):
    """Predicts the noise added to a target SDF, conditioned on a coarse SDF (channel-concat)
    and a diffusion timestep (FiLM). Input/output: `(B,1,R,R,R)` at any `R` divisible by
    `2**len(channel_mults)`."""

    def __init__(self, base_channels: int = 32, channel_mults=(1, 2, 4), temb_dim: int = 64):
        super().__init__()
        temb_out = temb_dim * 4
        self.temb_mlp = TimestepMLP(temb_dim)
        self.in_conv = nn.Conv3d(2, base_channels, 3, padding=1)  # noisy target + coarse

        self.downs = nn.ModuleList()
        cur = base_channels
        for mult in channel_mults:
            out_c = base_channels * mult
            self.downs.append(FiLMConvBlock3d(cur, out_c, temb_out))
            cur = out_c
        self.mid = FiLMConvBlock3d(cur, cur, temb_out)
        self.ups = nn.ModuleList()
        for mult in reversed(channel_mults):
            out_c = base_channels * mult
            self.ups.append(FiLMConvBlock3d(cur + out_c, out_c, temb_out))
            cur = out_c

        self.out_norm = nn.GroupNorm(min(8, cur), cur)
        self.out_conv = nn.Conv3d(cur, 1, 3, padding=1)
        # zero-init the output layer (standard DDPM trick): an untrained network starts by
        # predicting exactly zero noise, so early training doesn't inject a random directional
        # bias before the model has seen any data.
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, noisy_target: torch.Tensor, coarse: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        temb = self.temb_mlp(t)
        h = self.in_conv(torch.cat([noisy_target, coarse], dim=1))
        skips = []
        for down in self.downs:
            h = down(h, temb)
            skips.append(h)
            h = F.avg_pool3d(h, 2)
        h = self.mid(h, temb)
        for up in self.ups:
            h = F.interpolate(h, scale_factor=2, mode="trilinear", align_corners=False)
            h = up(torch.cat([h, skips.pop()], dim=1), temb)
        return self.out_conv(F.silu(self.out_norm(h)))
