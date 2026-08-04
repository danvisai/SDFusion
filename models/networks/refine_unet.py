"""Amortized learned refine — a residual 3D UNet that maps a crude sculpt SDF to a
coherent, detailed building SDF in one forward pass.

This is the generative core of the sculpt-and-refine vision: instead of per-building
optimization (scripts/server/refine.py displacement mode), the UNet has LEARNED, from
(rough -> good) pairs, what a good building looks like, and applies that instantly to
whatever the user sculpts. Residual formulation (out = crude + delta) so it starts near
identity and only has to learn the correction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
            nn.Conv3d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class RefineUNet3D(nn.Module):
    """(B,1,R,R,R) crude SDF -> (B,1,R,R,R) refined SDF. R divisible by 8."""

    def __init__(self, base=24, delta_scale=1.0):
        super().__init__()
        c = base
        self.e1 = ConvBlock(1, c)
        self.e2 = ConvBlock(c, 2 * c)
        self.e3 = ConvBlock(2 * c, 4 * c)
        self.pool = nn.MaxPool3d(2)
        self.bott = ConvBlock(4 * c, 8 * c)
        self.u3 = nn.ConvTranspose3d(8 * c, 4 * c, 2, 2)
        self.d3 = ConvBlock(8 * c, 4 * c)
        self.u2 = nn.ConvTranspose3d(4 * c, 2 * c, 2, 2)
        self.d2 = ConvBlock(4 * c, 2 * c)
        self.u1 = nn.ConvTranspose3d(2 * c, c, 2, 2)
        self.d1 = ConvBlock(2 * c, c)
        self.out = nn.Conv3d(c, 1, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)  # start at identity
        self.delta_scale = delta_scale

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b = self.bott(self.pool(e3))
        d3 = self.d3(torch.cat([self.u3(b), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        delta = torch.tanh(self.out(d1)) * self.delta_scale
        return x + delta


class LatentCorrectorUNet3D(nn.Module):
    """(B,C,16,16,16) wavy VQVAE latent -> (B,C,16,16,16) corrected latent (#59: latent-space
    corrector de-risk). Same residual/zero-init-identity contract as `RefineUNet3D`, but a
    SIBLING rather than a reuse: it operates on the frozen VQVAE's raw 3-channel latent grid
    (already only 16^3) instead of a 64^3 decoded SDF, so it only needs 2 downsamples
    (16->8->4) instead of 3."""

    def __init__(self, channels=3, base=48, delta_scale=1.0):
        super().__init__()
        c = base
        self.e1 = ConvBlock(channels, c)
        self.e2 = ConvBlock(c, 2 * c)
        self.pool = nn.MaxPool3d(2)
        self.bott = ConvBlock(2 * c, 4 * c)
        self.u2 = nn.ConvTranspose3d(4 * c, 2 * c, 2, 2)
        self.d2 = ConvBlock(4 * c, 2 * c)
        self.u1 = nn.ConvTranspose3d(2 * c, c, 2, 2)
        self.d1 = ConvBlock(2 * c, c)
        self.out = nn.Conv3d(c, channels, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)  # start at identity
        self.delta_scale = delta_scale

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        b = self.bott(self.pool(e2))
        d2 = self.d2(torch.cat([self.u2(b), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        delta = torch.tanh(self.out(d1)) * self.delta_scale
        return x + delta


def surface_weighted_l1(pred, target, band=0.1):
    """L1 weighted near the TARGET (good) surface so detail is matched, plus a small
    everywhere term for sign correctness."""
    w = torch.exp(-(target / band) ** 2) + 0.1
    return ((pred - target).abs() * w).sum() / w.sum()


def _central_grad(v):
    """Central-difference spatial gradient of (B,1,D,H,W) -> (B,3,D,H,W) (voxel units; boundary=0).
    Order of the 3 channels (dz,dy,dx) is irrelevant to the magnitude/cosine terms below."""
    dz = torch.zeros_like(v); dy = torch.zeros_like(v); dx = torch.zeros_like(v)
    dz[:, :, 1:-1] = (v[:, :, 2:] - v[:, :, :-2]) * 0.5
    dy[:, :, :, 1:-1] = (v[:, :, :, 2:] - v[:, :, :, :-2]) * 0.5
    dx[:, :, :, :, 1:-1] = (v[:, :, :, :, 2:] - v[:, :, :, :, :-2]) * 0.5
    return torch.cat([dz, dy, dx], dim=1)


def sharpness_loss(pred, target, band=0.1, w_nrm=0.05, w_eik=0.01, eps=1e-6):
    """#54: surface-weighted L1 anchor + SHARPNESS-aware terms that plain L1 (v1/v2) lacks.

    A 'wavy wall' is exactly a surface whose NORMAL wobbles where the crisp target's is constant, so:
      - normal term: 1 - cos(grad(pred), grad(target)), weighted near the target surface -> pushes the
        refined normals onto the crisp target's -> de-ripples walls.
      - eikonal-ish term: |grad(pred)| matched to |grad(target)| near the surface (frame-agnostic: no
        need for the exact voxel spacing; the crisp target defines the correct gradient profile) ->
        keeps a valid, non-collapsed distance field.
    Returns (total_loss, {l1, nrm, eik}) with the components as detached floats for logging."""
    l1 = surface_weighted_l1(pred, target, band)
    w = torch.exp(-(target / band) ** 2)                       # (B,1,D,H,W) near-surface weight
    wsum = w.sum().clamp_min(eps)
    gp, gt = _central_grad(pred), _central_grad(target)
    npn = gp.norm(dim=1, keepdim=True); ntn = gt.norm(dim=1, keepdim=True)
    cos = (gp * gt).sum(dim=1, keepdim=True) / (npn * ntn + eps)
    nrm = (w * (1.0 - cos)).sum() / wsum
    eik = (w * (npn - ntn).abs()).sum() / wsum
    total = l1 + w_nrm * nrm + w_eik * eik
    return total, {"l1": float(l1.detach()), "nrm": float(nrm.detach()), "eik": float(eik.detach())}
