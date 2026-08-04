from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock3d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SDFResidualUNet(nn.Module):
    """Small 3D U-Net for correcting aligned retrieval SDFs.

    Input channels:
      0: aligned source SDF
      1: target footprint volume repeated along the vertical axis

    Output:
      predicted residual SDF in the same (D, H, W) frame.
    """

    def __init__(self, in_channels: int = 2, base_channels: int = 16, residual_clip: float = 1.0):
        super().__init__()
        c = int(base_channels)
        self.residual_clip = float(residual_clip)
        self.enc1 = ConvBlock3d(in_channels, c)
        self.enc2 = ConvBlock3d(c, c * 2)
        self.enc3 = ConvBlock3d(c * 2, c * 4)
        self.mid = ConvBlock3d(c * 4, c * 4)
        self.dec2 = ConvBlock3d(c * 6, c * 2)
        self.dec1 = ConvBlock3d(c * 3, c)
        self.out = nn.Conv3d(c, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool3d(e1, 2))
        e3 = self.enc3(F.avg_pool3d(e2, 2))
        m = self.mid(e3)
        u2 = F.interpolate(m, size=e2.shape[-3:], mode="trilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = F.interpolate(d2, size=e1.shape[-3:], mode="trilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        pred = self.out(d1)
        if self.residual_clip > 0:
            pred = torch.tanh(pred) * self.residual_clip
        return pred
