"""REPA — REPresentation Alignment for the Stage3a massing prior (training-gaps plan step 4).

Yu et al. 2024 (arXiv 2410.06940): adding an auxiliary loss that aligns the denoiser's
intermediate features with a frozen pretrained encoder's features of the CLEAN sample makes
diffusion training converge ~17x faster and sharper. Adapted to our 3D SDF setting:

  target  = DINOv2 patch features of two orthographic DEPTH RENDERS of the clean 64^3 SDF
            (top-down height map + front elevation), computed ON-THE-FLY so the targets stay
            consistent with dataset augmentation (rotations/flips). ~ms per batch on A100.
  student = the UNet middle_block feature map, axis-pooled to the matching view, projected
            by a small MLP (the only trainable part here), cosine-aligned per patch.

Per "REPA Works Until It Doesn't" (2505.16792) the alignment should be early-stopped — the
caller anneals/stops via its `repa_stop_iter` (see stage3a_model.forward).
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# keep hub/HF weights off the quota'd $HOME
_EXT = Path(__file__).resolve().parents[2] / "external"
os.environ.setdefault("TORCH_HOME", str(_EXT / "torch_hub"))
os.environ.setdefault("HF_HOME", str(_EXT / "hf_cache"))

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _mlp(c_in: int, c_out: int, hidden: int = 1024) -> nn.Sequential:
    return nn.Sequential(nn.Linear(c_in, hidden), nn.SiLU(),
                         nn.Linear(hidden, hidden), nn.SiLU(),
                         nn.Linear(hidden, c_out))


class RepaAlign(nn.Module):
    """loss = mean over views of (1 - cos(proj(pooled unet feat), DINOv2(depth render)))."""

    VIEWS = ("top", "front")

    def __init__(self, feat_ch: int, device: str = "cuda", dino: str = "facebook/dinov2-small"):
        super().__init__()
        self.device = device
        # transformers backend (torch.hub's dinov2 main branch needs py>=3.10; venv is 3.9)
        from transformers import Dinov2Model
        self.dino = Dinov2Model.from_pretrained(dino).to(device).eval()
        for p in self.dino.parameters():
            p.requires_grad_(False)
        self.dino_dim = int(self.dino.config.hidden_size)   # 384 for dinov2-small
        self.proj = nn.ModuleDict({v: _mlp(feat_ch, self.dino_dim) for v in self.VIEWS}).to(device)
        self._mean = _IMAGENET_MEAN.to(device)
        self._std = _IMAGENET_STD.to(device)

    # ---- depth renders of the clean SDF (B,1,D,H,W), axes (D=z, H=y up, W=x) -----------
    @torch.no_grad()
    def depth_views(self, sdf: torch.Tensor) -> dict:
        occ = (sdf[:, 0] <= 0).float()                              # (B, D, H, W)
        B, D, H, W = occ.shape
        ys = torch.linspace(0, 1, H, device=occ.device).view(1, 1, H, 1)
        top = (occ * ys).amax(dim=2)                                # (B, D, W) height-from-above
        zs = torch.linspace(1, 0, D, device=occ.device).view(1, D, 1, 1)
        front = (occ * zs).amax(dim=1)                              # (B, H, W) depth-from-front
        return {"top": top, "front": front}

    @torch.no_grad()
    def _dino_grid(self, img2d: torch.Tensor) -> torch.Tensor:
        """(B, h, w) in [0,1] -> DINOv2 patch features (B, dino_dim, 16, 16)."""
        x = img2d.unsqueeze(1).repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self._mean) / self._std
        toks = self.dino(pixel_values=x).last_hidden_state[:, 1:]   # drop CLS -> (B, 256, dim)
        g = int(toks.shape[1] ** 0.5)
        return toks.transpose(1, 2).reshape(-1, self.dino_dim, g, g)

    def forward(self, unet_feat: torch.Tensor, clean_sdf: torch.Tensor) -> torch.Tensor:
        """unet_feat: middle_block output (B, C, d, h, w) with axes (z, y, x)."""
        views = self.depth_views(clean_sdf)
        loss = 0.0
        for v in self.VIEWS:
            tgt = self._dino_grid(views[v])                          # (B, dim, 16, 16)
            pooled = unet_feat.mean(dim=3) if v == "top" else unet_feat.mean(dim=2)
            # pooled: (B, C, a, b) -> per-location MLP -> (B, dim, a, b) -> match DINO grid
            B, C, a, b = pooled.shape
            p = self.proj[v](pooled.permute(0, 2, 3, 1).reshape(-1, C))
            p = p.view(B, a, b, self.dino_dim).permute(0, 3, 1, 2)
            p = F.interpolate(p, size=tgt.shape[-2:], mode="bilinear", align_corners=False)
            loss = loss + (1.0 - F.cosine_similarity(p, tgt, dim=1)).mean()
        return loss / len(self.VIEWS)
