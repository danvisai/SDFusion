"""Shared 3DGS data type + Inria-format PLY I/O.

All scene.gsplat_* modules (placement, compose, guardrail, renderer) operate on
the GaussianSet dataclass below. Parameters are stored in their *raw*
(pre-activation) form, matching the Inria 3DGS PLY convention, so the same
tensors can be saved/loaded round-trip with no loss.

Activations at render time:
    means     -> means
    raw_scales-> exp()
    raw_quats -> L2-normalize
    raw_opac  -> sigmoid()
    sh_dc     -> rgb = sh_dc * SH_C0 + 0.5
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from plyfile import PlyData, PlyElement


SH_C0 = 0.28209479177387814  # spherical harmonics degree-0 normalization


@dataclass
class GaussianSet:
    """3DGS in raw (pre-activation) form. Tensors share device + dtype."""
    means: torch.Tensor       # (N, 3)
    raw_scales: torch.Tensor  # (N, 3)  pre-exp
    raw_quats: torch.Tensor   # (N, 4)  un-normalized; convention (w, x, y, z)
    raw_opac: torch.Tensor    # (N,)    pre-sigmoid
    sh_dc: torch.Tensor       # (N, 3)  SH degree-0 coefficient; rgb = sh_dc * SH_C0 + 0.5

    @property
    def device(self):
        return self.means.device

    @property
    def n(self) -> int:
        return self.means.shape[0]

    def to(self, device) -> "GaussianSet":
        return GaussianSet(
            means=self.means.to(device),
            raw_scales=self.raw_scales.to(device),
            raw_quats=self.raw_quats.to(device),
            raw_opac=self.raw_opac.to(device),
            sh_dc=self.sh_dc.to(device),
        )

    def clone(self) -> "GaussianSet":
        return GaussianSet(
            means=self.means.clone(),
            raw_scales=self.raw_scales.clone(),
            raw_quats=self.raw_quats.clone(),
            raw_opac=self.raw_opac.clone(),
            sh_dc=self.sh_dc.clone(),
        )

    def filter(self, mask: torch.Tensor) -> "GaussianSet":
        """Boolean mask over Gaussians; returns a new set."""
        return GaussianSet(
            means=self.means[mask],
            raw_scales=self.raw_scales[mask],
            raw_quats=self.raw_quats[mask],
            raw_opac=self.raw_opac[mask],
            sh_dc=self.sh_dc[mask],
        )

    # --- activated views (no grad-tracking; for rendering) ----------------
    def activated_scales(self) -> torch.Tensor:
        return torch.exp(self.raw_scales)

    def activated_quats(self) -> torch.Tensor:
        return self.raw_quats / (self.raw_quats.norm(dim=-1, keepdim=True) + 1e-8)

    def activated_opac(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_opac)

    def activated_colors(self) -> torch.Tensor:
        return self.sh_dc * SH_C0 + 0.5


# --- I/O ---------------------------------------------------------------------

def load_inria_ply(path: str, device: str = "cpu") -> GaussianSet:
    """Load a 3DGS PLY in Inria format. Supports degree-0 (no f_rest) and
    higher-degree (with f_rest_*); we keep only the DC component."""
    ply = PlyData.read(path)
    v = ply["vertex"]
    means = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
    sh_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1).astype(np.float32)
    raw_opac = np.asarray(v["opacity"]).astype(np.float32)
    raw_scales = np.stack(
        [v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1
    ).astype(np.float32)
    raw_quats = np.stack(
        [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1
    ).astype(np.float32)
    t = lambda a: torch.from_numpy(a).to(device)
    return GaussianSet(
        means=t(means),
        raw_scales=t(raw_scales),
        raw_quats=t(raw_quats),
        raw_opac=t(raw_opac),
        sh_dc=t(sh_dc),
    )


def save_inria_ply(path: str, g: GaussianSet) -> None:
    """Save a GaussianSet to Inria PLY (degree-0 SH only)."""
    means_np = g.means.detach().cpu().numpy().astype(np.float32)
    scales_np = g.raw_scales.detach().cpu().numpy().astype(np.float32)
    quats_np = g.raw_quats.detach().cpu().numpy().astype(np.float32)
    opac_np = g.raw_opac.detach().cpu().numpy().reshape(-1, 1).astype(np.float32)
    sh_np = g.sh_dc.detach().cpu().numpy().astype(np.float32)
    N = means_np.shape[0]
    normals = np.zeros((N, 3), dtype=np.float32)

    dtype_full = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]
    elements = np.empty(N, dtype=dtype_full)
    attrs = np.concatenate(
        [means_np, normals, sh_np, opac_np, scales_np, quats_np], axis=1,
    )
    for i in range(N):
        elements[i] = tuple(attrs[i])
    PlyData([PlyElement.describe(elements, "vertex")], text=False).write(path)
