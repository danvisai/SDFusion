"""Compose multiple placed GaussianSets into a single scene-wide GaussianSet.

Analog of scripts/compose_hunyuan_scene_smoke.py for meshes: takes a list of
already-placed (Frame W) GaussianSets and concatenates them into one big set
whose raw fields are simple row-stacks.
"""
from __future__ import annotations
from typing import Iterable, List

import torch

from scene.gsplat_common import GaussianSet


def compose(parts: Iterable[GaussianSet]) -> GaussianSet:
    """Stack-concatenate a list of placed GaussianSets."""
    parts: List[GaussianSet] = list(parts)
    if not parts:
        raise ValueError("compose() called with no parts")
    device = parts[0].means.device
    dtype = parts[0].means.dtype
    parts = [p.to(device) for p in parts]

    means = torch.cat([p.means for p in parts], dim=0)
    raw_scales = torch.cat([p.raw_scales for p in parts], dim=0)
    raw_quats = torch.cat([p.raw_quats for p in parts], dim=0)
    raw_opac = torch.cat([p.raw_opac for p in parts], dim=0)
    sh_dc = torch.cat([p.sh_dc for p in parts], dim=0)
    return GaussianSet(
        means=means.to(dtype),
        raw_scales=raw_scales.to(dtype),
        raw_quats=raw_quats.to(dtype),
        raw_opac=raw_opac.to(dtype),
        sh_dc=sh_dc.to(dtype),
    )
