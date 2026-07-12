"""Ticket 11: loads ticket 07's real (coarse, target) monolith pairs for training.

No SDF grids are duplicated to disk (ticket 07's own design: `data/monolith_pairs_v1/` holds
only ids + provenance). Both tensors are recomputed on the fly from the same functions ticket
07 already validated over all 1572 `train_100` ids (0 failures, 0 leakage, footprint-axis IoU
1.0/1.0): `render_facades.load_buildingnet_sdf` for the real target and
`build_monolith_pairs.low_pass_sdf` for ADR 0004's locked coarse-input primary.

Network input representation: the raw target/coarse SDF grids have a wide, per-building dynamic
range (BuildingNet's `pc_sdf_sample` field is not pre-truncated -- measured range roughly
[-0.45, 17.3] across `train_100`). Stage3a's own live input contract already solves exactly
this problem (`scripts/foundations/eval_harness.py`'s `frame_n_input`: derive occupancy, recompute
an EDT-true SDF, clip to +-0.2 m) -- reused here unchanged rather than inventing a second
normalization, then rescaled by `trunc` to land in [-1, 1] (the range a diffusion process
expects). `frame_n_input` is applied to OCCUPANCY, so it is used identically for the coarse
input (`low_pass_sdf(...) <= 0`, mirroring `transform_vs_noise.py`'s own
`footprint_extrude_blockout` -> `frame_n_input` pipeline for the SDEdit blockout condition).

The train/gradient-validation split here is carved OUT OF `train_100` itself (never the sealed
`data/splits_v1/test.json`, which stays reserved for tickets 12/13's headline comparison) --
purely so training has a held-out slice to monitor convergence against.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

REPO = Path(__file__).resolve().parents[1]
for _p in ("scripts/eval", "scripts/foundations"):
    sys.path.insert(0, str(REPO / _p))

import render_facades as rf  # noqa: E402
from build_monolith_pairs import WORKING_RES, S_STAR_VOXELS, low_pass_sdf  # noqa: E402
from eval_harness import frame_n_input  # noqa: E402

TRUNC = 0.2  # matches eval_harness.frame_n_input's own default


def train_val_ids(ids, val_frac: float = 0.1, seed: int = 0):
    """Deterministic held-out-from-gradients slice of `ids` (e.g. `train_100`'s pairs) for
    monitoring convergence. Disjoint from and does not touch the sealed test split."""
    ids = sorted(ids)
    rng = np.random.default_rng(seed)
    perm = [ids[i] for i in rng.permutation(len(ids))]
    n_val = round(val_frac * len(ids))
    val, train = perm[:n_val], perm[n_val:]
    return sorted(train), sorted(val)


def apply_axis_aug(arr: np.ndarray, k_rot: int, flip_x: bool, flip_z: bool) -> np.ndarray:
    """Rotate a `(D,H,W)` volume around the up axis (H, axis=1) by `k_rot*90` degrees and
    optionally mirror along D/W -- 1849 real buildings is a small corpus (Layer 1's known data
    bottleneck), and every one of these 8 transforms is an equally valid "real building" (the
    up axis is physically meaningful and never touched)."""
    out = np.rot90(arr, k=k_rot, axes=(0, 2)) if k_rot else arr
    if flip_x:
        out = np.flip(out, axis=2)
    if flip_z:
        out = np.flip(out, axis=0)
    return np.ascontiguousarray(out)


class MonolithPairDataset(Dataset):
    """`ids`: building ids to draw from (e.g. `train_val_ids(...)`'s train or val half)."""

    def __init__(self, ids, working_res: int = WORKING_RES, s_star_vox: float = S_STAR_VOXELS,
                 trunc: float = TRUNC, augment: bool = False, device: str = "cpu"):
        self.ids = list(ids)
        self.working_res = working_res
        self.s_star_vox = s_star_vox
        self.trunc = trunc
        self.augment = augment
        self.device = device

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict:
        bid = self.ids[index]
        target_raw = rf.load_buildingnet_sdf(bid, working_res=self.working_res, device=self.device)
        coarse_raw = low_pass_sdf(target_raw, self.working_res, self.s_star_vox, device=self.device)
        target_occ = target_raw <= 0
        coarse_occ = coarse_raw <= 0

        if self.augment:
            rng = np.random.default_rng()
            k_rot, flip_x, flip_z = int(rng.integers(4)), bool(rng.integers(2)), bool(rng.integers(2))
            target_occ = apply_axis_aug(target_occ, k_rot, flip_x, flip_z)
            coarse_occ = apply_axis_aug(coarse_occ, k_rot, flip_x, flip_z)

        target_sdf, _, _ = frame_n_input(target_occ, self.device, trunc=self.trunc)
        coarse_sdf, _, _ = frame_n_input(coarse_occ, self.device, trunc=self.trunc)
        return dict(
            building=bid,
            target=(target_sdf[0] / self.trunc).float(),  # (1,R,R,R), rescaled to [-1,1]
            coarse=(coarse_sdf[0] / self.trunc).float(),
        )
