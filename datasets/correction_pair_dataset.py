from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _apply_aug3d(arr: np.ndarray, k_rot: int, flip_x: bool, flip_z: bool) -> np.ndarray:
    """Rotate around Y (axes 0,2 = z,x) by k*90° and optionally flip along z/x.
    Y-axis (axis=1) is unchanged because buildings have a fixed 'up'."""
    out = arr
    if k_rot:
        out = np.rot90(out, k=k_rot, axes=(0, 2))
    if flip_x:
        out = np.flip(out, axis=2)
    if flip_z:
        out = np.flip(out, axis=0)
    return np.ascontiguousarray(out)


def _apply_aug2d(fp: np.ndarray, k_rot: int, flip_x: bool, flip_z: bool) -> np.ndarray:
    """Apply matching rotation/flip to a (z, x) footprint."""
    out = fp
    if k_rot:
        out = np.rot90(out, k=k_rot, axes=(0, 1))
    if flip_x:
        out = np.flip(out, axis=1)
    if flip_z:
        out = np.flip(out, axis=0)
    return np.ascontiguousarray(out)


class CorrectionPairDataset(Dataset):
    """Loads aligned-source/target SDF pairs produced by build_correction_pairs.py."""

    def __init__(
        self,
        pair_root: str | Path = "data/BuildingNet_dataset_v0_1/correction_pairs",
        phase: str = "train",
        max_samples: int = 0,
        residual_clip: float = 1.0,
        augment: bool = False,
    ):
        self.pair_root = Path(pair_root)
        self.phase = phase
        self.residual_clip = float(residual_clip)
        self.augment = bool(augment)
        meta_path = self.pair_root / phase / "pair_metadata.csv"
        self.meta = pd.read_csv(meta_path)
        if max_samples and max_samples > 0:
            self.meta = self.meta.iloc[:max_samples].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.meta.iloc[index]
        data = np.load(row["pair_path"], allow_pickle=True)
        source = data["source_aligned_sdf"].astype(np.float32)
        target = data["target_sdf"].astype(np.float32)
        residual = data["residual_sdf"].astype(np.float32)
        footprint = data["target_footprint"].astype(np.float32)

        if self.residual_clip > 0:
            residual = np.clip(residual, -self.residual_clip, self.residual_clip)

        if self.augment:
            k_rot = random.randint(0, 3)
            flip_x = random.random() < 0.5
            flip_z = random.random() < 0.5
            source = _apply_aug3d(source, k_rot, flip_x, flip_z)
            target = _apply_aug3d(target, k_rot, flip_x, flip_z)
            residual = _apply_aug3d(residual, k_rot, flip_x, flip_z)
            footprint = _apply_aug2d(footprint, k_rot, flip_x, flip_z)

        # SDF tensors are stored in (z, y, x). Conv3d expects (C, D, H, W).
        fp_vol = np.repeat(footprint[:, None, :], source.shape[1], axis=1)
        x = np.stack([source, fp_vol], axis=0)
        return {
            "input": torch.from_numpy(x),
            "source_sdf": torch.from_numpy(source[None]),
            "target_sdf": torch.from_numpy(target[None]),
            "residual_sdf": torch.from_numpy(residual[None]),
            "query_id": row["query_id"],
            "source_id": row["source_id"],
        }
