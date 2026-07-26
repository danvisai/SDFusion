"""3D BAG dataset — real watertight LoD2.2 buildings (Netherlands) as a clean SDF
massing corpus for the SDEdit prior (Stage3a model).

Same per-sample contract as Stage3aDataset, so the unchanged Stage3a model consumes it:
    {'sdf': (1,64,64,64) clamped to +-trunc, 'fp': (1,64,64), 'class_id': long,
     'style_id': long (=8 'unknown'/real), 'height': float (Frame-N Y extent), ...}

Source corpus: data/bag3d_v1/bag3d.h5 (built by scripts/ingest_3dbag.py). No style/class
labels (real buildings), so style_id=8 and class_id=0 — this is a MASSING prior; richness
comes from the real geometry, not the labels. See memory project_sdedit_corpus_ceiling.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from datasets.base_dataset import BaseDataset
from datasets.buildingnet_dataset import _augment_sdf_fp

STYLE_UNKNOWN_ID = 8


class Bag3dDataset(BaseDataset):
    def name(self) -> str:
        return f"Bag3dDataset-{self.phase}"

    def initialize(self, opt, phase: str = "train", cat: str = "all", res: int = 64) -> None:
        self.opt = opt
        self.phase = phase
        self.h5_path = Path(getattr(opt, "bag3d_h5", "data/bag3d_v1/bag3d.h5"))
        self.trunc_thres = float(getattr(opt, "trunc_thres", 0.2))
        self.augment = bool(getattr(opt, "augment", False)) and phase == "train"
        with h5py.File(self.h5_path, "r") as f:
            self.n_total = int(f["sdf"].shape[0])
        # deterministic 96/2/2 split
        perm = np.random.default_rng(0).permutation(self.n_total)
        n_val = max(1, int(0.02 * self.n_total))
        splits = {"val": perm[:n_val], "test": perm[n_val:2 * n_val], "train": perm[2 * n_val:]}
        self.idxs = splits[phase]
        self._h5 = None
        print(f"[bag3d] phase={phase}  n={len(self.idxs)} / {self.n_total}  h5={self.h5_path}")

    def _get_h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __len__(self) -> int:
        return len(self.idxs)

    def __getitem__(self, i: int) -> dict:
        gi = int(self.idxs[i])
        h = self._get_h5()
        sdf_np = h["sdf"][gi].astype(np.float32)        # (64,64,64) signed, Frame-N
        fp_np = h["footprint"][gi].astype(np.uint8)     # (64,64)
        # region/culture token: real.h5 carries source_id (0=NL 1=DE 2=JP); legacy bag3d.h5 = NL(0)
        region_id = int(h["source_id"][gi]) if "source_id" in h else 0
        # height conditioning = Frame-N Y extent of the iso=0 region (matches Stage3aDataset)
        occ_y = (sdf_np <= 0).any(axis=(0, 2))           # along H (axis 1)
        if occ_y.any():
            ys = np.where(occ_y)[0]
            height_n = float((ys.max() - ys.min() + 1) * (2.0 / 63.0))
        else:
            height_n = 0.0

        sdf = torch.from_numpy(sdf_np).unsqueeze(0)
        fp = torch.from_numpy(fp_np).unsqueeze(0).float()
        if self.trunc_thres > 0.0:
            sdf = torch.clamp(sdf, min=-self.trunc_thres, max=self.trunc_thres)
        if self.augment:
            rng = np.random.default_rng((gi * 0x9E3779B1) & 0xFFFFFFFF)
            sdf, fp = _augment_sdf_fp(sdf, fp, rng)

        return {
            "sdf": sdf,
            "fp": fp,
            "class_id": torch.tensor(0, dtype=torch.long),
            "style_id": torch.tensor(STYLE_UNKNOWN_ID, dtype=torch.long),
            "height": torch.tensor(height_n, dtype=torch.float32),
            "region_id": torch.tensor(region_id, dtype=torch.long),
            "source": "bag3d",
            "path": f"{self.h5_path}:#{gi}",
        }
