"""Stage 3b training dataset — pairs each BuildingNet SDF with its voxelized Gaussian target.

Per-sample dict:
    sdf:        (1, 64, 64, 64)  float32   GT SDF (truncated to ±trunc_thres)
    fp:         (1, 64, 64)      float32   binary footprint
    class_id:   ()               long      0..52
    style_id:   ()               long      0..8 (8 recipes + 1 unknown = 8 here)
    height:     ()               float     Frame-N Y extent

    # Voxelized Gaussian target (from scripts/voxelize_gsplats.py):
    slots:      (32, 32, 32, 8, 14)  float32
    occ_count:  (32, 32, 32)         uint8    (number of slots in [0..8])
    bbox:       (2, 3)               float32  (lo, hi) Frame-N bbox
    n_total:    ()                   int32
    n_kept:     ()                   int32

    source: str
    path:   str

NOTE: this dataset trains ONLY on real BuildingNet ids (we don't have v2
Gaussians for the recipe-aug samples). Recipe augmentation is a Stage 3a
strategy; Stage 3b inherits the SDF distribution that Stage 3a learns and
maps it to Gaussians via the v2 baked targets.
"""
from __future__ import annotations
from pathlib import Path

import h5py
import numpy as np
import torch

from datasets.base_dataset import BaseDataset
from datasets.buildingnet_dataset import _augment_sdf_fp
from datasets.buildingnet_retrieval_dataset import (
    build_label_maps, load_split_ids, subtype_label,
)
from datasets.stage3a_dataset import (
    STYLE_UNKNOWN_ID, _heights_csv_to_dict,
)


class Stage3bDataset(BaseDataset):
    """SDF + footprint + voxelized Gaussian target pairs."""

    def name(self) -> str:
        return f"Stage3bDataset-{self.phase}"

    def initialize(self, opt, phase: str = "train", cat: str = "all", res: int = 64) -> None:
        self.opt = opt
        self.data_root = Path(getattr(opt, "dataroot", "data"))
        self.phase = phase
        self.trunc_thres = float(getattr(opt, "trunc_thres", 0.3))
        self.augment = bool(getattr(opt, "augment", False)) and phase == "train"
        self.seed = int(getattr(opt, "seed", 0))

        bn_root = self.data_root / "BuildingNet_dataset_v0_1"
        all_ids = (
            load_split_ids(bn_root, "train")
            + load_split_ids(bn_root, "val")
            + load_split_ids(bn_root, "test")
        )
        self.subtype_to_idx, _ = build_label_maps(all_ids)

        phase_ids = load_split_ids(bn_root, phase)
        vox_dir = Path(getattr(opt, "vox_root",
                               bn_root / "gsplat_voxelized_32k8"))
        sdf_dir = bn_root / "resolution_64"

        # Drop ids that don't have BOTH a voxelized GS and an SDF on disk.
        kept: list[tuple[str, Path, Path]] = []
        n_missing_vox, n_missing_sdf = 0, 0
        for mid in phase_ids:
            sdf_h5 = sdf_dir / mid / "ori_sample_grid.h5"
            vox_npz = vox_dir / f"{mid}.npz"
            if not sdf_h5.exists():
                n_missing_sdf += 1
                continue
            if not vox_npz.exists():
                n_missing_vox += 1
                continue
            kept.append((mid, sdf_h5, vox_npz))
        self.ids = [k[0] for k in kept]
        self.sdf_paths = [str(k[1]) for k in kept]
        self.vox_paths = [str(k[2]) for k in kept]

        heights = _heights_csv_to_dict(
            Path(getattr(opt, "heights_csv",
                         "outputs/stage3_metadata/asset_dimensions.csv"))
        )
        self.heights = [float(heights.get(mid, 0.0)) for mid in self.ids]

        # h5py handles are per-worker, opened lazily.
        self._h5_handles: dict[str, h5py.File] = {}

        print(f"[stage3b] phase={phase} ids={len(self.ids)} "
              f"missing_vox={n_missing_vox} missing_sdf={n_missing_sdf}")
        if not self.ids:
            raise RuntimeError(
                f"Stage3bDataset: no paired (SDF, voxelized GS) ids found for phase={phase}."
            )

    def __len__(self) -> int:
        return len(self.ids)

    def _get_h5(self, path: str) -> h5py.File:
        h = self._h5_handles.get(path)
        if h is None:
            h = h5py.File(path, "r")
            self._h5_handles[path] = h
        return h

    def __getitem__(self, index: int) -> dict:
        mid = self.ids[index]
        sdf_path = self.sdf_paths[index]
        vox_path = self.vox_paths[index]

        # SDF + footprint from the h5.
        h = self._get_h5(sdf_path)
        sdf_np = h["pc_sdf_sample"][:].astype(np.float32)
        fp_np = h["footprint"][:].astype(np.uint8)
        sdf = torch.from_numpy(sdf_np).view(1, 64, 64, 64)
        fp = torch.from_numpy(fp_np).float()  # (1, 64, 64)
        if self.trunc_thres > 0.0:
            sdf = torch.clamp(sdf, -self.trunc_thres, self.trunc_thres)

        # Voxelized Gaussian target from the .npz cache.
        with np.load(vox_path) as z:
            slots = z["slots"].astype(np.float32)        # (32, 32, 32, 8, 14)
            occ_count = z["occ_count"].astype(np.uint8)  # (32, 32, 32)
            bbox = z["bbox"].astype(np.float32)           # (2, 3)
            n_total = int(z["n_total"])
            n_kept = int(z["n_kept"])

        slots_t = torch.from_numpy(slots)
        occ_t = torch.from_numpy(occ_count)
        bbox_t = torch.from_numpy(bbox)

        # Optional axis-aligned augmentation. We rotate/flip the SDF + fp together,
        # but NOT the voxelized Gaussian slots — those encode oriented features
        # (means relative to a particular cell, quaternions etc.) whose semantics
        # under a 90 deg rotation require careful axis remapping of every attribute.
        # For v1 we just disable augmentation; future work can implement the
        # corresponding quaternion rotation + index remap.
        if self.augment:
            seed = (
                hash(mid) ^
                (torch.initial_seed() & 0xFFFFFFFF) ^
                (index * 0x9E3779B1)
            ) & 0xFFFFFFFF
            # Disabled — see comment.
            pass

        class_id = self.subtype_to_idx.get(subtype_label(mid), 0)
        height = self.heights[index]

        return {
            "sdf": sdf,
            "fp": fp,
            "class_id": torch.tensor(class_id, dtype=torch.long),
            "style_id": torch.tensor(STYLE_UNKNOWN_ID, dtype=torch.long),  # CLIP labels still TBD
            "height": torch.tensor(height, dtype=torch.float32),
            "slots": slots_t,
            "occ_count": occ_t,
            "bbox": bbox_t,
            "n_total": torch.tensor(n_total, dtype=torch.int32),
            "n_kept": torch.tensor(n_kept, dtype=torch.int32),
            "source": "building",
            "path": sdf_path,
        }
