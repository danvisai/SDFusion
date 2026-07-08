"""Hybrid prior-retrain dataset (foundations task #4): real 3D BAG massing (+ era/floors labels)
mixed with the labeled procedural recipe corpus (8 named styles) — the user's chosen conditioning.

BAG is read from the fast (re-chunked) /dev/shm h5 via Bag3dDataset. The recipe corpus has SLOW
chunking ((391,4,4,8) -> random single-sample reads starve the GPU), so we PRELOAD a subsample of
each style into RAM once (sequential reads are fast); per-sample access is then instant.

Per-sample dict: {sdf, fp, class_id, style_id, height, era_id, floors_id, source}. BAG -> style 8 +
era/floors labels; procedural -> named style 0..7 + era/floors = unknown tokens. class_id=0 for now
(this retrain revives STYLE conditioning; class is a later add).
"""
from __future__ import annotations
from pathlib import Path

import h5py
import numpy as np
import torch

from datasets.base_dataset import BaseDataset
from datasets.bag3d_dataset import Bag3dDataset
from datasets.buildingnet_dataset import _augment_sdf_fp
from datasets.stage3a_dataset import _RECIPE_STYLE_ORDER

ERA_UNK, FLOORS_UNK, REGION_UNK = 5, 4, 3   # REGION_UNK = num_regions-1 (synthetic recipes have no culture)


def floors_bucket(f: np.ndarray) -> np.ndarray:
    b = np.full_like(f, FLOORS_UNK)
    b[(f >= 1) & (f <= 2)] = 0
    b[(f >= 3) & (f <= 5)] = 1
    b[(f >= 6) & (f <= 9)] = 2
    b[f >= 10] = 3
    return b


class HybridDataset(BaseDataset):
    def name(self) -> str:
        return f"HybridDataset-{self.phase}"

    def initialize(self, opt, phase: str = "train", cat: str = "all", res: int = 64) -> None:
        self.opt = opt
        self.phase = phase
        self.bag_ratio = float(getattr(opt, "bag_ratio", 0.5))
        self.trunc = float(getattr(opt, "trunc_thres", 0.2))
        self.augment = bool(getattr(opt, "augment", False)) and phase == "train"
        seed = int(getattr(opt, "seed", 0))

        # real BAG massing (fast /dev/shm reads) + extra labels (aligned to GLOBAL corpus order).
        # era/floors only exist for the NL BAG corpus; when the bag h5 is the bigger cross-cultural
        # real.h5 the labels no longer align -> disable them (region_id carries culture instead).
        self.bag = Bag3dDataset(); self.bag.initialize(opt, phase)
        self.era = self.floors_id = None
        lab_path = Path(getattr(opt, "bag_labels", "data/bag3d_v1/bag_labels.npz"))
        if lab_path.exists():
            lab = np.load(lab_path)
            self.era = lab["era"].astype(np.int64)
            self.floors_id = floors_bucket(lab["floors"].astype(np.int64))
            if len(self.era) < self.bag.n_total:
                # real.h5 = [NL(labelled, in bag order) | DE | JP]; labels cover the NL prefix only.
                print(f"[hybrid] bag_labels ({len(self.era)}) < corpus ({self.bag.n_total}); "
                      "applying era/floors to in-range (NL) rows, unknown for DE/JP")

        # procedural recipes -> PRELOAD a subsample of each style into RAM (sequential = fast)
        M = int(getattr(opt, "recipe_per_style", 1500))
        root = Path(getattr(opt, "recipe_aug_root", "data/recipe_augmentation_v1"))
        sdfs, fps, sids, hns = [], [], [], []
        for st in _RECIPE_STYLE_ORDER:
            p = root / f"{st}.h5"
            if not p.exists():
                continue
            with h5py.File(p, "r") as h:
                n = min(M, int(h["sdf"].shape[0]))
                sdf = h["sdf"][:n].astype(np.float16)          # sequential read
                fp = h["footprint"][:n].astype(np.uint8)
            occ_y = (sdf <= 0).any(axis=(1, 3))                # (n, H) over y
            for k in range(n):
                yk = np.where(occ_y[k])[0]
                hns.append(float((yk.max() - yk.min() + 1) * (2.0 / 63.0)) if yk.size else 0.0)
            sdfs.append(sdf); fps.append(fp); sids += [_RECIPE_STYLE_ORDER.index(st)] * n
        self.rec_sdf = np.concatenate(sdfs)                    # (Nrec,64,64,64) float16
        self.rec_fp = np.concatenate(fps)                      # (Nrec,64,64) uint8
        self.rec_sid = np.asarray(sids, np.int64)
        self.rec_hn = np.asarray(hns, np.float32)
        nrec = len(self.rec_sid)
        print(f"[hybrid] recipe preloaded {nrec} samples into RAM ({self.rec_sdf.nbytes/1e9:.1f} GB)")

        n = max(len(self.bag), nrec)
        self.virtual_len = 2 * n
        rng = np.random.default_rng(seed)
        self._is_bag = rng.random(self.virtual_len) < self.bag_ratio
        self._bag_idx = rng.integers(0, max(len(self.bag), 1), self.virtual_len)
        self._rec_idx = rng.integers(0, max(nrec, 1), self.virtual_len)
        print(f"[hybrid] phase={phase} bag={len(self.bag)} recipe={nrec} "
              f"bag_ratio={self.bag_ratio} virtual_len={self.virtual_len}")

    def __len__(self) -> int:
        return self.virtual_len

    def __getitem__(self, i: int) -> dict:
        if self._is_bag[i] and len(self.bag):
            bi = int(self._bag_idx[i]) % len(self.bag)
            item = dict(self.bag[bi])                          # carries region_id from source_id
            gi = int(self.bag.idxs[bi])                        # global index for labels
            # ALWAYS emit era/floors (unknown when no label for this row) so batch keys stay uniform
            # across the bag + recipe paths — default_collate requires identical keys. NL rows keep
            # their real era/floors (preserves the deployed prior's conditioning); DE/JP -> unknown.
            if self.era is not None and gi < len(self.era):
                item["era_id"] = torch.tensor(int(self.era[gi]), dtype=torch.long)
                item["floors_id"] = torch.tensor(int(self.floors_id[gi]), dtype=torch.long)
            else:
                item["era_id"] = torch.tensor(ERA_UNK, dtype=torch.long)
                item["floors_id"] = torch.tensor(FLOORS_UNK, dtype=torch.long)
            return item

        ri = int(self._rec_idx[i]) % len(self.rec_sid)
        sdf = torch.from_numpy(self.rec_sdf[ri].astype(np.float32)).clamp(-self.trunc, self.trunc).unsqueeze(0)
        fp = torch.from_numpy(self.rec_fp[ri].astype(np.float32)).unsqueeze(0)
        if self.augment:
            rng = np.random.default_rng((i * 0x9E3779B1) & 0xFFFFFFFF)
            sdf, fp = _augment_sdf_fp(sdf, fp, rng)
        return {
            "sdf": sdf, "fp": fp,
            "class_id": torch.tensor(0, dtype=torch.long),
            "style_id": torch.tensor(int(self.rec_sid[ri]), dtype=torch.long),
            "height": torch.tensor(float(self.rec_hn[ri]), dtype=torch.float32),
            "era_id": torch.tensor(ERA_UNK, dtype=torch.long),
            "floors_id": torch.tensor(FLOORS_UNK, dtype=torch.long),
            "region_id": torch.tensor(REGION_UNK, dtype=torch.long),   # synthetic = no culture
            "source": f"recipe:{int(self.rec_sid[ri])}", "path": "ram",
        }
