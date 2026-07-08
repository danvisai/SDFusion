"""Stage 3a training dataset — mixes real BuildingNet + recipe-aug samples.

Per-sample dict:
    {
        'sdf':       (1, 64, 64, 64) float32, optionally truncated to ±trunc_thres,
        'fp':        (1, 64, 64)    float32   binary footprint (axis convention: (D=z, W=x))
        'class_id':  ()              long     index into the unified 53-way subtype map
        'style_id':  ()              long     0..7 for recipes, 8 for "unknown" (real assets)
        'height':    ()              float    Frame-N Y extent (range ~ 0..2)
        'source':    str                     'building' | 'recipe:<style>'
        'path':      str                     source file path
    }

Mixes the two corpora by `--recipe_aug_ratio` (default 0.7 = 70% recipes, 30%
real). Per-epoch behavior: every epoch samples `len_recipe + len_buildingnet`
items by drawing from one corpus or the other per `__getitem__`. The intent is
that the model sees both distributions evenly without manual interleaving.

Augmentation (train phase only, `--augment`): Y-rotations (k*90deg) + X/Z flips
on the SDF + footprint together. Identical machinery to
`datasets/buildingnet_dataset.py:_augment_sdf_fp`.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch

from datasets.base_dataset import BaseDataset
from datasets.buildingnet_dataset import _augment_sdf_fp
from datasets.buildingnet_retrieval_dataset import (
    build_label_maps, load_split_ids, subtype_label,
)


STYLE_UNKNOWN_ID = 8  # 8 recipe styles (0..7) + 1 "unknown" at index 8


def _heights_csv_to_dict(csv_path: Path) -> dict[str, float]:
    """Parse outputs/stage3_metadata/asset_dimensions.csv -> {id: y_extent_n}."""
    out: dict[str, float] = {}
    if not csv_path.exists():
        return out
    import csv
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            out[row["id"]] = float(row["y_extent_n"])
    return out


class Stage3aDataset(BaseDataset):
    """Mixed BuildingNet + recipe-aug dataset for Stage 3a conditional diffusion."""

    def name(self) -> str:
        return f"Stage3aDataset-{self.phase}"

    def initialize(
        self,
        opt,
        phase: str = "train",
        cat: str = "all",
        res: int = 64,
    ) -> None:
        """Match the (opt, phase, cat, res) init pattern used by CreateDataset."""
        self.opt = opt
        self.data_root = Path(getattr(opt, "dataroot", "data"))
        self.phase = phase
        self.recipe_aug_root = Path(getattr(opt, "recipe_aug_root",
                                            "data/recipe_augmentation_v1"))
        heights_csv = Path(getattr(opt, "heights_csv",
                                   "outputs/stage3_metadata/asset_dimensions.csv"))
        self.recipe_aug_ratio = float(getattr(opt, "recipe_aug_ratio", 0.7))
        self.trunc_thres = float(getattr(opt, "trunc_thres", 0.2))  # v1 VQVAE's native trunc
        self.augment = bool(getattr(opt, "augment", False)) and phase == "train"
        self.seed = int(getattr(opt, "seed", 0))
        recipe_styles = getattr(opt, "recipe_styles", None)

        # 1) Real BuildingNet split — gives us the unified 53-way subtype map.
        bn_root = self.data_root / "BuildingNet_dataset_v0_1"
        train_ids = load_split_ids(bn_root, "train")
        val_ids   = load_split_ids(bn_root, "val")
        test_ids  = load_split_ids(bn_root, "test")
        all_ids   = train_ids + val_ids + test_ids
        self.subtype_to_idx, self.top_to_idx = build_label_maps(all_ids)

        phase_ids = {"train": train_ids, "val": val_ids, "test": test_ids}[phase]
        self.bn_ids: list[str] = phase_ids
        self.bn_h5_paths = [
            bn_root / "resolution_64" / mid / "ori_sample_grid.h5"
            for mid in self.bn_ids
        ]

        # 2) Per-asset Frame-N Y extents (= our height proxy at training time).
        heights = _heights_csv_to_dict(heights_csv)
        self.bn_heights = [
            float(heights.get(mid, 0.0)) for mid in self.bn_ids
        ]
        missing_heights = sum(1 for h in self.bn_heights if h == 0.0)
        if missing_heights:
            print(f"[stage3a] WARN: {missing_heights}/{len(self.bn_ids)} BuildingNet ids "
                  f"missing height entries in {heights_csv}; defaulted to 0.0")

        # 3) Recipe-aug corpus — one h5 per style.
        if recipe_styles is None:
            recipe_styles = sorted(
                p.stem for p in self.recipe_aug_root.glob("*.h5")
            )
        self.recipe_styles = recipe_styles

        # We index recipes by (style_idx, sample_idx). Open each h5 lazily to
        # support DataLoader workers; cache the (n_samples, class_label_list)
        # at construction time.
        self.recipe_meta: list[dict] = []  # one entry per style file
        self.recipe_cum: list[int] = [0]
        for st in self.recipe_styles:
            h5_path = self.recipe_aug_root / f"{st}.h5"
            if not h5_path.exists():
                continue
            with h5py.File(h5_path, "r") as f:
                n = int(f["sdf"].shape[0])
                class_labels = [b.decode("utf-8") for b in f["class_label"][:]]
            self.recipe_meta.append({
                "style": st,
                "style_id": st_to_id(st),
                "h5_path": str(h5_path),
                "n": n,
                "class_labels": class_labels,
            })
            self.recipe_cum.append(self.recipe_cum[-1] + n)
        self.recipe_total = self.recipe_cum[-1] if self.recipe_meta else 0

        # h5py file handles are per-worker, opened lazily in __getitem__.
        self._h5_handles: dict[str, h5py.File] = {}

        # 4) Compose a flat virtual length by union, with a stable per-index
        # source/global-id map so __getitem__ is deterministic.
        # We choose virtual length = max(n_bn, ceil(n_recipe / aug_ratio))
        # so neither corpus gets starved at the chosen mix ratio.
        n_bn = len(self.bn_ids)
        n_rc = self.recipe_total
        if n_bn == 0 and n_rc == 0:
            raise RuntimeError("Stage3aDataset: no samples found in either corpus.")
        if self.recipe_aug_ratio >= 1.0:
            virt = n_rc
        elif self.recipe_aug_ratio <= 0.0:
            virt = n_bn
        else:
            virt = max(n_bn, int(n_rc / max(self.recipe_aug_ratio, 1e-6)))
        self.virtual_len = virt

        rng = np.random.default_rng(self.seed)
        # Pre-roll the source per virtual index (bool: True = recipe).
        self._is_recipe = rng.random(self.virtual_len) < self.recipe_aug_ratio
        # And the underlying index within that corpus.
        self._inner_idx = np.where(
            self._is_recipe,
            rng.integers(0, max(n_rc, 1), size=self.virtual_len),
            rng.integers(0, max(n_bn, 1), size=self.virtual_len),
        )

        print(f"[stage3a] phase={phase}  bn={n_bn}  recipe={n_rc} "
              f"styles={len(self.recipe_meta)}  ratio={self.recipe_aug_ratio:.2f} "
              f"virtual_len={self.virtual_len}")

    # ---- helpers --------------------------------------------------------

    def _get_h5(self, path: str) -> h5py.File:
        h = self._h5_handles.get(path)
        if h is None:
            h = h5py.File(path, "r")
            self._h5_handles[path] = h
        return h

    def _resolve_recipe_idx(self, ridx: int) -> tuple[dict, int]:
        """ridx is flat across all style files; return (meta, sample_idx_in_style)."""
        # Binary-search the cumulative-count array.
        from bisect import bisect_right
        si = bisect_right(self.recipe_cum, ridx) - 1
        si = max(0, min(si, len(self.recipe_meta) - 1))
        local = ridx - self.recipe_cum[si]
        return self.recipe_meta[si], int(local)

    def _load_buildingnet(self, bn_idx: int) -> dict:
        bn_idx = int(bn_idx) % max(len(self.bn_ids), 1)
        mid = self.bn_ids[bn_idx]
        h5_path = str(self.bn_h5_paths[bn_idx])
        h = self._get_h5(h5_path)
        sdf_np = h["pc_sdf_sample"][:].astype(np.float32)
        fp_np = h["footprint"][:].astype(np.uint8)
        sdf = torch.from_numpy(sdf_np).view(1, 64, 64, 64)
        fp = torch.from_numpy(fp_np).float()  # (1, 64, 64)
        class_id = self.subtype_to_idx.get(subtype_label(mid), 0)
        height = self.bn_heights[bn_idx]
        return {
            "sdf": sdf,
            "fp": fp,
            "class_id": int(class_id),
            "style_id": STYLE_UNKNOWN_ID,
            "height": float(height),
            "source": "building",
            "path": h5_path,
        }

    def _load_recipe(self, ridx: int) -> dict:
        ridx = int(ridx) % max(self.recipe_total, 1)
        meta, local = self._resolve_recipe_idx(ridx)
        h = self._get_h5(meta["h5_path"])
        sdf_np = h["sdf"][local].astype(np.float32)             # (64, 64, 64)
        fp_np = h["footprint"][local].astype(np.uint8)          # (64, 64)
        height_m = float(h["height_m"][local])
        # Recipe class label -> unified 53-way subtype id (the 5 recipe classes
        # are all present in the BuildingNet subtype space).
        class_label = meta["class_labels"][local]
        class_id = self.subtype_to_idx.get(class_label, 0)

        sdf = torch.from_numpy(sdf_np).unsqueeze(0)   # (1, 64, 64, 64)
        fp = torch.from_numpy(fp_np).unsqueeze(0).float()  # (1, 64, 64)

        # Recipe SDFs are in world coords (meters); convert to a Frame-N-comparable
        # height (used as the height-conditioning scalar): bbox Y extent of the
        # iso=0 region divided by 64 voxels' worth of Frame-N units (2 units total).
        # We use the recipe's own footprint occupancy along the H axis as a proxy.
        occupied_y = (sdf_np <= 0).any(axis=(0, 2))  # (H,) along y
        if occupied_y.any():
            ys = np.where(occupied_y)[0]
            height_n = float((ys.max() - ys.min() + 1) * (2.0 / 63.0))
        else:
            height_n = 0.0

        return {
            "sdf": sdf,
            "fp": fp,
            "class_id": int(class_id),
            "style_id": int(meta["style_id"]),
            "height": float(height_n),
            "source": f"recipe:{meta['style']}",
            "path": meta["h5_path"] + f":#{local}",
        }

    # ---- Dataset interface ----------------------------------------------

    def __len__(self) -> int:
        return self.virtual_len

    def __getitem__(self, index: int) -> dict:
        is_recipe = bool(self._is_recipe[index])
        inner = int(self._inner_idx[index])
        if is_recipe and self.recipe_total > 0:
            item = self._load_recipe(inner)
        else:
            item = self._load_buildingnet(inner)

        sdf, fp = item["sdf"], item["fp"]
        if self.trunc_thres > 0.0:
            sdf = torch.clamp(sdf, min=-self.trunc_thres, max=self.trunc_thres)

        if self.augment:
            # Per-call RNG so workers don't desync.
            seed = (
                hash(item["path"]) ^
                (torch.initial_seed() & 0xFFFFFFFF) ^
                (index * 0x9E3779B1)
            ) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)
            sdf, fp = _augment_sdf_fp(sdf, fp, rng)

        return {
            "sdf": sdf,
            "fp": fp,
            "class_id": torch.tensor(item["class_id"], dtype=torch.long),
            "style_id": torch.tensor(item["style_id"], dtype=torch.long),
            "height": torch.tensor(item["height"], dtype=torch.float32),
            "source": item["source"],
            "path": item["path"],
        }


# --- helpers used by the constructor + by training-time inference glue ------

_RECIPE_STYLE_ORDER = (
    "modern", "colonial", "victorian", "industrial",
    "craftsman", "mediterranean", "contemporary", "public_civic",
)


def st_to_id(style_name: str) -> int:
    """Stable style index used by Stage 3a. Matches scene/sdf_recipes.py:STYLES order."""
    return _RECIPE_STYLE_ORDER.index(style_name)
