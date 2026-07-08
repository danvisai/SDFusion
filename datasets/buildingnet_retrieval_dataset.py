from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


TOP_LEVELS = ("RESIDENTIAL", "RELIGIOUS", "COMMERCIAL", "MILITARY", "PUBLIC")


def subtype_label(model_id: str) -> str:
    return model_id.split("_mesh", 1)[0]


def top_level_label(model_id: str) -> str:
    for prefix in TOP_LEVELS:
        if model_id.startswith(prefix):
            return prefix
    return "UNKNOWN"


def build_label_maps(ids: list[str]) -> tuple[dict[str, int], dict[str, int]]:
    subtypes = sorted({subtype_label(mid) for mid in ids})
    tops = sorted({top_level_label(mid) for mid in ids})
    return {x: i for i, x in enumerate(subtypes)}, {x: i for i, x in enumerate(tops)}


def load_split_ids(data_root: str | Path, phase: str) -> list[str]:
    path = Path(data_root) / "splits" / f"{phase}_split.txt"
    with path.open() as f:
        return [ln.strip() for ln in f if ln.strip()]


def load_footprint_png(data_root: str | Path, phase: str, model_id: str) -> torch.Tensor:
    path = Path(data_root) / "footprints_png" / phase / f"{model_id}.png"
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    arr = (arr > 127).astype(np.float32)
    return torch.from_numpy(arr)[None, ...]


def augment_footprint(fp: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Light augmentation preserving footprint identity for contrastive training."""
    x = fp.clone()
    k = int(torch.randint(0, 4, (1,), generator=generator).item())
    x = torch.rot90(x, k, dims=(-2, -1))
    if bool(torch.randint(0, 2, (1,), generator=generator).item()):
        x = torch.flip(x, dims=(-1,))
    if bool(torch.randint(0, 2, (1,), generator=generator).item()):
        x = torch.flip(x, dims=(-2,))

    # Small translation with zero padding.
    max_shift = 3
    dy = int(torch.randint(-max_shift, max_shift + 1, (1,), generator=generator).item())
    dx = int(torch.randint(-max_shift, max_shift + 1, (1,), generator=generator).item())
    x = F.pad(x[None], (max_shift, max_shift, max_shift, max_shift), mode="constant", value=0.0)[0]
    y0 = max_shift + dy
    x0 = max_shift + dx
    x = x[:, y0:y0 + fp.shape[-2], x0:x0 + fp.shape[-1]]
    return x.contiguous()


class BuildingNetRetrievalDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path = "data/BuildingNet_dataset_v0_1",
        phase: str = "train",
        subtype_to_idx: dict[str, int] | None = None,
        top_to_idx: dict[str, int] | None = None,
        augment: bool = False,
        max_samples: int | None = None,
    ):
        self.data_root = Path(data_root)
        self.phase = phase
        self.ids = load_split_ids(self.data_root, phase)
        if max_samples is not None and max_samples > 0:
            self.ids = self.ids[:max_samples]
        if subtype_to_idx is None or top_to_idx is None:
            subtype_to_idx, top_to_idx = build_label_maps(self.ids)
        self.subtype_to_idx = subtype_to_idx
        self.top_to_idx = top_to_idx
        self.augment = augment

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict[str, object]:
        mid = self.ids[index]
        fp = load_footprint_png(self.data_root, self.phase, mid)
        if self.augment:
            fp_a = augment_footprint(fp)
            fp_b = augment_footprint(fp)
        else:
            fp_a = fp
            fp_b = fp.clone()
        subtype = subtype_label(mid)
        top = top_level_label(mid)
        return {
            "id": mid,
            "fp": fp,
            "fp_a": fp_a,
            "fp_b": fp_b,
            "class_id": torch.tensor(self.subtype_to_idx[subtype], dtype=torch.long),
            "top_id": torch.tensor(self.top_to_idx[top], dtype=torch.long),
        }
