# debug_loader.py (put this at the repo root)

import os
import torch
from torch.utils.data import DataLoader

from datasets.building_fp2shape_dataset import BuildingFp2ShapeDataset

def test_loader(dataroot):
    # 1) Instantiate your dataset
    ds = BuildingFp2ShapeDataset(dataroot=dataroot, split="train")
    print("✅ Dataset loaded, length =", len(ds))

    # 2) Try a single getitem
    sample = ds[0]
    print("   Sample keys:", list(sample.keys()))
    print("   img shape:",   sample["img"].shape)
    print("   sdf shape:",   sample["sdf"].shape)

    # 3) Wrap in a DataLoader with no workers, no pinning
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
    print("✅ DataLoader created. Fetching first batch…")
    batch = next(iter(loader))
    print("   Batch keys:", batch.keys())
    print("   img batch shape:", batch["img"].shape)
    print("   sdf batch shape:", batch["sdf"].shape)
    print("🆗 Loader test passed without crash.")

if __name__ == "__main__":
    # adjust this path if needed
    dataroot = "data/BuildingNet_dataset_v0_1"
    test_loader(dataroot)
