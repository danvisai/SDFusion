import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import h5py

class BuildingFp2ShapeDataset(Dataset):
    """Loads (footprint_image, SDF_tensor) pairs from BuildingNet."""
    def __init__(self, dataroot, split="train", img_transforms=None):
        """
        Expects directory structure:
          dataroot/
            footprints_png/
              train/
              val/
              test/
            resolution_64/
              <id1>/ori_sample_grid.h5
              <id2>/ori_sample_grid.h5
            splits/
              train_split.txt
              val_split.txt
              test_split.txt
        """
        self.dataroot = dataroot
        self.split = split

        # load the list of IDs
        split_file = os.path.join(dataroot, "splits", f"{split}_split.txt")
        with open(split_file, 'r') as f:
            self.ids = [l.strip() for l in f if l.strip()]

        # directories for images and SDFs
        self.img_dir = os.path.join(dataroot, "footprints_png", split)
        self.sdf_dir = os.path.join(dataroot, "resolution_64")

        # default image transforms
        self.img_transforms = img_transforms or transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]

        # 1) load footprint image
        img_path = os.path.join(self.img_dir, f"{id_}.png")
        img = Image.open(img_path).convert("RGB")
        img = self.img_transforms(img)

        # 2) load SDF from HDF5
        h5_path = os.path.join(self.sdf_dir, id_, "ori_sample_grid.h5")
        with h5py.File(h5_path, "r") as f:
            # try preferred keys first
            if "pc_sdf_sample" in f:
                arr = f["pc_sdf_sample"][()]
            elif "pc_sdf_original" in f:
                arr = f["pc_sdf_original"][()]
            else:
                # fallback: pick the first 3D+ array
                for key in f.keys():
                    data = f[key][()]
                    if data.ndim >= 3:
                        arr = data
                        print(f"[BuildingFp2ShapeDataset] auto-loaded volumetric data from key='{key}'")
                        break
                else:
                    raise RuntimeError(
                        f"No 3D array found in {h5_path}; available keys: {list(f.keys())}"
                    )

        # 3) if it's flattened (shape = N,), reshape back to (D,D,D)
        #    infer D from the 'resolution_64' folder name
        res = int(os.path.basename(self.sdf_dir).split("_")[-1])
        arr = arr.reshape((res, res, res))

        # 4) wrap into a (1, D, D, D) tensor
        sdf = torch.from_numpy(arr).unsqueeze(0).float()

        return {
            "img": img,    # torch.Size([3,256,256])
            "sdf": sdf,    # torch.Size([1,64,64,64])
        }

    def name(self):
        return "BuildingFp2ShapeDataset"
