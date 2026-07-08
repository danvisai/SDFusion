"""
    adopted from: https://github.com/shubhtuls/PixelTransformer/blob/03b65b8612fe583b3e35fc82b446b5503dd7b6bd/data/shapenet.py
"""
import os
import h5py
import numpy as np
from PIL import Image
from termcolor import colored, cprint

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.transforms as transforms

from datasets.base_dataset import BaseDataset


def _augment_sdf_fp(sdf, fp, rng):
    """Y-rotation (k*90 deg) + X/Z flip augmentation that preserves the
    BuildingNet axis convention (D=z, H=y, W=x; channel dim is 0).

    sdf : (1, D, H, W) float    fp : (3, D, W) float
    Both rotated/flipped together so footprint stays consistent with the SDF.
    """
    k = int(rng.integers(0, 4))         # 0, 1, 2, 3 quarter-turns about Y
    flip_x = bool(rng.integers(0, 2))
    flip_z = bool(rng.integers(0, 2))
    if k:
        # Rotate in the (D, W) plane = (axis 1, axis 3) for sdf, (axis 1, axis 2) for fp.
        sdf = torch.rot90(sdf, k=k, dims=(1, 3))
        fp = torch.rot90(fp, k=k, dims=(1, 2))
    if flip_x:
        sdf = torch.flip(sdf, dims=(3,))   # flip W
        fp = torch.flip(fp, dims=(2,))
    if flip_z:
        sdf = torch.flip(sdf, dims=(1,))   # flip D
        fp = torch.flip(fp, dims=(1,))
    return sdf, fp


# from https://github.com/laughtervv/DISN/blob/master/preprocessing/info.json
class BuildingNetDataset(BaseDataset):

    def initialize(self, opt, phase='train', cat='all', res=64):
        self.opt = opt
        self.phase = phase
        self.load_from_cached = False
        self.max_dataset_size = opt.max_dataset_size
        self.res = res
        # Per-worker RNG for augmentation; seeded later in __getitem__.
        self._augment = bool(getattr(opt, 'augment', False)) and phase == 'train'

        dataroot = opt.dataroot
        file_list = f'{dataroot}/BuildingNet_dataset_v0_1/splits/{phase}_split.txt'

        SDF_dir = f'{dataroot}/BuildingNet_dataset_v0_1/resolution_{res}'

        self.model_list = []
        self.z_list = []
        with open(file_list) as f:
            model_list_s = []
            z_list_s = []
            for l in f.readlines():
                model_id = l.rstrip('\n')
                
                path = f'{SDF_dir}/{model_id}/ori_sample_grid.h5'
                model_list_s.append(path)
            
            self.model_list += model_list_s
            self.z_list += z_list_s

        np.random.default_rng(seed=0).shuffle(self.model_list)
        np.random.default_rng(seed=0).shuffle(self.z_list)

        self.model_list = self.model_list[:self.max_dataset_size]
        self.z_list = self.z_list[:self.max_dataset_size]
        cprint('[*] %d samples loaded.' % (len(self.model_list)), 'yellow')

        self.N = len(self.model_list)

        self.to_tensor = transforms.ToTensor()

        self.img_root = os.path.join(
            opt.dataroot,
            "BuildingNet_dataset_v0_1",
            "footprints_png",
            phase
        )
        #transforms to get a 3xHxW tensor
        self.img_transform = transforms.Compose([
            transforms.Resize((res,res)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3,1,1)), ]
        )


    def __getitem__(self, index):

        sdf_h5_file = self.model_list[index]
        
        #h5_f = h5py.File(sdf_h5_file, 'r')
        #sdf = h5_f['pc_sdf_sample'][:].astype(np.float32)
        # open once, read both SDF and footprint
        with h5py.File(sdf_h5_file, 'r') as h5_f:
           # (N,1) float32 -> (1,res,res,res)
            sdf_np = h5_f['pc_sdf_sample'][:].astype(np.float32)
            fp_np  = h5_f['footprint'][:].astype(np.uint8)
        
        #sdf = torch.Tensor(sdf).view(1, self.res, self.res, self.res) #old code
        sdf = torch.from_numpy(sdf_np).view(1, self.res, self.res, self.res)
        # footprint comes in as (1, H, W)
        fp = torch.from_numpy(fp_np).float()    # convert 0/1 -> float
        fp = fp.repeat(3,1,1)
        #--------newcode
        # now load the matching footprint‐PNG as your “image” branch
        #model_id = os.path.splitext(os.path.basename(sdf_h5_file))[0]
        model_id = os.path.basename(os.path.dirname(sdf_h5_file))
        png_path = os.path.join(self.img_root, model_id + ".png")
        img = Image.open(png_path).convert("L")
        img = self.img_transform(img)   # → torch.FloatTensor (3×res×res)
        #----------

        thres = self.opt.trunc_thres
        if thres != 0.0:
            sdf = torch.clamp(sdf, min=-thres, max=thres)

        # Optional axis-aligned augmentation (Y-rotations + X/Z flips). The
        # `img` (PNG-loaded footprint render) is intentionally NOT rotated —
        # the VQVAE doesn't consume `img`, but downstream image-conditioned
        # paths might, and silently rotating their conditioning would break
        # them. The `fp` (binary footprint from h5) is rotated to stay
        # consistent with the SDF, since VQVAE-v2 training uses fp via the
        # soft_footprint_bce aux loss.
        if self._augment:
            seed_data = (
                hash(sdf_h5_file) ^
                (torch.initial_seed() & 0xFFFFFFFF) ^
                (index * 0x9E3779B1)
            ) & 0xFFFFFFFF
            rng = np.random.default_rng(seed_data)
            sdf, fp = _augment_sdf_fp(sdf, fp, rng)

        ret = {
            'sdf': sdf,
            'fp':fp,
            'img':img,  #new code
            'path': sdf_h5_file,

        }

        if self.load_from_cached:
            z_path = self.z_list[index]
            z = torch.from_numpy(np.load(z_path))
            ret['z'] = z
            ret['z_path'] = z_path

        return ret

    def __len__(self):
        return self.N

    def name(self):
        return f'BuildingNetDataset-{self.res}'