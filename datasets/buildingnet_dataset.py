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


# from https://github.com/laughtervv/DISN/blob/master/preprocessing/info.json
class BuildingNetDataset(BaseDataset):

    def initialize(self, opt, phase='train', cat='all', res=64):
        self.opt = opt
        self.load_from_cached = False
        self.max_dataset_size = opt.max_dataset_size
        self.res = res

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