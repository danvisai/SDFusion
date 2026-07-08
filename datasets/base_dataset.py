
import numpy as np
from imageio import imread
from PIL import Image

from termcolor import colored, cprint

import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms

from torchvision import datasets

# from configs.paths import dataroot


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def CreateDataset(opt):
    dataset = None
    val_dataset = None  # most modes have no val split; only buildingnet does

    # decide resolution later at model
    if opt.dataset_mode == 'snet':
        from datasets.snet_dataset import ShapeNetDataset
        train_dataset = ShapeNetDataset()
        test_dataset = ShapeNetDataset()
        train_dataset.initialize(opt, 'train', cat=opt.cat, res=opt.res)
        test_dataset.initialize(opt, 'test', cat=opt.cat, res=opt.res)

    elif opt.dataset_mode == 'pix3d_img2shape':
        from datasets.pix3d_dataset import Pix3DImg2ShapeDataset
        train_dataset = Pix3DImg2ShapeDataset()
        test_dataset = Pix3DImg2ShapeDataset()
        train_dataset.initialize(opt, 'train', cat=opt.cat, res=opt.res)
        test_dataset.initialize(opt, 'test', cat=opt.cat, res=opt.res)

    elif opt.dataset_mode == "building_fp2shape":
        from datasets.building_fp2shape_dataset import BuildingFp2ShapeDataset

        train_dataset = BuildingFp2ShapeDataset(
        dataroot       = opt.dataroot,
        split          = "train",        # train split
        img_transforms = None            # or pass opt.img_transforms
        )
        test_dataset = BuildingFp2ShapeDataset(
        dataroot       = opt.dataroot,
        split          = "val",          # or "test", whichever you evaluate on
        img_transforms = None
        )


    elif opt.dataset_mode == 'text2shape':
        from datasets.text2shape_dataset import Text2ShapeDataset
        train_dataset = Text2ShapeDataset()
        test_dataset = Text2ShapeDataset()
        train_dataset.initialize(opt, 'train', cat=opt.cat, res=opt.res)
        test_dataset.initialize(opt, 'test', cat=opt.cat, res=opt.res)


    elif opt.dataset_mode == 'snet_mm2shape':
        from datasets.snet_mm2shape_dataset import ShapeNetMultiModal2ShapeDataset
        train_dataset = ShapeNetMultiModal2ShapeDataset()
        test_dataset = ShapeNetMultiModal2ShapeDataset()
        train_dataset.initialize(opt, 'train', cat=opt.cat, res=opt.res)
        test_dataset.initialize(opt, 'test', cat=opt.cat, res=opt.res)

    elif opt.dataset_mode == 'building':
        from datasets.buildingnet_dataset import BuildingNetDataset
        train_dataset = BuildingNetDataset()
        val_dataset = BuildingNetDataset()
        test_dataset = BuildingNetDataset()
        train_dataset.initialize(opt, 'train', cat=opt.cat, res=opt.res)
        val_dataset.initialize(opt, 'val', cat=opt.cat, res=opt.res)
        test_dataset.initialize(opt, 'test', cat=opt.cat, res=opt.res)

    elif opt.dataset_mode == 'stage3a':
        from datasets.stage3a_dataset import Stage3aDataset
        train_dataset = Stage3aDataset()
        val_dataset = Stage3aDataset()
        test_dataset = Stage3aDataset()
        train_dataset.initialize(opt, 'train', cat=opt.cat, res=opt.res)
        val_dataset.initialize(opt, 'val', cat=opt.cat, res=opt.res)
        test_dataset.initialize(opt, 'test', cat=opt.cat, res=opt.res)

    elif opt.dataset_mode == 'stage3b':
        from datasets.stage3b_dataset import Stage3bDataset
        train_dataset = Stage3bDataset()
        val_dataset = Stage3bDataset()
        test_dataset = Stage3bDataset()
        train_dataset.initialize(opt, 'train', cat=opt.cat, res=opt.res)
        val_dataset.initialize(opt, 'val', cat=opt.cat, res=opt.res)
        test_dataset.initialize(opt, 'test', cat=opt.cat, res=opt.res)

    elif opt.dataset_mode == 'bag3d':
        from datasets.bag3d_dataset import Bag3dDataset
        train_dataset = Bag3dDataset()
        val_dataset = Bag3dDataset()
        test_dataset = Bag3dDataset()
        train_dataset.initialize(opt, 'train', cat=opt.cat, res=opt.res)
        val_dataset.initialize(opt, 'val', cat=opt.cat, res=opt.res)
        test_dataset.initialize(opt, 'test', cat=opt.cat, res=opt.res)

    elif opt.dataset_mode == 'hybrid':
        from datasets.hybrid_dataset import HybridDataset
        train_dataset = HybridDataset()
        val_dataset = HybridDataset()
        test_dataset = HybridDataset()
        train_dataset.initialize(opt, 'train', cat=opt.cat, res=opt.res)
        val_dataset.initialize(opt, 'val', cat=opt.cat, res=opt.res)
        test_dataset.initialize(opt, 'test', cat=opt.cat, res=opt.res)

    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    cprint("[*] Dataset has been created: %s" % (train_dataset.name()), 'blue')
    return train_dataset, val_dataset, test_dataset
