
import numpy as np
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image

import torch
import torchvision.utils as vutils

from datasets.base_dataset import CreateDataset
from datasets.dataloader import CreateDataLoader, get_data_generator

from models.base_model import create_model

from utils.util import seed_everything

def tensor_to_pil(tensor):
    # """ assume shape: c h w """
    if tensor.dim() == 4:
        tensor = vutils.make_grid(tensor)

    return Image.fromarray( (rearrange(tensor, 'c h w -> h w c').cpu().numpy() * 255.).astype(np.uint8) )

############ START: all Opt classes ############

class BaseOpt(object):
    def __init__(self, gpu_ids=0, seed=None):
        # important args
        self.isTrain = False
        self.gpu_ids = [gpu_ids]
        # self.device = f'cuda:{gpu_ids}'
        self.device = 'cuda'
        self.debug = '0'

        # default args
        self.serial_batches = False
        self.nThreads = 4
        self.distributed = False

        # hyperparams
        self.batch_size = 1

        # dataset args
        self.max_dataset_size = 10000000
        self.trunc_thres = 0.2

        if seed is not None:
            seed_everything(seed)
            
        self.phase = 'test'

    def name(self):

        return 'BaseOpt'

class VQVAEOpt(BaseOpt):
    def __init__(self, gpu_ids=0, seed=None):
        super().__init__(gpu_ids)

        # some other custom args here

        print(f'[*] {self.name()} initialized.')

    def name(self):
        return 'VQVAETestOpt'

class SDFusionOpt(BaseOpt):
    def __init__(self, gpu_ids=0, seed=None):
        super().__init__(gpu_ids, seed=seed)

        # some other custom args here
        
        ## building net
        # opt.res = 128
        # opt.dataset_mode = 'buildingnet'
        # opt.cat = 'all'
        print(f'[*] {self.name()} initialized.')
        
    def init_dset_args(self, dataroot='data', dataset_mode='snet', cat='all', res=64, cached_dir=None):
        # dataset - snet
        self.dataroot = dataroot
        self.cached_dir = cached_dir
        self.ratio = 1.0
        self.res = res
        self.dataset_mode = dataset_mode
        self.cat = cat

    def init_model_args(
            self,
            ckpt_path='saved_ckpt/sdfusion-snet-all.pth',
            vq_ckpt_path='saved_ckpt/vqvae-snet-all.pth',
        ):
        self.model = 'sdfusion'
        self.df_cfg = 'configs/sdfusion_snet.yaml'
        self.ckpt = ckpt_path
        
        self.vq_model = 'vqvae'
        self.vq_cfg = 'configs/vqvae_snet.yaml'
        self.vq_ckpt = vq_ckpt_path
        self.vq_dset = 'snet'
        self.vq_cat = 'all'

    def name(self):
        return 'SDFusionTestOption'


class SDFusionText2ShapeOpt(BaseOpt):
    def __init__(self, gpu_ids=0, seed=None):
        super().__init__(gpu_ids, seed=seed)

        # some other custom args here
        print(f'[*] {self.name()} initialized.')
        
    def init_dset_args(self, dataroot='data', dataset_mode='text2shape', cat='all', res=64, cached_dir=None):
        # dataset - snet
        self.dataroot = dataroot
        self.cached_dir = cached_dir
        self.ratio = 1.0
        self.res = res
        self.dataset_mode = dataset_mode
        self.cat = cat
        
    def init_model_args(
            self,
            ckpt_path='saved_ckpt/sdfusion-txt2shape.pth',
            vq_ckpt_path='saved_ckpt/vqvae-snet-all.pth',
        ):
        self.model = 'sdfusion-txt2shape'
        self.df_cfg = 'configs/sdfusion-txt2shape.yaml'
        self.ckpt = ckpt_path
        
        self.vq_model = 'vqvae'
        self.vq_cfg = 'configs/vqvae_snet.yaml'
        self.vq_ckpt = vq_ckpt_path
        self.vq_dset = 'snet'
        self.vq_cat = 'all'

    def name(self):
        return 'SDFusionText2ShapeOption'

class SDFusionImage2ShapeOpt(BaseOpt):
    def __init__(self, gpu_ids=0, seed=None):
        super().__init__(gpu_ids, seed=seed)
        self.latent_size_HW = [16,16]
        self.latent_size_D = 16
        # some other custom args here
        print(f'[*] {self.name()} initialized.')
        
    def init_dset_args(self, dataroot='data', dataset_mode='pix3d_img2shape', cat='all', res=64, cached_dir=None):
        # dataset - snet
        self.dataroot = dataroot
        self.cached_dir = cached_dir
        self.ratio = 1.0
        self.res = res
        self.dataset_mode = dataset_mode
        self.cat = cat
        
    def init_model_args(
            self,
            #ckpt_path='saved_ckpt/sdfusion-img2shape.pth',
            ckpt_path = '/scratch/gilbreth/dsimhadr/GenerativeTowns/SDFusion/Logs_GT/2025-09-23T20-28-50-sdfusion_model_img2shape-building-LR1e-5/ckpt/df_steps-latest.pth',
            #vq_ckpt_path='saved_ckpt/vqvae-snet-all.pth',
            vq_ckpt_path = '/scratch/gilbreth/dsimhadr/GenerativeTowns/SDFusion/logs_building/2025-05-19T19-58-28-vqvae-building-all-res64-LR1e-4-T0.2-release/ckpt/vqvae_steps-latest.pth',
        ):
        #self.model = 'sdfusion-img2shape'
        self.model = 'sdfusion_model_img2shape'
        self.df_cfg = 'configs/sdfusion-img2shape.yaml'
        self.ckpt = ckpt_path
        
        self.vq_model = 'vqvae'
        self.vq_cfg = 'configs/vqvae_bnet.yaml'
        self.vq_ckpt = vq_ckpt_path
        self.vq_dset = 'snet'
        self.vq_cat = 'all'

    def name(self):
        return 'SDFusionImage2ShapeOption'


class SDFusionMM2ShapeOpt(BaseOpt):
    def __init__(self, gpu_ids=0, seed=None):
        super().__init__(gpu_ids, seed=seed)

        # some other custom args here
        print(f'[*] {self.name()} initialized.')
        
    def init_dset_args(self, dataroot='data', dataset_mode='mm2shape', cat='all', res=64, cached_dir=None):
        # dataset - snet
        self.dataroot = dataroot
        self.cached_dir = cached_dir
        self.ratio = 1.0
        self.res = res
        self.dataset_mode = dataset_mode
        self.cat = cat
        
    def init_model_args(
            self,
            #ckpt_path='saved_ckpt/sdfusion-mm2shape.pth',
            ckpt_path = '/scratch/gilbreth/dsimhadr/GenerativeTowns/SDFusion/Logs_GT/2025-09-23T20-28-50-sdfusion_model_img2shape-building-LR1e-5/ckpt/df_steps-latest.pth',
            #vq_ckpt_path='saved_ckpt/vqvae-snet-all.pth',
            vq_ckpt_path = '/scratch/gilbreth/dsimhadr/GenerativeTowns/SDFusion/logs_building/2025-05-19T19-58-28-vqvae-building-all-res64-LR1e-4-T0.2-release/ckpt/vqvae_steps-latest.pth',
        ):
        self.model = 'sdfusion_model_img2shape'
        self.df_cfg = 'configs/sdfusion-img2shape.yaml'
        self.ckpt = ckpt_path
        
        self.vq_model = 'vqvae'
        self.vq_cfg = 'configs/vqvae_bnet.yaml'
        self.vq_ckpt = vq_ckpt_path
        self.vq_dset = 'snet'
        self.vq_cat = 'all'

    def name(self):
        return 'SDFusionMM2ShapeOption'


############ END: all Opt classes ############

# get partial shape from range
def get_partial_shape(shape, xyz_dict, z=None):
    """
        args:  
            shape: input sdf. (B, 1, H, W, D)
            xyz_dict: user-specified range.
                x: left to right
                y: bottom to top
                z: front to back
    """
    x = shape
    device = x.device
    (x_min, x_max) = xyz_dict['x']
    (y_min, y_max) = xyz_dict['y']
    (z_min, z_max) = xyz_dict['z']
    
    # clamp to [-1, 1]
    x_min, x_max = max(-1, x_min), min(1, x_max)
    y_min, y_max = max(-1, y_min), min(1, y_max)
    z_min, z_max = max(-1, z_min), min(1, z_max)

    B, _, H, W, D = x.shape # assume D = H = W

    x_st = int( (x_min - (-1))/2 * H )
    x_ed = int( (x_max - (-1))/2 * H )
    
    y_st = int( (y_min - (-1))/2 * W )
    y_ed = int( (y_max - (-1))/2 * W )
    
    z_st = int( (z_min - (-1))/2 * D )
    z_ed = int( (z_max - (-1))/2 * D )
    
    # print('x: ', xyz_dict['x'], x_st, x_ed)
    # print('y: ', xyz_dict['y'], y_st, y_ed)
    # print('z: ', xyz_dict['z'], z_st, z_ed)

    # where to keep    
    x_mask = torch.ones(B, 1, H, W, D).bool().to(device)
    x_mask[:, :, :x_st, :, :] = False
    x_mask[:, :, x_ed:, :, :] = False
    
    x_mask[:, :, :, :y_st, :] = False
    x_mask[:, :, :, y_ed:, :] = False
    
    x_mask[:, :, :, :, :z_st] = False
    x_mask[:, :, :, :, z_ed:] = False
        
    shape_part = x.clone()
    shape_missing = x.clone()
    shape_part[~x_mask] = 0.2 # T-SDF
    shape_missing[x_mask] = 0.2
    
    ret = {
        'shape_part': shape_part,
        'shape_missing': shape_missing,
        'shape_mask': x_mask,
    }
    
    if z is not None:
        B, _, zH, zW, zD = z.shape # assume D = H = W

        x_st = int( (x_min - (-1))/2 * zH )
        x_ed = int( (x_max - (-1))/2 * zH )
        
        y_st = int( (y_min - (-1))/2 * zW )
        y_ed = int( (y_max - (-1))/2 * zW )
        
        z_st = int( (z_min - (-1))/2 * zD )
        z_ed = int( (z_max - (-1))/2 * zD )
        
        # where to keep    
        z_mask = torch.ones(B, 1, zH, zW, zD).to(device)
        z_mask[:, :, :x_st, :, :] = 0.
        z_mask[:, :, x_ed:, :, :] = 0.
        
        z_mask[:, :, :, :y_st, :] = 0.
        z_mask[:, :, :, y_ed:, :] = 0.
    
        z_mask[:, :, :, :, :z_st] = 0.
        z_mask[:, :, :, :, z_ed:] = 0.
        
        ret['z_mask'] = z_mask

    return ret

# for img2shape
# https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
def mask2bbox(mask):
    # mask: w x h
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # return rmin, rmax, cmin, cmax
    return cmin, rmin, cmax, rmax

# ref: pix2vox: https://github.com/hzxie/Pix2Vox/blob/f1b82823e79d4afeedddfadb3da0940bcf1c536d/utils/data_transforms.py
def crop_square(img, bbox, img_size_h=256, img_size_w=256):
    # from pix2vox
    img_height, img_width, c = img.shape

    x0, y0, x1, y1 = bbox

    # Calculate the size of bounding boxes
    bbox_width = x1 - x0
    bbox_height = y1 - y0
    bbox_x_mid = (x0 + x1) * .5
    bbox_y_mid = (y0 + y1) * .5

    # Make the crop area as a square
    square_object_size = max(bbox_width, bbox_height)
    x_left = int(bbox_x_mid - square_object_size * .5)
    x_right = int(bbox_x_mid + square_object_size * .5)
    y_top = int(bbox_y_mid - square_object_size * .5)
    y_bottom = int(bbox_y_mid + square_object_size * .5)

    # If the crop position is out of the image, fix it with padding
    pad_x_left = 0
    if x_left < 0:
        pad_x_left = -x_left
        x_left = 0
    pad_x_right = 0
    if x_right >= img_width:
        pad_x_right = x_right - img_width + 1
        x_right = img_width - 1
    pad_y_top = 0
    if y_top < 0:
        pad_y_top = -y_top
        y_top = 0
    pad_y_bottom = 0
    if y_bottom >= img_height:
        pad_y_bottom = y_bottom - img_height + 1
        y_bottom = img_height - 1

    # Padding the image and resize the image
    processed_image = np.pad(img[y_top:y_bottom + 1, x_left:x_right + 1],
                                ((pad_y_top, pad_y_bottom), (pad_x_left, pad_x_right), (0, 0)),
                                mode='edge')
    
    pil_img = Image.fromarray(processed_image)
    pil_img = pil_img.resize((img_size_w, img_size_h))
    # processed_image = cv2.resize(processed_image, (img_size_w, img_size_h))

    return pil_img

def preprocess_image(image, mask, *, fallback_bg=178, img_hw=256):
    """
    Prepare inputs for CLIP when you have a mask and (optionally) an image.

    Returns:
        img_comp (PIL.Image): composited or synthesized RGB, square crop @ img_hw
        img_clean (PIL.Image): masked-only RGB, square crop @ img_hw

    Behavior:
      - If image is None: create a neutral RGB canvas and render only the masked
        region (mask silhouette). This allows mask-only demos.
      - If image is array/str: behave like before (composite with softened bg).
    """
    # --- Load / normalize mask to uint8 {0,1} ---
    if isinstance(mask, str):
        mask_np = np.array(Image.open(mask).convert('1'), dtype=np.uint8)  # HxW {0,1}
    else:
        mask_np = mask.astype(np.uint8)

    # --- Load or synthesize image ---
    if image is None:
        # Neutral gray background
        # You can also choose white (255) or black (0). Gray is a good neutral.
        h, w = mask_np.shape[:2]
        img_np = np.full((h, w, 3), fallback_bg, dtype=np.uint8)
    else:
        img_np = np.array(Image.open(image).convert('RGB')) if isinstance(image, str) else image

    # --- Guard: empty mask -> use whole image ---
    if mask_np.max() == 0:
        # No foreground; create a 1-pixel bbox in the center
        h, w = mask_np.shape
        cy, cx = h // 2, w // 2
        mask_np[cy:cy+1, cx:cx+1] = 1

    # --- Get bbox from mask & crop square ---
    x0, y0, x1, y1 = mask2bbox(mask_np)
    bbox = [x0, y0, x1, y1]

    # --- Composite (when image exists) or keep neutral background (when None) ---
    # r controls how much background is pushed toward a flat color;
    # when image is None, img_np is already flat so this is fine too.
    r = 0.7
    img_comp_np = (
        img_np * mask_np[:, :, None] +
        (1 - mask_np[:, :, None]) * (r * 255 + (1 - r) * img_np)
    ).astype(np.uint8)

    img_clean_np = (img_np * mask_np[:, :, None]).astype(np.uint8)

    # --- Square crop to bbox and resize to CLIP-friendly size ---
    img_comp = crop_square(img_comp_np, bbox, img_size_h=img_hw, img_size_w=img_hw)
    img_clean = crop_square(img_clean_np, bbox, img_size_h=img_hw, img_size_w=img_hw)

    return img_comp, img_clean

# def preprocess_image(image, mask):
#     if type(image) is str:
#         img_np = np.array(Image.open(image).convert('RGB'))
#     else:
#         img_np = image
#     if type(mask) is str:
#         mask_np = np.array(Image.open(mask).convert('1'))
#     else:
#         mask_np = mask
        
#     # get bbox from mask
#     x0, y0, x1, y1 = mask2bbox(mask_np)
#     bbox = [x0, y0, x1, y1]
        
#     r = 0.7
#     img_comp = img_np * mask_np[:, :, None] + (1 - mask_np[:, :, None]) * (r*255 + (1 - r) * img_np)
#     img_comp = crop_square(img_comp.astype(np.uint8), bbox)
    
#     img_clean = img_np * mask_np[:, :, None]
#     img_clean = crop_square(img_clean.astype(np.uint8), bbox)
    
#     return img_comp, img_clean

def get_shape_mask(mask_mode, device='cuda'):

    B = 1
    D = 64
    d = 16
    
    x_mask = torch.ones(B, 1, D, D, D).bool().to(device)
    z_mask = torch.ones(B, 1, d, d, d).to(device)
    
    if mask_mode == 'bottom':
        # given bottom-half
        x_mask[:, :, :, D//2:, :] = False
        z_mask[:, :, :, d//2:, :] = 0
    
    elif mask_mode == 'top':
        # given top-half
        x_mask[:, :, :, :D//2, :] = False
        z_mask[:, :, :, :d//2, :] = 0

    elif mask_mode == 'back':
        # given back-half
        x_mask[:, :, :, :, D//2:] = False
        z_mask[:, :, :, :, d//2:] = 0

    elif mask_mode == 'front':
        # given front-half
        x_mask[:, :, :, :, :D//2] = False
        z_mask[:, :, :, :, :d//2] = 0

    return x_mask, z_mask
    