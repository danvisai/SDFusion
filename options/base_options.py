import os
import argparse

from termcolor import colored
from omegaconf import OmegaConf

import torch
from torch.utils.tensorboard import SummaryWriter

import utils

from utils.distributed import (
    get_rank,
    synchronize,
)

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        # hyper parameters
        self.parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        self.parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

        # log stuff
        self.parser.add_argument('--logs_dir', type=str, default='./logs', help='the root of the logs dir. All training logs are saved here')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')

        # dataset stuff
        self.parser.add_argument('--dataroot', type=str, default=None, help='root dir for data. if None, specify by `hostname` in configs/paths.py')
        self.parser.add_argument('--dataset_mode', type=str, default='snet', help='chooses how datasets are loaded. [mnist, snet, abc, snet-abc]')
        self.parser.add_argument('--res', type=int, default=64, help='dataset resolution')
        self.parser.add_argument('--cat', type=str, default='chair', help='category for shapenet')
        self.parser.add_argument('--trunc_thres', type=float, default=0.2, help='threshold for truncated sdf.')
        
        self.parser.add_argument('--ratio', type=float, default=1., help='ratio of the dataset to use. for debugging and overfitting')
        self.parser.add_argument('--max_dataset_size', default=2147483648, type=int, help='chooses the maximum dataset size.')
        self.parser.add_argument('--nThreads', default=9, type=int, help='# threads for loading data')        
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')

        ############## START: model related options ################
        self.parser.add_argument(
                            '--model', type=str, default='sdfusion',
                            choices=['vqvae', 'sdfusion', 'sdfusion-img2shape', 'sdfusion-txt2shape','sdfusion-mm2shape','sdfusion_model_img2shape'],
                            help='chooses which model to use.'
                        )
        self.parser.add_argument('--ckpt', type=str, default=None, help='ckpt to load.')

        # diffusion stuff
        self.parser.add_argument('--df_cfg', type=str, default='configs/sdfusion_snet.yaml', help="diffusion model's config file")
        self.parser.add_argument('--ddim_steps', type=int, default=100, help='steps for ddim sampler')
        self.parser.add_argument('--ddim_eta', type=float, default=0.0)
        self.parser.add_argument('--uc_scale', type=float, default=1.0, help='scale for un guidance')
        
        #new code
        # --- latent size (optional CLI overrides; can be omitted if YAML provides them) ---
        self.parser.add_argument(
            '--latent_size_D', type=int, default=None,
            help='Latent depth (D) for the 3D latent grid; if None, will be read from df_cfg or inferred from vq_cfg.'
        )
        self.parser.add_argument(
            '--latent_size_HW', type=int, nargs=2, default=None,
            help='Latent height and width (H W) for the 3D latent grid; if None, will be read from df_cfg or inferred from vq_cfg.'
        )


        # vqvae stuff
        self.parser.add_argument('--vq_model', type=str, default='vqvae', help='for choosing the vqvae model to use.')
        self.parser.add_argument('--vq_cfg', type=str, default='configs/vqvae_snet.yaml', help='vqvae model config file')
        self.parser.add_argument('--vq_dset', type=str, default=None, help='dataset vqvae originally trained on')
        self.parser.add_argument('--vq_cat', type=str, default=None, help='dataset category vqvae originally trained on')
        self.parser.add_argument('--vq_ckpt', type=str, default=None, help='vqvae ckpt to load.')
        ############## END: model related options ################

        # misc
        self.parser.add_argument('--debug', default='0', type=str, choices=['0', '1'], help='if true, debug mode')
        self.parser.add_argument('--seed', default=111, type=int, help='seed')

        # multi-gpu stuff
        self.parser.add_argument("--backend", type=str, default="gloo", help="which backend to use")
        self.parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")

        self.initialized = True

    def parse_and_setup(self):
        import sys
        cmd = ' '.join(sys.argv)
        print(f'python {cmd}')

        if not self.initialized:
            self.initialize()

        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        if self.opt.isTrain:
            self.opt.phase = 'train'
        else:
            self.opt.phase = 'test'

        # setup multi-gpu stuffs here
        # basically from stylegan2-pytorch, train.py by rosinality
        self.opt.device = 'cuda'
        n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        self.opt.distributed = n_gpu > 1

        if self.opt.distributed:
            torch.cuda.set_device(self.opt.local_rank)
            torch.distributed.init_process_group(backend=self.opt.backend, init_method="env://")
            synchronize()

        name = self.opt.name
        if self.opt.isTrain and self.opt.ckpt is not None:
            name = f'continue-{name}'

        self.opt.name = name
        
        self.opt.gpu_ids_str = self.opt.gpu_ids

        # NOTE: seed or not?
        # seed = opt.seed
        # util.seed_everything(seed)

        self.opt.rank = get_rank()

        #new code
        def _infer_latent_from_vqcfg(vq_cfg_path):
            try:
                vq_yaml = OmegaConf.load(vq_cfg_path)
                dd = vq_yaml.model.params.ddconfig
                base_res = int(dd.resolution)          # e.g., 64, 128, 256
                ch_mult  = list(dd.ch_mult)            # e.g., [1, 2, 4] or [1, 2, 4, 4]

                # Two common heuristics for downsample factor
                factor_a = 2 ** max(len(ch_mult) - 1, 0)
                factor_b = 2 ** max(len(ch_mult), 0)

                # Prefer f=4 if divisible (common "f4" latent noted in your cfg comments)
                candidates = []
                if base_res % 4 == 0:
                    candidates.append(4)
                candidates.extend([factor_a, factor_b])

                for f in candidates:
                    if f > 0 and base_res % f == 0:
                        side = base_res // f
                        if side >= 1:
                            return int(side), int(side), int(side)
            except Exception as e:
                print(colored(f"[warn] Could not infer latent size from vq_cfg ({vq_cfg_path}): {e}", "yellow"))

            print(colored("[warn] Falling back to latent 64^3", "yellow"))
            return 64, 64, 64

        # Load df_cfg YAML if possible
        try:
            df_yaml = OmegaConf.load(self.opt.df_cfg)
        except Exception as e:
            df_yaml = OmegaConf.create()
            print(colored(f"[warn] Could not load df_cfg ({self.opt.df_cfg}): {e}", "yellow"))

        # CLI overrides (already parsed) take precedence; if not provided, try YAML
        if getattr(self.opt, 'latent_size_HW', None) is None:
            if 'latent_size_HW' in df_yaml:
                hw = df_yaml.latent_size_HW
                self.opt.latent_size_HW = (int(hw[0]), int(hw[1]))
            else:
                self.opt.latent_size_HW = None

        if getattr(self.opt, 'latent_size_D', None) is None:
            if 'latent_size_D' in df_yaml:
                self.opt.latent_size_D = int(df_yaml.latent_size_D)
            else:
                self.opt.latent_size_D = None

        # If still missing, infer from VQ-VAE config
        if self.opt.latent_size_HW is None or self.opt.latent_size_D is None:
            D, H, W = _infer_latent_from_vqcfg(self.opt.vq_cfg)
            if self.opt.latent_size_HW is None:
                self.opt.latent_size_HW = (H, W)
            if self.opt.latent_size_D is None:
                self.opt.latent_size_D = D

        # Normalize types
        self.opt.latent_size_HW = (int(self.opt.latent_size_HW[0]), int(self.opt.latent_size_HW[1]))
        self.opt.latent_size_D  = int(self.opt.latent_size_D)
        # --------------------------------------------------------------------



        if get_rank() == 0:
            # print args
            args = vars(self.opt)

            print('------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')

            # make experiment dir
            if self.opt.isTrain:
                expr_dir = os.path.join(self.opt.logs_dir, self.opt.name)
                utils.util.mkdirs(expr_dir)
                
                ckpt_dir = os.path.join(self.opt.logs_dir, self.opt.name, 'ckpt')
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                self.opt.ckpt_dir = ckpt_dir
                    
                file_name = os.path.join(expr_dir, 'opt.txt')
                with open(file_name, 'wt') as opt_file:
                    opt_file.write('------------ Options -------------\n')
                    for k, v in sorted(args.items()):
                        opt_file.write('%s: %s\n' % (str(k), str(v)))
                    opt_file.write('-------------- End ----------------\n')
                
                # tensorboard writer
                tb_dir = '%s/tboard' % expr_dir
                if not os.path.exists(tb_dir):
                    os.makedirs(tb_dir)
                self.opt.tb_dir = tb_dir
                writer = SummaryWriter(log_dir=tb_dir)
                self.opt.writer = writer

        return self.opt
