import os
from collections import OrderedDict
from functools import partial
from utils.util import tensor2im

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from omegaconf import OmegaConf
from termcolor import colored, cprint
from tqdm import tqdm


from models.base_model import BaseModel
from models.model_utils import load_vqvae
from models.networks.diffusion_networks.network import DiffusionUNet
from models.networks.clip_networks.network import CLIPImageEncoder
from models.networks.diffusion_networks.samplers.ddim import DDIMSampler
from models.networks.diffusion_networks.ldm_diffusion_util import (
    make_beta_schedule, extract_into_tensor, default, exists
)
from utils.distributed import reduce_loss_dict
from utils.util_3d import init_mesh_renderer, render_sdf

class SDFusionImageFPShapeModel(BaseModel):
    """
    Diffusion-based 3D SDF generator conditioned on both image and footprint masks.
    """
    def name(self):
        return 'SDFusionImageFPShapeModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.device = opt.device

        # load configs
        df_conf = OmegaConf.load(opt.df_cfg)
        vq_conf = OmegaConf.load(opt.vq_cfg)

        # build diffusion network
        unet_params = df_conf.unet.params
        use_ckpt = bool(unet_params.get("use_checkpoint", False))
        self.df = DiffusionUNet(unet_params, vq_conf=vq_conf,
                                conditioning_key=df_conf.model.params.conditioning_key)
        self.df.to(self.device)
        self.init_diffusion_params(opt)


        self.ddim_steps = getattr(opt, "ddim_steps", 100)
        if opt.debug == "1":
            self.ddim_steps = min(self.ddim_steps, 20)
        # sampler
        self.ddim_sampler = DDIMSampler(self)

        # load pretrained VQ-VAE
        self.vqvae = load_vqvae(vq_conf, vq_ckpt=opt.vq_ckpt, opt=opt)

        

        # image encoder
        clip_param = df_conf.clip.params
        self.img_encoder = CLIPImageEncoder(model=clip_param.model)
        self.img_encoder.to(self.device)
        # footprint encoder (same backbone)
        self.fp_encoder = CLIPImageEncoder(model=clip_param.model)
        self.fp_encoder.to(self.device)

        # set up optimizer
        if self.isTrain:
            train_params = [p for p in self.df.parameters() if p.requires_grad]
            train_params += [p for p in self.img_encoder.parameters() if p.requires_grad]
            train_params += [p for p in self.fp_encoder.parameters() if p.requires_grad]
            self.optimizer = optim.AdamW(train_params, lr=opt.lr)
            self.schedulers = [optim.lr_scheduler.StepLR(self.optimizer, 1000, 0.9)]
            self.optimizers = [self.optimizer]

        # optionally load checkpoint
        if opt.ckpt:
            self.load_ckpt(opt.ckpt, load_opt=self.isTrain)

        # # renderer only for inference/visuals
        # if not self.isTrain:
        #     dist, elev, azim = 1.7, 20, 20
        #     self.renderer = init_mesh_renderer(image_size=256,
        #                                       dist=dist, elev=elev, azim=azim,
        #                                       device=self.device)
        # else:
        #     self.renderer = None

        # always set up a renderer so we can save train-time visuals
        dist, elev, azim = 1.7, 20, 20
        self.renderer = init_mesh_renderer(
            image_size=256, dist=dist, elev=elev, azim=azim,
            device=self.device
        )

        cprint(f"[*] SDFusionImageFPShapeModel initialized (train={self.isTrain}).", 'cyan')

    def init_diffusion_params(self, opt):
        df_conf = OmegaConf.load(opt.df_cfg)
        dp = df_conf.model.params
        self.parameterization = 'eps'
        self.learn_logvar = False
        self.register_schedule(given_betas=None, beta_schedule='linear',
                               timesteps=dp.timesteps,
                               linear_start=dp.linear_start,
                               linear_end=dp.linear_end)
        self.logvar = torch.zeros(self.num_timesteps, device=self.device)
        self.l_simple_weight = 1.
        self.original_elbo_weight = 0.
        self.uc_scale = 1.

    def register_schedule(self, given_betas, beta_schedule, timesteps,
                          linear_start, linear_end, cosine_s=8e-3):
        betas = given_betas or make_beta_schedule(beta_schedule, timesteps,
                                                  linear_start=linear_start,
                                                  linear_end=linear_end,
                                                  cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        prev = np.append(1., alphas_cumprod[:-1])
        to_t = lambda arr: torch.tensor(arr, dtype=torch.float32, device=self.device)
        self.betas = to_t(betas)
        self.alphas_cumprod = to_t(alphas_cumprod)
        self.alphas_cumprod_prev = to_t(prev)
        self.sqrt_alphas_cumprod = to_t(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_t(np.sqrt(1 - alphas_cumprod))
        self.posterior_variance = to_t((1 - 0.) * betas * (1 - prev) / (1 - alphas_cumprod))
        self.posterior_log_variance_clipped = to_t(np.log(np.maximum(self.posterior_variance.cpu().numpy(), 1e-20)))
        self.posterior_mean_coef1 = to_t(betas * np.sqrt(prev) / (1 - alphas_cumprod))
        self.posterior_mean_coef2 = to_t((1 - prev) * np.sqrt(alphas) / (1 - alphas_cumprod))
        lvlb = betas**2 / (2 * self.posterior_variance.cpu().numpy() * alphas * (1 - alphas_cumprod))
        lvlb[0] = lvlb[1]
        self.lvlb_weights = to_t(lvlb)
        self.num_timesteps = timesteps

    def set_input(self, input, max_sample=None):
        self.x   = input['sdf']
        self.img = input['img']
        self.fp  = input['fp']
        self.uc_img = torch.zeros_like(self.img)
        self.uc_fp  = torch.zeros_like(self.fp)
        if max_sample:
            self.x   = self.x[:max_sample]
            self.img = self.img[:max_sample]
            self.fp  = self.fp[:max_sample]
            self.uc_img = self.uc_img[:max_sample]
            self.uc_fp  = self.uc_fp[:max_sample]
        self.x, self.img, self.fp, self.uc_img, self.uc_fp = (
            t.to(self.device) for t in (self.x, self.img, self.fp, self.uc_img, self.uc_fp)
        )

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    # def apply_model(self, x_noisy, t, cond):
    #     key = 'c_concat' if self.df.conditioning_key=='concat' else 'c_crossattn'
    #     cond = {key: cond} if not isinstance(cond, dict) else cond
    #     out = self.df(x_noisy, t, **cond)
    #     return out[0] if isinstance(out, tuple) else out
    
    def apply_model(self, x_noisy, t, cond):
        key = 'c_concat' if self.df.conditioning_key=='concat' else 'c_crossattn'
        if isinstance(cond, dict):
        # already a dict of conditioning
            pass
        else:
        # make sure cond is a list of tensors
            if not isinstance(cond, list):
                cond = [cond]
            cond = {key: cond}
        out = self.df(x_noisy, t, **cond)
        return out[0] if isinstance(out, tuple) else out


    # def p_losses(self, x_start, cond_list, t, noise=None):
    #     noise = noise or torch.randn_like(x_start)
    #     x_noisy = self.q_sample(x_start, t, noise)
    #     model_out = self.apply_model(x_noisy, t, cond_list)
    #     target = noise if self.parameterization=='eps' else x_start
    #     loss_simple = F.mse_loss(model_out, target, reduction='none').mean([1,2,3,4])
    #     loss = (loss_simple / torch.exp(self.logvar[t]) + self.logvar[t]).mean()
    #     return x_noisy, target, loss, {'loss_total': loss}
    
    def p_losses(self, x_start, cond, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        model_out = self.apply_model(x_noisy, t, cond)
        target = noise if self.parameterization == 'eps' else x_start
        loss_simple = F.mse_loss(model_out, target, reduction='none').mean([1,2,3,4])
        loss = (loss_simple / torch.exp(self.logvar[t]) + self.logvar[t]).mean()
        return x_noisy, target, loss, {'loss_total': loss}

    # def forward(self):
    #     c_img = self.img_encoder(self.img).float()
    #     c_fp  = self.fp_encoder(self.fp).float()
    #     with torch.no_grad():
    #         z = self.vqvae(self.x, forward_no_quant=True, encode_only=True).detach()
    #     t = torch.randint(0, self.num_timesteps, (z.shape[0],), device=self.device)
    #     _, _, loss, loss_dict = self.p_losses(z, [c_img, c_fp], t)
    #     self.loss_df   = loss
    #     self.loss_dict = loss_dict

    def forward(self):
       # 1) encode image & footprint to two CLIP vectors
        c_img = self.img_encoder(self.img).float()
        c_fp  = self.fp_encoder(self.fp).float()
       # 2) fuse by summation
        cond = c_img + c_fp
        # 3) encode SDF to latent z (no quant)
        with torch.no_grad():
            z = self.vqvae(self.x, forward_no_quant=True, encode_only=True).detach()
                     # 4) sample random timestep
        t = torch.randint(0, self.num_timesteps, (z.shape[0],), device=self.device)
                # 5) compute diffusion loss under fused condition
        _, _, loss, loss_dict = self.p_losses(z, cond, t)
        self.loss_df   = loss
        self.loss_dict = loss_dict

    def switch_train(self):
        # put all trainable nets into train() mode
        self.df.train()
        self.vqvae.train()
        self.img_encoder.train()
        self.fp_encoder.train()

    def backward(self):
        self.loss = self.loss_df
        self.loss.backward()
    
    def switch_eval(self):
        # put everything into eval() mode for inference/visuals
        self.df.eval()
        self.vqvae.eval()
        self.img_encoder.eval()
        self.fp_encoder.eval()

    def optimize_parameters(self, total_steps):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    @torch.no_grad()
    def eval_metrics(self, dataloader, thres=0.0, global_step=0):
        self.switch_eval()
        
        ret = OrderedDict([
            ('dummy_metrics', 0.0),
        ])

        self.switch_train()
        return ret

    # @torch.no_grad()
    # def inference(self, data, ddim_steps=None, ddim_eta=0., uc_scale=None, infer_all=False, max_sample=16):
    #     self.switch_eval()
    #     self.set_input(data)
    #     uc_img = self.img_encoder(self.uc_img).float()
    #     uc_fp  = self.fp_encoder(self.uc_fp).float()
    #     c_img  = self.img_encoder(self.img).float()
    #     c_fp   = self.fp_encoder(self.fp).float()
       
    #     uc = torch.cat([uc_img, uc_fp], dim=1)
    #     cond = torch.cat([c_img, c_fp], dim=1)

    #     B = c_img.shape[0]
    #     shape = (self.vqvae.ddconfig.z_channels, ) + tuple(self.vqvae.ddconfig.resolution // (2**len(self.vqvae.ddconfig.ch_mult)) for _ in range(3))
    #     samples, _ = self.ddim_sampler.sample(S=ddim_steps or self.ddim_steps,
    #                                         batch_size=B,
    #                                         shape=shape,
    #                                         conditioning=cond,
    #                                         unconditional_guidance_scale=uc_scale or self.uc_scale,
    #                                         unconditional_conditioning=uc,
    #                                         eta=ddim_eta,
    #                                         quantize_x0=False)
    #     self.gen_df = self.vqvae.decode_no_quant(samples)
    #     return self.gen_df


    @torch.no_grad()
    def inference(self, data, ddim_steps=None, ddim_eta=0., uc_scale=None, infer_all=False, max_sample=16):
        self.switch_eval()
        self.set_input(data)
        # encode zero‐conditions and true conditions
        uc_img = self.img_encoder(self.uc_img).float()
        uc_fp  = self.fp_encoder(self.uc_fp).float()
        c_img  = self.img_encoder(self.img).float()
        c_fp   = self.fp_encoder(self.fp).float()

        # fuse by summation
        uc   = uc_img + uc_fp
        cond = c_img  + c_fp

        B = cond.shape[0]
        shape = (self.vqvae.ddconfig.z_channels, ) + tuple(
            self.vqvae.ddconfig.resolution // (2**len(self.vqvae.ddconfig.ch_mult))
            for _ in range(3)
        )
        samples, _ = self.ddim_sampler.sample(
            S=ddim_steps or self.ddim_steps,
            batch_size=B,
            shape=shape,
            conditioning=cond,
            unconditional_guidance_scale=uc_scale or self.uc_scale,
            unconditional_conditioning=uc,
            eta=ddim_eta,
            quantize_x0=False,
        )
        self.gen_df = self.vqvae.decode_no_quant(samples)
        return self.gen_df

    # def get_current_visuals(self):
    #     if self.renderer is None:
    #         return {}
    #     ims = {
    #         'img': self.img,
    #         'gt': render_sdf(self.renderer, self.x),
    #         'gen': render_sdf(self.renderer, self.gen_df)
    #     }
    #     return OrderedDict((k, self.tnsrs2ims([k])[0]) for k in ims)
    
  

    @torch.no_grad()
    def get_current_visuals(self):
        if self.renderer is None:
            return {}
        # render SDFs
        gt_t = render_sdf(self.renderer, self.x)
        gen_t = render_sdf(self.renderer, self.gen_df)

        # convert to H×W×3 uint8 arrays
        img_im = tensor2im(self.img.data)
        gt_im  = tensor2im(gt_t .data)
        gen_im = tensor2im(gen_t.data)

        return OrderedDict([
        ('img', img_im),
        ('gt',  gt_im),
        ('gen', gen_im),
        ])

    def save(self, label, global_step, save_opt=False):
        state = {'vqvae': self.vqvae.state_dict(),
                 'df': self.df.state_dict(),
                 'img_enc': self.img_encoder.state_dict(),
                 'fp_enc': self.fp_encoder.state_dict(),
                 'global_step': global_step}
        if save_opt:
            state['opt'] = self.optimizer.state_dict()
        torch.save(state, os.path.join(self.opt.ckpt_dir, f'df_{label}.pth'))

    def load_ckpt(self, ckpt, load_opt=False):
    # allow passing in state‐dict or path
        map_fn = lambda storage, loc: storage
        state_dict = torch.load(ckpt, map_location=map_fn) if isinstance(ckpt, str) else ckpt

        self.vqvae.load_state_dict(state_dict['vqvae'])
        self.df.load_state_dict(state_dict['df'])
        self.img_encoder.load_state_dict(state_dict['img_enc'])
        self.fp_encoder.load_state_dict(state_dict['fp_enc'])
        print(colored(f"[*] weights loaded from {ckpt}", 'blue'))

        if load_opt and 'opt' in state_dict:
            self.optimizer.load_state_dict(state_dict['opt'])
            print(colored(f"[*] optimizer state loaded from {ckpt}", 'blue'))
        elif load_opt:
            print(colored("[!] optimizer state not found in checkpoint.", 'yellow'))
