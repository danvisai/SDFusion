"""Stage 3a — Conditional SDF latent diffusion for BuildingNet.

Conditioning: (footprint, class, height, style).
    - footprint: dual path. Spatial via _build_fp3d_for -> c_concat (1 ch).
      Global via frozen FootprintEmbedNet -> 256-d emb to crossattn context.
    - class:  nn.Embedding(53, 32).
    - style:  nn.Embedding(9, 16)   (8 recipes + 1 "unknown").
    - height: MLP(1 -> 32).
    Global vector (256+32+16+32 = 336) is projected to context_dim=512 and
    fed to the UNet's spatial transformer as c_crossattn.

Backbone: existing DiffusionUNet (3D) with conditioning_key='hybrid'. The UNet
expects in_channels = (latent_z_channels + spatial_concat_channels) = 3 + 1.

Frozen components:
    - VQVAE  (encodes GT SDF -> latent z target for diffusion supervision;
      decodes back to SDF when computing the guardrail aux loss).
    - FootprintEmbedNet (the retrieval network from Logs_GT/retrieval_footprint_full).

Optional guardrail (every K=50 steps): decode current eps prediction back to
SDF and add `surface_band_smooth_l1 + soft_footprint_bce` against the GT SDF.
Code reused from train_sdf_residual.py:43-62.
"""
from __future__ import annotations
import math
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import OmegaConf
from termcolor import colored, cprint

from models.base_model import BaseModel
from models.model_utils import load_vqvae
from models.networks.diffusion_networks.network import DiffusionUNet
from models.networks.diffusion_networks.samplers.ddim import DDIMSampler
from models.networks.diffusion_networks.ldm_diffusion_util import (
    extract_into_tensor, make_beta_schedule,
)
from models.networks.retrieval.footprint_embed import FootprintEmbedNet
from utils.distributed import reduce_loss_dict
from utils.util_3d import init_mesh_renderer, render_sdf


# --- Aux losses (verbatim from train_sdf_residual.py:43-62) -----------------

def _soft_inside(sdf: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.sigmoid(-sdf / max(tau, 1e-6))


def _surface_band_smooth_l1(corrected, target, sigma: float, beta: float = 0.1):
    band = torch.exp(-target.abs() / max(sigma, 1e-6))
    per_voxel = F.smooth_l1_loss(corrected, target, reduction="none", beta=beta)
    return (band * per_voxel).sum() / band.sum().clamp_min(1e-8)


def _soft_footprint_bce(corrected, target, tau: float):
    # (B, C, D, H, W); H = Y (vertical).
    p = _soft_inside(corrected, tau).amax(dim=3).clamp(1e-6, 1.0 - 1e-6)
    t = (target <= 0).any(dim=3).float()
    return F.binary_cross_entropy(p, t)


# ---------------------------------------------------------------------------

class Stage3aModel(BaseModel):
    """Conditional SDF latent diffusion. See module docstring."""

    def name(self):
        return "Stage3aSDFDiffusion"

    # -- init -----------------------------------------------------------

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.device = opt.device

        # configs
        self.df_conf = OmegaConf.load(opt.df_cfg)
        self.vq_conf = OmegaConf.load(opt.vq_cfg)
        s3 = self.df_conf.get("stage3a", {})
        gd = self.df_conf.get("guardrail", {})

        # backbone diffusion UNet
        unet_params = self.df_conf.unet.params
        # adaLN (gated, gap #1 fix): inject the global cond vector into the time embedding so it
        # modulates every ResBlock — the cross-attn-only path is ignored (style-collapse audit).
        self.use_adaln = bool(getattr(opt, "use_adaln", s3.get("use_adaln", False)))
        # Layer-A context conditioning (gated): concat (known_body, edit_mask, primitive) latent
        # channels to the UNet input so an added mass integrates coherently with the existing body.
        # Extends in_channels by 3; load_ckpt copies the old input-conv into the first channels and
        # ZERO-inits the new ones -> a warm-started ckpt begins identical to the base prior.
        self.use_context = bool(getattr(opt, "use_context", s3.get("use_context", False)))
        self.n_ctx_ch = 3
        self._base_in_ch = int(unet_params.in_channels)
        if self.use_context:
            unet_params = OmegaConf.create(OmegaConf.to_container(unet_params, resolve=True))
            unet_params.in_channels = self._base_in_ch + self.n_ctx_ch
        self._ctx_rng = np.random.default_rng(int(getattr(opt, "seed", 0)) + 7)
        self.df = DiffusionUNet(
            unet_params, vq_conf=self.vq_conf,
            conditioning_key=self.df_conf.model.params.conditioning_key,
            adaln_context_dim=(int(unet_params.context_dim) if self.use_adaln else None),
        )
        self.df.to(self.device)
        self._init_diffusion_params()

        # DDIM sampler (used at inference time)
        self.ddim_steps = getattr(opt, "ddim_steps", 100)
        if opt.debug == "1":
            self.ddim_steps = min(self.ddim_steps, 20)
        self.ddim_sampler = DDIMSampler(self)

        # VQVAE (frozen)
        self.vqvae = load_vqvae(self.vq_conf, vq_ckpt=opt.vq_ckpt, opt=opt)

        # FootprintEmbedNet (frozen, gives global footprint embedding)
        self.fp_emb_dim = int(s3.get("fp_emb_dim", 256))
        self.num_classes = int(s3.get("num_classes", 53))
        self.num_styles = int(s3.get("num_styles_plus_unknown", 9))
        self.class_emb_dim = int(s3.get("class_emb_dim", 32))
        self.style_emb_dim = int(s3.get("style_emb_dim", 16))
        self.height_emb_dim = int(s3.get("height_emb_dim", 32))
        self.context_dim = int(s3.get("context_dim", 512))
        self.fp3d_concat_scale = float(s3.get("fp3d_concat_scale", 0.5))
        self.freeze_fp_encoder = bool(s3.get("freeze_fp_encoder", True))

        self.fp_encoder = FootprintEmbedNet(
            num_classes=self.num_classes,
            embedding_dim=self.fp_emb_dim,
        )
        fp_ckpt_path = s3.get("fp_emb_ckpt", None)
        if fp_ckpt_path and os.path.exists(fp_ckpt_path):
            state = torch.load(fp_ckpt_path, map_location="cpu")
            sd = state.get("model", state)
            try:
                self.fp_encoder.load_state_dict(sd, strict=False)
                cprint(f"[*] FootprintEmbedNet loaded from {fp_ckpt_path}", "blue")
            except Exception as exc:
                cprint(f"[!] FootprintEmbedNet load failed ({exc}); using random init.", "yellow")
        else:
            cprint(f"[!] FootprintEmbedNet ckpt not found at {fp_ckpt_path}; using random init.", "yellow")
        if self.freeze_fp_encoder:
            for p in self.fp_encoder.parameters():
                p.requires_grad = False
            self.fp_encoder.eval()
        self.fp_encoder.to(self.device)

        # Conditioning embeddings.
        self.class_emb = nn.Embedding(self.num_classes, self.class_emb_dim).to(self.device)
        self.style_emb = nn.Embedding(self.num_styles, self.style_emb_dim).to(self.device)
        self.height_mlp = nn.Sequential(
            nn.Linear(1, self.height_emb_dim),
            nn.SiLU(),
            nn.Linear(self.height_emb_dim, self.height_emb_dim),
        ).to(self.device)
        # Hybrid extra conditioning (gap #1 revival): BAG era + floors. GATED so the OLD
        # prior ckpt (snap demo) still loads when use_extra_cond is off (the default).
        self.use_extra_cond = bool(getattr(opt, "use_extra_cond", s3.get("use_extra_cond", False)))
        self.extra_emb_dim = int(s3.get("extra_emb_dim", 16))
        self.p_uncond = float(getattr(opt, "p_uncond", s3.get("p_uncond", 0.0)))
        if self.use_extra_cond:
            self.era_emb = nn.Embedding(6, self.extra_emb_dim).to(self.device)     # 0..4 + unknown(5)
            self.floors_emb = nn.Embedding(5, self.extra_emb_dim).to(self.device)  # 4 buckets + unknown(4)
        # Region/culture token (cross-cultural corpus). GATED like use_extra_cond.
        self.use_region = bool(getattr(opt, "use_region", s3.get("use_region", False)))
        self.num_regions = int(s3.get("num_regions", 4))                            # 0=NL 1=DE 2=JP, 3=unknown
        self.region_emb_dim = int(s3.get("region_emb_dim", 16))
        if self.use_region:
            self.region_emb = nn.Embedding(self.num_regions, self.region_emb_dim).to(self.device)
        # Layer-B element-type token (window/door/balcony/pilaster/bay), classified from the
        # Layer-A primitive's shape (see _classify_element_type). GATED like use_region.
        self.use_element_type = bool(getattr(opt, "use_element_type", s3.get("use_element_type", False)))
        self.num_element_types = int(s3.get("num_element_types", 6))
        self.element_type_emb_dim = int(s3.get("element_type_emb_dim", 16))
        if self.use_element_type:
            self.element_type_emb = nn.Embedding(self.num_element_types, self.element_type_emb_dim).to(self.device)
        total_global = (
            self.fp_emb_dim + self.class_emb_dim
            + self.style_emb_dim + self.height_emb_dim
            + (2 * self.extra_emb_dim if self.use_extra_cond else 0)
            + (self.region_emb_dim if self.use_region else 0)
            + (self.element_type_emb_dim if self.use_element_type else 0)
        )
        self.global_proj = nn.Sequential(
            nn.Linear(total_global, self.context_dim),
            nn.SiLU(),
            nn.Linear(self.context_dim, self.context_dim),
        ).to(self.device)

        # Guardrail config.
        self.gd_enabled = bool(gd.get("enabled", False))
        self.gd_every = int(gd.get("every_k_steps", 50))
        self.gd_band_weight = float(gd.get("band_weight", 0.5))
        self.gd_fp_weight = float(gd.get("fp_weight", 0.25))
        self.gd_band_sigma = float(gd.get("band_sigma", 0.05))
        self.gd_fp_tau = float(gd.get("fp_tau", 0.05))

        # REPA (training-gaps step 4, arXiv 2410.06940) — GATED, training-only: align the
        # UNet middle_block features with DINOv2 features of depth renders of the clean SDF.
        self.use_repa = bool(getattr(opt, "use_repa", False)) and self.isTrain
        if self.use_repa:
            from models.networks.repa import RepaAlign
            up = self.df_conf.unet.params
            mid_ch = int(up.model_channels) * int(list(up.channel_mult)[-1])
            self.repa = RepaAlign(mid_ch, device=self.device)
            self.repa_weight = float(getattr(opt, "repa_weight", 0.5))
            self.repa_stop_iter = int(getattr(opt, "repa_stop_iter", 0) or 10 ** 9)
            self._repa_feat = None
            self.df.diffusion_net.middle_block.register_forward_hook(
                lambda _m, _i, out: setattr(self, "_repa_feat", out))
            cprint(f"[*] REPA on: mid_ch={mid_ch} w={self.repa_weight} stop@{self.repa_stop_iter}", "blue")

        # Optimizer: AdamW + linear warmup + cosine.
        if self.isTrain:
            train_params = (
                list(self.df.parameters())
                + list(self.class_emb.parameters())
                + list(self.style_emb.parameters())
                + list(self.height_mlp.parameters())
                + list(self.global_proj.parameters())
            )
            if self.use_extra_cond:
                train_params += list(self.era_emb.parameters()) + list(self.floors_emb.parameters())
            if self.use_region:
                train_params += list(self.region_emb.parameters())
            if self.use_element_type:
                train_params += list(self.element_type_emb.parameters())
            if self.use_repa:
                train_params += list(self.repa.proj.parameters())
            if not self.freeze_fp_encoder:
                train_params += list(self.fp_encoder.parameters())
            self.optimizer = optim.AdamW(
                train_params, lr=opt.lr,
                betas=(0.9, 0.999), weight_decay=1e-4,
            )
            warmup = max(int(getattr(opt, "warmup_steps", 1000)), 1)
            total = max(int(getattr(opt, "cosine_total_steps", 150000)),
                        warmup + 1)
            def _lr_lambda(step: int) -> float:
                if step < warmup:
                    return float(step + 1) / float(warmup)
                p = (step - warmup) / max(total - warmup, 1)
                return 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, _lr_lambda)
            self.optimizers = [self.optimizer]
            self.schedulers = [self.scheduler]

            # EMA of the diffusion UNet (gap #5) — softer/cleaner samples.
            self.use_ema = bool(getattr(opt, "use_ema", False))
            self.ema_decay = float(getattr(opt, "ema_decay", 0.999))
            if self.use_ema:
                import copy
                self.ema_df = copy.deepcopy(self.df).eval()
                for p in self.ema_df.parameters():
                    p.requires_grad_(False)

        if opt.ckpt:
            self.load_ckpt(opt.ckpt, load_opt=self.isTrain)

        # Renderer for visuals.
        self.renderer = init_mesh_renderer(
            image_size=256, dist=1.7, elev=20, azim=20, device=self.device,
        )
        cprint(f"[*] Stage3aModel initialized (train={self.isTrain}).", "cyan")

        # Step counter (independent of trainer's total_steps for guardrail gating).
        self._step = 0

    # -- diffusion schedule --------------------------------------------

    def _init_diffusion_params(self):
        dp = self.df_conf.model.params
        self.parameterization = "eps"
        self.learn_logvar = False
        self.register_schedule(
            given_betas=None, beta_schedule="linear",
            timesteps=dp.timesteps,
            linear_start=dp.linear_start, linear_end=dp.linear_end,
        )
        self.logvar = torch.zeros(self.num_timesteps, device=self.device)
        self.l_simple_weight = 1.0
        self.original_elbo_weight = 0.0
        self.uc_scale = 1.0
        self.scale_factor = float(dp.get("scale_factor", 1.0))
        cprint(f"[*] diffusion scale_factor = {self.scale_factor}", "yellow")

    def register_schedule(self, given_betas, beta_schedule, timesteps,
                          linear_start, linear_end, cosine_s=8e-3):
        betas = given_betas or make_beta_schedule(
            beta_schedule, timesteps,
            linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s,
        )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        prev = np.append(1.0, alphas_cumprod[:-1])
        to_t = lambda arr: torch.tensor(arr, dtype=torch.float32, device=self.device)
        self.betas = to_t(betas)
        self.alphas_cumprod = to_t(alphas_cumprod)
        self.alphas_cumprod_prev = to_t(prev)
        self.sqrt_alphas_cumprod = to_t(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_t(np.sqrt(1 - alphas_cumprod))
        self.posterior_variance = to_t((1 - 0.0) * betas * (1 - prev) / (1 - alphas_cumprod))
        self.posterior_log_variance_clipped = to_t(
            np.log(np.maximum(self.posterior_variance.cpu().numpy(), 1e-20))
        )
        self.posterior_mean_coef1 = to_t(betas * np.sqrt(prev) / (1 - alphas_cumprod))
        self.posterior_mean_coef2 = to_t((1 - prev) * np.sqrt(alphas) / (1 - alphas_cumprod))
        lvlb = betas ** 2 / (
            2 * self.posterior_variance.cpu().numpy() * alphas * (1 - alphas_cumprod)
        )
        lvlb[0] = lvlb[1]
        self.lvlb_weights = to_t(lvlb)
        self.num_timesteps = timesteps

    # -- input shaping ---------------------------------------------------

    def set_input(self, input, max_sample=None):
        self.x = input["sdf"].to(self.device, non_blocking=True)
        self.fp = input["fp"].to(self.device, non_blocking=True)
        self.class_id = input["class_id"].to(self.device, non_blocking=True)
        self.style_id = input["style_id"].to(self.device, non_blocking=True)
        self.height = input["height"].to(self.device, non_blocking=True).float()
        self.era_id = input["era_id"].to(self.device, non_blocking=True) if "era_id" in input else None
        self.floors_id = input["floors_id"].to(self.device, non_blocking=True) if "floors_id" in input else None
        self.region_id = input["region_id"].to(self.device, non_blocking=True) if "region_id" in input else None
        if max_sample is not None:
            self.x = self.x[:max_sample]
            self.fp = self.fp[:max_sample]
            self.class_id = self.class_id[:max_sample]
            self.style_id = self.style_id[:max_sample]
            self.height = self.height[:max_sample]
            if self.era_id is not None: self.era_id = self.era_id[:max_sample]
            if self.floors_id is not None: self.floors_id = self.floors_id[:max_sample]
            if self.region_id is not None: self.region_id = self.region_id[:max_sample]

    def _build_fp3d_for(self, D: int, H: int, W: int, C: int = 1) -> torch.Tensor:
        """Broadcast 2D footprint to (B, C, D, H, W) latent grid using the
        BuildingNet axis convention: footprint sits on (D, W) and replicates
        along H. Mirrors sdfusion_model_img2shape._build_fp3d_for verbatim."""
        fp2d = self.fp
        if fp2d.dim() == 4 and fp2d.shape[1] > 1:
            fp2d = fp2d[:, 0:1, ...]
        elif fp2d.dim() == 3:
            fp2d = fp2d.unsqueeze(1)
        fp2d = fp2d.to(dtype=torch.float32, device=self.device)
        fp2d_lat = F.interpolate(fp2d, size=(D, W), mode="nearest")
        fp3d = fp2d_lat.unsqueeze(3).repeat(1, C, 1, H, 1)
        return fp3d

    # Layer-B element-type vocabulary: 0=unknown 1=window 2=door 3=balcony 4=pilaster 5=bay.
    _ELEMENT_TYPES = ["unknown", "window", "door", "balcony", "pilaster", "bay"]

    @staticmethod
    def _classify_element_type(hz: float, hy: float, hw: float, y_center: float, H: float) -> int:
        """Voxel-space analog of scripts/server/facade_grammar.py:classify_shape (mode='add'),
        normalized by the full grid height H. (hz,hy,hw) = half-extents of the crude primitive's
        occupied bbox; y_center = its vertical center. Same thresholds as classify_shape, just
        re-expressed in voxel units instead of the normalized [0,1] cube."""
        if hz <= 0 and hy <= 0 and hw <= 0:
            return 0                                              # empty region -> unknown
        lateral = max(hw, hz, 1e-6)
        ground = y_center < 0.28 * H
        if hy < 0.55 * lateral:
            return 3                                              # wide & short -> balcony
        if hy > 1.7 * lateral:
            return 2 if ground else 4                             # tall & slender -> door : pilaster
        if min(hw, hy, hz) < 0.45 * (np.median([hw, hy, hz]) + 1e-6):
            return 2 if (ground and hy > 0.14 * H) else 1         # thin plane -> door : window
        return 5                                                  # chunky box -> bay

    def _build_context(self, x: torch.Tensor):
        """Layer-A context channels (latent res) derived self-supervised from the clean target x
        (B,1,D,H,W; D=z,H=y-up,W=x). Picks a random upper-biased edit region, then:
          known_body = body OUTSIDE the region (region emptied to +T),
          edit_mask  = the region,
          primitive  = a crude solid bbox of the body INSIDE the region (the 'crude placed mass').
        Teaches 'crude mass in region + surrounding body -> the coherent real element'. Also
        classifies the primitive's shape -> a Layer-B element-type id (see _classify_element_type).
        Returns three (B,1,d,h,w) latent-res channels (avg-pooled 64->16) + an (B,) elem-type id."""
        B, _, D, H, W = x.shape
        T = 0.2
        occ = (x[:, 0] <= 0.0)                                   # (B,D,H,W)
        mask = torch.zeros_like(x)
        prim = torch.full_like(x, T)
        elem_ids = []
        r = self._ctx_rng
        for b in range(B):
            dz = int(r.integers(D // 4, D // 2 + 1)); dy = int(r.integers(H // 4, H // 2 + 1)); dw = int(r.integers(W // 4, W // 2 + 1))
            z0 = int(r.integers(0, max(D - dz, 1))); w0 = int(r.integers(0, max(W - dw, 1)))
            y0 = int(r.integers(H // 3, max(H - dy, H // 3 + 1)))    # upper-biased (towers/roofs/dormers)
            z1, y1, w1 = z0 + dz, y0 + dy, w0 + dw
            mask[b, 0, z0:z1, y0:y1, w0:w1] = 1.0
            sub = occ[b, z0:z1, y0:y1, w0:w1]
            if bool(sub.any()):
                idx = torch.nonzero(sub, as_tuple=False)
                lo = idx.min(0).values.tolist(); hi = (idx.max(0).values + 1).tolist()
                prim[b, 0, z0 + lo[0]:z0 + hi[0], y0 + lo[1]:y0 + hi[1], w0 + lo[2]:w0 + hi[2]] = -T
                hz = (hi[0] - lo[0]) / 2.0; hy = (hi[1] - lo[1]) / 2.0; hw = (hi[2] - lo[2]) / 2.0
                y_center = y0 + (lo[1] + hi[1]) / 2.0
                elem_ids.append(self._classify_element_type(hz, hy, hw, y_center, float(H)))
            else:
                elem_ids.append(0)                                # empty region -> unknown
        known = torch.where(mask > 0.5, torch.full_like(x, T), x)
        f = D // int(self.df_conf.unet.params.image_size)           # 64 // 16 = 4
        ds = lambda v: F.avg_pool3d(v, kernel_size=f, stride=f)
        elem_id = torch.tensor(elem_ids, dtype=torch.long, device=x.device)
        return ds(known), ds(mask), ds(prim), elem_id

    def _null_ctx(self, fp3d):
        """3 zero context channels (Layer-A) matching fp3d (B,1,D,H,W) — identity under the
        zero-init context conv; used for plain generation/display where there is no edit."""
        z = torch.zeros_like(fp3d)
        return [z, z, z]

    def _build_global_context(self) -> torch.Tensor:
        """Encode (footprint, class, style, height) -> (B, 1, context_dim)."""
        # Footprint: feed FootprintEmbedNet a (B, 1, 64, 64) tensor.
        fp_in = self.fp
        if fp_in.dim() == 4 and fp_in.shape[1] > 1:
            fp_in = fp_in[:, 0:1, ...]
        elif fp_in.dim() == 3:
            fp_in = fp_in.unsqueeze(1)
        if fp_in.shape[-1] != 64 or fp_in.shape[-2] != 64:
            fp_in = F.interpolate(fp_in, size=(64, 64), mode="nearest")
        if self.freeze_fp_encoder:
            with torch.no_grad():
                fp_emb, _ = self.fp_encoder(fp_in.float(), class_id=self.class_id)
        else:
            fp_emb, _ = self.fp_encoder(fp_in.float(), class_id=self.class_id)
        cls_emb = self.class_emb(self.class_id)
        sty_emb = self.style_emb(self.style_id)
        hgt_emb = self.height_mlp(self.height.view(-1, 1))
        parts = [fp_emb, cls_emb, sty_emb, hgt_emb]
        if self.use_extra_cond:
            era = self.era_id if getattr(self, "era_id", None) is not None else torch.full_like(self.class_id, 5)
            flo = self.floors_id if getattr(self, "floors_id", None) is not None else torch.full_like(self.class_id, 4)
            parts += [self.era_emb(era), self.floors_emb(flo)]
        if self.use_region:
            reg = (self.region_id if getattr(self, "region_id", None) is not None
                   else torch.full_like(self.class_id, self.num_regions - 1))   # unknown
            parts.append(self.region_emb(reg))
        if self.use_element_type:
            et = (self.elem_type_id if getattr(self, "elem_type_id", None) is not None
                  else torch.full_like(self.class_id, 0))                       # unknown
            parts.append(self.element_type_emb(et))
        g = torch.cat(parts, dim=1)
        return self.global_proj(g).unsqueeze(1)  # (B, 1, context_dim)

    # -- diffusion ops --------------------------------------------------

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def apply_model(self, x_noisy, t, cond):
        cond = dict(cond)
        B = x_noisy.shape[0]
        cc = []
        for c in cond["c_concat"]:                      # keep ALL concat channels (fp + Layer-A context)
            if c.shape[0] != B:
                c = c.expand(B, -1, -1, -1, -1).contiguous()
            cc.append(c)
        cond["c_concat"] = cc
        if getattr(self, "use_adaln", False) and "adaln_vec" not in cond:
            # central derivation -> covers train, sdedit AND inference; CFG-nulled ctx -> zero
            # vector = the learned null embedding, consistent across train/inference.
            ctx = cond["c_crossattn"][0]
            if ctx.shape[0] != B:
                ctx = ctx.expand(B, -1, -1).contiguous()
            cond["adaln_vec"] = ctx.mean(dim=1)
        out = self.df(x_noisy, t, **cond)
        return out[0] if isinstance(out, tuple) else out

    def _predict_x0_from_eps(self, x_t, t, eps):
        a = extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape)
        s = extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - s * eps) / a.clamp_min(1e-8)

    def forward(self):
        # Encode GT SDF to latent, rescale to ~unit-std.
        with torch.no_grad():
            z = self.vqvae(self.x, forward_no_quant=True, encode_only=True).detach()
        z = z * self.scale_factor

        # Build conditioning.
        _, _, D, H, W = z.shape
        fp3d = self._build_fp3d_for(D, H, W) * self.fp3d_concat_scale
        cc = [fp3d]
        self.elem_type_id = None
        if self.use_context:                                       # Layer-A: + known_body, mask, primitive
            body, mask, prim, elem_id = self._build_context(self.x)
            cc = cc + [body, mask, prim]
            self.elem_type_id = elem_id                             # Layer-B: primitive's shape -> type
        ctx = self._build_global_context()                          # (after context: needs elem_type_id)
        # Classifier-free-guidance dropout (gap #3): with prob p_uncond, null the whole
        # conditioning (zero footprint + context + crossattn) so the unconditional branch trains.
        if self.p_uncond > 0:
            drop = (torch.rand(z.shape[0], device=self.device) < self.p_uncond)
            if drop.any():
                m = (~drop).view(-1, 1, 1, 1, 1).float()
                cc = [c * m for c in cc]
                ctx = ctx * (~drop).view(-1, 1, 1).float()
        cond = {"c_concat": cc, "c_crossattn": [ctx]}

        # Sample timestep, run diffusion loss.
        t = torch.randint(0, self.num_timesteps, (z.shape[0],), device=self.device)
        noise = torch.randn_like(z)
        x_noisy = self.q_sample(z, t, noise)
        eps_pred = self.apply_model(x_noisy, t, cond)
        target = noise  # eps parameterization
        loss_simple = F.mse_loss(eps_pred, target, reduction="none").mean([1, 2, 3, 4])
        loss_simple_mean = loss_simple.mean()
        loss = (loss_simple / torch.exp(self.logvar[t]) + self.logvar[t]).mean()
        loss_dict = {"total": loss.detach(), "simple": loss_simple_mean.detach()}

        # REPA alignment (early-stopped per 2505.16792 via repa_stop_iter).
        if getattr(self, "use_repa", False) and self._step < self.repa_stop_iter \
                and self._repa_feat is not None:
            r = self.repa(self._repa_feat, self.x)
            loss = loss + self.repa_weight * r
            loss_dict["repa"] = r.detach()
            loss_dict["total"] = loss.detach()

        # Optional guardrail: decode predicted x0 to SDF every K steps,
        # add (surface-band SmoothL1 + footprint BCE) against GT SDF.
        if (self.gd_enabled
                and self.gd_every > 0
                and (self._step % self.gd_every == 0)):
            with torch.no_grad():
                pass  # decoder eval mode; latent path requires grad
            x0_lat = self._predict_x0_from_eps(x_noisy, t, eps_pred) / self.scale_factor
            sdf_pred = self.vqvae.decode_no_quant(x0_lat)
            band = _surface_band_smooth_l1(sdf_pred, self.x,
                                           sigma=self.gd_band_sigma)
            fp_b = _soft_footprint_bce(sdf_pred, self.x, tau=self.gd_fp_tau)
            aux = self.gd_band_weight * band + self.gd_fp_weight * fp_b
            loss = loss + aux
            loss_dict["band"] = band.detach()
            loss_dict["fp"] = fp_b.detach()
            loss_dict["total"] = loss.detach()

        self.loss_df = loss
        self.loss_dict = loss_dict

    def backward(self):
        self.loss = self.loss_df
        self.loss.backward()

    def optimize_parameters(self, total_steps):
        self.forward()
        self.optimizer.zero_grad(set_to_none=True)
        self.backward()
        self.optimizer.step()
        self._step += 1
        if getattr(self, "use_ema", False):
            with torch.no_grad():
                for ep, p in zip(self.ema_df.parameters(), self.df.parameters()):
                    ep.mul_(self.ema_decay).add_(p, alpha=1.0 - self.ema_decay)

    def switch_eval(self):
        self.df.eval()
        self.vqvae.eval()
        self.fp_encoder.eval()
        self.class_emb.eval()
        self.style_emb.eval()
        self.height_mlp.eval()
        self.global_proj.eval()
        if self.use_element_type:
            self.element_type_emb.eval()

    def switch_train(self):
        self.df.train()
        # VQVAE + FootprintEmbedNet are frozen — keep eval()
        self.vqvae.eval()
        if not self.freeze_fp_encoder:
            self.fp_encoder.train()
        self.class_emb.train()
        self.style_emb.train()
        self.height_mlp.train()
        self.global_proj.train()
        if self.use_element_type:
            self.element_type_emb.train()

    def get_current_errors(self):
        ret = OrderedDict()
        if hasattr(self, "loss_dict") and isinstance(self.loss_dict, dict):
            for k, v in self.loss_dict.items():
                ret[k] = float(v.item()) if torch.is_tensor(v) else float(v)
        elif hasattr(self, "loss_df"):
            ret["total"] = float(self.loss_df.detach().item())
        return ret

    # -- evaluation / inference ----------------------------------------

    @torch.no_grad()
    def eval_metrics(self, dataloader, thres=0.0, global_step=0):
        """Stub for trainer compatibility. Heavier eval is in scripts/eval_stage3a.py."""
        self.switch_eval()
        ret = OrderedDict([("dummy", 0.0)])
        self.switch_train()
        return ret

    @torch.no_grad()
    def get_current_visuals(self):
        """Used by train.py at display_freq. Renders GT vs generated SDF."""
        from utils.util import tensor2im
        ret = OrderedDict()
        if self.renderer is None:
            return ret
        try:
            gt_t = render_sdf(self.renderer, self.x)
            ret["gt"] = tensor2im(gt_t.data)
            if hasattr(self, "gen_sdf"):
                gen_t = render_sdf(self.renderer, self.gen_sdf)
                ret["gen"] = tensor2im(gen_t.data)
        except Exception as exc:
            cprint(f"[!] Stage3a get_current_visuals failed: {exc}", "yellow")
        return ret

    @torch.no_grad()
    def inference(self, data, ddim_steps=None, ddim_eta=0.0,
                  uc_scale=None, infer_all=False, max_sample=16):
        self.switch_eval()
        self.set_input(data, max_sample=max_sample)
        ctx = self._build_global_context()
        # Latent shape from VQVAE config.
        zC = self.vq_conf.model.params.embed_dim
        n_down = len(self.vq_conf.model.params.ddconfig.ch_mult) - 1
        D = self.vq_conf.model.params.ddconfig.resolution // (2 ** n_down)
        H, W = D, D
        fp3d = self._build_fp3d_for(D, H, W) * self.fp3d_concat_scale
        ctx_ch = self._null_ctx(fp3d) if self.use_context else []   # plain gen: no edit context
        cond = {"c_concat": [fp3d] + ctx_ch, "c_crossattn": [ctx]}
        # Unconditional branch: zero footprint + zero context.
        uc_fp3d = torch.zeros_like(fp3d)
        uc_ctx = torch.zeros_like(ctx)
        uc = {"c_concat": [uc_fp3d] + ctx_ch, "c_crossattn": [uc_ctx]}
        samples, _ = self.ddim_sampler.sample(
            S=ddim_steps or self.ddim_steps,
            batch_size=ctx.shape[0],
            shape=(zC, D, H, W),
            conditioning=cond,
            unconditional_guidance_scale=uc_scale or self.uc_scale,
            unconditional_conditioning=uc,
            eta=ddim_eta,
            quantize_x0=False,
        )
        samples = samples / self.scale_factor
        self.gen_sdf = self.vqvae.decode_no_quant(samples)
        return self.gen_sdf

    @torch.no_grad()
    def sdedit(self, data, strength=0.5, ddim_steps=None, ddim_eta=0.0,
               uc_scale=None, max_sample=16, noise=None,
               guide_model=None, auto_scale=2.0, neutral_style=None,
               ref_latent=None, ref_alpha=0.0, ctx_channels=None, elem_type_id=None):
        """SDEdit: project a crude *edited* SDF onto the learned building manifold.

        Encodes the edited SDF (data['sdf']) to the VQVAE latent, adds `strength` worth of
        noise (0 = identity round-trip, 1 = full regeneration), then runs the conditional
        reverse diffusion over ONLY the tail of the DDIM schedule. Low strength keeps the
        user's sculpted massing; higher strength lets the prior reshape it into a more
        plausible building. Conditioning (footprint, class, style, height) steers the result.
        Returns the decoded SDF (B,1,64,64,64).

        Guidance:
          - Default (guide_model=None): classifier-free guidance with `uc_scale` against
            a zero-conditioning branch.
          - Autoguidance (guide_model set): guide THIS (strong) model with a weaker
            checkpoint of itself (Karras et al., NeurIPS'24). Both run the SAME conditioning;
            the score becomes  e = e_weak + auto_scale * (e_strong - e_weak).
            This sidesteps our untrained unconditional branch (the CFG failure mode) — no
            retraining needed. `guide_model` is another Stage3aModel loaded from an earlier
            checkpoint that shares this model's frozen VQVAE/latent space.
        """
        self.switch_eval()
        self.set_input(data, max_sample=max_sample)
        # Layer-B: real element-type id for the actual placed primitive (eval callers classify it
        # themselves via _classify_element_type, since sdedit doesn't run _build_context). None ->
        # unknown (matches training's no-context default).
        self.elem_type_id = elem_type_id
        S = int(ddim_steps or self.ddim_steps)
        strength = float(min(max(strength, 0.0), 1.0))
        # 1) encode the edited SDF -> scaled latent
        z = self.vqvae(self.x, forward_no_quant=True, encode_only=True).detach() * self.scale_factor
        # 1b) reference-guided probe (no retrain): bias the SDEdit start toward a retrieved real
        #     building's latent. GATED (ref_alpha=0 -> identity). Global low-weight blend pulls the
        #     trajectory toward the reference's manifold; the localized SDF blend still keeps the body.
        if ref_latent is not None and ref_alpha > 0.0:
            z = z * (1.0 - ref_alpha) + ref_latent.to(z.device, z.dtype) * ref_alpha
        # 2) conditioning (mirrors inference())
        _, _, D, H, W = z.shape
        fp3d = self._build_fp3d_for(D, H, W) * self.fp3d_concat_scale
        ctx = self._build_global_context()
        # Layer-A context channels (known_body, edit_mask, primitive) at latent res — real if passed,
        # else null (no-edit). uc branch always uses null context.
        ctx_ch = ((list(ctx_channels) if ctx_channels is not None else self._null_ctx(fp3d))
                  if self.use_context else [])
        cond = {"c_concat": [fp3d] + ctx_ch, "c_crossattn": [ctx]}
        uc = {"c_concat": [torch.zeros_like(fp3d)] + [torch.zeros_like(c) for c in ctx_ch],
              "c_crossattn": [torch.zeros_like(ctx)]}
        # 2b) autoguidance: build the weak model's conditioning + an eps_fn that
        #     combines strong/weak conditional predictions (replaces CFG when set).
        eps_fn = None
        if guide_model is not None:
            guide_model.switch_eval()
            guide_model.set_input(data, max_sample=max_sample)
            g_fp3d = guide_model._build_fp3d_for(D, H, W) * guide_model.fp3d_concat_scale
            g_ctx = guide_model._build_global_context()
            g_ctx_ch = guide_model._null_ctx(g_fp3d) if getattr(guide_model, "use_context", False) else []
            cond_bad = {"c_concat": [g_fp3d] + g_ctx_ch, "c_crossattn": [g_ctx]}
            w = float(auto_scale)

            def eps_fn(x_in, t_in):
                e_strong = self.apply_model(x_in, t_in, cond)
                e_weak = guide_model.apply_model(x_in, t_in, cond_bad)
                return e_weak + w * (e_strong - e_weak)
        elif neutral_style is not None:
            # Style-isolating guidance: guide the target style against a NEUTRAL style with the
            # SAME footprint, so the guidance term is ONLY the style difference. e = e_neutral +
            # w*(e_style - e_neutral). If styles are ignored, e_style==e_neutral -> identical
            # regardless of w (a definitive collapse test).
            saved = self.style_id
            self.style_id = torch.full_like(self.style_id, int(neutral_style))
            ctx_n = self._build_global_context()
            self.style_id = saved
            cond_n = {"c_concat": [fp3d] + ctx_ch, "c_crossattn": [ctx_n]}
            w = float(uc_scale if uc_scale is not None else self.uc_scale)

            def eps_fn(x_in, t_in):
                e_style = self.apply_model(x_in, t_in, cond)
                e_neutral = self.apply_model(x_in, t_in, cond_n)
                return e_neutral + w * (e_style - e_neutral)
        # 3) truncated DDIM schedule
        samp = self.ddim_sampler
        samp.make_schedule(ddim_num_steps=S, ddim_eta=ddim_eta, verbose=False)
        t_enc = min(int(round(strength * S)), S)
        if t_enc <= 0:                       # identity: just round-trip the latent
            self.gen_sdf = self.vqvae.decode_no_quant(z / self.scale_factor)
            return self.gen_sdf
        ddim_ts = samp.ddim_timesteps        # ascending DDPM timestep indices, len S
        b = z.shape[0]
        # 4) noise the latent to the start of the truncated range
        ts_start = torch.full((b,), int(ddim_ts[t_enc - 1]), device=self.device, dtype=torch.long)
        img = self.q_sample(z, ts_start, noise=noise)
        # 5) reverse-sample over indices t_enc-1 .. 0 (reuses p_sample_ddim's CFG)
        scale = uc_scale if uc_scale is not None else self.uc_scale
        for index in reversed(range(t_enc)):
            ts = torch.full((b,), int(ddim_ts[index]), device=self.device, dtype=torch.long)
            img, _ = samp.p_sample_ddim(
                img, cond, ts, index=index,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=uc,
                eps_fn=eps_fn,
            )
        samples = img / self.scale_factor
        self.gen_sdf = self.vqvae.decode_no_quant(samples)
        return self.gen_sdf

    # -- ckpt I/O -----------------------------------------------------

    def save(self, label, global_step, save_opt=True):
        state = {
            "df": self.df.state_dict(),
            "vqvae": self.vqvae.state_dict(),
            "fp_encoder": self.fp_encoder.state_dict(),
            "class_emb": self.class_emb.state_dict(),
            "style_emb": self.style_emb.state_dict(),
            "height_mlp": self.height_mlp.state_dict(),
            "global_proj": self.global_proj.state_dict(),
            "global_step": global_step,
            "step": self._step,
        }
        if self.use_extra_cond:
            state["era_emb"] = self.era_emb.state_dict()
            state["floors_emb"] = self.floors_emb.state_dict()
        if self.use_element_type:
            state["element_type_emb"] = self.element_type_emb.state_dict()
        if getattr(self, "use_ema", False):
            state["ema_df"] = self.ema_df.state_dict()
        if save_opt:
            state["opt"] = self.optimizer.state_dict()
            state["sched"] = self.scheduler.state_dict()
        torch.save(state, os.path.join(self.opt.ckpt_dir, f"stage3a_{label}.pth"))

    def load_ckpt(self, ckpt, load_opt=False):
        state = torch.load(ckpt, map_location="cpu") if isinstance(ckpt, str) else ckpt
        def _fit_in_conv(sd):
            # Layer-A: expand the input conv base_in_ch -> base_in_ch+n_ctx_ch, copying the old
            # weights into the first channels and ZERO-initing the new context channels (identity start).
            if not getattr(self, "use_context", False):
                return sd
            k = "diffusion_net.input_blocks.0.0.weight"
            if k in sd and sd[k].shape[1] == self._base_in_ch:
                w = self.df.state_dict()[k].clone()          # (C, base+ctx, 3,3,3)
                w[:, :self._base_in_ch] = sd[k]
                w[:, self._base_in_ch:] = 0.0
                sd = dict(sd); sd[k] = w
                cprint(f"[*] Layer-A: expanded input conv {self._base_in_ch}->{w.shape[1]} ch (context zero-init)", "blue")
            return sd
        def _fit_global_proj(sd):
            # Layer-B: expand global_proj's first Linear in_features when a warm-start ckpt lacks
            # the element_type_emb columns; copies old weights, ZERO-inits the new columns (the
            # new embedding starts contributing nothing, i.e. an identity start, same trick as
            # _fit_in_conv above).
            if not getattr(self, "use_element_type", False):
                return sd
            k = "0.weight"
            cur = self.global_proj.state_dict()
            if k in sd and sd[k].shape[1] != cur[k].shape[1]:
                w = cur[k].clone()
                old_in = sd[k].shape[1]
                w[:, :old_in] = sd[k]
                w[:, old_in:] = 0.0
                sd = dict(sd); sd[k] = w
                cprint(f"[*] Layer-B: expanded global_proj input {old_in}->{w.shape[1]} (element_type zero-init)", "blue")
            return sd
        self.df.load_state_dict(_fit_in_conv(state["df"]))
        if not getattr(self, "isTrain", False) and "ema_df" in state:
            self.df.load_state_dict(_fit_in_conv(state["ema_df"]))   # EMA weights sample cleaner; inference only
            cprint("[*] Stage3a using EMA weights (ema_df)", "blue")
        if "vqvae" in state:
            self.vqvae.load_state_dict(state["vqvae"])
        if "fp_encoder" in state:
            self.fp_encoder.load_state_dict(state["fp_encoder"])
        self.class_emb.load_state_dict(state["class_emb"])
        self.style_emb.load_state_dict(state["style_emb"])
        self.height_mlp.load_state_dict(state["height_mlp"])
        self.global_proj.load_state_dict(_fit_global_proj(state["global_proj"]))
        if self.use_extra_cond and "era_emb" in state:
            self.era_emb.load_state_dict(state["era_emb"])
            self.floors_emb.load_state_dict(state["floors_emb"])
        if self.use_element_type and "element_type_emb" in state:
            self.element_type_emb.load_state_dict(state["element_type_emb"])
        self._step = int(state.get("step", 0))
        cprint(f"[*] Stage3a weights loaded from {ckpt}", "blue")
        if load_opt and "opt" in state and getattr(self, "use_context", False):
            # Layer-A expanded the input conv -> the saved optimizer moments (4ch) are incompatible
            # with the new 7ch param. Use a FRESH optimizer (fine for a warm-start finetune).
            cprint("[*] Layer-A: skipping optimizer restore (input conv expanded; fresh optimizer)", "blue")
        elif load_opt and "opt" in state:
            self.optimizer.load_state_dict(state["opt"])
            if "sched" in state:
                self.scheduler.load_state_dict(state["sched"])
            cprint(f"[*] Stage3a optimizer + scheduler restored from {ckpt}", "blue")
