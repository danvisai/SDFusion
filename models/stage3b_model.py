"""Stage 3b model — SDF -> Gaussian voxel-slot lifter for BuildingNet.

Conditioning (same encoder pattern as Stage 3a, minus the FootprintEmbedNet
crossattn since Stage 3b's UNet uses FiLM):
    - class_id (53-way) -> 32-d embedding
    - style_id (9-way)  -> 16-d embedding   (recipes aren't used for 3b training
                                            since we lack v2 Gaussians for them,
                                            but the slot exists for inference
                                            consistency with Stage 3a)
    - height (1-d MLP)  -> 32-d
    - footprint global features from FootprintEmbedNet -> 256-d (frozen)
    Concatenated 256+32+16+32 = 336-d -> Linear(ctx_dim) -> FiLM context.

Spatial input to lifter UNet: (B, 2, 32, 32, 32) = (sdf_32 concat fp3d_32).
Output: per-cell slots + per-slot occupancy logits.

Losses (per-step, no every-K gating; the lifter is small enough to train fast):
    - means:      SmoothL1, mask by valid_slot
    - raw_scales: SmoothL1, mask by valid_slot
    - quats:      1 - |cos(angle)| (handles q == -q ambiguity), mask by valid_slot
    - raw_opac:   SmoothL1
    - sh_dc:      SmoothL1
    - occupancy:  BCEWithLogits per slot

Optional render-consistency loss (gated by --use_render_consistency in launcher)
to be added in a follow-up if v1 quality bottlenecks.
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
from termcolor import cprint, colored

from models.base_model import BaseModel
from models.networks.sdf_to_gs_lifter import SDFToGSLifter
from models.networks.retrieval.footprint_embed import FootprintEmbedNet
from utils.util_3d import init_mesh_renderer


SLOT_ATTR_DIM = 14


def _quat_cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - |dot(pred, target)| over normalized quaternions. Handles q==-q."""
    pred_n = pred / (pred.norm(dim=-1, keepdim=True) + 1e-8)
    targ_n = target / (target.norm(dim=-1, keepdim=True) + 1e-8)
    cos_sim = (pred_n * targ_n).sum(dim=-1).abs()
    return (1.0 - cos_sim).mean()


class Stage3bModel(BaseModel):
    """SDF -> Gaussian slot lifter."""

    def name(self) -> str:
        return "Stage3bModel"

    def initialize(self, opt) -> None:
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.device = opt.device

        self.df_conf = OmegaConf.load(opt.df_cfg)
        s3 = self.df_conf.get("stage3b", {})

        # Embedding dims (default to Stage 3a's so the conditioning encoder
        # weights could be re-used later if we want to share).
        self.num_classes = int(s3.get("num_classes", 53))
        self.num_styles = int(s3.get("num_styles_plus_unknown", 9))
        self.class_emb_dim = int(s3.get("class_emb_dim", 32))
        self.style_emb_dim = int(s3.get("style_emb_dim", 16))
        self.height_emb_dim = int(s3.get("height_emb_dim", 32))
        self.fp_emb_dim = int(s3.get("fp_emb_dim", 256))
        self.ctx_dim = int(s3.get("ctx_dim", 256))
        self.base_channels = int(s3.get("base_channels", 32))
        self.k_slots = int(s3.get("k_slots", 8))
        self.grid_res = int(s3.get("grid_res", 32))
        self.fp3d_concat_scale = float(s3.get("fp3d_concat_scale", 0.5))
        self.occ_threshold = float(s3.get("occ_threshold", 0.5))
        self.freeze_fp_encoder = bool(s3.get("freeze_fp_encoder", True))

        # Lifter network.
        self.lifter = SDFToGSLifter(
            in_channels=2,                  # sdf + fp3d
            base_channels=self.base_channels,
            ctx_dim=self.ctx_dim,
            k_slots=self.k_slots,
            attr_dim=SLOT_ATTR_DIM,
            grid_res=self.grid_res,
        ).to(self.device)

        # Footprint encoder (frozen, for global ctx vector).
        self.fp_encoder = FootprintEmbedNet(
            num_classes=self.num_classes,
            embedding_dim=self.fp_emb_dim,
        ).to(self.device)
        fp_ckpt = s3.get("fp_emb_ckpt", "Logs_GT/retrieval_footprint_full/ckpt_best.pth")
        if fp_ckpt and os.path.exists(fp_ckpt):
            state = torch.load(fp_ckpt, map_location="cpu")
            sd = state.get("model", state)
            try:
                self.fp_encoder.load_state_dict(sd, strict=False)
                cprint(f"[*] FootprintEmbedNet loaded from {fp_ckpt}", "blue")
            except Exception as exc:
                cprint(f"[!] FootprintEmbedNet load failed: {exc}", "yellow")
        if self.freeze_fp_encoder:
            for p in self.fp_encoder.parameters():
                p.requires_grad = False
            self.fp_encoder.eval()

        # Conditioning encoders.
        self.class_emb = nn.Embedding(self.num_classes, self.class_emb_dim).to(self.device)
        self.style_emb = nn.Embedding(self.num_styles, self.style_emb_dim).to(self.device)
        self.height_mlp = nn.Sequential(
            nn.Linear(1, self.height_emb_dim),
            nn.SiLU(),
            nn.Linear(self.height_emb_dim, self.height_emb_dim),
        ).to(self.device)
        total_global = (
            self.fp_emb_dim + self.class_emb_dim
            + self.style_emb_dim + self.height_emb_dim
        )
        self.global_proj = nn.Sequential(
            nn.Linear(total_global, self.ctx_dim),
            nn.SiLU(),
            nn.Linear(self.ctx_dim, self.ctx_dim),
        ).to(self.device)

        # Loss weights.
        lw = self.df_conf.get("stage3b_loss", {})
        self.w_means = float(lw.get("means", 1.0))
        self.w_scales = float(lw.get("scales", 0.5))
        self.w_quats = float(lw.get("quats", 0.5))
        self.w_opac = float(lw.get("opac", 0.5))
        self.w_sh = float(lw.get("sh", 0.5))
        self.w_occ = float(lw.get("occ", 1.0))

        # Optimizer + cosine schedule.
        if self.isTrain:
            train_params = (
                list(self.lifter.parameters())
                + list(self.class_emb.parameters())
                + list(self.style_emb.parameters())
                + list(self.height_mlp.parameters())
                + list(self.global_proj.parameters())
            )
            if not self.freeze_fp_encoder:
                train_params += list(self.fp_encoder.parameters())
            self.optimizer = optim.AdamW(
                train_params, lr=opt.lr, betas=(0.9, 0.999), weight_decay=1e-4,
            )
            warmup = max(int(getattr(opt, "warmup_steps", 1000)), 1)
            total = max(int(getattr(opt, "cosine_total_steps", 60000)), warmup + 1)
            def _lr_lambda(step: int) -> float:
                if step < warmup:
                    return float(step + 1) / float(warmup)
                p = (step - warmup) / max(total - warmup, 1)
                return 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, _lr_lambda)
            self.optimizers = [self.optimizer]
            self.schedulers = [self.scheduler]

        if opt.ckpt:
            self.load_ckpt(opt.ckpt, load_opt=self.isTrain)

        self.renderer = init_mesh_renderer(
            image_size=256, dist=1.7, elev=20, azim=20, device=self.device,
        )
        cprint(f"[*] Stage3bModel initialized (train={self.isTrain}).", "cyan")
        self._step = 0

    # ---- input plumbing ------------------------------------------------

    def set_input(self, input, max_sample=None) -> None:
        self.x = input["sdf"].to(self.device, non_blocking=True)
        self.fp = input["fp"].to(self.device, non_blocking=True)
        self.class_id = input["class_id"].to(self.device, non_blocking=True)
        self.style_id = input["style_id"].to(self.device, non_blocking=True)
        self.height = input["height"].to(self.device, non_blocking=True).float()
        # Training-only target tensors:
        if "slots" in input:
            self.slots_gt = input["slots"].to(self.device, non_blocking=True)
            self.occ_gt = input["occ_count"].to(self.device, non_blocking=True)  # (B, 32, 32, 32) uint8
            self.bbox_gt = input["bbox"].to(self.device, non_blocking=True)
        if max_sample is not None:
            for n in ("x", "fp", "class_id", "style_id", "height",
                      "slots_gt", "occ_gt", "bbox_gt"):
                if hasattr(self, n):
                    setattr(self, n, getattr(self, n)[:max_sample])

    def _build_fp3d_32(self) -> torch.Tensor:
        """Footprint broadcast to (B, 1, 32, 32, 32) on the (D=z, W=x) plane."""
        fp = self.fp
        if fp.dim() == 4 and fp.shape[1] > 1:
            fp = fp[:, 0:1, ...]
        elif fp.dim() == 3:
            fp = fp.unsqueeze(1)
        fp = fp.to(dtype=torch.float32, device=self.device)
        fp = F.interpolate(fp, size=(32, 32), mode="nearest")  # (B, 1, 32, 32)
        fp3d = fp.unsqueeze(3).repeat(1, 1, 1, 32, 1)            # (B, 1, 32, 32, 32)
        return fp3d * self.fp3d_concat_scale

    def _build_ctx(self) -> torch.Tensor:
        """Global conditioning vector for the lifter's FiLM blocks."""
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
        g = torch.cat([fp_emb, cls_emb, sty_emb, hgt_emb], dim=1)
        return self.global_proj(g)  # (B, ctx_dim)

    # ---- forward + loss ------------------------------------------------

    def forward(self) -> None:
        # Downsample 64 -> 32 SDF, concat fp3d.
        sdf_32 = F.avg_pool3d(self.x, kernel_size=2, stride=2)  # (B, 1, 32, 32, 32)
        fp3d_32 = self._build_fp3d_32()                          # (B, 1, 32, 32, 32)
        vol = torch.cat([sdf_32, fp3d_32], dim=1)                # (B, 2, 32, 32, 32)
        ctx = self._build_ctx()                                  # (B, ctx_dim)

        slots_pred, occ_logits = self.lifter(vol, ctx)
        # slots_pred:  (B, 32, 32, 32, K, 14)
        # occ_logits:  (B, K, 32, 32, 32)

        # Build per-slot validity mask from occ_count: slot k valid iff k < occ_count.
        B, D, H, W, K, A = slots_pred.shape
        ks = torch.arange(K, device=self.device).view(1, 1, 1, 1, K)
        occ_gt_b = self.occ_gt.unsqueeze(-1)                      # (B, D, H, W, 1)
        valid = (ks < occ_gt_b).float()                           # (B, D, H, W, K)

        # Per-slot attribute losses, masked by validity.
        attr_p = slots_pred
        attr_t = self.slots_gt                                    # (B, D, H, W, K, 14)
        v = valid.unsqueeze(-1)                                   # (B, D, H, W, K, 1)
        denom = v.sum().clamp_min(1.0)

        def _masked_smooth_l1(pred, target, mask):
            return (F.smooth_l1_loss(pred, target, reduction="none", beta=0.1) * mask).sum() / denom

        l_means = _masked_smooth_l1(attr_p[..., 0:3],  attr_t[..., 0:3],  v) * self.w_means
        l_scales = _masked_smooth_l1(attr_p[..., 3:6], attr_t[..., 3:6], v) * self.w_scales
        l_opac = _masked_smooth_l1(attr_p[..., 10:11], attr_t[..., 10:11], v) * self.w_opac
        l_sh = _masked_smooth_l1(attr_p[..., 11:14], attr_t[..., 11:14], v) * self.w_sh

        # Quaternion loss — only over valid slots.
        if valid.sum() > 0:
            pq = attr_p[..., 6:10][valid > 0]                     # (N_valid, 4)
            tq = attr_t[..., 6:10][valid > 0]
            l_quats = _quat_cosine_loss(pq, tq) * self.w_quats
        else:
            l_quats = torch.zeros((), device=self.device)

        # Per-slot occupancy BCE.
        # occ_logits is (B, K, D, H, W). Reorder to align with valid (B, D, H, W, K).
        occ_logits_aligned = occ_logits.permute(0, 2, 3, 4, 1)
        target_occ = (ks < occ_gt_b).float()                      # (B, D, H, W, K)
        l_occ = F.binary_cross_entropy_with_logits(
            occ_logits_aligned, target_occ, reduction="mean"
        ) * self.w_occ

        loss = l_means + l_scales + l_quats + l_opac + l_sh + l_occ
        self.loss_df = loss
        self.loss_dict = {
            "total": loss.detach(),
            "means": l_means.detach(),
            "scales": l_scales.detach(),
            "quats": l_quats.detach(),
            "opac": l_opac.detach(),
            "sh": l_sh.detach(),
            "occ": l_occ.detach(),
        }
        # Stash predictions for visualization.
        self._last_pred = (slots_pred.detach(), occ_logits.detach())

    def backward(self) -> None:
        self.loss = self.loss_df
        self.loss.backward()

    def optimize_parameters(self, total_steps) -> None:
        self.forward()
        self.optimizer.zero_grad(set_to_none=True)
        self.backward()
        self.optimizer.step()
        self._step += 1

    def switch_eval(self) -> None:
        self.lifter.eval()
        self.fp_encoder.eval()
        for m in (self.class_emb, self.style_emb, self.height_mlp, self.global_proj):
            m.eval()

    def switch_train(self) -> None:
        self.lifter.train()
        if not self.freeze_fp_encoder:
            self.fp_encoder.train()
        for m in (self.class_emb, self.style_emb, self.height_mlp, self.global_proj):
            m.train()

    def get_current_errors(self) -> OrderedDict:
        ret = OrderedDict()
        if hasattr(self, "loss_dict"):
            for k, v in self.loss_dict.items():
                ret[k] = float(v.item()) if torch.is_tensor(v) else float(v)
        return ret

    @torch.no_grad()
    def eval_metrics(self, dataloader, thres=0.0, global_step=0):
        self.switch_eval()
        ret = OrderedDict([("dummy", 0.0)])
        self.switch_train()
        return ret

    @torch.no_grad()
    def get_current_visuals(self):
        return OrderedDict()

    @torch.no_grad()
    def inference(self, data=None, **kwargs):
        """Stub for trainer compatibility — no DDIM here, just a single forward pass."""
        if data is not None:
            self.set_input(data)
        self.switch_eval()
        sdf_32 = F.avg_pool3d(self.x, kernel_size=2, stride=2)
        fp3d_32 = self._build_fp3d_32()
        vol = torch.cat([sdf_32, fp3d_32], dim=1)
        ctx = self._build_ctx()
        slots_pred, occ_logits = self.lifter(vol, ctx)
        self.switch_train()
        return slots_pred, occ_logits

    # ---- ckpt I/O -----------------------------------------------------

    def save(self, label, global_step, save_opt=False) -> None:
        state = {
            "lifter": self.lifter.state_dict(),
            "class_emb": self.class_emb.state_dict(),
            "style_emb": self.style_emb.state_dict(),
            "height_mlp": self.height_mlp.state_dict(),
            "global_proj": self.global_proj.state_dict(),
            "fp_encoder": self.fp_encoder.state_dict(),
            "global_step": global_step,
            "step": self._step,
        }
        if save_opt:
            state["opt"] = self.optimizer.state_dict()
            state["sched"] = self.scheduler.state_dict()
        torch.save(state, os.path.join(self.opt.ckpt_dir, f"stage3b_{label}.pth"))

    def load_ckpt(self, ckpt, load_opt=False) -> None:
        state = torch.load(ckpt, map_location="cpu") if isinstance(ckpt, str) else ckpt
        self.lifter.load_state_dict(state["lifter"])
        self.class_emb.load_state_dict(state["class_emb"])
        self.style_emb.load_state_dict(state["style_emb"])
        self.height_mlp.load_state_dict(state["height_mlp"])
        self.global_proj.load_state_dict(state["global_proj"])
        if "fp_encoder" in state:
            self.fp_encoder.load_state_dict(state["fp_encoder"])
        self._step = int(state.get("step", 0))
        cprint(f"[*] Stage3b weights loaded from {ckpt}", "blue")
        if load_opt and "opt" in state:
            self.optimizer.load_state_dict(state["opt"])
            if "sched" in state:
                self.scheduler.load_state_dict(state["sched"])
