"""Part-SET refiner — detail-plan step 4 (SPLICE-style joint refinement, the coherence engine).

A building's parts as a PADDED SET (40 slots × [type-onehot 10 | box 6 | validity 1]); a
conditional DDPM denoises the WHOLE SET jointly, cross-attending to the spatial massing grid.
Editing = set-SDEdit: perturb the set (add/move/duplicate a part), partially noise, denoise —
the set re-coheres (poses adjust, validity drops redundant parts, types stay consistent).
Permutation-equivariant (no slot positional encoding).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.networks.part_layout_planner import MassingEncoderSpatial, N_TYPES

SLOTS = 40
PART_DIM = N_TYPES + 6 + 1          # one-hot type | box6 | validity


def cosine_betas(T, s=0.008):
    t = torch.linspace(0, T, T + 1) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    betas = (1 - f[1:] / f[:-1]).clamp(1e-5, 0.999)
    return betas


class SetDenoiser(nn.Module):
    def __init__(self, dim=256, depth=6, heads=4):
        super().__init__()
        self.inp = nn.Linear(PART_DIM, dim)
        self.enc = MassingEncoderSpatial(dim)
        self.t_emb = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        layer = nn.TransformerDecoderLayer(dim, heads, dim * 4, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.tr = nn.TransformerDecoder(layer, depth)
        self.out = nn.Linear(dim, PART_DIM)
        self.dim = dim

    def _timestep(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        ang = t[:, None].float() * freqs[None]
        return torch.cat([ang.sin(), ang.cos()], dim=-1)

    def forward(self, x, t, sdf):
        """x (B, SLOTS, PART_DIM) noised set; sdf (B,1,64,64,64) massing."""
        _, mem = self.enc(sdf)
        h = self.inp(x) + self.t_emb(self._timestep(t))[:, None]
        h = self.tr(h, mem)                                   # full self-attn over slots (no mask)
        return self.out(h)


class PartSetRefiner(nn.Module):
    def __init__(self, T=1000, device="cuda"):
        super().__init__()
        self.T = T
        self.net = SetDenoiser().to(device)
        betas = cosine_betas(T).to(device)
        alphas = (1 - betas).cumprod(0)
        self.register_buffer("sqrt_ab", alphas.sqrt())
        self.register_buffer("sqrt_1mab", (1 - alphas).sqrt())
        self.device = device

    def q_sample(self, x0, t, noise):
        return self.sqrt_ab[t][:, None, None] * x0 + self.sqrt_1mab[t][:, None, None] * noise

    # validity channel weighted up: v1 only KILLED junk 17% of the time — deletion must be
    # as learnable as relocation (the validity dim is 1/17th of the signal otherwise).
    CH_W = None

    def loss(self, x0, sdf, x_corrupt=None):
        """Corruption-robust denoising: when x_corrupt (= x0 + junk parts) is given, the noised
        input comes from x_corrupt but the eps target points the implied x0-prediction at the
        CLEAN x0 — i.e. the model is SUPERVISED to delete junk. Without corrupted training
        pairs the refiner never learns deletion (measured: junk killed 0%)."""
        B = x0.shape[0]
        if x_corrupt is None:
            x_corrupt = x0
        t = torch.randint(0, self.T, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x_corrupt, t, noise)
        coef = (self.sqrt_ab[t] / self.sqrt_1mab[t].clamp_min(1e-4)).clamp(max=3.0)
        eps_tgt = noise + coef[:, None, None] * (x_corrupt - x0)
        eps = self.net(x_t, t, sdf)
        if PartSetRefiner.CH_W is None:
            w = torch.ones(PART_DIM, device=x0.device)
            w[-1] = 3.0                                   # validity
            w[N_TYPES:N_TYPES + 6] = 1.5                  # boxes (pose stability)
            PartSetRefiner.CH_W = w / w.mean()
        return (F.mse_loss(eps, eps_tgt, reduction="none") * PartSetRefiner.CH_W).mean()

    @torch.no_grad()
    def refine(self, x_init, sdf, strength=0.3, steps=12):
        """set-SDEdit: noise the given set to strength*T, denoise jointly (DDIM eta=0)."""
        t0 = int(max(min(strength, 0.999), 1e-3) * (self.T - 1))
        ts = torch.linspace(t0, 0, steps + 1, dtype=torch.long, device=x_init.device)
        x = self.q_sample(x_init, torch.full((x_init.shape[0],), t0, device=x_init.device,
                                             dtype=torch.long), torch.randn_like(x_init))
        for i in range(steps):
            t, t_next = ts[i], ts[i + 1]
            tb = torch.full((x.shape[0],), int(t), device=x.device, dtype=torch.long)
            eps = self.net(x, tb, sdf)
            x0 = (x - self.sqrt_1mab[t] * eps) / self.sqrt_ab[t].clamp_min(1e-6)
            x = self.sqrt_ab[t_next] * x0 + self.sqrt_1mab[t_next] * eps
        return x0


# ===========================================================================
# CoherentPartRefiner — the coherent-add-primitive upgrade (NEW; leaves PartSetRefiner +
# refiner.pth + /recohere_details untouched). Adds the no-image conditioning (spec §2):
#   image            -> massing SDF      (MassingEncoderSpatial cross-attn, as before)
#   2D part masks    -> ADDED-PRIMITIVE MARKER (per-slot bit: "integrate THIS piece")
#   object class     -> symbolic class embedding (FiLM)
# Trained on edit-pairs (x_corrupt, marker) -> x_clean; refine = set-SDEdit with optional
# X-Part neighbour-locality (freeze slots far from the marked piece).
# ===========================================================================
N_CLASSES = 4                                              # COMMERCIAL/PUBLIC/RELIGIOUS/RESIDENTIAL


class CoherentSetDenoiser(nn.Module):
    def __init__(self, dim=256, depth=6, heads=4):
        super().__init__()
        self.inp = nn.Linear(PART_DIM + 1, dim)            # +1: added-primitive marker
        self.enc = MassingEncoderSpatial(dim)
        self.t_emb = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.cls_emb = nn.Embedding(N_CLASSES, dim)
        layer = nn.TransformerDecoderLayer(dim, heads, dim * 4, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.tr = nn.TransformerDecoder(layer, depth)
        self.out = nn.Linear(dim, PART_DIM)
        self.dim = dim

    def _timestep(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        ang = t[:, None].float() * freqs[None]
        return torch.cat([ang.sin(), ang.cos()], dim=-1)

    def forward(self, x, t, sdf, marker, cls):
        """x (B,SLOTS,PART_DIM); marker (B,SLOTS); cls (B,) long."""
        _, mem = self.enc(sdf)
        film = (self.t_emb(self._timestep(t)) + self.cls_emb(cls))[:, None]   # (B,1,dim)
        h = self.inp(torch.cat([x, marker[..., None]], -1)) + film
        h = self.tr(h, mem)
        return self.out(h)


class CoherentPartRefiner(nn.Module):
    def __init__(self, T=1000, device="cuda"):
        super().__init__()
        self.T = T
        self.net = CoherentSetDenoiser().to(device)
        betas = cosine_betas(T).to(device)
        alphas = (1 - betas).cumprod(0)
        self.register_buffer("sqrt_ab", alphas.sqrt())
        self.register_buffer("sqrt_1mab", (1 - alphas).sqrt())
        self.device = device

    def q_sample(self, x0, t, noise):
        return self.sqrt_ab[t][:, None, None] * x0 + self.sqrt_1mab[t][:, None, None] * noise

    # wall-attachable element type indices (window/door/balcony/balcony_up) in the planner vocab
    WALL_TYPES = (0, 2, 5, 6)

    def coherence(self, x0, x_clean, sdf):
        """Relational coherence on the predicted x0 (recohere plan §1.B), self-supervised from the
        clean target in-batch — no structure cache needed:
          L_row    same-type parts in the SAME GT band stay co-planar (cy)
          L_size   same-type/band parts share size (uniform rhythm)
          L_attach wall parts' predicted centre sits on the massing surface (|sdf|->0)."""
        NT = N_TYPES
        valid = x_clean[..., -1] > 0
        typ = x_clean[..., :NT].argmax(-1)
        cy_p, cy_g = x0[..., NT + 1], x_clean[..., NT + 1]
        sz_p = x0[..., NT + 3:NT + 6]
        B = x0.shape[0]
        L_row = x0.new_zeros(()); L_size = x0.new_zeros(()); npair = 0
        for b in range(B):
            idx = torch.where(valid[b])[0]
            if len(idx) < 2:
                continue
            tb, cg, cp, sp = typ[b][idx], cy_g[b][idx], cy_p[b][idx], sz_p[b][idx]
            m = (tb[:, None] == tb[None]) & ((cg[:, None] - cg[None]).abs() < 0.04)
            m &= ~torch.eye(len(idx), dtype=torch.bool, device=x0.device)
            if m.any():
                L_row = L_row + ((cp[:, None] - cp[None]) ** 2)[m].mean()
                L_size = L_size + ((sp[:, None] - sp[None]) ** 2).sum(-1)[m].mean()
                npair += 1
        if npair:
            L_row = L_row / npair; L_size = L_size / npair
        wall = valid & torch.stack([typ == t for t in self.WALL_TYPES]).any(0)
        cen = x0[..., NT:NT + 3].clamp(-1, 1).view(B, 1, 1, SLOTS, 3)     # grid: (x,y,z)->(W,H,D)
        samp = F.grid_sample(sdf, cen, align_corners=True).view(B, SLOTS)
        L_attach = (samp[wall] ** 2).mean() if wall.any() else x0.new_zeros(())
        return L_row + L_size + L_attach

    def loss(self, x_clean, x_corrupt, marker, sdf, cls, cohw=0.0):
        """Edit-pair denoising: noise x_corrupt, but the eps target implies the CLEAN x0 — i.e.
        supervise the marked moldy piece (and any drift) toward the coherent layout. cohw>0 adds
        the relational coherence losses on the implied x0."""
        B = x_clean.shape[0]
        t = torch.randint(0, self.T, (B,), device=x_clean.device)
        noise = torch.randn_like(x_clean)
        x_t = self.q_sample(x_corrupt, t, noise)
        coef = (self.sqrt_ab[t] / self.sqrt_1mab[t].clamp_min(1e-4)).clamp(max=3.0)
        eps_tgt = noise + coef[:, None, None] * (x_corrupt - x_clean)
        eps = self.net(x_t, t, sdf, marker, cls)
        if PartSetRefiner.CH_W is None:
            w = torch.ones(PART_DIM, device=x_clean.device); w[-1] = 3.0
            w[N_TYPES:N_TYPES + 6] = 1.5; PartSetRefiner.CH_W = w / w.mean()
        l = (F.mse_loss(eps, eps_tgt, reduction="none") * PartSetRefiner.CH_W).mean()
        if cohw > 0:
            # x0 estimate is only meaningful at LOW noise; at high t it explodes -> gate + clamp
            low = t < int(0.30 * self.T)
            if low.any():
                x0 = (x_t[low] - self.sqrt_1mab[t[low]][:, None, None] * eps[low]) \
                    / self.sqrt_ab[t[low]][:, None, None].clamp_min(1e-6)
                l = l + cohw * self.coherence(x0.clamp(-1.5, 1.5), x_clean[low], sdf[low])
        return l

    @torch.no_grad()
    def refine(self, x_init, sdf, marker, cls, strength=0.3, steps=12, neighbor_k=0):
        """set-SDEdit conditioned on (sdf, marker, cls). neighbor_k>0 = X-Part locality: only
        the marked slot + its k nearest (by box-centre) get full noise; the rest stay ~frozen."""
        B = x_init.shape[0]
        t0 = int(max(min(strength, 0.999), 1e-3) * (self.T - 1))
        scale = torch.ones(B, SLOTS, device=x_init.device)
        if neighbor_k > 0:
            cen = x_init[..., N_TYPES:N_TYPES + 3]
            for b in range(B):
                mi = torch.where(marker[b] > 0)[0]
                if len(mi) == 0:
                    continue
                d = (cen[b][None] - cen[b][mi][:, None]).norm(dim=-1).min(0).values
                near = torch.argsort(d)[: neighbor_k + len(mi)]
                m = torch.full((SLOTS,), 0.06, device=x_init.device); m[near] = 1.0
                scale[b] = m
        ts = torch.linspace(t0, 0, steps + 1, dtype=torch.long, device=x_init.device)
        noise = torch.randn_like(x_init)
        x = self.q_sample(x_init, torch.full((B,), t0, device=x_init.device, dtype=torch.long),
                          noise * scale[..., None])
        x = x_init + scale[..., None] * (x - x_init)        # frozen slots ~ unchanged
        for i in range(steps):
            t, t_next = ts[i], ts[i + 1]
            tb = torch.full((B,), int(t), device=x.device, dtype=torch.long)
            eps = self.net(x, tb, sdf, marker, cls)
            x0 = (x - self.sqrt_1mab[t] * eps) / self.sqrt_ab[t].clamp_min(1e-6)
            step = self.sqrt_ab[t_next] * x0 + self.sqrt_1mab[t_next] * eps
            x = x_init + scale[..., None] * (step - x_init)
        return x0
