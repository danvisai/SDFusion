"""B+.6 — conditional diffusion over the recipe-parameter space (the generative head).

This is the "truly generative" replacement for the B+.5 deterministic MLP: instead of a
single (cond -> params) regression, it learns p(recipe_params | conditioning) so that
sampling produces *diverse* buildings consistent with the symbolic input
(footprint proportions, class, style).

It operates on the SAME normalized padded param vector (MAX_PARAMS=12) and the SAME
scale-invariant conditioning (COND_DIM=46) as B+.5, so it reuses
`models/networks/recipe_param_space.py` wholesale. Per-style validity masking is applied
to the denoising loss; padded/invalid dims are discarded after sampling.

Pieces:
  - ConditionalDenoiser : eps_theta(x_t, t, cond) MLP with sinusoidal time embedding,
                          cond projection, and a learned null embedding for
                          classifier-free guidance.
  - GaussianDiffusion   : cosine schedule, q_sample, masked eps loss with cond-dropout,
                          DDIM sampling with `eta` (stochastic -> diverse) and a guidance
                          scale.

References (see README §9): SALAD (part-level latent diffusion), GenCAD (conditioned
CAD-parameter diffusion), Ho et al. DDPM, Nichol & Dhariwal cosine schedule, Song DDIM.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from models.networks.recipe_param_space import COND_DIM, MAX_PARAMS


# ---------------------------------------------------------------------------
# Denoiser
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:  # t: (B,) long/float
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=device).float() / max(half - 1, 1)
        )
        ang = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        if self.dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class ConditionalDenoiser(nn.Module):
    """eps_theta(x_t, t, cond) -> predicted noise over MAX_PARAMS dims.

    Conditioning is injected by adding a (time + cond) embedding into each residual
    block. A learned `null_cond` vector replaces `cond` for the samples flagged by
    `drop` (classifier-free guidance).
    """

    def __init__(self, cond_dim: int = COND_DIM, n_params: int = MAX_PARAMS,
                 hidden: int = 256, depth: int = 4, time_dim: int = 128):
        super().__init__()
        self.cond_dim = cond_dim
        self.n_params = n_params
        self.null_cond = nn.Parameter(torch.zeros(cond_dim))

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
        )
        self.in_proj = nn.Linear(n_params, hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
            for _ in range(depth)
        ])
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(hidden, n_params))

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor,
                drop: torch.Tensor | None = None) -> torch.Tensor:
        if drop is not None:
            cond = torch.where(drop[:, None], self.null_cond[None, :].expand_as(cond), cond)
        cond_emb = self.cond_mlp(cond)
        t_emb = self.time_mlp(t)
        h = self.in_proj(x_t)
        film = t_emb + cond_emb
        for blk in self.blocks:
            h = h + blk(h + film)
        return self.out(h)


# ---------------------------------------------------------------------------
# Diffusion process
# ---------------------------------------------------------------------------

def cosine_alpha_bar(T: int, s: float = 0.008) -> np.ndarray:
    steps = T + 1
    x = np.linspace(0, T, steps) / T
    ab = np.cos((x + s) / (1 + s) * math.pi / 2) ** 2
    return (ab / ab[0]).astype(np.float64)  # alpha_bar at t=0..T, ab[0]=1


class GaussianDiffusion:
    """eps-prediction DDPM with cosine schedule + masked loss + DDIM sampling."""

    def __init__(self, num_timesteps: int = 1000, device: str = "cpu"):
        self.T = num_timesteps
        ab = cosine_alpha_bar(num_timesteps)
        betas = np.clip(1.0 - ab[1:] / ab[:-1], 0.0, 0.999)
        self.betas = torch.tensor(betas, dtype=torch.float32, device=device)
        acp = torch.tensor(ab[1:], dtype=torch.float32, device=device)  # (T,) alpha_bar_1..T
        self.alphas_cumprod = acp
        self.sqrt_acp = acp.sqrt()
        self.sqrt_one_minus_acp = (1.0 - acp).sqrt()
        self.device = device

    def to(self, device: str) -> "GaussianDiffusion":
        for k in ("betas", "alphas_cumprod", "sqrt_acp", "sqrt_one_minus_acp"):
            setattr(self, k, getattr(self, k).to(device))
        self.device = device
        return self

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor) -> torch.Tensor:
        return (self.sqrt_acp[t][:, None] * x0
                + self.sqrt_one_minus_acp[t][:, None] * noise)

    def p_losses(self, denoiser: ConditionalDenoiser, x0: torch.Tensor,
                 cond: torch.Tensor, mask: torch.Tensor,
                 p_uncond: float = 0.1) -> torch.Tensor:
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        drop = (torch.rand(B, device=x0.device) < p_uncond) if p_uncond > 0 else None
        eps = denoiser(x_t, t, cond, drop)
        se = (eps - noise) ** 2 * mask
        return se.sum() / mask.sum().clamp_min(1.0)

    @torch.no_grad()
    def ddim_sample(self, denoiser: ConditionalDenoiser, cond: torch.Tensor,
                    n_params: int = MAX_PARAMS, steps: int = 50, eta: float = 1.0,
                    guidance: float = 1.0, clamp: float = 5.0) -> torch.Tensor:
        """Sample normalized params (B, n_params). eta>0 -> stochastic (diverse);
        guidance>1 -> classifier-free guidance toward `cond`."""
        device = cond.device
        B = cond.shape[0]
        x = torch.randn(B, n_params, device=device)
        ts = torch.linspace(self.T - 1, 0, steps, device=device).round().long()
        for i in range(steps):
            t_cur = int(ts[i].item())
            t_b = torch.full((B,), t_cur, device=device, dtype=torch.long)
            if guidance != 1.0:
                eps_c = denoiser(x, t_b, cond, torch.zeros(B, dtype=torch.bool, device=device))
                eps_u = denoiser(x, t_b, cond, torch.ones(B, dtype=torch.bool, device=device))
                eps = eps_u + guidance * (eps_c - eps_u)
            else:
                eps = denoiser(x, t_b, cond, None)
            acp_t = self.alphas_cumprod[t_cur]
            x0_pred = (x - (1 - acp_t).sqrt() * eps) / acp_t.sqrt()
            if clamp:
                x0_pred = x0_pred.clamp(-clamp, clamp)
            if i + 1 < steps:
                t_next = int(ts[i + 1].item())
                acp_n = self.alphas_cumprod[t_next]
                sigma = eta * ((1 - acp_n) / (1 - acp_t)).sqrt() * (1 - acp_t / acp_n).sqrt()
                dir_xt = (1 - acp_n - sigma ** 2).clamp_min(0).sqrt() * eps
                x = acp_n.sqrt() * x0_pred + dir_xt + sigma * torch.randn_like(x)
            else:
                x = x0_pred
        return x
