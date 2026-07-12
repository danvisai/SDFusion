"""Ticket 11: the Gaussian diffusion process wrapping `models/networks/monolith_unet.py`.

Separated from the network module the same way this codebase already separates architecture
(`models/networks/`) from the higher-level model/training logic (`models/*_model.py`): the
schedule math and sampler are reused unchanged regardless of which UNet backs them, and are
pure enough to unit test without a GPU or a trained checkpoint.

DDIM sampling (not ancestral ddpm sampling) is used for the same reason `Stage3aModel` uses it
(`scripts/eval/transform_vs_noise.py`'s `ddim_steps=None` -> the model's own default): eta=0
DDIM is fully deterministic given the seed that draws the initial noise, so "inference
reproducibility" (ticket 11's own acceptance criterion) is exact by construction rather than
needing a fixed-seed *and* fixed-dropout-mask argument.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    """The original DDPM linear noise schedule."""
    return torch.linspace(beta_start, beta_end, timesteps)


def alphas_cumprod_from_betas(betas: torch.Tensor) -> torch.Tensor:
    return torch.cumprod(1.0 - betas, dim=0)


def q_sample(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor,
             alphas_cumprod: torch.Tensor) -> torch.Tensor:
    """Forward diffusion: `x_t = sqrt(alpha_bar_t) x0 + sqrt(1 - alpha_bar_t) noise`."""
    ac_t = alphas_cumprod[t].view(-1, *([1] * (x0.dim() - 1)))
    return ac_t.sqrt() * x0 + (1 - ac_t).sqrt() * noise


class GaussianDiffusion:
    """Ties a noise-prediction network to a fixed linear schedule. `model(noisy, coarse, t)`
    must return predicted noise of the same shape as `noisy`."""

    def __init__(self, model, timesteps: int = 1000, beta_start: float = 1e-4,
                 beta_end: float = 2e-2, device: str = "cpu",
                 surface_band: float = 0.3, surface_weight: float = 1.0):
        self.model = model
        self.timesteps = timesteps
        self.device = device
        self.betas = linear_beta_schedule(timesteps, beta_start, beta_end).to(device)
        self.alphas_cumprod = alphas_cumprod_from_betas(self.betas)
        # BuildingNet targets are >90% constant background (real occupancy is typically
        # <5% of the 96^3 volume, CONTEXT.md's own "Detail" scale is tied to thin facade
        # structure) -- an unweighted MSE over every voxel is dominated by the easy, building-
        # independent background and gives almost no gradient signal for the thin informative
        # surface band. Verified empirically (ticket 11 answer): an unweighted 15k-step
        # checkpoint reached near-zero aggregate loss (~0.001) while DDIM sampling still
        # produced ~30% occupancy against ~2% real targets -- the loss was "converged" on the
        # trivial part of the volume, not the part that matters.
        self.surface_band = surface_band
        self.surface_weight = surface_weight

    def p_losses(self, x0: torch.Tensor, coarse: torch.Tensor, t: torch.Tensor | None = None,
                 noise: torch.Tensor | None = None) -> torch.Tensor:
        """Noise-prediction MSE, weighted up near the true surface (`|x0| < surface_band`) so
        the thin informative region isn't drowned out by the constant-background majority of
        the volume (see `__init__`)."""
        b = x0.shape[0]
        if t is None:
            t = torch.randint(0, self.timesteps, (b,), device=x0.device)
        if noise is None:
            noise = torch.randn_like(x0)
        noisy = q_sample(x0, t, noise, self.alphas_cumprod)
        pred = self.model(noisy, coarse, t)
        weight = 1.0 + self.surface_weight * (x0.abs() < self.surface_band).float()
        return (weight * (pred - noise) ** 2).mean()

    @torch.no_grad()
    def ddim_sample(self, coarse: torch.Tensor, shape, ddim_steps: int = 50,
                     seed: int = 0, eta: float = 0.0, clip_x0: float = 1.0) -> torch.Tensor:
        """Deterministic (eta=0) DDIM sampling, conditioned on `coarse`. `seed` fixes the only
        source of randomness (the initial noise), so equal inputs + equal seed -> bit-identical
        output.

        `x0_pred` is clamped to `+-clip_x0` (the data's own known range -- training divides by
        `TRUNC` so real inputs live in [-1, 1]) every step. Without this, dividing by
        `sqrt(alphas_cumprod[t])` -- which is tiny at high t, by design -- amplifies any error
        in an imperfectly-trained `eps` prediction, and that error compounds multiplicatively
        over `ddim_steps`; verified empirically against an early checkpoint: unclamped sampling
        diverged to values outside [-16, 7] and ~65% predicted occupancy on real buildings with
        ~1-5% true occupancy. This is the standard `clip_denoised` DDPM/DDIM practice, not a new
        hyperparameter search."""
        device = coarse.device
        generator = torch.Generator(device=device if device.type != "mps" else "cpu")
        generator.manual_seed(seed)
        x = torch.randn(shape, generator=generator, device=device if device.type != "mps" else "cpu").to(device)

        step_indices = torch.linspace(self.timesteps - 1, 0, ddim_steps, device=device).round().long()
        step_indices = torch.unique_consecutive(step_indices)
        for i, t in enumerate(step_indices):
            t_batch = torch.full((shape[0],), int(t), device=device, dtype=torch.long)
            eps = self.model(x, coarse, t_batch)
            ac_t = self.alphas_cumprod[t]
            x0_pred = (x - (1 - ac_t).sqrt() * eps) / ac_t.sqrt().clamp(min=1e-8)
            if clip_x0 > 0:
                x0_pred = x0_pred.clamp(-clip_x0, clip_x0)
            ac_next = self.alphas_cumprod[step_indices[i + 1]] if i + 1 < len(step_indices) \
                else torch.tensor(1.0, device=device)
            x = ac_next.sqrt() * x0_pred + (1 - ac_next).sqrt() * eps
        return x
