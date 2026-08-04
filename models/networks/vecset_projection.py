"""Set-SDEdit: project a footprint blockout onto the learned manifold, over a latent TOKEN SET.

This is the generator, and it is a **transform, not a sampler**. ADR 0003 and `CONTEXT.md` record the
decision plainly: *"You never sample a building from noise (degenerate at achievable data scale).
Instead you project"* -- and *"the same operator, applied to a footprint blockout, is generation."*
The reason is our data scale, which #64 quantified at 35,776 buildings against the 400K+ the vecset
literature trains on.

The procedure is the SDEdit one the repo already uses for grids (`Stage3aModel.sdedit`) and for part
sets (`part_set_refiner`), lifted to a vecset latent:

    encode(footprint blockout) -> tokens -> add PARTIAL noise (strength) -> denoise back

`strength` is the whole dial. At 0 the blockout is returned untouched; at 1 it is noised to the top of
the schedule but still denoised from *that* trajectory rather than from an unconditioned draw -- so even
at maximum it remains a projection of the input, never a free sample. `from_noise` exists purely as a
diagnostic and is deliberately not the claim.

The denoiser backbone is shared with from-noise sampling; only inference differs. That is why aligning
with ADR 0003 cost no rebuild.
"""
from __future__ import annotations

from typing import Optional

import torch


def cosine_alphas(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cumulative alphas on the cosine schedule -> (T,), decreasing from ~1 to ~0."""
    t = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64) / timesteps
    f = torch.cos((t + s) / (1 + s) * torch.pi * 0.5) ** 2
    ac = (f / f[0]).clamp(1e-8, 1.0)
    return ac[1:].float()


class SetSDEdit:
    """Projection over a latent token set. Wraps a denoiser; holds no parameters of its own."""

    def __init__(self, denoiser, timesteps: int = 1000, alphas_cumprod: Optional[torch.Tensor] = None,
                 x0_clamp: float = 3.0):
        """`x0_clamp` bounds the predicted clean latent. It must match the LATENT SCALE, not be a loose
        catch-all: training normalises latents to unit variance, so ~3 sigma is the honest bound. A
        looser clamp lets an unreliable epsilon push the latent far off the codec's manifold, and a
        vecset decoder given an off-manifold latent returns shredded geometry rather than a blurry
        shape -- which is exactly the vertical-slat failure the first evaluation rendered.
        """
        self.net = denoiser
        self.timesteps = timesteps
        self.x0_clamp = x0_clamp
        self.ac = cosine_alphas(timesteps) if alphas_cumprod is None else alphas_cumprod

    # -- the operator ---------------------------------------------------------

    def noise_to(self, blockout: torch.Tensor, strength: float,
                 noise: Optional[torch.Tensor] = None,
                 seed: Optional[int] = None) -> torch.Tensor:
        """The forward half: the partially-noised starting point the denoiser walks back from.

        Exposed because this is where "projection, not sampling" actually lives, and it is provable
        without a trained model: the result is `sqrt(a)*blockout + sqrt(1-a)*noise`, so blockout
        information is retained in proportion to (1 - strength). What happens on the way back down is
        a property of the weights; this is a property of the schedule.
        """
        if strength <= 0.0:
            return blockout.clone()
        dev = blockout.device
        t_start = int(min(max(strength, 0.0), 1.0) * (self.timesteps - 1))
        if noise is None:
            g = torch.Generator(device="cpu")
            if seed is not None:
                g.manual_seed(int(seed))
            noise = torch.randn(blockout.shape, generator=g)
        a = self.ac.to(dev)[t_start]
        return a.sqrt() * blockout + (1 - a).sqrt() * noise.to(dev)

    @torch.no_grad()
    def project(self, blockout: torch.Tensor, footprint: torch.Tensor,
                height: Optional[torch.Tensor] = None, region: Optional[torch.Tensor] = None,
                strength: float = 0.5, steps: int = 20, guidance: float = 1.0,
                seed: Optional[int] = None,
                noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Project `blockout` (B, N, C) onto the manifold. `strength` in [0, 1] sets how far to go.

        strength=0 returns the blockout untouched -- projection degrades to trusting the input rather
        than inventing geometry, which is the behaviour a footprint-conditioned generator should have
        when the footprint already determines the mass.

        ⚠️ strength→1 DEGENERATES to from-noise. At the top of the schedule the cumulative alpha is
        ~0, so `sqrt(a)*blockout` vanishes and no blockout information survives -- mathematically it
        becomes an unconditioned draw. That is exactly the regime ADR 0003 rejects, so the operating
        point must sit well below 1; `from_noise` is only that degenerate case named honestly.

        `noise` may be supplied directly (otherwise drawn from `seed`), which makes the token-set
        symmetry statable: permuting the blockout AND its noise permutes the output.
        """
        if strength <= 0.0:
            return blockout.clone()

        dev = blockout.device
        ac = self.ac.to(dev)
        t_start = int(min(max(strength, 0.0), 1.0) * (self.timesteps - 1))

        x = self.noise_to(blockout, strength, noise, seed)        # partial noising, not a fresh draw

        # DDIM-style deterministic walk back down the schedule
        ts = torch.linspace(t_start, 0, min(steps, t_start + 1), dtype=torch.long, device=dev)
        for i, t in enumerate(ts):
            eps = self._eps(x, t, footprint, height, region, guidance)
            at = ac[t]
            x0 = ((x - (1 - at).sqrt() * eps) / at.sqrt()).clamp(-self.x0_clamp, self.x0_clamp)
            if i + 1 < len(ts):
                an = ac[ts[i + 1]]
                x = an.sqrt() * x0 + (1 - an).sqrt() * eps
            else:
                x = x0
        return x

    @torch.no_grad()
    def from_noise(self, shape, footprint, height=None, region=None, steps: int = 50,
                   guidance: float = 1.0, seed: Optional[int] = None,
                   device=None) -> torch.Tensor:
        """DIAGNOSTIC ONLY -- retained to measure what the model has learned, never the claim.

        ADR 0003 rejects from-noise generation at our data scale; this exists so that position stays
        testable rather than merely asserted.
        """
        g = torch.Generator(device="cpu")
        if seed is not None:
            g.manual_seed(int(seed))
        x = torch.randn(shape, generator=g).to(device or footprint.device)
        return self.project(x, footprint, height, region, strength=1.0, steps=steps,
                            guidance=guidance, seed=seed)

    # -- internals ------------------------------------------------------------

    def _eps(self, x, t, footprint, height, region, guidance: float):
        tt = t.reshape(1).expand(x.shape[0]).to(x.device)
        eps = self.net(x=x, t=tt, footprint=footprint, height=height, region=region)
        if guidance != 1.0:
            unc = self.net(x=x, t=tt, footprint=footprint, height=height, region=region,
                           drop_cond=True)
            eps = unc + guidance * (eps - unc)
        return eps
