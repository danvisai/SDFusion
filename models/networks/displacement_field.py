"""Implicit displacement field for detail-preserving refine (Option D / HF-NeuS style).

The snap-refine (`scripts/server/refine.py` fast/quality) projects a sculpt onto the recipe
manifold and LOSES detail the recipe can't express (a tower, a carved notch). This instead
represents the finalized building as

    final(x) = recipe_base(x) + displacement(x)

where `displacement` is a small per-building-fit MLP so that `base + displacement` matches
the *edited* SDF. The base captures the clean, style-consistent coarse shape; the
displacement carries the user's sculpted detail as a smooth residual. The result keeps the
detail, is a clean closed implicit surface (no CSG seams / non-watertight holes), and stays
differentiable.

This is per-building optimization (overfit one shape, like HF-NeuS/SIREN), not a trained
generalizing model — that amortized version is the natural follow-on (predict the
displacement from conditioning, mirroring B+.5 -> B+.6).

SIREN (Sitzmann et al. 2020, https://arxiv.org/abs/2006.09661), not Fourier-feature + ReLU/
SiLU: a Fourier-encoded input into a standard MLP is spectrally biased toward smooth,
low-frequency solutions, which was producing over-smoothed "blobby" relief detail (verified
2026-07-07 on paint_relief output — sharp edges from the source art/depth map were getting
rounded off). Periodic sine activations throughout the network represent sharp edges and
fine local structure far better at the same network size — this was the design this
module's docstring already pointed at ("like HF-NeuS/SIREN") but never actually implemented.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SineLayer(nn.Module):
    """One SIREN layer: sin(w0 * (W x + b)). `w0` on the first layer sets the base
    frequency (30.0, the paper's default, works well since inputs are pre-normalized to
    ~[-1,1]); later layers use a smaller w0 so depth adds refinement, not aliasing. Weight
    init follows the paper's scheme, which keeps the sine's input distribution stable
    across layers regardless of depth."""

    def __init__(self, in_features: int, out_features: int, is_first: bool = False,
                w0: float = 30.0):
        super().__init__()
        self.w0 = w0
        self.linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            bound = (1.0 / in_features) if is_first else (math.sqrt(6.0 / in_features) / w0)
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * self.linear(x))


class DisplacementField(nn.Module):
    """d(x): R^3 -> R, bounded residual via a SIREN. Inputs are normalized to ~[-1, 1].

    `n_freq` is kept only for call-site/signature compatibility with the old Fourier-
    feature version (refine.py and scripts/test_displacement_representation.py both pass
    it) — SIREN doesn't need an explicit frequency bank, `w0_first` plays that role.
    """

    def __init__(self, n_freq: int = 6, hidden: int = 128, depth: int = 3,
                 out_scale: float = 0.5, w0_first: float = 30.0, w0_hidden: float = 1.0):
        super().__init__()
        del n_freq  # unused (SIREN's w0 replaces the Fourier frequency bank)
        layers = [SineLayer(3, hidden, is_first=True, w0=w0_first)]
        for _ in range(max(depth - 1, 0)):
            layers.append(SineLayer(hidden, hidden, is_first=False, w0=w0_hidden))
        self.net = nn.Sequential(*layers)
        self.out_layer = nn.Linear(hidden, 1)
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden) / w0_hidden
            self.out_layer.weight.uniform_(-bound, bound)
            self.out_layer.bias.zero_()
        self.out_scale = out_scale

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        d = self.out_layer(self.net(x_norm)).squeeze(-1)
        return torch.tanh(d) * self.out_scale


def fit_displacement(base_vals: torch.Tensor, target_vals: torch.Tensor,
                     pts_norm: torch.Tensor, *, steps: int = 400, lr: float = 1e-3,
                     band: float = 0.15, reg: float = 0.02, device: str = "cpu",
                     n_freq: int = 6, hidden: int = 128, out_scale: float = 0.6
                     ) -> DisplacementField:
    """Fit d so that base + d ~= target, emphasizing the target surface band.

    base_vals/target_vals: (Q,) precomputed SDF values; pts_norm: (Q,3) in ~[-1,1].
    `reg` keeps the displacement minimal where the base already matches (clean residual).
    `out_scale` bounds |d|: small (~0.6) = fine surface detail only (HF-NeuS); large
    (set to the residual range) = can also reproduce big user-added masses (towers).
    """
    field = DisplacementField(n_freq=n_freq, hidden=hidden, out_scale=out_scale).to(device)
    opt = torch.optim.Adam(field.parameters(), lr=lr)
    # Weight points near EITHER surface (base or target) so seams/detail are captured.
    w = torch.exp(-(target_vals.abs() / band) ** 2) + 0.3 * torch.exp(-(base_vals.abs() / band) ** 2)
    w = (w + 0.05)
    wn = w.sum().clamp_min(1.0)
    for _ in range(steps):
        opt.zero_grad()
        d = field(pts_norm)
        fit = ((base_vals + d - target_vals).abs() * w).sum() / wn
        loss = fit + reg * d.abs().mean()
        loss.backward()
        opt.step()
    return field


def normalizer(bbox):
    """Return fn mapping world points -> ~[-1,1] for the field, given a bbox."""
    x0, y0, z0, x1, y1, z1 = [float(v) for v in bbox]
    center = torch.tensor([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
    half = torch.tensor([max((x1 - x0) / 2, 1e-3), max((y1 - y0) / 2, 1e-3),
                         max((z1 - z0) / 2, 1e-3)])

    def f(pts: torch.Tensor) -> torch.Tensor:
        return (pts - center.to(pts.device)) / half.to(pts.device)
    return f
