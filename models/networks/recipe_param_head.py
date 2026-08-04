"""B+.5 deterministic recipe-parameter prediction head.

A small MLP mapping symbolic conditioning -> normalized padded recipe params:

    cond (COND_DIM) ──► [Linear-GELU-(Dropout)] x depth ──► Linear ──► params (MAX_PARAMS)

The head predicts the *normalized* param vector (see ParamNormalizer); call the
normalizer's inverse to recover raw params for `diff_recipe`. Padded/invalid dims are
predicted too but masked out of the loss, so the head only ever has to learn the dims
that exist for a style (style identity arrives via the conditioning one-hot).

This is the B+.5 sanity baseline that must overfit the (cond -> params) mapping before
B+.6 swaps in a diffusion model over the same param space.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.networks.recipe_param_space import COND_DIM, MAX_PARAMS


class RecipeParamHead(nn.Module):
    def __init__(self, cond_dim: int = COND_DIM, n_params: int = MAX_PARAMS,
                 hidden: int = 256, depth: int = 3, dropout: float = 0.0):
        super().__init__()
        layers = []
        d_in = cond_dim
        for _ in range(depth):
            layers += [nn.Linear(d_in, hidden), nn.GELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d_in = hidden
        layers.append(nn.Linear(d_in, n_params))
        self.net = nn.Sequential(*layers)
        self.cond_dim = cond_dim
        self.n_params = n_params

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        """cond: (B, COND_DIM) -> normalized params (B, MAX_PARAMS)."""
        return self.net(cond)


def masked_param_loss(pred: torch.Tensor, target: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
    """Mean-squared error over valid dims only.

    pred/target/mask: (B, MAX_PARAMS); mask is {0,1}. Normalised by the number of
    valid dims so styles with more params don't dominate the batch loss.
    """
    se = (pred - target) ** 2 * mask
    denom = mask.sum().clamp_min(1.0)
    return se.sum() / denom
