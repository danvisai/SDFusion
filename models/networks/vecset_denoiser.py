"""Footprint-conditioned denoiser over a latent TOKEN SET (spec #67, the A2 generative model).

This is the part that stays ours. The autoencoder is inherited -- it is the half that provably needs
hundreds of thousands of shapes -- while the generative model, its conditioning, and the research claim
are built here and trained on our own corpus.

Why a token set rather than a grid, restated because it drives every design choice below: in the
dense-grid stack the diffusion must hit an exact value in every cell, and its errors surface as *surface
waviness* -- five separate efforts failed to fix that, and the deployed model still inflates buildings to
1.45x their true volume with 0.60 volumetric IoU. Over a token set the decoder maps *any* set to a crisp
surface, so a diffusion error becomes a wrong-but-clean building rather than a right-but-melted one.

Design consequences, each pinned by a test:
  * **No positional encoding on the token axis.** The latent is a set, so the network is permutation
    equivariant: permuting tokens permutes the output. This is the structural difference from a grid.
  * **No fixed token count.** Nothing may bake in a sequence length.
  * **Conditioning is purely geometric** -- footprint, height, region. No text, no images. That is
    load-bearing for the contribution claim, which is footprint-ONLY generation.
  * **An explicit unconditional path** (`drop_cond`), so classifier-free guidance needs no retraining.

Footprint enters as cross-attention over tokens from a small conv encoder of the 64^2 mask, rather than
as a pooled vector: massing is spatially structured, and a single vector cannot say *where* the plan is
notched.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Standard sinusoidal timestep embedding -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period)
                      * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    a = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(a), torch.sin(a)], dim=-1)
    return F.pad(emb, (0, dim - emb.shape[-1])) if emb.shape[-1] < dim else emb


class FootprintEncoder(nn.Module):
    """64^2 mask -> a short sequence of conditioning tokens, preserving *where* the plan is."""

    def __init__(self, width: int, res: int = 64, tokens: int = 16):
        super().__init__()
        self.tokens = tokens
        ch = max(width // 4, 16)
        self.net = nn.Sequential(
            nn.Conv2d(1, ch, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(ch, ch * 2, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(ch * 2, width, 4, 2, 1), nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(int(math.sqrt(tokens)))
        self.norm = nn.LayerNorm(width)

    def forward(self, fp: torch.Tensor) -> torch.Tensor:
        h = self.pool(self.net(fp))                       # (B, width, s, s)
        return self.norm(h.flatten(2).transpose(1, 2))    # (B, s*s, width)


class Block(nn.Module):
    """Self-attention over the token set, cross-attention to the conditioning, then an MLP.

    Timestep (plus the scalar conditioning) modulates via adaLN, which is what lets a single scalar
    steer the whole set without breaking permutation equivariance.
    """

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.n1, self.n2, self.n3 = (nn.LayerNorm(width, elementwise_affine=False) for _ in range(3))
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.cross = nn.MultiheadAttention(width, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(width, width * 6))
        nn.init.zeros_(self.ada[-1].weight); nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, cond_tokens, vec):
        s1, b1, g1, s2, b2, g2 = self.ada(vec)[:, None].chunk(6, dim=-1)
        h = self.n1(x) * (1 + s1) + b1
        x = x + g1 * self.attn(h, h, h, need_weights=False)[0]
        x = x + self.cross(self.n2(x), cond_tokens, cond_tokens, need_weights=False)[0]
        h = self.n3(x) * (1 + s2) + b2
        return x + g2 * self.mlp(h)


class VecsetDenoiser(nn.Module):
    """Predicts noise on a latent token set, conditioned on footprint + height + (optionally) region.

    `n_regions` is a property of the CORPUS, not of this module. It defaulted to 3 and no call site
    ever overrode it, which is how the frozen A2 source (`vecset_v4_surf` @240k) came to carry
    `region = nn.Embedding(3, 512)` -- and why it raises `IndexError` on every BuildingWorld row,
    all of which carry region id 3-8 (#188). Pass the corpus's own width.

    `n_regions=0` is the region-FREE variant (owner direction 2026-09-18): the embedding is not
    built at all, so the checkpoint structurally carries no region channel. That is deliberately
    stronger than passing `region=None` to a 3-region model -- that combination is an *untrained
    mode* of a model whose `cfg_drop` only ever dropped the whole condition jointly, which is
    exactly what #119 refused to generate from. Footprint and height still condition, so the
    footprint-only contribution claim is unaffected.
    """

    def __init__(self, latent_channels: int = 64, width: int = 768, depth: int = 12,
                 heads: int = 12, footprint_res: int = 64, n_regions: int = 3,
                 cond_tokens: int = 16):
        super().__init__()
        if n_regions < 0:
            raise ValueError(f"n_regions must be >= 0 (0 means region-free), got {n_regions}")
        self.width = width
        self.n_regions = int(n_regions)
        self.inp = nn.Linear(latent_channels, width)
        self.out = nn.Linear(width, latent_channels)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

        self.t_mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.fp_enc = FootprintEncoder(width, footprint_res, cond_tokens)
        self.height = nn.Linear(1, width)
        # `None`, not `nn.Embedding(0, width)`: an empty embedding is still a submodule, so it would
        # put a `region.weight` key in the state_dict and leave "is this checkpoint region-free?"
        # answerable only by reading a shape. Absence is the honest encoding of absence.
        self.region = nn.Embedding(n_regions, width) if n_regions else None
        # learned stand-ins for "no conditioning", so classifier-free guidance needs no retraining
        self.null_tokens = nn.Parameter(torch.randn(1, cond_tokens, width) * 0.02)
        self.null_vec = nn.Parameter(torch.zeros(1, width))

        self.blocks = nn.ModuleList([Block(width, heads) for _ in range(depth)])
        self.final = nn.LayerNorm(width)

    def forward(self, x, t, footprint, height=None, region=None, drop_cond: bool = False):
        """x (B,N,C) noisy tokens · t (B,) · footprint (B,1,R,R) · height (B,) · region (B,) longs."""
        B = x.shape[0]
        if region is not None and self.region is None:
            # Loudly, rather than dropping it: a caller passing region ids believes it holds a
            # region-conditioned model. Silently ignoring them would let a gate score a
            # region-conditioned harness against a region-free checkpoint and report the
            # difference as a corpus result.
            raise ValueError(
                "this denoiser is region-free (n_regions=0) but was handed region ids; pass "
                "region=None, or load a checkpoint that carries a region embedding")
        vec = self.t_mlp(timestep_embedding(t, self.width))

        if drop_cond:
            cond = self.null_tokens.expand(B, -1, -1)
            vec = vec + self.null_vec.expand(B, -1)
        else:
            cond = self.fp_enc(footprint)
            if height is not None:
                vec = vec + self.height(height.float().view(B, 1))
            if region is not None:
                vec = vec + self.region(region.long().view(B))

        h = self.inp(x)
        for blk in self.blocks:
            h = blk(h, cond, vec)
        return self.out(self.final(h))


def region_width_of(checkpoint: dict) -> int:
    """How many regions this saved denoiser conditions on. 0 means region-free.

    #188. Every consumer of a checkpoint -- the eval harness, the town service, #119's cache path --
    has to rebuild the net before it can load the weights, and each one currently hardcodes the
    3-region default. That default is the whole defect: it is why the frozen A2 source cannot
    condition on a single BuildingWorld row. So the answer comes from ONE place, and it is read off
    the weights rather than assumed.

    `n_regions` is preferred when the blob declares it, but every checkpoint written before #188 --
    the frozen source included -- predates that key, so the embedding's own shape is the fallback.
    A declaration its weights contradict is refused rather than guessed at.
    """
    weights = checkpoint.get("model", checkpoint)
    stored = weights.get("region.weight")
    from_weights = 0 if stored is None else int(stored.shape[0])
    declared = checkpoint.get("n_regions")
    if declared is None:
        return from_weights
    declared = int(declared)
    if declared != from_weights:
        raise ValueError(
            f"checkpoint declares n_regions={declared} but carries a region embedding of width "
            f"{from_weights} -- refusing to guess which is right")
    return declared


def denoiser_from_checkpoint(checkpoint: dict, device=None) -> "VecsetDenoiser":
    """Rebuild the denoiser a checkpoint describes, with its weights loaded and in eval mode.

    The single place that knows how a saved blob maps onto constructor arguments, so a consumer
    cannot quietly rebuild a *differently shaped* model and load weights into it (#188).
    """
    args = checkpoint.get("args", {})
    net = VecsetDenoiser(latent_channels=checkpoint["latent_channels"],
                         width=args["width"], depth=args["depth"], heads=args["heads"],
                         footprint_res=checkpoint["footprint_res"],
                         n_regions=region_width_of(checkpoint))
    net.load_state_dict(checkpoint["model"])
    if device is not None:
        net = net.to(device)
    return net.eval()


def region_tensor(net: "VecsetDenoiser", value, device=None):
    """The `region` argument to pass this denoiser for one building: a (1,) tensor, or None.

    Six call sites needed the same three lines -- "a region-free checkpoint has no embedding to
    index and raises if handed one" -- and a guard that is copy-pasted is a guard that gets missed
    at the seventh site. `forward` raises on a mismatch by design (#188); this is how a caller
    avoids provoking it without having to know which kind of checkpoint it is holding.
    """
    if not net.n_regions:
        return None
    import torch

    return torch.tensor([int(value)], device=device)
