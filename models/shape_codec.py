"""Seam 1 (spec #68): the shape-codec contract, and adapters for the two codecs we have.

The diffusion should not know which autoencoder is behind it. Today the VQVAE is reachable from a
handful of places in the Stage3a model, so this boundary already exists implicitly -- this writes it
down, and lets a query-based codec sit behind the same calls.

Three operations:

  * ``encode(building) -> latent``            what the diffusion models
  * ``query(latent, points) -> sdf``          signed distance at ARBITRARY points -- the actual
                                              representation change; resolution becomes a decode-time
                                              choice rather than a training-time constant
  * ``decode_grid(latent, res) -> (B,1,R,R,R)``  a grid, so every existing caller and the whole
                                              evaluation harness keep working untouched

`decode_grid` is deliberately expressible via `query`, which is what makes the swap safe: the gate,
the roughness diagnostic and the shared level-0.0 meshing helper need no changes.

**Conventions are part of the contract, not an adapter's private business**, because the two codecs
disagree on both and a silent mismatch inverts the shape:
  * sign  -- NEGATIVE INSIDE, matching `real.h5` (Dora returns positive-inside; TripoSG matches ours)
  * frame -- Frame-N, [-1,1]^3

A `Building` carries both a grid and a mesh because the codecs consume different projections of the
same object: a dense-grid codec reads the field, a vecset codec reads points on the surface.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

FRAME_LO, FRAME_HI = -1.0, 1.0


@dataclass
class Building:
    """One building, in Frame-N, in whichever projections are available.

    `sdf` is (R,R,R) with negative inside. `verts`/`faces` are the surface, wound outward. A codec
    raises if the projection it needs is absent, rather than silently degrading.
    """
    sdf: Optional[np.ndarray] = None
    verts: Optional[np.ndarray] = None
    faces: Optional[np.ndarray] = None

    def require_sdf(self) -> np.ndarray:
        if self.sdf is None:
            raise ValueError("this codec needs a grid, but the Building carries no sdf")
        return self.sdf

    def require_mesh(self):
        if self.verts is None or self.faces is None:
            raise ValueError("this codec needs a surface, but the Building carries no mesh")
        import trimesh
        from scene.surface_sampling import ensure_outward
        return ensure_outward(trimesh.Trimesh(np.asarray(self.verts, np.float64),
                                              np.asarray(self.faces), process=False))


def grid_points(res: int, device=None) -> torch.Tensor:
    """The (res^3, 3) query points of a Frame-N grid, in the array's own [x, y, z] index order."""
    ax = torch.linspace(FRAME_LO, FRAME_HI, res, dtype=torch.float32, device=device)
    gx, gy, gz = torch.meshgrid(ax, ax, ax, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=1)


class ShapeCodec(abc.ABC):
    """Encode a building to a latent; answer signed distance anywhere; materialise a grid."""

    name: str = "codec"
    #: chunk size for query batching, so a full grid never has to fit in one forward pass
    query_chunk: int = 32768

    @abc.abstractmethod
    def encode(self, building: Building) -> torch.Tensor:
        """-> latent, shape (1, ...). What the diffusion learns to produce."""

    @abc.abstractmethod
    def query(self, latent: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """(1, N, 3) points -> (1, N) signed distance, NEGATIVE INSIDE, Frame-N."""

    def decode_grid(self, latent: torch.Tensor, res: int = 64) -> torch.Tensor:
        """-> (1, 1, res, res, res). Default implementation is `query` over the grid, so any codec
        satisfying the contract is automatically scorable by the existing evaluation harness."""
        pts = grid_points(res, device=self._device(latent))[None]
        out = torch.cat([self.query(latent, pts[:, i:i + self.query_chunk])
                         for i in range(0, pts.shape[1], self.query_chunk)], dim=1)
        return out.view(1, 1, res, res, res)

    @staticmethod
    def _device(t: torch.Tensor):
        return t.device if torch.is_tensor(t) else None


class VQVAECodec(ShapeCodec):
    """The current dense-grid codec, wrapped unchanged.

    Its decoder is native-grid, so `query` is trilinear interpolation of the decoded field. That makes
    it a legitimate member of the contract while being honest about the difference: for this codec,
    resolution is fixed at train time and querying above it interpolates rather than resolves.
    """

    name = "vqvae"

    def __init__(self, vqvae, res: int = 64):
        self.vqvae, self.res = vqvae, res

    def encode(self, building: Building) -> torch.Tensor:
        x = torch.as_tensor(building.require_sdf(), dtype=torch.float32)[None, None]
        x = x.to(next(self.vqvae.parameters()).device)
        with torch.no_grad():
            return self.vqvae(x, forward_no_quant=True, encode_only=True)

    def decode_grid(self, latent: torch.Tensor, res: int = 64) -> torch.Tensor:
        with torch.no_grad():
            g = self.vqvae.decode_no_quant(latent)
        if res != g.shape[-1]:
            g = torch.nn.functional.interpolate(g, size=(res,) * 3, mode="trilinear",
                                                align_corners=True)
        return g

    def query(self, latent: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        g = self.decode_grid(latent, self.res)
        # grid_sample wants (x, y, z) in [-1,1] indexed as (W, H, D); our array axes are [x, y, z]
        p = points.view(1, 1, 1, -1, 3).flip(-1).to(g.dtype).to(g.device)
        return torch.nn.functional.grid_sample(g, p, mode="bilinear", align_corners=True).view(1, -1)


class DoraCodec(ShapeCodec):
    """The pretrained vecset codec, wrapped so its idiosyncrasies stay in one place.

    Two conventions are normalised here rather than leaking outward: it consumes two point streams
    (uniform + sharp-edge) instead of a grid, and it returns a **positive-inside** TSDF, which is
    negated to match ours.
    """

    name = "dora"

    def __init__(self, model, n_coarse: int = 8192, n_sharp: int = 8192, seed: int = 0):
        self.model, self.n_coarse, self.n_sharp = model, n_coarse, n_sharp
        self.rng = np.random.default_rng(seed)

    def encode(self, building: Building) -> torch.Tensor:
        from scene.surface_sampling import sample_streams
        mesh = building.require_mesh()
        dev = next(self.model.parameters()).device
        coarse, sharp = sample_streams(mesh, self.n_coarse, self.n_sharp, self.rng)
        with torch.no_grad():
            _, kl, _ = self.model.encode(torch.from_numpy(coarse)[None].to(dev),
                                         torch.from_numpy(sharp)[None].to(dev),
                                         sample_posterior=False)
        return kl

    def query(self, latent: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        dev = next(self.model.parameters()).device
        with torch.no_grad():
            lat = self.model.decode(latent)
            out = self.model.query(points.to(dev).float(), lat).float()
        return -out.view(1, -1)          # positive-inside -> our negative-inside convention
