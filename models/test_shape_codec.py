"""Contract tests for the shape-codec seam (seam 1, spec #68).

The point of these tests is that **the contract is under test, not either codec**: the same suite runs
against every implementation, so a codec that satisfies it can be swapped in without touching the
diffusion or the evaluation harness.

Two tiers:
  * a **fake** codec with an analytic field -- pure CPU, no weights, always runs, and pins the contract
    itself (sign convention, grid/query agreement, chunking, degenerate input);
  * the **real** codecs -- skipped unless their weights are present, so the suite stays runnable
    anywhere while still checking the adapters when it can.

Run: env -u LD_PRELOAD ./sdfusion/bin/python models/test_shape_codec.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from models.shape_codec import Building, ShapeCodec, grid_points  # noqa: E402

DORA_CKPT = REPO / "external/dora_vae_1_1.ckpt"
VQ_CKPT = REPO / "logs_building/vqvae_clean_ft/vqvae_clean.pth"
VQ_CFG = REPO / "configs/vqvae_snet.yaml"


def _sphere_building(r=0.5, res=32):
    """A shape with an exact analytic SDF, so 'is the sign right' has a ground truth."""
    import trimesh
    ax = np.linspace(-1, 1, res)
    gx, gy, gz = np.meshgrid(ax, ax, ax, indexing="ij")
    sdf = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2) - r
    m = trimesh.creation.icosphere(subdivisions=3, radius=r)
    return Building(sdf=sdf.astype(np.float32), verts=np.asarray(m.vertices),
                    faces=np.asarray(m.faces))


class AnalyticCodec(ShapeCodec):
    """Encodes a sphere to its radius and answers exactly -- the contract with no model in the way."""

    name = "analytic"
    query_chunk = 1000            # deliberately small, to exercise chunked decode_grid

    def encode(self, building: Building) -> torch.Tensor:
        sdf = building.require_sdf()
        res = sdf.shape[0]
        occ = (sdf <= 0).sum()
        r = (occ / sdf.size * 8.0 * 3.0 / (4.0 * np.pi)) ** (1.0 / 3.0)
        return torch.tensor([[r]], dtype=torch.float32)

    def query(self, latent, points):
        r = latent[0, 0]
        return (points[0].norm(dim=-1) - r)[None]


class ContractSuite:
    """Behaviours every codec must exhibit. Subclasses supply `codec()` and `building()`."""

    def codec(self) -> ShapeCodec:
        raise NotImplementedError

    def building(self) -> Building:
        return _sphere_building()

    def test_sign_convention_is_negative_inside(self):
        c, b = self.codec(), self.building()
        z = c.encode(b)
        pts = torch.tensor([[[0.0, 0.0, 0.0], [0.95, 0.0, 0.0]]])
        d = c.query(z, pts)[0]
        self.assertLess(float(d[0]), 0.0, "centre of the solid must be negative")
        self.assertGreater(float(d[1]), 0.0, "a point outside must be positive")

    def test_decode_grid_agrees_with_query(self):
        """The grid path must be the query path -- that equivalence is what lets the existing
        evaluation harness score a query-based codec unchanged."""
        c, b = self.codec(), self.building()
        z = c.encode(b)
        res = 16
        g = c.decode_grid(z, res).view(-1)
        q = c.query(z, grid_points(res)[None].to(g.device))[0]
        torch.testing.assert_close(g.cpu(), q.cpu(), atol=2e-3, rtol=2e-3)

    def test_decode_grid_shape_and_finiteness(self):
        c, b = self.codec(), self.building()
        z = c.encode(b)
        for res in (8, 24):
            g = c.decode_grid(z, res)
            self.assertEqual(tuple(g.shape), (1, 1, res, res, res))
            self.assertTrue(torch.isfinite(g).all())

    def test_occupancy_is_plausible(self):
        """A sphere of radius 0.5 in [-1,1]^3 fills 4/3*pi*r^3 / 8 ~= 6.5% of the box."""
        c, b = self.codec(), self.building()
        g = c.decode_grid(c.encode(b), 32)
        self.assertTrue(0.02 < float((g <= 0).float().mean()) < 0.15)


class TestAnalyticCodec(ContractSuite, unittest.TestCase):
    def codec(self):
        return AnalyticCodec()


class TestBuildingProjections(unittest.TestCase):
    """A codec must fail loudly when the projection it needs is missing, not degrade silently."""

    def test_missing_sdf_raises(self):
        with self.assertRaises(ValueError):
            Building(verts=np.zeros((3, 3)), faces=np.zeros((1, 3), int)).require_sdf()

    def test_missing_mesh_raises(self):
        with self.assertRaises(ValueError):
            Building(sdf=np.zeros((4, 4, 4), np.float32)).require_mesh()

    def test_require_mesh_returns_outward_wound_surface(self):
        b = _sphere_building()
        b.faces = np.asarray(b.faces)[:, ::-1]        # invert it
        self.assertGreater(b.require_mesh().volume, 0)


@unittest.skipUnless(DORA_CKPT.exists(), "Dora checkpoint not present")
class TestDoraCodec(ContractSuite, unittest.TestCase):
    """Same contract, real weights. Skipped when the checkpoint is absent."""

    _codec = None

    def codec(self):
        if TestDoraCodec._codec is None:
            from models.shape_codec import DoraCodec
            from scripts.foundations.dora_roundtrip_probe import load_dora
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            TestDoraCodec._codec = DoraCodec(load_dora(dev), n_coarse=2048, n_sharp=2048)
        return TestDoraCodec._codec

    def test_decode_grid_agrees_with_query(self):
        # same equivalence, looser tolerance: a learned decoder is not bit-exact across chunkings
        c, b = self.codec(), self.building()
        z = c.encode(b)
        res = 16
        g = c.decode_grid(z, res).view(-1).cpu()
        q = c.query(z, grid_points(res)[None])[0].cpu()
        torch.testing.assert_close(g, q, atol=1e-3, rtol=1e-3)


@unittest.skipUnless(VQ_CKPT.exists(), "VQVAE checkpoint not present")
class TestVQVAECodec(ContractSuite, unittest.TestCase):
    """The SAME contract, against the existing dense-grid codec.

    This is the test that makes the seam meaningful: both codecs answer the same calls with the same
    conventions, so the diffusion and the evaluation harness cannot tell them apart.
    """

    _codec = None

    def codec(self):
        if TestVQVAECodec._codec is None:
            from types import SimpleNamespace
            from omegaconf import OmegaConf
            from models.model_utils import load_vqvae
            from models.shape_codec import VQVAECodec
            cfg = next((c for c in (VQ_CFG, REPO / "configs/vqvae_bnet_v2.yaml") if c.exists()), None)
            if cfg is None:
                self.skipTest("no vqvae config found")
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            conf = OmegaConf.load(str(cfg))
            vq = load_vqvae(conf, vq_ckpt=str(VQ_CKPT),
                            opt=SimpleNamespace(device=dev, gpu_ids=[0], isTrain=False))
            TestVQVAECodec._codec = VQVAECodec(vq.eval().to(dev))
        return TestVQVAECodec._codec

    def building(self):
        return _sphere_building(res=64)     # this codec is native-64^3

    def test_decode_grid_agrees_with_query(self):
        # for a native-grid codec, `query` is interpolation of the decoded field, so agreement is
        # exact only at grid nodes; assert there rather than pretending otherwise.
        c = self.codec()
        z = c.encode(self.building())
        g = c.decode_grid(z, 64).view(-1).cpu()
        q = c.query(z, grid_points(64)[None])[0].cpu()
        torch.testing.assert_close(g, q, atol=2e-3, rtol=2e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
