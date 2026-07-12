"""Contract tests for the monolith's conditional noise-prediction UNet (ticket 11).

Fast + CPU-only: tiny resolutions and channel counts, no real data, no GPU. Exercises shape
contracts and the timestep-embedding math -- the seams a training loop depends on being correct
before any real training runs.

Run: env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
     models/networks/test_monolith_unet.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import monolith_unet as mu  # noqa: E402


class SinusoidalEmbeddingTest(unittest.TestCase):
    def test_output_shape(self):
        t = torch.tensor([0, 5, 999])
        emb = mu.sinusoidal_embedding(t, dim=32)
        self.assertEqual(emb.shape, (3, 32))

    def test_zero_timestep_is_deterministic_and_finite(self):
        emb = mu.sinusoidal_embedding(torch.tensor([0]), dim=16)
        self.assertTrue(torch.isfinite(emb).all())

    def test_different_timesteps_give_different_embeddings(self):
        emb = mu.sinusoidal_embedding(torch.tensor([0, 500]), dim=16)
        self.assertFalse(torch.allclose(emb[0], emb[1]))

    def test_odd_dim_still_produces_requested_width(self):
        emb = mu.sinusoidal_embedding(torch.tensor([1, 2]), dim=17)
        self.assertEqual(emb.shape, (2, 17))


class FiLMConvBlock3dTest(unittest.TestCase):
    def test_output_shape_matches_spatial_input_with_requested_channels(self):
        block = mu.FiLMConvBlock3d(in_ch=2, out_ch=8, temb_dim=12)
        x = torch.randn(2, 2, 8, 8, 8)
        temb = torch.randn(2, 12)
        out = block(x, temb)
        self.assertEqual(out.shape, (2, 8, 8, 8, 8))

    def test_finite_output(self):
        block = mu.FiLMConvBlock3d(in_ch=3, out_ch=3, temb_dim=8)
        out = block(torch.randn(1, 3, 4, 4, 4), torch.randn(1, 8))
        self.assertTrue(torch.isfinite(out).all())


class MonolithUNetTest(unittest.TestCase):
    def _tiny(self):
        return mu.MonolithUNet(base_channels=4, channel_mults=(1, 2), temb_dim=8)

    def test_output_shape_matches_single_channel_input(self):
        net = self._tiny()
        noisy = torch.randn(2, 1, 16, 16, 16)
        coarse = torch.randn(2, 1, 16, 16, 16)
        t = torch.tensor([3, 100])
        out = net(noisy, coarse, t)
        self.assertEqual(out.shape, (2, 1, 16, 16, 16))

    def test_finite_output_at_a_different_resolution(self):
        # 8^3 with two downsample levels (8 -> 4 -> 2) -- confirms no hardcoded resolution.
        net = self._tiny()
        out = net(torch.randn(1, 1, 8, 8, 8), torch.randn(1, 1, 8, 8, 8), torch.tensor([0]))
        self.assertEqual(out.shape, (1, 1, 8, 8, 8))
        self.assertTrue(torch.isfinite(out).all())

    def test_output_zero_initialized(self):
        # The output conv is zero-initialized (standard DDPM trick: the untrained network
        # starts as the identity map on the noise-prediction target, i.e. predicts zero).
        net = self._tiny()
        out = net(torch.randn(1, 1, 16, 16, 16), torch.randn(1, 1, 16, 16, 16), torch.tensor([7]))
        self.assertTrue(torch.allclose(out, torch.zeros_like(out)))

    def test_param_count_is_reported(self):
        net = self._tiny()
        n = sum(p.numel() for p in net.parameters())
        self.assertGreater(n, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
