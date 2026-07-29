"""Contract tests for set-SDEdit projection over a latent token set (spec #67, ADR 0003).

ADR 0003 and `CONTEXT.md` record that a building is never sampled from noise -- *"Instead you
project"* -- because from-noise generation is degenerate at our data scale. So the generator is a
PROJECTION: encode a footprint blockout, add PARTIAL noise, denoise back onto the learned manifold.

These tests pin the properties that make it a projection rather than a sampler, so the distinction
cannot quietly erode:
  * strength 0 is a no-op -- the blockout survives untouched
  * higher strength departs further from the blockout (it is a dial, not a switch)
  * the footprint still conditions the result
  * it is deterministic given a seed

CPU-only, tiny, no weights. Run:
    env -u LD_PRELOAD ./sdfusion/bin/python models/networks/test_vecset_projection.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from models.networks.vecset_denoiser import VecsetDenoiser  # noqa: E402
from models.networks.vecset_projection import SetSDEdit  # noqa: E402

N, C, FP = 16, 8, 16


def _net():
    torch.manual_seed(0)
    net = VecsetDenoiser(latent_channels=C, width=32, depth=2, heads=4, footprint_res=FP)
    with torch.no_grad():                      # lift it out of the zero-init degenerate state
        net.out.weight.normal_(0, 0.05)
        for blk in net.blocks:
            blk.ada[-1].weight.normal_(0, 0.05)
    return net.eval()


def _inputs(n=1):
    torch.manual_seed(1)
    return dict(blockout=torch.randn(n, N, C),
                footprint=torch.rand(n, 1, FP, FP),
                height=torch.rand(n),
                region=torch.zeros(n, dtype=torch.long))


class TestProjectionSemantics(unittest.TestCase):
    def setUp(self):
        self.op = SetSDEdit(_net(), timesteps=100)
        self.b = _inputs()

    def test_zero_strength_is_a_no_op(self):
        """The defining property: at strength 0 the blockout is returned untouched, so projection
        degrades gracefully to 'trust the input' rather than inventing geometry."""
        out = self.op.project(strength=0.0, seed=0, **self.b)
        torch.testing.assert_close(out, self.b["blockout"], atol=1e-6, rtol=1e-6)

    def test_strength_is_a_dial_not_a_switch(self):
        base = self.b["blockout"]
        d = [ (self.op.project(strength=s, seed=0, **self.b) - base).norm().item()
              for s in (0.2, 0.5, 0.9) ]
        self.assertLess(d[0], d[1])
        self.assertLess(d[1], d[2])

    def test_noising_retains_blockout_information(self):
        """Where "projection, not sampling" actually lives, and the only part provable WITHOUT a
        trained model: the starting point keeps blockout signal in proportion to (1 - strength).

        Deliberately not asserted on `project`'s output. Whether the walk back lands near the blockout
        depends on the weights -- at strength 0.6 the start is already ~80% noise, and an untrained
        denoiser has no reason to return anywhere in particular. Testing that here would be testing
        the fixture, not the operator.
        """
        base = self.b["blockout"]
        corr = []
        for s in (0.1, 0.4, 0.8):
            x = self.op.noise_to(base, strength=s, seed=0)
            corr.append(torch.nn.functional.cosine_similarity(
                x.flatten(), base.flatten(), dim=0).item())
        self.assertGreater(corr[0], 0.9, "a light touch must leave the blockout nearly intact")
        self.assertGreater(corr[0], corr[1])
        self.assertGreater(corr[1], corr[2])

    def test_noising_at_full_strength_degenerates_to_noise(self):
        """Named honestly rather than hidden: at strength 1 the blockout term vanishes, which is the
        regime ADR 0003 rejects and the reason the operating point must sit below 1."""
        base = self.b["blockout"]
        x = self.op.noise_to(base, strength=1.0, seed=0)
        self.assertLess(abs(torch.nn.functional.cosine_similarity(
            x.flatten(), base.flatten(), dim=0).item()), 0.25)

    def test_output_shape_and_finiteness(self):
        out = self.op.project(strength=0.5, seed=0, **self.b)
        self.assertEqual(out.shape, self.b["blockout"].shape)
        self.assertTrue(torch.isfinite(out).all())

    def test_deterministic_given_a_seed(self):
        a = self.op.project(strength=0.6, seed=3, **self.b)
        c = self.op.project(strength=0.6, seed=3, **self.b)
        torch.testing.assert_close(a, c)

    def test_footprint_still_conditions_the_projection(self):
        a = self.op.project(strength=0.6, seed=0, **self.b)
        b2 = dict(self.b, footprint=1.0 - self.b["footprint"])
        c = self.op.project(strength=0.6, seed=0, **b2)
        self.assertGreater((a - c).abs().max().item(), 1e-5)

    def test_permutation_equivariance_survives_projection(self):
        """The set symmetry must survive the operator -- but it is a statement about the token set
        AND its noise, since noise is drawn per token. Permuting both must permute the output."""
        torch.manual_seed(11)
        noise = torch.randn(1, N, C)
        perm = torch.randperm(N)
        a = self.op.project(strength=0.5, noise=noise, **self.b)
        c = self.op.project(strength=0.5, noise=noise[:, perm],
                            **dict(self.b, blockout=self.b["blockout"][:, perm]))
        torch.testing.assert_close(a[:, perm], c, atol=1e-4, rtol=1e-4)


class TestGuidance(unittest.TestCase):
    def test_guidance_scale_changes_the_result(self):
        op, b = SetSDEdit(_net(), timesteps=100), _inputs()
        a = op.project(strength=0.6, seed=0, guidance=1.0, **b)
        c = op.project(strength=0.6, seed=0, guidance=3.0, **b)
        self.assertGreater((a - c).abs().max().item(), 1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
