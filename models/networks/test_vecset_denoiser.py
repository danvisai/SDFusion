"""Contract tests for the footprint-conditioned token-set denoiser (spec #67).

Written before the module, at the seam the codec contract established: the denoiser sees a **latent
token set**, never a grid, and its only conditioning is geometric (footprint + height + region). Tests
assert external behaviour -- shapes, that each conditioning signal actually changes the output, and that
the thing can learn -- never internals, so they survive architecture changes.

CPU-only and small; the overfit test is the one real smoke signal that the wiring is sound.

Run: env -u LD_PRELOAD ./sdfusion/bin/python models/networks/test_vecset_denoiser.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from models.networks.vecset_denoiser import VecsetDenoiser  # noqa: E402

B, N, C, FP = 2, 32, 16, 16          # tiny: batch, tokens, latent channels, footprint resolution


def _net(**kw):
    """A denoiser in a NON-DEGENERATE state.

    The module zero-inits its output projection and its adaLN gates -- standard practice, and what
    makes training stable -- so at initialisation it emits exactly zero and no conditioning could
    possibly show. That is correct for training and useless for testing, so the fixture perturbs those
    layers to stand in for "a model that has taken a few steps". Production init is untouched.
    """
    torch.manual_seed(0)
    net = VecsetDenoiser(latent_channels=C, width=64, depth=2, heads=4, footprint_res=FP, **kw)
    with torch.no_grad():
        net.out.weight.normal_(0, 0.05)
        for blk in net.blocks:
            blk.ada[-1].weight.normal_(0, 0.05)
    return net


def _batch(n=B):
    torch.manual_seed(1)
    return dict(x=torch.randn(n, N, C),
                t=torch.randint(0, 1000, (n,)),
                footprint=torch.rand(n, 1, FP, FP),
                height=torch.rand(n),
                region=torch.randint(0, 3, (n,)))


class TestShapes(unittest.TestCase):
    def test_output_matches_input_token_set(self):
        net, b = _net(), _batch()
        out = net(**b)
        self.assertEqual(out.shape, b["x"].shape)
        self.assertTrue(torch.isfinite(out).all())

    def test_variable_token_count(self):
        """A token set has no fixed length -- the model must not bake one in."""
        net = _net()
        for n_tok in (8, 32, 64):
            x = torch.randn(1, n_tok, C)
            out = net(x=x, t=torch.tensor([5]), footprint=torch.rand(1, 1, FP, FP),
                      height=torch.rand(1), region=torch.zeros(1, dtype=torch.long))
            self.assertEqual(out.shape, (1, n_tok, C))

    def test_permutation_equivariance(self):
        """The latent is a SET: permuting tokens must permute the output, not change it.

        This is the property that distinguishes a token-set model from a grid model, and the reason
        no positional encoding is applied to the token axis.
        """
        net = _net().eval()
        b = _batch(1)
        with torch.no_grad():
            a = net(**b)
            perm = torch.randperm(N)
            b2 = dict(b, x=b["x"][:, perm])
            c = net(**b2)
        torch.testing.assert_close(a[:, perm], c, atol=1e-5, rtol=1e-5)


class TestConditioningIsUsed(unittest.TestCase):
    """Every conditioning signal must measurably change the output, or it is decorative."""

    def _differs(self, net, b1, b2):
        with torch.no_grad():
            return (net(**b1) - net(**b2)).abs().max().item()

    def test_timestep_changes_output(self):
        net, b = _net().eval(), _batch()
        b2 = dict(b, t=b["t"] + 500)
        self.assertGreater(self._differs(net, b, b2), 1e-4)

    def test_footprint_changes_output(self):
        net, b = _net().eval(), _batch()
        b2 = dict(b, footprint=1.0 - b["footprint"])
        self.assertGreater(self._differs(net, b, b2), 1e-4,
                           "footprint conditioning is not reaching the output")

    def test_height_changes_output(self):
        net, b = _net().eval(), _batch()
        b2 = dict(b, height=b["height"] + 0.5)
        self.assertGreater(self._differs(net, b, b2), 1e-5)

    def test_region_changes_output(self):
        net, b = _net().eval(), _batch()
        b2 = dict(b, region=(b["region"] + 1) % 3)
        self.assertGreater(self._differs(net, b, b2), 1e-5)

    def test_null_conditioning_is_available_for_cfg(self):
        """Classifier-free guidance needs an explicit unconditional path."""
        net, b = _net().eval(), _batch()
        with torch.no_grad():
            cond = net(**b)
            unc = net(**dict(b, drop_cond=True))
        self.assertGreater((cond - unc).abs().max().item(), 1e-4)


class TestItLearns(unittest.TestCase):
    """The smoke signal that the wiring is actually sound: it can overfit one batch."""

    def test_overfits_a_single_batch(self):
        net = _net()
        b = _batch()
        target = torch.randn_like(b["x"])
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        first = last = None
        for step in range(120):
            loss = torch.nn.functional.mse_loss(net(**b), target)
            opt.zero_grad(); loss.backward(); opt.step()
            if step == 0:
                first = loss.item()
            last = loss.item()
        self.assertLess(last, first * 0.5, f"loss did not fall: {first:.4f} -> {last:.4f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
