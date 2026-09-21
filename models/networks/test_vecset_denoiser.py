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

from models.networks.vecset_denoiser import (  # noqa: E402
    VecsetDenoiser, denoiser_from_checkpoint, region_width_of,
)

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


class TestRegionWidth(unittest.TestCase):
    """#188: the region embedding's width is a corpus property, not a module constant.

    The frozen A2 source (`vecset_v4_surf` @240k) declares `region = nn.Embedding(3, 512)` because
    `n_regions` defaulted to 3 and no call site ever overrode it. Every BuildingWorld row carries
    region id 3-8, so that checkpoint raises `IndexError: index out of range in self` on the corpus
    this effort has moved to -- a structural block, not a quality gap. These tests pin the two ways
    out: a wider channel, and no channel at all.
    """

    def test_a_wider_channel_accepts_the_new_corpus_ids(self):
        net = _net(n_regions=9).eval()
        b = _batch()
        for rid in range(9):                      # 0-2 legacy, 3-8 BuildingWorld style buckets
            out = net(**dict(b, region=torch.full((B,), rid, dtype=torch.long)))
            self.assertTrue(torch.isfinite(out).all(), f"region id {rid} did not produce a finite output")

    def test_the_default_width_still_rejects_a_buildingworld_id(self):
        """The frozen source's failure mode, pinned so the fix cannot be claimed without the flag."""
        net = _net().eval()
        with self.assertRaises(IndexError):
            net(**dict(_batch(), region=torch.full((B,), 3, dtype=torch.long)))


class TestRegionFree(unittest.TestCase):
    """#188: `n_regions=0` is region conditioning REMOVED, not widened.

    Owner direction, 2026-09-18. This is a distinct conditioning variant -- the same shape as the
    height-map family's arm 3 (`--drop_region`, `n_regions=0`), which is a fifth channel variant
    rather than arm 2 with a flag flipped. `forward` already tolerated `region=None`, but a model
    that still CARRIES an untrained embedding it is never handed is the combination #119 rejected as
    "generating from an untrained mode"; the parameter has to be absent, and the absence has to be
    legible in the checkpoint.
    """

    def test_carries_no_region_parameter_at_all(self):
        net = _net(n_regions=0)
        self.assertIsNone(net.region)
        keys = [k for k in net.state_dict() if "region" in k]
        self.assertEqual(keys, [], f"a region-free checkpoint still carries {keys}")

    def test_forwards_without_a_region(self):
        net = _net(n_regions=0).eval()
        b = dict(_batch()); b.pop("region")
        out = net(**b)
        self.assertEqual(out.shape, b["x"].shape)
        self.assertTrue(torch.isfinite(out).all())

    def test_footprint_and_height_still_condition(self):
        """Region-free is not condition-free: the footprint-ONLY claim rests on these two."""
        net = _net(n_regions=0).eval()
        b = dict(_batch()); b.pop("region")
        with torch.no_grad():
            base = net(**b)
            fp = net(**dict(b, footprint=1.0 - b["footprint"]))
            ht = net(**dict(b, height=b["height"] + 0.5))
        self.assertGreater((base - fp).abs().max().item(), 1e-4, "footprint stopped conditioning")
        self.assertGreater((base - ht).abs().max().item(), 1e-5, "height stopped conditioning")

    def test_being_handed_a_region_is_an_error_not_a_silent_no_op(self):
        """A caller that still passes region ids is wrong about which model it holds.

        Silently ignoring them is the expensive failure: a gate would score a region-conditioned
        harness against a region-free checkpoint and report the difference as a corpus result.
        """
        net = _net(n_regions=0).eval()
        with self.assertRaises(ValueError):
            net(**_batch())

    def test_classifier_free_guidance_still_has_its_unconditional_path(self):
        net = _net(n_regions=0).eval()
        b = dict(_batch()); b.pop("region")
        with torch.no_grad():
            cond, unc = net(**b), net(**dict(b, drop_cond=True))
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


def _blob(net, **extra):
    """The shape `train_vecset.py` actually saves."""
    blob = {"model": net.state_dict(), "step": 1000, "latent_channels": C, "footprint_res": FP,
            "args": {"width": 64, "depth": 2, "heads": 4}}
    blob.update(extra)
    return blob


class TestRegionWidthOfACheckpoint(unittest.TestCase):
    """#188: how wide is this checkpoint's region channel? Read it off the WEIGHTS.

    Every consumer -- the eval harness, the town service, #119's cache path -- rebuilds a net from a
    saved blob, and each one currently hardcodes the 3-region default. That is precisely the defect
    #188 exists to remove, so the answer has to come from one place.
    """

    def test_reads_a_declared_width(self):
        self.assertEqual(region_width_of(_blob(_net(n_regions=9), n_regions=9)), 9)

    def test_reads_a_region_free_declaration(self):
        self.assertEqual(region_width_of(_blob(_net(n_regions=0), n_regions=0)), 0)

    def test_infers_the_width_of_a_checkpoint_saved_before_the_key_existed(self):
        """Every checkpoint on disk today predates `n_regions`, including the frozen A2 source."""
        self.assertEqual(region_width_of(_blob(_net())), 3)

    def test_infers_region_free_from_the_absence_of_the_weight(self):
        self.assertEqual(region_width_of(_blob(_net(n_regions=0))), 0)

    def test_refuses_a_declaration_its_own_weights_contradict(self):
        """A blob claiming 9 regions while carrying a 3-wide embedding is not safe to guess at."""
        with self.assertRaises(ValueError):
            region_width_of(_blob(_net(n_regions=3), n_regions=9))


class TestRebuildingFromACheckpoint(unittest.TestCase):
    def test_round_trips_a_region_free_checkpoint(self):
        net = _net(n_regions=0)
        rebuilt = denoiser_from_checkpoint(_blob(net))
        self.assertIsNone(rebuilt.region)
        b = dict(_batch()); b.pop("region")
        with torch.no_grad():
            torch.testing.assert_close(net.eval()(**b), rebuilt.eval()(**b))

    def test_round_trips_a_region_conditioned_checkpoint(self):
        net = _net(n_regions=9)
        rebuilt = denoiser_from_checkpoint(_blob(net, n_regions=9))
        b = dict(_batch(), region=torch.full((B,), 7, dtype=torch.long))
        with torch.no_grad():
            torch.testing.assert_close(net.eval()(**b), rebuilt.eval()(**b))

    def test_rebuilds_the_frozen_sources_shape_without_a_declaration(self):
        rebuilt = denoiser_from_checkpoint(_blob(_net()))
        self.assertEqual(rebuilt.n_regions, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
