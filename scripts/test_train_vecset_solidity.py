"""#84: contract for the per-sample surface weighting. Synthetic, fast, no GPU.

⚠️ These tests import `surface_term` from `scripts.train_vecset` and call **the production function**.
The first version of this file re-implemented the reduction locally and asserted against the copy. It
passed while the shipped code was an exact no-op at the flag values actually used, and the copy is why
nobody noticed for two runs. A test that re-implements its subject tests nothing.
"""
from __future__ import annotations

import unittest

import torch

from scripts.train_vecset import surface_term


def _mk(n, p=16, err=1.0):
    """(got, tgt, w_t) with identical per-sample error, so only the weighting varies."""
    return torch.full((n, p), err), torch.zeros(n, p), torch.ones(n).reshape(n, 1)


class TestSurfaceTermNoWeighting(unittest.TestCase):
    def test_matches_the_plain_scalar_reduction(self):
        torch.manual_seed(0)
        got, tgt = torch.randn(4, 512), torch.randn(4, 512)
        w_t = torch.rand(4).reshape(4, 1)
        plain = (w_t * (got - tgt) ** 2).mean()               # the pre-#84 expression
        weighted, logged = surface_term(got, tgt, w_t)
        self.assertTrue(torch.allclose(plain, weighted, atol=1e-7))
        self.assertTrue(torch.allclose(plain, logged, atol=1e-7))


class TestSurfaceTermWeighting(unittest.TestCase):
    """The properties #84 asked for, checked on the real function."""

    def test_weighting_still_applies_at_surf_bs_1(self):
        """🔑 The regression this file exists for.

        `--surf_bs` defaults to 1, so exactly one sample carries the term. The original renormaliser
        divided by the mean over that selection -- a 1-element mean -- making the weight identically
        1.0 and the whole flag a no-op. Normalising by the CORPUS mean fixes it, and this is the test
        that would have caught it.
        """
        got, tgt, w_t = _mk(1)
        low, _ = surface_term(got, tgt, w_t, torch.tensor([0.4]), norm=0.8)
        high, _ = surface_term(got, tgt, w_t, torch.tensor([1.2]), norm=0.8)
        self.assertNotAlmostEqual(float(low), float(high),
                                  msg="weighting must still act when only one sample is selected")
        self.assertLess(float(low), float(high))

    def test_a_below_average_weight_exerts_less_pressure(self):
        """#84: 'lower weight on low-solidity / high-complexity footprints'."""
        got, tgt, w_t = _mk(4)
        w = torch.tensor([1.0, 1.0, 0.5, 0.5])
        base, _ = surface_term(got, tgt, w_t, w, norm=0.75)
        lo = got.clone(); lo[2] *= 2.0        # extra error on a LOW-weight sample
        hi = got.clone(); hi[0] *= 2.0        # the same on a HIGH-weight one
        d_lo = surface_term(lo, tgt, w_t, w, norm=0.75)[0] - base
        d_hi = surface_term(hi, tgt, w_t, w, norm=0.75)[0] - base
        self.assertLess(float(d_lo), float(d_hi))
        self.assertAlmostEqual(float(d_lo / d_hi), 0.5, places=5)

    def test_corpus_normalisation_preserves_total_pressure(self):
        """Redistribute, do not reduce.

        Raw per-region weights average 0.58. Applied unnormalised, the flag is also a 42% cut in
        --surf_weight and a gain cannot be attributed to either. Normalising by the corpus mean keeps
        the magnitude and changes only the distribution.
        """
        got, tgt, w_t = _mk(6)
        w = torch.tensor([0.387, 0.387, 0.574, 0.574, 0.779, 0.779])
        corpus_mean = float(w.mean())
        flat, _ = surface_term(got, tgt, w_t)
        raw, _ = surface_term(got, tgt, w_t, w, norm=None)
        norm, _ = surface_term(got, tgt, w_t, w, norm=corpus_mean)
        self.assertLess(float(raw), float(flat) * 0.7)          # unnormalised is a large silent cut
        self.assertAlmostEqual(float(norm), float(flat), places=5)

    def test_normalisation_does_not_depend_on_the_selected_window(self):
        """⚠️ The other half of the original bug: the divisor was the mean over `sel`.

        A window that happened to be all-low-weight got weight 1.0 across the board, so the same
        building was penalised differently depending on who it was batched with. A corpus constant
        cannot do that.
        """
        got, tgt, w_t = _mk(1)
        a, _ = surface_term(got, tgt, w_t, torch.tensor([0.387]), norm=0.58)
        got2, tgt2, w_t2 = _mk(3)
        b, _ = surface_term(got2, tgt2, w_t2, torch.tensor([0.387, 0.387, 0.387]), norm=0.58)
        self.assertAlmostEqual(float(a), float(b), places=6)

    def test_logged_magnitude_is_always_the_unweighted_one(self):
        """`surf_hist` must stay comparable to runs without the flag."""
        got, tgt, w_t = _mk(3)
        weighted, logged = surface_term(got, tgt, w_t, torch.full((3,), 0.2), norm=1.0)
        self.assertLess(float(weighted), float(logged))
        self.assertAlmostEqual(float(logged), 1.0, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
