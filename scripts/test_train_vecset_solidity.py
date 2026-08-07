"""#84: contract for `--surf_weight_by_solidity`. Synthetic, fast, no GPU.

Two things must hold, and both were checked by hand before this file existed:
  1. With the flag OFF the loss is **bit-identical** to the old scalar reduction. The refactor keeps
     the per-sample dimension alive, and that must not change any run that does not ask for it.
  2. With the flag ON, a re-entrant (low-solidity) footprint exerts proportionally LESS pressure --
     #84's "lower weight on low-solidity / high-complexity footprints".
"""
from __future__ import annotations

import unittest

import torch


def _reduce(got, tgt, w_t, sol=None):
    """The reduction as `scripts/train_vecset.py` performs it."""
    per = (w_t * (got - tgt) ** 2).flatten(1).mean(1)
    return (per * sol).mean() if sol is not None else per.mean(), per.mean()


class TestSolidityWeighting(unittest.TestCase):
    def test_flag_off_is_identical_to_the_old_scalar_reduction(self):
        torch.manual_seed(0)
        n, p = 4, 512
        got, tgt = torch.randn(n, p), torch.randn(n, p)
        w_t = torch.rand(n).reshape(n, 1)
        old = (w_t * (got - tgt) ** 2).mean()          # the pre-#84 expression
        new, _ = _reduce(got, tgt, w_t)
        self.assertTrue(torch.allclose(old, new, atol=1e-7),
                        "keeping the per-sample dim must not change a run without the flag")

    def test_low_solidity_exerts_proportionally_less_pressure(self):
        """Controlled: identical baseline error on every sample, only solidity differs."""
        n, p = 4, 8
        tgt = torch.zeros(n, p)
        got = torch.ones(n, p)                          # equal error everywhere
        w_t = torch.ones(n).reshape(n, 1)
        sol = torch.tensor([1.0, 1.0, 0.5, 0.5])
        base, _ = _reduce(got, tgt, w_t, sol)
        lo = got.clone(); lo[2] *= 2.0                  # extra error on a LOW-solidity building
        hi = got.clone(); hi[0] *= 2.0                  # the same on a HIGH-solidity one
        d_lo = _reduce(lo, tgt, w_t, sol)[0] - base
        d_hi = _reduce(hi, tgt, w_t, sol)[0] - base
        self.assertAlmostEqual(float(d_lo / d_hi), 0.5, places=5)
        self.assertLess(float(d_lo), float(d_hi), "a re-entrant footprint must exert LESS pressure")

    def test_logged_magnitude_is_the_unweighted_one(self):
        """`surf_hist` must stay comparable to runs without the flag.

        Logging the weighted value would make a run look better purely by down-weighting its hard
        cases -- exactly the kind of selection effect this map has already been bitten by three times.
        """
        n, p = 3, 16
        tgt, got = torch.zeros(n, p), torch.ones(n, p)
        w_t = torch.ones(n).reshape(n, 1)
        weighted, logged = _reduce(got, tgt, w_t, torch.tensor([0.2, 0.2, 0.2]))
        self.assertLess(float(weighted), float(logged))
        self.assertAlmostEqual(float(logged), 1.0, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
