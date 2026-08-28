"""Contract tests for #126's measurement. Synthetic, fast, no GPU, no corpus.

#126 asks whether the paired metric's optimum is the envelope rather than a real building. Its
evidence is a *constructed* arm -- one real building offered as the answer for another -- and the
whole decision turns on that construction being fair. So what is pinned here is exactly the two
places it could be unfair:

  * `matched_pairs` selects the population. If it lets through pairs that are not plausible answers
    for each other, the arm scores badly for the wrong reason.
  * `transplant_height` renders the alternative. If it does not come out footprint-exact and at the
    conditioned height, the arm is charged for errors a footprint-conditioned generator cannot make,
    and #126's number would be an artefact of the construction rather than a fact about the metric.

Run: env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/test_measure_scoring_optimum.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.foundations.eval_massing_arms import RES  # noqa: E402
from scripts.foundations.measure_scoring_optimum import (  # noqa: E402
    compare_to_envelope, matched_pairs, score_pair, transplant_height,
)


def _rect(res, z0, z1, x0, x1):
    m = np.zeros((res, res), bool)
    m[z0:z1, x0:x1] = True
    return m


class TestMatchedPairs(unittest.TestCase):
    """The population: only buildings that are plausible answers for each other may enter."""

    def test_identical_footprint_and_height_pairs(self):
        fp = _rect(16, 4, 12, 4, 12)
        fps = np.stack([fp, fp.copy()])
        self.assertEqual(matched_pairs(fps, np.array([10, 10])), [(0, 1)])

    def test_disjoint_footprints_never_pair(self):
        fps = np.stack([_rect(16, 0, 4, 0, 4), _rect(16, 10, 14, 10, 14)])
        self.assertEqual(matched_pairs(fps, np.array([10, 10])), [])

    def test_footprint_iou_below_threshold_is_excluded(self):
        """IoU 8/16 = 0.5, well under the 0.90 the ticket specifies."""
        fps = np.stack([_rect(16, 0, 4, 0, 4), _rect(16, 0, 4, 2, 6)])
        self.assertEqual(matched_pairs(fps, np.array([10, 10])), [])

    def test_height_outside_tolerance_is_excluded(self):
        """Same footprint, but one is twice as tall -- not an answer for the other."""
        fp = _rect(16, 4, 12, 4, 12)
        fps = np.stack([fp, fp.copy()])
        self.assertEqual(matched_pairs(fps, np.array([10, 20])), [])
        self.assertEqual(matched_pairs(fps, np.array([20, 21])), [(0, 1)])

    def test_pairs_are_unordered_and_self_pairs_excluded(self):
        fp = _rect(16, 4, 12, 4, 12)
        fps = np.stack([fp, fp.copy(), fp.copy()])
        self.assertEqual(matched_pairs(fps, np.array([10, 10, 10])),
                         [(0, 1), (0, 2), (1, 2)])


class TestTransplantHeight(unittest.TestCase):
    """The arm: an alternative roof, rendered as a footprint-conditioned generator would have to."""

    def test_transplant_onto_itself_is_the_identity(self):
        """The one case with a known answer. If this drifts, every number below is unanchored."""
        fp = _rect(16, 4, 12, 4, 12)
        h = np.where(fp, np.arange(16, dtype=np.int16)[:, None] % 7 + 1, 0).astype(np.int16)
        out = transplant_height(h, fp, 10, fp, 10)
        np.testing.assert_array_equal(out, h)

    def test_output_is_footprint_exact(self):
        """Criterion 2 is free by construction: the arm is 0 off the target footprint, >=1 on it."""
        src, dst = _rect(16, 4, 12, 4, 12), _rect(16, 5, 13, 5, 13)
        h = np.where(src, 6, 0).astype(np.int16)
        out = transplant_height(h, src, 10, dst, 10)
        self.assertFalse(out[~dst].any())
        self.assertTrue((out[dst] >= 1).all())

    def test_profile_is_rescaled_to_the_conditioned_height(self):
        """Height is a user input (#81), so only the ROOF SHAPE is borrowed, never the height."""
        fp = _rect(16, 4, 12, 4, 12)
        h = np.where(fp, 4, 0).astype(np.int16)          # half of an extent of 8
        out = transplant_height(h, fp, 8, fp, 16)         # same shape, twice as tall
        self.assertEqual(int(out[fp].max()), 8)

    def test_never_exceeds_the_conditioned_extent(self):
        fp = _rect(16, 4, 12, 4, 12)
        h = np.where(fp, 9, 0).astype(np.int16)
        self.assertEqual(int(transplant_height(h, fp, 9, fp, 5)[fp].max()), 5)

    def test_uncovered_cells_are_filled_from_the_nearest_source(self):
        """A dst cell the source footprint misses must still get a roof, not a hole."""
        src, dst = _rect(16, 4, 8, 4, 8), _rect(16, 4, 12, 4, 12)
        h = np.where(src, 7, 0).astype(np.int16)
        out = transplant_height(h, src, 10, dst, 10)
        self.assertTrue((out[dst] == 7).all())


class TestScorePairLadder(unittest.TestCase):
    """The ladder: each rung must be a fair offer, and the anchor rung must be exact."""

    def _building(self, bid, fp, y0, h):
        return dict(id=bid, fp=fp, y0=y0,
                    target=np.where(fp, h, 0).astype(np.int16), extent=h)

    def test_a_building_offered_as_its_own_answer_is_perfect_on_every_rung(self):
        """No rung may cost anything when the alternative IS the held-out building."""
        b = self._building(1, _rect(RES, 4, 12, 4, 12), 3, 8)
        row = score_pair(b, b)
        for arm in ("alt_raw", "alt_aligned", "alt_exact"):
            self.assertEqual(row[arm]["vol_iou"], 1.0, arm)

    def test_the_blockout_rung_is_the_envelope(self):
        """It fills every footprint column to the conditioned height: it can only ever over-fill."""
        fp = _rect(RES, 4, 12, 4, 12)
        a = dict(id=1, fp=fp, y0=3, extent=8,
                 target=np.where(fp, 5, 0).astype(np.int16))
        row = score_pair(a, a)
        self.assertEqual(row["blockout"]["missing"], 0.0)
        self.assertGreater(row["blockout"]["extra"], 0.0)
        self.assertTrue(row["carve_needed"])

    def test_base_placement_alone_costs_the_raw_rung(self):
        """Why `alt_aligned` exists: two identical buildings at different y0 are not identical here."""
        fp = _rect(RES, 4, 12, 4, 12)
        a, b = self._building(1, fp, 3, 8), self._building(2, fp, 9, 8)
        row = score_pair(a, b)
        self.assertLess(row["alt_raw"]["vol_iou"], 1.0)
        self.assertEqual(row["alt_aligned"]["vol_iou"], 1.0)


class TestCompareToEnvelope(unittest.TestCase):
    """`extra` is better when SMALLER and IoU when LARGER, so the direction argument is load-bearing.

    Ties are counted apart from losses: #126's "coin flip" reading came from pooling them.
    """

    ROWS = [
        {"blockout": {"vol_iou": 0.80, "extra": 0.20}, "alt": {"vol_iou": 0.90, "extra": 0.10}},
        {"blockout": {"vol_iou": 0.80, "extra": 0.20}, "alt": {"vol_iou": 0.70, "extra": 0.30}},
        {"blockout": {"vol_iou": 0.80, "extra": 0.20}, "alt": {"vol_iou": 0.80, "extra": 0.20}},
    ]

    def test_higher_is_better_counts_only_strict_wins(self):
        c = compare_to_envelope(self.ROWS, "alt", "vol_iou", True)
        self.assertEqual((c["wins"], c["losses"], c["ties"]), (1, 1, 1))
        self.assertAlmostEqual(c["rate"], 1 / 3)

    def test_lower_is_better_inverts_the_comparison(self):
        c = compare_to_envelope(self.ROWS, "alt", "extra", False)
        self.assertEqual((c["wins"], c["losses"], c["ties"]), (1, 1, 1))
        self.assertAlmostEqual(c["rate"], 1 / 3)

    def test_ties_are_excluded_from_the_decided_rate(self):
        """The correction #126's framing needed: 1 win and 1 loss is 50%, not 33%."""
        c = compare_to_envelope(self.ROWS, "alt", "vol_iou", True)
        self.assertAlmostEqual(c["rate_ex_ties"], 0.5)

    def test_a_tie_is_not_a_win_in_either_direction(self):
        tie = [self.ROWS[2]]
        for key, hib in (("vol_iou", True), ("extra", False)):
            c = compare_to_envelope(tie, "alt", key, hib)
            self.assertEqual((c["wins"], c["ties"]), (0, 1))
            self.assertEqual(c["rate"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
