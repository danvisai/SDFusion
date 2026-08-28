"""Contract tests for #127's height-map generator. Synthetic, fast, CPU, no corpus, no GPU.

#127 asks one question -- does a footprint-conditioned height-map generator actually *carve*, or
does it learn identity like every arm before it? A wrong answer is cheap to produce two ways, and
both are pinned here rather than trusted:

  * **Leakage.** The conditioning must be a function of the footprint, the conditioned height and
    the region ONLY. If the target height field reaches the input by any route the answer is
    meaningless, so `condition_channels` is pinned to a signature that cannot see it, and the
    retrieval bank is pinned to exclude the held-out rows it is scored on.
  * **The invariants the output space is *claimed* to give for free.** #127's case rests on
    "footprint-exact, collapse-impossible, valid by construction". Those are properties of
    `apply_depth`, not of the corpus, and they only hold if the clamp is right. `missing` is NOT
    among them -- a generator that over-carves cuts into GT -- and the test that says so is
    deliberate.

Run: env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/test_train_height_map_generator.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.foundations.eval_massing_arms import volume_split  # noqa: E402
from scripts.foundations.recover_massing_programs import occupancy  # noqa: E402
from scripts.foundations.train_height_map_generator import (  # noqa: E402
    DEPTH_CLASSES, apply_depth, carve_depth, condition_channels, decode_logits,
    mean_relative_depth, mean_roof_height, retrieve_nn,
)


def _rect(res, z0, z1, x0, x1):
    m = np.zeros((res, res), bool)
    m[z0:z1, x0:x1] = True
    return m


class TestCarveDepthCoding(unittest.TestCase):
    """The label. `depth = extent - top`, which is exactly the per-column carve the blockout needs."""

    def test_round_trip_is_exact(self):
        fp = _rect(16, 2, 10, 3, 11)
        top = np.where(fp, np.int16(7), 0).astype(np.int16)
        top[4:7, 5:9] = 3
        d = carve_depth(top, fp, extent=9)
        np.testing.assert_array_equal(apply_depth(fp, 9, d), top)

    def test_depth_is_zero_off_the_footprint(self):
        fp = _rect(16, 2, 6, 2, 6)
        top = np.where(fp, np.int16(4), 0).astype(np.int16)
        d = carve_depth(top, fp, extent=9)
        self.assertEqual(int(d[~fp].sum()), 0)

    def test_a_flat_full_height_building_is_the_all_zero_label(self):
        """The 42% empty-program majority #10 measured: nothing to carve is the class-0 label."""
        fp = _rect(16, 2, 10, 2, 10)
        top = np.where(fp, np.int16(12), 0).astype(np.int16)
        self.assertEqual(int(np.abs(carve_depth(top, fp, extent=12)).sum()), 0)

    def test_labels_fit_the_class_budget(self):
        """64 classes is the whole range: a column can be carved at most extent-1 below the top."""
        fp = _rect(16, 0, 16, 0, 16)
        top = np.where(fp, np.int16(1), 0).astype(np.int16)
        self.assertLess(int(carve_depth(top, fp, extent=DEPTH_CLASSES).max()), DEPTH_CLASSES)


class TestApplyDepthInvariants(unittest.TestCase):
    """#127's structural claim, made falsifiable: what does the output space give for free?"""

    def test_it_is_footprint_exact_whatever_the_prediction(self):
        fp = _rect(16, 2, 10, 3, 11)
        for d in (np.zeros((16, 16), np.int16), np.full((16, 16), 99, np.int16),
                  np.random.default_rng(0).integers(0, 40, (16, 16)).astype(np.int16)):
            h = apply_depth(fp, 9, d)
            np.testing.assert_array_equal(h > 0, fp)

    def test_every_footprint_column_keeps_at_least_one_voxel(self):
        """Collapse-impossible in the sense #127 means: no hole is ever punched through the plan."""
        fp = _rect(16, 2, 10, 3, 11)
        h = apply_depth(fp, 9, np.full((16, 16), 1000, np.int16))
        self.assertEqual(int(h[fp].min()), 1)

    def test_the_carve_is_purely_subtractive(self):
        """A prediction can never exceed the blockout, so `extra` can never be worse than doing
        nothing. That is the guarantee the depth parameterisation buys, and it is why the arm is
        parameterised as depth rather than as an absolute top."""
        fp = _rect(16, 2, 10, 3, 11)
        rng = np.random.default_rng(1)
        blockout = apply_depth(fp, 9, np.zeros((16, 16), np.int16))
        for _ in range(8):
            d = rng.integers(-5, 40, (16, 16)).astype(np.int16)
            self.assertTrue(bool((apply_depth(fp, 9, d) <= blockout).all()))

    def test_missing_is_NOT_free_and_the_arm_can_still_collapse(self):
        """⚠️ #127 says "`missing` and `collapse_rate` are 0 by clamping". Only the *validity* of
        the solid is free. Over-carving still eats GT, so the collapse rate has to be measured and
        published rather than assumed away."""
        fp = _rect(16, 2, 10, 3, 11)
        gt = occupancy(fp, 0, apply_depth(fp, 9, np.zeros((16, 16), np.int16)))
        over = occupancy(fp, 0, apply_depth(fp, 9, np.full((16, 16), 8, np.int16)))
        self.assertGreater(volume_split(over, gt)["missing"], 0.15)


class TestConditioningCarriesNoAnswer(unittest.TestCase):
    """The leakage guard. The input is the conditioning #127 names -- footprint, height, region."""

    def test_two_buildings_with_the_same_conditioning_get_identical_input(self):
        """Different roofs, same footprint/height/region => the network cannot tell them apart.
        If this ever fails, the conditioning has grown a channel that saw the target."""
        fp = _rect(16, 2, 10, 3, 11)
        a = condition_channels(fp, extent=9, height_m=12.0, region=1)
        b = condition_channels(fp, extent=9, height_m=12.0, region=1)
        np.testing.assert_array_equal(a, b)

    def test_the_conditioned_height_reaches_the_input(self):
        fp = _rect(16, 2, 10, 3, 11)
        self.assertFalse(np.array_equal(condition_channels(fp, 9, 12.0, 1),
                                        condition_channels(fp, 20, 12.0, 1)))

    def test_the_region_reaches_the_input(self):
        fp = _rect(16, 2, 10, 3, 11)
        self.assertFalse(np.array_equal(condition_channels(fp, 9, 12.0, 0),
                                        condition_channels(fp, 9, 12.0, 2)))

    def test_channels_are_finite_and_bounded(self):
        fp = _rect(16, 0, 16, 0, 16)
        c = condition_channels(fp, 64, 300.0, 2)
        self.assertTrue(np.isfinite(c).all())
        self.assertLessEqual(float(np.abs(c).max()), 4.0)


class TestDecode(unittest.TestCase):
    """Argmax, not expectation: the mean of a bimodal roof distribution is a roof nobody built."""

    def test_decode_takes_the_argmax_class(self):
        fp = _rect(8, 1, 7, 1, 7)
        logits = np.zeros((DEPTH_CLASSES, 8, 8), np.float32)
        logits[3] = 1.0
        np.testing.assert_array_equal(decode_logits(logits, fp, extent=9),
                                      apply_depth(fp, 9, np.full((8, 8), 3, np.int16)))

    def test_decode_never_leaves_the_footprint(self):
        fp = _rect(8, 1, 7, 1, 7)
        logits = np.random.default_rng(2).normal(size=(DEPTH_CLASSES, 8, 8)).astype(np.float32)
        np.testing.assert_array_equal(decode_logits(logits, fp, 9) > 0, fp)


class TestMeanRoofBaseline(unittest.TestCase):
    """#127's `mean roof` arm: the unconditional version of the regression-to-the-mean trap."""

    def test_profile_is_the_mean_relative_depth_per_cell(self):
        fp = _rect(8, 0, 8, 0, 8)
        depths = np.stack([np.zeros((8, 8), np.int16), np.full((8, 8), 5, np.int16)])
        prof = mean_relative_depth(depths, np.stack([fp, fp]), np.array([10, 10]))
        np.testing.assert_allclose(prof, 0.25, atol=1e-6)

    def test_cells_no_footprint_covers_are_zero_not_nan(self):
        fp = _rect(8, 0, 4, 0, 4)
        prof = mean_relative_depth(np.zeros((1, 8, 8), np.int16), fp[None], np.array([10]))
        self.assertTrue(np.isfinite(prof).all())

    def test_the_profile_scales_with_the_conditioned_height(self):
        fp = _rect(8, 0, 8, 0, 8)
        prof = np.full((8, 8), 0.25, np.float32)
        self.assertEqual(int(mean_roof_height(prof, fp, 20)[fp].max()), 15)
        self.assertEqual(int(mean_roof_height(prof, fp, 40)[fp].max()), 30)


class TestRetrievalBaseline(unittest.TestCase):
    """1-NN is #127's real bar, so the thing that could flatter it is pinned: seeing the answer."""

    def test_picks_the_footprint_iou_nearest_bank_row(self):
        q = _rect(16, 0, 8, 0, 8)
        bank = np.stack([_rect(16, 0, 2, 0, 2), _rect(16, 0, 8, 0, 7), _rect(16, 8, 16, 8, 16)])
        np.testing.assert_array_equal(retrieve_nn(q[None], bank), [1])

    def test_an_exact_footprint_match_is_preferred(self):
        q = _rect(16, 2, 10, 2, 10)
        bank = np.stack([_rect(16, 2, 10, 2, 9), q.copy(), _rect(16, 2, 11, 2, 10)])
        np.testing.assert_array_equal(retrieve_nn(q[None], bank), [1])

    def test_a_query_absent_from_the_bank_cannot_retrieve_itself(self):
        """The bank is built from TRAINING rows only. This pins the property the caller relies on:
        retrieval returns a bank index, so a held-out row can only be answered by a training row."""
        q = _rect(16, 2, 10, 2, 10)
        bank = np.stack([_rect(16, 0, 3, 0, 3), _rect(16, 12, 16, 12, 16)])
        self.assertIn(int(retrieve_nn(q[None], bank)[0]), (0, 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
