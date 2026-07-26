"""Contract tests for ticket 17's two-AFC study: pure sampling/blinding/analysis seams.

Fast + data-free: exercises the pure functions only, no rendering, no server, no GPU. The
render/collection pipeline itself is verified separately by a small real prototype batch (see the
ticket answer), matching this project's established convention for GPU-dependent code.

Run: env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
     scripts/eval/test_two_afc_study.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/eval
import two_afc_study as afc  # noqa: E402


class SamplePairIdsTest(unittest.TestCase):
    def test_deterministic_for_same_seed(self):
        ids = [f"b{i}" for i in range(50)]
        a = afc.sample_pair_ids(ids, 10, seed=3)
        b = afc.sample_pair_ids(ids, 10, seed=3)
        self.assertEqual(a, b)

    def test_returns_requested_count(self):
        ids = [f"b{i}" for i in range(50)]
        out = afc.sample_pair_ids(ids, 10, seed=0)
        self.assertEqual(len(out), 10)
        self.assertEqual(len(set(out)), 10)  # no duplicates

    def test_returns_all_when_n_exceeds_population(self):
        ids = [f"b{i}" for i in range(5)]
        out = afc.sample_pair_ids(ids, 20, seed=0)
        self.assertEqual(sorted(out), sorted(ids))

    def test_sampled_ids_are_a_subset_of_the_population(self):
        ids = [f"b{i}" for i in range(50)]
        out = afc.sample_pair_ids(ids, 10, seed=1)
        self.assertTrue(set(out) <= set(ids))


class AssignBlindSidesTest(unittest.TestCase):
    def test_each_building_gets_both_arms_exactly_once(self):
        ids = [f"b{i}" for i in range(20)]
        out = afc.assign_blind_sides(ids, seed=0)
        for row in out:
            self.assertEqual({row["left"], row["right"]}, {"monolith", "decomposition"})

    def test_deterministic_for_same_seed(self):
        ids = [f"b{i}" for i in range(20)]
        a = afc.assign_blind_sides(ids, seed=5)
        b = afc.assign_blind_sides(ids, seed=5)
        self.assertEqual(a, b)

    def test_produces_a_mix_of_left_and_right_over_enough_buildings(self):
        ids = [f"b{i}" for i in range(40)]
        out = afc.assign_blind_sides(ids, seed=0)
        lefts = [row["left"] for row in out]
        self.assertIn("monolith", lefts)
        self.assertIn("decomposition", lefts)

    def test_covers_every_building_exactly_once(self):
        ids = [f"b{i}" for i in range(20)]
        out = afc.assign_blind_sides(ids, seed=0)
        self.assertEqual([row["building"] for row in out], ids)


class WilsonCiTest(unittest.TestCase):
    def test_returns_point_estimate_matching_plain_proportion(self):
        phat, lo, hi = afc.wilson_ci(7, 10)
        self.assertAlmostEqual(phat, 0.7)
        self.assertLess(lo, phat)
        self.assertGreater(hi, phat)

    def test_ci_is_symmetric_around_point_when_k_is_half_n(self):
        phat, lo, hi = afc.wilson_ci(5, 10)
        self.assertAlmostEqual(phat, 0.5)
        self.assertAlmostEqual((lo + hi) / 2, phat, places=2)

    def test_handles_zero_n_without_crashing(self):
        phat, lo, hi = afc.wilson_ci(0, 0)
        self.assertEqual((phat, lo, hi), (0.0, 0.0, 1.0))

    def test_bounds_stay_within_zero_one(self):
        phat, lo, hi = afc.wilson_ci(10, 10)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)

    def test_narrower_ci_for_larger_n_at_same_proportion(self):
        _, lo_small, hi_small = afc.wilson_ci(7, 10)
        _, lo_large, hi_large = afc.wilson_ci(70, 100)
        self.assertLess(hi_large - lo_large, hi_small - lo_small)


class TwoAfcResultTest(unittest.TestCase):
    def setUp(self):
        self.answer_key = [
            dict(building="b0", left="monolith", right="decomposition"),
            dict(building="b1", left="decomposition", right="monolith"),
            dict(building="b2", left="monolith", right="decomposition"),
            dict(building="b3", left="decomposition", right="monolith"),
        ]

    def test_counts_preferences_correctly(self):
        # b0: picked right -> decomposition. b1: picked left -> decomposition.
        # b2: picked left -> monolith. b3: picked right -> monolith.
        responses = {"b0": "right", "b1": "left", "b2": "left", "b3": "right"}
        out = afc.two_afc_result(responses, self.answer_key)
        self.assertEqual(out["n"], 4)
        self.assertEqual(out["n_preferred_decomposition"], 2)
        self.assertAlmostEqual(out["proportion"], 0.5)

    def test_ignores_responses_for_unknown_building_ids(self):
        responses = {"b0": "right", "unknown_building": "left"}
        out = afc.two_afc_result(responses, self.answer_key)
        self.assertEqual(out["n"], 1)
        self.assertEqual(out["missing_ids"], ["unknown_building"])

    def test_flags_significant_vs_chance_when_all_prefer_decomposition(self):
        responses = {"b0": "right", "b1": "left", "b2": "right", "b3": "left"}
        out = afc.two_afc_result(responses, self.answer_key)
        self.assertEqual(out["n_preferred_decomposition"], 4)
        self.assertTrue(out["significant_vs_chance"])

    def test_not_significant_when_split_evenly(self):
        responses = {"b0": "right", "b1": "left", "b2": "left", "b3": "right"}
        out = afc.two_afc_result(responses, self.answer_key)
        self.assertFalse(out["significant_vs_chance"])

    def test_empty_responses_gives_none_proportion(self):
        out = afc.two_afc_result({}, self.answer_key)
        self.assertEqual(out["n"], 0)
        self.assertIsNone(out["proportion"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
