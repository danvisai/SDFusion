"""Contract tests for the C2 kill-gate decision rule (ticket 13).

Fast + data-free: exercises the one pure seam -- the kill-gate decision rule itself -- without
touching the GPU model, checkpoints, or rendering. The generation/comparison pipeline is
verified separately by an integration run (see the ticket answer), matching this project's
established convention for GPU-dependent code.

Run: env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
     scripts/foundations/test_decide_c2_kill_gate.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/foundations
import decide_c2_kill_gate as gate  # noqa: E402


class KillGateDecisionTest(unittest.TestCase):
    def test_passes_when_decomposition_wins_detail_and_massing_is_comparable(self):
        d = gate.kill_gate_decision(detail_fid_decomp=100.0, detail_fid_monolith=150.0,
                                    massing_fp_iou_decomp=0.30, massing_fp_iou_monolith=0.28)
        self.assertTrue(d["wins_detail"])
        self.assertTrue(d["comparable_massing"])
        self.assertEqual(d["gate"], "pass")
        self.assertAlmostEqual(d["detail_fid_gap"], 50.0)

    def test_fails_when_decomposition_loses_on_detail_even_with_better_massing(self):
        d = gate.kill_gate_decision(detail_fid_decomp=200.0, detail_fid_monolith=150.0,
                                    massing_fp_iou_decomp=0.40, massing_fp_iou_monolith=0.20)
        self.assertFalse(d["wins_detail"])
        self.assertEqual(d["gate"], "fail")

    def test_fails_when_massing_drops_far_below_tolerance_even_with_detail_win(self):
        d = gate.kill_gate_decision(detail_fid_decomp=100.0, detail_fid_monolith=150.0,
                                    massing_fp_iou_decomp=0.10, massing_fp_iou_monolith=0.30,
                                    comparable_tolerance=0.05)
        self.assertTrue(d["wins_detail"])
        self.assertFalse(d["comparable_massing"])
        self.assertEqual(d["gate"], "fail")

    def test_massing_within_tolerance_below_monolith_still_counts_as_comparable(self):
        # "comparable" is not "at least as good" -- a small, disclosed tolerance below the
        # monolith's own massing fidelity still passes the gate.
        d = gate.kill_gate_decision(detail_fid_decomp=100.0, detail_fid_monolith=150.0,
                                    massing_fp_iou_decomp=0.27, massing_fp_iou_monolith=0.30,
                                    comparable_tolerance=0.05)
        self.assertTrue(d["comparable_massing"])
        self.assertEqual(d["gate"], "pass")

    def test_equal_fid_does_not_count_as_a_win(self):
        d = gate.kill_gate_decision(detail_fid_decomp=150.0, detail_fid_monolith=150.0,
                                    massing_fp_iou_decomp=0.30, massing_fp_iou_monolith=0.30)
        self.assertFalse(d["wins_detail"])
        self.assertEqual(d["gate"], "fail")


class BootstrapMeanCiTest(unittest.TestCase):
    def test_point_estimate_matches_plain_mean(self):
        pt, lo, hi = gate.bootstrap_mean_ci([0.1, 0.2, 0.3, 0.4], n_boot=500, seed=0)
        self.assertAlmostEqual(pt, 0.25)
        self.assertLessEqual(lo, pt)
        self.assertGreaterEqual(hi, pt)

    def test_constant_values_give_a_zero_width_ci(self):
        pt, lo, hi = gate.bootstrap_mean_ci([0.5] * 10, n_boot=200, seed=0)
        self.assertAlmostEqual(pt, 0.5)
        self.assertAlmostEqual(lo, 0.5)
        self.assertAlmostEqual(hi, 0.5)

    def test_same_seed_is_deterministic(self):
        a = gate.bootstrap_mean_ci([0.1, 0.5, 0.9, 0.2], n_boot=300, seed=7)
        b = gate.bootstrap_mean_ci([0.1, 0.5, 0.9, 0.2], n_boot=300, seed=7)
        self.assertEqual(a, b)


class LocalizeDecompositionFailuresTest(unittest.TestCase):
    def test_splits_by_leakage_tier_and_retrieval_activity(self):
        rows = [
            dict(decomposition_tier="clean", decomposition_iou=0.10, decomposition_n_retrieved=0,
                 decomposition_massing_iou=0.10),
            dict(decomposition_tier="clean", decomposition_iou=0.20, decomposition_n_retrieved=2,
                 decomposition_massing_iou=0.20),
            dict(decomposition_tier="train_leak", decomposition_iou=0.40, decomposition_n_retrieved=0,
                 decomposition_massing_iou=0.40),
        ]
        out = gate.localize_decomposition_failures(rows)
        self.assertAlmostEqual(out["mean_iou_by_leakage_tier"]["clean"], 0.15)
        self.assertAlmostEqual(out["mean_iou_by_leakage_tier"]["train_leak"], 0.40)
        self.assertAlmostEqual(out["mean_iou_with_retrieval"], 0.20)
        self.assertAlmostEqual(out["mean_iou_without_retrieval"], 0.25)
        self.assertEqual(out["n_with_retrieval"], 1)
        self.assertEqual(out["n_without_retrieval"], 2)

    def test_none_when_a_bucket_is_empty(self):
        rows = [dict(decomposition_tier="clean", decomposition_iou=0.5, decomposition_n_retrieved=0,
                     decomposition_massing_iou=0.5)]
        out = gate.localize_decomposition_failures(rows)
        self.assertIsNone(out["mean_iou_with_retrieval"])
        self.assertEqual(out["n_with_retrieval"], 0)

    def test_composition_iou_drop_isolates_the_compose_step_from_massing(self):
        # massing_iou is ticket 12's base-massing-only IoU; decomposition_iou is the FINAL
        # composed-shape IoU -- their difference is what the compose step itself cost.
        rows = [
            dict(decomposition_tier="clean", decomposition_iou=0.10, decomposition_n_retrieved=1,
                 decomposition_massing_iou=0.30),
            dict(decomposition_tier="clean", decomposition_iou=0.20, decomposition_n_retrieved=0,
                 decomposition_massing_iou=0.25),
        ]
        out = gate.localize_decomposition_failures(rows)
        self.assertAlmostEqual(out["mean_composition_iou_drop"], 0.125)
        self.assertAlmostEqual(out["mean_composition_iou_drop_with_retrieval"], 0.20)
        self.assertAlmostEqual(out["mean_composition_iou_drop_without_retrieval"], 0.05)


class LocalizeMonolithFailuresTest(unittest.TestCase):
    def test_splits_by_class_and_near_empty_generation(self):
        rows = [
            dict(building_class="RESIDENTIAL", monolith_iou=0.0, monolith_gen_occ_frac=1e-6),
            dict(building_class="RESIDENTIAL", monolith_iou=0.4, monolith_gen_occ_frac=0.01),
            dict(building_class="COMMERCIAL", monolith_iou=0.2, monolith_gen_occ_frac=0.02),
        ]
        out = gate.localize_monolith_failures(rows)
        self.assertAlmostEqual(out["mean_iou_by_class"]["RESIDENTIAL"], 0.2)
        self.assertAlmostEqual(out["mean_iou_by_class"]["COMMERCIAL"], 0.2)
        self.assertEqual(out["n_near_empty_generations"], 1)
        self.assertEqual(out["n_non_empty_generations"], 2)
        self.assertAlmostEqual(out["mean_iou_near_empty"], 0.0)
        self.assertAlmostEqual(out["mean_iou_non_empty"], 0.3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
