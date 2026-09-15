"""Contract tests for #92's final 2x2 scorecard artifact."""
from __future__ import annotations

import unittest

from scripts.foundations.summarize_issue92 import summarize_arms, summarize_matched_curves


def _artifact(ids, step, rows):
    return {
        "meta": {"a2": f"step{step}.pth", "n": len(ids)},
        "ids": ids,
        "summary": {f"a2_s{strength}": values for strength, values in rows.items()},
    }


class TestIssue92Summary(unittest.TestCase):
    def test_each_arm_is_read_at_the_strength_with_best_median_iou(self):
        ids = [10, 20, 30]
        arms = {
            "A": _artifact(ids, 220000, {
                0.4: {"vol_iou": 0.87, "vs_input": 0.97, "beats_envelope_rate": 0.04,
                      "collapse_rate": 0.01},
                0.5: {"vol_iou": 0.88, "vs_input": 0.96, "beats_envelope_rate": 0.06,
                      "collapse_rate": 0.02},
            }),
            "B": _artifact(ids, 210000, {
                0.4: {"vol_iou": 0.89, "vs_input": 0.95, "beats_envelope_rate": 0.07,
                      "collapse_rate": 0.03},
                0.5: {"vol_iou": 0.86, "vs_input": 0.90, "beats_envelope_rate": 0.08,
                      "collapse_rate": 0.04},
            }),
            "C": _artifact(ids, 230000, {
                0.5: {"vol_iou": 0.85, "vs_input": 0.94, "beats_envelope_rate": 0.03,
                      "collapse_rate": 0.05},
            }),
            "D": _artifact(ids, 240000, {
                0.5: {"vol_iou": 0.84, "vs_input": 0.93, "beats_envelope_rate": 0.02,
                      "collapse_rate": 0.06},
            }),
        }

        report = summarize_arms(arms)

        self.assertEqual(report["selected"]["A"]["strength"], 0.5)
        self.assertEqual(report["selected"]["B"]["strength"], 0.4)
        self.assertEqual(report["selected"]["A"]["evaluated_strengths"], [0.4, 0.5])
        self.assertEqual(report["candidate_bar_arm"], "B")
        self.assertTrue(report["selected"]["A"]["bar_met"])
        self.assertTrue(report["selected"]["B"]["bar_met"])
        self.assertAlmostEqual(report["surface_marginal"]["encoded"]["vol_iou"], 0.03)
        self.assertAlmostEqual(report["surface_marginal"]["aligned"]["vol_iou"], 0.05)

    def test_the_bar_is_an_and_and_five_percent_is_not_enough(self):
        ids = [1]
        row = {0.5: {"vol_iou": 0.876, "vs_input": 0.979,
                     "beats_envelope_rate": 0.05, "collapse_rate": 0.0}}
        report = summarize_arms({arm: _artifact(ids, 240000, row) for arm in "ABCD"})
        self.assertFalse(report["selected"]["A"]["bar_met"])
        self.assertEqual(report["selected"]["A"]["failed_clauses"], ["beats_envelope_rate > 0.05"])

    def test_all_four_arms_must_use_the_same_building_ids(self):
        row = {0.5: {"vol_iou": 0.8, "vs_input": 0.9,
                     "beats_envelope_rate": 0.0, "collapse_rate": 0.0}}
        arms = {arm: _artifact([1, 2], 240000, row) for arm in "ABCD"}
        arms["D"] = _artifact([1, 3], 240000, row)
        with self.assertRaisesRegex(ValueError, "same fixed ids"):
            summarize_arms(arms)

    def test_factorial_marginals_are_computed_at_matched_steps_and_strength(self):
        def point(step, vol_iou, vs_input):
            return {
                "step": step, "n": 714, "vol_iou": vol_iou, "vs_input": vs_input,
                "fp_iou": 0.9, "missing": 0.1, "extra": 0.2, "collapse_rate": 0.1,
                "beats_envelope_rate": 0.06,
            }

        curves = {
            "A": [point(190000, 0.80, 0.90), point(200000, 0.82, 0.91)],
            "B": [point(190000, 0.85, 0.92), point(200000, 0.86, 0.93)],
            "C": [point(190000, 0.78, 0.89), point(200000, 0.79, 0.90)],
            "D": [point(190000, 0.84, 0.91), point(200000, 0.83, 0.92)],
        }

        report = summarize_matched_curves(curves, strength=0.5)

        self.assertEqual(report["steps"], [190000, 200000])
        self.assertEqual(report["endpoint_step"], 200000)
        self.assertAlmostEqual(
            report["by_step"]["190000"]["surface_marginal"]["encoded"]["vol_iou"],
            0.02,
        )
        self.assertAlmostEqual(
            report["by_step"]["190000"]["surface_marginal"]["aligned"]["vol_iou"],
            0.01,
        )
        self.assertAlmostEqual(
            report["endpoint"]["alignment_effect"]["with_surface"]["vol_iou"],
            0.04,
        )

    def test_matched_curve_report_refuses_different_checkpoint_sets(self):
        row = {"step": 190000, "n": 714, "vol_iou": 0.8, "vs_input": 0.9}
        curves = {arm: [row.copy()] for arm in "ABCD"}
        curves["D"].append({**row, "step": 200000})
        with self.assertRaisesRegex(ValueError, "same checkpoint steps"):
            summarize_matched_curves(curves, strength=0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
