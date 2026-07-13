"""Contract tests for the C1b sculpt strength sweep (ticket 10).

Fast + data-free: exercises the one pure seam -- aggregating per-(case, strength) rows into
per-strength faithfulness stats for the faithfulness-vs-realism plot -- without touching the
GPU model. The snap/render/FID pipeline itself is verified separately by an integration run
(see the ticket answer), matching this project's established convention for GPU-dependent code
(tickets 05/09).

Run: env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
     scripts/eval/test_sculpt_strength_sweep.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/eval
import sculpt_strength_sweep as sss  # noqa: E402


class SummarizeByStrengthTest(unittest.TestCase):
    def test_aggregates_mean_min_max_across_cases_at_one_strength(self):
        rows = [
            dict(strength=0.5, case="tower", iou_to_edit=0.8),
            dict(strength=0.5, case="dome", iou_to_edit=0.6),
            dict(strength=0.5, case="carve", iou_to_edit=1.0),
        ]
        out = sss.summarize_by_strength(rows)
        self.assertEqual(len(out), 1)
        row = out[0]
        self.assertEqual(row["strength"], 0.5)
        self.assertEqual(row["n_cases"], 3)
        self.assertAlmostEqual(row["mean_iou_to_edit"], 0.8)
        self.assertAlmostEqual(row["min_iou_to_edit"], 0.6)
        self.assertAlmostEqual(row["max_iou_to_edit"], 1.0)

    def test_returns_one_row_per_distinct_strength_sorted_ascending(self):
        rows = [
            dict(strength=0.9, case="tower", iou_to_edit=0.2),
            dict(strength=0.1, case="tower", iou_to_edit=0.95),
            dict(strength=0.5, case="tower", iou_to_edit=0.6),
        ]
        out = sss.summarize_by_strength(rows)
        self.assertEqual([r["strength"] for r in out], [0.1, 0.5, 0.9])

    def test_single_row_strength_has_equal_mean_min_max(self):
        rows = [dict(strength=0.3, case="carve", iou_to_edit=0.42)]
        out = sss.summarize_by_strength(rows)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["mean_iou_to_edit"], 0.42)
        self.assertEqual(out[0]["min_iou_to_edit"], 0.42)
        self.assertEqual(out[0]["max_iou_to_edit"], 0.42)

    def test_empty_input_yields_empty_output(self):
        self.assertEqual(sss.summarize_by_strength([]), [])


class EditCasesTest(unittest.TestCase):
    def test_case_names_are_unique(self):
        names = [name for name, _ in sss.EDIT_CASES]
        self.assertEqual(len(names), len(set(names)))

    def test_every_case_is_a_well_formed_editop_dict(self):
        for name, edit in sss.EDIT_CASES:
            self.assertIn("kind", edit)
            self.assertIn("mode", edit)
            self.assertIn(edit["mode"], ("add", "subtract"))

    def test_strengths_are_sorted_and_within_unit_interval(self):
        self.assertEqual(sss.STRENGTHS, sorted(sss.STRENGTHS))
        self.assertTrue(all(0.0 < s < 1.0 for s in sss.STRENGTHS))


if __name__ == "__main__":
    unittest.main(verbosity=2)
