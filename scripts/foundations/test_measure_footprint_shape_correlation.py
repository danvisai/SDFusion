"""Contract tests for #164's footprint-shape/error correlation probe. Synthetic, fast, CPU.

#164 asks whether the served height-map model's per-building error correlates with footprint-shape
statistics the tentative design would add as new conditioning channels (solidity, aspect ratio,
perimeter^2/area, vertex count). Two things are pinned independently of any real corpus:

  * **The shape statistics themselves**, against masks with known geometry (a square, an elongated
    rectangle, a re-entrant L) -- so a broken convex-hull or contour call fails here, not silently
    inside a 714-building correlation run.
  * **The join and the correlation arithmetic** on synthetic records, including the two ways real
    data degenerates: an id the height-field cache does not have, and a mask too small to form a
    hull/contour from. Neither may be silently dropped from the reported counts.

Run: env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/test_measure_footprint_shape_correlation.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.foundations.measure_footprint_shape_correlation import (  # noqa: E402
    ERROR_METRICS, collect_records, correlate, footprint_polygon, footprint_shape_stats,
)

RES = 64


def _square(size=20, res=RES):
    m = np.zeros((res, res), bool)
    r0 = c0 = (res - size) // 2
    m[r0:r0 + size, c0:c0 + size] = True
    return m


def _rectangle(h, w, res=RES):
    m = np.zeros((res, res), bool)
    r0, c0 = (res - h) // 2, (res - w) // 2
    m[r0:r0 + h, c0:c0 + w] = True
    return m


def _l_shape(res=RES):
    """A 40x40 square with its 20x20 top-right quadrant removed -- one re-entrant corner.

    Half the square must go missing for solidity to move clearly: the hull-of-pixel-centers
    convention (train_vecset.py's precedent, reused here) already undercounts a solid block's own
    area relative to its pixel count, so a small corner nick is masked by that same discretisation
    bias. Verified empirically before picking this size (0.2x notch: solidity 1.0 indistinguishable
    from convex; 0.5x: 0.908).
    """
    m = _rectangle(40, 40, res)
    rs, cs = np.nonzero(m)
    r0, c0 = rs.min(), cs.min()
    m[r0:r0 + 20, c0 + 20:c0 + 40] = False
    return m


class TestFootprintShapeStats(unittest.TestCase):
    def test_a_square_has_solidity_near_one_aspect_near_one_and_perimeter_ratio_near_sixteen(self):
        stats = footprint_shape_stats(_square(20))
        self.assertIsNotNone(stats)
        self.assertGreater(stats["solidity"], 0.95)
        self.assertAlmostEqual(stats["aspect_ratio"], 1.0, delta=0.1)
        # perimeter^2 / area = 16 for an exact square (P=4s, A=s^2); discretisation gives ~[14, 18].
        self.assertAlmostEqual(stats["perimeter_sq_over_area"], 16.0, delta=2.0)
        self.assertGreaterEqual(stats["vertex_count"], 4)
        self.assertLessEqual(stats["vertex_count"], 8)

    def test_an_elongated_rectangle_has_high_aspect_ratio_and_solidity_near_one(self):
        stats = footprint_shape_stats(_rectangle(8, 40))
        self.assertIsNotNone(stats)
        self.assertGreater(stats["solidity"], 0.95)
        self.assertAlmostEqual(stats["aspect_ratio"], 5.0, delta=0.5)

    def test_an_l_shaped_footprint_has_solidity_below_a_square_of_the_same_bbox(self):
        square = footprint_shape_stats(_rectangle(40, 40))
        l_shape = footprint_shape_stats(_l_shape())
        self.assertIsNotNone(l_shape)
        self.assertLess(l_shape["solidity"], square["solidity"])
        self.assertLess(l_shape["solidity"], 0.95)
        self.assertEqual(l_shape["vertex_count"], 6)

    def test_an_empty_mask_returns_none(self):
        self.assertIsNone(footprint_shape_stats(np.zeros((RES, RES), bool)))

    def test_a_mask_too_small_to_form_a_hull_returns_none(self):
        m = np.zeros((RES, RES), bool)
        m[0, 0] = m[0, 1] = True   # two pixels: no convex hull, no polygon
        self.assertIsNone(footprint_shape_stats(m))

    def test_footprint_polygon_is_none_for_a_degenerate_mask(self):
        self.assertIsNone(footprint_polygon(np.zeros((RES, RES), bool)))


class TestCorrelate(unittest.TestCase):
    def test_perfectly_correlated_values_score_one(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0 * v + 1.0 for v in x]
        result = correlate(x, y)
        self.assertEqual(result["n"], 5)
        self.assertAlmostEqual(result["pearson_r"], 1.0, places=6)
        self.assertAlmostEqual(result["spearman_r"], 1.0, places=6)

    def test_perfectly_anticorrelated_values_score_negative_one(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [-v for v in x]
        result = correlate(x, y)
        self.assertAlmostEqual(result["pearson_r"], -1.0, places=6)
        self.assertAlmostEqual(result["spearman_r"], -1.0, places=6)

    def test_fewer_than_three_points_is_nan_guarded_not_raised(self):
        result = correlate([1.0, 2.0], [3.0, 4.0])
        self.assertEqual(result["n"], 2)
        self.assertTrue(np.isnan(result["pearson_r"]))
        self.assertTrue(np.isnan(result["spearman_r"]))


class TestCollectRecords(unittest.TestCase):
    def setUp(self):
        fps = np.zeros((3, RES, RES), np.uint8)
        fps[0] = _square(20)
        fps[1] = _rectangle(8, 40)
        fps[2, 0, 0] = 1   # id 30: single-pixel mask, degenerate
        self.cache = {"row": np.asarray([10, 20, 30], np.int32), "fp": fps}
        self.per_building = [
            {"id": 10, "extra": 0.01, "missing": 0.02, "vol_iou": 0.9},
            {"id": 20, "extra": 0.03, "missing": 0.04, "vol_iou": 0.8},
            {"id": 30, "extra": 0.05, "missing": 0.06, "vol_iou": 0.7},   # degenerate mask
            {"id": 99, "extra": 0.07, "missing": 0.08, "vol_iou": 0.6},  # not in cache
        ]

    def test_matched_records_carry_both_shape_stats_and_every_error_metric(self):
        records, skipped = collect_records(self.cache, self.per_building)
        self.assertEqual(skipped, 2)
        self.assertEqual({r["id"] for r in records}, {10, 20})
        for r in records:
            self.assertIn("solidity", r)
            self.assertIn("aspect_ratio", r)
            self.assertIn("perimeter_sq_over_area", r)
            self.assertIn("vertex_count", r)
            for metric in ERROR_METRICS:
                self.assertIn(metric, r)

    def test_a_degenerate_mask_and_a_missing_row_are_both_counted_as_skipped(self):
        _, skipped = collect_records(self.cache, self.per_building)
        self.assertEqual(skipped, 2)


if __name__ == "__main__":
    unittest.main()
