"""CPU-only contract tests for verify_buildingworld_voids.py's measurement seam."""
from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.foundations.verify_buildingworld_voids import analyze_volume, below_roofline_gap_mask


class TestBuildingWorldVoidVerification(unittest.TestCase):
    def test_solid_box_has_no_below_roofline_gap(self):
        occ = np.zeros((16, 16, 16), dtype=bool)
        occ[2:14, 3:13, 2:14] = True
        result, gap, _ = analyze_volume(occ)
        self.assertFalse(result.has_gap)
        self.assertEqual(int(gap.sum()), 0)

    def test_open_passage_is_exterior_reachable_not_sealed(self):
        occ = np.zeros((16, 16, 16), dtype=bool)
        occ[2:14, 2:14, 2:14] = True
        occ[6:10, 5:9, :] = False
        result, _, _ = analyze_volume(occ)
        self.assertTrue(result.has_gap)
        self.assertGreater(result.reachable_gap_voxels, 0)
        self.assertEqual(result.sealed_gap_voxels, 0)
        self.assertEqual(result.classification, "reachable_only")

    def test_sealed_cavity_is_not_counted_as_reachable_signal(self):
        occ = np.zeros((16, 16, 16), dtype=bool)
        occ[2:14, 2:14, 2:14] = True
        occ[6:10, 5:9, 6:10] = False
        result, _, _ = analyze_volume(occ)
        self.assertTrue(result.has_gap)
        self.assertEqual(result.reachable_gap_voxels, 0)
        self.assertGreater(result.sealed_gap_voxels, 0)
        self.assertEqual(result.classification, "sealed_only")

    def test_global_base_is_inferred_in_centered_corpus_frame(self):
        occ = np.zeros((10, 10, 10), dtype=bool)
        occ[2:8, 4:9, 2:8] = True
        occ[4:6, 4:6, 4:6] = False
        gap = below_roofline_gap_mask(occ)
        self.assertFalse(gap[:, :4, :].any())
        self.assertTrue(gap[4:6, 4:6, 4:6].all())


if __name__ == "__main__":
    unittest.main()
