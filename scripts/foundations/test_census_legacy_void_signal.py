"""#119: synthetic controls for the A2-eligible legacy void census.

CPU only, no corpus and no checkpoint. These fix the census's meaning on geometry whose answer is
known by construction, so a regression in `hollow_shell_voxels` or the gap mask shows up here
rather than as a silently shifted corpus statistic.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.foundations.census_legacy_void_signal import (  # noqa: E402
    STRUCTURAL_VOXELS, below_roofline_gap_mask, census_row,
)


def _solid_box() -> np.ndarray:
    occ = np.zeros((16, 16, 16), bool)
    occ[4:12, 0:10, 4:12] = True
    return occ


class TestCensusRow(unittest.TestCase):
    def test_solid_box_has_no_gap(self):
        solid, gap, reach, sealed = census_row(_solid_box())
        self.assertEqual((gap, reach, sealed), (0, 0, 0))
        self.assertEqual(solid, 8 * 10 * 8)

    def test_empty_volume_is_all_zero(self):
        self.assertEqual(census_row(np.zeros((16, 16, 16), bool)), (0, 0, 0, 0))

    def test_exterior_passage_counts_as_reachable_not_sealed(self):
        occ = _solid_box()
        # A slot cut clean through the z extent: open to the volume boundary at both ends.
        occ[:, 2:6, 6:9] = False
        solid, gap, reach, sealed = census_row(occ)
        self.assertGreater(reach, 0)
        self.assertEqual(sealed, 0)
        self.assertEqual(gap, reach)

    def test_sealed_cavity_counts_as_sealed_not_reachable(self):
        occ = _solid_box()
        occ[6:10, 3:7, 6:10] = False  # fully enclosed by solid on every side
        solid, gap, reach, sealed = census_row(occ)
        self.assertGreater(sealed, 0)
        self.assertEqual(reach, 0)

    def test_mixed_volume_splits_reachable_and_sealed(self):
        occ = _solid_box()
        occ[:, 2:4, 6:8] = False        # through-passage, reachable
        occ[6:9, 6:9, 6:9] = False      # buried cavity, sealed
        _, gap, reach, sealed = census_row(occ)
        self.assertGreater(reach, 0)
        self.assertGreater(sealed, 0)
        self.assertEqual(gap, reach + sealed)

    def test_structural_threshold_is_s_star_cubed(self):
        self.assertEqual(STRUCTURAL_VOXELS, 27)


class TestGapMask(unittest.TestCase):
    def test_base_is_inferred_not_assumed_to_be_plane_zero(self):
        """The ingested corpus is centered per building, so plane 0 is not ground."""
        occ = np.zeros((8, 16, 8), bool)
        occ[2:6, 5:12, 2:6] = True      # floats well above H=0 in the stored frame
        occ[3:5, 7:9, 3:5] = False      # a cavity strictly inside the occupied band
        gap = below_roofline_gap_mask(occ)
        self.assertTrue(gap[3:5, 7:9, 3:5].all())
        self.assertFalse(gap[:, :5, :].any(), "planes below the inferred base must not count")

    def test_rejects_non_volume_input(self):
        with self.assertRaisesRegex(ValueError, "3-D occupancy volume"):
            below_roofline_gap_mask(np.zeros((4, 4), bool))

    def test_empty_volume_yields_empty_mask(self):
        self.assertFalse(below_roofline_gap_mask(np.zeros((6, 6, 6), bool)).any())


class TestMaskMatchesTheMergedVerification(unittest.TestCase):
    """The census duplicates the merged verification's mask rather than importing it. Assert the
    copies agree wherever both are present, so a future edit to one cannot silently diverge."""

    def test_identical_to_verify_buildingworld_voids(self):
        try:
            from scripts.foundations.verify_buildingworld_voids import (
                below_roofline_gap_mask as merged_mask,
            )
        except ImportError:
            self.skipTest("verify_buildingworld_voids not present on this branch")
        rng = np.random.default_rng(119)
        for _ in range(12):
            occ = rng.random((12, 12, 12)) < 0.35
            np.testing.assert_array_equal(below_roofline_gap_mask(occ), merged_mask(occ))


if __name__ == "__main__":
    unittest.main()
