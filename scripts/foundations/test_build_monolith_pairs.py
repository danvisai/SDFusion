"""Contract tests for the real full-data monolith-pair builder (ticket 07).

Fast + data-free: exercises the pure seams -- the s*-tied low-pass coarse-input transform, the
derived-vs-stored footprint axis check, and the per-pair structural validation -- with tiny
synthetic grids, no BuildingNet files and no GPU. The full 100%-fraction build against real
BuildingNet data is verified separately by an integration run (see the ticket answer), matching
this project's established convention for that kind of code (tickets 04/05/09).

Run: env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
     scripts/foundations/test_build_monolith_pairs.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/foundations
import build_monolith_pairs as bmp  # noqa: E402


class CoarseResolutionTest(unittest.TestCase):
    def test_locked_operating_point(self):
        # ADR 0004: working_res=96, s*=5 voxels -> coarse grid pitch ~= s*.
        self.assertEqual(bmp.coarse_resolution(96, 5), 19)

    def test_scales_with_s_star(self):
        self.assertEqual(bmp.coarse_resolution(96, 8), 12)

    def test_never_below_one(self):
        self.assertEqual(bmp.coarse_resolution(4, 50), 1)


class LowPassSdfTest(unittest.TestCase):
    def test_output_shape_matches_working_res(self):
        sdf = np.random.default_rng(0).normal(size=(12, 12, 12)).astype(np.float32)
        out = bmp.low_pass_sdf(sdf, working_res=12, s_star_vox=3, device="cpu")
        self.assertEqual(out.shape, (12, 12, 12))

    def test_constant_field_is_preserved(self):
        sdf = np.full((12, 12, 12), -0.3, dtype=np.float32)
        out = bmp.low_pass_sdf(sdf, working_res=12, s_star_vox=3, device="cpu")
        self.assertTrue(np.allclose(out, -0.3, atol=1e-4))

    def test_attenuates_a_sub_scale_spike(self):
        # A single deeply-negative voxel surrounded by zero is far smaller than s* -- the
        # low-pass primary must not let it survive to the coarse "massing" input unchanged.
        sdf = np.zeros((24, 24, 24), dtype=np.float32)
        sdf[12, 12, 12] = -5.0
        out = bmp.low_pass_sdf(sdf, working_res=24, s_star_vox=3, device="cpu")
        self.assertLess(float(np.abs(out).max()), 5.0)

    def test_deterministic(self):
        sdf = np.random.default_rng(1).normal(size=(18, 18, 18)).astype(np.float32)
        a = bmp.low_pass_sdf(sdf, working_res=18, s_star_vox=4, device="cpu")
        b = bmp.low_pass_sdf(sdf, working_res=18, s_star_vox=4, device="cpu")
        self.assertTrue(np.array_equal(a, b))

    def test_preserves_which_side_a_feature_is_on(self):
        # Solid low octant, empty high octant -- the down+up resample chain must not
        # transpose or flip an axis along the way.
        sdf = np.ones((16, 16, 16), dtype=np.float32)
        sdf[:6, :6, :6] = -1.0
        out = bmp.low_pass_sdf(sdf, working_res=16, s_star_vox=4, device="cpu")
        iz, iy, ix = np.unravel_index(np.argmin(out), out.shape)
        self.assertLess(iz, 8)
        self.assertLess(iy, 8)
        self.assertLess(ix, 8)


class FootprintAlignmentIouTest(unittest.TestCase):
    def test_identical_masks_give_iou_one(self):
        occ = np.zeros((4, 3, 4), dtype=bool)
        occ[1, :, 2] = True
        fp = occ.any(axis=1)
        self.assertEqual(bmp.footprint_alignment_iou(occ, fp), 1.0)

    def test_disjoint_masks_give_iou_zero(self):
        occ = np.zeros((4, 3, 4), dtype=bool)
        occ[0, :, 0] = True
        stored = np.zeros((4, 4), dtype=bool)
        stored[3, 3] = True
        self.assertEqual(bmp.footprint_alignment_iou(occ, stored), 0.0)

    def test_accepts_a_leading_singleton_axis_like_the_h5_field(self):
        occ = np.zeros((4, 3, 4), dtype=bool)
        occ[1, :, 2] = True
        fp = occ.any(axis=1)[None, ...]  # (1, D, W), as stored in ori_sample_grid.h5
        self.assertEqual(bmp.footprint_alignment_iou(occ, fp), 1.0)

    def test_empty_union_is_reported_as_perfect(self):
        occ = np.zeros((2, 2, 2), dtype=bool)
        stored = np.zeros((2, 2), dtype=bool)
        self.assertEqual(bmp.footprint_alignment_iou(occ, stored), 1.0)


class ValidatePairTest(unittest.TestCase):
    def test_valid_pair_reports_expected_fields(self):
        target = np.full((8, 8, 8), 0.1, dtype=np.float32)
        target[3:5, 3:5, 3:5] = -0.1
        coarse = np.full((8, 8, 8), 0.2, dtype=np.float32)
        rec = bmp.validate_pair("bldg_1", target, coarse, working_res=8, footprint_iou=0.97)
        self.assertEqual(rec["building"], "bldg_1")
        self.assertAlmostEqual(rec["target_occupancy_frac"], (target <= 0).mean())
        self.assertAlmostEqual(rec["coarse_occupancy_frac"], (coarse <= 0).mean())
        self.assertEqual(rec["footprint_axis_iou"], 0.97)

    def test_wrong_resolution_raises(self):
        target = np.zeros((8, 8, 8), dtype=np.float32)
        coarse = np.zeros((8, 8, 8), dtype=np.float32)
        with self.assertRaises(AssertionError):
            bmp.validate_pair("bldg_2", target, coarse, working_res=16, footprint_iou=1.0)

    def test_non_finite_values_raise(self):
        target = np.zeros((8, 8, 8), dtype=np.float32)
        target[0, 0, 0] = np.nan
        coarse = np.zeros((8, 8, 8), dtype=np.float32)
        with self.assertRaises(AssertionError):
            bmp.validate_pair("bldg_3", target, coarse, working_res=8, footprint_iou=1.0)


class SelectPairIdsReuseTest(unittest.TestCase):
    """The leakage-safe select/exclude seam is ticket 04's `select_building_ids`, reused here
    rather than reimplemented -- this only checks the builder actually calls the real thing."""

    def test_excludes_the_sealed_test_split(self):
        train = ["a", "b", "c"]
        test = ["b"]
        out = bmp.select_building_ids(train, include_ids=None, exclude_ids=test)
        self.assertEqual(out, ["a", "c"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
