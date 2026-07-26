"""Contract tests for the FID metric math + deterministic cameras (ticket 05).

Pure (numpy/scipy), no GPU or network. The Inception extractor + real neutral renders are verified
separately by a real-vs-real sanity integration (see the ticket answer).

Run: env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python scripts/eval/test_fid.py
"""
from __future__ import annotations

import sys
import unittest
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/eval
import fid  # noqa: E402
import render_facades as rf  # noqa: E402


class FidMathTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_fid_zero_for_identical_sets(self):
        f = self.rng.normal(size=(200, 16))
        self.assertLess(abs(fid.fid_from_features(f, f)), 1e-2)

    def test_fid_symmetric(self):
        a = self.rng.normal(size=(200, 16))
        b = self.rng.normal(1.0, size=(200, 16))
        self.assertAlmostEqual(fid.fid_from_features(a, b), fid.fid_from_features(b, a), places=4)

    def test_fid_nonnegative(self):
        a = self.rng.normal(size=(150, 8))
        b = self.rng.normal(2.0, size=(150, 8))
        self.assertGreaterEqual(fid.fid_from_features(a, b), -1e-6)

    def test_frechet_closed_form_diagonal(self):
        d = 5
        mu1, mu2 = np.zeros(d), np.ones(d)
        s1, s2 = np.full(d, 2.0), np.full(d, 0.5)
        expect = float(((mu1 - mu2) ** 2).sum() + (s1 + s2 - 2 * np.sqrt(s1 * s2)).sum())
        got = fid.frechet_distance(mu1, np.diag(s1), mu2, np.diag(s2))
        self.assertAlmostEqual(got, expect, places=4)

    def test_fid_grows_with_mean_shift(self):
        a = self.rng.normal(size=(300, 12))
        near = self.rng.normal(0.5, size=(300, 12))
        far = self.rng.normal(3.0, size=(300, 12))
        self.assertLess(fid.fid_from_features(a, near), fid.fid_from_features(a, far))

    def test_bootstrap_ci_brackets_point_and_orders(self):
        a = self.rng.normal(size=(120, 8))
        b = self.rng.normal(1.0, size=(120, 8))
        point, lo, hi = fid.bootstrap_fid_ci(a, b, n_boot=40, seed=1)
        self.assertLessEqual(lo, hi)
        self.assertLessEqual(lo, point + 1e-6)
        self.assertGreaterEqual(hi, point - 1e-6)

    def test_bootstrap_deterministic(self):
        a = self.rng.normal(size=(80, 6))
        b = self.rng.normal(1.0, size=(80, 6))
        self.assertEqual(fid.bootstrap_fid_ci(a, b, n_boot=30, seed=7),
                         fid.bootstrap_fid_ci(a, b, n_boot=30, seed=7))

    def test_provenance_recorded(self):
        self.assertEqual(fid.EXTRACTOR_PROVENANCE["feature_dim"], 2048)
        self.assertIn("inception", fid.EXTRACTOR_PROVENANCE["name"].lower())

    def test_group_bootstrap_matches_row_bootstrap_when_groups_are_singletons(self):
        a = self.rng.normal(size=(60, 8))
        b = self.rng.normal(1.0, size=(60, 8))
        row = fid.bootstrap_fid_ci(a, b, n_boot=25, seed=3)
        grouped = fid.bootstrap_fid_ci(a, b, n_boot=25, seed=3,
                                       groups_a=np.arange(60), groups_b=np.arange(60))
        self.assertEqual(row, grouped)

    def test_undersampled_flags_n_less_than_feature_dim(self):
        self.assertTrue(fid.undersampled(np.zeros((10, 20)), np.zeros((30, 20))))
        self.assertTrue(fid.undersampled(np.zeros((30, 20)), np.zeros((10, 20))))
        self.assertFalse(fid.undersampled(np.zeros((30, 20)), np.zeros((30, 20))))

    def test_fid_from_features_warns_when_undersampled(self):
        a = self.rng.normal(size=(10, 20))
        b = self.rng.normal(size=(30, 20))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fid.fid_from_features(a, b)
            self.assertTrue(any("biased" in str(x.message) for x in w))

    def test_fid_from_features_silent_when_adequately_sampled(self):
        a = self.rng.normal(size=(30, 20))
        b = self.rng.normal(size=(30, 20))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fid.fid_from_features(a, b)
            self.assertFalse(any("biased" in str(x.message) for x in w))

    def test_bootstrap_ci_warns_once_not_per_iteration(self):
        a = self.rng.normal(size=(10, 20))
        b = self.rng.normal(size=(30, 20))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fid.bootstrap_fid_ci(a, b, n_boot=15, seed=0)
            hits = [x for x in w if "biased" in str(x.message)]
            self.assertEqual(len(hits), 1)

    def test_group_bootstrap_resamples_whole_groups_together(self):
        """3 buildings x 4 correlated views each; group-level resampling must always include all
        4 views of any building it picks (never a partial group)."""
        groups = np.repeat([0, 1, 2], 4)
        rng = np.random.default_rng(0)
        idx = fid._resample_indices(rng, 12, groups)
        picked = set(idx.tolist())
        for g in np.unique(groups[idx]):
            members = set(np.where(groups == g)[0].tolist())
            self.assertTrue(members <= picked, f"group {g} only partially resampled")


class CameraTest(unittest.TestCase):
    def test_orbit_cameras_deterministic_and_count(self):
        self.assertEqual(rf.orbit_cameras(n_views=6), rf.orbit_cameras(n_views=6))
        self.assertEqual(len(rf.orbit_cameras(n_views=8)), 8)

    def test_orbit_cameras_span_distinct_azimuths(self):
        cams = rf.orbit_cameras(n_views=4, radius=2.0)
        xz = {(round(c["cam_pos"][0], 3), round(c["cam_pos"][2], 3)) for c in cams}
        self.assertEqual(len(xz), 4)

    def test_orbit_radius_respected(self):
        cams = rf.orbit_cameras(n_views=5, radius=3.0, elev_deg=0.0)
        for c in cams:
            r = vec_norm(c["cam_pos"])
            self.assertAlmostEqual(r, 3.0, places=5)


class ResampleSdfGridTest(unittest.TestCase):
    """CPU-only: the resolution-parity fix (ADR 0004) that resamples real BuildingNet's native 64³
    field up to the locked WORKING_RES before rendering, so real and generated arms are never
    compared at unequal sampling density."""

    def test_identity_when_resolution_matches(self):
        g = np.random.default_rng(0).normal(size=(20, 20, 20)).astype(np.float32)
        out = rf.resample_sdf_grid(g, 20, device="cpu")
        np.testing.assert_array_equal(out, g)

    def test_output_shape_matches_requested_resolution(self):
        g = np.zeros((32, 32, 32), np.float32)
        out = rf.resample_sdf_grid(g, 48, device="cpu")
        self.assertEqual(out.shape, (48, 48, 48))

    def test_constant_field_is_preserved(self):
        g = np.full((16, 16, 16), -0.37, np.float32)
        out = rf.resample_sdf_grid(g, 24, device="cpu")
        np.testing.assert_allclose(out, -0.37, atol=1e-5)

    def test_corner_values_preserved_under_align_corners(self):
        """align_corners=True must match sphere_trace's own grid_sample convention: voxel (0,0,0)
        stays cube corner -1, voxel (res-1,res-1,res-1) stays +1 regardless of resolution change."""
        g = np.zeros((10, 10, 10), np.float32)
        g[0, 0, 0] = 1.0
        g[-1, -1, -1] = -1.0
        out = rf.resample_sdf_grid(g, 20, device="cpu")
        self.assertAlmostEqual(float(out[0, 0, 0]), 1.0, places=4)
        self.assertAlmostEqual(float(out[-1, -1, -1]), -1.0, places=4)

    def test_working_res_constant_is_96(self):
        """ADR 0004 locks the shared working resolution at 96^3."""
        self.assertEqual(rf.WORKING_RES, 96)


def vec_norm(p):
    return float(np.linalg.norm(np.asarray(p)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
