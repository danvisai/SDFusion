"""Contract tests for token alignment (#90).

Pure CPU, no model and no cache: the properties under test are about the permutation, not about any
particular pair of buildings. The one property that cost the most to learn -- that a method must
return a genuine bijection, because #89's headline number came from a many-to-one map that cannot
reorder anything -- is pinned here rather than left to the probe.

Run: env -u LD_PRELOAD ./venv/bin/python models/test_token_alignment.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from models.token_alignment import METHODS, VOXEL, align, report  # noqa: E402


def _cloud(n=256, seed=0):
    return np.random.default_rng(seed).uniform(-1, 1, (n, 3))


class TestEveryMethodReturnsAPermutation(unittest.TestCase):
    """A cache written from a many-to-one map would duplicate tokens and drop others."""

    def test_permutation(self):
        pa, pb = _cloud(), _cloud(seed=1)
        for m in [x for x in METHODS if x != "nn"]:
            with self.subTest(method=m):
                perm = align(pa, pb, m)
                np.testing.assert_array_equal(np.sort(perm), np.arange(len(pb)))

    def test_nn_is_declared_not_a_permutation_here(self):
        # kept as an explicit expectation: `nn` is the bound, and its non-bijectivity is the reason
        pa, pb = _cloud(), _cloud(seed=1)
        self.assertLess(len(np.unique(align(pa, pb, "nn"))), len(pb))


class TestMatchingRecoversAKnownPermutation(unittest.TestCase):
    """The case with a right answer: same points, shuffled. A matcher must undo the shuffle."""

    def setUp(self):
        self.pa = _cloud(n=512)
        self.shuffle = np.random.default_rng(7).permutation(len(self.pa))
        self.pb = self.pa[self.shuffle]

    def test_greedy_recovers_it_exactly(self):
        perm = align(self.pa, self.pb, "greedy")
        np.testing.assert_array_equal(self.pb[perm], self.pa)

    def test_hungarian_recovers_it_exactly(self):
        perm = align(self.pa, self.pb, "hungarian")
        np.testing.assert_array_equal(self.pb[perm], self.pa)

    def test_as_encoded_does_not(self):
        perm = align(self.pa, self.pb, "as_encoded")
        self.assertFalse(np.allclose(self.pb[perm], self.pa))


class TestGreedyIsStableAndDeterministic(unittest.TestCase):
    def test_same_input_same_permutation(self):
        pa, pb = _cloud(), _cloud(seed=1)
        np.testing.assert_array_equal(align(pa, pb, "greedy"), align(pa, pb, "greedy"))

    def test_k_is_clamped_to_the_token_count(self):
        pa, pb = _cloud(n=64), _cloud(n=64, seed=1)
        np.testing.assert_array_equal(np.sort(align(pa, pb, "greedy", k=10_000)), np.arange(64))

    def test_k_of_one_still_returns_a_permutation(self):
        pa, pb = _cloud(n=64), _cloud(n=64, seed=1)
        np.testing.assert_array_equal(np.sort(align(pa, pb, "greedy", k=1)), np.arange(64))


class TestKRestrictionChangesTheAlgorithm(unittest.TestCase):
    """⚠️ #90 nearly chose wrong twice by comparing restricted and unrestricted greedy as if they
    were one method. The dependence is real, so it is asserted rather than trusted."""

    def test_larger_k_is_at_least_as_good_on_total_distance(self):
        pa, pb = _cloud(n=256), _cloud(n=256, seed=1)
        d = [np.linalg.norm(pa - pb[align(pa, pb, "greedy", k)], axis=-1).mean() for k in (1, 8, 256)]
        self.assertLessEqual(d[2], d[0])
        self.assertLessEqual(d[2], d[1] + 1e-9)


class TestReport(unittest.TestCase):
    def test_identical_latents_score_one(self):
        pa = _cloud(n=128)
        z = np.random.default_rng(3).normal(size=(128, 16)).astype(np.float32)
        rep = report(z, z, pa, pa, align(pa, pa, "greedy"))
        self.assertAlmostEqual(rep["cosine"], 1.0, places=4)
        self.assertEqual(rep["matched_frac"], 1.0)

    def test_matched_is_judged_at_the_voxel_pitch(self):
        pa = _cloud(n=128)
        pb = pa + np.array([2 * VOXEL, 0.0, 0.0])          # every pair just beyond a voxel
        z = np.random.default_rng(3).normal(size=(128, 16)).astype(np.float32)
        self.assertEqual(report(z, z, pa, pb, np.arange(128))["matched_frac"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
