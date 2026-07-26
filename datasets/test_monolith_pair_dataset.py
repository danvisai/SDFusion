"""Contract tests for the monolith pair dataset's pure seams (ticket 11).

Fast + data-free: the train/val split (which ids are ever gradient-trained on) and the
axis-preserving augmentation, without touching real BuildingNet files. `MonolithPairDataset`
itself (real H5 loading) is verified separately by the training smoke run (see the ticket
answer), matching this project's established convention for that kind of code.

Run: env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
     datasets/test_monolith_pair_dataset.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import monolith_pair_dataset as mpd  # noqa: E402


class TrainValIdsTest(unittest.TestCase):
    IDS = [f"bldg_{i:04d}" for i in range(200)]

    def test_sizes_match_the_requested_fraction(self):
        train, val = mpd.train_val_ids(self.IDS, val_frac=0.1, seed=0)
        self.assertEqual(len(val), 20)
        self.assertEqual(len(train), 180)

    def test_disjoint_and_covers_the_full_input(self):
        train, val = mpd.train_val_ids(self.IDS, val_frac=0.2, seed=0)
        self.assertEqual(set(train) & set(val), set())
        self.assertEqual(set(train) | set(val), set(self.IDS))

    def test_deterministic_given_the_same_seed(self):
        a = mpd.train_val_ids(self.IDS, val_frac=0.15, seed=3)
        b = mpd.train_val_ids(self.IDS, val_frac=0.15, seed=3)
        self.assertEqual(a, b)

    def test_different_seeds_give_different_splits(self):
        a = mpd.train_val_ids(self.IDS, val_frac=0.15, seed=3)
        b = mpd.train_val_ids(self.IDS, val_frac=0.15, seed=4)
        self.assertNotEqual(a[1], b[1])

    def test_order_independent(self):
        shuffled = list(reversed(self.IDS))
        a = mpd.train_val_ids(self.IDS, val_frac=0.1, seed=0)
        b = mpd.train_val_ids(shuffled, val_frac=0.1, seed=0)
        self.assertEqual(set(a[1]), set(b[1]))


class RandomAxisAugTest(unittest.TestCase):
    def test_identity_params_return_an_unchanged_array(self):
        arr = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
        out = mpd.apply_axis_aug(arr, k_rot=0, flip_x=False, flip_z=False)
        self.assertTrue(np.array_equal(out, arr))

    def test_rotation_preserves_up_axis_extent(self):
        # (D, H, W): rotating around the up axis (H) swaps D and W, H must stay untouched.
        arr = np.zeros((6, 4, 3), dtype=np.float32)
        out = mpd.apply_axis_aug(arr, k_rot=1, flip_x=False, flip_z=False)
        self.assertEqual(out.shape, (3, 4, 6))

    def test_flip_preserves_shape(self):
        arr = np.random.default_rng(0).random((4, 5, 6)).astype(np.float32)
        out = mpd.apply_axis_aug(arr, k_rot=0, flip_x=True, flip_z=True)
        self.assertEqual(out.shape, arr.shape)
        self.assertFalse(np.array_equal(out, arr))

    def test_same_params_applied_identically_to_two_arrays_stay_aligned(self):
        # The coarse/target pair must receive the SAME random augmentation, or they'd
        # decorrelate spatially -- this checks the params (not the RNG) fully determine output.
        a = np.random.default_rng(1).random((4, 4, 4)).astype(np.float32)
        b = np.random.default_rng(2).random((4, 4, 4)).astype(np.float32)
        # a marker voxel at the same index in both arrays should land at the same output index.
        a[1, 2, 3] = 99.0
        b[1, 2, 3] = 99.0
        out_a = mpd.apply_axis_aug(a, k_rot=2, flip_x=True, flip_z=False)
        out_b = mpd.apply_axis_aug(b, k_rot=2, flip_x=True, flip_z=False)
        self.assertEqual(tuple(np.argwhere(out_a == 99.0)[0]), tuple(np.argwhere(out_b == 99.0)[0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
