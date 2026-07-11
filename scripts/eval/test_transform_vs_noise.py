"""Contract tests for the C1a from-noise-vs-blockout comparison (ticket 09).

Fast + data-free: exercises the two pure seams -- the footprint-extrude blockout construction
and the Stage3a-training-leakage classification -- without touching the GPU model. The model
inference itself is verified separately by a small integration run (see the ticket answer),
matching this project's established convention for GPU-dependent code (tickets 04/05).

Run: env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
     scripts/eval/test_transform_vs_noise.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/eval
import transform_vs_noise as tvn  # noqa: E402


class FootprintExtrudeBlockoutTest(unittest.TestCase):
    def test_extrudes_union_footprint_to_the_max_occupied_height(self):
        # (D=3, H=4, W=3): one voxel occupied at y=0, another at y=2, at different (z,x).
        occ = np.zeros((3, 4, 3), dtype=bool)
        occ[0, 0, 0] = True   # low corner
        occ[2, 2, 2] = True   # taller corner, different footprint cell
        out = tvn.footprint_extrude_blockout(occ)
        # Footprint is the UNION of both occupied cells; height reaches the taller one (y=2).
        self.assertTrue(out[0, 0, 0])
        self.assertTrue(out[0, 2, 0])       # extruded up at the first footprint cell too
        self.assertTrue(out[2, 2, 2])
        self.assertFalse(out[0, 3, 0])      # above the max occupied height: empty
        self.assertFalse(out[1, 0, 0])      # outside the footprint entirely: empty

    def test_solid_below_top_within_footprint(self):
        occ = np.zeros((2, 5, 2), dtype=bool)
        occ[0, 3, 0] = True
        out = tvn.footprint_extrude_blockout(occ)
        # Every y-level from 0..3 is filled at the occupied footprint cell (solid extrude).
        self.assertTrue(out[0, :4, 0].all())
        self.assertFalse(out[0, 4, 0].any())

    def test_empty_occupancy_yields_empty_blockout(self):
        occ = np.zeros((2, 2, 2), dtype=bool)
        out = tvn.footprint_extrude_blockout(occ)
        self.assertFalse(out.any())

    def test_output_shape_matches_input(self):
        occ = np.random.default_rng(0).random((4, 6, 5)) < 0.1
        out = tvn.footprint_extrude_blockout(occ)
        self.assertEqual(out.shape, occ.shape)
        self.assertEqual(out.dtype, bool)


class ClassifyLeakageTest(unittest.TestCase):
    def test_partitions_by_which_stage3a_split_saw_each_id(self):
        ids = ["a", "b", "c", "d"]
        tiers = tvn.classify_leakage(ids, bn_train_ids=["a"], bn_val_ids=["b"], bn_test_ids=["c"])
        self.assertEqual(tiers["train_leak"], ["a"])
        self.assertEqual(tiers["val_leak"], ["b"])
        self.assertEqual(tiers["clean"], ["c"])
        self.assertEqual(tiers["unknown"], ["d"])

    def test_train_wins_if_an_id_is_in_multiple_lists_defensively(self):
        # Shouldn't happen in real BuildingNet splits, but train-membership should be checked
        # first (the most severe leakage), not silently overwritten by a later branch.
        tiers = tvn.classify_leakage(["x"], bn_train_ids=["x"], bn_val_ids=["x"], bn_test_ids=["x"])
        self.assertEqual(tiers["train_leak"], ["x"])
        self.assertEqual(tiers["val_leak"], [])
        self.assertEqual(tiers["clean"], [])

    def test_empty_input_yields_empty_tiers(self):
        tiers = tvn.classify_leakage([], bn_train_ids=["a"], bn_val_ids=["b"], bn_test_ids=["c"])
        self.assertEqual(tiers, dict(clean=[], val_leak=[], train_leak=[], unknown=[]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
