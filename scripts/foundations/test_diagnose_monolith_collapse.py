"""Contract tests for diagnose_monolith_collapse.py's pure seams.

Fast + data-free. The I/O-heavy full diagnosis (real BuildingNet loads, MonolithPairDataset over
train_100) is verified separately by an integration run (see ticket 13's Answer addendum).

Run: env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
     scripts/foundations/test_diagnose_monolith_collapse.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/foundations
import diagnose_monolith_collapse as dmc  # noqa: E402


class ClassifyByOccupancyTest(unittest.TestCase):
    def test_splits_by_threshold(self):
        rows = [
            dict(building="a", gen_occ_frac=0.0),
            dict(building="b", gen_occ_frac=5e-5),
            dict(building="c", gen_occ_frac=1e-4),
            dict(building="d", gen_occ_frac=0.5),
        ]
        near_empty, non_empty = dmc.classify_by_occupancy(rows, threshold=1e-4)
        self.assertEqual(near_empty, ["a", "b"])
        self.assertEqual(non_empty, ["c", "d"])

    def test_default_threshold_matches_ticket_13(self):
        rows = [dict(building="a", gen_occ_frac=0.00009), dict(building="b", gen_occ_frac=0.0002)]
        near_empty, non_empty = dmc.classify_by_occupancy(rows)
        self.assertEqual(near_empty, ["a"])
        self.assertEqual(non_empty, ["b"])


class GroupByHashTest(unittest.TestCase):
    def test_identical_bytes_land_in_one_group(self):
        data = {"a": b"same", "b": b"same", "c": b"different"}
        groups = dmc.group_by_hash(data)
        self.assertEqual(len(groups), 2)
        sizes = sorted(len(v) for v in groups.values())
        self.assertEqual(sizes, [1, 2])

    def test_all_distinct_gives_one_group_per_id(self):
        data = {"a": b"1", "b": b"2", "c": b"3"}
        groups = dmc.group_by_hash(data)
        self.assertEqual(len(groups), 3)

    def test_all_identical_gives_a_single_group(self):
        data = {"a": b"x", "b": b"x", "c": b"x"}
        groups = dmc.group_by_hash(data)
        self.assertEqual(len(groups), 1)
        self.assertEqual(sorted(list(groups.values())[0]), ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
