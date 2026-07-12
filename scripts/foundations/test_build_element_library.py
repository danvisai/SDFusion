"""Contract tests for the leakage-safe element-library builder (ticket 04) and the per-type
solidity/scale quantification it grew for ticket 08.

Fast + data-free: exercises the pure `select_building_ids` / `load_id_list` seam that guarantees a
held-out test split can never contribute an element, plus the pure `crop_solidity` /
`distribution_stats` / `scale_rel` seams ticket 08 added. The full voxelizing build is verified
separately by a small integration run (see the ticket answer).

Run: env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
     scripts/foundations/test_build_element_library.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/foundations
import build_element_library as bel  # noqa: E402

UNIV = ["COMMERCIALcastle_mesh0001", "RELIGIOUStemple_mesh0002",
        "RESIDENTIALhouse_mesh0003", "RESIDENTIALhouse_mesh0004", "PUBLICmuseum_mesh0005"]


class SelectBuildingIdsTest(unittest.TestCase):
    def test_no_filters_returns_all_sorted(self):
        self.assertEqual(bel.select_building_ids(UNIV), sorted(UNIV))

    def test_include_restricts_to_the_set(self):
        inc = {"RESIDENTIALhouse_mesh0003", "PUBLICmuseum_mesh0005"}
        self.assertEqual(set(bel.select_building_ids(UNIV, include_ids=inc)), inc)

    def test_exclude_removes_the_set(self):
        out = bel.select_building_ids(UNIV, exclude_ids={"COMMERCIALcastle_mesh0001"})
        self.assertNotIn("COMMERCIALcastle_mesh0001", out)
        self.assertEqual(len(out), len(UNIV) - 1)

    def test_exclude_wins_over_include(self):
        """A held-out test id that also appears in the include (train) set must be dropped."""
        both = "RESIDENTIALhouse_mesh0003"
        out = bel.select_building_ids(
            UNIV, include_ids={both, "PUBLICmuseum_mesh0005"}, exclude_ids={both})
        self.assertEqual(out, ["PUBLICmuseum_mesh0005"])

    def test_unknown_include_ids_are_ignored(self):
        self.assertEqual(bel.select_building_ids(UNIV, include_ids={"NOT_A_REAL_ID"}), [])

    def test_sorted_and_order_independent(self):
        exc = {"RESIDENTIALhouse_mesh0004"}
        a = bel.select_building_ids(UNIV, exclude_ids=exc)
        b = bel.select_building_ids(list(reversed(UNIV)), exclude_ids=exc)
        self.assertEqual(a, b)
        self.assertEqual(a, sorted(a))

    def test_load_id_list_none_and_json(self):
        self.assertIsNone(bel.load_id_list(None))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ids.json"
            p.write_text(json.dumps(["a", "b", "a"]))
            self.assertEqual(bel.load_id_list(str(p)), {"a", "b"})


class CropSolidityTest(unittest.TestCase):
    def test_all_inside_gives_solidity_one(self):
        crop = np.full((4, 4, 4), -1.0, dtype=np.float16)
        self.assertEqual(bel.crop_solidity(crop), 1.0)

    def test_all_outside_gives_solidity_zero(self):
        crop = np.full((4, 4, 4), 1.0, dtype=np.float16)
        self.assertEqual(bel.crop_solidity(crop), 0.0)

    def test_half_inside_half_outside(self):
        crop = np.ones((2, 2, 2), dtype=np.float32)
        crop[0] = -1.0
        self.assertAlmostEqual(bel.crop_solidity(crop), 0.5)

    def test_boundary_value_zero_counts_as_inside(self):
        # sdf<=0 -- matches element_fit.py's `_solidity` fallback definition exactly.
        crop = np.array([[[0.0, 1.0]]], dtype=np.float32)
        self.assertAlmostEqual(bel.crop_solidity(crop), 0.5)


class DistributionStatsTest(unittest.TestCase):
    def test_basic_stats(self):
        stats = bel.distribution_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(stats["n"], 5)
        self.assertEqual(stats["mean"], 3.0)
        self.assertEqual(stats["median"], 3.0)
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 5.0)

    def test_empty_input_reports_zero_count_without_crashing(self):
        stats = bel.distribution_stats([])
        self.assertEqual(stats["n"], 0)
        self.assertIsNone(stats["mean"])
        self.assertIsNone(stats["median"])

    def test_single_value(self):
        stats = bel.distribution_stats([2.5])
        self.assertEqual(stats["n"], 1)
        self.assertEqual(stats["mean"], 2.5)
        self.assertEqual(stats["min"], 2.5)
        self.assertEqual(stats["max"], 2.5)


class ScaleRelTest(unittest.TestCase):
    def test_returns_max_of_ext_rel(self):
        self.assertAlmostEqual(bel.scale_rel({"ext_rel": [0.1, 0.5, 0.2]}), 0.5)

    def test_handles_all_equal(self):
        self.assertAlmostEqual(bel.scale_rel({"ext_rel": [0.3, 0.3, 0.3]}), 0.3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
