"""Contract tests for the leakage-safe element-library builder (ticket 04).

Fast + data-free: exercises the pure `select_building_ids` / `load_id_list` seam that guarantees a
held-out test split can never contribute an element. The full voxelizing build is verified separately
by a small integration run (see the ticket answer).

Run: env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
     scripts/foundations/test_build_element_library.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
