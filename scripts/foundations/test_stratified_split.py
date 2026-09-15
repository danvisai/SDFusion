"""Contract tests for #153's region/tile-stratified split (`stratified_split.py`).

Two tiers, matching this repo's existing convention (see `test_train_monolith.py`):
fast + data-free synthetic-fixture tests for the algorithm's structural properties,
plus one integration test against the real corpus that pins the ACHIEVED proportions
(acceptance criterion 1) as a regression -- skipped, not failed, if `real.h5` isn't
present on this machine.

Run: env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
     scripts/foundations/test_stratified_split.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stratified_split import (  # noqa: E402
    assert_no_tile_crosses_boundary,
    make_split,
    tile_key,
)

REAL_H5 = Path(__file__).resolve().parents[2] / "data/real_massing_v1/real.h5"


class TileKeyTest(unittest.TestCase):
    def test_nl_key_is_the_gemeente_code(self):
        self.assertEqual(tile_key(0, "NL.IMBAG.Pand.0599100000058822"), "NL:0599")

    def test_de_key_is_the_gml_tile_filename(self):
        self.assertEqual(
            tile_key(1, "LoD2_32_280_5657_1_NW.gml#DENW43AL00001j66"),
            "DE:LoD2_32_280_5657_1_NW.gml",
        )

    def test_jp_key_is_the_zip_and_gml_tile_pair(self):
        self.assertEqual(
            tile_key(2, "533937_2.zip:53393770_bldg_6697_op2.gml#BLD_64cd89a1"),
            "JP:533937_2.zip:53393770_bldg_6697_op2.gml",
        )

    def test_malformed_nl_id_raises_rather_than_silently_misgrouping(self):
        with self.assertRaises(ValueError):
            tile_key(0, "not-a-bag-id")

    def test_malformed_de_jp_id_raises_rather_than_silently_misgrouping(self):
        with self.assertRaises(ValueError):
            tile_key(1, "no-hash-separator")

    def test_buildingworld_source_id_is_rejected_not_collapsed_into_one_giant_tile(self):
        """#177: BuildingWorld's 'City#member' bag_id would otherwise match the generic '#'
        branch and silently merge every one of a city's rows onto a single tile key."""
        with self.assertRaises(ValueError):
            tile_key(-1, "Berlin#mesh/x.obj")


def _synthetic_corpus(seed=0):
    """3 regions shaped like the real corpus's actual coarseness asymmetry: one region
    with a handful of large tiles (like NL's 5 city bboxes), two with many small tiles
    (like DE's 41 / JP's 9 CityGML tiles) -- enough to exercise the shared-floor logic."""
    rng = np.random.default_rng(seed)
    source_id, bag_id = [], []
    # Region 0: 5 big tiles, ~2000-3200 buildings each (NL-shaped).
    for t in range(5):
        n = int(rng.integers(2000, 3200))
        for _ in range(n):
            source_id.append(0)
            bag_id.append(f"NL.IMBAG.Pand.{1000 + t}00000000001")
    # Region 1: 40 small tiles, ~50-800 buildings each (DE-shaped).
    for t in range(40):
        n = int(rng.integers(50, 800))
        for _ in range(n):
            source_id.append(1)
            bag_id.append(f"tile_de_{t}.gml#B{t}")
    # Region 2: 9 medium tiles, ~200-1500 buildings each (JP-shaped).
    for t in range(9):
        n = int(rng.integers(200, 1500))
        for _ in range(n):
            source_id.append(2)
            bag_id.append(f"z{t}.zip:tile_jp_{t}.gml#B{t}")
    return np.array(source_id), np.array(bag_id)


class MakeSplitStructuralTest(unittest.TestCase):
    """Fast, data-free: properties that must hold for ANY corpus shaped like this one."""

    def setUp(self):
        self.source_id, self.bag_id = _synthetic_corpus()
        self.split, self.report = make_split(self.source_id, self.bag_id, seed=0)

    def test_every_row_gets_exactly_one_label(self):
        self.assertEqual(len(self.split), len(self.source_id))
        self.assertTrue(set(self.split.tolist()) <= {"train", "val", "test"})

    def test_no_tile_crosses_the_train_held_out_boundary(self):
        assert_no_tile_crosses_boundary(self.source_id, self.bag_id, self.split)

    def test_val_and_test_are_disjoint_everywhere(self):
        self.assertEqual(
            set(np.nonzero(self.split == "val")[0]) & set(np.nonzero(self.split == "test")[0]),
            set(),
        )

    def test_every_region_is_represented_in_val_and_in_test(self):
        for sid in (0, 1, 2):
            region_split = self.split[self.source_id == sid]
            self.assertIn("val", region_split, f"region {sid} has zero val representation")
            self.assertIn("test", region_split, f"region {sid} has zero test representation")

    def test_deterministic_for_a_fixed_seed(self):
        split2, _ = make_split(self.source_id, self.bag_id, seed=0)
        self.assertTrue((self.split == split2).all())

    def test_regions_are_within_reach_of_the_shared_floor(self):
        # Not exact (whole-tile blocking forbids exact matches in general) but no region's
        # val/test share of the pooled held-out set should be wildly disproportionate --
        # the bug #5/#127 found in the OLD row-random split (100% one region).
        val_by_region = {
            sid: int((self.split[self.source_id == sid] == "val").sum()) for sid in (0, 1, 2)
        }
        total_val = sum(val_by_region.values())
        for sid, n in val_by_region.items():
            self.assertGreater(n / total_val, 0.10, f"region {sid} under 10% of val: {val_by_region}")


class RealCorpusRegressionTest(unittest.TestCase):
    """Pins the ACHIEVED proportions against the actual corpus (acceptance criterion 1).

    Skipped, not failed, when real.h5 isn't present -- matches this repo's existing
    "the real-data path is verified separately" pattern (see test_train_monolith.py).
    If this ever fails, either the corpus changed (expected once BuildingWorld/#174
    lands -- rerun `stratified_split.py` and update these pins deliberately) or the
    split algorithm regressed (not expected -- investigate).
    """

    @unittest.skipUnless(REAL_H5.exists(), f"{REAL_H5} not present on this machine")
    def test_achieved_proportions_match_the_recorded_baseline(self):
        import h5py

        from utils.frozen_corpus import FROZEN_SPLIT_N_TOTAL

        # #177 appended BuildingWorld rows after the frozen NL/DE/JP prefix; this regression pins
        # #153's own recorded baseline, computed against that historical prefix, not whatever
        # real.h5 has grown to since (see five_arm_scorecard.materialize_split's own docstring).
        with h5py.File(REAL_H5, "r") as f:
            source_id = f["source_id"][:FROZEN_SPLIT_N_TOTAL]
            bag_id = f["bag_id"][:FROZEN_SPLIT_N_TOTAL]
        split, report = make_split(source_id, bag_id, seed=0)
        assert_no_tile_crosses_boundary(source_id, bag_id, split)

        self.assertEqual(len(split), 35776, "corpus size drifted from #153's baseline snapshot")
        pooled = {
            "train": int((split == "train").sum()),
            "val": int((split == "val").sum()),
            "test": int((split == "test").sum()),
        }
        # Recorded in docs/wayfinding/solid-first-subtractive-modeling/153-stratified-split.md.
        self.assertEqual(pooled, {"train": 26064, "val": 2592, "test": 7120})
        self.assertEqual(
            {k: v["val_n"] for k, v in report.items() if k != "_shared_targets"},
            {"NL": 858, "DE": 858, "JP": 876},
        )
        self.assertEqual(
            {k: v["test_n"] for k, v in report.items() if k != "_shared_targets"},
            {"NL": 2387, "DE": 2387, "JP": 2346},
        )


if __name__ == "__main__":
    unittest.main()
