"""Contract tests for #160's geometric duplicate gate. Synthetic, fast, CPU, no network, no h5.

#160's risk is real and specific: BuildingWorld Tokyo mesh ids follow PLATEAU's own gml:id
convention, and the corpus's existing JP rows come from the same PLATEAU tiles -- including rows
in the FROZEN held-out set the served arm's headline number is measured against. The ticket is
explicit that id-based matching cannot be trusted (`bag_id` is a truncated fixed-width field), so
the actual gate must be geometric: footprint IoU plus |height_m| match, in a shared real-world
coordinate frame. Four things are pinned here, independent of any live PLATEAU/NRW/BuildingWorld
data:

  * **The footprint/centroid geometry** itself, against points of known shape (a unit square).
  * **Polygon IoU**, against pairs of known overlap (identical, disjoint, a known fraction).
  * **The match decision**, which requires BOTH footprint and height to agree -- high IoU alone or
    matching height alone must not be enough, since either one is cheap to satisfy by coincidence
    across two independent building datasets.
  * **The spatial pre-filter** (bucketing + the cheap city-level bbox-separation check that this
    ticket's real Berlin-vs-NRW run turned out to need): a true duplicate pair must survive it, an
    unrelated pair hundreds of metres apart must never even reach the expensive IoU computation.

Run: env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/test_dedup_buildingworld_geometric.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.foundations.dedup_buildingworld_geometric import (  # noqa: E402
    _run_pair, bbox_overlaps, height_consistency_mismatches, hull_polygon_and_centroid,
    is_duplicate, match_candidates, nearest_neighbor_distances, nrw_tile_files_from_bag_ids,
    polygon_iou, spatial_bucket,
)


def _building(id_, cx, cy, size, height_m):
    """A synthetic square-footprint building record centred at (cx, cy)."""
    half = size / 2.0
    xy = np.array([[cx - half, cy - half], [cx + half, cy - half],
                   [cx + half, cy + half], [cx - half, cy + half]])
    poly, centroid = hull_polygon_and_centroid(xy)
    return dict(id=id_, polygon=poly, height_m=height_m, centroid=centroid)


class TestHullPolygonAndCentroid(unittest.TestCase):
    def test_a_unit_square_has_area_one_and_a_centred_centroid(self):
        xy = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        poly, centroid = hull_polygon_and_centroid(xy)
        self.assertAlmostEqual(poly.area, 1.0, places=6)
        np.testing.assert_allclose(centroid, [0.5, 0.5], atol=1e-6)

    def test_fewer_than_three_points_returns_none(self):
        self.assertIsNone(hull_polygon_and_centroid(np.array([[0, 0], [1, 1]])))

    def test_three_collinear_points_returns_none(self):
        self.assertIsNone(hull_polygon_and_centroid(np.array([[0, 0], [1, 0], [2, 0]])))


class TestPolygonIou(unittest.TestCase):
    def test_identical_polygons_score_one(self):
        poly, _ = hull_polygon_and_centroid(np.array([[0, 0], [4, 0], [4, 4], [0, 4]]))
        self.assertAlmostEqual(polygon_iou(poly, poly), 1.0, places=6)

    def test_disjoint_polygons_score_zero(self):
        a, _ = hull_polygon_and_centroid(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]))
        b, _ = hull_polygon_and_centroid(np.array([[10, 10], [11, 10], [11, 11], [10, 11]]))
        self.assertEqual(polygon_iou(a, b), 0.0)

    def test_a_known_partial_overlap_matches_the_hand_computed_fraction(self):
        # Two 2x2 squares offset by 1 in x: intersection 1x2=2, union 4+4-2=6 -> IoU = 1/3.
        a, _ = hull_polygon_and_centroid(np.array([[0, 0], [2, 0], [2, 2], [0, 2]]))
        b, _ = hull_polygon_and_centroid(np.array([[1, 0], [3, 0], [3, 2], [1, 2]]))
        self.assertAlmostEqual(polygon_iou(a, b), 1.0 / 3.0, places=6)


class TestIsDuplicate(unittest.TestCase):
    def test_matching_footprint_and_height_is_a_duplicate(self):
        self.assertTrue(is_duplicate(height_a=10.0, height_b=10.4, iou=0.8))

    def test_high_iou_alone_is_not_enough_if_height_disagrees(self):
        self.assertFalse(is_duplicate(height_a=10.0, height_b=25.0, iou=0.95))

    def test_matching_height_alone_is_not_enough_if_footprint_disagrees(self):
        self.assertFalse(is_duplicate(height_a=10.0, height_b=10.1, iou=0.05))

    def test_thresholds_are_the_caller_s_to_set(self):
        self.assertFalse(is_duplicate(height_a=10.0, height_b=10.0, iou=0.4, iou_threshold=0.5))
        self.assertTrue(is_duplicate(height_a=10.0, height_b=10.0, iou=0.4, iou_threshold=0.3))


class TestSpatialBucket(unittest.TestCase):
    def test_nearby_points_share_a_bucket(self):
        self.assertEqual(spatial_bucket(np.array([100.2, 200.4]), cell_m=15.0),
                         spatial_bucket(np.array([103.1, 202.9]), cell_m=15.0))

    def test_points_a_cell_apart_land_in_different_buckets(self):
        self.assertNotEqual(spatial_bucket(np.array([0.0, 0.0]), cell_m=15.0),
                            spatial_bucket(np.array([20.0, 0.0]), cell_m=15.0))


class TestBboxOverlaps(unittest.TestCase):
    def test_overlapping_boxes_overlap(self):
        self.assertTrue(bbox_overlaps((0, 0, 10, 10), (5, 5, 15, 15)))

    def test_boxes_hundreds_of_kilometres_apart_do_not_overlap(self):
        # The real #160 Berlin-vs-NRW finding: NRW easting 280-344 km, Berlin 780-814 km (UTM32).
        nrw = (280_000, 5_580_000, 344_000, 5_747_000)
        berlin = (780_744, 5_813_073, 814_549, 5_841_336)
        self.assertFalse(bbox_overlaps(nrw, berlin, margin_m=5_000))

    def test_margin_can_bridge_a_small_gap(self):
        a, b = (0, 0, 10, 10), (11, 0, 20, 10)
        self.assertFalse(bbox_overlaps(a, b, margin_m=0.0))
        self.assertTrue(bbox_overlaps(a, b, margin_m=1.0))


class TestMatchCandidates(unittest.TestCase):
    def test_finds_a_true_duplicate_pair_among_decoys(self):
        existing = [
            _building("plateau_A", cx=1000.0, cy=2000.0, size=8.0, height_m=12.0),
            _building("plateau_far", cx=5000.0, cy=5000.0, size=8.0, height_m=12.0),
        ]
        candidates = [
            _building("bw_A", cx=1000.3, cy=2000.2, size=8.2, height_m=12.3),   # true duplicate
            _building("bw_unrelated", cx=1000.0, cy=2000.0, size=8.0, height_m=40.0),  # same spot, diff height
        ]
        matches = match_candidates(candidates, existing, cell_m=15.0)
        ids = {(m["candidate_id"], m["existing_id"]) for m in matches}
        self.assertEqual(ids, {("bw_A", "plateau_A")})

    def test_returns_empty_when_nothing_is_close(self):
        existing = [_building("e1", 0.0, 0.0, 8.0, 10.0)]
        candidates = [_building("c1", 500.0, 500.0, 8.0, 10.0)]
        self.assertEqual(match_candidates(candidates, existing, cell_m=15.0), [])

    def test_a_pair_hundreds_of_metres_apart_is_never_even_compared(self):
        # A poisoned "existing" record whose iou/height would match if compared -- proves the
        # bucketing pre-filter, not the match decision, is what keeps it out.
        existing = [_building("far_but_identical", cx=900.0, cy=900.0, size=8.0, height_m=12.0)]
        candidates = [_building("near", cx=1000.0, cy=1000.0, size=8.0, height_m=12.0)]
        self.assertEqual(match_candidates(candidates, existing, cell_m=15.0), [])


class TestNearestNeighborDistances(unittest.TestCase):
    def test_distance_to_the_true_nearest_existing_centroid(self):
        existing = [_building("e_near", 0.0, 0.0, 4.0, 10.0),
                   _building("e_far", 100.0, 0.0, 4.0, 10.0)]
        candidates = [_building("c1", 3.0, 4.0, 4.0, 10.0)]   # 5m from e_near, 97m from e_far
        d = nearest_neighbor_distances(candidates, existing)
        self.assertEqual(len(d), 1)
        self.assertAlmostEqual(d[0], 5.0, places=6)

    def test_empty_existing_population_is_reported_not_raised(self):
        candidates = [_building("c1", 0.0, 0.0, 4.0, 10.0)]
        d = nearest_neighbor_distances(candidates, [])
        self.assertTrue(np.all(np.isinf(d)))

    def test_empty_candidate_population_returns_empty_not_raised(self):
        existing = [_building("e1", 0.0, 0.0, 4.0, 10.0)]
        d = nearest_neighbor_distances([], existing)
        self.assertEqual(len(d), 0)


class TestNrwTileFilesFromBagIds(unittest.TestCase):
    def test_extracts_unique_sorted_tile_filenames(self):
        bag_ids = ["LoD2_32_300_5675_1_NW.gml#DENW1", "LoD2_32_280_5657_1_NW.gml#DENW2",
                  "LoD2_32_300_5675_1_NW.gml#DENW3"]
        self.assertEqual(nrw_tile_files_from_bag_ids(bag_ids),
                        ["LoD2_32_280_5657_1_NW.gml", "LoD2_32_300_5675_1_NW.gml"])

    def test_an_unrecognised_pattern_raises_rather_than_silently_skipping(self):
        with self.assertRaises(ValueError):
            nrw_tile_files_from_bag_ids(["not_an_nrw_tile_id"])


class TestHeightConsistencyMismatches(unittest.TestCase):
    def test_agreeing_heights_produce_no_mismatch(self):
        existing_rows = [{"bag_id": "a", "row": 0, "height_m": 10.0}]
        world = {"a": {"height_m": 10.02}}
        self.assertEqual(height_consistency_mismatches(existing_rows, world), [])

    def test_a_large_height_disagreement_is_reported(self):
        existing_rows = [{"bag_id": "a", "row": 0, "height_m": 10.0}]
        world = {"a": {"height_m": 25.0}}
        mismatches = height_consistency_mismatches(existing_rows, world)
        self.assertEqual(len(mismatches), 1)
        self.assertEqual(mismatches[0]["bag_id"], "a")

    def test_a_row_never_relocated_is_not_a_mismatch(self):
        existing_rows = [{"bag_id": "missing", "row": 0, "height_m": 10.0}]
        self.assertEqual(height_consistency_mismatches(existing_rows, {}), [])


def _existing_row_and_world_entry(bag_id, row, cx, cy, size, height_m):
    r = {"bag_id": bag_id, "row": row, "height_m": height_m}
    return r, _building(bag_id, cx, cy, size, height_m)


class TestRunPairCompletenessFloor(unittest.TestCase):
    def test_a_small_disclosed_gap_is_tolerated_and_reported(self):
        rows, world = [], {}
        for i in range(100):
            r, b = _existing_row_and_world_entry(f"id{i}", i, i * 100.0, 0.0, 8.0, 10.0)
            rows.append(r)
            if i != 0:                              # one row (1%) never re-located
                world[b["id"]] = b
        result = _run_pair("test_pair", rows, world, [], cell_m=15.0, iou_threshold=0.3,
                          height_tol_m=2.0, min_relocate_frac=0.98)
        self.assertEqual(result["n_existing_relocated"], 99)
        self.assertEqual(result["unrelocated_bag_ids"], ["id0"])

    def test_a_gap_below_the_floor_fails_loudly(self):
        rows, world = [], {}
        for i in range(100):
            r, b = _existing_row_and_world_entry(f"id{i}", i, i * 100.0, 0.0, 8.0, 10.0)
            rows.append(r)
            if i < 50:                              # only half re-located
                world[b["id"]] = b
        with self.assertRaises(SystemExit):
            _run_pair("test_pair", rows, world, [], cell_m=15.0, iou_threshold=0.3,
                     height_tol_m=2.0, min_relocate_frac=0.98)

    def test_a_match_against_a_frozen_held_out_row_is_annotated(self):
        # #160's own stated highest-risk case: a BuildingWorld candidate duplicating a row in the
        # FROZEN held-out set. `existing_held_out` on the match is the only signal that would ever
        # surface it -- every other test here passes candidates=[], so this path (the `matches`
        # loop body in `_run_pair`) had never actually executed before this test.
        held_r, held_b = _existing_row_and_world_entry("held", 0, 0.0, 0.0, 8.0, 12.0)
        train_r, train_b = _existing_row_and_world_entry("train", 1, 500.0, 500.0, 8.0, 12.0)
        rows = [held_r, train_r]
        world = {"held": held_b, "train": train_b}
        dup_of_held = _building("bw_dup", cx=0.2, cy=0.1, size=8.1, height_m=12.2)
        result = _run_pair("test_pair", rows, world, [dup_of_held], cell_m=15.0,
                          iou_threshold=0.3, height_tol_m=2.0, held_out={0: True, 1: False})
        self.assertEqual(result["n_matches"], 1)
        match = result["matches"][0]
        self.assertEqual(match["existing_id"], "held")
        self.assertEqual(match["existing_row"], 0)
        self.assertTrue(match["existing_held_out"])

    def test_a_match_against_a_row_missing_from_the_held_out_map_is_none(self):
        row, b = _existing_row_and_world_entry("row0", 0, 0.0, 0.0, 8.0, 12.0)
        dup = _building("bw_dup", cx=0.2, cy=0.1, size=8.1, height_m=12.2)
        result = _run_pair("test_pair", [row], {"row0": b}, [dup], cell_m=15.0,
                          iou_threshold=0.3, height_tol_m=2.0)
        self.assertEqual(result["n_matches"], 1)
        self.assertIsNone(result["matches"][0]["existing_held_out"])

    def test_a_height_drifted_row_is_excluded_from_matching_not_a_hard_failure(self):
        drifted_r, drifted_b = _existing_row_and_world_entry("drifted", 0, 0.0, 0.0, 8.0, 10.0)
        drifted_r["height_m"] = 25.0                 # disagrees with the world-derived 10.0
        clean_r, clean_b = _existing_row_and_world_entry("clean", 1, 500.0, 500.0, 8.0, 10.0)
        rows = [drifted_r, clean_r]
        world = {"drifted": drifted_b, "clean": clean_b}
        result = _run_pair("test_pair", rows, world, [], cell_m=15.0, iou_threshold=0.3,
                          height_tol_m=2.0)
        self.assertEqual(result["n_existing_relocated"], 2)
        self.assertEqual(result["n_existing_verified"], 1)
        self.assertEqual([d["bag_id"] for d in result["height_drifted"]], ["drifted"])


if __name__ == "__main__":
    unittest.main()
