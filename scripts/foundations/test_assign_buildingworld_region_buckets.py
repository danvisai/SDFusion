"""Contract tests for #171's region-bucket backfill. Synthetic, fast, no corpus, no GPU.

Run: env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/test_assign_buildingworld_region_buckets.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.foundations.assign_buildingworld_region_buckets import (  # noqa: E402
    UNASSIGNED_SENTINEL, compute_updated_region,
)
from scripts.foundations.recover_program_labels_buildingworld import (  # noqa: E402
    BUILDINGWORLD_SOURCE_ID,
)
from scripts.foundations.source_provenance import BUILDINGWORLD_CITY_BUCKET, region_id_of  # noqa: E402


def _fixture(regions):
    """Three legacy rows (region 0/1/2) plus one BuildingWorld row per `regions` entry."""
    n_legacy = 3
    rows = np.arange(n_legacy + len(regions), dtype=np.int32)
    source_id = np.array([0, 1, 2] + [BUILDINGWORLD_SOURCE_ID] * len(regions), np.int32)
    source_key = np.array(
        [b"bag3d:NL", b"nrw:LoD2_32_280_5657", b"plateau:tokyo23ku"]
        + [f"bw:{city}".encode() for city in regions], "S64")
    region = np.array([0, 1, 2] + [UNASSIGNED_SENTINEL] * len(regions), np.int32)
    ledger = dict(row=rows, region=region,
                 held_out=np.zeros(len(rows), np.uint8),
                 height_m=np.ones(len(rows), np.float32))
    return ledger, source_id, source_key


class TestComputeUpdatedRegion(unittest.TestCase):
    def test_legacy_rows_pass_through_unchanged(self):
        ledger, source_id, source_key = _fixture(["Berlin"])
        updated = compute_updated_region(ledger, source_id, source_key)
        np.testing.assert_array_equal(updated[:3], [0, 1, 2])

    def test_buildingworld_rows_resolve_to_their_real_bucket(self):
        ledger, source_id, source_key = _fixture(["Berlin", "Tokyo", "CapeTown"])
        updated = compute_updated_region(ledger, source_id, source_key)
        np.testing.assert_array_equal(
            updated[3:], [BUILDINGWORLD_CITY_BUCKET["Berlin"], BUILDINGWORLD_CITY_BUCKET["Tokyo"],
                         BUILDINGWORLD_CITY_BUCKET["CapeTown"]])

    def test_no_buildingworld_row_is_left_at_the_sentinel(self):
        ledger, source_id, source_key = _fixture(list(BUILDINGWORLD_CITY_BUCKET))
        updated = compute_updated_region(ledger, source_id, source_key)
        self.assertFalse(np.any(updated == UNASSIGNED_SENTINEL))

    def test_is_idempotent_a_second_call_on_the_same_output_raises_rather_than_reassigning(self):
        ledger, source_id, source_key = _fixture(["Berlin"])
        once = compute_updated_region(ledger, source_id, source_key)
        ledger["region"] = once
        with self.assertRaisesRegex(ValueError, "already carry"):
            compute_updated_region(ledger, source_id, source_key)

    def test_a_legacy_row_wrongly_carrying_the_sentinel_is_rejected_as_corrupt(self):
        ledger, source_id, source_key = _fixture(["Berlin"])
        ledger["region"][0] = UNASSIGNED_SENTINEL
        with self.assertRaisesRegex(ValueError, "corrupt"):
            compute_updated_region(ledger, source_id, source_key)

    def test_matches_region_id_of_for_every_registered_city(self):
        cities = list(BUILDINGWORLD_CITY_BUCKET)
        ledger, source_id, source_key = _fixture(cities)
        updated = compute_updated_region(ledger, source_id, source_key)
        for city, value in zip(cities, updated[3:]):
            self.assertEqual(int(value), region_id_of(f"bw:{city}"))

    def test_an_unregistered_place_still_raises_through_region_id_of(self):
        ledger, source_id, source_key = _fixture(["NotACity"])
        with self.assertRaises(ValueError):
            compute_updated_region(ledger, source_id, source_key)

    def test_no_buildingworld_rows_is_a_harmless_no_op(self):
        ledger, source_id, source_key = _fixture([])
        updated = compute_updated_region(ledger, source_id, source_key)
        np.testing.assert_array_equal(updated, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
