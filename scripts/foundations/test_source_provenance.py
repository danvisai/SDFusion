"""Contract tests for #167's provenance-authority field. Synthetic, fast, no corpus, no GPU.

Run: env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/test_source_provenance.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.foundations.source_provenance import (  # noqa: E402
    BUILDINGWORLD_BUCKET_NAMES, BUILDINGWORLD_CITY_BUCKET, PIPELINE_REGION_ID,
    REGION_MAPPING_VERSION, SOURCE_KEY_MAX_BYTES,
    make_source_key, parse_source_key, region_id_of, region_mapping_sha256,
)


class TestSourceKeyFormat(unittest.TestCase):
    """`source_key` ('pipeline:place') is the one field every scheme downstream derives from."""

    def test_round_trips_through_parse(self):
        key = make_source_key("plateau", "tokyo23ku")
        self.assertEqual(key, "plateau:tokyo23ku")
        self.assertEqual(parse_source_key(key), ("plateau", "tokyo23ku"))

    def test_the_issues_own_examples_are_valid(self):
        for pipeline, place in (("bw", "Berlin"), ("plateau", "tokyo23ku"),
                                ("nrw", "LoD2_32_280_5657")):
            with self.subTest(pipeline=pipeline, place=place):
                self.assertEqual(parse_source_key(make_source_key(pipeline, place)),
                                 (pipeline, place))

    def test_malformed_keys_are_rejected(self):
        for bad in ("", "noColon", "UPPER:place", "bag3d:", ":place",
                    "bag3d:place with space", "bag3d:place:extra"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_source_key(bad)

    def test_a_trailing_newline_is_rejected_not_silently_accepted(self):
        # Code review: `$` in the old regex matches before a trailing newline, so
        # "plateau:tokyo23ku\n" used to parse as if it were "plateau:tokyo23ku" -- two source_keys
        # any newline-touched ingest (#174) would otherwise treat as the same place.
        with self.assertRaises(ValueError):
            parse_source_key("plateau:tokyo23ku\n")

    def test_make_source_key_rejects_the_same_malformed_pieces(self):
        with self.assertRaises(ValueError):
            make_source_key("UPPER", "place")
        with self.assertRaises(ValueError):
            make_source_key("bag3d", "")


class TestSourceKeyStorageWidth(unittest.TestCase):
    """#167's own motivating example (`class_label` (S16) truncating 'BW_GreaterGeelong', 17 bytes)
    is the storage-width decision this pins: source_key's real width is SOURCE_KEY_MAX_BYTES,
    decided once here rather than left for #174 to invent."""

    def test_the_tickets_own_motivating_example_fits(self):
        key = make_source_key("bw", "GreaterGeelong")
        self.assertLessEqual(len(key.encode("utf-8")), SOURCE_KEY_MAX_BYTES)

    def test_a_key_over_the_storage_width_raises_rather_than_truncating(self):
        too_long = "bag3d:" + "x" * SOURCE_KEY_MAX_BYTES
        with self.assertRaises(ValueError):
            make_source_key("bag3d", "x" * SOURCE_KEY_MAX_BYTES)
        with self.assertRaises(ValueError):
            parse_source_key(too_long)

    def test_a_key_exactly_at_the_storage_width_is_accepted(self):
        place = "x" * (SOURCE_KEY_MAX_BYTES - len("bag3d:"))
        key = make_source_key("bag3d", place)
        self.assertEqual(len(key.encode("utf-8")), SOURCE_KEY_MAX_BYTES)


class TestRegionOfSourceKey(unittest.TestCase):
    """`region_id_of` is the versioned, pure function every region/stratification scheme calls."""

    def test_todays_three_pipelines_match_the_frozen_source_id_convention(self):
        # #163's N_REGIONS comment: "source corpora: 0 NL / 1 DE / 2 JP".
        self.assertEqual(region_id_of(make_source_key("bag3d", "NL")), 0)
        self.assertEqual(region_id_of(make_source_key("nrw", "LoD2_32_280_5657")), 1)
        self.assertEqual(region_id_of(make_source_key("plateau", "tokyo23ku")), 2)

    def test_place_never_affects_the_region_id_for_a_legacy_pipeline(self):
        a = region_id_of(make_source_key("plateau", "tokyo23ku"))
        b = region_id_of(make_source_key("plateau", "osaka"))
        self.assertEqual(a, b)

    def test_an_unregistered_legacy_pipeline_raises_and_points_at_171(self):
        with self.assertRaisesRegex(ValueError, "#171"):
            region_id_of(make_source_key("someothersource", "anyplace"))

    def test_every_registered_pipeline_is_reachable_through_region_id_of(self):
        for pipeline in PIPELINE_REGION_ID:
            region_id_of(make_source_key(pipeline, "anyplace"))


class TestBuildingWorldStyleBucket(unittest.TestCase):
    """#171's decision: BuildingWorld's region channel is broad style buckets, keyed by place,
    not per-city and not merged into the legacy NL/DE/JP ids."""

    def test_place_determines_the_bucket_for_the_bw_pipeline(self):
        self.assertEqual(region_id_of(make_source_key("bw", "Berlin")),
                         BUILDINGWORLD_CITY_BUCKET["Berlin"])
        self.assertNotEqual(region_id_of(make_source_key("bw", "Berlin")),
                            region_id_of(make_source_key("bw", "Tokyo")))

    def test_every_ingestable_city_has_a_bucket(self):
        # #174's ALL_CITIES minus the dropped Toronto -- kept as a literal list, not an import,
        # so this test also catches an accidental import-order/circular-import mistake.
        cities = ("Adelaide", "Berlin", "Boston", "Calgary", "Cambridge", "CapeTown", "Edmonton",
                 "GreaterGeelong", "Melbourne", "Mississauga", "Montreal", "NewYork", "Perth",
                 "Philadelphia", "SanFrancisco", "Tokyo", "Wellington", "Yarra")
        for city in cities:
            with self.subTest(city=city):
                region_id_of(make_source_key("bw", city))  # must not raise

    def test_buckets_never_collide_with_the_legacy_region_ids(self):
        self.assertFalse(set(BUILDINGWORLD_CITY_BUCKET.values()) & set(PIPELINE_REGION_ID.values()))

    def test_an_unregistered_place_raises_and_points_at_171(self):
        with self.assertRaisesRegex(ValueError, "#171"):
            region_id_of(make_source_key("bw", "NotACity"))

    def test_every_bucket_id_used_has_a_human_readable_name(self):
        self.assertEqual(set(BUILDINGWORLD_CITY_BUCKET.values()), set(BUILDINGWORLD_BUCKET_NAMES))

    def test_no_bucket_is_a_single_city_carrying_a_diversity_label_alone(self):
        # #171 point 3: a thin, low-population bucket can silently absorb a data-quality quirk and
        # present it as "the style." Berlin, Tokyo, and CapeTown are the three single-city buckets
        # on record, each large on its own (hundreds of thousands / tens of thousands of rows,
        # never fewer than Cape Town's 243,932) -- a deliberate exception, not an oversight. Pin
        # the exception list so a future edit that adds a THIN single-city bucket fails this test
        # instead of silently repeating #171's warned-against failure mode.
        from collections import Counter
        counts = Counter(BUILDINGWORLD_CITY_BUCKET.values())
        single_city_buckets = {bucket for bucket, n in counts.items() if n == 1}
        single_city_names = {city for city, bucket in BUILDINGWORLD_CITY_BUCKET.items()
                             if bucket in single_city_buckets}
        self.assertEqual(single_city_names, {"Berlin", "Tokyo", "CapeTown"})


class TestRegionMappingHash(unittest.TestCase):
    """The mapping's version/hash is what #163 stamps into height-map checkpoints."""

    def test_is_a_stable_sha256_hex_digest(self):
        h = region_mapping_sha256()
        self.assertEqual(len(h), 64)
        int(h, 16)  # raises ValueError if not hex
        self.assertEqual(h, region_mapping_sha256())

    def test_is_order_independent_over_either_table(self):
        reordered = {"version": REGION_MAPPING_VERSION,
                     "pipeline_region_id": dict(reversed(list(PIPELINE_REGION_ID.items()))),
                     "buildingworld_city_bucket": dict(reversed(list(
                         BUILDINGWORLD_CITY_BUCKET.items())))}
        expected = hashlib.sha256(
            json.dumps(reordered, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.assertEqual(region_mapping_sha256(), expected)

    def test_reflects_the_version_string_not_just_the_tables(self):
        other_version = hashlib.sha256(
            json.dumps({"version": "not-" + REGION_MAPPING_VERSION,
                        "pipeline_region_id": PIPELINE_REGION_ID,
                        "buildingworld_city_bucket": BUILDINGWORLD_CITY_BUCKET},
                       sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.assertNotEqual(region_mapping_sha256(), other_version)

    def test_a_buildingworld_bucket_reassignment_changes_the_hash(self):
        mutated = dict(BUILDINGWORLD_CITY_BUCKET, Berlin=BUILDINGWORLD_CITY_BUCKET["Tokyo"])
        other = hashlib.sha256(
            json.dumps({"version": REGION_MAPPING_VERSION,
                        "pipeline_region_id": PIPELINE_REGION_ID,
                        "buildingworld_city_bucket": mutated},
                       sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.assertNotEqual(region_mapping_sha256(), other)


if __name__ == "__main__":
    unittest.main()
