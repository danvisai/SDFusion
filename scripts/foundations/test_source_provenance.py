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
    PIPELINE_REGION_ID, REGION_MAPPING_VERSION, SOURCE_KEY_MAX_BYTES,
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

    def test_place_never_affects_the_region_id_only_pipeline_does(self):
        a = region_id_of(make_source_key("plateau", "tokyo23ku"))
        b = region_id_of(make_source_key("plateau", "osaka"))
        self.assertEqual(a, b)

    def test_an_unregistered_pipeline_raises_and_points_at_171(self):
        with self.assertRaisesRegex(ValueError, "#171"):
            region_id_of(make_source_key("bw", "Berlin"))

    def test_every_registered_pipeline_is_reachable_through_region_id_of(self):
        for pipeline in PIPELINE_REGION_ID:
            region_id_of(make_source_key(pipeline, "anyplace"))


class TestRegionMappingHash(unittest.TestCase):
    """The mapping's version/hash is what #163 stamps into height-map checkpoints."""

    def test_is_a_stable_sha256_hex_digest(self):
        h = region_mapping_sha256()
        self.assertEqual(len(h), 64)
        int(h, 16)  # raises ValueError if not hex
        self.assertEqual(h, region_mapping_sha256())

    def test_is_order_independent_over_the_pipeline_table(self):
        reordered = {"version": REGION_MAPPING_VERSION,
                     "pipeline_region_id": dict(reversed(list(PIPELINE_REGION_ID.items())))}
        expected = hashlib.sha256(
            json.dumps(reordered, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.assertEqual(region_mapping_sha256(), expected)

    def test_reflects_the_version_string_not_just_the_table(self):
        other_version = hashlib.sha256(
            json.dumps({"version": "not-" + REGION_MAPPING_VERSION,
                        "pipeline_region_id": PIPELINE_REGION_ID},
                       sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.assertNotEqual(region_mapping_sha256(), other_version)


if __name__ == "__main__":
    unittest.main()
