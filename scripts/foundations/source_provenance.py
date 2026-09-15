"""#167 -- the single provenance-authority field for region and stratification schemes.

Region conditioning is currently derived from WHICH SURFACE FILE a row came from
(`dora_frozen_gate.SOURCES`, `stratified_split.SOURCE_NAMES`: bag3d/NL=0, nrw/DE=1, plateau/JP=2)
-- a source/pipeline id that happens to be 1:1 with country today. BuildingWorld breaks that 1:1
immediately: BuildingWorld Tokyo (same country as PLATEAU, different capture pipeline) and
BuildingWorld Berlin (same country as NRW, different capture pipeline) both need identities the
existing scheme has no slot for, and `class_label` (S16) would truncate a per-city label like
'BW_GreaterGeelong' (17 bytes).

`source_key` ('pipeline:place', e.g. 'plateau:tokyo23ku', 'nrw:LoD2_32_280_5657') is that one
authority field going forward. Every region/stratification/dedup scheme becomes a versioned, pure
function of it: `region_id_of` here is the first such function, and today only reproduces the
three pipelines the frozen corpus already has (bag3d/nrw/plateau) -- deliberately. Standing
preference #4 on map #156 rules out letting a fine-grained id absorb a data-quality defect
invisibly, so which BuildingWorld pipelines get which region id (or a new one) is #171's decision,
not this module's; `region_id_of` raises rather than guesses for any pipeline not yet registered.

`region_mapping_sha256` is the durable version/hash #163 reserved a stamp for in every
height-map-generator checkpoint: a checkpoint trained against one version of this table can never
silently be read as if it means another.
"""
from __future__ import annotations

import hashlib
import json
import re

SOURCE_KEY_RE = re.compile(r"^[a-z0-9_]+:[A-Za-z0-9_.-]+\Z")

# #167's own motivating problem was `class_label` (S16) truncating a per-city label like
# 'BW_GreaterGeelong' (17 bytes) with no error. This is the storage width decision that closes it:
# `bag_id` already uses a fixed-width S64 HDF5 field for a similar per-row identity string
# (concat_real_massing.py), so source_key reuses that same width rather than inventing a new one --
# #174 (the first writer of source_key) has a width to encode against instead of picking its own.
SOURCE_KEY_DTYPE = "S64"
SOURCE_KEY_MAX_BYTES = 64

REGION_MAPPING_VERSION = "v2"

# pipeline -> region id. Unchanged from today's `source_id` convention (#163's N_REGIONS comment:
# "source corpora: 0 NL / 1 DE / 2 JP", matching `stratified_split.SOURCE_NAMES` and
# `dora_frozen_gate.SOURCES`) -- #167 freezes it as version "v1" so #171/#174 have one place to
# derive region ids from `source_key` rather than inventing their own. This ticket does not
# retrofit `stratified_split.py`/`dora_frozen_gate.py` to read from this table -- they keep their
# own copies for now; only new, `source_key`-based consumers are guaranteed to agree with this one.
PIPELINE_REGION_ID: dict[str, int] = {
    "bag3d": 0,
    "nrw": 1,
    "plateau": 2,
}

# #171 (2026-09-13, project-owner-approved via /grilling): BuildingWorld's region channel is
# "broad style buckets," explicitly NOT per-city and NOT per-source_key -- a narrow, low-population
# flag can silently absorb a data-quality quirk specific to that population and present it as "the
# style" (#171 point 3). "region" is a style knob a real generation-time caller picks by hand
# ("Berlin-style roof"), not a geography lookup -- the retired "cross-cultural conditioning" framing
# does not apply; use "style bucket" / "source-style id" in new code and docs.
#
# Grouped by country/continent and sized against each bucket's actual row count (measured against
# the production `real.h5` at ingestion) so no bucket is a single thin city wearing a "diversity"
# label: the smallest bucket here (Oceania, 13,092 rows) is still comparable to a legacy pipeline's
# own ~11-12k rows, and the largest single city (Berlin, 456,569) keeps its own bucket rather than
# diluting a nominal "Europe" bucket it would swamp anyway.
#
# New ids (3-8), never merged into the existing NL/DE/JP ids (0-2): BuildingWorld is a different
# capture pipeline throughout, and #160's exhaustive geometric dedup gate already found BuildingWorld
# Tokyo/Berlin geographically distinct from the existing PLATEAU/NRW rows (nearest match 6,258 m /
# 454,075 m away) -- #171 dropped the same-country/different-pipeline arm specifically because that
# comparison can no longer cleanly separate "pipeline artifact" from "genuine architectural
# difference," so folding BuildingWorld's rows into the SAME learned region id as PLATEAU/NRW would
# quietly reintroduce exactly the confound #171 ruled untestable. Keys are the `source_key` "place"
# component -- BuildingWorld's own `CITY_SLUG` convention (`ingest_buildingworld.py`: the city name
# with spaces stripped), not the display name; not imported from there to avoid a circular import
# (`ingest_buildingworld.py` imports `make_source_key` from this module).
BUILDINGWORLD_CITY_BUCKET: dict[str, int] = {
    "Berlin": 3,                                                          # Germany (456,569 rows)
    "Tokyo": 4,                                                           # Japan (33,315 rows)
    "CapeTown": 5,                                                        # South Africa (243,932)
    "Edmonton": 6, "Calgary": 6, "Mississauga": 6, "Montreal": 6,         # Canada (640,020)
    "Boston": 7, "NewYork": 7, "SanFrancisco": 7,                        # USA (139,850)
    "Philadelphia": 7, "Cambridge": 7,
    "Melbourne": 8, "Yarra": 8, "Adelaide": 8,                            # Oceania: AU + NZ (13,092)
    "GreaterGeelong": 8, "Wellington": 8, "Perth": 8,
}
BUILDINGWORLD_BUCKET_NAMES: dict[int, str] = {
    3: "bw_germany", 4: "bw_japan", 5: "bw_south_africa",
    6: "bw_canada", 7: "bw_usa", 8: "bw_oceania",
}


def parse_source_key(source_key: str) -> tuple[str, str]:
    """Split a `source_key` into (pipeline, place), rejecting anything malformed (#167)."""
    if not SOURCE_KEY_RE.match(source_key):
        raise ValueError(f"#167: malformed source_key {source_key!r}; expected 'pipeline:place' "
                         f"with a lowercase pipeline (e.g. 'plateau:tokyo23ku')")
    if len(source_key.encode("utf-8")) > SOURCE_KEY_MAX_BYTES:
        raise ValueError(f"#167: source_key {source_key!r} is "
                         f"{len(source_key.encode('utf-8'))} bytes, over the {SOURCE_KEY_MAX_BYTES}-"
                         f"byte ({SOURCE_KEY_DTYPE}) storage width -- this is the truncation this "
                         f"ticket exists to prevent, so it raises here rather than silently cutting "
                         f"the key down to fit an HDF5 column")
    pipeline, _, place = source_key.partition(":")
    return pipeline, place


def make_source_key(pipeline: str, place: str) -> str:
    """Build the canonical 'pipeline:place' authority key, rejecting anything malformed (#167)."""
    key = f"{pipeline}:{place}"
    parse_source_key(key)
    return key


def region_id_of(source_key: str) -> int:
    """The region id a `source_key` belongs to.

    Every legacy pipeline (bag3d/nrw/plateau) maps 1:1 to a region id via `PIPELINE_REGION_ID`,
    regardless of place -- `region_id_of(make_source_key("plateau", "tokyo23ku"))` and
    `region_id_of(make_source_key("plateau", "osaka"))` agree. `"bw"` (BuildingWorld) is the one
    pipeline where place matters: #171 decided a broad style-bucket granularity, so the id comes
    from `BUILDINGWORLD_CITY_BUCKET[place]` instead. Any other unregistered pipeline, or an
    unregistered BuildingWorld place, raises rather than guessing.
    """
    pipeline, place = parse_source_key(source_key)
    if pipeline == "bw":
        if place not in BUILDINGWORLD_CITY_BUCKET:
            raise ValueError(f"#171: BuildingWorld place {place!r} (from {source_key!r}) has no "
                             f"style-bucket entry in BUILDINGWORLD_CITY_BUCKET "
                             f"{REGION_MAPPING_VERSION}")
        return BUILDINGWORLD_CITY_BUCKET[place]
    if pipeline not in PIPELINE_REGION_ID:
        raise ValueError(f"#167: pipeline {pipeline!r} (from {source_key!r}) has no region-id "
                         f"entry in PIPELINE_REGION_ID {REGION_MAPPING_VERSION}; #171 decides "
                         f"new pipelines' region-conditioning granularity before they can be "
                         f"added here")
    return PIPELINE_REGION_ID[pipeline]


def region_mapping_sha256() -> str:
    """SHA-256 of the versioned mapping's canonical serialization.

    Stamped into height-map-generator checkpoints (#163) so a checkpoint silently trained against
    one version of this mapping can never be loaded as if it means another. Covers both tables --
    a BuildingWorld bucket reassignment changes this hash exactly as a legacy pipeline remap would.
    """
    manifest = {"version": REGION_MAPPING_VERSION, "pipeline_region_id": PIPELINE_REGION_ID,
               "buildingworld_city_bucket": BUILDINGWORLD_CITY_BUCKET}
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
