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

SOURCE_KEY_RE = re.compile(r"^[a-z0-9_]+:[A-Za-z0-9_.-]+$")

REGION_MAPPING_VERSION = "v1"

# pipeline -> region id. Unchanged from today's `source_id` convention (#163's N_REGIONS comment:
# "source corpora: 0 NL / 1 DE / 2 JP", matching `stratified_split.SOURCE_NAMES` and
# `dora_frozen_gate.SOURCES`) -- #167 freezes it as version "v1" so #171/#174 have one place to
# derive region ids from `source_key` rather than inventing their own. This ticket does not
# retrofit `stratified_split.py`/`dora_frozen_gate.py` to read from this table -- they keep their
# own copies for now; only new, `source_key`-based consumers are guaranteed to agree with this one.
# New pipelines (BuildingWorld's) are added by a future, separately versioned entry once #171
# decides their region-conditioning granularity.
PIPELINE_REGION_ID: dict[str, int] = {
    "bag3d": 0,
    "nrw": 1,
    "plateau": 2,
}


def parse_source_key(source_key: str) -> tuple[str, str]:
    """Split a `source_key` into (pipeline, place), rejecting anything malformed (#167)."""
    if not SOURCE_KEY_RE.match(source_key):
        raise ValueError(f"#167: malformed source_key {source_key!r}; expected 'pipeline:place' "
                         f"with a lowercase pipeline (e.g. 'plateau:tokyo23ku')")
    pipeline, _, place = source_key.partition(":")
    return pipeline, place


def make_source_key(pipeline: str, place: str) -> str:
    """Build the canonical 'pipeline:place' authority key, rejecting anything malformed (#167)."""
    key = f"{pipeline}:{place}"
    parse_source_key(key)
    return key


def region_id_of(source_key: str) -> int:
    """The region id a `source_key` belongs to, per the versioned `PIPELINE_REGION_ID` table.

    Raises for any pipeline not yet in the table rather than guessing -- BuildingWorld's pipelines
    are deliberately absent until #171 decides their granularity (see module docstring).
    """
    pipeline, _ = parse_source_key(source_key)
    if pipeline not in PIPELINE_REGION_ID:
        raise ValueError(f"#167: pipeline {pipeline!r} (from {source_key!r}) has no region-id "
                         f"entry in PIPELINE_REGION_ID {REGION_MAPPING_VERSION}; #171 decides "
                         f"new pipelines' region-conditioning granularity before they can be "
                         f"added here")
    return PIPELINE_REGION_ID[pipeline]


def region_mapping_sha256() -> str:
    """SHA-256 of the versioned mapping's canonical serialization.

    Stamped into height-map-generator checkpoints (#163) so a checkpoint silently trained against
    one version of this mapping can never be loaded as if it means another.
    """
    manifest = {"version": REGION_MAPPING_VERSION, "pipeline_region_id": PIPELINE_REGION_ID}
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
