"""#171: backfill BuildingWorld rows' `region` ledger column from the -1 ingestion sentinel to
their real style-bucket id, now that #171 has decided the bucket table
(`source_provenance.BUILDINGWORLD_CITY_BUCKET`).

#174 left every BuildingWorld row's `region` at -1 ("a versioned-nowhere sentinel, not a guess at
#171's eventual region-id scheme" -- its own comment). This script resolves that sentinel via
`region_id_of(source_key)` and rewrites `corpus_ledger.h5` in place. Legacy (NL/DE/JP) rows'
region values are read back and passed through bit-identical; only BuildingWorld rows (identified
by `source_id`, never by row-id range) are ever changed, and only from -1 to a real bucket id -- an
already-resolved BuildingWorld row is refused, not silently reassigned, so a second run is a no-op
rather than a further mutation, and a genuinely already-assigned row is corruption to investigate,
not overwrite.

Run: env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/assign_buildingworld_region_buckets.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.foundations.corpus_ledger import LEDGER_PATH, read_ledger, write_ledger  # noqa: E402
from scripts.foundations.recover_program_labels_buildingworld import (  # noqa: E402
    BUILDINGWORLD_SOURCE_ID,
)
from scripts.foundations.source_provenance import region_id_of  # noqa: E402
from utils.frozen_corpus import REAL_CORPUS_PATH, open_real_corpus  # noqa: E402

UNASSIGNED_SENTINEL = -1


def compute_updated_region(ledger: dict, source_id: np.ndarray, source_key: np.ndarray) -> np.ndarray:
    """The ledger's `region` column with every BuildingWorld row's sentinel resolved.

    `source_id`/`source_key` are indexed by raw corpus row id, exactly like every other consumer
    in this codebase (e.g. `buildingworld_baseline.select_populations`) -- `ledger["row"]` supplies
    that index, never an assumption about row-id ranges.
    """
    rows = ledger["row"]
    region = ledger["region"].copy()
    is_bw = source_id[rows] == BUILDINGWORLD_SOURCE_ID
    already_assigned = (region != UNASSIGNED_SENTINEL) & is_bw
    if np.any(already_assigned):
        bad = rows[already_assigned][:5].tolist()
        raise ValueError(f"#171: {int(already_assigned.sum())} BuildingWorld row(s) (e.g. {bad}) "
                         f"already carry a non-sentinel region; refusing to silently reassign them "
                         f"-- investigate before rerunning")
    mislabelled = (region == UNASSIGNED_SENTINEL) & ~is_bw
    if np.any(mislabelled):
        bad = rows[mislabelled][:5].tolist()
        raise ValueError(f"#171: {int(mislabelled.sum())} non-BuildingWorld row(s) (e.g. {bad}) "
                         f"carry the BuildingWorld sentinel region; ledger looks corrupt")
    bw_idx = np.flatnonzero(is_bw)
    if not len(bw_idx):
        return region
    # Resolve each of the ~18 distinct cities once, not each of the ~1.5M rows.
    keys = source_key[rows[bw_idx]]
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    resolved = np.array([region_id_of(k.decode()) for k in unique_keys], np.int32)
    region[bw_idx] = resolved[inverse]
    return region


def run(h5_path: Path = REAL_CORPUS_PATH, ledger_path: Path = LEDGER_PATH) -> dict:
    ledger = read_ledger(ledger_path)
    with open_real_corpus(h5_path) as corpus:
        source_id = corpus["source_id"][:]
        source_key = corpus["source_key"][:]
    new_region = compute_updated_region(ledger, source_id, source_key)
    n_changed = int(np.sum(new_region != ledger["region"]))
    write_ledger(ledger["row"], new_region, ledger["held_out"], ledger["height_m"],
                path=ledger_path, source="assign_buildingworld_region_buckets (#171)")
    report = dict(rows_changed=n_changed, rows_total=len(ledger["row"]),
                 bucket_counts={int(b): int(np.sum(new_region[
                     source_id[ledger["row"]] == BUILDINGWORLD_SOURCE_ID] == b))
                     for b in sorted(set(new_region.tolist()))})
    print(f"[region-bucket] {n_changed}/{len(ledger['row'])} rows resolved; "
         f"bucket counts {report['bucket_counts']}", flush=True)
    return report


if __name__ == "__main__":
    run()
