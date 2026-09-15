"""#152 -- the license/provenance manifest for every source in `data/real_massing_v1/real.h5`.

#5's audit found the compliance gap this closes: two of the corpus's three sources are CC BY 4.0
(3D BAG confirmed; PLATEAU conditionally -- MLIT offers a choice of CC BY 4.0 / ODbL / ODC-BY and
this repo had never recorded which one it claims), and neither ingester wrote a license, credit
string, or source snapshot anywhere. `DATA_SOURCES` below is the single machine-checkable source of
truth; `docs/DATA_SOURCES.md` is its human-readable rendering, kept in sync by hand (only 3 rows) and
cross-checked by `test_data_sources.py` against drift.

Run:  env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/data_sources.py
Test: env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/test_data_sources.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.foundations.stratified_split import SOURCE_NAMES  # noqa: E402

H5 = REPO / "data/real_massing_v1/real.h5"
DOC = REPO / "docs/DATA_SOURCES.md"

# `source_id` -> provenance. `region` is drawn from `stratified_split.SOURCE_NAMES` directly (not
# re-literalized) so the two can never independently drift on what id 0/1/2 mean.
_LICENSES: Dict[int, dict] = {
    0: dict(
        name="3D BAG",
        license="CC BY 4.0",
        credit="© 3DBAG by tudelft3d and 3DGI",
        url="https://docs.3dbag.nl/en/copyright/",
        ingester="scripts/ingest_3dbag.py",
    ),
    1: dict(
        name="NRW LoD2 CityGML",
        license="dl-de/zero-2-0",
        credit="none required",
        url="https://www.govdata.de/dl-de/zero-2-0",
        ingester="scripts/foundations/ingest_citygml_lod2.py --source nrw",
    ),
    2: dict(
        name="PLATEAU",
        # PLATEAU offers a choice of CC BY 4.0 / ODbL / ODC-BY; CC BY 4.0 chosen here for
        # consistency with 3D BAG's own license, per #152's own recommendation.
        license="CC BY 4.0",
        credit="Data source: Project PLATEAU by MLIT (Ministry of Land, Infrastructure, "
              "Transport and Tourism, Japan)",
        url="https://www.mlit.go.jp/plateau/",
        ingester="scripts/foundations/ingest_citygml_lod2.py --source plateau",
    ),
    # #174: BuildingWorld rows. -1 is a deliberate, disclosed "region not yet assigned" sentinel
    # (#167: region_id_of raises rather than guesses for an unregistered pipeline; #171 has not
    # yet decided BuildingWorld's region-conditioning granularity). Per-row provenance is carried
    # by `source_key` ('bw:<CitySlug>'), the actual #167 authority field -- this entry exists only
    # so `test_data_sources.py::TestRealH5Coverage` has a manifest row for the sentinel itself, not
    # because -1 identifies a real region. See
    # docs/wayfinding/buildingworld-corpus/174-ingest-buildingworld.md.
    -1: dict(
        name="BuildingWorld (region not yet assigned; see source_key per row)",
        license="see per-city source_key; #171 decides region-conditioning granularity before "
                "training conditions on these rows",
        credit="BuildingWorld: A Structured 3D Building Dataset for Urban Foundation Models "
              "(arXiv:2511.06337)",
        url="https://arxiv.org/abs/2511.06337",
        ingester="scripts/foundations/ingest_buildingworld.py",
    ),
}
DATA_SOURCES: Dict[int, dict] = {
    sid: dict(region=SOURCE_NAMES[sid], **fields) for sid, fields in _LICENSES.items()
}


def missing_sources(h5_path: Path = H5) -> List[int]:
    """Every distinct `source_id` present in `h5_path` with no matching `DATA_SOURCES` entry.
    Empty means every source actually in the corpus is covered by the manifest."""
    import h5py
    import numpy as np

    with h5py.File(h5_path, "r") as f:
        present = {int(s) for s in np.unique(f["source_id"][:])}
    return sorted(present - set(DATA_SOURCES))


def main() -> None:
    missing = missing_sources()
    if missing:
        print(f"[data_sources] {len(missing)} source_id(s) in {H5.relative_to(REPO)} have no "
             f"DATA_SOURCES entry: {missing} -- update scripts/foundations/data_sources.py and "
             f"docs/DATA_SOURCES.md")
        raise SystemExit(1)
    print(f"[data_sources] every source_id in {H5.relative_to(REPO)} is covered by "
         f"{len(DATA_SOURCES)} DATA_SOURCES entries")
    for sid, s in sorted(DATA_SOURCES.items()):
        print(f"  {sid}  {s['region']:<3s} {s['name']:<40s} {s['license']}")


if __name__ == "__main__":
    main()
