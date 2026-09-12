# Data sources

License and provenance for every real-building source ingested into `data/real_massing_v1/real.h5`
(and its precursor `data/bag3d_v1/bag3d.h5`). Closes the compliance gap
[#5](https://github.com/danvisai/SDFusion/issues/5)'s audit found (see also its own write-up,
`docs/wayfinding/solid-first-subtractive-modeling/5-data-audit.md`): none of the ingesters recorded
a license, credit string, or source snapshot anywhere, and no manifest existed.

Machine-checkable source of truth: `scripts/foundations/data_sources.py`'s `DATA_SOURCES` dict.
`scripts/foundations/test_data_sources.py::TestRealH5Coverage` fails loudly if a `source_id` is
ever added to `real.h5` without a matching entry here and in that dict.

| `source_id` | region | source | license | required credit | ingester |
|---|---|---|---|---|---|
| 0 | NL | [3D BAG](https://docs.3dbag.nl/en/copyright/) | CC BY 4.0 | `© 3DBAG by tudelft3d and 3DGI` | `scripts/ingest_3dbag.py` |
| 1 | DE | [NRW LoD2 CityGML](https://www.govdata.de/dl-de/zero-2-0) (source data at opengeodata.nrw.de) | `dl-de/zero-2-0` | none required | `scripts/foundations/ingest_citygml_lod2.py --source nrw` |
| 2 | JP | [PLATEAU](https://www.mlit.go.jp/plateau/) (MLIT) | CC BY 4.0 | `Data source: Project PLATEAU by MLIT (Ministry of Land, Infrastructure, Transport and Tourism, Japan)` | `scripts/foundations/ingest_citygml_lod2.py --source plateau` |

## Notes

- **3D BAG** (`source_id=0`) is unambiguously CC BY 4.0 — confirmed directly against the source's
  own copyright page, linked above.
- **NRW** (`source_id=1`) is Germany's `dl-de/zero-2-0` ("Data licence Germany – Zero – Version
  2.0"), a public-domain-equivalent license that requires no attribution.
- **PLATEAU** (`source_id=2`) offers a choice of CC BY 4.0 / ODbL / ODC-BY per dataset; this project
  had never recorded which one it was claiming. **CC BY 4.0 is chosen here**, for consistency with
  3D BAG's own license and because it is the least restrictive of the three that still requires
  attribution — recorded as a decision, not left implicit.
- Every ingester stamps its output `.h5` with `attrs["source"]` and `attrs["ingested_at"]` (an ISO
  timestamp of the ingestion run) — both 3D BAG and PLATEAU are living, periodically-republished
  datasets, so a per-run snapshot stamp is the fetch date, not a fixed dataset version number
  neither source's own API exposes. `scripts/foundations/concat_real_massing.py` reads each
  source's stamp and aggregates them onto `real.h5` itself as `attrs["source_provenance"]` (a JSON
  object keyed by `source_id`) — the ingesters' own files are intermediates nothing downstream
  reads directly, so `real.h5` is the file that actually needs to carry this.
- Any new `source_id` added to `real.h5` must get a matching row here and in
  `scripts/foundations/data_sources.py`'s `DATA_SOURCES` dict, or
  `test_data_sources.py::TestRealH5Coverage` fails the next time it runs.
