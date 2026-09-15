# #167 — Decide the single provenance-authority field for region and stratification schemes

*Effort: fold BuildingWorld into the massing corpus (arm six on the gable bar), map
[#156](https://github.com/danvisai/SDFusion/issues/156). `wayfinder:grilling`. Decided with
project-owner sign-off in-session; blocks [#171](https://github.com/danvisai/SDFusion/issues/171)
and [#174](https://github.com/danvisai/SDFusion/issues/174).*

## The problem

Region conditioning today is derived from *which surface file a row came from*
(`precompute_vecset_latents.py`: `region=src_id[src]`, with `src` set by
`dora_frozen_gate.SOURCES = {bag3d, nrw, plateau}`), not from `real.h5`'s own `source_id` column —
a pipeline id that happens to be 1:1 with country today. BuildingWorld breaks that 1:1 immediately:
BuildingWorld Tokyo (same country as PLATEAU, different capture pipeline) and BuildingWorld Berlin
(same country as NRW, different capture pipeline) both need identities today's scheme has no slot
for. `class_label` (S16, fixed-width) is currently an exact alias of `source_id` and would truncate
a per-city label like `'BW_GreaterGeelong'` (17 bytes); `style_id` is uniformly `8` across all
35,776 existing rows — a dead field today.

## Decision

`source_key` — a string of the form `'pipeline:place'` (e.g. `'plateau:tokyo23ku'`,
`'nrw:LoD2_32_280_5657'`, `'bw:Berlin'`) — is the single provenance authority going forward, per
the issue's own proposed design. Every region/stratification/dedup scheme downstream becomes a
versioned, pure function of that one field, and the mapping's version/hash is recorded durably.

Implemented as `scripts/foundations/source_provenance.py`:

- `make_source_key(pipeline, place)` / `parse_source_key(key)` — build and validate the canonical
  format; malformed keys (uppercase pipeline, empty place, stray colons, embedded whitespace)
  raise rather than silently coerce.
- `PIPELINE_REGION_ID` — the versioned (`REGION_MAPPING_VERSION = "v1"`) pipeline → region-id
  table. `region_id_of(source_key)` is the first scheme built on it, and today only reproduces the
  three pipelines the frozen corpus already has (`bag3d`→0, `nrw`→1, `plateau`→2 — unchanged from
  `stratified_split.SOURCE_NAMES` / `dora_frozen_gate.SOURCES` / #163's `N_REGIONS` comment).
- `region_mapping_sha256()` — SHA-256 of the table's canonical (order-independent) JSON
  serialization, now stamped into every height-map-generator checkpoint via the seam #163 reserved
  for it (`cache_provenance`'s `region_mapping_sha256` field, previously hardcoded `None`).
  `validate_checkpoint_provenance` now also checks it on the no-cache load path, since the hash
  needs no training cache to compute — closing a gap in #163 where that path validated channel
  count and region count but not the mapping identity.
- `SOURCE_KEY_MAX_BYTES = 64` (`SOURCE_KEY_DTYPE = "S64"`) — the storage-width decision this
  ticket's own motivating example asked for and had not actually made (code review): reuses
  `bag_id`'s existing S64 HDF5 width (`concat_real_massing.py`) rather than #174 inventing a new
  one. `parse_source_key`/`make_source_key` raise on anything over that width instead of letting an
  HDF5 write truncate it silently — the exact hazard `class_label` (S16) demonstrated.

## Scope: this ticket defines the authority; it does not yet populate it

`region_id_of` deliberately has **no entries for any BuildingWorld pipeline**. Map #156's standing
preference #4 rules out letting a fine-grained id absorb a data-quality defect invisibly, and which
region id (or a new one) BuildingWorld's per-city pipelines get is a granularity question, not a
mechanism question — that is #171's decision, blocked on this ticket precisely so #171 has an
authority field to derive schemes from rather than inventing its own. Calling `region_id_of` on an
unregistered pipeline (e.g. `'bw:Berlin'`) raises, pointing at #171, rather than guessing.

The existing 35,776 rows are **not** retrofitted with a `source_key` column in this ticket —
`real.h5`/`corpus_ledger.h5` keep `source_id` unchanged. `source_key` is forward-looking: #174's
`ingest_buildingworld.py` is the first writer of it, once #171 extends `PIPELINE_REGION_ID` (or
introduces a sibling table) to cover BuildingWorld's pipelines and bumps
`REGION_MAPPING_VERSION` accordingly.

## Verification

```bash
env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/test_source_provenance.py
env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/test_train_height_map_generator.py
```

The first file covers `source_key` format validation, `region_id_of`'s known-pipeline/unregistered-
pipeline behavior, and the mapping hash's determinism and order-independence. The second's
`TestCheckpointProvenance` class was updated to expect a real hash instead of `None`, plus new
coverage for a `region_mapping_sha256` mismatch and for the no-cache validation path.

## Status: decided (2026-09-11)

Project-owner-approved: adopt `source_key` as proposed by the issue; scope this ticket to defining
the authority (module + versioned mapping + checkpoint hash), not retrofitting existing sources.
