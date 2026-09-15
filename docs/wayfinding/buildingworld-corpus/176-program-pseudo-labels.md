# #176 — Regenerate program pseudo-labels for new BuildingWorld rows

*Effort: fold BuildingWorld into the massing corpus (arm six on the gable bar), map
[#156](https://github.com/danvisai/SDFusion/issues/156). `ready-for-agent` label. Blocked by
[#175](https://github.com/danvisai/SDFusion/issues/175) (isosurface extraction); blocks
[#177](https://github.com/danvisai/SDFusion/issues/177) (the roof-family-stratified split for
these rows), which needs this ticket's `family` output as its stratification axis.*

> Run the existing, proven beam-search fitter (`recover_massing_programs.py`, `fit_program_beam`)
> against the newly-ingested BuildingWorld rows to produce program pseudo-labels
> (Layer/Ramp/CutRoof, with `dl_ops`/`dl_planar_fraction` diagnostics), exactly as the existing
> NL/DE/JP rows already have.

## What this ticket builds

`scripts/foundations/recover_program_labels_buildingworld.py`. For every `real.h5` row #174
stamped as BuildingWorld (`source_id == -1`): read the row's own `sdf`/`footprint` occupancy,
`height_field()` it, run `fit_program_beam` at its own defaults, and record `family`
(flat/gable/hip/complex), `dl_ops`, `dl_planar_fraction`, `is_shed`, and the fitted op-type
sequence. Output: `data/real_massing_v1/program_labels_buildingworld.h5`, one row per building.

## Decisions this ticket had to make that its own issue text didn't fully specify

### Why `real.h5` directly, not a re-voxelised `surfaces_buildingworld.h5` mesh

#175's own module docstring names this script as a downstream consumer of
`surfaces_buildingworld.h5`/`load_surfaces()`. But the fitter itself
(`recover_massing_programs.height_field` → `fit_program_beam`) has never taken a mesh as input —
not for NL/DE/JP in `recover_massing_programs.main()`, not for the 35,623-row training cache in
`train_height_map_generator.build_cache`, and not for the NL/DE/JP comparison population in #159's
own pilot (`pilot_buildingworld_roof_families._region_building_task`), which reads `real.h5`'s own
`sdf`/`footprint` directly. #174 wrote BuildingWorld's `sdf`/`footprint` through the identical
`building_to_sdf` call #175's mesh recovery re-derives its own geometry from, so re-voxelising
#175's mesh a second time would be strictly more work for the same numbers, and would put this
population on a different input than every other population's fit already uses. `real.h5` is
therefore the fitter's real input; `surfaces_buildingworld.h5` is used only to KNOW which rows are
BuildingWorld's (see next section) and as a corpus-consistency cross-check.

### Row selection: `source_id == -1`, cross-checked against #175's row count

`real.h5`'s `source_id` column carries #174's disclosed sentinel (`-1`) for every BuildingWorld
row, since no real region id exists for it yet (#171 not landed). This is the primary row
selector — direct, and independent of any other file. `cross_check_row_count()` additionally
asserts this count matches `surfaces_buildingworld.h5`'s own `row` count (#175 already verified
that file's population is exactly BuildingWorld's, deduplicated past the Perth `bag_id`-truncation
collision) and refuses to run if they disagree, catching either file having drifted before a
multi-hour fit runs against the wrong population — not because the mesh file supplies geometry
here, but because it is the one place BuildingWorld's row count was independently re-verified.

### Reusing #159's fit recipe, but not its function

`fit_one` inlines `pilot_buildingworld_roof_families.fit_and_classify`'s own body (same
`fit_program_beam` defaults, same `roof_family` call, both imported and reused unchanged) rather
than calling that function directly, because `fit_and_classify`'s return contract doesn't include
the op-type sequence this ticket also wants to record — the ticket text names `Layer/Ramp/CutRoof`
explicitly, not just the roof-family summary. Calling it and then re-fitting separately to recover
`ops` would double the per-building cost (~0.25s → ~0.5s), doubling a multi-hour production run for
no new information; inlining the ~10-line wrapper was judged cheaper than that, and than changing
`fit_and_classify`'s own contract underneath #159's already-committed, already-tested pilot.

### Output format: a flat per-row table, not #6's CutRoof-withheld slot cache

`train_height_map_generator.build_program_cache` is what NL/DE/JP rows' program labels are
consumed as for *training* (`(assign, types, planes)` slots, `program_labels.npz`) — but that
format is deliberately fit with `CutRoof` withheld (`SLOT_TYPES = ("Layer", "Ramp")`), because
`program_to_slots` needs every operation to carry a plane and `CutRoof`'s surface is a distance
transform, not one. A `CutRoof`-withheld fit cannot produce `family` (`roof_family` needs to see
`CutRoof`'s own `kind` to tell hip from gable), which is the actual deliverable #177 is blocked on.
That cache is also keyed by #161's ledger (`row`/`region`/`held_out`/`height_m`, split out of
`vecset_latents.h5`), which does not yet contain BuildingWorld rows — adding them is #177's own
job, and #177 is blocked BY this ticket, so this artifact cannot depend on the ledger already
having them. `program_labels_buildingworld.h5` is therefore a new, independent, row-id-keyed table
against `real.h5` directly: `row`, `ok`, `family` (S8), `dl_ops` (int16), `dl_planar_fraction`
(float32), `is_shed` (uint8), `ops` (S64, semicolon-joined type sequence), `fail_stage` (S32).
Polygon regions are not stored — the ticket asks for pseudo-labels and diagnostics, not replayable
programs, and at ~1.5M rows a full-vocabulary program's ring geometry (comparable to
`program_recovery_714.json`'s ~9 KB/building) would be on the order of 14 GB of JSON, which no
existing consumer of this ticket's output needs.

## Resumability and the pool-batching lesson from #174

`IncrementalRowWriter` mirrors `ingest_buildingworld.IncrementalCityWriter`'s resizable,
resumable, `committed_rows`-gated flush pattern. `run()`'s worker-pool path submits tasks in
bounded `POOL_BATCH_SIZE` (2,000) chunks, each fully drained before the next, for the identical
reason `ingest_buildingworld.py`'s own `POOL_BATCH_SIZE` exists: an earlier unbounded
`imap_unordered` over a whole city's tasks let completed-result buffering in the main process grow
without bound under I/O contention and got OOM-killed partway through Berlin (documented in
[#174's own doc](174-ingest-buildingworld.md)). This script runs the same shape of job — ~1.5M
CPU-bound tasks, one h5py-writing consumer — so it inherits that fix rather than re-discovering it.

## Code review, before the production run

A `/code-review` pass against the freshly-written script (before it was launched at full scale)
found one hard Standards violation, two judgement calls, and one real Spec bug — all fixed before
the production run below, so the numbers it reports are from the fixed code:

- **Duplicated read logic** between `_row_task` (pool worker) and `run()`'s sequential path.
  Extracted `_fit_corpus_row(g, row)`, used by both.
- **`fit_one` near-duplicated `pilot_buildingworld_roof_families.fit_and_classify`** just to also
  return the op-type sequence. Rather than keep two copies of the same fit-plus-classify logic,
  `fit_and_classify` itself was extended to return `ops` (a small, disclosed, backward-compatible
  addition — no other caller depended on its old return shape), and `fit_row` now calls it
  directly; the separate `fit_one` wrapper was deleted.
- **`report()` read the output file one row at a time** (`f["ok"][i]`, etc.) — fine at smoke scale,
  a real cost at 1.5M rows. Rewritten to bulk-read each column once.
- **Real bug, fixed:** `cross_check_row_count` treated ANY mismatch between `real.h5`'s
  `source_id`-derived count and `surfaces_buildingworld.h5`'s row count as fatal (`SystemExit`) —
  but #175's own `run()` legitimately drops rows it can't confidently recover a mesh for, so a
  smaller surfaces count is an expected outcome, not corruption. Softened to a non-fatal warning in
  both directions.

## What was actually run this session

- The full unit-test suite (`test_recover_program_labels_buildingworld.py`, 16 tests, post-fix):
  row selection against the `source_id` sentinel, the `surfaces_buildingworld.h5` row-count
  cross-check (missing file warns; matching, smaller, AND larger counts all warn-only, never
  raise), `fit_row`'s ok/fail paths (a flat synthetic box fits zero ops and reads `flat`; a ridged
  synthetic box fits a non-empty, well-formed op sequence; an empty footprint fails with
  `empty_height_field`; `fit_row` agrees with `fit_and_classify` called directly),
  `IncrementalRowWriter`'s write/flush/resume/no-resume/schema-mismatch behaviour, and `run()` end
  to end against a synthetic `real.h5`-shaped fixture built the same way
  `test_ingest_buildingworld.py::TestCombineIntoReal` does (the real frozen 35,776-row
  `bag_id`/`height_m` prefix, required by #162's own hash check, plus synthetic BuildingWorld rows
  appended after it) — confirming only `source_id == -1` rows are fit and that resuming a
  partially-committed run skips already-done rows without duplicating them.
  `test_pilot_buildingworld_roof_families.py`'s own suite (21 tests post the `ops`-field addition)
  was re-run and still passes.
- Real smoke runs against production `real.h5`: 50 rows sequential (`--workers 1`) and 500 rows on
  an 8-worker pool (`--workers 8`), 0 failures across all 550, ~0.24s/building/core observed on
  both paths — matching the issue's own "~0.25s/building" measurement, and confirming the
  `source_id`/`surfaces_buildingworld.h5` row-count cross-check passes against the real,
  already-committed #175 artifact.

## The production run (2026-09-14/15)

Run for real against every BuildingWorld row, `--workers 30` (this node has 32 cores), resumable
writer, default full-vocabulary recipe. Started 21:07, completed 00:23 — **11,756 s (3h16m)** for
**1,526,778 rows**, averaging **~0.23 s/building/core** across 30 workers, matching the smoke-run
rate and the issue's own "~0.25 s/building" estimate. No `run_in_background` interruption, no OOM,
no resume needed — the resumable writer existed as a safety net (matching #174/#175's own
established convention for a job of this scale and duration), not because anything failed.

**Result**: `data/real_massing_v1/program_labels_buildingworld.h5`, 1,526,778 rows, **0 failures**
(`ok=1,526,778`, `fail_stages={}` — every row's `height_field` succeeded and `fit_program_beam`
completed without raising).

| family | count | fraction |
|---|---:|---:|
| flat | 378,223 | 24.8% |
| gable | 560,865 | 36.7% |
| complex | 521,117 | 34.1% |
| hip | 66,573 | 4.4% |

`dl_ops` median **2.00**, `dl_planar_fraction` median **0.250**, `n_shed` (lone-`Ramp` programs)
**69,089** (4.5%). All four families are populated at corpus scale (not collapsed into "complex,
many-Layer" the way #159's pilot asked whether BuildingWorld would) — #177 has a real stratification
axis to build its held-out split against.

## Status: implemented, unit-tested, code-reviewed and fixed, and run to completion in production
(2026-09-14/15)

`data/real_massing_v1/program_labels_buildingworld.h5` now holds program pseudo-labels for all
1,526,778 new BuildingWorld rows. #177 (roof-family-stratified split for these rows) is unblocked.
