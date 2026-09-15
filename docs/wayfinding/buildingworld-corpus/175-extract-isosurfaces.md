# #175 — Extract isosurfaces for new BuildingWorld rows and register the source in the surfaces pipeline

*Effort: fold BuildingWorld into the massing corpus (arm six on the gable bar), map
[#156](https://github.com/danvisai/SDFusion/issues/156). `ready-for-agent` label. Blocked by
[#174](https://github.com/danvisai/SDFusion/issues/174) (`ingest_buildingworld.py`); unblocks
[#176](https://github.com/danvisai/SDFusion/issues/176) (program pseudo-label regeneration) and
the vecset-latent pipeline.*

> Extract isosurfaces for the new BuildingWorld rows into `surfaces_buildingworld.h5` and register
> the new source in `dora_frozen_gate.SOURCES`/`load_surfaces()` so downstream consumers can find
> them.

## What this ticket built

`scripts/foundations/ingest_surfaces_buildingworld.py`: re-walks the same canonical zips #174's
`stage_city` read, re-applying the SAME per-city geometric correction so recovered vertices line
up with the SDF `real.h5` already stores, and recovers each BuildingWorld row's own Frame-N mesh
into one merged `surfaces_buildingworld.h5` (all 18 ingestable cities, keyed by `source_key`) —
mirroring `ingest_surfaces.py`'s existing per-source pipeline. `_resolve_collision` handles the one
real join-key hazard: `bag_id_for`'s 64-byte truncation makes Perth's long
`..._part<N>.obj` member names collide, so more than one `real.h5` row can legitimately share a
truncated `bag_id` — disambiguated by recomputing each candidate's SDF and matching it against
each contending row's stored one. `"buildingworld": "BW"` was registered in
`dora_frozen_gate.SOURCES`.

**Production run (2026-09-13/14):** `surfaces_buildingworld.h5` holds 1,526,778 rows — exactly
`real.h5`'s BuildingWorld count, no duplicates, no missing rows, `--verify` sampling at
IoU=1.0000/L1=0.00000 across multiple cities, including all 29 of Perth's collision rows correctly
resolved.

## Code review (this session) and the fixes it drove

A `/code-review` pass against the committed diff (`d68073c...39d683c`) found four real issues —
two Standards-axis, two Spec-axis, overlapping on the same root causes. Fixed in this same session,
with the production artifact re-verified rather than assumed unaffected (see each item).

### 1. `load_surfaces()`'s default silently grew ~44x, breaking a live caller

Registering `"buildingworld"` in `SOURCES` made `load_surfaces()`'s default row set jump from
35,776 to 1,562,554 — every caller taking the default now pays to materialise ~1.5M extra
verts/faces into RAM and build a `trimesh.Trimesh(...).volume` per row just for the winding check.
Worse: **`precompute_vecset_latents.py`'s default invocation (no `--limit`/`--stratify`) now
raised `SystemExit`** — its #161 pre-encode check reads `load_surfaces()`'s rows against
`corpus_ledger.h5`, which doesn't cover BuildingWorld yet (that's #177's job), so ~1.53M rows read
as "missing ledger entry" instead of the script quietly encoding the historical corpus it always
had. `dora_frozen_gate.main()`'s own gate paid the same unscoped cost for rows its own code already
says it structurally can't select (`held` is bounded to `< FROZEN_SPLIT_N_TOTAL`).

**Fix:** `load_surfaces(sources=None)` takes an explicit source list, defaulting to every
registered source (unchanged behaviour for a caller that wants everything). The three call sites
that don't want BuildingWorld yet — `dora_frozen_gate.main()`, `precompute_vecset_latents.py`,
`audit_surface_corpus.py`'s bag3d-only lookup — now pass `sources` explicitly, restoring their
pre-#175 cost and (for `precompute_vecset_latents.py`) their pre-#175 default behaviour. Covered by
a new `test_dora_frozen_gate.py` (default reads everything; `sources` filters; a missing per-source
file still just warns).

### 2. `_resolve_collision` had no actual confidence threshold

Its own docstring claimed to return rows with a "confident match" and drop the rest — but the
greedy nearest-pairing had no error floor at all; a candidate was accepted whenever it was the
least-bad option remaining, however poor the fit. Production data happened to be fine (see below),
but the code didn't guarantee it.

**Fix:** `COLLISION_MIN_IOU = 0.95` — the same "ALIGNED" bar `ingest_surfaces.revoxelize_iou_l1`
uses everywhere else in this pipeline — is now enforced: a pairing is only accepted once its
occupancy IoU against the stored SDF clears it; otherwise the row is left unresolved (logged, not
guessed). **Re-verified against production data directly** rather than assumed safe: re-running
`_resolve_collision` over both of Perth's real collision groups (29 rows total) against the actual
committed `real.h5`/canonical zip reproduced IoU=1.0000/L1=0.00000 on every one — the shipped
`surfaces_buildingworld.h5` needed no changes. New tests pin the threshold firing (a candidate
that's merely the only option left, but nothing like the stored geometry, is now rejected) and
being configurable (`min_iou=0.0` accepts the same pairing).

### 3. No resumability or bounded memory in `run()`

`run()` accumulated every city's verts/faces in plain Python lists and wrote once at the end — no
flush, no resume, unbounded memory over the full ~1.5M-row population. `ingest_buildingworld.py`
(#174) hit a real OOM from an analogous pattern on these same zips and fixed it with
`IncrementalCityWriter`'s periodic, `committed_rows`-gated flush; `run()` had regressed to the
pre-fix shape for a job of the identical class and scale.

**Fix:** `IncrementalSurfaceWriter`, adapted for `surfaces_buildingworld.h5`'s ragged per-row mesh
data (verts/faces grow by a variable count per row, unlike #174's fixed-shape SDF/footprint
columns — resuming derives the verts/faces boundary from `vert_offset[committed_rows]`/
`face_offset[committed_rows]` rather than a separate counter). One subtlety caught in testing before
it could bite production: a naive "auto-flush inside `add()`" would let a flush land BETWEEN two
`add()` calls for the same `bag_id` (a multi-row collision group), marking that `bag_id`
`already_done` after only the first row committed — a resume would then skip re-deriving the rest,
silently losing rows forever. Fixed by decoupling `add()` (buffer only) from `maybe_flush()`
(called by `run()` only at a `bag_id` group boundary), pinned by
`test_a_flush_mid_call_never_splits_one_bag_ids_rows_across_it` at `flush_every=1` (the worst case).
`--no_resume` was added to the CLI for an intentional fresh start.

Deliberately **not** added in this pass: a worker pool (`ingest_buildingworld.stage_city`'s
`workers>1` path). The reproduced, concrete risk here was unbounded memory + no resume, not
wall-clock speed — #175's own production run already completed single-process — and #174's own
history landed the sequential/robust version first, adding parallelism separately once that
baseline worked. `POOL_BATCH_SIZE` is named and reserved for that future addition rather than left
for it to invent a second bound.

The already-committed `surfaces_buildingworld.h5` did not need rewriting for this fix — it
finished a complete run before crashing was ever a risk. This is about the NEXT run (a new city, a
bug fix) inheriting the same robustness #174's own postmortem earned, not a retroactive repair.

### 4. Minor: `audit_surface_corpus.py`'s own `SOURCES` tuple

Still hardcodes `("bag3d", "nrw", "plateau")` for its per-source audit loop — BuildingWorld is not
audited by this diagnostic tool. Left as a disclosed, deliberate non-fix: whether this manual audit
script should also cover BuildingWorld is a scope decision, not a bug, and out of place in a
correctness pass. Its one `load_surfaces()` call site (an unrelated bag3d-row lookup) was still
scoped down (`sources=("bag3d",)`) since that's the same class of unscoped-cost issue as items 1-2.

## What was actually run this session (the fixes above)

- `test_ingest_surfaces_buildingworld.py`: 21 tests (was 15) — all existing tests still pass
  unchanged; six new ones cover `IncrementalSurfaceWriter` (write/flush/resume/no-resume/schema
  mismatch/the group-boundary flush-timing bug) and one new `TestRunEndToEnd` case
  (`test_run_resumes_a_partial_run_without_dropping_or_duplicating_rows`, using `--limit` to
  simulate an interrupted run).
- `test_dora_frozen_gate.py` (new, 4 tests): `load_surfaces`'s `sources` parameter — default reads
  every registered source, an explicit list filters, excluding buildingworld leaves other rows
  untouched, a missing per-source file still just warns.
- `test_precompute_vecset_latents.py`: unaffected, re-run to confirm (3 tests, unrelated
  `IncrementalCache` coverage).
- The Perth collision re-verification described in item 2, against real production `real.h5` and
  the real canonical zip (not synthetic data) — 29/29 rows re-resolved at IoU=1.0000.

## Status: fixed and tested; production artifact confirmed unaffected

`surfaces_buildingworld.h5` (1,526,778 rows, produced before this session) required no changes.
`dora_frozen_gate.py`, `precompute_vecset_latents.py`, and `ingest_surfaces_buildingworld.py` were
fixed and are covered by new or updated tests, all passing. #176 (which runs independently of
`load_surfaces()`/`surfaces_buildingworld.h5` — see its own doc) was unaffected by any of this.
