# #174 — Write `ingest_buildingworld.py`

*Effort: fold BuildingWorld into the massing corpus (arm six on the gable bar), map
[#156](https://github.com/danvisai/SDFusion/issues/156). `ready-for-agent` label. Applies five
already-decided tickets (`#160`, `#165`, `#166`, `#167`, `#168`) to real data for the first time;
does not reopen any of them.*

## What this ticket builds

`scripts/foundations/ingest_buildingworld.py`, mirroring `ingest_citygml_lod2.py`'s structure and
calling the existing, UNMODIFIED `building_to_sdf` (`scripts/ingest_3dbag.py`). Two stages:

1. **Stage** (`stage_city`) — per city, incrementally ingest every candidate mesh in its canonical
   zip (`profile_buildingworld_meshes.canonical_zip`, reused rather than reimplemented) into
   `data/real_massing_v1/buildingworld_staging/<CitySlug>.h5`, via `IncrementalCityWriter` — a
   resizable, resumable writer modelled directly on `precompute_vecset_latents.py`'s own
   flush/committed-rows pattern. This is not an aesthetic choice: Berlin alone has ~523k candidate
   meshes, and holding every kept SDF volume for a city that size in memory before one write is not
   viable (hundreds of GB), nor is losing all progress to a single crash acceptable for a job that
   may run for hours.
2. **Combine** (`combine_into_real`) — rebuild a `real.h5`-shaped file = the untouched original's
   rows + every staged city's rows, verify the frozen-prefix pin (#162) on the REBUILT file, then
   atomically replace `--out`. `real.h5`'s existing datasets have no `maxshape` (confirmed directly:
   every column's `maxshape == shape`), so true in-place resize is impossible — "append" is
   necessarily a safe rebuild, never a mutation of the original file until the very last
   `Path.replace`.

## Decisions this ticket had to make that its own issue text didn't fully specify

The blocking tickets (#160/#165/#166/#167/#168) answer *what* to filter/correct/write. They do not
answer every mechanical *how*. Three of those gaps are disclosed here rather than resolved silently.

### `source_id = -1` for every BuildingWorld row

The row schema #174's own issue text names is `(sdf, footprint, height_m, provenance key,
class_label, style_id)` — no `source_id`. But `real.h5`'s `source_id` column is a required,
fixed-width `int32` array with one entry per row; a newly appended row cannot simply lack a value.
Three options were considered:

- **Guess a real region id** (e.g. a new integer `3` for "BuildingWorld"). Rejected: this is
  exactly the granularity decision #171 owns ("which region id... is #171's decision, not this
  module's" — #167's own text), and #171 has not landed. A single pooled id would also be wrong the
  moment #171 decides per-city ids are needed, and by then it would mean rewriting `source_id` for
  every already-ingested BuildingWorld row.
- **Leave it at HDF5's default fill (0)**. Rejected outright: 0 already means NL. Silently
  mislabelling every BuildingWorld row as Dutch is precisely the "letting a fine-grained id absorb a
  data-quality defect invisibly" anti-pattern map #156's standing preference #4 warns against.
- **An explicit, disclosed sentinel (`-1`)**. Chosen. It cannot collide with a real region id, it is
  registered (not silently uncovered) in `data_sources.py`/`docs/DATA_SOURCES.md` — required because
  `test_data_sources.py::TestRealH5Coverage` reads `real.h5`'s own `source_id` values against that
  manifest and fails loudly on anything unregistered — and in `stratified_split.SOURCE_NAMES` (a
  pure display-name lookup with a safe `.get()` fallback everywhere except
  `void_semantic_sample.py`'s hard index, which this addition also protects). None of these three
  files' actual STRATIFICATION or REGION-CONDITIONING logic changes; only a display name and a
  manifest row are added. `region_id_of` (#167's real authority-derivation function) is never
  called for BuildingWorld rows — it would raise, correctly, since `region_id_of` is documented to
  raise rather than guess for an unregistered pipeline.

Region conditioning at TRAINING time (the `region` column of `vecset_latents.h5`/`corpus_ledger.h5`)
is derived downstream, in the surface-extraction/vecset-precompute pipeline (#175/#176), not from
`real.h5`'s `source_id` directly (#167's own text: "region conditioning today is derived from WHICH
SURFACE FILE a row came from... not from real.h5's own source_id column"). Since #175/#176 have not
yet run against these rows, `-1` cannot silently leak into a training run today — it will surface
the moment #171/#175 need to decide what BuildingWorld rows' region actually is.

### `source_key`/`defect_class` as new `real.h5` columns

Two columns beyond the ticket's literal list, both required by the tickets that decided them, not
invented here:

- `source_key` (`S64`, `'bw:<CitySlug>'`) — #167: "`source_key`... is the single provenance
  authority going forward... #174's `ingest_buildingworld.py` is the first writer of it."
- `defect_class` (`S32`, `watertight`/`floor_open`/`non_boundary_defect`) — #166: "Per-row
  provenance for BuildingWorld must carry a `defect_class` field... so this bucket stays traceable."

Per #167's own scope note ("The existing 35,776 rows are not retrofitted with a `source_key`
column"), the historical rows get an empty placeholder (`b""`) for both columns when the schema is
extended, not a derived-but-invented value — verified by
`TestCombineIntoReal.test_combine_appends_and_preserves_frozen_prefix`.

### City-slug convention (`source_key`'s `place` component)

`source_provenance.SOURCE_KEY_RE` requires `place` to match `[A-Za-z0-9_.-]+` — no spaces. Four of
BuildingWorld's 19 city directory names have spaces ("Cape Town", "Greater Geelong", "New York", "San
Francisco"). `CITY_SLUG` strips them (`"Cape Town" -> "CapeTown"`), matching #167's own motivating
example verbatim (`'BW_GreaterGeelong'`). `class_label` (`S16`, legacy) is built from the same slug
and explicitly truncated to 16 bytes with `.encode()[:16]` — the exact, disclosed truncation #167's
own text flags as `class_label`'s known failure mode, done on purpose and visibly rather than left to
numpy's silent fixed-width truncation.

## Per-city corrections applied (#165), exactly as decided

| City / subfolder | Correction |
|---|---|
| New York, Boston, Cambridge (whole city) | `* (1200/3937)` — US survey feet -> metres, isotropic |
| Philadelphia `2010_ph_downtown/` only | same factor; `2015_scene/` untouched (already metric) |
| Mississauga | `pyproj` reproject X/Y from EPSG:3857 (Web Mercator, what the raw coordinates actually match) to EPSG:26917 (NAD83 UTM 17N, the city's own stated convention); Z untouched (already real metres) |
| Yarra `richmond/` only | negate Y **and** reverse each face's winding (`faces[:, ::-1]`) — the code-review-corrected version of #165's fix: negating Y alone is a mirror transform that would ingest an inside-out mesh |
| Toronto | excluded entirely (independently poor mesh quality per #158, not worth reprojecting) |
| every other city | left as-is (#157: already metric-correct) |

Corrections run in `apply_geometric_correction`, BEFORE the 90 m extent filter and BEFORE
`building_to_sdf` — required by #165's own text ("the 90-unit extent cut must be applied in
correctly-scaled metres per city, not raw file units"). Verified directly:
`test_feet_city_extent_filter_applies_to_corrected_units` builds a 300-US-survey-foot (91.4 m)
Boston box and confirms it is rejected on the 90 m cutoff only because the correction ran first.

## Watertightness gate (#166), reusing the profiler's own classifier

`classify_defect` calls `profile_buildingworld_meshes.boundary_defect` directly rather than
reimplementing the floor_open/scattered/mixed/non_boundary_defect taxonomy — the profiling pass and
the real ingest must never silently diverge on what a defect class means.
`KEEP_DEFECT_CLASSES = {watertight, floor_open, non_boundary_defect}`; `scattered`/`mixed` are
rejected. `defect_class` is stamped per kept row (see above).

## Duplicate exclusion (#160)

`load_dedup_excluded_ids` reads `execution/artifacts/buildingworld_duplicate_gate.json` and excludes
any Tokyo/Berlin candidate id #160's gate flagged. At the recorded snapshot both match lists are
empty, so this is a live, tested mechanism with no current effect — `main()` refuses to ingest
Tokyo or Berlin at all if the artifact is missing, rather than silently skipping the check. #160's
own doc already flags this as a snapshot result, not a standing guarantee: re-run
`dedup_buildingworld_geometric.py` before a production ingest if BuildingWorld's local files, or the
existing PLATEAU/NRW rows, have changed since the artifact was generated.

## Sampling (#168)

No cap, no reweighting, applied literally: `stage_city` takes every candidate that survives the
gates above, for every requested city, with no per-city or total limit in the code path itself.
`--limit` exists only as a testing/smoke-run convenience, never a corpus-shaping default.

## Z-up sanity assertion (#157 cross-check, not a smoke test)

`z_reference_check` cross-checks a per-city sample of (corrected) mesh z-origins against #157's own
documented reference elevation, in three modes:

- **`asl`** (13 cities) — median sampled z_min must fall within #157's stated elevation band
  ±40 m. A real ValueError, not a warning: this is meant to catch a wrong CRS/scale slipping through,
  which a "heights look plausible" smoke test (what the ticket explicitly asks to replace) would
  not.
- **`relative_zero`** (Melbourne) — #157 found z_min is always exactly 0 (a per-building
  height-above-base, not elevation); the check asserts the sample is still near zero (±2 m), which
  would catch Melbourne's export format changing.
- **`unreliable`** (Adelaide, Perth) / **`undocumented`** (San Francisco, Wellington) — #157 either
  found these cities' z internally inconsistent (Adelaide ranges from −92 m to +41 m depending on the
  specific sub-structure file) or never established a figure at all. Gating on either would be false
  confidence, not a real check — logged, never raised.

Bands (`CITY_Z_REFERENCE`) are transcribed directly from #157's own per-city numbers/prose, padded
generously (±40 m) for real building-to-building variation the audit's 20-file sample would not have
captured. This is a judgement call disclosed here, not asserted as precise: a tighter or different
band is a reasonable future refinement, not a correctness bug in this ticket.

The extent-cutoff acceptance test #165 pre-registered ("`frac_extent_gt_90` should drop to a range
comparable with already-metric-correct cities... after correction") is reported as a natural
byproduct (`n_extent_gt_90`/`population` in each city's staging summary) rather than hard-gated —
it's a distributional sanity check to review per city, not a single pass/fail threshold. #165's other
two acceptance items (a cross-city roof-pitch distribution comparison via the beam-search fitter, and
a manual satellite-imagery spot-check) are **not** implemented in this script: the pitch comparison
duplicates #159's already-existing pilot machinery and the imagery check is inherently a human task.
Both are disclosed here as deferred follow-up, not silently dropped.

## What was actually run this session

- The full unit-test suite (`test_ingest_buildingworld.py`, 35 tests) against synthetic meshes/h5
  files, covering every correction, the watertightness classifier, the z-reference check in all four
  modes, the dedup-exclusion mechanism, `IncrementalCityWriter`'s write/flush/resume behaviour, and
  `combine_into_real`'s frozen-prefix preservation, tamper rejection, source-file non-mutation, and
  handling of a city that stages zero rows.
- A real smoke run of the **staging** stage against actual local BuildingWorld data (`--limit 30
  --res 16`, 8 cities spanning every correction path: Berlin, Tokyo, Adelaide, Yarra, Boston,
  Mississauga, Philadelphia, Melbourne) — confirms the whole per-mesh pipeline (zip -> trimesh load
  -> correction -> defect classification -> extent filter -> `building_to_sdf` -> row construction)
  works against the real files, with real per-city kept/skip counts consistent with #166's own
  sampled pass-fraction table (e.g. Philadelphia's small sample kept 0/30, unsurprising against its
  documented ~18% real yield).
- A real smoke run of the **combine** stage was started against production `real.h5` (reading it,
  writing only to a throwaway `/tmp` output — the production file was never touched) and stopped
  early once it was confirmed to be legitimately I/O-bound (block-copying the existing ~32 GB of SDF
  volumes on a shared cluster filesystem, not a bug) rather than run to completion, which would have
  taken on the order of an hour for zero additional correctness signal beyond what the unit tests and
  the staging smoke run already established. **A full production ingest across all 18 cities with no
  cap (#168) was not run in this session** — the candidate pool is roughly two orders of magnitude
  larger than the existing corpus (Berlin alone: ~523k candidates) and combine's own cost scales with
  the existing corpus size on every run; this is real, multi-hour-to-multi-day compute that should be
  a deliberate, monitored operation, not something to launch unattended as part of writing the
  script.

## The production run (2026-09-13/14)

Run for real across all 18 non-Toronto cities, no cap, `--workers 28` (a fork-pool parallelization
added after the initial smoke-testing above showed the single-threaded rate would take roughly a day;
`stage_city`'s `workers`/`pool_batch_size` parameters and `IncrementalCityWriter`'s resumability exist
because of this run, not speculatively).

**Mid-run incident, real and disclosed rather than smoothed over**: about 2 hours in (partway through
Berlin, the largest city), the kernel OOM-killed the main process --
`dmesg`: `Memory cgroup out of memory: Killed process ... anon-rss:238894500kB` against this job's
250 GB cgroup. Root cause: `multiprocessing.Pool.imap_unordered` buffers every completed result in
the main process regardless of consumer speed; submitting a whole city's tasks (Berlin: 523,213) to
one call let that buffer grow unbounded when disk writes temporarily fell behind the 28 workers under
Lustre contention from another job sharing the node. Fixed by submitting tasks in bounded batches
(`POOL_BATCH_SIZE = 2000`, each batch fully drained before the next is submitted), which caps
outstanding buffered results regardless of consumer speed -- tested
(`test_stage_city_pool_batches_tasks_rather_than_submitting_all_at_once`) before relaunching. Thanks
to `IncrementalCityWriter`'s per-city resumability, only ~31,600 unflushed Berlin rows were lost to
the crash, not the ~2 hours of prior work. Both run logs are kept under
`data/real_massing_v1/buildingworld_staging/`: `_ingest_run_1_oom_crash.log` (the crashed attempt,
kept as the postmortem record) and `_ingest_run.log` (the full run that completed).

**Result**: `real.h5` grew from 35,776 to **1,562,554 rows** (1,526,778 new BuildingWorld rows,
~1.56 TB). Verified directly via h5py after completion: row 0 (historical) still carries its original
`bag_id`/`height_m` with `source_key`/`defect_class` empty as designed; spot-checked rows across the
new range (Adelaide, Berlin, Boston, Cape Town, Montreal, and the very last row -- a Yarra `richmond/`
mesh that landed as `floor_open`, confirming the winding-fix worked rather than producing a
`scattered` inside-out mesh) all carry correct `source_key`/`source_id=-1`/`defect_class`. The
script's own internal `assert_frozen_corpus` re-verification on the rebuilt file (run before the
atomic replace) passed -- `real.h5`'s fresh mtime is the evidence, since a failure there would have
left the original file untouched.

Per-city kept/seen (`[stage:done]` lines, `_ingest_run.log`; Adelaide's total below includes the
1,104 kept before the crash, which the resumed run's own summary line doesn't repeat since those rows
were already committed):

| City | Kept | Seen | Yield |
|---|---:|---:|---:|
| Adelaide | 1,104 | 4,580 | 24% |
| Berlin | 456,569 | 523,213 | 87% |
| Boston | 114,494 | 167,418 | 68% |
| Calgary | 165,688 | 457,474 | 36% |
| Cambridge | 182 | 17,377 | 1% |
| Cape Town | 243,932 | 274,598 | 89% |
| Edmonton | 281,211 | 370,677 | 76% |
| Greater Geelong | 604 | 886 | 68% |
| Melbourne | 7,742 | 8,934 | 87% |
| Mississauga | 139,063 | 145,081 | 96% |
| Montreal | 54,058 | 60,528 | 89% |
| New York | 20,650 | 47,815 | 43% |
| Perth | 418 | 4,136 | 10% |
| Philadelphia | 276 | 2,938 | 9% |
| San Francisco | 4,248 | 91,360 | 5% |
| Tokyo | 33,315 | 35,577 | 94% |
| Wellington | 530 | 17,368 | 3% |
| Yarra | 2,694 | 26,050 | 10% |

Low-yield cities (Cambridge, Perth, Philadelphia, San Francisco, Wellington, Calgary, Yarra, Adelaide)
are dominated by `defect:scattered` rejections, matching #166's own sampled pass-fraction predictions
for these cities almost exactly. High-yield cities (Berlin, Boston, Cape Town, Edmonton, Mississauga,
Montreal, Tokyo) confirm the CRS corrections and watertightness gate work correctly at full scale, not
just on the smoke sample -- Mississauga's 96% yield in particular is strong evidence the Web-Mercator
-> UTM17N reprojection (#165) is landing extents correctly, since a wrong reprojection would show up
as a spike in `extent` rejections instead.

## Running it for real

```bash
# Stage one or more cities (safe: never touches real.h5; resumable per city)
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/foundations/ingest_buildingworld.py --no_combine --cities Berlin

# Stage everything (all 18 non-Toronto cities), no cap -- the actual #168-decided scope
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/foundations/ingest_buildingworld.py --no_combine

# Once every requested city is staged, rebuild real.h5 (reads the whole existing corpus; expect
# this to dominate wall time -- it prints periodic copy progress)
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/foundations/ingest_buildingworld.py --no_stage --combine
```

## Status: implemented, unit-tested, and run to completion in production (2026-09-13/14)

`real.h5` now holds 1,562,554 rows (35,776 historical + 1,526,778 new BuildingWorld rows). #175
(extract isosurfaces for the new rows and register the source in the surfaces pipeline) is unblocked.
