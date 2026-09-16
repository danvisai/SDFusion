# #183 - Run the BuildingWorld retrain ladder: noise control, corpus fold-in, drop-region control

*Effort: fold BuildingWorld into the massing corpus (arm six on the gable bar), map
[#156](https://github.com/danvisai/SDFusion/issues/156). Child of #156, decided by #172 (ladder
mechanics) and #171 (region-bucket granularity). Blocked by #178 (CLOSED
2026-09-15 - see `178-buildingworld-baseline-signoff.md`).*

> Execute the retrain ladder decided in #172: full retrain from scratch for every arm, no
> warm-start. Arm 1 (noise control) first, pre-registering a noise-band tolerance before looking
> at arm 2. Arms 2-3 fold BuildingWorld in (with, then without, #171's style-bucket region
> channel), each judged against both the noise band and #170's pre-registered floors. Stop early
> on any KILL.

## Arm 1 - retrain-noise control: DONE, noise band pre-registered

Same recipe as the currently-served `heightmap_ce.pt` (`--objective ce --epochs 40`, median
decode), same pre-BuildingWorld corpus (`--corpus_scope legacy`, the exact frozen 35,623-row cache
`heightmap_ce.pt` itself trained on), only the seed changed (0 -> 1) and the tag
(`heightmap_ce_noise_control_seed1`). No BuildingWorld data touched. Command:

```
python scripts/foundations/train_height_map_generator.py \
  --objective ce --tag heightmap_ce_noise_control_seed1 --seed 1 --epochs 40 \
  --median_decode --montage 0 --corpus_scope legacy \
  --out execution/artifacts/183_arm1_noise_control.json
```

Result: `execution/artifacts/183_arm1_noise_control.json`. The frozen control's own numbers are
read from the existing `execution/artifacts/height_map_generator_714.json` (`heightmap_ce_median`
arm), never re-scored - same convention #178's `frozen_control()` already established.

### Pre-registered noise band (2026-09-15, before arm 2 was run or looked at)

Absolute delta, seed1 minus the frozen (seed0) control, on the carve-needing population (n=411 -
the population the pinned-714 headline numbers and `PROGRAM_BAR` are set on):

| metric | frozen (seed0) | seed1 | delta | noise-band tolerance |
|---|---|---|---|---|
| `extra` | 0.0603 | 0.0685 | +0.0082 | ±0.0082 |
| `missing` | 0.0385 | 0.0335 | -0.0051 | ±0.0051 |
| `vs_input` | 0.8432 | 0.8452 | +0.0020 | ±0.0020 |
| `collapse_rate` | 0.0268 | 0.0243 | -0.0024 | ±0.0024 |
| `vol_iou` | 0.8948 | 0.8940 | -0.0008 | ±0.0008 |
| `dl_planar_fraction` | 0.2000 | 0.2000 | 0.0000 | ±0.0000 |
| `dl_ops` (median) | 6.0 | 5.0 | -1.0 | ±1.0 |

On the full pinned-714 population (n=714, reported for completeness, not the primary gate):
`extra` +0.0025, `missing` -0.0023, `vs_input` +0.0041, `collapse_rate` -0.0014, `vol_iou`
+0.0017, `dl_ops`/`dl_planar_fraction` unchanged.

**Regression-guard rule for arms 2-3 (per #172(c)):** an arm's carve-needing score on each metric
above must fall within `frozen_control_value ± noise_band_tolerance`. This is a single-run
estimate, not a distribution - #183's own text calls for exactly one noise-control run, not a
multi-seed sweep - so treat the tolerance as a floor on what "no real regression" can mean, not a
tight confidence interval. `dl_ops`'s ±1.0 band is real signal, not slack padding: it is a median
over 411 integer-valued per-building counts, and a single-seed swing from 6 to 5 shows that
statistic sits close enough to a boundary that ordinary retrain noise alone can flip it by a full
unit. A future arm landing exactly on this boundary should be read with that in mind, not treated
as a precise verdict.

Arm 1 is a control, not judged against #170's floors (informational verdict only, shown in its own
run output: `heightmap_ce_noise_control_seed1_median` reads `PASS` against the historical
`PROGRAM_BAR`, `heightmap_ce_noise_control_seed1` (argmax decode) reads `NOT MET` - expected, since
the served arm is specifically the median-decode one).

## #171 landed: BuildingWorld style-bucket region scheme

#171's decision (broad style buckets, not per-city, new ids rather than merged into NL/DE/JP) had
no concrete bucket boundaries on record. Implemented in `source_provenance.py`
(`BUILDINGWORLD_CITY_BUCKET`, `REGION_MAPPING_VERSION` bumped to v2) - 6 buckets grouped by
country/continent and sized against each bucket's actual row count so no bucket is a thin single
city wearing a "diversity" label: Germany (Berlin, 456,569), Japan (Tokyo, 33,315), South Africa
(Cape Town, 243,932), Canada (640,020 across 4 cities), USA (139,850 across 5 cities), Oceania
(13,092 across 6 AU/NZ cities). Full reasoning and the "why new ids, not merged" argument (#160's
dedup gate already found BuildingWorld Tokyo/Berlin geographically distinct from PLATEAU/NRW) is
in that module's own comments - see commit `90867e4`.

`scripts/foundations/assign_buildingworld_region_buckets.py` backfilled `corpus_ledger.h5`'s
`region` column for all 1,526,778 BuildingWorld rows from #174's `-1` sentinel to their real
bucket id. Run against production; verified the 35,623 legacy rows read back bit-identical and no
row is left at the sentinel.

## Trainer wiring for a 9-region channel (commit `0ec14fc`)

`N_REGIONS`/`COND_CHANNELS` were fixed module-level constants (3 regions, NL/DE/JP). Threaded an
`n_regions` parameter through `conditioning_channel_names`, `validate_region_ids`,
`condition_channels`, `cache_provenance`, `validate_checkpoint_provenance`'s no-cache fallback, and
`make_model`, every one defaulting to the legacy `N_REGIONS=3` so no existing caller's behavior
changed (confirmed: full existing suite, `town_generate_service.py` import, unaffected).
`build_cache` now stamps `n_regions` into the cache dict itself (`N_REGIONS` for `"legacy"`,
`N_REGIONS_ALL=9` for `"all"`), so it flows downstream automatically without every caller needing
to know which scope produced its cache.

Verified against the real (post-backfill) ledger, without paying for a full corpus_scope="all"
cache build (which reads ~1.56M raw SDF volumes and was not run this session): both `"legacy"`
(35,623 rows, region range [0, 2]) and `"all"` (1,562,401 rows, region range [0, 8]) pass
`validate_region_ids` cleanly. The region-validation landmine that would have made an unscoped
`corpus_scope="all"` run fail immediately (BuildingWorld rows carrying an out-of-range sentinel) is
gone; the remaining cost of an actual arm 2 run is wall-clock (~1.56M SDF reads to build the cache
once, then training), not a correctness blocker.

## Known gap, disclosed and deliberately not fixed here

`scripts/server/town_generate_service.py`'s own model-construction and `condition_channels` call
sites still assume the legacy 3-region default and do not read `n_regions` off a served
checkpoint's own claim. Harmless today (only legacy checkpoints are served), but a landmine for
whoever eventually wires a BuildingWorld-trained checkpoint into the live demo - same character as
#155's `fit_decode`, which was shipped as an arm with its town-service integration explicitly left
as separate, not-yet-done work. Not part of #183's own scope.

## Arm 2 - corpus + style-bucket fold-in: DONE, KILL on the BuildingWorld gate (2026-09-16)

Full retrain from scratch, `--corpus_scope all` (9-region channel, BuildingWorld folded in),
otherwise the exact same recipe as arm 1/the served checkpoint (`--objective ce --epochs 40
--median_decode --seed 0`). `build_cache(corpus_scope="all")` ran first (1,562,401 rows, 18.5 min
parallelized - see the `build_cache` parallelization entry below). Training itself: 40 epochs,
~5.93h (21,341s), after fixing a real CPU/GPU overlap bug mid-run (see below) that would otherwise
have made this an ~11-13h run. Checkpoint: `outputs/height_map_generator/heightmap_ce_bw_bucket.pt`.

**Gate 1 - pinned-714 regression guard (carve-needing n=411), vs arm 1's noise band:**

| metric | frozen | arm 2 | delta | noise band | within? |
|---|---|---|---|---|---|
| extra | 0.0603 | 0.0665 | +0.0062 | ±0.0082 | yes |
| missing | 0.0385 | 0.0325 | -0.0060 | ±0.0051 | no (improved) |
| vs_input | 0.8432 | 0.8433 | +0.0001 | ±0.0020 | yes |
| collapse_rate | 0.0268 | 0.0243 | -0.0024 | ±0.0024 | at boundary (improved) |
| vol_iou | 0.8948 | 0.8961 | +0.0012 | ±0.0008 | no (improved) |
| dl_planar_fraction | 0.2000 | 0.2000 | 0.0000 | ±0.0000 | yes |
| dl_ops (median) | 6.0 | 6.0 | 0.0 | ±1.0 | yes |

Every metric that nominally exceeds its noise-band tolerance does so in the IMPROVING direction
(lower missing/collapse, higher vol_iou) - the regression guard's purpose is to catch a metric
getting WORSE by more than ordinary retrain noise can explain, and nothing did. **Gate 1: clears.**
BuildingWorld fold-in did not hurt (and mildly helped) the legacy pinned-714 population.

**Gate 2 - #178's signed-off BuildingWorld bar, scored via the new
`scripts/foundations/score_buildingworld_checkpoint.py` (24,464 held-out buildings across the 9
named cities): every one of the 11 floors reads `pass=False`. numeric_pass=False.**

The failure is concentrated and legible, not scattered noise: `clear_kill` (the KILL clause,
`dl_planar_fraction > kill_planar`) fails on **every single floor**, because arm 2's measured
`dl_planar_fraction` is 0.000 almost everywhere (`gable_hip` alone reaches 0.250, still under its
0.261 kill line). This is not simply "the population is mostly flat, so zero is correct" - the
1-NN baseline itself (a non-generative nearest-footprint copy, no learning at all) already
measured nonzero planar fraction on Berlin (0.20), Edmonton (0.125), and gable_hip (0.261);
arm 2's trained model comes in at 0.00, 0.00, and 0.25 respectively on those same three
populations - **underperforming a copy-paste baseline specifically on roof form.** `extra` and
`collapse` clear their bars comfortably almost everywhere (`beats_extra`/`collapse` mostly True) -
the arm is not destroying volume or over-carving; it is producing safe, close-to-flat surfaces
that never commit to a pitched roof. This is the exact "volume/safety PASS, form KILL" pattern
already on record for this objective on the LEGACY population (`1-map-status-and-parked-work.md`:
"no trained arm clears the 0.40 planar bar while also holding extra/collapse") - now also
confirmed, with a hard KILL rather than a soft NOT MET, on BuildingWorld's own population and
`#170`'s stricter three-floor bar. BuildingWorld fold-in did not cause this; it did not fix it
either. Full per-floor numbers: `execution/artifacts/183_arm2_buildingworld_gate.json`.

**Verdict: arm 2 is a KILL.** Per #172(b), "a KILL verdict on any arm stops the ladder rather than
running the remaining arms regardless." **Arm 3 (drop-region control) is not run.**

## Incidental fix found and fixed mid-arm-2: train()'s CPU/GPU overlap

`train()`'s hot loop called `HeightFieldSet.batch()` (CPU-bound: `_d4` augmentation, then
`condition_channels`'s per-row `distance_transform_edt`) synchronously, immediately followed by
GPU forward/backward - invisible at the legacy 35,776-row scale, but a live run at BuildingWorld's
scale measured 52% GPU utilization, confirmed by reading the code: no overlap at all. Fixed with
`prefetch_iter()`, a single-background-thread producer (deliberately not a multi-process
`DataLoader`, which would have let each worker's own forked, independently-advancing copy of
`HeightFieldSet`'s per-call `self.rng` diverge from what a sequential caller produces - silently
changing which augmentation draws a batch gets, not just wall-clock). Measured: 1204s/epoch -> 540s
(2.23x), GPU utilization 52% -> 96%. Arm 2's full run used the fixed version throughout.

## What's left

1. **Arm 3 will not run** - the ladder's own pre-registered stopping rule (#172(b)) ends it here.
   A future revisit needs a new decision, not a continuation: is the form problem addressed before
   trying a drop-region control on top of it, or is a drop-region arm still informative on its own?
2. #173's footprint-shape channel was already out of scope for this ladder (per #183's own text)
   and remains so.
3. The form problem itself (#1's "not yet specified" list already named this the owner's next
   focus after #154, independent of BuildingWorld) is now confirmed on a second, larger, more
   diverse population - not a new problem, but no longer only a legacy-corpus observation either.

## Status: arm 1 complete; arm 2 complete and KILLed on the BuildingWorld gate; arm 3 not run per the ladder's stopping rule; ladder concluded (2026-09-16)
