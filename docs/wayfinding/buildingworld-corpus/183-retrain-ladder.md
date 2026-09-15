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

## What's left

1. **Arm 2 - corpus + style-bucket fold-in.** `--corpus_scope all` (region channel intact, 9-wide),
   full retrain from scratch, BuildingWorld folded in. Needs a real `build_cache(corpus_scope="all")`
   run first (multi-hour SDF read across ~1.56M rows, not yet started) before training can begin.
   Score on the carve-needing pinned-714 AND on #178's signed-off BuildingWorld bar (9 floors, 4 of
   them GUARD-only per the 2026-09-15 sign-off) - both gates must clear.
2. **Arm 3 - drop-region control**, only if arm 2 does not KILL. Same enlarged corpus, region
   channel removed entirely (`shape_channels`/base four channels only, no `region_r` band) - this
   needs its own explicit "no region" path, not yet built (arms 2 and 3 are NOT the same code path
   with a flag flipped on `n_regions`; dropping the channel is a fifth conditioning-channel
   variant, distinct from widening it).
3. #173's footprint-shape channel remains explicitly out of scope for this ladder (per #183's own
   text) - a separate, later, single-variable arm if #173 ever resolves to add one.

## Status: arm 1 complete and pre-registered; arms 2-3 blocked on a corpus_scope="all" cache build, not yet started (2026-09-15)
