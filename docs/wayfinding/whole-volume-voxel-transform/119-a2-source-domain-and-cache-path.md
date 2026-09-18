# The A2 source's domain, and the state of the cache path

*2026-09-18. Work on [Cache authentic A2 outputs for the whole-volume feasibility
decision](https://github.com/danvisai/SDFusion/issues/119) for map [Whole-volume voxel
transformation of A2 massing](https://github.com/danvisai/SDFusion/issues/113). The census is CPU
only. The cache-path smoke used the GPU in an isolated window (A100 idle, no A2 training run
present); it loaded the frozen checkpoint read-only and wrote no checkpoint and no cache.*

## Finding

**The frozen A2 source cannot generate for the corpus this effort has moved to, and the corpus it
can generate for cannot answer the effort's question.** Those two facts together block #119 as
written; they are not a cohort-design problem that better stratification could solve.

#119 is therefore left open and blocked on [Retrain the A2 massing source on the new corpus so
BuildingWorld rows can be generated](https://github.com/danvisai/SDFusion/issues/188), per owner
direction on 2026-09-18. No cache artifact was written: a cache sealed against a source about to be
replaced would be dead on arrival.

## 1. The frozen source's domain is exactly the legacy population

[#115](https://github.com/danvisai/SDFusion/issues/115) sealed `vecset_v4_surf` @240k (SHA-256
`643aed0896e2edc36ab3ecb073da63847881dc2ba95459eed896a19b65fed04d`) as the one permitted source.
That checkpoint conditions through a three-entry region embedding:

```
A2 region embedding: (3, 512)
  region id 2: accepted
  region id 3: IndexError: index out of range in self
  region id 8: IndexError: index out of range in self
```

Executed against the real checkpoint, not read off the source. `VecsetDenoiser.__init__` declares
`n_regions: int = 3` and `prototype_voxel_editor.cache_command` constructs the net without
overriding it. An exhaustive sweep of all 75 vecset checkpoints under `logs_building/` and
`transfer/` read `region.weight` from every one: 73 at `(3, 512)` and 2 at `(3, 256)`. **None has
more than three regions**, so this is a property of the model family here, not of one checkpoint.

The corpus ledger's region column places every BuildingWorld row outside that range:

| region ids | source | rows |
|---|---|---:|
| 0, 1, 2 | legacy `bag3d` / `nrw` / `plateau` | 35,623 |
| 3-8 | BuildingWorld style buckets (`BUILDINGWORLD_CITY_BUCKET`) | 1,526,778 |

```
{0: 11773, 1: 11850, 2: 12000,
 3: 456569, 4: 33315, 5: 243932, 6: 640020, 7: 139850, 8: 13092}
```

So "authentic A2 output" is only a definable thing for the 35,623 legacy rows. For BuildingWorld
rows the frozen generator does not produce a worse source; it produces none.

Two workarounds exist and both were rejected as the basis for a sealed corpus. Substituting a
legacy region id for a BuildingWorld row conditions the generator on a label the row does not
carry. Passing `region=None` is structurally supported by `VecsetDenoiser.forward` (footprint and
height still condition), but the model only ever saw region-present or all-conditioning-dropped
during training, since `cfg_drop=0.1` drops the whole condition jointly; the combination is
untrained. The owner's direction is to retrain region-free rather than to generate from an
untrained mode of a checkpoint that was sealed for a different population.

## 2. The legacy population cannot carry the question

[The BuildingWorld void/passage verification](119-buildingworld-void-passage-verification.md)
established that BuildingWorld's below-roofline gaps are predominantly exterior-reachable
architecture rather than sealed defect cavities, and concluded the decision cohort should come from
BuildingWorld. That left open whether the legacy population could serve instead. It cannot, and the
margin is not close.

[`census_legacy_void_signal.py`](../../../scripts/foundations/census_legacy_void_signal.py)
censuses **every** row the frozen A2 can condition on, rather than sampling, reusing the committed
`hollow_shell_voxels()` reachability contract and the same below-roofline gap mask as the
BuildingWorld verification.

| region | rows | any gap | any reachable void | structural (>= s\*<sup>3</sup> = 27 vox) | structural and held out |
|---|---:|---:|---:|---:|---:|
| BAG / NL | 11,773 | 150 | 123 | 41 | 0 |
| NRW / DE | 11,850 | 153 | 148 | 80 | 2 |
| PLATEAU / JP | 12,000 | 4 | 4 | **0** | 0 |
| **total** | **35,623** | **307** | **275 (0.77%)** | **121 (0.34%)** | **2** |

The decisive number is the last column. The screen and full gates draw from the held-out
population, and it contains **two** structural-void rows in total. A 96-row screen would be
expected to contain none; the 714-row gate likewise. Whatever the stratification of the training
cohort, the capability could not be *measured* on held-out data, which is what the gate exists to
do.

Sealed cavities are negligible in this population: 72 sealed voxels against 43,326 reachable, and
32 sealed-only rows. That is consistent with the BuildingWorld result rather than in tension with
it. Legacy PLATEAU is the sharpest case: 12,000 Tokyo rows, four of which have any gap at all,
totalling 31 reachable voxels, and not one reaching structural scale.

Row-stream digest over all 35,623 censused rows:
`c7dfc481eece69e3d610a4300a30da078f9e654da3e06143abc05971e8da85aa`. The artifact
[`119-legacy-void-census.json`](119-legacy-void-census.json) commits that digest, the per-region
summary, the full list of 121 structural row ids, and a per-row record for each of the 307
gap-carrying rows.

## 3. The cache path had never run, and crashed on its first line of model setup

`cache_command` was listed as not-done in
[the #125 plumbing record](125-absolute-occupancy-plumbing.md): "untested beyond unit coverage of
its pure helpers ... the full Dora/A2-decode path itself has not been run." Running it surfaced a
defect that no unit test could have caught, because the failure is in the part that needs a real
checkpoint:

```python
mu, sd = ck["latent_mu"].to(device), ck["latent_sd"].to(device)
# AttributeError: 'float' object has no attribute 'to'
```

`train_vecset.py` persists `ds.mu`/`ds.sd` verbatim, so what a checkpoint stores depends on the
dataset that produced it. The frozen A2 source stores Python floats (`latent_mu = -0.0243…`,
`latent_sd = 0.8388…`), so this line raised before a single row could be generated. The repository's
other call sites already normalise with `torch.as_tensor`.

Fixed by `normalize_latent_stats()` in `prototype_voxel_editor.py`, which accepts floats, NumPy
scalars, or tensors and fails loudly on a checkpoint missing either statistic. Covered by
`TestLatentStatsNormalisation` in `test_prototype_voxel_editor.py`.

## 4. With that fixed, the path works and caching is cheap

Six train-role rows generated end to end through the real Dora encode, SDEdit projection, and
`decode_grid` path, with the committed reseed formula (`master_seed * 1000003 + row`) applied to
both the envelope encoding and the projection noise.

| measure | value |
|---|---|
| model load (A2 + Dora, 191.6M params) | 11 s, once |
| steady-state generation | **~1.4 s/row** |
| projected 384-row train role | ~9 min |
| projected 96-row screen role | ~2 min |
| projected 714-row full role | ~17 min |
| cache record | 1.57 MB/row uncompressed (`source_field` f16 + four uint8 volumes + footprint) |

Those projections used the retired 384/96/714 design. Against the cohort sizes #118 actually
settled, the same per-row cost lands very differently:

| role (per #118) | rows | cache time | cache size |
|---|---:|---:|---:|
| screen (`GATE_SCREEN_N`) | 250 | ~6 min | ~0.4 GB |
| confirmation (`GATE_CONFIRMATION_N`, #178's held-out set) | 24,464 | **~9.5 GPU-hours** | **~38 GB** |
| training bank | full bank, sized in #115 | scales at ~1.4 s and 1.57 MB per row | |

The confirmation population is the binding constraint on scheduling, not the screen.

Sanitized-A2 against real occupancy IoU on those six rows: 0.88, 0.78, 0.66, 0.70, 0.68, 0.83.
Reported only as evidence the decode path produces plausible geometry, not as a result: six
non-randomly-chosen rows measure nothing.

This also discharges one of [#118](https://github.com/danvisai/SDFusion/issues/118)'s
still-to-do items, "the A2 timing probe, before the training cohort is final". The probe ran
against the old source, but per-row cost is dominated by the Dora encode, the 20-step projection,
and the `decode_grid` call, none of which the retrain changes, so the figure transfers to whatever
checkpoint #188 produces.

## Consequence

Map #113 was re-chartered onto the BuildingWorld corpus on 2026-09-18, at owner direction, and
these findings are part of why. The resulting ticket state:

- **#119** stays open, unclaimed, rewritten for this corpus, and blocked by
  [#115](https://github.com/danvisai/SDFusion/issues/115) (cohort composition) and
  [#188](https://github.com/danvisai/SDFusion/issues/188) (the retrained source).
- **#115 is reopened and rewritten.** Its prior resolution was a cohort decision about a population
  this effort no longer uses. Its *method* survives verbatim: one source per global row, row-keyed
  reseeding of both envelope encoding and SDEdit noise, no sweeping, no retries, no best-of-*k*,
  outcome-blind screening, editor sealed before the confirmation population.
- **#121's gate population changes** from the retired pinned 714 to #178's 24,464 held-out
  BuildingWorld rows, which is what makes the ~9.5 GPU-hour figure above the number that matters.
- **#125 is banner-marked stale on the corpus.** Its plumbing, representation, and provenance
  discipline survive; its 384/96/714 numbers and three-region design do not.
- The pinned 714 remains a **named legacy regression control**, the role
  [the BuildingWorld verification](119-buildingworld-void-passage-verification.md) already assigned
  it and the one #118's regression guard uses. No gate verdict is scored on it.

## Reproduction

Census (CPU only, ~2 min, reads `real.h5` and `vecset_latents.h5`, writes one JSON):

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/foundations/census_legacy_void_signal.py \
  --out docs/wayfinding/whole-volume-voxel-transform/119-legacy-void-census.json
```

Synthetic controls for the census, including a solid box, an exterior-reachable passage, a sealed
cavity, a mixed volume, and the centered-frame base inference:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  -m unittest scripts.foundations.test_census_legacy_void_signal
```

The cache-path smoke was a throwaway measurement rather than a committed script: it replicates
`cache_command`'s inner loop over the first few manifest rows without writing a cache. Its two
durable outputs are the `normalize_latent_stats` fix and the timings in section 4. Re-running
`cache --role train` against a manifest is the real path, and is now unblocked on the code side.
