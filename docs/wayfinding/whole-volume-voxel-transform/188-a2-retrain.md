# #188 — Retraining the A2 massing source on the new corpus, region-free

*Work on [Retrain the A2 massing source on the new corpus so BuildingWorld rows can be
generated](https://github.com/danvisai/SDFusion/issues/188), for map [Whole-volume voxel
transformation of A2 massing](https://github.com/danvisai/SDFusion/issues/113). This is the record
of the decisions, the pre-registered bar, and the run. Owner direction of 2026-09-18 (region-free)
is followed as given.*

## Why the sealed source had to be replaced

Established in [`119-a2-source-domain-and-cache-path.md`](119-a2-source-domain-and-cache-path.md)
and not re-derived here: the frozen source `vecset_v4_surf` @240k (SHA-256 `643aed08…`) conditions
through `region = nn.Embedding(3, 512)`, and every BuildingWorld row carries region id 3–8. The
failure is `IndexError`, not a quality gap.

That distinction governs everything below. **#188 is a substitution, not a new quality claim** —
#115's amendment says the method carries over unchanged and "only the identity of the frozen
checkpoint changes". The bar is set accordingly (see *The pre-registered bar*).

## Four decisions this ticket had to make

### 1. Region-free, and what that means structurally

Owner direction. Implemented as `n_regions=0`, which does **not** build the embedding at all, so
the checkpoint structurally carries no `region.weight` and raises if a caller hands it region ids.

That is deliberately stronger than passing `region=None` to a 3-region model. #119 rejected exactly
that combination: `cfg_drop=0.1` drops the whole condition jointly, so "footprint and height but no
region" is an *untrained mode* of the frozen checkpoint, not a supported one. Removing the
parameter removes the ambiguity.

The named alternative if this underperforms is #171's 9-bucket style channel, and #188's own text
requires it be run as its own single-variable arm, never bundled. `--region_free` and `--regions`
are therefore mutually exclusive at the parser, and the trainer now derives the region width from
the corpus rather than the module default — which is what stops this defect recurring for a
region-conditioned arm.

### 2. `corpus_scope="all"` cannot mean "every row" for this family

The height-map ladder's `corpus_scope="all"` reads all 1,562,401 ledger rows because its cache is a
cheap SDF read (18.5 min parallelized, #183). The vecset family cannot copy that.

**Measured on this box, not estimated:**

| step | cost |
|---|---|
| one real-surface vecset latent | **207 ms** (2.9 ms CPU sampling, 200 ms Dora forward) |
| one blockout latent | 415 ms serial (141 ms `real.h5` read, 51 ms extrude+mesh, 222 ms encode) |
| all 1,526,778 BuildingWorld rows, real pass only | **~88 GPU-hours, ~412 GB** |

So the fold-in is a **sample**, defined in `scripts/foundations/vecset_cohort.py` and committed as
an artifact rather than improvised at a call site. The encode is GPU-bound on the real pass — there
is nothing to overlap — but the blockout pass spends 192 ms per row on CPU with the A100 idle,
which `prefetch()` now overlaps.

### 3. Cohort composition: capped-proportional

Chosen by the owner from three options, 2026-09-21. The corpus is severely imbalanced — Canada
640,020 rows against Oceania 13,092, a 49× spread.

- *strictly proportional* — Canada + Germany take ~72% of the draw, Oceania lands under 1%.
- *equal per bucket* — a distribution no real city has, and self-defeating under region-free
  conditioning: the model cannot tell buckets apart, so equalising only reweights the geometry.
- **capped-proportional** — proportional in the middle, clipped at both ends.

`cap_frac=0.25`, `floor_frac=0.05`, water-filled. ⚠️ **These fractions are #188's own local call,
not settled policy.** [#168](https://github.com/danvisai/SDFusion/issues/168) (sampling cap /
corpus-balance) is open and undecided; the cohort is reproducible from its salt and quota alone if
it settles otherwise.

The draw (`execution/artifacts/188_cohort.json`, train digest `9295ac59e5a5a569…`):

| region | bucket | trainable rows | drawn | share of draw | share of bucket |
|---|---|---:|---:|---:|---:|
| 3 | Germany | 447,428 | 15,000 | 25.0% | 3.35% |
| 4 | Japan | 32,547 | 3,000 | 5.0% | 9.22% |
| 5 | South Africa | 239,025 | 15,000 | 25.0% | 6.28% |
| 6 | Canada | 627,137 | 15,000 | 25.0% | 2.39% |
| 7 | USA | 137,046 | 9,000 | 15.0% | 6.57% |
| 8 | Oceania | 12,816 | 3,000 | 5.0% | 23.41% |
| | **total** | **1,495,999** | **60,000** | | |

Selection is **outcome-blind**, per #115's surviving method: the only input is a salted hash of the
row id — never source, geometry, difficulty, or any measured outcome. Held-out rows (#177's split)
are structurally ineligible. Hash *ordering* rather than shuffling also makes a larger quota a
superset of a smaller one, so a later top-up extends the encode instead of invalidating ~8
GPU-hours of it.

⚠️ **The legacy 35,623 rows are not sampled.** Their latent and blockout caches already exist and
enter the retrain whole, so they remain the regression anchor. The trainer reads several caches at
once rather than materialising a merged ~34 GB copy that could drift from either source.

### 4. The recipe is the frozen source's own, with one disclosed departure

`vecset_v4_surf` is not a single run — it is a three-stage lineage, read off the checkpoints
themselves:

```
vecset_v2_pair        0 →  60k   surf_weight 0.0
vecset_v3_pair_long  60k → 180k  surf_weight 0.0   (resumed)
vecset_v4_surf      180k → 240k  surf_weight 1.0   (resumed)
```

So "retrain from scratch" is **180,000 steps with the surface term off, then 60,000 with it on**,
batch 8, lr 1e-4, width 512, depth 8, heads 8, `pair_frac` 0.8, `pair_t_min` 0.35, `cfg_drop` 0.1.

⚠️ **One departure, disclosed rather than hidden.** `vecset_v4_surf` ran before `--surf_t_center`
existed, so its surface term used the lowest-t selection that #80 later measured as a mistake — it
fed the term only near-clean latents and taught the model to reproduce its input (vs-input 0.993).
The current default (0.55) is the corrected behaviour. Replicating the old value exactly would
reproduce a known defect on purpose, so #188 uses the fixed default and records that this is *not*
byte-identical to the frozen source's recipe.

## The pre-registered bar

Written to `execution/artifacts/188_preregistration.json` **before the training run started**, from
reference numbers measured on the gate population before any retrained checkpoint existed. #183 is
the precedent #188's own text names: its arms 2 and 3 both failed #178's bar, and the lesson is to
state the bar in advance rather than discover it afterwards.

### What the bar deliberately does not ask

**The frozen source does not beat its own footprint envelope**, measured on the pinned 714
(`criterion2_full714.json`, strength 0.5):

| arm | fp_iou | missing | extra | vol_iou |
|---|---|---|---|---|
| blockout (doing nothing) | 1.0000 | 0.0000 | 0.0714 | 0.9334 |
| frozen A2 @240k | 0.9589 | 0.0026 | 0.0876 | 0.8625 |

It wins on `extra` for 11.3% of rows and on `vol_iou` for **0.56%**. A bar demanding the retrain
beat the envelope would be one the sealed source itself fails by a wide margin, and would quietly
convert #188 from "replace a structurally unusable checkpoint" into "solve the open quality problem
map #1 records as unsolved under *safety vs form*". Those are different tickets, and this one does
not pretend to be the other.

### Reference posts, measured on the gate population before training

900 held-out BuildingWorld rows, 6 buckets, gate digest `47af806a908eb0f5…`:

| arm | n | fp_iou | missing | extra | vol_iou | collapse |
|---|---:|---:|---:|---:|---:|---:|
| gt | 900 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| blockout | 900 | 1.000 | 0.000 | **0.139** | **0.878** | 0.000 |
| codec_ceiling | 900 | 0.998 | 0.000 | 0.001 | 0.999 | 0.009 |

Note BuildingWorld's envelope is a worse fit than legacy's (`extra` 0.139 against 0.0714), i.e.
there is more for a generator to carve here.

### The clauses

| id | kind | clause | resolves to |
|---|---|---|---|
| K1 | KILL | `generation_failures == 0` | every scored row generated |
| K2 | KILL | `buckets_scored == 6` | all six style buckets generated |
| K3 | KILL | `fp_iou >= 0.90` | (frozen source's own legacy median is 0.9589, p10 0.8803) |
| K4 | KILL | `collapse_rate <= 0.25` | between the frozen source's 0.1667 and #92's failed 0.4636 |
| B1 | BAR | `vol_iou / blockout >= 0.9240` | `vol_iou >= 0.8113` |
| B2 | BAR | `extra / blockout <= 1.2269` | `extra <= 0.1705` |
| B3 | BAR | `missing <= 0.05` | absolute; the blockout's `missing` is 0, so a ratio is undefined |
| I1 | INFORMATIONAL | `vs_input <= 0.99` | reported, never scored |

**B1/B2 are non-inferiority normalised by population.** The retrained source is scored on
BuildingWorld rows and the frozen one on legacy rows, so raw numbers are not comparable — the
populations differ in difficulty. The ratio form asks: *relative to doing nothing on its own
population, is the replacement at least as good as the frozen source was relative to doing nothing
on its population?* Both thresholds sit on the wrong side of 1.0 precisely because the source being
replaced loses to its own envelope.

A KILL is not outvoted by the clauses that passed (#183's rule, per #172(b)). I1 is informational
because [#119](https://github.com/danvisai/SDFusion/issues/119) ruled no gate verdict is scored on
the legacy pinned 714, and because no arm in this family has ever cleared `vs_input`.

### Operating point, fixed in advance

`strength 0.5, steps 20, guidance 1.0` — the point the frozen source was scored at, and the one
#119 will seal alongside the replacement. #115's method forbids sweeping, so this is not tuned.

## Status

*This section is the run's own record and is updated as it proceeds. Nothing below is a claim about
a result that has not been measured.*

- ✅ Code landed and tested: region-free denoiser, corpus-derived region width, multi-cache trainer,
  cohort selection, row-scoped surface loading, blockout prefetch overlap, bar + verdict scorer.
- ✅ Every checkpoint consumer now reads its region width off the weights
  (`denoiser_from_checkpoint`), including #119's own cache path and the town service — the latter
  being the landmine #183 disclosed and explicitly left for whoever produced a BuildingWorld-trained
  checkpoint. That is this ticket.
- ✅ Cohort drawn and committed; gate population (900 rows) encoded; reference eval run; bar
  pre-registered and committed **before** training.
- ⏳ Training-cohort encode (60,000 rows, real + blockout).
- ⏳ Retrain (180k steps surface-term-off, then 60k on).
- ⏳ Score against the bar; freeze, hash, and hand the digest to #119.
