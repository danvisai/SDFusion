# #188 - Retraining the A2 massing source on the new corpus, region-free

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

That distinction governs everything below. **#188 is a substitution, not a new quality claim** -
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
the corpus rather than the module default - which is what stops this defect recurring for a
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
an artifact rather than improvised at a call site. The encode is GPU-bound on the real pass - there
is nothing to overlap - but the blockout pass spends 192 ms per row on CPU with the A100 idle,
which `prefetch()` now overlaps.

### 3. Cohort composition: capped-proportional

Chosen by the owner from three options, 2026-09-21. The corpus is severely imbalanced - Canada
640,020 rows against Oceania 13,092, a 49× spread.

- *strictly proportional* - Canada + Germany take ~72% of the draw, Oceania lands under 1%.
- *equal per bucket* - a distribution no real city has, and self-defeating under region-free
  conditioning: the model cannot tell buckets apart, so equalising only reweights the geometry.
- **capped-proportional** - proportional in the middle, clipped at both ends.

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
row id - never source, geometry, difficulty, or any measured outcome. Held-out rows (#177's split)
are structurally ineligible. Hash *ordering* rather than shuffling also makes a larger quota a
superset of a smaller one, so a later top-up extends the encode instead of invalidating ~8
GPU-hours of it.

⚠️ **The legacy 35,623 rows are not sampled.** Their latent and blockout caches already exist and
enter the retrain whole, so they remain the regression anchor. The trainer reads several caches at
once rather than materialising a merged ~34 GB copy that could drift from either source.

### 4. The recipe is the frozen source's own, with one disclosed departure

`vecset_v4_surf` is not a single run - it is a three-stage lineage, read off the checkpoints
themselves:

```
vecset_v2_pair        0 →  60k   surf_weight 0.0
vecset_v3_pair_long  60k → 180k  surf_weight 0.0   (resumed)
vecset_v4_surf      180k → 240k  surf_weight 1.0   (resumed)
```

So "retrain from scratch" is **180,000 steps with the surface term off, then 60,000 with it on**,
batch 8, lr 1e-4, width 512, depth 8, heads 8, `pair_frac` 0.8, `pair_t_min` 0.35, `cfg_drop` 0.1.

⚠️ **One departure, disclosed rather than hidden.** `vecset_v4_surf` ran before `--surf_t_center`
existed, so its surface term used the lowest-t selection that #80 later measured as a mistake - it
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

Structure and vocabulary are `CONTEXT.md`'s own (PASS / GUARD / KILL), not a parallel scheme:
"**PASS** is what the arm must achieve. **GUARD** is what it may not break on the way (collapse and
`vs_input`). **KILL** is a pre-registered clause that answers the ticket 'no'."

| id | kind | clause | resolves to |
|---|---|---|---|
| K1 | KILL | `generation_failures == 0` | every scored row generated; derived from the run, not asserted |
| K2 | KILL | `buckets_scored == 6` | all six style buckets generated |
| G1 | GUARD | `vs_input < 0.98` | `CONTEXT.md`'s documented guard, at its documented value |
| G2 | GUARD | `collapse_rate <= 0.1582` | `CONTEXT.md`'s documented bar (1-NN retrieval's rate) |
| P0 | PASS | `fp_iou >= 0.90` | frozen source's own legacy median is 0.9589, p10 0.8803 |
| B1 | PASS | `vol_iou / blockout >= 0.9240` | `vol_iou >= 0.81102` |
| B2 | PASS | `extra / blockout <= 1.2269` | `extra <= 0.17091` |
| B3 | PASS | `missing <= 0.05` | absolute; the blockout's `missing` is 0, so a ratio is undefined |
| I1 | INFORMATIONAL | `beats_envelope_rate >= 0.0056` | reported, never scored |

**B1/B2 are non-inferiority normalised by population.** The retrained source is scored on
BuildingWorld rows and the frozen one on legacy rows, so raw numbers are not comparable - the
populations differ in difficulty. The ratio form asks: *relative to doing nothing on its own
population, is the replacement at least as good as the frozen source was relative to doing nothing
on its population?* Both thresholds sit on the wrong side of 1.0 precisely because the source being
replaced loses to its own envelope.

⚠️ **B1/B2 inherit a small inconsistency from the recipe departure above.** Their thresholds come
from `vecset_v4_surf`'s measured numbers, and §4's `surf_t_center` correction means the retrain does
not reproduce that checkpoint's recipe exactly. The reference is still the right one - it is the
source actually being replaced - but "non-inferiority to the frozen source" is therefore
non-inferiority to a checkpoint trained with a defect this run does not repeat, which if anything
makes the comparison harder for the retrain rather than easier.

A KILL is not outvoted by the clauses that passed (#183's rule, per #172(b)). A broken GUARD means
the arm is not servable whatever else it scored. I1 is informational for the reason set out above:
requiring the replacement to beat an envelope the sealed source loses to would make this a
different ticket.

### ⚖️ Revision, 2026-09-21, before any training step ran

*Recorded rather than quietly applied, because a pre-registration that can be edited is not one.*

The first committed version of this registration (`c5789a1`) **could be passed by doing nothing.**
Scoring the committed `blockout` reference arm against it: `fp_iou` 1.0 ≥ 0.90, `collapse_rate` 0.0
≤ 0.25, and both ratio clauses exactly 1.0 against themselves - every scored clause PASS. The only
no-op detector, `vs_input`, had been demoted to INFORMATIONAL and loosened to 0.99.

Two documented standards had been silently overridden in the process, which
`docs/agents/domain.md` specifically forbids ("Surface any conflict ... explicitly instead of
silently overriding the recorded decision"):

- `CONTEXT.md`: "`vs_input` ... **1.0 means it did nothing.** An arm at 0.99 has not been measured
  as a generator however good its other numbers look (#75). **The guard is < 0.98.**"
- `CONTEXT.md`: "`collapse_rate` ... the bar is **1-NN retrieval's 0.1582**: a generator that
  destroys more buildings than naive retrieval is not servable whatever else it scores."

Both are now scored GUARD clauses at their documented values. The revision **tightens** the bar in
every direction and was made **before the first training step**, with the original version intact
in git history at `c5789a1`; no result influenced it. Checked afterwards, and worth recording: the
frozen source clears both guards on its own population (`collapse_rate` 0.098 on the pinned 714;
`vs_input` 0.9616 at n=12), so adopting them does not import a bar the predecessor fails.

Verified: a pure no-op now scores **GUARD BROKEN**, failing G1 alone while passing all eight other
clauses - which is exactly the shape of the hole that was there.

### Operating point, fixed in advance

`strength 0.5, steps 20, guidance 1.0` - the point the frozen source was scored at, and the one
#119 will seal alongside the replacement. #115's method forbids sweeping, so this is not tuned.

## Result: GUARD BROKEN (2026-09-22)

The retrain ran to its registered 240,000 steps and was scored **once**, on the 900-row held-out
gate population, at the pre-registered operating point. Artifacts:
`execution/artifacts/massing_arms_eval_188_candidate_step240000.json` and
`execution/artifacts/188_verdict.json`.

| id | kind | clause | measured | |
|---|---|---|---:|---|
| K1 | KILL | `generation_failures == 0` | 0 | PASS |
| K2 | KILL | `buckets_scored == 6` | 6 | PASS |
| G1 | GUARD | `vs_input < 0.98` | 0.1350 | PASS |
| G2 | GUARD | `collapse_rate <= 0.1582` | **0.8178** | **FAIL** |
| P0 | PASS | `fp_iou >= 0.90` | **0.7658** | **FAIL** |
| B1 | PASS | `vol_iou / blockout >= 0.9240` | **0.1268** | **FAIL** |
| B2 | PASS | `extra / blockout <= 1.2269` | 1.0586 | PASS |
| B3 | PASS | `missing <= 0.05` | **0.8655** | **FAIL** |
| I1 | INFORMATIONAL | `beats_envelope_rate >= 0.0056` | 0.0067 | reported only |

**Both KILL clauses passed, and that is the one thing #188 did achieve.** All 900 BuildingWorld
rows generated, across all six style buckets, with zero failures. The `IndexError` that made the
frozen source structurally unusable on this corpus is gone. The capability exists; the quality does
not.

`B2` passing is an artifact rather than a result: an arm cannot leave much surplus behind when it
has removed 87% of the building.

### The failure is bimodal, not uniformly bad

| percentile | `missing` |
|---|---:|
| p10 | 0.009 |
| p25 | 0.447 |
| median | 0.866 |
| p75 | 0.936 |
| p90 | 0.961 |

130 of 900 rows (14.4%) come back essentially intact (`missing` < 0.05); 736 (81.8%) are collapsed.
The model either reproduces a building or demolishes it, and rarely lands in between. Whatever is
wrong is close to binary per row, not a uniform degradation.

### What it produces: hollow shells, and volume in the wrong place

Rendered three ways, all committed under `outputs/massing_arms_eval/`:

- `montage_188_candidate_step240000.png` - the scored arm beside gt / blockout / codec_ceiling.
- `voxel_188_PREVIEW_step220000.png` - **the informative one.** True voxel occupancy beside the
  same volume collapsed through a height-map lens.

The voxel render shows the mechanism the scalars only imply: the arm produces **hollow shells** -
walls and partial floors around an empty interior. On one row it reaches **0.995x GT volume while
missing 77% of it**: close to exactly the right quantity of material, in the wrong places.

⚠️ **A height-map lens cannot represent this failure.** A height map is one solid run per column by
construction, so collapsing this output to a height field fills every cavity back in and renders a
plausible solid building. This is worth recording beyond #188: any future arm in this family judged
through a height-field or skyline statistic would score this checkpoint as acceptable.

### Diagnosis: the surface term is an accelerant, not the cause

Two controls, same 12 ids, same harness:

| run | fp_iou | missing | extra | vol_iou | collapse | vs_input |
|---|---:|---:|---:|---:|---:|---:|
| **control**: frozen source on LEGACY rows | 0.9580 | 0.0016 | 0.1146 | 0.8975 | 0.1667 | 0.9819 |
| phase 1 (surf OFF, step 180k), BuildingWorld | 0.9403 | 0.0758 | 0.1629 | 0.7151 | 0.4167 | 0.8765 |
| phase 2 (surf ON, step 220k), BuildingWorld | 0.7694 | 0.7416 | 0.1623 | 0.2249 | 0.8333 | 0.2773 |

**The harness is sound.** The control reproduces the frozen source's own recorded behaviour
(`fp_iou` 0.9580 against its 0.9589 at n=714; `collapse_rate` 0.1667 matching its recorded n=12
exactly), so the region-free changes to `eval_massing_arms.py` did not break the measurement.

**Phase 1 already failed the collapse guard** (0.4167 against 0.1582) before the decoded-surface
term was ever switched on. Phase 2 then multiplied `missing` roughly tenfold, which matches #60's
recorded finding that this term diverges into rubble through the `1/sqrt(alpha_bar)` amplification
of eps-error. So removing the surface term would buy a cheaper failure, not a fix - the cause
predates it.

### Correction to this document's own earlier claim

An earlier revision of the clause rationale stated that the frozen source "measures 0.9616 (n=12),
so it clears" the `vs_input < 0.98` guard. The control run above measures **0.9819 on a different
12 ids, which fails it**. The frozen source sits on that line and lands either side depending on
the sample; #87's own per-region medians (0.9882 / 0.9832 / 0.9813) are above it. The threshold is
`CONTEXT.md`'s and is unchanged - only the justifying sentence was wrong, and it is corrected here
rather than silently edited in place.

### Consequences

- **Not sealed.** `seal_a2_source.py` refuses a non-PASS verdict without a recorded override, and
  there is no case for one. No digest is handed to #119.
- **#119 remains blocked**, and with it #120, #121 and #124 - the chain below the source.
- **#115 and #187 are unaffected** and remain workable.
- The open question this hands back to the map is whether #113's source must be a vecset A2
  checkpoint at all, or whether the route is re-chartered onto another generator. That is an owner
  decision, not one this ticket can settle.

## Status

- Code, cohort, gate population, pre-registered bar: **done and committed**.
- Cohort encode (60,000 rows, real + blockout): **done**, 7 h 44 m, zero skipped rows.
- Retrain, 180k steps surface-off then 60k on: **done**, 2026-09-21 13:13 to 2026-09-22 17:13 UTC.
- Scored once against the pre-registered bar: **done** - **GUARD BROKEN**.
- Freeze / hash / seal: **deliberately not done**; the bar did not accept the checkpoint.
