# Authentic A2-to-real supervision without identity or leakage bias

**Decision for [Design authentic A2-to-real supervision without identity or leakage bias](https://github.com/danvisai/SDFusion/issues/115).**

## Answer

Freeze one already-accepted A2 operating point, materialize one reproducible authentic output per
building, and train against real occupancy. The first corpus is **384 training buildings**, not 384
arbitrary rows: 326 rows that require envelope-to-real correction (163 BAG/NL, 163 NRW/DE) and 58
explicit identity controls (19 BAG, 19 NRW, 20 PLATEAU). Screen once on a fixed, region-balanced 96
held-out rows; only a passing, frozen editor checkpoint may be evaluated on the fixed 714-building
population.

The learned result is compared with a **footprint-sanitized A2 baseline**, not raw A2. Deterministic
removal of A2 spill therefore earns no model credit. Identity and correction opportunities are
reported separately, and the final report includes the sealed 618-row complement of the screening
set beside the compatibility result on all 714 rows.

This is authentic supervision because the editor input is the actual field decoded from the A2
projection it will receive at inference. Issue #91's paired cache is not this corpus: it pairs encoded
footprint-envelope tokens with real tokens for A2 training; it does not contain A2 outputs
([aligned-retrain contract](../../../scripts/foundations/run_aligned_retrain.py)).

## Why this policy is necessary

The repository has already demonstrated each attribution hazard:

- The shipped A2 result at its accepted operating point is worse than its own envelope: on all 714
  held-out buildings, median 3D IoU is 0.8756 for A2 versus 0.9334 for the envelope, with A2 median
  `vs_input=0.9846` ([full artifact](../../../execution/artifacts/massing_arms_eval_ship714.json)). An
  editor must improve A2 without being credited for returning or deterministically clipping that
  strong input.
- Exactly 291/714 held-out envelopes equal the target. All 210 PLATEAU held-out rows are identity
  rows, versus 10/232 BAG and 71/272 NRW. PLATEAU was ingested at LoD1, and the training code records
  that its envelope-to-real pair has exactly zero target change
  ([dataset contract](../../../scripts/train_vecset.py#L67-L89)). Sampling the corpus in its natural
  proportions would therefore reward `KEEP` for the wrong reason.
- Ascending corpus row order tracks source region. The old first-48 prefix was all Dutch; the shared
  evaluator now interleaves regions and records IDs precisely because region changes height, envelope
  overfill, and collapse behaviour ([ID-selection implementation](../../../scripts/foundations/eval_massing_arms.py#L523-L565)).
- Per-building sampling is mandatory. The evaluator reduced seeded median-IoU drift to about 0.001,
  while unseeded median IoU moved by 0.027
  ([seed policy and measured noise floor](../../../scripts/foundations/eval_massing_arms.py#L600-L609)).
  Dora's surface sampler is stateful, so the codec must also be reseeded per row; otherwise a row's
  envelope latent depends on every row encoded before it
  ([codec contract](../../../models/shape_codec.py#L94-L102)).
- Strength is a cliff on the shipped checkpoint: 0.5 is often a near-no-op while 0.7 can collapse a
  rectangular envelope. Strength therefore cannot be chosen after observing editor performance
  ([recorded strength measurement](../latent-token-order/93-strength-band.md)).

## Frozen A2 source distribution

The corpus generator has no checkpoint or sampler sweep.

| item | frozen value |
|---|---|
| A2 weights | `weights/massing-vecset/vecset_v4_surf.pth` |
| checkpoint identity | step 240,000; SHA-256 `643aed0896e2edc36ab3ecb073da63847881dc2ba95459eed896a19b65fed04d` |
| projection | strength `0.5`, 20 DDIM-style steps, guidance `1.0` |
| task | footprint plus **specified normalized vertical extent** |
| master seed | `0` |
| row seed | `master_seed * 1_000_003 + global_real_h5_row`, therefore the canonical seed is the row ID |
| samples | exactly one A2 sample per building; no retry, rejection, average, or best-of-*k* |

The checkpoint is the shipped A2 checkpoint already measured on 714 rows, so its selection predates
the voxel editor. The currently running issue-#92 arms are not eligible. If a later A2 checkpoint is
accepted, it creates `authentic-a2-v2`: regenerate every source and rerun the experiment. Never replace
the checkpoint inside `v1` or choose whichever A2 checkpoint makes the editor look best.

For every row, perform this exact source construction:

1. Build the signed-EDT footprint envelope at the specified `(y0, y1)`.
2. Call `DoraCodec.reseed(row_seed)` immediately before encoding that envelope.
3. Call `SetSDEdit.project(..., seed=row_seed)` with the frozen operating point.
4. Decode once to a float32 `64^3` field and derive `raw_occ = field32 <= 0` **before** any float16
   storage conversion.

This extends the evaluator's row-keyed diffusion seed to the stateful Dora encoder. The projection
itself is deterministic after its supplied noise draw
([projection implementation](../../../models/networks/vecset_projection.py#L78-L116)). It also means
subset and full-population replay produce the same source for a row regardless of loop order.

The existing benchmark constructs `(y0, y1)` from the target and calls the task “footprint + given
height” ([harness](../../../scripts/foundations/eval_massing_arms.py#L630-L715)). Metric `height_m`
alone is not equivalent: buildings are normalized per instance, and voxel extent correlates with
metric height by only 0.43 ([height diagnosis](../../../scripts/foundations/probe_height_inference.py#L1-L16)).
For leakage clarity, this corpus names and stores `(y0, y1)` as an **allowed conditioning field**.
No other value derived from target geometry may enter the A2 source, editor input, mask, constraint,
or postprocessor. Results must not be described as a footprint-only task.

## Fixed row manifests

Rows are global `real.h5` IDs, never positions in another cache. Selection version is
`whole-volume-authentic-supervision-v1`; within each cohort, order rows by
`SHA256("whole-volume-authentic-supervision-v1:" + decimal_row)`.

### Training 384

- `held_out == 0` only.
- Opportunity means the specified footprint envelope is **not byte-identical** to `real.sdf <= 0`.
- Take the first 163 opportunity rows from region 0 and first 163 from region 1.
- Identity means envelope occupancy is byte-identical to target occupancy. Take the first 19 identity
  rows from region 0, 19 from region 1, and 20 from region 2.
- Sort the resulting 384 rows by the same salted hash. PLATEAU contributes preservation examples, not
  a 32.7% vote for `KEEP`.

Canonical compact-JSON list SHA-256:
`b8e1cf84bc7f030a269d74021abe43886ad353cf3a90542ddba2f8be7cb537bb`.

### Screening 96 and full 714

The full population is the 714 non-degenerate held-out global rows recorded by the accepted full
artifact. Sort each region by the salted hash and round-robin regions. Screening uses the first 32
rows from each region (96 total), selected without reading A2 or editor outcomes. It contains 54
envelope-opportunity and 42 envelope-identity rows (1 BAG, 9 NRW, 32 PLATEAU), reflecting rather than
hiding the benchmark's identity mass.

Canonical compact-JSON list SHA-256 values:

- screening 96: `dde3583874333651169ad3fae0a5a7fdcdedff1302bd91b2ad0e5151f03c103c`
- full 714: `9584cf66b61be665995c3f028ee8e1590147bfb3969f3dc1867e308a4dcdac94`
- canonical manifest `{version,salt,train384,val96,full714}`:
  `9e697321d31e0b378d381575a4e483c0dba2ee93f0d8735124745a3ea911bcca`

The cache builder must materialize the lists in a manifest and refuse to run if their digests differ.
It must also record the selected target-occupancy digest; the list digest alone cannot detect an HDF5
file being replaced in place.

## Cache contract

One HDF5 row carries:

| field | purpose |
|---|---|
| `row`, `cohort`, `region` | stable identity and declared sampling stratum |
| `height_m`, `extent_y0`, `extent_y1` | explicit allowed height/extent conditions |
| `footprint` | packed conditioning mask |
| `a2_field_f16` | clipped/normalized read-only continuous A2 evidence; never recompute occupancy from it |
| `a2_occ` | packed exact occupancy from float32 A2 field before quantization |
| `sanitized_a2_occ` | packed deterministic baseline `a2_occ & footprint[:,None,:]` |
| `target_occ` | packed `real.sdf[row] <= 0`; labels/scoring only |
| `envelope_occ` | packed specified-height footprint envelope; control and stratum only |

File attributes record the A2 checkpoint path, step and SHA-256; repository revision; Dora config and
checkpoint identity; resolution, axes and sign convention; strength/steps/guidance; seed formula;
manifest digest; and per-dataset content digests. The source field is stored at float16 only as neural
conditioning. `a2_occ` preserves the exact float32 zero crossing, so a storage conversion cannot
silently change the categorical source state.

Do not cache a mesh and voxelize it. A2 already materializes the continuous field before meshing; its
native occupancy is `field <= 0` ([town generation path](../../../scripts/server/town_generate_service.py#L220-L247)).

## Sanitized control and attributable improvement

Let `C(x) = x & footprint[:,None,:]`. Both the learned arm and its baseline start from `C(raw A2)`.
The headline paired delta is:

`IoU(editor(C(A2)), target) - IoU(C(A2), target)`.

Always publish raw A2 separately so the cost/benefit of `C` remains visible. The editor receives raw
A2 occupancy, A2 field, footprint and declared conditions, but hard footprint containment is applied
identically to its result and the baseline. It cannot claim deterministic spill removal as learning.

`C` is intentionally minimal and target-independent. Ground connection, solidity, minimum thickness
and connectivity are validity gates, not hidden geometry-repair heuristics. If a later design adds a
deterministic repair `R`, the comparison must become `editor(R(C(A2)))` versus `R(C(A2))` on the same
rows, and it is a new named corpus/version.

## Opportunity and identity scorecard

Report the population result and these predeclared strata; never rebalance the held-out aggregate:

- **E0 envelope identity:** `envelope_occ == target_occ`.
- **E1/E2/E3 envelope opportunity:** normalized envelope correction
  `|envelope XOR target| / |target|` in `(0, .05]`, `(.05, .15]`, and `(.15, +inf)`.
- **A0/A1/A2/A3 authentic-source error:**
  `|sanitized_A2 XOR target| / |target|` equal to `0`, in `(0, .05]`, `(.05, .15]`, and
  `(.15, +inf)`.

The E strata expose the benchmark's inherent identity target; the A strata expose what the editor
actually receives. Publish their cross-tab by region. Train loss may balance `KEEP/ADD/REMOVE` actions,
but held-out metrics are never action-balanced or identity-weighted. A result passes only by the
predeclared population gate **and** does not degrade E0; a gain confined to E0 correction or one region
is reported as such, not generalized to the task.

## Leakage and replay rules

1. The trainer opens only the training-384 cache. Target arrays from screening/full caches are not
   importable by the training process.
2. Architecture, loss, optimizer, step count, thresholding and all validity gates are frozen before
   screening. Screening 96 is one go/no-go run, not a hyperparameter-validation loop.
3. Freeze and hash the editor checkpoint before generating or opening full-714 targets. The compatible
   headline remains all 714, but also publish the **sealed 618-row complement** because proceeding was
   conditional on the 96-row screen.
4. Replay uses manifest IDs, row-keyed encoder/noise seeds and the cached source-content digest. Any
   missing, duplicate, cross-split or mismatched row is fatal.
5. No model may receive `target_occ`, `envelope_occ`, identity/opportunity label, source-error stratum,
   or a target-derived edit mask. They exist only for sampling training rows and for reporting.
6. No rejection sampling, collapsed-source replacement, retry, best-of-*k*, checkpoint cherry-pick,
   strength sweep, or per-row level-set tuning is allowed. Failures remain in the corpus.
7. The 96 and 714 stages reuse byte-identical A2 sources for their shared IDs. Full replay is not a
   new stochastic draw.

## Scope and recorded-decision boundary

This corpus tests whole-volume occupancy as an experimental **massing transform** above `s*`; it does
not make a voxel mask the canonical building. Treating it as the final state would conflict with the
symbolic recipe and constrained-volume vocabulary in [`CONTEXT.md`](../../../CONTEXT.md), while
from-noise voxel generation would conflict with the accepted “transform, not generate” claim
([ADR 0003](../../adr/0003-two-claim-thesis.md)). The corpus decision neither resolves nor bypasses
that later recipe-compatibility decision.

