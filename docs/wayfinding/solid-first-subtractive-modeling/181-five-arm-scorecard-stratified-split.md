# #181 — Run the five-arm autonomous-generation scorecard against the stratified split

*Effort: solid-first semantic architectural carving. Opened 2026-09-06, built 2026-09-10. Part of
[#1](https://github.com/danvisai/SDFusion/issues/1). Implements the H1a evaluation protocol
[#8](8-falsifiable-proof.md) decided, against the split
[#153](153-stratified-split.md) produces.*

> Using #153's implemented region/tile-stratified split, re-run the standard massing-eval harness
> (`missing`/`extra`/`vs_input`/collapse/`dl_ops`/`dl_planar_fraction`) across exactly five arms:
> `blockout`, `1-NN` retrieval, the raw #127/#155 height-map generator, `fit_decode`, and
> `fit_decode` + #9's block-coordination bias. Report all five side by side, never collapsing the
> raw-generator and `fit_decode` rows into one number. PASS/PARTIAL/KILL for H1a is decided per
> axis: volume/safety (does the arm clear 1-NN's `extra` on the carve-needing subset) and
> architectural form (does `dl_planar_fraction` show real form).

Code `scripts/foundations/five_arm_scorecard.py` (orchestration: split materialization, overlap
audit, sampling, curated report) + a minimal, additive `--bank_exclude_ids` flag on
`scripts/foundations/train_height_map_generator.py` (the actual scoring pipeline, reused
unmodified otherwise). Tests `scripts/foundations/test_five_arm_scorecard.py` (21 cases) +
`TestBankEligibility` in `test_train_height_map_generator.py` (4 cases). Artifacts
`execution/artifacts/height_map_generator_153_test.json` (the full harness output, every arm) and
`execution/artifacts/five_arm_scorecard_153.json` (the curated five-arm report this doc reads).
Run: **1,000** buildings sampled from #153's test split, **512 carve-needing**, checkpoint
`weights/massing-heightmap/heightmap_ce.pt` (existing, no new training).

---

## What "the standard massing-eval harness" actually is

#181's own text names `missing`/`extra`/`vs_input`/collapse/`dl_ops`/`dl_planar_fraction` — this is
**not** `eval_massing_arms.py` (a different, closed A2/vecset comparison with none of these
metrics), it is `scripts/foundations/train_height_map_generator.py`, which despite its filename
contains #127/#129/#132/#138/#139/#155's entire scoring CLI. It already builds a `1-NN` arm
(`retrieve_nn`/`transplant_height`, footprint-IoU nearest-neighbor, hyperparameter-free) and a
`fit_decode` arm (`fit_decode`, wrapping the generator's raw output through #10's beam fitter) end
to end, and already accepts a `FitBias` for the fifth arm via its existing
`--fit_decode_roof_family` flag — none of this needed reimplementing.

## Materializing #153's split

`stratified_split.make_split` deliberately returns an in-memory label array, not a persisted id
list (see that module's own docstring: dozens of already-closed tickets are pinned to the legacy
population, and retroactively touching it was explicitly out of scope for #153). `materialize_split`
runs it fresh (seed 0, the published default) and writes the ids the harness needs. Of #153's 7,120
raw test rows, **7,079** are present in the height-field cache (`vecset_latents.h5` fails to encode
153 of real.h5's 35,776 rows entirely, an unrelated, pre-existing gap).

**The retrieval bank is rebuilt to exclude #153's own held-out rows** — `--bank_exclude_ids`,
9,459 rows (test ∪ val), a minimal additive flag with no effect when omitted (4 dedicated tests).
The bank is a non-parametric lookup rebuilt fresh every run, so this is not retraining; it is what
makes the `1-NN` arm's number in this report a genuine held-out measurement rather than one where
"the old bank may retrieve the test row itself," per this ticket's own reconciliation note.

## 🔑🔑 The train/test overlap audit — most of this is retrospective, not held-out

Per this ticket's own reconciliation note, existing checkpoints were trained under the OLD split;
measured directly rather than trusted from the cited figures:

| | overlap | of 7,079 |
|---|---|---|
| in the legacy Bag3d **training** split (`perm[2·n_val:]`, seed 0) | **6,798** | **96.0%** |
| in the height-map generator's own OLD training-eligible pool (`held==0`) | **6,937** | **98.0%** |

(The reconciliation note's own cited figures — 6,837 and 6,937 on 7,120 raw rows — match almost
exactly once the 41-row cache gap is accounted for; the second figure is identical.)

⚠️ **This means the scorecard below is a genuine held-out measurement for `blockout` (no learning
at all) and `1-NN` (bank rebuilt above), but a *retrospective* measurement for the raw generator and
both `fit_decode` arms** — ~96-98% of the rows they are scored on were almost certainly in the
checkpoint's own training pool. This is disclosed per the reconciliation note's own instruction
("audit train/test overlap... disclose retrospective evaluation if needed. This is not
authorization to violate the ticket's no-new-training constraint") and is exactly why no new
checkpoint was trained here — doing so would either violate that constraint or produce a *fresh*
number nobody asked for while still not answering what #153's split alone can determine about an
*existing* checkpoint. No per-checkpoint training manifest exists to check row-level (not just
pool-level) membership; see the module docstring for why that is a real, disclosed ceiling on this
audit, not an oversight.

## Result — n = 512 carve-needing, median `extra` with a bootstrap CI

| arm | extra | 95% CI | missing | collapse | dl_ops | planar |
|---|---|---|---|---|---|---|
| blockout | 0.2236 | [0.2112, 0.2391] | 0.0000 | 0.0000 | 0.0 | 0.00 |
| 1-NN retrieval | 0.1112 | [0.0993, 0.1305] | 0.0310 | **0.1719** | 2.0 | 0.25 |
| raw height-map generator (#127/#155) | **0.0551** | [0.0459, 0.0617] | 0.0323 | 0.0430 | 6.0 | **0.20** |
| fit_decode (#155) | 0.0724 | [0.0652, 0.0796] | 0.0205 | **0.0234** | 3.0 | **0.00** |
| fit_decode + #9 coordination bias | 0.0722 | [0.0652, 0.0796] | 0.0207 | 0.0234 | 3.0 | 0.00 |

**Oracle-fed / ceiling-only, never part of the pass/fail comparison** — quoted from
[130-baselines-diffusion-curriculum.md](130-baselines-diffusion-curriculum.md), on the **legacy
pinned 411** carve-needing rows, not #153, not re-run:

| named baseline | extra | missing | collapse | dl_ops | planar |
|---|---|---|---|---|---|
| ceiling (program recovery, sees GT) | 0.0035 | 0.0000 | 0.0000 | 2.0 | 0.50 |
| ArcPro/CoMa (`flatten_ramps`, oracle-fed) | 0.0528 | 0.0000 | 0.0024 | 1.0 | 0.00 |

n = 512 is well above this package's own undersampled threshold (300); not flagged. The bootstrap
resamples 512 distinct carve-needing buildings (one row each, not repeated conditions per building
the way #179/#180's own pair-level bootstraps were) — the one residual caveat is that #153's own
tile-stratification exists precisely because nearby buildings correlate structurally, so the 512
draws are not perfectly independent of each other either; read the CI as descriptive under that
caveat, not as a guarantee over fully i.i.d. draws.

## Per-axis verdict, mechanically (`train_height_map_generator.verdict`, unmodified)

| arm | volume/safety (beats 1-NN `extra`) | architectural form (`planar ≥ 0.40`) |
|---|---|---|
| raw height-map generator (#127/#155) | **PASS** (0.0551 < 0.1112) | **KILL** (0.20 < 0.40) |
| fit_decode (#155) | **PASS** (0.0724 < 0.1112) | **KILL** (0.00 < 0.40) |
| fit_decode + #9 coordination bias | **PASS** (0.0722 < 0.1112) | **KILL** (0.00 < 0.40) |

## This replicates #155's own finding almost exactly, on a different, mostly-overlapping population

#155's own closing numbers (legacy population): raw median `extra` 0.0603/collapse 0.0268/planar
0.20, `fit_decode` `extra` 0.0807/collapse 0.0049/planar 0.00 — "safer, worse form." Here: raw
`extra` 0.0551/collapse 0.0430/planar 0.20, `fit_decode` `extra` 0.0724/collapse 0.0234/planar
0.00 — **the same qualitative pattern in every column**: `fit_decode` trades some `extra` for lower
collapse and loses essentially all its architectural form (planar 0.20 → 0.00). Given ~97% of this
population overlaps the checkpoint's own training pool (above), this is best read as a **replication
of the same trade-off's direction and rough shape**, not independent confirmation that it holds
under genuine generalization — that question needs either a freshly-trained checkpoint under #153's
split (out of scope: "no new training") or #153's small non-overlapping remainder scored alone
(disclosed as a follow-on, not attempted here).

⚠️ **The coordination bias (`fit_decode + #9`) makes almost no measurable difference from plain
`fit_decode`** (extra 0.0722 vs 0.0724, planar 0.00 vs 0.00) — expected, not a bug: `roof_family`
biasing TOWARD `"flat"` has nothing to move when `fit_decode`'s own output is already
overwhelmingly flat (planar already at 0.00), the same "nothing to bite into" finding [#180](180-multi-footprint-coordination-real-footprints.md)
made independently against its own flat placeholder massing.

## Verdict, per #8's own per-axis shape

- **H1a / volume-safety: SUPPORTED.** Both the raw generator and both `fit_decode` arms clear
  1-NN's `extra` on the carve-needing subset (n=512), comfortably outside each other's CI.
- **H1a / architectural form: NOT SUPPORTED (KILL) for all three trained arms**, `fit_decode`
  more severely (planar 0.00) than the raw generator (0.20) — exactly #155's own already-known
  mound-shaped-not-roof-shaped negative, reported honestly rather than as a new failure.
- **Per #8's own rule, these two axes disagree, so H1a reads PARTIALLY SUPPORTED**, not a single
  collapsed word — volume/safety clears its bar, architectural form does not, on this split exactly
  as on the legacy one.
- **Scope caveat, stated plainly:** because ~97% of #153's test population overlaps the checkpoint's
  own OLD training pool, this PARTIALLY SUPPORTED reading is a **retrospective replication**, not
  fresh held-out generalization evidence, for the two axes that depend on the trained checkpoint.
  The `1-NN` reference point (bank genuinely rebuilt) and `blockout` are the only arms here that are
  unaffected by the overlap.

## Scope and disclosed limitations

- 1,000 of 7,079 available test buildings were scored (512 carve-needing), not the full population
  — `fit_decode`'s beam search and the per-arm description-length fit both cost real per-building
  time, and scoring all ~7,000 was a multi-hour run; disclosed as a deliberate, seeded (181),
  reproducible sampling choice rather than a silently shrunk population.
- No new checkpoint was trained, per the ticket's own constraint — the overlap audit above is the
  honest cost of that choice, not hidden.
- #130's oracle numbers are quoted from the legacy 411-row population, not re-run against #153 —
  a deliberate choice (they are ceiling references, never part of the pass/fail comparison; #181
  doesn't require re-deriving them, only including them clearly labeled).
- The "block-coordination bias" arm applies one representative bias (`roof_family="flat"`)
  uniformly across the whole population, not to real, adjacency-grouped multi-footprint blocks —
  #153's split carries no adjacency structure to group by (the same gap
  [#180](180-multi-footprint-coordination-real-footprints.md) exists specifically to source real
  block scenes for); H1a's own question is aggregate generation quality under a coordination-shaped
  bias, which #180 already covers the genuine multi-footprint-block validity question for
  separately.

## What follows

- A genuinely fresh held-out reading of the raw generator / `fit_decode` arms needs either a
  checkpoint trained under #153's split (its own future ticket, not this one) or scoring only
  #153's small non-overlapping remainder (~2-4% of the test set) in isolation — not attempted here,
  named as open.
- H1a's scorecard row (volume/safety SUPPORTED, architectural form NOT SUPPORTED, PARTIALLY
  SUPPORTED overall, scope-caveated as retrospective) is ready for #8's own cross-hypothesis
  rollup alongside H1b ([#179](179-guided-edit-completion-proxy.md), FALSIFIED as implemented) and
  H3 ([#180](180-multi-footprint-coordination-real-footprints.md), SUPPORTED) — all three of #8's
  own spawned tickets are now closed.
