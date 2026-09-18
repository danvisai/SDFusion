# #118 - Preregistered success and kill gates for whole-volume voxel correction

*Settled 2026-09-18 by interview (`/grilling`) with the ticket owner, as map #113 requires for its
human decisions. Signed off by the owner. Depends on #117 (validity), #125 (populations and
plumbing), #126 (how massing is scored) and #178 (the BuildingWorld bar and its sign-off
machinery). No experiment has been run; nothing here is a result.*

Every threshold below is either **quoted** from an arm of record under #170's falsification rule,
**computed** from a published statistical design, or **the owner's** explicit call recorded as
such. Where nothing fixes a number, this document says so rather than inventing one.

## Scope

The bar that decides whether the whole-volume A2 voxel corrector stops after its screen, advances
to the fixed confirmation population, or earns a later stochastic arm. It governs the deterministic
correction arm only. It does not authorize production integration, and it does not settle recipe
compatibility.

## Two owner corrections reshaped this bar

Both arrived during the interview, and both are load-bearing.

### 1. The transform adds mass as well as removing it

The prior framing - inherited from #126 and its "carve-needing subset" - treats the job as carving.
That is correct on the legacy pinned 714, where the envelope is a footprint x height extrusion that
**contains GT by construction**, so `missing` can only ever be damage: A2's `missing` is 0.0024
against `extra` 0.0922, an asymmetry of 38 to 1.

On BuildingWorld it does not hold. 1-NN retrieval over the 24,464 held-out rows:

| quantity | value |
|---|---|
| median `missing` | **0.0242** |
| median `extra` | **0.0262** |
| rows under-filled by more than 2% | **52.3%** |
| rows over-filled by more than 2% | 53.0% |
| rows under-filled **more than** over-filled | **44.2%** |
| rows wrong in both directions at once | 19.7% |

Near-symmetric. And the two directions are two different building populations, not two tails of
one:

| roof family | n | `missing` | `extra` | collapse |
|---|---|---|---|---|
| flat | 7,321 | **0.0676** | 0.0000 | **0.2532** |
| gable | 8,017 | 0.0168 | 0.0815 | 0.0730 |
| hip | 437 | 0.0032 | 0.1123 | 0.0526 |
| complex | 8,689 | 0.0210 | 0.0547 | 0.0954 |

On the 6,623 rows where the envelope is already correct - nothing to carve - 1-NN still scores
`missing` 0.0712 and collapses **26.3%** of them. An `extra`-led bar is blind exactly where the
baseline is worst.

**Consequence: `missing` is a PASS axis, not a collapse guard.** The two are never summed.
CONTEXT.md already states why - they are "not symmetric in consequence", surplus is a building that
looks unfinished and `missing` is a building with a trench through it - and #126 spent a ticket
killing the one-number form.

### 2. Interior fill is not architecture

A building may be modelled hollow or solid; from outside they are the same building, and ISO 19107
treats an interior shell as a first-class feature - which is why #117 declines to fail cavities.
A raw volume comparison charges every voxel of air inside a shell target as the candidate's
surplus. That is bookkeeping about air, and it is not rare: #117 measured roughly **4.6% of
BuildingWorld (~70,600 rows)** arriving as 1-3 voxel skins, Calgary about 97% of them.

There is a second, quieter reason. #178's floors are computed by
`train_height_map_generator.height_split` **in height-column space**, where a column is a solid run
and an interior cannot exist. Grading a hollow-sensitive candidate against a hollow-blind floor
compares two different measurements.

**Consequence: every volume comparison in this bar is made on sealed occupancy.** A cavity is
sealed iff it is unreachable from outside, which is `hollow_shell_voxels`' existing test, so a
courtyard open to the sky or a passage clear through the building survives untouched.
`fill_sealed_cavities` and `sealed_volume_metrics` implement it; no new geometry logic was needed.

## Populations

The corpus grew: `real.h5` now holds BuildingWorld, 1,562,554 rows ingested 2026-09-14, and the
ledger carries 31,493 held-out rows. #162 made that growth **append-only** and pins the first
35,776 rows, so the legacy prefix is intact and the pinned-714 row indices still resolve to the
same buildings. The two populations coexist by design, and CONTEXT.md governs which does what.

| role | population | n | duty |
|---|---|---|---|
| **regression guard** | the pinned 714 | 714 | did this damage what already worked |
| **screen** | family-stratified draw, disjoint from training | **250** | stage-1 futility boundary, one time only |
| **confirmation** | #178's own nine-city held-out slice | **24,464** | the PASS bar |
| sealed complement | confirmation minus screen | 24,214 | published separately, never used to decide |
| training | the full held-out=0 bank | **1,530,908** | owner's call: all of it |

**The confirmation population is #178's, exactly.** Scoring on any other set silently voids every
quoted floor - the population-mismatch error this project keeps catching, and the one #126's own
first draft made.

⚠️ **A discrepancy in the existing record, flagged not resolved.** `178-buildingworld-baseline-signoff.md`
states the query population as **20,720 buildings**, while
`execution/artifacts/buildingworld_baseline_178.json` reports `n = 24,464` and carries 24,464
per-building rows, whose per-city counts sum to exactly that. Every number in this document is
taken from the artifact, since that is what the floors were actually computed on. Which of the two
is the typo is #178's to settle; it does not change a floor either way, but it should not be left
sitting in the record given that this is precisely the class of error #126 exists to catch.

Widening the bar to Mississauga or #169's pooled cities is #170's decision to reopen, not this
ticket's.

⚠️ **The screen is family-stratified, not city-stratified.** Family is where the two directions
separate, so a family draw tests both by construction. A city draw cannot - and Cambridge's five
held-out rows make even coverage impossible. This also fixes a defect in #125's own 32-per-region
screen: measured on the legacy corpus, **PLATEAU contributes 0 of 231 carve-needing rows**, so a
third of that cohort was structurally incapable of testing carving at all.

**Training on the full bank is the owner's explicit instruction**, overriding #125's 384-row
cohort. #125 sized that for a corpus that no longer exists, and at 384 a negative result cannot
distinguish "no correction signal" from "not enough data", which is the one thing #113 needs it to
do. ⚠️ Recorded consequence: this stops being the cheap throwaway probe #113 describes and becomes
a full training run. The screen's job changes with it - from "kill it before we spend" to
"checkpoint on the way through".

⚠️ **Unmeasured, and required before the cohort is final.** No record exists of how fast one A2
generation runs on this machine, and 1,530,908 of them is the dominant cost. **The first action in
the isolated GPU window is a 200-building timing probe**, and its result is published beside the
cohort. Caching the full bank is roughly 1 TB. If the probe makes the full bank infeasible inside
the window #113 allows, the cohort is reduced *by an explicit amendment to this document*, never
silently.

## What is measured

Vocabulary is CONTEXT.md's, unchanged. `missing` is GT the arm failed to fill; `extra` is volume it
added outside GT; both are fractions of the real building's volume, both on sealed occupancy.
`vs_input` is IoU against the A2 output the arm started from, where 1.0 means it did nothing.
`collapse_rate` is the fraction of rows with `missing` >= 0.15.

Reported for the candidate, the A2 baseline and the GT target alike, so that no rate is ever quoted
without a control - #117's rule, carried forward.

**Aggregate 3D IoU is a diagnostic and appears in no clause.** #126 demoted it: on the median a
real building and the envelope are indistinguishable (0.82948 against 0.82951), so a threshold on
it cannot mean what a threshold is for. #126 binds #118 to two constraints, and both are honoured
here - paired improvement is **not** expressed as an aggregate-IoU threshold, and the
beats-envelope clause **names its metric and its tie handling**.

**Ties are excluded from every win rate and published beside it.** Pooling them into the
denominator is what turned a real 60% into a 46% "coin flip" on #126's own 72 rows, where on 17 of
them the alternative's roof simply *was* the envelope.

## The bar

Three parts, following this project's own PASS / GUARD / KILL shape, and machine-checked in
`screening_gate`, `confirmation_gate` and `kill_clauses` rather than in prose.

The PASS gate is a **conjunction**, which is Berger (1982)'s intersection-union test: each clause
may be run at the same level and the conjunction is still level-alpha, so stacking clauses needs no
multiplicity correction. The KILL gate is a **disjunction**, where that result runs the other way -
three clauses at 5% would give roughly a 14% false-kill rate - so its clauses are stated as
point-estimate facts carrying no alpha at all.

### Stage 1 - the screen (250 rows, one time)

| clause | threshold | source |
|---|---|---|
| paired `extra` wins over A2, **opportunity rows** | **Simon boundary on the decided rows** - 136 if 250 decide | computed |
| paired `missing` wins over A2, **opportunity rows** | same | computed |
| decided comparisons per axis | >= 30 **and** >= half the scored rows | quoted; fraction mechanical |
| `vs_input` on opportunity rows | < 0.98 | CONTEXT.md; `quoted_bar`'s `max_vs_input` |
| collapse rate | <= the A2 baseline's own, same rows | quoted |
| identity rows, both axes | no worse than A2 | quoted |
| footprint (criterion 2) | pass rate >= A2's own | #85, allowance 0.05 |
| validity (#117) | invalid rate <= A2's own | quoted |
| human review | 12-row montage, #148's rubric | owner |
| Sharp Normal Error | measured, narrowband no worse than the EDT control | owner + #79 |

**Where the boundary comes from.** Simon (1989)'s two-stage design stops for futility at or below
r1 wins of n1. Against the alternative worth detecting (p1 = 0.60) with the null at a coin flip,
n1 = 250 gives r1 = 136 - proceed above 54.4% - which **kills a useless arm 92.7% of the time for a
4.1% power cost**. At #125's original 96 the same design kills only 62.0%. The number was computed,
not chosen; the owner chose only the cohort size.

⚠️ **The boundary is computed on the DECIDED comparisons, not on the cohort size.** Ties are
excluded from every rate in this document (#126), so `wins` counts decided rows; holding that count
to a boundary computed on all 250 would score every tie as a failure to win, which is the pooling
#126 forbids. With 200 of 250 decided the boundary is 108, not 136. **This was not settled in the
interview** - it is a mechanical consequence of the tie rule, recorded here rather than left
implicit, and so is its guard: at least half the cohort must produce a decided comparison on each
axis, so a thin decided slice cannot buy a cheap boundary. 40 unanimous wins of 250 clears its own
boundary and still fails the screen.

⚠️ **The magnitude of the improvement is deliberately not a screen clause.** The EMA guideline on
the choice of non-inferiority margin and ICH E9/E10 require a margin justified on domain grounds,
not statistical convenience - and a first draft of this bar set one at "+0.05 because it is 11x the
CI half-width", which is exactly the disallowed reasoning. The only domain anchor available is
#117's own LOD2.1 minimum building-part size: on the median *legacy* carve-needing building
(40,165 voxels - measured there because the pinned 714 is where per-building volumes are already
published) one minimum part is about 120 voxels, so a 0.01 change in `extra` is roughly 3.3
building-part-sized chunks and 0.05 is about 17. The domain scale rules out anything below ~0.003 as sub-visible and
is silent above it. Rather than invent a bar in that silence, the screen grades **direction**
(the sign test) and the confirmation grades **magnitude** (the 1-NN floors).

### Stage 2 - the confirmation (24,464 rows, frozen checkpoint)

Primarily `buildingworld_baseline.buildingworld_verdict` against a #118 registration built with
`include_missing=True`, so both axes are quoted from 1-NN on each floor's own population per #170.

| floor | n | `max_missing` | `max_extra` | `max_collapse` | gating |
|---|---|---|---|---|---|
| overall | 24,464 | 0.0242 | 0.0262 | 0.1345 | full |
| Berlin | 9,141 | 0.0250 | 0.0382 | 0.1276 | full |
| Cape Town | 4,907 | 0.0247 | 0.0779 | 0.1213 | full |
| Edmonton | 5,669 | 0.0198 | 0.0189 | 0.1067 | full |
| Montreal | 1,111 | 0.0272 | 0.0322 | 0.1125 | full |
| Tokyo | 768 | 0.0261 | 0.0483 | 0.1445 | full |
| gable_hip | 8,454 | 0.0159 | 0.0828 | 0.0719 | full |
| Boston | 2,294 | **0.0619** | 0.0000 | 0.2681 | `beats_extra` signed off |
| Melbourne | 156 | **0.0603** | 0.0000 | 0.2308 | `beats_extra` signed off |
| Cambridge | 5 | 0.0959 | 0.0000 | 0.0000 | GUARD-only |
| New York | 413 | 0.0000 | 0.0000 | 0.0920 | GUARD-only |

🔑 **Adding the `missing` axis repairs three of #178's four infeasible floors.** Boston, Melbourne
and Cambridge were signed GUARD-only on 2026-09-15 because their `extra` floors quote to 0.0 and
`extra < 0.0` is never true. Their error was always in the `missing` direction, so with that axis
they have a real, clearable floor again. Only New York remains genuinely unreachable - 1-NN scores
0.0000 on both, it retrieves exactly.

**Owner's call on how that is applied.** Boston and Melbourne return to gating **on the `missing`
axis only**, through per-clause sign-off: `beats_extra` alone stops gating, because that single
clause is analytically unreachable while the rest of the floor is not. Retiring the whole floor
would discard a clause the arm can be held to; promoting it to full gating would demand
`extra < 0.0`. Cambridge stays GUARD-only at n=5 - five buildings cannot gate anything - and so
does New York.

⚠️ **This applies to #118's arm only.** #178's registration and #183's running ladder are untouched:
the `missing` axis is opt-in (`include_missing=False` by default) and #178's existing artifact
grades bit-identically. Amending a signed-off bar under a running experiment would move the
goalposts mid-ladder.

Added on top of the floors, because a carve-only height map cannot fail them:

| clause | threshold | source |
|---|---|---|
| beats the envelope on `extra` | 95% lower bound > 0.50 of **decided** rows | owner + #126 |
| collapse | no regression against the screen | quoted |
| validity | no regression against the screen | quoted |
| footprint (criterion 2) | pass rate >= A2's own | #85 |
| human review | full montage, blind, #148's rubric | owner |
| Sharp Normal Error | measured, narrowband no worse than control | owner |
| assay sensitivity | the baseline reproduced its arm of record | ICH E10 |

**The beats-envelope clause, stated exactly as #126 demands.** Metric: `extra`. Ties: excluded from
the rate, published beside it. Threshold: the owner set "more decided rows than not"; the
statistically honest form is the 95% Wilson lower bound above 0.50, which at n = 24,464 is
**12,361 of 24,464 = 0.5053**. On the legacy 411 the same clause would have needed 0.5426 - at this
population size the statistical clause costs about half a point and **the 1-NN floors carry the
entire bar**, the reverse of the legacy-714 design.

### KILL - the sentences that answer #113 "no"

| clause | fires when | source |
|---|---|---|
| `killed_not_transforming` | no better than the envelope on **both** axes | `verdict()`'s `killed_identity`, made two-sided |
| `killed_collapse_over_1nn` | collapse > 0.1582 | CONTEXT.md: "not servable whatever else it scores" |

Beating the envelope on one axis is a real, if partial, transform and is not killed.

## Uncertainty

- **Win rates**: Wilson score lower bound. Brown, Cai & DasGupta (2001) recommend Wilson at these n
  and show the Wald interval's coverage is erratic.
- **Medians**: percentile bootstrap, 2,000 resamples, seed fixed - `five_arm_scorecard`'s existing
  `bootstrap_median_ci`, reused rather than reimplemented. Half-width falls from +/-0.0044 at n=57
  to about +/-0.0021 at n=250 and +/-0.0001 at n=24,464.
- **Seed noise does not apply to this arm.** The corrector is deterministic and #125 freezes its
  source byte-identical across stages, so paired differences carry no seed term. The 0.04 figure in
  `massing_arms_eval_ship714.json`'s `noise_floor` is the seed-to-seed range of a *stochastic* arm
  and must not be quoted as this arm's noise floor. It reappears in exactly two places below.
- **A stochastic arm inherits it.** Preregistered now, at no cost: any later discrete-diffusion arm
  must be run on **at least 3 seeds** and judged on the seed-pooled median with the across-seed
  range published. Its floors are otherwise these.

## Human review and Sharp Normal Error

**Human review blocks at both stages** - 12 rows at the screen, the full montage at the
confirmation - using #148's fixed three-question rubric, with fixed-frame plan/facade/isometric/
section views, and explicit architectural-versus-noise and collapse checks.

This is not a courtesy clause. Blau & Michaeli (2018) prove that distortion and perceptual quality
are at odds, and that the result "holds true for any distortion measure". So no scalar in this
document - `missing`, `extra`, criterion 2 or Sharp Normal Error - can be substituted for a human
looking at the building. #126's finding that the envelope wins on paired distortion metrics is an
instance of that theorem rather than a quirk of this corpus.

**Sharp Normal Error is graded within one arm only.** The owner asked for SNE with nonzero views as
a gate condition; the honest shape is narrower than that, for two measured reasons. Its own
docstring records that it is *contaminated across arms* - on a row whose occupancy is byte-identical
to GT it still reads 0.241, not 0 - and says it is "reported, never ranked on". And **SNE has never
actually been measured on this project**: every committed artifact carries `views: 0, values: {}`,
so there is no number of record to quote a cross-arm floor from. The clause is therefore: the
narrowband surface recovery must be no worse than the signed-EDT control over *identical*
occupancy, at nonzero views - the one comparison the instrument supports - and `views == 0`
**fails** the gate. A measurement that was not taken reads as absent, never as a pass, which is
`verdict()`'s own `--no_form` rule.

## Failure accounting

Invalid rows, collapsed rows and visually poor rows **stay in the primary analysis**, with the
invalid rate carried beside the headline as its own guard. This is ICH E9's intention-to-treat
principle: excluding them is a per-protocol analysis, which is known to bias toward the arm under
test. #125 asks for the same thing in its own words - failed and collapsed outputs stay in metrics
and montages, nothing is retried or replaced.

#118's other requirement - **"prevent deterministic footprint sanitization or returning the input
from claiming a learned win"** - is secured in three places, and it is worth being exact about
which of them are live today.

1. **Sanitization earns no credit, today, by parity.** `_score_row` applies `sanitize_footprint` to
   the candidate and the cached A2 baseline alike, so footprint intersection cancels out of every
   paired difference in this bar. That is #125's own baseline-parity rule and it holds now.
2. **It disappears entirely once #117 lands.** #117 settled that nothing is projected and nothing
   is rejected, which deletes `sanitize_footprint` from scoring and leaves no deterministic
   cleanup to credit at all. ⚠️ That is **#187's implementing pass, not yet applied** - until it
   is, the baseline in every clause here is the sanitized A2 arm rather than the raw one. Both
   arms move together either way, so no clause changes its meaning when #187 lands.
3. **Returning the input cannot pass.** `vs_input < 0.98` on the opportunity rows, which is
   CONTEXT.md's own guard rather than #125's looser 0.99 placeholder, and both futility clauses
   are wins *against that same input*, so a no-op scores zero wins on both axes.

## Assay sensitivity

Every number here is a paired difference against the cached A2 baseline. If the cache does not
reproduce the shipped operating point, the run measures something else and still returns a tidy
verdict. ICH E10 names this: a comparison is uninterpretable unless the control behaved as expected.

**The run is void, and the gate is not evaluated, unless the cached A2 baseline's median `extra` on
the pinned 714 lands within 0.04 of 0.09219727319195108** - the shipped arm's own committed number
in `massing_arms_eval_ship714.json`. The tolerance is this project's measured seed-to-seed range for
`extra`, which is exactly the term #125's per-row reseeding introduces. The pinned 714 is the
natural control because it is the one population where A2 has a number of record; A2 has never been
scored on BuildingWorld.

## Where nothing fixes the number, stated plainly

- **The magnitude of a meaningful surplus improvement.** The LOD specification sizes a visible
  building part and therefore a floor of about 0.003; above that it is silent. This bar does not
  invent one - see the screen section.
- **The screen cohort size.** Simon's design computes the *boundary* once n1 is fixed; it does not
  fix n1. 250 is the owner's choice, made against the measured trade-off (62% -> 92.7% kill rate
  for the same power cost).
- **Whether a failed screen permits one re-fit.** Not settled here, and deliberately: the screen is
  one-time by #125's own text, and Dwork et al. (2015) is why - a held-out set reused adaptively
  stops being held out. A second look requires a new ticket and a new cohort, not a rerun.
- **The training cohort's feasibility**, pending the timing probe.

## References

- Simon, R. (1989). Optimal two-stage designs for phase II clinical trials. *Controlled Clinical
  Trials* 10(1):1-10.
- Berger, R.L. (1982). Multiparameter hypothesis testing and acceptance sampling. *Technometrics*
  24:295-300. See also Berger & Hsu (1996), *Statistical Science* 11(4).
- Brown, L.D., Cai, T.T., DasGupta, A. (2001). Interval estimation for a binomial proportion.
  *Statistical Science* 16(2):101-133.
- Blau, Y., Michaeli, T. (2018). The perception-distortion tradeoff. *CVPR 2018*.
- Maier-Hein, L., Reinke, A., Godau, P., et al. (2024). Metrics reloaded: recommendations for image
  analysis validation. *Nature Methods* 21(2):195-212.
- Dwork, C., Feldman, V., Hardt, M., Pitassi, T., Reingold, O., Roth, A. (2015). The reusable
  holdout: preserving validity in adaptive data analysis. *Science* 349(6248):636-638.
- Nosek, B.A., Ebersole, C.R., DeHaven, A.C., Mellor, D.T. (2018). The preregistration revolution.
  *PNAS* 115(11):2600-2606.
- ICH E9, *Statistical Principles for Clinical Trials* (intention-to-treat, pre-specification);
  ICH E10, *Choice of Control Group in Clinical Trials* (assay sensitivity); EMA,
  *Guideline on the choice of the non-inferiority margin* (CPMP/EWP/2158/99).
- Biljecki, F., Ledoux, H., Stoter, J. (2016). An improved LOD specification for 3D building
  models. *CEUS* 59:25-37. (Via #117, for the building-part size anchor.)
- Project: #117 (validity contract), #125 (spec), #126 (how massing is scored), #148 (human
  rubric), #170 (falsification rule), #178 (BuildingWorld bar and sign-off), #85 (criterion 2),
  #79 (Sharp Normal Error), ADR 0004 (`s*`).

## Code

Landed with this document, CPU only, no GPU touched and no experiment run.

- `prototype_voxel_editor.fill_sealed_cavities` / `sealed_volume_metrics` - hollow-invariant
  scoring, wired into `_score_row`.
- `prototype_voxel_editor.wilson_lower_bound`, `futility_boundary`, `probability_of_early_stop`,
  `paired_win_record` - the statistics, with ties excluded and published.
- `prototype_voxel_editor.screening_gate`, `confirmation_gate`, `kill_clauses`,
  `assay_sensitivity` - the bar, machine-checked.
- `buildingworld_baseline.quoted_bar(include_missing=)`, `infeasible_clauses_for`,
  `preregister(include_missing=)`, `buildingworld_verdict`'s `beats_missing` clause, and
  `apply_signoff(guard_only_clauses=)` - the `missing` axis and per-clause sign-off, opt-in so
  #178 and #183 are unaffected.

### Deliberately not done here

- **`CONTEXT.md` is untouched.** Two findings in this document are project-wide vocabulary and not
  specific to this arm - that volume comparisons must be made on sealed occupancy, and that
  `missing` is a PASS axis wherever an arm can add mass. Neither contradicts anything `CONTEXT.md`
  currently says, so nothing there is wrong; both are additions. #113's Out of Scope and #125's
  Implementation Decisions both forbid this effort from editing `CONTEXT.md`, so they are recorded
  here and left for a ticket that is allowed to make them.
- **#178's registration and #183's ladder are untouched**, verified by re-grading the committed
  arm-3 report with the committed #178 registration and getting all eleven floor verdicts back
  bit-identical. Pinned by `test_arm3_regrades_bit_identically_under_the_extended_verdict`.

### Still to do, for the implementing pass

- **Rebuild the cohorts on BuildingWorld.** `build_manifest` and `select_screen_rows` are still
  #125's three-region, 384/96/714 design. The screen must become a 250-row family-stratified draw
  and the confirmation #178's 24,464. `GATE_SCREEN_N` and `GATE_CONFIRMATION_N` already hold the
  settled values and the gate enforces them, so a cohort that was not rebuilt fails the bar rather
  than passing quietly.
- **Apply #117's consequences**, which are that ticket's to implement (#187) and are assumed here:
  `sanitize_footprint` no longer applied at `_score_row` or `build_cache`, the two-rule validity
  contract, `validity_projection_delta` removed. Until then the baseline in these clauses is the
  sanitized A2 arm rather than the raw one.
- **Build #118's registration**: run `preregister(report, include_missing=True)` against #178's
  own measured report, then `apply_signoff` with `guard_only_floors=["city:Cambridge",
  "city:New York"]` and `guard_only_clauses={"city:Boston": ["beats_extra"], "city:Melbourne":
  ["beats_extra"]}`, and commit it as a separate artifact from #178's.
- **The A2 timing probe**, before the training cohort is final.
- **Wire the montage and SNE passes** into the screen and confirmation reports. Both clauses read
  false until then, which is intended.
