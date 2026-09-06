# #8 — Specify the minimal falsifiable proof

*Effort: solid-first semantic architectural carving. Opened 2026-07-15, grilled 2026-09-06. Blocked
by [#6](6-program-generator.md), [#7](7-validity-gates-and-visual-carving-traces.md),
[#9](9-multi-footprint-coordination.md), [#10](10-program-recovery.md), all closed; blocks
[#2](https://github.com/danvisai/SDFusion/issues/2) directly.*

> What is the smallest falsifiable experiment package that can distinguish the proposed
> contribution from integration of known components? Specify hypotheses, held-out real data,
> autonomous and guided-edit tasks, semantic-void and edit-locality measurements, multi-footprint
> coordination tests, direct baselines and ablations, statistical reporting, visual carving traces,
> failure criteria, and a kill gate for the dual-mode semantic-carving claim.

Resolved by interview (`/grilling`) with the ticket owner, not by measurement — this is a proof-package
*design* decision, not an experiment. Facts cited below (corpus void counts, code paths that do or
do not exist, prior baseline protocols) were checked against the repository during the session; the
decisions themselves are the owner's.


## 🔑 The five potentially-novel hypotheses from `NOVELTY_SURVEY.md` are the falsifiable claims, not new ones

`NOVELTY_SURVEY.md`'s own "Potentially novel hypotheses" section already named the five claims worth
treating as falsifiable research claims rather than assumed novelty. This ticket does not invent new
hypotheses — it decides which of those five are in the minimal package, and how each is measured:

| # | hypothesis | in package? |
|---|---|---|
| H1a | one semantic edit distribution serves autonomous footprint-to-program generation | ✅ yes |
| H1b | ...and interprets/completes a rough local add/subtract gesture into the same op space | ✅ yes, headless proxy only |
| H2 | architectural voids (courtyards, passages, light wells) as first-class generative objects | ❌ dropped, see below |
| H3 | a selected footprint block can share coordinated decisions while staying independently valid | ✅ yes |
| H4 | edit locality survives program regeneration | ✅ yes, cited not re-measured |
| H5 | a real-geometry benchmark for semantic carving programs exists | ✅ yes, this is the benchmark being assembled |


## H2 (voids as first-class objects) is dropped, not silently cut

[#4](4-edit-algebra.md) measured **0 through-void voxels in 4,324,919** across the entire corpus;
[#5](5-data-audit.md) independently confirmed no LoD2 source (3D BAG, NRW CityGML, PLATEAU) carries
courtyards, passages, arcades, or light wells at all — it is a **ceiling of the real data**, not a
gap this proof package can close by trying harder. Falsifying a claim against data that structurally
cannot support either outcome would not be a real test.

**Decision:** H2 is out of this package. It remains a named, deferred hypothesis — matching
[#9](9-multi-footprint-coordination.md)'s own treatment of the courtyard/style axes — pending
procedural synthetic void data, the one route [#5](5-data-audit.md) found un-superseded by any real
source, narrowly, for the volumetric tier.


## H4 (edit locality) is cited, not re-measured

[#7](7-validity-gates-and-visual-carving-traces.md) already retired edit locality as a *scored*
metric, on the grounds that [#144](https://github.com/danvisai/SDFusion/issues/144)'s structural
proof already exceeds what a statistical score could show. Re-running a statistical locality test on
top of a structural proof would be strictly weaker evidence, not stronger.

**Decision:** the proof package cites [#144](https://github.com/danvisai/SDFusion/issues/144)
directly for H4. The one exception is H1b below, where locality is re-checked on a *new* code path
(the guided-edit re-fit), not re-litigated on the path #144 already covers.


## H1a — autonomous generation: five arms, never collapsed to one number

**Ablation set, exactly five, nothing added:**

1. `blockout` — do nothing (the safety floor).
2. `1-NN` retrieval — zero-training, fully automatic. This is **the pass/fail baseline**, not
   `blockout`.
3. the raw [#127](127-height-map-generator.md)/[#155](https://github.com/danvisai/SDFusion/issues/155)
   height-map generator, unfused.
4. `fit_decode` — the generator fused through [#10](10-program-recovery.md)'s fitter, the arm
   actually served today.
5. `fit_decode` + [#9](9-multi-footprint-coordination.md)'s block-coordination bias.

Multi-hypothesis (`k_hyp`) training is deliberately excluded:
[#155](https://github.com/danvisai/SDFusion/issues/155) already showed it is oracle-decoded only,
never a servable arm — it is a diagnostic this map already ran, not a candidate for this proof.

**Decision: arms 3 and 4 are always reported side by side, never collapsed into "the system."**
[#155](https://github.com/danvisai/SDFusion/issues/155) found `fit_decode` is *safer* (collapse
0.0268 → 0.0049) but has *worse* architectural form (`dl_planar_fraction` 0.20 → 0.00) than the raw
generator it fuses. Picking one as "the" arm to score would hide exactly that split — the same
mistake this map has refused to make in [#130](130-baselines-diffusion-curriculum.md),
[#126](126-massing-scoring.md), and [#155](https://github.com/danvisai/SDFusion/issues/155) itself.

**Baseline protocol, and why 1-NN's status here differs from #127's own ruling on it:**
[#127](127-height-map-generator.md)'s pre-registration originally named 1-NN "the real bar," but its
own later, standing ruling (dated 2026-08-28, made after seeing results) **demoted** it to "a
reference point, not a gate" — because requiring one specific *parametric* trained network to beat a
*non-parametric* baseline that carries all 34,909 training roofs to inference time treats a fair
compression constraint as a quality failure for that network. That ruling is about judging a single
generator's own training run, and stands unchanged.

This ticket asks a different question — not "is this network good enough" but "does the system beat
mere integration of known components" — and 1-NN retrieval *is* one of the simplest such known
components. For that comparison, being non-parametric and automatic is exactly what makes it the
right line, not a reason to excuse it. [#130](130-baselines-diffusion-curriculum.md)'s
named-baseline-equivalents (ArcPro/CoMa's `flatten_ramps`, CityGenAgent's `blockout`), by contrast,
were run **oracle-fed** — given [#10](10-program-recovery.md)'s ground-truth recovered program, not
run automatically — which makes them a ceiling check, not a fair automatic-vs-automatic comparison.

**Decision:** `1-NN` is this package's pass/fail gate for H1a specifically (distinct from, and not a
reversal of, #127's own generator-quality ruling). [#130](130-baselines-diffusion-curriculum.md)'s
oracle-fed numbers stay in every report table, explicitly labeled oracle-fed/ceiling-only, never as
the pass/fail comparison.

**Verdict shape:** each hypothesis resolves to one of three readings — **SUPPORTED**,
**PARTIALLY SUPPORTED**, or **FALSIFIED** — reported **per axis** for H1a (volume/safety vs.
architectural form), not collapsed into one word when the axes disagree. This is a hypothesis-level
rollup, a different altitude from `verdict()`'s own PASS/GUARD/KILL bar on a single measured arm
(CONTEXT.md's "the bar itself" entry) — an axis reads PARTIALLY SUPPORTED exactly when its
constituent arms disagree the way [#155](https://github.com/danvisai/SDFusion/issues/155) found
(safety clears its bar, form does not), never by inventing a fourth bar component.


## H1b — guided-edit: a headless proxy, not new interaction code

No gesture, accept/reject, or confidence UI exists anywhere in the repository —
[#3](3-dual-mode-carving-edit-locality.md) explicitly left that layer not-yet-specified rather than
guessed against a demo the owner disowned mid-session, and this session's own repository search
confirms no completion/gesture code path exists to test yet. Building it is out of scope for a
planning-only map ([#1](https://github.com/danvisai/SDFusion/issues/1)).

**Decision: test the underlying completion capability headlessly, without any UI.** Task: take a
held-out building's program already recovered by [#10](10-program-recovery.md), apply a synthetic
"rough gesture" — a box-shaped add or subtract volume, footprint-column-local, at several sizes and
positions — and re-run the constrained beam-search fitter to produce a completed program.

**Pass bar, exactly two checks, nothing else:**
1. The completion passes [#7](7-validity-gates-and-visual-carving-traces.md)'s finalize-time gate.
2. [#144](https://github.com/danvisai/SDFusion/issues/144)'s locality invariant holds **on this
   specific re-fit path**: every op whose footprint columns lie outside the gesture's own
   footprint-column footprint must be provably unchanged. This is a fresh check, not a re-litigation
   of #144 — #144 proved locality on a hand-built mixed program, never on a re-fit triggered by a
   synthetic gesture, and the fitter's re-fit path is different code from #144's direct algebra
   edits.

**Decision: no gesture-accuracy score.** There is no real "intended" shape a rough box implies —
scoring the completion's IoU against some invented target would just declare the fitter's own output
ground truth, exactly the "synthetic-generator ceiling" risk `NOVELTY_SURVEY.md` already names as a
novelty risk (risk 5).


## H3 — coordination: real product footprints, one binary check

[#9](9-multi-footprint-coordination.md) built the mechanism and explicitly deferred its evaluation
to this ticket. No real multi-footprint ground truth exists anywhere in the repository or corpus:
`town_generate_service.py` generates every building fully independently, on a decorrelated seed, and
neither `real.h5` nor `Bag3dDataset` carries an adjacency or parcel field. There is no real paired
block to score massing accuracy against.

**Decision: block scenes are real footprints as the product actually encounters them** — drawn or
imported together in the town demo — not a synthetic assembly of held-out corpus rows pretending to
be a block.

**Decision: the pass bar is footprint adherence alone.** Run [#9](9-multi-footprint-coordination.md)'s
block-coordination bias (each of the four live axes independently, and combined) across each block
scene. PASS requires **100% of the footprints in the block still clear
[#7](7-validity-gates-and-visual-carving-traces.md)'s finalize-time gate** after the coordinated
re-fit — any regression relative to that footprint's own uncoordinated fit is a named failure. No
"did the shared axis actually get more consistent" statistic is computed — that reading is
qualitative and left to a human reviewing [#7](7-validity-gates-and-visual-carving-traces.md)'s
existing visual carving trace on the assembled block, not a number this package reports.


## H5 — the benchmark itself, gated on the split fix already in flight

[#5](5-data-audit.md)'s audit already found the standing evaluation set has a live, only
eval-sampling-patched defect: today's row-random permutation in `Bag3dDataset` has a proven region
confound. [#153](https://github.com/danvisai/SDFusion/issues/153) exists specifically to fix that at
the split level.

**Decision: this proof package's held-out data is [#153](https://github.com/danvisai/SDFusion/issues/153)'s
region/tile-stratified split, not the legacy pinned-714** — running the most rigorous evaluation this
map produces on a split with a known, named confound would undercut the package's own credibility.

**Decision: [#154](https://github.com/danvisai/SDFusion/issues/154)'s ~60-building human-audited
semantic annotation set is re-pointed at [#153](https://github.com/danvisai/SDFusion/issues/153)'s
output too**, sequenced after it rather than run in parallel against the set it's about to supersede
— annotating against a split you already know is confounded just means redoing the pass later for no
reason.

**Consequence:** H5's benchmark — [#153](https://github.com/danvisai/SDFusion/issues/153)'s split +
[#154](https://github.com/danvisai/SDFusion/issues/154)'s semantic annotations +
[#7](7-validity-gates-and-visual-carving-traces.md)'s visual carving trace and
[#148](https://github.com/danvisai/SDFusion/issues/148)'s rubric — exists as *infrastructure* today
(H5 itself: SUPPORTED), but this ticket's other measurements (H1a's five-arm scorecard) do not run
until [#153](https://github.com/danvisai/SDFusion/issues/153) lands.


## Evidence and reporting standard, reused not reinvented

**Visual evidence:** [#7](7-validity-gates-and-visual-carving-traces.md)'s existing 4-fixed-view,
delta-highlighted carving trace and [#148](https://github.com/danvisai/SDFusion/issues/148)'s existing 3-question human
rubric are the qualitative reporting mechanism for every judged output in this package — autonomous
generations, coordinated blocks, and guided-edit-proxy completions alike. Nothing new is built.

**Statistical reporting:** every number in this package states its **true, non-inflated N** (distinct
buildings, block scenes, or gesture/building pairs — never camera-angle-multiplied or otherwise
inflated counts) with a bootstrap confidence interval computed on that true N, explicitly flagged
when too small to trust. This is [#10](10-program-recovery.md)'s own sculpt-strength-sweep
convention (`fid.undersampled` reading false at an inflated N while the real effective N stayed
small), adopted project-wide for this package rather than re-derived.

**Reporting shape: a per-hypothesis scorecard, never a single collapsed verdict.** H1a
(volume/safety and architectural form as separate axes), H1b, H3, and H5 each get an independent
SUPPORTED / PARTIALLY SUPPORTED / FALSIFIED reading, per the verdict shape defined under H1a above.
There is no single "did dual-mode semantic carving pass" sentence this package produces.


## What this ticket explicitly does not decide

- **The exact synthetic-gesture parameterization** (box sizes, positions, how many per building) and
  **the exact block-scene sourcing mechanics** (how many real product footprint sets, how they're
  collected) are left to the spawned execution tickets below, at the same altitude
  [#9](9-multi-footprint-coordination.md) left its own scoring-bias formula to
  [#149](https://github.com/danvisai/SDFusion/issues/149)–[#151](https://github.com/danvisai/SDFusion/issues/151).
- **Final cross-hypothesis aggregation** (reading all four scorecard rows together once every
  spawned ticket lands) is not itself a ticket — it is a trivial last step once the last spawned
  ticket reports its row, not new work.


## Tickets this spawned

`ready-for-agent`, part of [#1](https://github.com/danvisai/SDFusion/issues/1):

- [#179](https://github.com/danvisai/SDFusion/issues/179) — Build the Guided-Edit Completion Proxy
  and Re-Verify Locality on It (H1b)
- [#180](https://github.com/danvisai/SDFusion/issues/180) — Test Multi-Footprint Coordination
  Against Real Product Footprint Sets (H3)
- [#181](https://github.com/danvisai/SDFusion/issues/181) — Run the Five-Arm Autonomous-Generation
  Scorecard Against the Stratified Split (H1a), blocked by
  [#153](https://github.com/danvisai/SDFusion/issues/153)

[#154](https://github.com/danvisai/SDFusion/issues/154) (H5's semantic annotation set) is amended,
not respawned: re-pointed at [#153](https://github.com/danvisai/SDFusion/issues/153)'s output and
now blocked by it.


## What follows

- [#2](https://github.com/danvisai/SDFusion/issues/2) (the integration boundary) is unblocked by
  this ticket directly.
- [#153](https://github.com/danvisai/SDFusion/issues/153) becomes the real gate on this proof
  package actually running — [#8](https://github.com/danvisai/SDFusion/issues/8) itself closes on
  the package's *design*, matching how [#3](3-dual-mode-carving-edit-locality.md),
  [#7](7-validity-gates-and-visual-carving-traces.md), and
  [#9](9-multi-footprint-coordination.md) each closed on their decision, not on execution.
- H2 (voids) remains named and deferred, unblocked by nothing yet chartered — not silently dropped,
  not ready-for-agent either.
