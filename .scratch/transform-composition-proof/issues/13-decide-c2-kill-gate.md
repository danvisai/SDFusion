# Decide the Full-Data C2 Kill-Gate

Type: research
Status: open
Blocked by: 05, 11, 12

## Question

Compare the real-pair monolith and Stage 3a-plus-retrieval decomposition at 100% equal detail data,
using paired massing fidelity and distributional facade-detail fidelity with uncertainty and audited
failures. Decide whether C2 survives the preregistered kill-gate, and localize the cause before
graduating either the scaling curve or a remediation branch from the map's fog.

## Answer

**Status: done (2026-07-15). Gate = FAIL.** The decomposition arm does not win detail fidelity
despite retaining comparable (in fact better) footprint massing fidelity — the PRD's exact
kill-gate condition ("does not win on detail fidelity while retaining comparable massing
fidelity") triggers. Per the PRD, this stops the equal-data scaling curve and requires
diagnosing the failure rather than retrofitting the hypothesis.

`scripts/foundations/generate_monolith_arm.py` generated `monolith_v3`'s held-out outputs on the
same 277-id `data/splits_v1/test.json` population ticket 12's decomposition arm already used
(**277/277 succeeded, 0 failures**, `ddim_steps=1000` per ticket 11's own finding).
`scripts/foundations/decide_c2_kill_gate.py` then ran both arms' final outputs plus the real
target through one shared neutral render+FID+IoU harness and applied the PRD's preregistered
rule. Full results in `execution/artifacts/c2_kill_gate_decision.json`; montage at
`outputs/c2_kill_gate/montage.png` (not committed — `*.png` is gitignored repo-wide, matching
every prior ticket's montage).

### Headline numbers

| Metric | Monolith | Decomposition | Gap (decomp − mono) |
|---|---|---|---|
| Footprint IoU, mean (95% CI) | 0.390 (0.350–0.433) | **0.472** (0.435–0.508) | **+0.081** |
| Full-volume IoU, mean (95% CI) | **0.379** (0.339–0.423) | 0.093 (0.081–0.106) | −0.286 |
| Detail FID, point (95% CI) | **116.2** (121.3–133.1) | 143.7 (150.4–161.1) | +27.4 (decomp worse) |

- `wins_detail = False` (decomposition's FID is *higher* — less realistic — than the monolith's).
- `comparable_massing = True` (decomposition's footprint IoU, the module's primary massing
  metric per its own docstring, is 0.081 *above* the monolith's, well clear of the disclosed
  ±0.05 tolerance — decomposition doesn't just retain comparable massing, it wins massing too).
- `gate = fail` because the rule is an AND: winning massing does not compensate for losing detail.
- **Both FID point estimates fall outside their own 95% CI** (`point_outside_ci=True` for both
  arms, at N=2216 images / 277 groups each, clearing the raw `undersampled` N>2048 threshold).
  This is the same small-sample-per-group bootstrap artifact ticket 10's sculpt-strength sweep
  first surfaced: clearing the raw image-count floor does not guarantee a well-behaved bootstrap
  when the *effective* N (distinct buildings) is what actually drives variance. Reported, not
  hidden, and it means the 27.4-point FID gap should be read with real caution about its exact
  magnitude — though the direction (decomposition worse) is consistent across both the point
  estimate and this run's full CI range.

### Footprint vs. full-volume IoU: a large, informative split

Decomposition wins footprint IoU (2D top-down silhouette) but loses full-volume IoU by a wide
margin. `localize_decomposition_failures`'s new `composition_iou_drop` metric (added during code
review — see below) isolates why: the compose step (adding retrieved/procedural elements onto
the base massing) only costs **0.009 mean IoU** relative to ticket 12's own base-massing-only
IoU (0.0085 with retrieval, 0.0098 without — retrieval-active buildings lose *slightly* less,
a mild positive signal for retrieval fidelity). The gap is not primarily a composition-step
failure. It is present already at the base massing level: ticket 12's own massing IoU (mean
0.102, median 0.055) is already this low before any detail is added, because a coarse massing
volume — even a well-fit one — inherently has low voxel overlap with a fully-detailed real
BuildingNet target once you score every voxel rather than just the footprint silhouette. The
kill-gate's own decision correctly reads footprint IoU (the module's designated primary massing
metric), not full-volume IoU, so this gap does not change `comparable_massing=True` — but it is
the right lens for any remediation ticket to inherit: the leverage point is base-massing
fidelity at full-volume resolution, not retrieval quality.

### Failure localization

**Decomposition** (`localize_decomposition_failures`): IoU is fairly flat across leakage tiers
(clean 0.082, train_leak 0.092, val_leak 0.118 — no dramatic leakage-driven inflation).
Retrieval-active buildings (158/277, 57%) average 0.103 IoU vs. 0.081 for non-retrieval
(119/277) — retrieval helps, modestly, but doesn't close the gap to the monolith.

**Monolith** (`localize_monolith_failures`): a genuinely bimodal pattern. **73/277 (26%)** of
generations are near-empty (`gen_occ_frac < 1e-4`, mean IoU ≈ 0.0000147 — essentially zero) vs.
**204/277 (74%)** non-empty averaging **0.515 IoU** — a strong number when the model doesn't
collapse. Fairly uniform across building classes (COMMERCIAL 0.475, RELIGIOUS 0.383, RESIDENTIAL
0.370, PUBLIC 0.348). This empirically confirms and quantifies the 2026-07-13 audit's
"near-empty/fragmentary" caveat — it's not a vague concern, it's a measured 26% collapse rate,
with the non-collapsed 74% substantially outperforming decomposition on full-volume IoU.

### Visual contradiction — read before trusting the FID number alone

The montage (`outputs/c2_kill_gate/montage.png`) shows something the FID number doesn't capture:
**decomposition outputs look far more recognizably building-like** — boxy massing with visible
window/door/architectural detail — **than the monolith's often sparse, fragmentary blob-clusters**
(consistent with the 26% near-empty rate above). Yet FID scores the monolith as *more* realistic.
This is a real, disclosed limitation of computing FID from neutral-shaded renders through an
ImageNet-pretrained Inception model: it may not track architectural-shape realism the way a human
eye does, especially against a reference set (real BuildingNet renders) it was never trained on.
Two other observations plausibly contribute to the gap independent of this limitation:
decomposition's renders show disconnected floating debris / placement artifacts for several
buildings (`castle_mesh0904`, `hotel_building_mesh0295`, `hotel_building_mesh0461` in the
montage), and the monolith's fragment-clusters — while not building-like — may simply sit closer
in Inception feature space to a sparse, thin-surfaced real BuildingNet render than a solid,
retrieval-composed shape with visible seams does. Neither undermines the quantitative FID result
as computed, but both are reasons to treat "decomposition loses on FID" as distinct from
"decomposition looks worse" — they are not the same claim, and a human blinded study (PRD user
story 33, not implemented — see Known gaps) would be needed to settle which one matters more for
the eventual paper's actual claim.

### Known gaps, disclosed not hidden

- **Chamfer distance**: never implemented anywhere in this codebase (confirmed by exhaustive
  grep before ticket 12 and reconfirmed here). PRD line 38 and user story 17 name "paired
  Chamfer and IoU" for massing; only IoU is reported, matching every prior massing-fidelity claim
  in this project (tickets 07/09/10/12). A distance-transform Chamfer on 96³ grids would be cheap
  to add but was not attempted in this ticket.
- **Blinded two-AFC study**: PRD lines 39-40 and user story 33 ask for "distributional facade FID
  plus a blinded two-AFC study for detail." Only the FID half exists. A human-subject blinded
  comparison UI is a substantial separate feature, not attempted here — directly relevant given
  the visual-contradiction finding above.
- **Renderer-effect isolation**: PRD user story 28 asks failure causes separated into "massing,
  retrieval, renderer, and monolith effects." This ticket separates massing/retrieval (via
  `composition_iou_drop`) and monolith (near-empty vs. non-empty), but renderer-effect isolation
  has no precedent anywhere in this codebase and was not attempted — there is nothing existing to
  compare the renderer against (disclosed in the script's own module docstring).

### Code review

`/code-review` ran Standards and Spec sub-agents (model `fable`) against
`b11c5da...HEAD`. Findings and disposition:

- **Fixed**: `kill_gate_decision`'s `massing_iou_decomp`/`massing_iou_monolith` parameters
  actually received footprint IoU, not full-volume IoU, but nothing in the name said so — renamed
  to `massing_fp_iou_*` and the docstring now states explicitly which metric the decision reads.
- **Fixed**: massing IoU had no uncertainty reported (PRD story 19 asks for uncertainty on
  "FID and paired metrics" — only FID had it). Added `bootstrap_mean_ci` (percentile bootstrap,
  same spirit as `fid.py`'s `bootstrap_fid_ci`) and threaded `ci95` through the manifest's
  `massing.footprint_iou`/`full_iou` for both arms.
- **Fixed**: `localize_decomposition_failures`'s own docstring claimed to separate "massing-step
  failure... from detail-composition-step failure" but never actually used the available
  `decomposition_massing_iou` field to compute that split — added `composition_iou_drop` (see
  above), which is what surfaced the "the compose step barely moves IoU" finding.
- **Fixed** (minor): a deferred `from make_splits import parse_class` import inside `main()` was
  moved to module top, matching sibling scripts' convention of only deferring genuinely heavy
  imports (torch).
- **Added**: `input_provenance` in the output manifest, propagating each input arm's own
  `git_rev`/`dirty_digest` (and the monolith's checkpoint identity) for full audit-chain
  traceability — the Spec sub-agent flagged the prior manifest didn't propagate this.
- **Investigated and found to be a false alarm**: the Spec sub-agent flagged that
  `data/decomposition_arm_v1/manifest.json` records `git_rev=be69768` — the commit *before*
  ticket 12's solidity-threshold fix (`b11c5da`) — and concluded the decomposition arm might be
  stale/invalid. Checked directly: the manifest's `dirty_digest` (a hash of uncommitted changes
  at run time) is non-empty, correctly disclosing that the run happened on top of `be69768` with
  uncommitted code — and the retrieval counts recorded in that same manifest (158/277 buildings,
  57%; det_type breakdown `column=286/balcony=7/tower=98/dome=13/chimney=69`) exactly match the
  documented **post-fix** results from the `b11c5da` follow-up commit. The arm genuinely used the
  fixed retrieval code; the provenance system worked as designed, the sub-agent just read
  `git_rev` in isolation without checking `dirty_digest`.
- **Not fixed** (judgement call, left as-is): several Standards-axis findings were minor
  duplication/naming observations on throwaway-after-decision research code (a third near-copy
  of `_montage`, parallel mono/decomp blocks in `main()` that could loop, a mild data clump in
  `kill_gate_decision`'s four float args) — explicitly called "all minor" and "judgement calls"
  by the reviewing sub-agent, not worth the churn/risk on a script whose headline result is
  already decided.

### What this means for the map

Per the PRD, the equal-data scaling curve ("Run the Equal-Data Scaling Curve") stops here rather
than proceeding to 25%/50% fractions. The fog item "the remediation branch if the 100% C2
kill-gate fails" (map.md's Not yet specified) is now informed by real evidence rather than
speculative: the failure is **not** primarily a retrieval-quality problem (composition costs
~0.009 IoU) and **not** primarily a leakage artifact (flat across tiers) — it's (a) the
monolith's own 26% near-empty collapse rate pulling its full-volume IoU picture down while still
winning FID, and (b) decomposition's base massing-to-full-volume-IoU gap being large regardless
of composition. Neither of these was resolved by this ticket; they're the concrete starting
points for whatever remediation ticket the project owner decides to open next.

## Comments
