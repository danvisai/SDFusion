# Tickets: Transform + Composition Research Proof

These tickets build the reproducible evidence package specified in
`.scratch/transform-composition-proof/PRD.md`.

Work the **frontier**: any ticket whose blockers are all done.

## Validate the Element Retrieval Baseline

**What to build:** Make the existing real-element retrieval path experiment-ready by proving that
retrieved elements are appropriately scaled, sufficiently solid, visible at realization resolution,
reversible as recipe operations, and compatible with existing service flows.

**Blocked by:** None - can start immediately.

- [x] Focused retrieval and analytic-composition checks cover the reported failure modes.
- [x] Applicable API and sculpt-flow gates pass, or any environmental limitation is recorded.
- [x] The validated implementation and evidence are committed without overwriting unrelated work.

## Prove the Experiment Run Contract

**What to build:** Make one small experiment flow from configuration through rendering and metrics into
a resumable artifact containing provenance, seeds, failures, outputs, and repository state.

**Blocked by:** None - can start immediately.

**Status (2026-07-12):** satisfied by convention, not a dedicated ticket/script —
`execution/IMPLEMENTATION_PLAN_2026-07-10.md` never spun this into its own `I0.x` item, and every
ticket since (03, 04, 05, 07, 08, 09, 11) independently implements this contract via its own
`manifest.json` (git rev, seeds, failures, per-sample status) and, for ticket 11's training run,
resumability. No further action planned unless a gap is found.

- [x] A smoke run produces a complete, versioned artifact. (per-ticket manifests, see above)
- [x] Repeating or resuming the run preserves its declared inputs and results. (ticket 11's
      checkpoint/resume contract, TDD-tested in `scripts/foundations/test_train_monolith.py`)

## Freeze Leakage-Safe BuildingNet Splits

**What to build:** Produce deterministic, class-stratified held-out and nested training fractions with
a report proving coverage, nesting, balance, and disjointness.

**Blocked by:** None - can start immediately.

**Status: done** — `scripts/foundations/make_splits.py`, `data/splits_v1/`. See map.md.

- [x] The sealed test set never appears in a training fraction.
- [x] The training fractions are nested and reproducible from their recorded seed.

## Capture the Existing Residual Transform Evidence

**What to build:** Turn the prior residual-correction result into a reproducible C1 artifact with metric
provenance and a paper-ready figure.

**Blocked by:** None - can start immediately.

**Status (2026-07-12):** not started, but explicitly scoped as trivial —
`execution/IMPLEMENTATION_PLAN_2026-07-10.md`'s `I-C1.3` marks this "**No new work**": pull the
already-trained `Logs_GT/sdf_residual_full_v4_aug_topk3` result
(`val_corrected_fp_iou≈0.999` vs `val_corrected_iou≈0.13–0.28`) into a figure at paper-writing
time (ticket 18). Genuinely open, but low-risk/low-effort when picked up.

- [ ] Footprint and detailed-shape metrics trace to their original run artifacts.
- [ ] The resulting summary states the evidence and its limitations without re-running training.

## Lock the Shared Geometry Representation

**What to build:** Run a pre-result detail-visibility gate that fixes the shared working resolution and
the metric definition of `s*` before headline results are observed.

**Blocked by:** Prove the Experiment Run Contract.

**Status: done** — `docs/adr/0004-experiment-operating-point.md` (res 96³, `s*`=5 vox). See map.md.

- [x] The gate records whether 96^3 preserves the declared facade-detail categories.
- [x] Every arm uses the same recorded resolution and fixed-a-priori `s*` definition.

## Establish Neutral Geometry Evaluation

**What to build:** Render held-out real geometry through deterministic neutral cameras and compute the
shared massing and detail metrics with uncertainty and failure accounting.

**Blocked by:** Freeze Leakage-Safe BuildingNet Splits; Lock the Shared Geometry Representation.

**Status: done** — `scripts/eval/{fid,render_facades,sanity_real_vs_real}.py`. See map.md.

- [x] Real geometry produces deterministic, comparable facade renders.
- [x] Real-versus-real FID and paired massing metric sanity checks are recorded.

## Build a Leakage-Safe Retrieval Slice

**What to build:** Build a provenance-carrying element library from one declared training fraction,
retrieve and realize an element, and prove that held-out buildings contribute no geometry.

**Blocked by:** Validate the Element Retrieval Baseline; Freeze Leakage-Safe BuildingNet Splits.

**Status: done** — `data/element_library_train100_v1/` (2744 elements, 0 leakage). See map.md.

- [x] Include and exclude contracts are enforced in emitted provenance.
- [x] One retrieved element completes the path from library source to realized geometry.

## Build a Real Monolith-Pair Slice

**What to build:** Convert a real BuildingNet building into an aligned low-pass-to-detailed training
pair with verified SDF conventions, provenance, and reviewable output.

**Blocked by:** Freeze Leakage-Safe BuildingNet Splits; Lock the Shared Geometry Representation.

**Status: done** — `data/monolith_pairs_v1/` (1572/1572 pairs, 0 leakage). See map.md.

- [x] The target is original real geometry, never a composer-generated target.
- [x] Alignment, axes, sign, resolution, and source provenance pass automated checks.

## Test the Fixed Detail-Scale Coincidence

**What to build:** Measure the fixed `s*` against BuildingNet massing and semantic-detail categories and
report pass, partial coincidence, or failure without moving the boundary.

**Blocked by:** Freeze Leakage-Safe BuildingNet Splits; Lock the Shared Geometry Representation.

**Status: done** — `scripts/eval/measure_scale_spectrum.py`. **Result: 6/11, partial_coincidence**
(honest, boundary not moved). See map.md.

- [x] The measurement uses the preregistered scale and category set.
- [x] A reproducible result artifact includes uncertainty and category-level outcomes.

## Measure Transform Versus From-Noise Sampling

**What to build:** Run held-out footprints through honest Stage 3a from-noise and footprint-blockout
SDEdit paths, then compare them through the canonical geometry evaluation.

**Blocked by:** Establish Neutral Geometry Evaluation.

**Status: done** — `scripts/eval/transform_vs_noise.py`. **C1 holds** (median footprint IoU 0.30→0.61).
See map.md.

- [x] Both arms expose and record their exact information and sampling contracts.
- [x] Massing metrics, FID, uncertainty, and qualitative failures are reported together.

## Prototype the Sculpt Transform Sweep

**What to build:** Run representative crude edits across SDEdit strengths and present the resulting
faithfulness-versus-realism curve for review.

**Blocked by:** Establish Neutral Geometry Evaluation.

**Status (2026-07-12): NEXT UP.** Was already unblocked early but got skipped over when ticket
"Train and Evaluate the Full-Data Monolith" was picked up first, deviating from
`execution/IMPLEMENTATION_PLAN_2026-07-10.md`'s prescribed order (sculpt sweep before any
training). Resuming now per the 2026-07-12 pivot (see map.md).

- [ ] Strength is the controlled variable across fixed edit cases.
- [ ] The prototype exposes successes and failures at every operating point.

## Train and Evaluate the Full-Data Monolith

**What to build:** Train, resume, validate, and neutrally evaluate a strong real-pair train_100 monolith
under the canonical experiment contract.

**Blocked by:** Establish Neutral Geometry Evaluation; Build a Real Monolith-Pair Slice.

**Status: done** — 3 runs (v1 unweighted, v2 surface-weighted, v3 x0-prediction). **v3 succeeded**:
generated occupancy 1.57% vs 1.66% real (matched). `monolith_v3` recommended as the reference
checkpoint. See map.md.

- [x] Training and validation behavior support using the checkpoint as an honest baseline.
- [x] Held-out results, failures, checkpoint identity, and compute budget are recorded.

## Generate and Evaluate the Full-Data Decomposition

**What to build:** Generate held-out Stage 3a massing plus train_100-only retrieved detail and evaluate
it with the same contract as the monolith.

**Blocked by:** Establish Neutral Geometry Evaluation; Build a Leakage-Safe Retrieval Slice.

**Status: PAUSED (2026-07-12), not blocked.** Fully scoped and code-verified (Stage3a SDEdit
massing + `propose_detail_ops` learned placement + `element_fit.retrieve` against
`element_library_train100_v1` + `EditableBuilding` CSG composition) but deprioritized by the
project owner in favor of the massing/shape-quality thread (ticket "Prototype the Sculpt Transform
Sweep", above). See map.md's "Pivot (2026-07-12)" section before resuming.

- [ ] Every output traces to allowed massing and element sources.
- [ ] Results use the same resolution, renderer, metrics, and failure policy as the monolith.

## Decide the Full-Data C2 Kill-Gate

**What to build:** Produce a documented continue-or-stop decision from the full-data comparison,
including uncertainty and failure localization.

**Blocked by:** Train and Evaluate the Full-Data Monolith; Generate and Evaluate the Full-Data Decomposition.

**Status: blocked** on "Generate and Evaluate the Full-Data Decomposition" (paused, see above).

- [ ] Detail fidelity and comparable massing fidelity are evaluated under the preregistered contract.
- [ ] A failed gate stops scaling work and identifies the next diagnostic question.

## Run the Equal-Data Scaling Curve

**What to build:** Run matched 25%, 50%, and 100% monolith and decomposition arms and estimate their
detail-data scaling trends.

**Blocked by:** Decide the Full-Data C2 Kill-Gate.

- [ ] Each fraction gives both arms exactly the same BuildingNet detail ids.
- [ ] Trends include uncertainty and do not overclaim behavior beyond observed data.

## Run the Massing and Element-Data Ablations

**What to build:** Report recipe-massing robustness and LoD3 element enrichment as separate ablations
without contaminating the headline comparison.

**Blocked by:** Decide the Full-Data C2 Kill-Gate.

- [ ] The recipe-massing arm changes only the massing source.
- [ ] LoD3 enrichment remains isolated from the equal-BuildingNet-data headline.

## Validate Detail Preference with Two-AFC

**What to build:** Create and validate a blinded, randomized two-AFC study and analyze collected
preferences under a declared protocol when participant recruitment is approved.

**Blocked by:** Decide the Full-Data C2 Kill-Gate.

- [ ] Pair sampling, blinding, randomization, consent, and analysis are fixed before collection.
- [ ] Human preference is reported as support for, not a replacement for, geometry metrics.

## Publish the Research Evidence Package

**What to build:** Generate the paper draft, related work, figures, tables, protocols, limitations,
negative results, and reproducibility references from recorded experiment artifacts.

**Blocked by:** Capture the Existing Residual Transform Evidence; Test the Fixed Detail-Scale Coincidence; Measure Transform Versus From-Noise Sampling; Prototype the Sculpt Transform Sweep; Run the Equal-Data Scaling Curve; Run the Massing and Element-Data Ablations; Validate Detail Preference with Two-AFC.

- [ ] Every manuscript claim and number traces to a recorded artifact.
- [ ] Core research claims, demo-wrapper features, limitations, and negative results are explicit.
