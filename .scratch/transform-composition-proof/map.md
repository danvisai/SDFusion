# Transform + Composition Research Proof

## Destination

Produce a reproducible, paper-ready evidence package that supports or falsifies both claims in
`docs/adr/0003-two-claim-thesis.md`: rough-input SDEdit beats from-noise generation for massing (C1),
and retrieved composition beats a real-pair monolith for detail at equal data (C2).

## Notes

- This map deliberately carries execution through the experiments and paper artifacts; the user
  asked to continue Claude's accepted July 10 research work and go ahead, overriding wayfinder's
  planning-only default.
- Read `CONTEXT.md` for project language and `docs/adr/0001-massing-scale-decomposition.md`,
  `docs/adr/0002-experiment-design.md`, and `docs/adr/0003-two-claim-thesis.md` for accepted decisions.
- The reconciled source plans are `execution/RESEARCH_PROOF_PLAN_2026-07-10.md`,
  `execution/DATA_PREP_PLAN_2026-07-10.md`, and `execution/IMPLEMENTATION_PLAN_2026-07-10.md`.
- The canonical agent-ready specification is `PRD.md`; child tickets remain the execution and
  dependency record.
- Preserve the existing uncommitted Phase R quality changes in `scripts/server/element_fit.py`,
  `scripts/server/inference_service.py`, `scripts/server/layout_detail.py`, and
  `scripts/server/refine.py`; they predate this map and are inputs to the experiment baseline.
- Never admit held-out test BuildingNet ids to training pairs or retrieval libraries. All arms use
  the same working resolution and neutral facade renderer. Detail is evaluated distributionally;
  massing is evaluated against its paired target.
- Resolve no more than one ticket per wayfinder session. Refer to tickets by linked title, not number.

## Decisions so far

<!-- Closed-ticket pointers are appended here. Accepted ADRs are context, not synthetic tickets. -->

- [Validate the Element Retrieval Baseline](issues/02-validate-element-retrieval-baseline.md) - Phase R now enforces bounded, solid retrieval and preserves ordered recipe edits through analytic output-resolution realization; CPU gates pass, with full CUDA suites recorded as environment-blocked. (Full CUDA branch + sculpt-flow suites since re-verified GREEN 8/8+11/11+19/19 on a fresh server from the committed code — 2026-07-10.)
- [Freeze Leakage-Safe BuildingNet Splits](issues/03-freeze-buildingnet-splits.md) - `scripts/foundations/make_splits.py` (TDD, 10 contract tests green) freezes a deterministic class-stratified sealed test set + nested train_25 ⊂ train_50 ⊂ train_100 (277 / 392 / 785 / 1572 over 1849) at `data/splits_v1/` (seed 0). Unblocks ticket 04.
- [Make the Element-Library Builder Leakage-Safe](issues/04-make-library-builder-leakage-safe.md) - `build_element_library.py` gains `--include-ids`/`--exclude-ids` (exclude wins), `--out`/`--no-qa`, and a `manifest.json` leakage audit (asserted empty). TDD 7/7 + integration proof: zero test contributors, byte-identical re-builds, and a removal proof (test buildings that contribute when included are absent when excluded). Unblocks tickets 05, 08.

## Not yet specified

- The remediation branch if the 100% C2 kill-gate fails. Its question depends on whether the failure
  comes from retrieval quality, massing mismatch, rendering bias, or a genuinely stronger monolith.
- The interpretation and paper narrative if the fixed-a-priori `s*` coincidence test fails.
- Exact training schedules and compute allocation beyond the 100% kill-gate; derive these from the
  measured throughput, memory use, and validation behavior of the first full-data run.
- Which qualitative cases and failure examples deserve final paper figures; select only after the
  quantitative results expose the representative regimes.

## Out of scope

- General demo polish, stale demo-bundle rebuilding, weathering preview work, ornament-library growth,
  and deferred disk cleanup. They do not establish C1 or C2.
- Phase G crop-inpainting as a product feature. Revisit only if an experiment result makes a tightly
  scoped generative-element ablation necessary.
- A Hunyuan3D-2 comparison: ADR 0002 rejects it because it cannot take the task's footprint input.
