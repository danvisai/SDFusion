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
- [Lock the Experiment Operating Point](issues/01-lock-experiment-operating-point.md) - `docs/adr/0004`: res 96³ (preflight escape → 128³); `s*` = 1.0 m = 5 vox @96³ tied to the 64³ massing-grid limit (corrects the loose ≈0.5 m; propagated to CONTEXT + ADRs 0001/0002 + plans); monolith coarse input = low-pass primary + footprint-extrude variant. Unblocks tickets 05, 06.
- [Build the Neutral Facade and FID Harness](issues/05-build-neutral-facade-fid-harness.md) - `scripts/eval/{fid,render_facades,sanity_real_vs_real}.py` (TDD, 22 contract tests). Fixed a real 64³-vs-96³ resolution-parity bug (`resample_sdf_grid`), pinned extractor provenance, group-aware bootstrap for correlated views, plus two bugs CAUGHT DURING VERIFICATION (a `--res` conflation between SDF voxel grid and image pixel size; a BLAS thread-oversubscription stall, 121 threads/40 cores, fixed via capped `OMP/MKL/OPENBLAS_NUM_THREADS`). Discovered and permanently guarded (`fid.undersampled`, warns) a genuine FID small-sample/high-dimension bias: at 48 buildings/144 images (< the 2048-d feature dim) the point estimate falls outside its own bootstrap CI, confirmed via synthetic data — not a bug, but tickets 07/08's headline FID needs substantially more samples than this sanity scale. Rendering is arm-agnostic by construction, ready for the monolith/decomposition arms.
- [Test the Fixed Detail-Scale Coincidence](issues/06-test-detail-scale-coincidence.md) - `scripts/eval/measure_scale_spectrum.py` (TDD, 12 contract tests incl. a synthetic-geometry integration test), run over all 1,572 `train_100` buildings (0 parse failures). **Result: 6/11, `partial_coincidence`.** Massing (wall, roof) lands cleanly above `s*`; thin facade-articulation detail (window, door, column, chimney) lands cleanly below it; but the discrete ADD-element vocabulary ticket 04 already adopted (tower, balcony, balcony_upper, stairs, dome) lands *above* `s*` — large in absolute scale despite being compositionally "detail". Honest finding, boundary not moved: `s*` (voxel-Nyquist representability) and "generatable vs. composable" are correlated, not identical, axes — the ADD elements are composed for being too *varied/rare* to generate, not too *small* to resolve. Does not weaken C2; reframes how the paper should cite this coincidence for the ADD-element categories specifically.
- [Measure SDEdit Transform Versus From-Noise Sampling](issues/09-measure-transform-versus-noise.md) - `scripts/eval/transform_vs_noise.py` (TDD, 7 contract tests for the pure seams). **Leakage catch first:** `data/splits_v1/test.json` is a different partition than Stage3a's own original training split — 224/277 of ticket 03's "held-out" ids were actually gradient-trained on; only 27 were genuinely never seen, and those 27 are what got evaluated. **Result over the 27, 0 failures: C1 holds.** Footprint IoU (primary massing metric) roughly doubles at the median, from-noise 0.30 -> blockout-SDEdit 0.61; full-volume IoU moves the same direction more modestly (several buildings have thin/sparse real ground truth, disclosed per-building, not filtered — the occupancy distribution is continuous with no natural cutoff). FID is honestly inconclusive at this n (both arms `undersampled=true`, matching ticket 05's known finding) and is not leaned on. Qualitative montage confirms it: from-noise samples are consistently lumpy/melted blobs, blockout-SDEdit samples are consistently more rectilinear and building-like.
- [Build Real Full-Data Monolith Pairs](issues/07-build-real-monolith-pairs.md) - `scripts/foundations/build_monolith_pairs.py` (TDD, 16 contract tests). Coarse input = ADR 0004's locked `low_pass_sdf` primary: resample the real target SDF down to a grid whose voxel pitch matches the fixed `s*` (19³) and back up to 96³ via the same `resample_sdf_grid` every other ticket uses — no new interpolation path, no new hyperparameter, source/target stay aligned by construction (coarse is derived from the loaded target itself). Axis convention independently verified against BuildingNet's own stored footprint field (`footprint_alignment_iou`: axis=1/H-up gives IoU=1.0 on real data, axis=0/2 give ~0.06). Full `train_100` build: **1572/1572 pairs, 0 failures, 0 leakage**, footprint-axis IoU mean/min = 1.0/1.0 across all buildings, class balance exact match to ticket 03's frozen `train_100` counts. No SDF grids duplicated to disk — only ids + provenance + diagnostics (`data/monolith_pairs_v1/{manifest,pairs,per_pair}.json`), real targets load on demand from the existing BuildingNet H5s. Montage (`outputs/monolith_pairs_v1/montage.png`) confirms the low-pass qualitatively strips thin sub-s* spires/ornament while the broad mass survives. Unblocks ticket 11.

## Not yet specified

- The remediation branch if the 100% C2 kill-gate fails. Its question depends on whether the failure
  comes from retrieval quality, massing mismatch, rendering bias, or a genuinely stronger monolith.
- The interpretation and paper narrative if the fixed-a-priori `s*` coincidence test fails.
- Exact training schedules and compute allocation beyond the 100% kill-gate; derive these from the
  measured throughput, memory use, and validation behavior of the first full-data run.
- Which qualitative cases and failure examples deserve final paper figures; select only after the
  quantitative results expose the representative regimes.
- Minimum sample count for a trustworthy headline FID (ticket 05 finding: N must substantially exceed
  the 2048-d Inception feature dimensionality — the 48-building/144-image sanity scale is provably
  too small). Concrete views-per-building / test-set-fraction budget for tickets 07/08 is undecided.

## Out of scope

- General demo polish, stale demo-bundle rebuilding, weathering preview work, ornament-library growth,
  and deferred disk cleanup. They do not establish C1 or C2.
- Phase G crop-inpainting as a product feature. Revisit only if an experiment result makes a tightly
  scoped generative-element ablation necessary.
- A Hunyuan3D-2 comparison: ADR 0002 rejects it because it cannot take the task's footprint input.
