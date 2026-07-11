# Research proof plan — transform + composition, not data-scaling (2026-07-10, rev3)

Two-claim thesis (`docs/adr/0003-two-claim-thesis.md`), experiment design (`docs/adr/0002`), original
factorization (`docs/adr/0001`). Vocabulary: `CONTEXT.md`. Companions: `DATA_PREP_PLAN`,
`IMPLEMENTATION_PLAN`, `PAPER_DRAFT`, `RELATED_WORK`.

## 0. Thesis
Generality comes from **transform + composition, not data scale.** Two falsifiable claims:
- **C1 (transform, not generate)** — never sample from noise; *project* a rough input onto the
  real-building manifold via SDEdit; the same projection does generation and editing; it recovers
  massing, not detail.
- **C2 (compose, not synthesize)** — detail is ill-posed to generate at achievable data scale, so it is
  composed/retrieved from real elements; this beats monolithic detail-generation at equal data.

---

## 1. C1 experiment — transform, not generate
**C1a. Transform ≫ from-noise.** Sample the Stage 3a prior two ways on the held-out footprints:
(a) unconditional/low-info **from noise**; (b) **SDEdit from a footprint blockout** (footprint-extrude,
partial-noise, denoise). Metric: **rendered FID** vs real (same neutral shader as C2). Predicted:
FID(from-noise) ≫ FID(SDEdit-blockout) — quantifies the "degenerate from noise" finding.
**C1b. Editing = the same transform.** A `strength` sweep on a crude user edit (e.g. tower-spike →
coherent tower): report faithfulness (IoU to the imposed edit) ↔ realism (FID) as one curve, showing a
single operator spans generation↔editing. Qualitative sculpt sheets + the quantitative curve.
**C1c. Transform aligns massing, not detail (bridge to C2).** Cite the residual-correction result
(`Logs_GT/sdf_residual_full_v4_aug_topk3`): footprint-IoU **0.999** vs detailed-shape-IoU **~0.13–0.28**
— a second, independent transform (retrieval + residual UNet) recovers massing/footprint but not detail.

## 2. C2 experiment — compose, not synthesize (the data-scaling curve)
Unchanged from rev2 (D1–D4). Summary:
- **Monolith (baseline):** one SDF net, footprint → detailed SDF, trained on **REAL** (coarse-massing →
  BuildingNet-detail) pairs — NOT synthetic `detail_pairs_v1`.
- **Decomposition (headline):** **Stage 3a massing** (full data) + detail **retrieved** from the element
  library. Robustness ablation: recipe-param → procedural massing + retrieval.
- **Equal-data rule:** at fraction *X*, both arms see exactly `train_X` of BuildingNet detail (monolith
  trains on `train_X` pairs; the library is **rebuilt from `train_X` ids**); massing stays full.
- **Metric:** massing → paired (Chamfer/IoU); **detail → rendered-facade FID** (neutral shader) + 2AFC.
- **Curve:** (monolith, decomposition) × (25/50/100 %). Predicted: monolith poor + slowly improving;
  decomposition good + gracefully degrading; the monolith slope, extrapolated, answers "just get more
  data" (no external giant — dropped, D4).

## 3. The metric split mirrors the thesis
Massing is determined by the footprint ⇒ **paired** (Chamfer/IoU). Detail is underdetermined ⇒
**distributional** (FID). Working resolution **≥ 96³**, identical neutral shader across all arms.
`s*` (≈0.5 m) is **fixed a priori**; the scale-spectrum measurement *tests* the massing/detail coincidence.

## 4. Feasibility — have vs. gaps
**Have:** Stage 3a prior (C1a/C1b) + SDEdit; residual result (C1c, done); element-library build script,
composer, render pipeline, monolith harness (C2).
**Gaps:** held-out split; **from-noise vs blockout sampling harness** (C1a); **sculpt strength-sweep eval**
(C1b); FID + facade-render harness; REAL massing→detail pairs; per-fraction leakage-safe libraries;
monolith runs; `s*` coincidence measurement; 2AFC study.

## 5. Sequencing (fail-fast)
- **Step 0:** held-out split + leakage-safe library + FID harness; `s*` coincidence in parallel.
- **Step 1 (C1, cheap, high-signal):** C1a from-noise-vs-blockout FID + C1b sculpt sweep — uses the LIVE
  prior, no training. Establishes the transform half early.
- **Step 2 (C2 kill-gate):** 100 % cell — monolith@100 vs decomposition. Stop if it loses.
- **Step 3:** fill 25/50 % → the curve; then recipe-massing ablation + 2AFC study.

## 6. Where the demo fits (not the proof)
The peripheral edit features (weathering, ornaments, sketch-relief, recipe-closure round-trip) are the
**demo wrapper** — qualitative existence-proof of an editable town. C1 (transform/sculpt) is **not**
wrapper — it is the core mechanism the paper proves.

## 7. Risks
Weak-baseline (real-pairs monolith, matched compute); coincidence may fail (`s*` fixed a priori → honest
negative); fair rendering (identical neutral shader); leakage (per-fraction split + library exclude test);
C1a fairness (the "from-noise" baseline must be the *best honest* unconditional sample, not a strawman).
