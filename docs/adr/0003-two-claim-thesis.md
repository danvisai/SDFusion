# Two-claim thesis: transform + composition, not data-scaling

**Status:** accepted (2026-07-10) — expands `0001-massing-scale-decomposition.md`

Re-reading the whole codebase and the June-2026 professor report showed the project's true thesis is
broader than 0001's "massing-scale decomposition." The unifying claim (from the professor report):
**generality comes from *transform* + *composition*, not from data scale** — you can never enumerate
every building, so instead of scaling data you (i) transform any rough input onto the manifold of real
buildings and (ii) compose it from understood architectural elements. Two falsifiable claims:

## C1 — transform, not generate
You never sample a building from noise (degenerate at achievable data scale). Instead you **project** a
rough input onto the real-building manifold via **SDEdit**. The *same* projection is **generation** (from
a footprint blockout) **and editing** (from a user sculpt) — therefore **editability is core, not a
wrapper**. Transform recovers **massing**, not detail.
- **Experiment:** (i) `FID(SDEdit-from-footprint-blockout) ≪ FID(sample-from-noise)`; (ii) a sculpt
  faithfulness↔realism `strength` sweep (crude edit → coherent building); (iii) the residual-correction
  datapoint as evidence transform aligns massing not detail: `val_corrected_fp_iou ≈ 0.999` vs
  `val_corrected_iou ≈ 0.13–0.28` (`Logs_GT/sdf_residual_full_v4_aug_topk3`).

## C2 — compose, not synthesize (the 0001 claim, unchanged)
Detail is ill-posed to *generate* at achievable data scale, so it is **composed/retrieved from real
elements**, which beats monolithic detail-generation at equal data — proven by the data-scaling curve
with the D1–D4 refinements (`0002-experiment-design.md`).

## Consequences
- **Editability re-elevated:** the SDEdit **transform** (snap/sculpt) is **C1 core**, NOT the demo
  wrapper. The *peripheral* edit features (weathering, ornaments, sketch-relief, recipe-closure
  round-trip) remain the demo wrapper.
- **Two experiments now** (C1 + C2). The residual-correction pipeline (`train_sdf_residual.py`,
  `HYBRID_PIPELINE_PLAN.md`) and the never-from-noise finding become **C1 evidence**, not orphaned work.
- **Deliverables:** `execution/PAPER_DRAFT.md` + `execution/RELATED_WORK.md` + the reconciled
  `execution/` plans (rev3, two-claim).
