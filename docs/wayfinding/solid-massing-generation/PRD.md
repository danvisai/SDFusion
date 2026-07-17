# Spec: Solid, Footprint-Matching Massing Generation

PRD companion to the Wayfinder map [Solid, Footprint-Matching Massing Generation](https://github.com/danvisai/SDFusion/issues/24). Decisions here mirror that map's tickets; when a ticket resolves, reflect its decision back into this spec.

## Problem Statement

The Stage3a massing generator is supposed to turn a footprint into a base mass that reads as a building — in the user's words, "at least a solid block with some shape on top, matching the footprint, similar to all the data we have — not just BuildingNet." A substantial share of held-out output does not: some generations shatter into disconnected floating fragments, and many targets are thin shells / wireframes rather than solid blocks.

This was surfaced by the C2 composition kill-gate, whose diagnosis localized the failure to the **massing source**, not the downstream composition. Two concrete defects:

1. **A 26% empty-input collapse.** 73/277 held-out monolith generations are *byte-identical* near-empty grids, traced to a zero-occupancy coarse input (the low-pass SDF degenerates to empty); 335/1572 training pairs share this degenerate coarse input.
2. **A thin-shell data reality.** 56% of held-out buildings voxelize below 0.5% occupancy at 96³ — the very targets the model learns from are thin shells, so it cannot learn a solid block where the ground truth is a hollow wireframe.

When input and target are healthy, the pipeline already draws clean solid blocks (`outputs/transform_vs_noise/montage.png`) — so the problem is a degenerate tail, not the whole distribution.

## Solution

Two coordinated fixes to the shared massing data-target + coarse-input pipeline, gated by a measurable acceptance test:

1. **Give the generator solid targets.** Establish whether BuildingNet's low occupancy is a genuinely hollow mesh or a surface-vs-filled measurement artifact. If solidify-in-place works, filled solids become the target representation; otherwise (or additionally) onboard solid-massing corpora (PLATEAU / LoD2, the Japanese massing set, lod3_tum) so the training distribution spans all our data, not just BuildingNet.
2. **Never collapse on degenerate input.** Add a footprint-extrusion fallback coarse input at the single choke point so that when the low-pass SDF degenerates to empty, the model still receives real footprint signal — at both training and generation time.

"Done" is a locked acceptance gate: zero empty-input collapse, a solidity/occupancy floor, a footprint-IoU floor on genuinely held-out buildings, met per-corpus, and signed off visually on neutral-render montages.

## User Stories

1. As a town designer, I want the massing generator to turn my footprint into a solid block with a roof/shape on top, so that generated buildings read as buildings rather than floating debris.
2. As a town designer, I want the generated mass to fill and match my footprint, so that buildings sit correctly on their plots.
3. As a town designer, I want solid buildings across all classes (commercial, residential, religious, public), so that no class systematically breaks apart.
4. As a town designer, I want the generator to work on footprints similar to all our data, not just BuildingNet, so that coverage isn't limited to one corpus's quirks.
5. As a researcher, I want to know whether BuildingNet's thin-shell appearance is a hollow-mesh reality or a surface-vs-filled artifact, so that I choose the cheapest correct data fix.
6. As a researcher, I want a quantified before/after occupancy and fill-success measurement on a representative sample, so that the hollow-vs-artifact decision is evidence-backed, not assumed.
7. As a researcher, I want an inventory of solid-massing corpora with per-source building counts, footprint availability, scale conventions, and 96³ occupancy, so that I know how much solid data we can actually assemble.
8. As a researcher, I want each corpus's alignment cost to the Stage3a footprint-conditioning contract recorded, so that onboarding effort is predictable.
9. As a developer, I want a single choke point for the coarse input, so that the empty-input fallback attaches in exactly one place.
10. As a developer, I want the footprint-extrusion fallback applied at both training and generation time, so that train/test coarse-input distributions match and the model learns to use it.
11. As a developer, I want the fallback to trigger only when the primary coarse input degenerates (near-zero occupancy), so that healthy inputs are unaffected.
12. As a reviewer, I want the 26% collapse to drop to 0% on the known-collapsing held-out cases, so that the fix is verifiably effective.
13. As a reviewer, I want previously-collapsed cases to produce distinct, non-identical outputs after the fix, so that we've eliminated the byte-identical degenerate grid.
14. As a researcher, I want a measurable acceptance gate (collapse rate, solidity floor, footprint-IoU floor, per-corpus), so that "solid, footprint-matching" is testable rather than subjective.
15. As a reviewer, I want neutral-render montages as part of gate sign-off, so that visual quality is judged directly, not only via scalar metrics.
16. As a researcher, I want the gate evaluated on the genuinely held-out (Stage3a-clean) population, so that results aren't inflated by leaked training buildings.
17. As a developer, I want the target representation (filled occupancy vs filled SDF vs solidify-step) chosen once, so that data prep is consistent across corpora.
18. As a developer, I want the solidification method (winding-number, flood-fill, alpha-wrap) chosen against the hollow-vs-artifact evidence, so that we don't fill meshes that can't be filled reliably.
19. As the downstream composition stage, I want the improved massing to preserve the roof/shape-on-top, so that element placement still has a plausible base.
20. As a researcher, I want the retrain recipe (fine-tune vs from-scratch, data mix/curriculum, unified vs corpus-conditioned, compute budget, go/no-go checkpoints) settled before any training runs, so that expensive compute isn't spent on an undecided plan.
21. As the project owner, I want the actual retrain to remain a human hand-off, so that no autonomous agent launches training without a go decision.
22. As a reviewer, I want the fallback fix and the eval harness to be implementable now, independent of the retrain, so that we make verifiable progress before the big decisions close.
23. As a developer, I want the eval harness to report per-building occupancy and footprint IoU, so that regressions are attributable to specific cases.
24. As a researcher, I want the gate to distinguish "model failed to draw a block" from "target was a fragment," so that we don't penalize the model for faithfully reproducing thin-shell ground truth.
25. As a town designer, I want healthy footprints (which already yield clean blocks under blockout-SDEdit) to stay clean after the changes, so that the fix doesn't regress the cases that already work.
26. As a reviewer, I want the spec's decisions cross-linked to the map's tickets, so that anyone can see which decisions are settled and which are still open.
27. As a developer, I want tests that assert on external behavior (occupancy, IoU, collapse count) rather than internal tensor shapes, so that the tests survive refactors.
28. As a researcher, I want the fallback's necessity re-checked after the representation change, so that we don't keep a redundant patch if solid targets already remove empty inputs.
29. As the project owner, I want assets stored under the effort's own folder, so that this work stays isolated from Codex's element-retrieval thread and the carving spec.
30. As a reviewer, I want the acceptance gate to require passing on non-BuildingNet corpora specifically, so that "across all data" is enforced, not aspirational.

## Implementation Decisions

- **Target.** The Stage3a footprint-conditioned massing generator (the C1 transform / massing prior). The shared, improvable substrate is the massing data-target + coarse-input pipeline used by both the Stage3a prior and the monolith foundations code (ADR 0004's coarse-input scheme).
- **Two-pronged approach.** Solid targets (data/representation) + empty-input fallback (robustness). Root causes are established, not hypothesized: 26% byte-identical collapse from zero-occupancy coarse input (335/1572 train pairs); 56% of held-out below 0.5% occupancy at 96³.
- **Empty-input fallback — decided ([#29](https://github.com/danvisai/SDFusion/issues/29)): a residual safety net, not the primary fix.** #28's target solidification is what fixes the collapse — it breaks the byte-identical degeneracy (prototype: 28/29 distinct coarse, 26/29 non-empty). Keep a footprint-extrusion coarse input at the single choke point `build_monolith_pairs.low_pass_sdf` (mirrored in `generate_monolith_arm`) that triggers **only when `low_pass(solid_target)` still degenerates** (the residual ~10%, extreme tiny-footprint tail), at both train and generation time.
- **Non-building assets in the tail — decided ([#32](https://github.com/danvisai/SDFusion/issues/32)): mask, don't filter.** The near-empty tail is *mostly thin buildings* (median non-building face share ~0%), not trees/fences — but a ~10–15% minority carries ground/plant/fence, and BuildingNet's per-face labels (`face_labels`, all 1,849 meshes) let us mask it out. Form the solid target **and the footprint** from building-labeled faces only (drop ground 9 / road 19,23 / plant 5 / fence 13), and filter only meshes with negligible building mass after masking. This means #28/#29 must use a **building-only footprint** (the raw stored footprint includes the ground sheet). Separately generating mass vs. assets for a coherent scene is downstream composition, out of scope here.
- **Target representation — decided ([#28](https://github.com/danvisai/SDFusion/issues/28)).** Keep the SDF representation; solidify *only the training target* (no architecture/loss/eval change). Build the solid by **hybrid footprint-extrusion** — per-column occupancy extrusion (preserves the roof/shape-on-top) as primary, stored-footprint flat extrusion as the near-empty-tail fallback — **keep-centered lowest-to-top** (no phantom base), **precomputed** per building (mask→SDF via signed EDT) with QA montages. The solidify step attaches at data-prep after `render_facades.load_buildingnet_sdf`. Grounded in the [hollow-vs-artifact finding](https://github.com/danvisai/SDFusion/issues/25).
- **Corpora — decided ([#26](https://github.com/danvisai/SDFusion/issues/26)): the solid data already exists.** `data/real_massing_v1/real.h5` holds **35,776 real LoD2 buildings** (NL 11,776 + DE 12,000 + JP 12,000), genuinely solid (occ 13–30% vs BuildingNet 0.56%), already footprint-paired in the Stage3a format (`sdf` + `footprint` + `height_m` + `source_id`) — alignment cost ≈ 0. LoD2 needs **no** solidification; three cultures give style diversity. This reframes the whole effort: BuildingNet (+ #28/#29/#32) is an *optional shape-variety add-on*, not the massing source. Whether BuildingNet is in the mix at all — plus NL/DE/JP weighting and `source_id` conditioning — is the retrain-recipe call ([#30](https://github.com/danvisai/SDFusion/issues/30)).
- **Acceptance gate — decided ([#27](https://github.com/danvisai/SDFusion/issues/27)).** Aggregated over corpora, on generated Stage3a-clean held-out, building-only footprint: **collapse ≤ 1%** (`gen_occ < 1e-4`); **anti-fragmentation — ≥ 85% of outputs with largest 6-connected component ≥ 0.90**; **footprint-IoU median ≥ 0.65, p10 ≥ 0.35**; per-corpus reported as non-gating diagnostic; **neutral-render montage sign-off is the final arbiter, FID excluded** (visual-contradiction limitation). Fill-vs-envelope dropped (penalizes shape). Calibrated in `execution/artifacts/massing_gate_calibration.json`. Forcing function: only ~55–60% of current solidified targets clear the LCC bar → the solidify must be strengthened (feeds #30).
- **Retrain recipe — decided ([#30](https://github.com/danvisai/SDFusion/issues/30)).** **LoD2-only** (`real.h5`, 35,776 solid; BuildingNet + #28/#29/#32 deferred to an optional phase-2), **from-scratch** footprint-conditioned latent SDF diffusion, conditioning = footprint + class + height + style + **region/`source_id`** (drop era/floors). **Baseline-first go/no-go:** eval the deployed Stage3a on LoD2 held-out against the gate → stop if it passes; else from-scratch retrain (~100–150k step ceiling, gate eval every ~25k, early-stop on two flat checkpoints); final acceptance = #27 gate + visual sign-off. Full plan: `docs/wayfinding/solid-massing-generation/30-retrain-recipe.md`. Execution is a human hand-off, not autonomous.
- **Invariants.** Respect ADR 0004 (res 96³; s* = 1.0 m = 5 vox). The fallback must not change the working resolution or s*. No new behavioral seam: reuse the eval harness; the two fixes attach at existing internal seams.

## Testing Decisions

- **What makes a good test.** Assert on external behavior — generated occupancy, footprint IoU, collapse count, distinctness of previously-collapsed outputs — for held-out inputs, never on internal tensor shapes or intermediate representations.
- **Modules tested now.** The empty-input fallback (at the `low_pass_sdf` / `generate_monolith_arm` choke point) and the acceptance-gate harness (`eval_monolith` + `render_facades`). The solidification step gets tests once the representation ticket lands.
- **Seam.** Behavioral gating happens at the single eval-harness seam. The fallback additionally gets a focused unit test at its choke point: degenerate input → non-empty, distinct, footprint-matching coarse; healthy input → unchanged.
- **Prior art.** `scripts/foundations/test_generate_monolith_arm.py`, `test_build_monolith_pairs.py`, `test_eval_*`, `scripts/eval/test_fid.py`. Follow their leakage-safety and provenance conventions (git rev recorded; sealed-test-set isolation).
- **Guardrail.** The FID small-sample/high-dimension bias already guarded as `fid.undersampled` applies — the gate must not read FID below the effective-N floor.

## Out of Scope

- Launching or running the retrain — this spec plans it; execution is a human hand-off (the map is planning-only).
- The C2 composition/detail proof and its remediation (concluded on the Transform + Composition Research Proof map — different destination).
- Codex's element-retrieval thread (local ticket 02 / Phase R).
- The solid-first subtractive-carving specification (separate spec map).
- The blinded two-AFC detail-fidelity study (measures detail realism, not massing solidity).
- Downstream re-integration of improved massing into the composition/detail pipeline — revisit only after the massing gate passes.
- Facade detail, windows, ornament, materials, appearance — massing only.

## Further Notes

- **Readiness scoping — RESOLVED as the map closed.** The original scoping ("build the fallback + gate harness, stop before retraining") was superseded by #26/#30: the corpora audit found solid LoD2 data already exists, so the massing path is **LoD2-only** and the empty-input fallback + solidify machinery (#28/#29/#32) are **deferred to an optional BuildingNet phase-2** — not built now. Delivered instead: the **acceptance-gate harness** (`scripts/foundations/baseline_gate_eval.py`, tested) and the **from-scratch LoD2 retrain** (`retrain_prior_hybrid.py` adapted). The retrain was launched only with **explicit human approval** (never autonomously), consistent with the planning-only intent.
- This spec mirrors the map's tickets #25–#30; keep them in sync as decisions land.
- Assets live under `docs/wayfinding/solid-massing-generation/`.
