# Spec: Massing Surface Fidelity

PRD companion to the Wayfinder map [Massing Surface Fidelity](https://github.com/danvisai/SDFusion/issues/34); decisions mirror tickets #35–#37.

## Problem Statement

The massing generator (from the completed *Solid, Footprint-Matching Massing* effort) now produces solid, footprint-matching blocks — but their **surfaces are rough**: wavy, eroded, blobby, with no crisp flat walls or sharp roof planes, unlike the clean real LoD2 buildings. The user wants the generated massing to *look like crisp architecture*, not lumpy approximations — **without** losing the solidity and footprint match already won.

## Solution

Diagnosis localized the roughness to the **diffusion prior's sampled SDF field** — not the VQVAE codec, the 64³ resolution, or the mesh render (the VQVAE round-trip reproduces crisp GT). So fix it **prior-side, cheapest-first**:

- **Phase 1 (no retrain):** sampling-side probes — use the checkpoint's **EMA weights**, sweep **DDIM steps**, sweep **CFG guidance** — scored against the fidelity gate. Likely the whole fix, at zero training cost.
- **Phase 2 (only if Phase 1 falls short):** fine-tune (warm-start) with an **SDF-field smoothness regularizer**.

Success = **crisp surfaces on visual-montage sign-off AND the #27 massing gate still passing** (no regression).

## User Stories

1. As a town designer, I want crisp flat walls and sharp roof planes on generated massing, so that buildings look like clean architecture, not lumpy blobs.
2. As a town designer, I want the crispness fix to not make buildings break apart or drift off their footprint, so that I don't trade one problem for another.
3. As a researcher, I want the roughness source pinned before a fix is chosen, so that compute isn't spent on the wrong lever.
4. As a researcher, I want confirmation the VQVAE round-trip reproduces crisp GT, so that I can rule out a codec/resolution upgrade.
5. As a researcher, I want confirmation the roughness lives in the sampled SDF field, so that the fix targets the prior, not the render.
6. As a developer, I want the cheapest fixes tried first (no retrain), so that we don't spend days of GPU before a possible free win.
7. As a developer, I want inference to use the checkpoint's EMA weights, so that samples are softer/cleaner at zero training cost.
8. As a developer, I want to sweep DDIM steps at inference, so that I can trade compute for less sampling noise.
9. As a developer, I want to sweep the CFG guidance scale at inference, so that I can tune away noisy or over-sharpened output.
10. As a reviewer, I want each Phase-1 configuration scored against the gate plus a montage, so that I judge crispness and no-regression together.
11. As a researcher, I want the fidelity gate to be visual-primary, so that a metric that can't detect the roughness doesn't gate the work.
12. As a researcher, I want it recorded that two scalar crispness metrics empirically failed to separate crisp from rough, so that we don't re-litigate the metric.
13. As a reviewer, I want the #27 massing gate reused as an automatable no-regression guard, so that solidity/footprint-match can't silently regress.
14. As a developer, I want the Phase-1 knobs to flow through the existing eval harness, so that no new evaluation path is built.
15. As the project owner, I want Phase 2 (retrain) attempted only if Phase 1 falls short, so that compute is spent only when needed.
16. As a developer, I want the Phase-2 fix to be a warm-start fine-tune, not from-scratch, so that we keep the gate-passing massing and only smooth the field.
17. As a developer, I want the Phase-2 regularizer to smooth the field without erasing crisp edges, so that we don't blur the sharpness we want.
18. As a reviewer, I want a go/no-go at each phase against the fidelity gate, so that we stop at the first phase that works.
19. As a researcher, I want resolution to stay 64³, so that we don't onboard a new VQVAE / re-voxelize corpora for a problem that isn't resolution-bound.
20. As a town designer, I want crispness judged against real LoD2, so that the bar is "looks like a real building."
21. As a developer, I want the EMA-weight path verified (the checkpoint saves `ema_df`), so that "use EMA" is a real lever, not a no-op.
22. As a reviewer, I want the montage to compare generated vs real side by side, so that the sign-off is grounded.
23. As a developer, I want the fidelity work to build on the map-#24 checkpoint, so that we improve the shipped massing, not a fresh model.
24. As the project owner, I want the effort to hand off cleanly to implementation, so that the decisions become a buildable plan.
25. As a reviewer, I want the residual 64³ voxel-staircasing (present in real data too) distinguished from the prior's roughness, so that we don't chase a resolution-inherent artifact.
26. As a developer, I want the autoguidance path already in the model available as a guidance option, so that a weak-model guide can be tried if plain CFG underperforms.
27. As a researcher, I want the fix evaluated on genuinely held-out LoD2, so that crispness isn't inflated by training overlap.
28. As a town designer, I want crispness across the distribution, not cherry-picked buildings, so that the improvement is real.

## Implementation Decisions

- **Target.** The map-#24 from-scratch LoD2 checkpoint (the gate-passing massing model). This effort improves its surface fidelity in place.
- **Root cause ([#35](https://github.com/danvisai/SDFusion/issues/35)).** The roughness is in the diffusion prior's **sampled SDF field** — the VQVAE round-trip reproduces crisp GT (IoU 0.995) and the field itself is noisy for samples. **Not** the codec, **not** 64³ resolution, **not** the mesh render. The higher-resolution / new-VQVAE path is **ruled out**; resolution stays 64³.
- **Fidelity gate ([#36](https://github.com/danvisai/SDFusion/issues/36)).** **Visual-montage sign-off is the primary/required arbiter** — no scalar geometry metric reliably separates crisp from rough (two normal-consistency metrics empirically failed, both ~0.99, because the roughness is mid-scale waviness on locally-smooth decoded fields). The **#27 massing gate** is the automatable **no-regression guard**. A global normal-concentration scalar *may* be added as a non-gating diagnostic.
- **Fix plan ([#37](https://github.com/danvisai/SDFusion/issues/37)) — cheapest-first, phased.**
  - **Phase 1 — sampling-side, no retrain:** (i) **EMA-weight inference** — the checkpoint saves `ema_df`, but inference currently uses raw weights, so enable EMA at inference (the build/use path may be train-gated — flipping it is the work); (ii) **DDIM-step sweep** (100 → 250/500); (iii) **CFG guidance-scale sweep** via the existing `unconditional_guidance_scale` (the autoguidance path is also available). Score each against the fidelity gate; ship the first crisp configuration with no regression.
  - **Phase 2 — only if Phase 1 short:** warm-start fine-tune with an **SDF-field smoothness regularizer** on the predicted x0 (gradient-TV / eikonal-deviation, small weight so it smooths without erasing edges); gate-checkpointed, early-stop.
- **Modules.** Phase-1 knobs are exposed through the eval harness's opt → the model's inference; Phase 2 attaches at the training loss (the model / the retrain entry). **No new evaluation path — reuse the existing gate harness.**
- **Go/no-go.** The fidelity gate at each phase; stop at the first that passes.

## Testing Decisions

- **What makes a good test.** Assert external behavior — the #27 no-regression metrics (occupancy/LCC/footprint-IoU) and that the sampling knobs actually take effect (EMA weights loaded and used; guidance scale applied) — not internals.
- **Modules tested.** The eval harness (`score_gate`/`per_corpus_diagnostics` already covered) plus the new knob plumbing (`use_ema` swaps in the checkpoint's `ema_df` at inference; guidance scale flows to inference).
- **Not auto-tested.** The visual crispness gate is human sign-off; the automatable surface is the #27 no-regression check + the knob plumbing.
- **Prior art.** `test_baseline_gate_eval.py` (synthetic, no-GPU) for the pure logic; a tiny model-load smoke for the behavioral knob checks.
- **Phase 2 (if reached).** Test the regularizer term is finite and its weight is applied; verify via a short smoke run.

## Out of Scope

- Higher resolution / new VQVAE / corpora re-voxelization — ruled out by the diagnostic (the roughness isn't resolution-bound).
- Architectural detail generation (windows, doors, facades, cornices) — a separate downstream layer.
- Appearance (textures, materials, lighting, neural render) — downstream.
- Regressing or re-opening the map-#24 massing solidity/footprint behavior — a hard no-regression constraint, not a target.
- The residual 64³ voxel-staircasing present in real data too — resolution-inherent, not this effort's roughness.

## Further Notes

- **`ready-for-agent` scope.** Phase 1 is the buildable-now surface (EMA inference + DDIM/guidance sweeps + gate/montage eval). **Phase 2 (retrain) is gated on Phase 1 falling short and on a human go decision — not autonomous.**
- Builds on map #24's shipped checkpoint; reuses its gate harness (`scripts/foundations/baseline_gate_eval.py`).
- **First concrete step:** wire EMA-weight inference and re-run the gate + montage — quite possibly the whole fix at zero training cost.
