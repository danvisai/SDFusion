# Surface-Fidelity Fix Plan & Go/No-Go — cheapest-first, prior-side

Resolves [Settle the surface-fidelity fix plan and go/no-go](https://github.com/danvisai/SDFusion/issues/37). Per #35 the roughness is in the diffusion prior's sampled SDF field; per #36 the gate is **visual-montage-primary + #27 no-regression**. Resolution stays **64³** (#35 ruled out the codec/resolution as the cause).

## Phase 1 — sampling-side, no retrain (cheapest; hours)

Probe these levers, evaluating each against the #36 gate (`baseline_gate_eval.py --ckpt … --tag …` for the #27 no-regression check + a visual montage):

- **EMA weights.** Inference currently uses the **raw** weights; the EMA copy (`ema_df`) is documented as "softer/cleaner samples" — the likely single biggest free win. Load EMA for inference.
- **DDIM steps.** Sweep 100 → 250/500 (more steps → less sampling noise).
- **CFG guidance scale.** Sweep it (the unconditional branch is trained via `p_uncond=0.1`).

**Ship if any combination reaches crisp visual sign-off with the #27 massing gate intact — no retrain.**

## Phase 2 — training-side, only if Phase 1 falls short

- **Fine-tune** (warm-start the 120k checkpoint, *not* from scratch) with an **SDF-field smoothness regularizer** on the predicted x0 — gradient-TV / eikonal-deviation, small weight, so it smooths the field without erasing crisp edges.
- Gate-checkpointed, early-stop on the #36 gate.

## Go / no-go

The #36 gate (visual crisp sign-off + #27 no-regression) at each phase; **stop at the first phase that passes.** Execution is the hand-off.

Entry points: `models/stage3a_model.py` (`ema_df`, guidance) for Phase-1 sampling; `scripts/foundations/baseline_gate_eval.py` for the no-regression check; `scripts/foundations/retrain_prior_hybrid.py` (warm-start) for Phase 2.
