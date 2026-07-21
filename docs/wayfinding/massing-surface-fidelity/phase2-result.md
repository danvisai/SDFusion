# Phase-2 Result — smoothness fine-tune FALLS SHORT; both levers negative

Execution record for the [Surface-Fidelity fix plan](https://github.com/danvisai/SDFusion/issues/37)
Phase 2 (warm-start fine-tune + SDF-field smoothness regularizer). Reached because Phase 1's sampling
knobs fell short ([phase1-result.md](phase1-result.md)). Warm-started from the map-#24 checkpoint
(`logs_building/2026-07-16-stage3a-lod2-fromscratch-region/...steps-latest.pth`), LoD2 `real.h5`,
`--use_region 1 --use_extra_cond 0`, constant lr 2e-5, guardrail on. Evaluated on the **raw** fine-tuned
weights (the shipped-EMA path had a warm-start bug, since fixed — see below) with the existing gate harness.

## Both PRD-named smoothness forms fail

| run | kind | weight | iters | fp-IoU median | fp-IoU p10 | #27 gate | crisp? |
|---|---|---|---|---|---|---|---|
| base (map #24, raw) | — | — | — | 0.835 | 0.70 | ✅ | no (baseline) |
| grad_tv | Laplacian/curvature | 0.4 | 2000 | 0.74 | 0.40 | ✅ (weakened) | **no** |
| eikonal | (\|∇\|−1)² | 0.1 | 1200 | **0.45** | 0.33 | **❌ FAIL** | **no** |

Artifacts: `execution/artifacts/baseline_gate_eval_ftsmooth-*.json`; montages
`outputs/baseline_gate_eval/montage_ftsmooth-*-raw.png`.

## Why each fails (visual + mechanistic)

- **grad_tv (curvature penalty).** No crispness on the montage — it **rounds** the shapes (footprint and
  roof edges included), dropping fp-IoU 0.835 → 0.74 while the faces stay wavy. Mechanistically a
  curvature penalty cannot tell *bad waviness on a face* from a *good sharp edge*: it removes both. It is
  **directionally wrong** — "crisp architecture" needs sharp edges, which smoothness erases.
- **eikonal (metric \|∇\|=1 penalty).** The eikonal term **never decreased** across the run (0.944 → 0.941):
  the decoded field's gradient magnitude is fixed by the **frozen VQVAE decoder** (truncated SDF), so the
  prior/latent cannot move it. What the eikonal gradient *did* do was push the geometry off-footprint —
  fp-IoU collapsed to 0.45 (**gate FAIL**), montage shows inflated cauliflower blobs. Worst of both: no
  crispness, regressed gate.

## Warm-start EMA bug (found & fixed, commit `72df568`)

`ema_df` is deepcopied in `initialize()` **before** `load_ckpt`, so a warm-start fine-tune's EMA shadow
started from RANDOM init — ~`decay^iters` (0.999^2000 ≈ 14%) of random weights contaminate the EMA the
checkpoint ships (and inference loads by default). Fixed by re-syncing `ema_df` to the loaded weights after
a warm-start load; committed gated test `TestWarmStartEma`. Evals above used raw weights, so the bug does
not affect these results — but it would have corrupted any shipped checkpoint.

## Go / no-go — the map's cheapest-first plan is exhausted

**Phase 1 (sampling knobs) and Phase 2 (smoothness fine-tune) both fall short.** The planned cheapest-first
levers do not produce crisp surfaces. Root reason, consistent with [#35](https://github.com/danvisai/SDFusion/issues/35):
the roughness is a wavy level-set in the **sampled latent decoded through the frozen VQVAE**; a
field-*smoothness* objective on the decoded output is the wrong tool — it rounds (grad_tv) or distorts
(eikonal), never sharpens. Eikonal also exposed that the frozen truncated-SDF decoder caps field quality.

**Recommendation (a re-scope beyond this map — needs a human go):**
1. **Edge-aware, not smoothness.** Target crispness directly — a piecewise-planar / normal-clustering
   objective that *rewards* flat faces + sharp creases (not a curvature penalty). Not a lever this map scoped.
2. **Revisit the frozen codec.** The eikonal cap suggests the truncated-SDF VQVAE decoder — not the prior —
   may be the fidelity ceiling; #35 ruled out *higher-res*, but not a *different/edge-preserving* codec.
3. **Accept + defer.** The map-#24 massing already passes the #27 solidity/footprint gate; crispness could be
   handled downstream (post-hoc plane-fit / mesh cleanup, or masked by the appearance layer).

The massing-surface-fidelity map does **not** reach crisp surfaces via its planned levers. Closing it here
with this negative result; option (1)/(2) would be a new effort.
