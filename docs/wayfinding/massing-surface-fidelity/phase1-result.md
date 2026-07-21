# Phase-1 Result — sampling knobs FALL SHORT → Phase-2 decision needed

Execution record for the [Surface-Fidelity fix plan](https://github.com/danvisai/SDFusion/issues/37) Phase 1
(no-retrain sampling probes). Harness: `scripts/foundations/baseline_gate_eval.py` on the map-#24
checkpoint `logs_building/2026-07-16-stage3a-lod2-fromscratch-region/ckpt/stage3a_steps-latest.pth`
(`--use_region 1 --use_extra_cond 0`), n=16 LoD2 held-out, seed 0.

## Correction to the plan: EMA was **already on**

The fix plan named EMA-weight inference as "the likely single biggest free win," assuming inference used
raw weights. It does not: `Stage3aModel.load_ckpt` already swaps in the checkpoint's `ema_df` for any
inference load (`isTrain=False`), and the map-#24 gate + the #35/#36 roughness diagnosis were **all run
with EMA weights active**. So EMA is not a new lever — it is the deployed default. This effort's plumbing
change makes EMA a *toggle* (`--use_ema 0/1`) so the raw config can be scored for comparison.

## The sweep (all configs PASS the #27 no-regression gate)

| tag | weights | ddim | guidance | fp-IoU median | fp-IoU p10 | collapse | LCC≥0.90 | #27 |
|---|---|---|---|---|---|---|---|---|
| A (deployed default) | EMA | 100 | 1.0 | 0.847 | 0.772 | 0% | 100% | ✅ |
| C (max steps) | EMA | 500 | 1.0 | 0.841 | 0.761 | 0% | 100% | ✅ |
| D (guidance up) | EMA | 100 | 2.0 | **0.892** | **0.826** | 0% | 100% | ✅ |
| F (raw weights) | raw | 100 | 1.0 | 0.835 | 0.704 | 0% | 100% | ✅ |

Artifacts: `execution/artifacts/baseline_gate_eval_p1-*.json`; montages
`outputs/baseline_gate_eval/montage_p1-*.png`.

## Visual verdict (the #36 primary arbiter): NOT crisp

No config produces crisp flat walls or sharp roof planes. On the montages:

- **DDIM 100 → 500 (A vs C):** surfaces are indistinguishable in roughness. 5× more steps does not smooth
  the waviness — **direct confirmation the roughness is learned prior-side field structure, not sampling
  noise** (which more steps would reduce). Matches #35.
- **Guidance 1.0 → 2.0 (A vs D):** tightens the footprint outline (best fp-IoU, 0.892) but **amplifies**
  the vertical striations rather than removing them. Sharper guidance ≠ crisper surface.
- **EMA vs raw (A vs F):** near-identical surfaces (mean-abs SDF diff 0.010); EMA is marginally better on
  fp-IoU, confirming it as the right default. Raw is not crisper.

## Go / no-go

**Phase 1 falls short.** All cheap sampling levers hold the #27 gate but none reaches crisp visual
sign-off. Per the #37 go/no-go, the next step is **Phase 2 — warm-start fine-tune of the 120k checkpoint
with an SDF-field smoothness regularizer** (gradient-TV / eikonal-deviation, small weight; gate-checkpointed,
early-stop). Autoguidance (a weak-model guide) was left untried: the CFG evidence (higher guidance amplified
striations) predicts a sharpening guide would worsen, not smooth, the waviness — so it is not recommended
ahead of Phase 2.

**Phase 2 is training-side and, per the PRD `ready-for-agent` scope, is gated on an explicit human go
decision — it is not autonomous.** This record is the hand-off for that decision.
