# Retrain Result — GATE PASSED, effort complete

The #30 from-scratch LoD2 retrain executed to completion and **passed the #27 acceptance gate**, including the human visual sign-off (2026-07-20). Map #24's destination — *solid, footprint-matching massing* — is reached.

## The run

`logs_building/2026-07-16-stage3a-lod2-fromscratch-region` — from-scratch, LoD2-only (`real.h5`, 34,346 train), region-conditioned (`--use_region 1`), era/floors dropped (`--use_extra_cond 0`), bs 16, lr 1e-4, p_uncond 0.1. **120,000 iterations in ~193,265 s (~53.7 h).** Loss 1.08 → **0.04**; the footprint-loss component fell **~0.40 → ~0.12** — the direct signal of improved footprint adherence.

## Gate: baseline vs retrained (60 LoD2 held-out)

| #27 criterion | Deployed baseline | **Retrained (120k)** | Floor |
|---|---|---|---|
| Collapse | 0% | **0%** | ≤1% ✓ |
| Anti-fragmentation (LCC≥0.90 share) | pass | **pass** | ≥85% ✓ |
| Footprint-IoU median | 0.43 ❌ | **0.89** ✓ | ≥0.65 |
| Footprint-IoU p10 | 0.28 ❌ | **0.80** ✓ | ≥0.35 |
| Visual montage sign-off (final arbiter) | blobs ❌ | **solid footprint-matching blocks ✓** | — |
| **Overall** | **FAIL** | **✅ PASS** | |

Held-out is clean for the retrain (trained on `real.h5` phase=train, scored phase=test — same permutation). Sanity: real-footprint self-IoU = 1.0.

Artifacts: `execution/artifacts/baseline_gate_eval_lod2-final.json`, `outputs/baseline_gate_eval/montage_lod2-final.png`. Harness: `scripts/foundations/baseline_gate_eval.py --ckpt <…>/stage3a_steps-latest.pth --use_region 1 --use_extra_cond 0`.

## The through-line, confirmed

"Breaking apart" was a BuildingNet-thin-shell artifact (#25/#26); the generator already had 35k solid footprint-paired LoD2 buildings. Retraining LoD2-only from scratch (dropping the deployed 20k hybrid checkpoint's recipe dilution, adding a culture token) turned solid-but-**blobby** output into solid-**footprint-matching** output — no BuildingNet solidify machinery (#28/#29/#32) needed.

## Open quality note (not a gate failure)

Generated surfaces are **rough/noisy** vs crisp real LoD2 (vertical striations, soft roof detail). This is a *fidelity/appearance* refinement, distinct from the massing-solidity gate that is now met — a candidate for a separate effort, not part of this map.

## Not exercised

Per-checkpoint trajectory evals (10k…110k) were not run; the final checkpoint passing was sufficient. If wanted, the tagged harness (`--tag`) can gate-eval each to find where the gate was first crossed (possible earlier early-stop point).
