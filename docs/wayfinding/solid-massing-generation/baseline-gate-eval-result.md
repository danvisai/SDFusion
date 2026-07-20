# Baseline Gate Eval (Checkpoint 0) — result: NO-GO, retrain warranted

> **Follow-up:** the from-scratch LoD2 retrain this NO-GO triggered subsequently **PASSED** the gate (footprint-IoU 0.43→0.89). See `retrain-result.md`.

Execution of the #30 plan's Checkpoint 0: run the **deployed** Stage3a (`logs_building/2026-06-08…stage3a-hybrid-clean`, 20k steps) on LoD2 held-out (`real.h5` test split, deterministic) and score against the [#27 gate](https://github.com/danvisai/SDFusion/issues/27). Script `scripts/foundations/baseline_gate_eval.py`; artifacts `execution/artifacts/baseline_gate_eval.json`, `outputs/baseline_gate_eval/montage.png`.

## Result (n=60)

| gate criterion | value | verdict |
|---|---|---|
| collapse rate (≤1%) | **0.0%** | ✅ pass |
| ≥85% outputs with LCC ≥ 0.90 | **pass** | ✅ pass |
| footprint-IoU median (≥0.65) | **0.428** | ❌ fail |
| footprint-IoU p10 (≥0.35) | **0.275** | ❌ fail |
| visual montage sign-off (final arbiter) | **amorphous blobs, not blocks** | ❌ fail |
| **overall** | | **❌ FAIL** |

Sanity: real-footprint self-IoU median = **1.0** → the footprint-IoU metric is correctly oriented; the low generated value is a genuine model weakness, not an axis bug.

## Reading

- **The "breaking apart" reframe partly holds:** on LoD2 the deployed model does **not** collapse (0%) and does **not** fragment (LCC pass) — that pathology was indeed a BuildingNet-thin-shell artifact.
- **But the baseline does NOT pass:** the output is solid-but-**blobby** — it honors the footprint only moderately (IoU 0.43) and visually reads as a cauliflower lump, not a clean footprint-matching block-with-roof. The deployed 20k "hybrid" checkpoint (bag_ratio 0.5, no region token, undertrained) is not good enough.
- **The scalar gate has a blind spot** the visual arbiter caught: LCC + collapse both pass on a blob. (A future "block-like-ness" scalar could help, but #27's visual-final-arbiter rule did its job.)

## Decision consequence

Per the #30 go/no-go: **proceed to the from-scratch retrain** — and its target is now sharply characterized. The retrain does **not** need to solve collapse or fragmentation (already solved on LoD2); it needs to fix **footprint adherence + block-like geometry** (blob → clean footprint-matching block). This is exactly what the planned recipe targets: LoD2-only, from-scratch (drop the hybrid dilution / recipe corpus), region-conditioned, longer budget.
