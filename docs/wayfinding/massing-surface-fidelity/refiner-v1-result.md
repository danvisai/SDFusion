# Surface-crispness refiner v1 — trained, integrated, re-gated (#46)

Execution record for [Train the wavy->crisp SDF refiner at scale + integrate + re-gate](https://github.com/danvisai/SDFusion/issues/46)
(the #41 lever, design proven in #45). The residual `RefineUNet3D` refiner is a **frozen post-process** on the
map-#24 prior's inference SDF (`prior -> refiner -> mesh`); the #27-gate-passing prior is untouched.

## Training (`scripts/foundations/train_refiner.py`)

- **Aligned synthetic pairs** (input = corrupted GT, target = the *same* GT), two corruption modes mixed
  per batch (`p_sdedit=0.5`):
  - **SDEdit partial regeneration** (primary) via the existing `model.sdedit` — encode GT -> z0, noise to a
    moderate timestep, run the truncated DDIM loop back, decode. Carries *real* prior waviness while staying
    anchored to the GT massing. Strength calibrated to **0.15** by matching real-prior roughness.
  - **σ-augmentation** — `decode(encode(GT) + σ·noise)`, σ ~ U(0.10, 0.20), the prototype's corruption widened.
- `RefineUNet3D` base=32, `delta_scale=0.25` (residual, zero-init output = identity start, tanh-bounded delta
  -> footprint-safe), `surface_weighted_l1` loss. 2000 GT buildings (800 with a precomputed SDEdit variant),
  3000 steps, batch 8, lr 2e-4. Train loss 0.0060 -> 0.0032. Wall time ~15 min on one GPU.

## Validation (n=24 held-out real prior samples — paired, same samples before/after)

| metric | before | after | GT floor |
|---|---|---|---|
| footprint-IoU (mean) | 0.871 | **0.886** | — |
| footprint-IoU (worst sample) | 0.705 | **0.719** | — |
| surface roughness (mean) | 0.00538 | **0.00474** | 0.00412 |

Every sample's footprint held or rose (no erosion — the Phase-2 failure mode). Roughness closed **~51%** of
the wavy->GT gap (up from the #45 prototype's ~26%).

## #27 re-gate at n=60 — WITH vs WITHOUT refinement

Both runs: map-#24 checkpoint, deployed config (`use_region 1, use_extra_cond 0, use_ema 1, ddim 100`). The two
runs are independent sample draws (the DDIM sampler is unseeded in the harness), so this is a distributional
check; the *paired* per-sample delta is the n=24 table above.

| metric | unrefined | **refined** | #27 threshold |
|---|---|---|---|
| collapse rate | 0.0% | 0.0% | ≤1% ✅ |
| LCC≥0.90 fraction | 100% | 100% | ≥85% ✅ |
| footprint-IoU median | 0.883 | **0.894** | ≥0.65 ✅ |
| footprint-IoU **p10** | 0.777 | **0.825** | ≥0.35 ✅ |
| OVERALL_SCALAR_PASS | PASS | **PASS** | |

**Both the median and the p10 improved after refinement** — the carried-forward p10 risk did not just survive,
it went up. No collapse, full connectivity. Artifacts: `execution/artifacts/baseline_gate_eval_{refined,unrefined}.json`.

## Visual (orchestrator sign-off, opus)

`outputs/refiner_v1/before_after_montage.png` (committed copy: `refiner-v1-before-after.png`): the refined
column has **visibly flatter walls, cleaner vertical edges, and smoother roof planes** than the lumpy "before"
column — the pitted/cratered tops that plagued the #45 prototype are substantially reduced. It is a **de-rippler,
not a structure generator**: it does not add missing GT detail (e.g. stepped rooflines the prior never produced),
and the ±0.2 SDF truncation caps how far a bounded residual can move the surface. No new artifacts, no footprint
bleed, no erased legitimate rooflines.

## Integration

`baseline_gate_eval.load_refiner(path)` reconstructs the net from the checkpoint's `{state_dict, base,
delta_scale}` and freezes it; `baseline_gate_eval.py --refine <ckpt>` applies it to `model.inference()` output
before occupancy/gate scoring. A CPU-only contract test (`TestLoadRefiner`) pins that an untrained (zero-init)
checkpoint is the exact identity map. Deploy as a single-forward (~ms) post-process; the prior is never retrained.

## Follow-up: more training plateaus (v2, 2026-07-24)

A v2 run scaled the *proven* recipe — 8000 steps (from 3000), 4000 pairs / 1600 SDEdit (from 2000/800),
`p_sdedit=0.6` — to test whether more training improves the result. **It does not.** On the same n=24
held-out real prior samples: roughness after = **0.00476** (v1: 0.00474) — flat, arguably a hair worse;
fp-IoU after = 0.890 (v1: 0.886) — marginally better; the before/after montage is visually indistinguishable
from v1. Train loss dropped a little (0.00565 -> 0.00304) but the *validation* roughness on real samples did
not move.

**Conclusion: the residual-refiner + synthetic-pairing recipe is already at its ceiling in v1** (~0.00474 vs
the 0.00412 GT floor); more of the same training does not close the remaining gap. That gap is the
*architectural* ceiling — a bounded residual on a ±0.2-truncated SDF, the synthetic-vs-real domain gap, and
the inability to add missing structure — not an under-training problem. Closing it further needs a *different*
lever (a stronger / structure-aware prior, or a different refiner formulation), a new effort beyond this map.
v2 was discarded; **v1 remains the deployed refiner.** (Also fixed a bug where `train_refiner.py` hardcoded the
result-JSON path and clobbered v1's record regardless of `--out_dir`; it now derives the name from `--out_dir`.)

## Status

**The map-#34 destination is reached:** generated massing renders crisper (flatter faces, cleaner edges),
footprint-safe, **#27 PASS at n=60 (median 0.894, p10 0.825)**. Trained weights: `outputs/refiner_v1/refiner_unet_v1.pth`
(on disk, not git-tracked per the repo's no-model-weights convention; regenerable via `train_refiner.py --seed 0`).
Foundations suite: 114 tests green.

**Honest limits (for whoever picks up crispness next):** the refiner de-ripples but doesn't fully reach the
GT crisp floor (roughness 0.00474 vs 0.00412) and can't add missing architectural structure — genuinely sharp
creases / stepped rooflines would need either a stronger prior or a structure-aware model, not a bounded
residual refiner. That is out of this map's scope.
