# Retrain Recipe & Go/No-Go Plan — the hand-off

Resolves [Settle the multi-corpus massing retrain recipe and go/no-go plan](https://github.com/danvisai/SDFusion/issues/30). This is the implementation-ready plan; every map decision folds in here. Execution (running it) is the hand-off — the map is planning-only.

## Data — LoD2-only

- **`data/real_massing_v1/real.h5`** — 35,776 solid LoD2 buildings (NL 11,776 + DE 12,000 + JP 12,000), footprint-paired, already solid (occ 13–30%). **No solidification** ([#26](https://github.com/danvisai/SDFusion/issues/26)).
- **BuildingNet excluded** from the massing mix. Its solidify/fallback/mask machinery (#28/#29/#32) is a **deferred phase-2 variety option**, used only if LoD2-only lacks architectural variety — not now.
- **Splits:** carve a clean, sealed LoD2 held-out test set (mirror the `make_splits.py` methodology: class- and source-stratified; sealed).

## Model — from scratch

- **From-scratch** Stage3a footprint-conditioned latent SDF diffusion (not warm-start) — drops the deployed checkpoint's baggage.
- **Conditioning:** footprint (core contract) + class + height + style + **region/`source_id`** (NL/DE/JP culture token, `use_region: True`). **Drop era/floors** (`use_extra_cond: False`).
- **Representation:** SDF, unchanged ([#28](https://github.com/danvisai/SDFusion/issues/28) keep-SDF holds trivially — LoD2 targets are already solid, no solidify step).

## Acceptance gate ([#27](https://github.com/danvisai/SDFusion/issues/27))

On generated LoD2 held-out (building-only footprint is moot here — LoD2 has no non-building labels):
collapse ≤ 1%; **≥ 85% of outputs with largest-connected-component ≥ 0.90**; footprint-IoU **median ≥ 0.65, p10 ≥ 0.35**; **visual montage sign-off is the final arbiter**; FID excluded.

## Plan / go–no-go

1. **Checkpoint 0 — baseline (zero compute).** Eval the *deployed* Stage3a on LoD2 held-out against the gate. **If it passes → STOP: destination reached, no retrain.**
2. **If it fails → from-scratch retrain.** Budget in **steps**: ceiling ~**100–150k**, gate eval every ~**25k**. Measure throughput from the first ~2k steps to project wall-clock (don't fix a wall-clock up front).
3. **Early-stop:** two consecutive checkpoints with no meaningful gate improvement → stop; or stop the moment the gate passes.
4. **Final acceptance:** full #27 gate **+ visual montage sign-off**.

## Notes

- `source_id` conditioning deliberately reverses the deployed config's "breadth is the lever, not a new token" stance — cross-cultural style diversity is explicitly wanted.
- The #27 solidify-strengthening forcing-function applies **only** if the deferred BuildingNet phase-2 is pursued.
- Entry point: `scripts/foundations/retrain_prior_hybrid.py` (adapt for from-scratch + region token), config `configs/stage3a_sdf_diffusion.yaml`.
