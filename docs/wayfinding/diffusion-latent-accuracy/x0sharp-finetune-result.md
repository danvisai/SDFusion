# x0-sharp diffusion finetune — result (#60)

Ticket [#60](https://github.com/danvisai/SDFusion/issues/60) of [map #58](https://github.com/danvisai/SDFusion/issues/58).
**Verdict: NEGATIVE — the decoded-x0 smoothness finetune does not reach crisp at any weight.** Cheap
latent-accuracy levers are now exhausted; the fix is a representation change (implicit decoder).

## What was run
Warm-start the map-#24 prior (`real.h5` LoD2, `use_region 1`, `use_extra_cond 0`) with the
decoded-predicted-x0 smoothness regularizer already in `stage3a_model.forward()`
(`_sdf_field_smoothness`, `grad_tv` = in-band mean |Laplacian|), via `retrain_prior_hybrid.py
--finetune_from <map-#24> --use_smooth 1 --smooth_kind grad_tv --smooth_weight W`. 3000 iters, bs 8.
This is map-#34's planned-but-unexecuted **Phase-2**.

## Results (paired, n=24 held-out; GT floor 0.0041; eval = roughness + #27 gate)

| run | roughness | footprint-IoU med / p10 | #27 gate | outcome |
|---|---|---|---|---|
| map-#24 baseline | 0.00552 | 0.872 / 0.805 | PASS | — |
| **w=0.1** (no clip) | **0.03045** | 0.385 / 0.295 | **FAIL** | **DIVERGED into rubble** |
| **w=0.05 + grad-clip 1.0** | **0.00547** | 0.819 / 0.527 | PASS | **no gain + mild footprint erosion** |

Montages: `x0sharp-vs-map24-montage.png` (w=0.1, rubble), `x0sharp-w05-vs-map24-montage.png` (w=0.05,
lumpy like map-#24, some rows grew appendages).

## The divergence (w=0.1), diagnosed
The `simple` (noise-prediction) loss was stable (~0.01–0.02) through iter ~2100, then **exploded to
~1.0 and never recovered** (eps-MSE≈1.0 = a broken denoiser → garbage latents → rubble decode). Cause:
`grad_tv` is computed on the decoded **predicted-x0**, reconstructed at a random timestep as
`x0 ≈ (x_t − √(1−ᾱ)·ε̂)/√ᾱ`; at high noise `√ᾱ→0`, so it **amplifies the model's ε-error enormously**,
and its Laplacian injects large, high-variance gradients — at w=0.1 this drove the weights out of the
map-#24 basin. (Exactly the instability StEik, NeurIPS'23, flags — see `docs/research/crisp-massing-literature.md`.)

## The fix + the clean result
Added a gated `--grad_clip` (max grad-norm; `stage3a_model.optimize_parameters` clips the optimized
params; 0=off, no change to other runs). At **w=0.05 + grad-clip 1.0** the run stayed stable the whole
3000 iters (no `simple` spike) — but roughness was **unchanged** (0.00547 vs 0.00552) and the footprint
mildly eroded (p10 0.805→0.527). So the smoothness finetune has **no crisp regime**: weak = no gain
(+small shape cost), strong = divergence. Cleanly reconfirms map-#34's Phase-2 negative on the map-#24 base.

## Conclusion for map #58
Every cheap lever is exhausted — codec is crisp (0.0044≈GT, #56); post-decode refiner (SDF #54, latent
#59) plateaus at 0.0047; x0-sharp finetune gives no gain (#60). **Nothing that keeps the 64³-dense-grid
diffusion reaches GT crispness.** The durable fix is a **query-based implicit / vecset decoder**
(`representation-ceiling-menu.md`, `docs/research/crisp-massing-literature.md`) — a fresh effort.
