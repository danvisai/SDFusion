# Residual-over-extrusion retrain — design

> **SUPERSEDED (2026-07-26).** The cheap de-risk (#56) showed that SDF-combining the analytic extrusion
> with the decoded residual on the 64³ grid **corrupts crispness**, so this residual-over-extrusion
> retrain was not pursued. The ceiling was instead located at the diffusion (not the codec) — see
> [`representation-ceiling-menu.md`](representation-ceiling-menu.md). Kept as a design record.

Lever ① of [map #52 "Crisp clean massing"](https://github.com/danvisai/SDFusion/issues/52), greenlit by
[#53](https://github.com/danvisai/SDFusion/issues/53). This is the design; execution follows in tickets
#56 (cheap de-risk) → #57 (retrain).

## The one change that matters

Today the diffusion generates the **whole** SDF, which the VQVAE + diffusion smooth into lumpy blobs
(map-#24 native render). Instead, the diffusion generates only the **residual** over an analytic
footprint-extrusion prior:

```
output_sdf = clip( ext(footprint, height)  +  decode(diffusion_residual) , ±0.2 )
```

The walls come from `ext(...)` — an **analytic** field added *outside* the lossy codec — so they are
**crisp by construction** and never touched by the smoothing that blobs the current model. #53 proved the
residual is small (~10% of field energy) and lives at the roof (wall residual ~0), so the diffusion only
has to learn the easy part.

## Inference-time contract (the design driver)

At inference the ONLY inputs are **footprint + height (+region)**. So the extrusion prior MUST be a
deterministic function of the *condition*, identical at train and inference time:

```
ext(footprint, height) = extrude( crisp_2d_sdf(footprint), y ∈ [ground, height] )
```

The #53 de-risk cheated by extruding GT's own mid-height slice (not available at inference) purely to
measure the residual — the retrain must instead build `ext` from the **footprint condition**. Everything
below follows from making that consistent and crisp.

## Components

### A. `footprint → crisp 2D SDF`  (make-or-break)

The prior's crispness == the 2D footprint SDF's crispness. In #53, an EDT of the raw 64² mask staircased
(roughness 0.020, ~5× GT); the crisp result came from a smooth cross-section. So:
- **Inference (demo):** footprints are polygons → **analytic polygon SDF** (exact, crisp).
- **Train (real.h5):** only a 64² mask exists → build a crisp SDF by **upsampling the mask ~4–8×,
  EDT, downsampling** (sub-voxel boundary), or fit a polygon then analytic SDF. Must match the inference
  crispness.
- Ship one shared `footprint_extrusion_sdf(footprint, height)` used by the dataset, the inference wrapper,
  and the demo bridge — a single source of truth (extends `scene/sdf_primitives.py` / reuses the #53
  `extrusion_from_midslice` combine formula, but sourced from the footprint).

### B. Target reparameterization  (dataset)

Fork `datasets/bag3d_dataset.py` → a residual variant that returns:
- `"sdf"` = `clip(GT_sdf − ext(footprint, GT_height), ±τ_r)` — **the residual is the new diffusion target**
- `footprint`, `height`, `region_id` — conditioning unchanged
- `"ext"` — the prior volume, for conditioning (C/D)

`τ_r`: #53 residual abs-max was ~0.09 → **τ_r = 0.1** truncation gives the codec a tighter, better-
conditioned target than the ±0.2 full-SDF range.

### C. Codec: finetune the VQVAE on residuals

Because the walls are added analytically *outside* the codec, the codec only ever sees residuals — but
`vqvae_clean.pth` was trained on full SDFs, so residuals are out-of-distribution. **Finetune
`vqvae_clean` on residual volumes** (cheap; mirrors the existing VQVAE finetune, `launchers/
train_vqvae_bnet_v2.sh` / `models/vqvae_model.py`) → `vqvae_residual.pth`.

### D. Diffusion: retrain Stage3a on residual latents

Fork `scripts/foundations/retrain_prior_hybrid.py` (the map-#24-style self-contained driver):
- frozen latent space = `vqvae_residual.pth` (C).
- target latent `z = vqvae_residual.encode(residual)` — the existing `stage3a_model.forward()` (line ~544
  `z = self.vqvae(self.x, encode_only=True)`) already encodes `self.x`; feeding the residual dataset (B) +
  the residual codec (C) is the whole mechanism, **no model surgery for the target**.
- conditioning unchanged (footprint, height, region) **+ recommended:** concat the **ext-prior latent** as
  an extra `c_concat` channel so the model knows the base it is residualising against.
- **loss on the reconstructed FULL SDF** (`ext + decode(residual)`) vs GT, not on the raw residual — so the
  objective targets *final* crispness. `_surface_band_smooth_l1` stays; `_soft_footprint_bce` can be
  **dropped** (ext already guarantees the footprint). Leave a hook to add #54's normal/eikonal loss.
- **warm-start** from the map-#24 checkpoint (`--finetune_from`) to cut iterations.

### E. Inference wrapper

Add a `residual` mode: build `ext`, decode the residual, return `clip(ext + residual, ±0.2)`, mesh at 0.0.
Wire it into `baseline_gate_eval.py` and the demo `proto_clean_structure.py` bridge.

## Eval / success criteria

- **Roughness** of the reconstructed full SDF → ~**0.004** GT floor (vs map-#24's 0.0047) — the headline metric.
- **#27 gate** footprint-IoU ≥ map-#24 (should be trivially high — `ext` is footprint-exact).
- **Honest-shaded montage** (native held-out + the demo's Munich/Lafayette footprints): crisp flat walls,
  sharp edges, no blobs — the map's HITL visual bar.

## Risks / scope

- **A (footprint→crisp-2D-SDF consistency)** is make-or-break; de-risked in #56 before any diffusion cost.
- **C (VQVAE-on-residuals OOD)** — finetune; round-trip de-risked in #56.
- **Setback-heavy buildings** (the #53 caveat, e.g. gi=5157): a single cross-section extrusion is a weaker
  prior → larger residual there. **Accept for v1**; a richer prior (per-height cross-section or a coarse
  setback term) is a later enhancement.
- **Compute:** VQVAE finetune is cheap (~hours); the diffusion retrain is expensive (map-#24 ≈ 120k
  iters / days) — warm-start reduces it.

## Milestones (cheapest first)

1. **#56 (cheap, gates the spend):** the `footprint→crisp-2D-SDF` module + residual dataset + VQVAE-on-
   residuals finetune + an **oracle round-trip check** — is `ext + decode(encode(residual))` crisp
   (roughness → GT floor) and footprint-faithful? If yes, the expensive diffusion retrain is de-risked.
2. **#57 (expensive, blocked by #56):** Stage3a residual retrain (D) + eval (E).
