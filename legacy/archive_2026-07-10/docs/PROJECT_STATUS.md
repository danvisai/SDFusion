# GenerativeTowns / Path Q+ — Project Status

Author: Danvi (project owner)
Host: Purdue Gilbreth, A100 80GB node
Repo root: `/scratch/gilbreth/dsimhadr/GenerativeTowns/SDFusion`

---

## End goal

Take a 2D map containing N building footprints (annotated with class labels
where available) and produce, per footprint, a 3D mesh that (a) projects back
exactly to the input footprint, (b) looks like a plausible building of the
given class, and (c) is positioned at the world coordinates implied by the map
so the meshes compose into a single town/scene without manual placement.

The intermediate target is a per-building SDF on a `64^3` grid in
BuildingNet's normalized coordinate frame — meshes are extracted by marching
cubes at iso=0 and then translated/scaled per the map's metric scale.

---

## Current architecture: "Path Q+"

```
   2D map
     |
     +--> per-footprint crop  (binary mask, 1 channel)
            |
            +--> class label (RESIDENTIAL/RELIGIOUS/COMMERCIAL/MILITARY/PUBLIC + sub)
                   |
                   v
            [Step 3]  ControlNet (SD1.5 + footprint cond)        TRAINED
                   |   --> single-view 3/4 ortho RGB render of "what
                   |        a class-X building with this footprint
                   |        would look like"
                   v
            [Step 4]  Hunyuan3D-2  (frozen, image-to-3D)         WORKING (smoke)
                   |   --> raw mesh / SDF₀ in unit-cube frame
                   v
            [Step 5]  SDF-residual diffusion  (planned)          NOT YET TRAINED
                   |   conditioned on (footprint, height_map, class)
                   |   target = GT_SDF  -  Hunyuan3D-2_SDF
                   |   reuses existing VQVAE+UNet (configs/sdfusion-img2shape.yaml)
                   v
            cleaned per-building SDF -> marching cubes -> mesh
            |
   union   (boolean OR over all per-building SDFs at world placement)
            |
   final composite town mesh
```

Rationale: SDFusion alone (a 3D diffusion in latent space) was not converging
on BuildingNet (more on this in **Failures and lessons** below). Path Q+
instead leans on a strong frozen image-to-3D prior (Hunyuan3D-2) and uses
SDFusion as a *residual cleanup* network. The 3D model only has to learn the
delta between Hunyuan3D-2's noisy mesh and the GT SDF, which is a much smaller,
better-conditioned learning problem and respects the footprint exactly via
explicit conditioning.

---

## What works today

- **VQVAE on BuildingNet (3D, res=64).** Trained `2025-05-19`. Checkpoint:
  `logs_building/2025-05-19T19-58-28-vqvae-building-all-res64-LR1e-4-T0.2-release/ckpt/vqvae_steps-latest.pth`.
  Latent shape `(B, 3, 16, 16, 16)` (3D conv, axis-preserving).

- **Hunyuan3D-2 inference set up locally.** Frozen, used as the image-to-3D
  prior. Source: `external/Hunyuan3D-2/`; weight cache:
  `external/hf_cache/hub/models--tencent--Hunyuan3D-2/`. Driven through
  `hy3dgen.shapegen.Hunyuan3DDiTFlowMatchingPipeline`. Empirical wall time
  ~30s/building on the A100 (per `scripts/path_q_smoke.py` log timings — TBD
  verify exact median).

- **BuildingNet ortho renders generated.** 1740 of 1744 model ids rendered to
  `data/BuildingNet_dataset_v0_1/buildingnet_renders/<phase>/<id>.png` at
  512×512 white-background 3/4 view (elev=20°, azim=30°, FoVOrthographic).
  Generator: `scripts/render_buildingnet_orthoviews.py`. Current counts on
  disk: `train=1470, val=76, test=198` (split files are post-medium-filter so
  the renders cover the union of pre- and post-filter ids — TBD verify the
  4-mesh marching-cubes failures).

- **Footprint→view ControlNet trained (v2, bf16).**
  - Run dir: `Logs_GT/CN-2026-05-05T22-55-39-footprint2view-15k-bf16/`
  - Final ckpt: `ckpt/controlnet-015000/`
  - Trained 15000 steps, batch 4, lr 1e-5, 500-step linear warmup, bf16
    throughout, `gradient_checkpointing` on. Source `train_controlnet.py`,
    launched via `launchers/train_controlnet_full_v2.sh`.
  - Loss trajectory (from `loss_log.txt`): step 50 → 0.0309, step 1000 →
    0.0255, step 15000 → 0.0295. Range across the run is roughly 0.007–0.08
    with the final-buckets mean near ~0.03. No divergence.
  - Sample grids saved every 500 steps to `samples/step{N:06d}_train_grid.png`
    and `samples/step{N:06d}_val_grid.png` (60 grid PNGs total).

- **val split carved out and wired into the trainer.**
  - 5%-stratified-by-category split via `scripts/make_val_split.py`. Originals
    backed up as `splits/train_split.txt.bak`.
  - `datasets/base_dataset.py:89-92` constructs a `BuildingNetDataset` with
    `phase='val'`. `datasets/dataloader.py:21-26` wraps it. `train.py:120-123`
    runs `_val_loss(model, val_dl)` at every `save_steps_freq` and reports
    `val_loss` alongside `eval_metrics`.

- **SDF axis convention verified (D=z, H=y, W=x).** Per
  `~/.claude/projects/-scratch-gilbreth-dsimhadr-GenerativeTowns-SDFusion/memory/project_sdfusion_axes.md`:
  encode→decode silhouette IoU is 0.83 along Y vs 0.64 along Z/X. Code in
  `models/sdfusion_model_img2shape.py:_build_fp3d_for` (lines 221–237) places
  the `(z, x)` silhouette on `(D, W)` and replicates along H, exactly as the
  axis convention requires.

- **Filtered splits (1091 train / 55 val / 132 test).** Threshold `inside%
  >= 0.20` via `scripts/filter_low_inside_splits.py`. Pre-filter counts (from
  `*.prefilter.bak`): 1470/76/197 — so the medium filter dropped 379+21+65 =
  465 ids. Originals in `splits/<split>_split.txt.prefilter.bak`. h5 files on
  disk are not deleted, just unreferenced.

- **Recomputed footprints from SDF.** `scripts/recompute_footprints_from_sdf.py`
  rewrites `h5['footprint']` as `(sdf<=0).any(axis=Y)` for all SDFs in
  `data/BuildingNet_dataset_v0_1/resolution_64/<id>/ori_sample_grid.h5` and
  regenerates the matching PNGs in `footprints_png/<phase>/`. PNG counts:
  `train=1470, val=76, test=198` (pre-filter).

- **Latent scale_factor calibrated.** `scripts/compute_scale_factor.py`
  computes Welford-style std over ~19M VQVAE-encoded latent values on the
  train split. Result baked into `configs/sdfusion-img2shape.yaml:11`:
  `scale_factor: 2.380615` (= 1/0.4201). The SDFusion model uses it at
  forward (`models/sdfusion_model_img2shape.py:351`) and inverts it before
  decoding (`:488`).

- **Path Q+ Layer 0 smoke complete.** `scripts/path_q_smoke.py` crops
  rows 1 and 2 from `step015000_train_grid.png`, feeds the ControlNet gens
  into Hunyuan3D-2, exports `.glb`, renders a contact sheet. Output:
  `outputs/path_q_smoke/path_q_smoke_summary.png` plus per-row inputs/glbs.

- **Path Q+ Layer 1 smoke (negative-prompt) generated.**
  `scripts/path_q_smoke_negprompt.py` re-runs the v2 ControlNet with
  `guidance_scale=8.0`, `num_inference_steps=30`, an aggressive negative prompt
  (windows/ornaments/photo-realism/etc.) and a "matte 3D model" positive
  prompt. ControlNet outputs are saved at
  `outputs/path_q_smoke_neg/row{1,2}_controlnet_gen_neg.png`. As of the most
  recent file listing, the Hunyuan3D-2 step of this script has not yet
  produced `_mesh_neg.glb` files in that directory — TBD verify whether the
  negative-prompt run actually completed end-to-end vs only ran the SD
  ControlNet half.

---

## Failures and lessons

- **v1 ControlNet diverged due to fp16 numerical instability.**
  - First 15k-step ControlNet run used `mixed_precision=fp16`. Loss climbed
    from ~0.02 at start to ~0.92 by mid-run. Outputs collapsed to noise.
  - Root cause: the trainable ControlNet (fp32 master weights, fp16 forward)
    handed its down/mid residuals into a frozen fp16 UNet. The fp32→fp16 cast
    at the residual hand-off was where AMP infra had no calibrated loss-scaler
    and where activations were already large; gradients in the trainable
    branch eventually overflowed silently.
  - Fix (v2, working): keep VAE / text encoder / base UNet *and* the trainable
    ControlNet all in `bf16` (`train_controlnet.py:146-154`). bf16 has fp32-like
    dynamic range, so the boundary cast is a no-op in practice. Also added a
    500-step linear LR warmup (`--warmup_steps 500`).
  - **Lesson:** never mix fp32-trainable + fp16-frozen at a residual hand-off
    without first verifying the AMP loss-scaler covers the trainable branch.
    Default to bf16 on Ampere+; fall back to fp32 if memory permits. fp16 only
    when you have working `GradScaler` infrastructure already proven on the
    exact computation graph.

- **`bake_footprint()` in `preprocess/create_sdf.py` was broken.**
  - Original behavior: project every mesh face onto the XZ plane, including
    roofs / eaves / ceilings, then per-mesh normalize XZ extent to `[0,1]^2`.
  - Empirical impact: 30-sample audit showed mean occupancy 81%, with 13% of
    samples 100% white. Per-mesh normalization also destroyed the absolute
    scale information that should have anchored "footprint area = N square
    grid cells".
  - Fix: `scripts/recompute_footprints_from_sdf.py` derives the footprint
    directly from the SDF voxel grid as `(sdf<=0).any(axis=Y_AXIS)`, in the
    centered/unit-sphere-normalized frame the SDF already lives in. This is
    the correct top-down silhouette and preserves scale relative to the SDF.
  - **Lesson:** if a derived label is suspicious, compute it from the most
    downstream artifact you trust (here, the SDF), not the upstream mesh.

- **`_build_fp3d_for` was rotating the ground plan 90° around X.**
  - Earlier code sized the footprint to `(H, W)` and replicated along the D
    axis. With latent layout `(B, C, D=z, H=y, W=x)`, that lifts a `(z, x)`
    silhouette into the `(y, x)` plane, which is a 90°-around-X rotation of
    the building's ground plan.
  - Fix: `models/sdfusion_model_img2shape.py:221-237`. Resize to `(D, W)` and
    replicate along H. Verified empirically (silhouette IoU 0.83 on Y vs 0.64
    elsewhere).
  - **Lesson:** at a cubic latent (16³ here), an axis swap doesn't crash and
    doesn't show up in scalar loss. It only manifests as poor coupling between
    the spatial conditioning and the generated geometry. Always verify axis
    correspondence with an encode→decode silhouette IoU sweep before training.

- **BuildingNet meshes are mostly non-watertight.**
  - Only ~51% have `inside%` (fraction of `sdf<=0` voxels in the 64³ grid)
    above 0.5%. The medium filter (>=0.20%) drops 466 unusable buildings;
    the post-filter splits are 1091/55/132. Below the threshold the
    marching-cubes mesh is either empty or shattered into floating fragments
    that produce an unusable training render.
  - **Lesson:** treat `inside%` as a primary data-quality gate, not a nice-
    to-have. Run `scripts/filter_low_inside_splits.py --dry_run` before any
    new dataset.

- **SDFusion-only training never converged on BuildingNet** (older runs in
  `Logs_GT/2025-*` and `SMOKE-2026-05-05*`). Combination of the
  `_build_fp3d_for` axis bug, broken footprints, and the missing
  `scale_factor` made the loss readings meaningless for many tens of
  thousands of steps. Path Q+ pivots away from "ask SDFusion to generate
  the building from scratch" and toward "ask SDFusion to clean up
  Hunyuan3D-2's output" — see Step 5 below.

---

## Outstanding: Path Q+ remaining steps

- **Step 4 (planned): build the SDF residual training set.**
  - For all 1091 train ids, render the GT footprint through the v2 ControlNet
    → feed result into Hunyuan3D-2 → voxelize the resulting mesh into a 64³
    SDF in BuildingNet's normalized frame → save as
    `data/BuildingNet_dataset_v0_1/hy3d_sdf/<id>.h5` (path TBD).
  - Estimated wall time: ~30s/building × 1091 ≈ 9h on a single A100 (sequential).
    Could be batched up if Hunyuan3D-2 supports it — TBD verify.

- **Step 5 (planned): train SDF-residual diffusion.**
  - Reuse `models/sdfusion_model_img2shape.py` (VQVAE encoder frozen, 3D UNet
    trained from scratch or from existing ckpt) but change the target from
    `z = encode(GT_SDF)` to `z = encode(GT_SDF − Hunyuan3D-2_SDF)`.
  - Conditioning: `(footprint_2D, height_map_2D, class_id)`. Footprint volume
    plumbing is already in place via `_build_fp3d_for` and the
    `c_concat`/`c_crossattn` hybrid path in `apply_model`. Height-map and
    class-id channels are TBD.
  - Estimated wall time: ~2h to a usable checkpoint at the same throughput as
    the current `train_sdfusion_img2shape_smoke.sh` smoke run. The smaller
    learning target (residual) should converge much faster than the full
    SDFusion-from-scratch training did.

- **Step 6 (planned): scene composition.**
  - Map → per-footprint crops → per-building SDFs (Step 5 output) → boolean
    union of all signed distance fields at metric world placement → single
    composite SDF → marching cubes for the final town mesh.
  - Open question: how to resolve overlaps where neighbouring building SDFs
    interpenetrate. Simplest: take the elementwise min (which is the SDF-union
    operator). For aesthetics we may want a smooth-min (R-function) variant.

---

## Known issues being investigated

- **Hunyuan3D-2 produces noisy meshes on photorealistic ControlNet gens.**
  The current ControlNet was trained against ortho RGB renders that have some
  shading detail; the resulting outputs lean photorealistic, which makes
  Hunyuan3D-2 hallucinate windows/balconies/ornaments that don't exist on the
  GT building.
  - **Layer 1 (in flight):** negative-prompt + clean massing positive prompt
    at high CFG. `scripts/path_q_smoke_negprompt.py` is the test harness;
    ControlNet outputs exist in `outputs/path_q_smoke_neg/` but the Hunyuan3D-2
    `_mesh_neg.glb` files are not yet present in the directory listing —
    needs verification of whether the run completed.
  - **Layer 2 (fallback if Layer 1 insufficient):** retrain the ControlNet
    against depth-map / normal-map renders instead of RGB. This forces the
    surface-form prior to be geometric, not photometric, which should make
    Hunyuan3D-2 hallucinate less.

- **Two outputs directories with similar names (`single_mask_outpputs` typo
  + `single_mask_outputs`).** Likely stale, but content not audited — TBD
  verify before reorganizing.

---

## Key file paths (quick reference)

Models / config:
- `models/sdfusion_model_img2shape.py` — 3D SDF diffusion model with footprint
  hybrid conditioning. Active class: `SDFusionImageFPShapeModel`.
- `configs/sdfusion-img2shape.yaml` — diffusion + UNet + CLIP config.
  `conditioning_key: hybrid`, `in_channels: 4` (3 latent + 1 footprint),
  `scale_factor: 2.380615`, `latent_size_HW: [64,64]`, `latent_size_D: 64`.
- `configs/vqvae_bnet.yaml` — VQVAE config used to encode SDFs.

Trainers:
- `train.py` — SDFusion latent diffusion trainer with val-loss reporting
  (`_val_loss` at lines 28–44, called at line 120).
- `train_controlnet.py` — ControlNet finetuner, single-GPU, no `accelerate`.
- `launchers/train_controlnet_full_v2.sh` — bf16, 15k steps, batch 4, lr 1e-5,
  500-step warmup, `gradient_checkpointing`.
- `launchers/train_sdfusion_img2shape_smoke.sh` — 2000-iter smoke for the 3D
  diffusion side.

Datasets:
- `datasets/buildingnet_dataset.py` — yields `{sdf, fp, img, path}` for the
  3D model. Reads SDF + footprint from `ori_sample_grid.h5`, footprint PNG
  separately as the "image" branch.
- `datasets/buildingnet_controlnet_dataset.py` — yields
  `{pixel_values, conditioning_pixel_values, input_ids, prompt}` for
  ControlNet. Builds prompts like `"a house building, residential, 3/4 view,
  white background"` from the model_id prefix.

Scripts:
- `scripts/recompute_footprints_from_sdf.py` — rewrites h5 footprints + PNGs.
- `scripts/compute_scale_factor.py` — Welford-style latent std over train set.
- `scripts/render_buildingnet_orthoviews.py` — per-id 3/4 ortho renders.
- `scripts/filter_low_inside_splits.py` — `inside% >= 0.2` filter, with
  `*.prefilter.bak` rollback files.
- `scripts/make_val_split.py` — 5%-stratified val carve-out.
- `scripts/path_q_smoke.py` — ControlNet → Hunyuan3D-2 end-to-end smoke.
- `scripts/path_q_smoke_negprompt.py` — Layer 1 negative-prompt smoke.
- `scripts/check_smoke_run.py` — post-hoc loss/visual sanity check on a
  `Logs_GT/<run>/` directory.

Checkpoints:
- VQVAE: `logs_building/2025-05-19T19-58-28-vqvae-building-all-res64-LR1e-4-T0.2-release/ckpt/vqvae_steps-latest.pth`
- ControlNet v2 (final): `Logs_GT/CN-2026-05-05T22-55-39-footprint2view-15k-bf16/ckpt/controlnet-015000/`
- Earlier `Saved_Checkpoint/df_steps-latest.pth` exists but is from a pre-fix
  SDFusion run — do not load for new experiments.

External / caches:
- `external/Hunyuan3D-2/` — Tencent's image-to-3D model code (frozen, cloned).
- `external/hf_cache/hub/models--tencent--Hunyuan3D-2/` — HF weight cache.
- `external/hf_cache/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/`
  — SD1.5 base used by ControlNet trainer.

Data layout:
- `data/BuildingNet_dataset_v0_1/resolution_64/<id>/ori_sample_grid.h5` —
  SDF (`pc_sdf_sample`, flat, reshape `(64,64,64)` in `(z,y,x)` order) plus
  footprint (`footprint`, `(1,64,64)` uint8).
- `data/BuildingNet_dataset_v0_1/footprints_png/{train,val,test}/<id>.png`
- `data/BuildingNet_dataset_v0_1/buildingnet_renders/{train,val,test}/<id>.png`
- `data/BuildingNet_dataset_v0_1/splits/{train,val,test}_split.txt` (post
  filter), with `*.prefilter.bak` and `train_split.txt.bak` rollback copies.

Outputs:
- `outputs/path_q_smoke/` — Layer 0 smoke artifacts (footprint, gen, mesh
  GLB, mesh render, 4-up multiview).
- `outputs/path_q_smoke_neg/` — Layer 1 smoke artifacts (ControlNet outputs
  with neg prompt; Hunyuan3D-2 outputs TBD verify).

---

## How to reproduce / commands cheat sheet

All commands run from `/scratch/gilbreth/dsimhadr/GenerativeTowns/SDFusion`.
The `env -u LD_PRELOAD -u LD_LIBRARY_PATH` prefix is required on Gilbreth to
strip XALT and the Spack-managed library path that collide with our pinned
PyTorch+CUDA. The `sdfusion/bin/python` path is the project's local venv.

```bash
# 0. Activate the workaround prefix once per shell (or source this from .bashrc):
PFX='env -u LD_PRELOAD -u LD_LIBRARY_PATH'
PY="$(pwd)/sdfusion/bin/python"

# 1. Recompute footprints from SDFs (idempotent, has --dry_run)
$PFX $PY scripts/recompute_footprints_from_sdf.py --dry_run
$PFX $PY scripts/recompute_footprints_from_sdf.py

# 2. Carve val split (one-time; refuses to overwrite)
$PFX $PY scripts/make_val_split.py --dry_run
$PFX $PY scripts/make_val_split.py

# 3. Filter low-inside ids (creates *.prefilter.bak on first run)
$PFX $PY scripts/filter_low_inside_splits.py --dry_run
$PFX $PY scripts/filter_low_inside_splits.py

# 4. Render BuildingNet 3/4 ortho views (already done; --overwrite to redo)
$PFX $PY scripts/render_buildingnet_orthoviews.py --phase all

# 5. Train ControlNet v2 (the working bf16 run)
$PFX ./launchers/train_controlnet_full_v2.sh

# 6. SDFusion smoke (2000 iters; for axis / loss-log / scale_factor sanity)
$PFX ./launchers/train_sdfusion_img2shape_smoke.sh

# 7. Inspect a smoke run after the fact
$PFX $PY scripts/check_smoke_run.py Logs_GT/SMOKE-<...>

# 8. End-to-end Path Q+ smoke (requires v2 ControlNet ckpt at step 015000)
$PFX HF_HOME=$(pwd)/external/hf_cache $PY scripts/path_q_smoke.py
$PFX HF_HOME=$(pwd)/external/hf_cache $PY scripts/path_q_smoke_negprompt.py

# 9. Recompute the latent scale_factor if you retrain the VQVAE
$PFX $PY scripts/compute_scale_factor.py \
    --vq_cfg  configs/vqvae_bnet.yaml \
    --vq_ckpt logs_building/2025-05-19T19-58-28-vqvae-building-all-res64-LR1e-4-T0.2-release/ckpt/vqvae_steps-latest.pth \
    --dataroot data --num_batches 32 --batch_size 8 --trunc_thres 0.2
# Then update configs/sdfusion-img2shape.yaml model.params.scale_factor.
```

Useful one-off checks:

```bash
# Verify split sizes match the filter
wc -l data/BuildingNet_dataset_v0_1/splits/{train,val,test}_split.txt
# expected: 1091 train / 55 val / 132 test

# Peek at the latest ControlNet loss
tail -50 Logs_GT/CN-2026-05-05T22-55-39-footprint2view-15k-bf16/loss_log.txt

# Inspect a sample grid
xdg-open Logs_GT/CN-2026-05-05T22-55-39-footprint2view-15k-bf16/samples/step015000_train_grid.png
```
