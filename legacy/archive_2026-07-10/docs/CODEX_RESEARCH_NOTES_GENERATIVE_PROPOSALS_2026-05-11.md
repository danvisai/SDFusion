# Codex Research Notes - Generative Proposal Images

Prepared by: Codex  
Project: `/scratch/gilbreth/dsimhadr/GenerativeTowns/SDFusion`  
Date: 2026-05-11

## Current Issue

The learned proposal-image path works end to end:

```text
OSM footprint -> learned proposal image -> Hunyuan3D -> placed/audited mesh
```

But the first pixel-space DDPM proposal model still produces RGB speckle in
sample sheets, even after an 8-example overfit run.

Important empirical state:

```text
supervised UNet:
  sharp-loss best val_l1_pixel: 0.0465
  preview: better but still blurry

pixel DDPM smoke:
  20 epochs, all data
  sample_epoch_020: RGB speckle

pixel DDPM overfit:
  8 examples, 300 epochs, EMA
  best val_noise_mse: 0.0223
  sample_epoch_300: still has RGB speckle
```

Interpretation:

- More full-dataset DDPM training is not the next move.
- The current pixel-space DDPM can reduce denoising loss, but the reverse
  sampler/model formulation is not producing clean proposal images.
- We should cleanly separate three tests:
  1. denoising reconstruction from a partially noised real image,
  2. deterministic DDIM sampling from noise,
  3. Hunyuan downstream usefulness.

## Relevant Papers

### Improved DDPM

Nichol and Dhariwal, "Improved Denoising Diffusion Probabilistic Models"

Source: https://huggingface.co/papers/2102.09672

Why it matters:

- Introduces practical DDPM improvements such as learned reverse-process
  variances and better sampling efficiency.
- Our current sampler is a minimal fixed-variance DDPM sampler. The paper
  supports the idea that naive sampling can be a limiting factor even when
  training loss improves.

Actionable takeaway:

- Add DDIM and/or improved variance handling before judging pixel diffusion.
- Track sample quality, not only noise MSE.

### One-2-3-45++

Liu et al., "One-2-3-45++: Fast Single Image to 3D Objects with Consistent
Multi-View Generation and 3D Diffusion"

Source: https://huggingface.co/papers/2311.07885

Why it matters:

- Reinforces the strategy of using image-conditioned generation as a strong
  control surface before 3D generation.
- Our pipeline already uses Hunyuan as image-to-3D; the proposal-image model
  should produce clean, object-like conditioning images before Hunyuan sees
  them.

Actionable takeaway:

- Keep Hunyuan downstream.
- Improve the proposal image generator rather than forcing weak images through
  image-to-3D.

### BuilDiff

Wei, Vosselman, and Yang, "BuilDiff: 3D Building Shape Generation using
Single-Image Conditional Point Cloud Diffusion Models"

Source: https://researchportal.bath.ac.uk/en/publications/buildiff-3d-building-shape-generation-using-single-image-conditio/

Why it matters:

- Directly relevant to building generation from images.
- Suggests an alternative to our current Hunyuan dependency: image-conditioned
  point-cloud diffusion for building shape.

Actionable takeaway:

- Medium-term: compare Hunyuan image-to-3D against a building-specialized
  diffusion shape model or point-cloud diffusion branch.

### ControlCity

Zhou et al., "ControlCity: A Multimodal Diffusion Model Based Approach for
Accurate Geospatial Data Generation and Urban Morphology Analysis"

Source: https://huggingface.co/papers/2409.17049

Why it matters:

- Uses multimodal diffusion for geospatial/building-footprint generation.
- Supports our move toward conditioning on OSM/geospatial channels rather than
  only retrieved assets.

Actionable takeaway:

- Our condition representation should become richer:
  footprint, roads/context, class, height, and neighborhood style.

### PrITTI

Tze et al., "PrITTI: Primitive-based Generation of Controllable and Editable
3D Semantic Urban Scenes"

Source: https://huggingface.co/papers/2506.19117

Why it matters:

- Uses latent diffusion over structured urban primitives rather than dense
  voxels.
- This aligns with our observation that direct 64^3 SDF output is too coarse,
  while retrieval/Hunyuan meshes are visually useful but need controllable
  structure.

Actionable takeaway:

- Long-term: represent towns as editable building primitives or latent object
  proposals, not just pixels or voxels.

### Human-Guided Urban Form Diffusion

"Human-guided urban form generation using multimodal diffusion models"

Source: https://www.sciencedirect.com/science/article/pii/S0360132325013629

Why it matters:

- Uses a staged urban generation pipeline and ControlNet-style constraints.
- This supports a staged approach like ours:
  map/layout constraints -> proposal image -> 3D generation -> audit/repair.

Actionable takeaway:

- Consider ControlNet-style conditioning over a pretrained image diffusion
  model rather than training pixel DDPM from scratch on 37 examples.

## Recommended Next Technical Steps

Do not continue full DDPM training yet.

Priority:

1. Add DDIM sampling to `scripts/train_osm_proposal_image_ddpm.py`.
2. Add a reconstruction sheet:
   - take real target image,
   - noise to selected timesteps,
   - reverse-denoise,
   - compare target/noisy/reconstructed.
3. If reconstruction works but pure samples speckle, sampler/noise schedule is
   the problem.
4. If reconstruction fails, the score model or conditioning architecture is
   insufficient.
5. If both fail, move to a pretrained/latent diffusion approach instead of
   pixel DDPM from scratch.

Most likely next implementation:

```text
DDIM sampler + reconstruction diagnostic sheet
```

Do this before more training.

## Implemented Diagnostic Result

Implemented in:

```text
scripts/train_osm_proposal_image_ddpm.py
```

Added:

- deterministic DDIM sampler,
- `--sampler ddpm|ddim`,
- `--recon_every`,
- `--recon_timesteps`,
- `--resume_ckpt`,
- reconstruction diagnostic sheets:
  `mask | target | noised target | reconstructed target`.

Diagnostic command:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/train_osm_proposal_image_ddpm.py \
  --train_jsonl outputs/osm_proposal_image_dataset_v1/campus_lafayette_proposal_v1_train.jsonl \
  --val_jsonl outputs/osm_proposal_image_dataset_v1/campus_lafayette_proposal_v1_train.jsonl \
  --out_dir outputs/osm_proposal_image_ddpm_v1_overfit8_ddim_diag \
  --image_size 128 \
  --batch_size 4 \
  --epochs 300 \
  --base_channels 48 \
  --timesteps 200 \
  --sample_steps 100 \
  --sampler ddim \
  --sample_every 50 \
  --recon_every 1 \
  --recon_timesteps 25,50,100,150 \
  --resume_ckpt outputs/osm_proposal_image_ddpm_v1_overfit8/ckpt_best.pth \
  --max_train_examples 8 \
  --max_val_examples 8 \
  --ema_decay 0.995 \
  --num_workers 2 \
  --device cuda
```

Outputs:

```text
outputs/osm_proposal_image_ddpm_v1_overfit8_ddim_diag/sample_ddim_resume_epoch_300.png
outputs/osm_proposal_image_ddpm_v1_overfit8_ddim_diag/recon_t025_epoch_300.png
outputs/osm_proposal_image_ddpm_v1_overfit8_ddim_diag/recon_t050_epoch_300.png
outputs/osm_proposal_image_ddpm_v1_overfit8_ddim_diag/recon_t100_epoch_300.png
outputs/osm_proposal_image_ddpm_v1_overfit8_ddim_diag/recon_t150_epoch_300.png
```

Observed result:

- Pure DDIM samples from noise still show RGB speckle.
- Reconstructions from partially noised real target images recover the correct
  building mass/silhouette, especially from lower noise levels.
- Reconstructions remain gray, soft, and noisy.

Interpretation:

- The model has learned some local denoising and footprint-conditioned shape
  structure.
- The model is not strong enough to generate clean images from pure noise.
- The next research step should not be "more epochs on the same tiny
  pixel-space DDPM." It should be one of:
  1. pretrained/latent image diffusion with OSM control channels,
  2. stronger conditional diffusion architecture and a larger proposal image
     dataset,
  3. use the current supervised/retrieval image path for Hunyuan while the
     generative branch matures.
