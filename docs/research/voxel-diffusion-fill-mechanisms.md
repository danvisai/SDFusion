# How a diffusion model learns to fill voxels

**Research date:** 2026-08-19

**Question:** We intend to voxelize the massing, remove material, and recover it with a diffusion
model, then ask whether the result is a coherent structure. *How does such a model learn to fill?*
What do the cited papers do exactly?

**Source policy:** every mechanism below is taken from the paper's own full text (arXiv HTML/PDF) or
its official repository, not from an abstract. Where a paper does not state a number, this note says
so rather than guessing. Repository claims are linked to checked-in code.

**Scope:** representation research. This note changes no ADR and no code.

---

## Executive verdict

1. There are **four** mechanisms for making a diffusion model fill a region, not one. They differ in
   *where the knowledge of "known vs unknown" lives*, and they are not interchangeable.
2. The mechanism this project needs is **(2) train-time block perturbation**, as in DVD. It is the
   only one that trains on the same corruption it meets at inference, and the only one that gets
   exact preservation outside the mask for free.
3. **This repository already implements it.** `stage3a_model._build_context()` builds a random
   axis-aligned box mask, an emptied `known_body`, and a crude `primitive`, and injects them through
   a zero-init conv. It is gated off, and it carries a recorded negative — **at detail scale**
   (ADR 0001, "Layer-A context-snap"). It has never been tested at massing scale.
4. The binding risk is **not** the model class. It is the **copy incentive**. Every paper that fills
   successfully does so because its input genuinely lacks the answer. Ours does not, unless we make
   it so.

---

## The four mechanisms

| # | mechanism | where the mask lives | trained on masks? | preservation outside mask | example |
|---|---|---|---|---|---|
| 1 | **Sampling-time replacement** | sampler only | no | exact, by construction | TRELLIS region edit (RePaint) |
| 2 | **Train-time block perturbation** | training corruption *and* sampler | yes | exact, by construction | DVD (BSP) |
| 3 | **Control-branch conditioning** | a second encoder | yes, via paired partial/complete data | none — soft only | DiffComplete |
| 4 | **Coarse-as-prior / hierarchical upsample** | nowhere; there is no mask | n/a | n/a | ArchComplete, Seed3D 2.0 |

The distinction that matters: **(1) and (2) can guarantee that untouched geometry is untouched.
(3) cannot.** A control branch conditions on the partial shape but the denoiser is free to move
every voxel, so "the rest of the building survived" becomes a measurement, not a property.

---

## What each paper does, exactly

### DVD — Discrete Voxel Diffusion ([2605.07971](https://arxiv.org/abs/2605.07971))

The closest algorithmic precedent, and the only one whose editing mechanism is a first-class trained
capability rather than a sampler trick.

- **Representation.** Binary occupancy `X ∈ {0,1}^(N×N×N)` at **N = 64** — 262,144 positions, each a
  Bernoulli variable (K=2), sequence length L = N³. Identical grid to ours.
- **Formulation.** A *uniform-state* discrete diffusion model (USDM), **not** a masked/absorbing
  variant. Forward marginal `p_t(·|x₀; α_t) = Cat(x_t; α_t·x₀ + (1−α_t)·π)` with uniform prior
  `π = 1/K`. Loss is the Rao-Blackwellized negative ELBO, with a centered tanh-truncated logit
  parameterization for numerical stability.
- **Why discrete matters.** It treats occupancy as natively discrete and so "avoids
  continuous-to-discrete thresholding". Our pipeline thresholds a continuous field at `fld <= 0`;
  DVD's claim is that the threshold is where information dies.
- **How it learns to fill — Block-Structured Perturbation (BSP).** During fine-tuning it randomly
  selects **several axis-aligned hypercubes ("blocks") at multiple scales**, as a union with
  different side lengths, and corrupts only those. Because training saw exactly this corruption,
  inference needs no special algorithm: initialize the edit region with uniform noise, run the
  ordinary sampler, and the mask holds unmasked regions fixed throughout. The paper's claim is that
  this enables inpainting and editing "within a single sampling round, requiring negligible
  auxiliary computation and **no additional model evaluations**."
- **Sampling.** 256 steps for generation from scratch (~20 s on an A800); **128 steps for
  inpainting**.
- **Conditioning.** Image (6 renders/asset, yaw every 60°, pitch 30°, radius 2.5) and text, with
  classifier-free guidance scheduled as `0.7 if t>0.5 else 0.4` (image) and `1.0 if t>0.5 else 0.4`
  (text).
- **Scale.** ~400M parameters, 400K steps at batch 128 (image) / 600K (text), on TRELLIS's ~500K set,
  of which ~450K survived preprocessing.
- **What it does not establish for us.** Its target is a **sparse surface scaffold** over 450K general
  assets with image/text conditioning — not a filled LoD2 building solid conditioned on a footprint.
  The checkpoint is not reusable. **The recipe is.**

### ArchComplete ([2412.17957](https://arxiv.org/abs/2412.17957), [CAG 2025](https://doi.org/10.1016/j.cag.2025.104477))

The strongest domain precedent — it is the only one about buildings.

- **Representation.** Binary occupancy: "each voxel contains a value of either 0 or 1, representing
  an empty (void) or occupied voxel (mass)".
- **Stage 1 — coarse.** 3D **voxel VQGAN** at **64³** (voxel size ≈ 75 cm), codebook K=512, D=128,
  latent resolution r=8, 3D PatchGAN discriminator (receptive field 8). Loss = reconstruction + a
  novel **2.5D perceptual loss** (orthogonal xy/yz/xz projections through VGG-16) + commitment +
  adversarial, weights α=100, β=10, λ=0.25, δ=0.1. A **decoder-only transformer** (context window
  512 embeddings) predicts the next patch token autoregressively under cross-entropy.
- **Stage 2 — detailization.** A **four-level hierarchy (L=4)** of patch-based 3D-U-Net diffusion
  upsamplers, 8×8× total upsampling, 64³ → **512³** (≈ 9 cm voxels). Each level concatenates the
  subdivided coarse grid and the fine grid at the input layer, and conditions on `(coarse patches
  C^l, fine patches C^(l+1), timestep T)`. Patches are 6 m³ chunks voxelized at 8³/16³/32³/64³.
  Inference uses **one-fourth patch overlap**, blends overlapping predictions by averaging, and DDIM
  to cut T=1000 → **100** steps. ~150,000 coarse–fine patch pairs.
- **How it learns to fill.** Two different ways, and only the first is a mask:
  - *Shape completion*: apply a **block mask on half of an unseen voxel grid**, then use the
    autoregressive transformer to sample the tokens of the missing region, restricting the NLL to
    the remaining entries. This is next-token prediction, not diffusion.
  - *Detailization*: the diffusion upsamplers learn `p(fine patch | coarse patch)`. There is no
    mask — the coarse patch genuinely does not contain the fine geometry. The paper describes the
    upsamplers as learning to "iteratively subdivide coarse patches into octants and prune excessive
    ones", introducing openings and thinning walls.
  - *Plan-drawing completion*: 2D plans drawn in Goxel are treated as a partial 3D input and lifted.
- **Data.** A **3D House dataset** of **1500 models** (originals + augmentations), 1:1 scale, fully
  modelled interiors, from teaching at the University of Innsbruck.
- **What it does not establish for us.** It generates and completes *its own* house grids. It never
  shows a post-hoc editor improving the authentic output of a *different* generator, and its
  interior-modelled house domain is not LoD2 exterior massing. Note also 1500 models vs our corpus —
  the small-data regime ADR 0001 argues for.

### DiffComplete ([2306.16329](https://arxiv.org/abs/2306.16329), NeurIPS 2023)

- **Representation.** Distance/SDF grids at **32³** (`shapenet_dim32_df` / `shapenet_dim32_sdf`).
- **How it learns to fill.** A **ControlNet-style control branch** — the repo carries
  `epn_control_train.yaml` and `net/control_weights` and credits ControlNet directly. The incomplete
  shape goes through this separate branch, whose features are injected into the 3D-U-Net denoiser in
  a **spatially aligned** way ("hierarchical feature aggregation"). An **occupancy-aware fusion**
  strategy then lets several partial shapes condition one completion.
- **The critical property.** Conditioning is **soft**. Neither the paper nor the released code
  clamps observed voxels during sampling. The model is *asked* to respect the input; it is not
  *prevented* from moving it.
- **Data.** 3D-EPN, eight ShapeNet categories, paired complete/partial.
- **What it does not establish for us.** Partial-scan completion, commonly class-specific in the
  released config — not architecture-conditioned correction of a full but implausible building.

### Seed3D 2.0 ([2605.13862](https://arxiv.org/abs/2605.13862))

The paper the question named. It is **not** a filling paper — it is the strongest evidence for
deterministic voxelization as a *spatial anchor*.

- **Both stages are VecSet latents**, the same family as our A2/Dora path, compressed by a
  **locality-aware VAE** that consolidates tokens across spatial neighborhoods ("concentrating
  representational capacity in geometrically complex regions") and decodes through a
  content-adaptive sparse routing mechanism restricting each query to a compact token subset.
- **Stage 1.** A scaled-up Seed3D 1.0 DiT generates coarse VecSet latents from image conditioning,
  decoded to an intermediate mesh by DMC on a **sparse 512³ grid**.
- **Stage 2 — the part that matters here.** The coarse mesh is voxelized by **GPU-accelerated
  voxelization and morphological dilation** into a spatial occupancy prior. That prior is injected
  as **"voxelized positional encodings"**, which anchor each latent token to a spatial location and
  thereby promote structural regularity. Stage 2 also conditions on **partially diffused Stage 1
  latents** as a coarse geometric reference. Final extraction prunes query points hierarchically
  using the Stage 1 occupancy prior, up to **1536³**.
- **Objective.** Rectified-flow diffusion from Gaussian noise to structured latents.
- **The lesson, precisely.** Voxels are used as **position and conditioning**, never as the edited
  building state, and the voxelization is **deterministic**. This is exactly the seam
  [voxelizing-generated-massing.md](voxelizing-generated-massing.md) recommends.
- Parameter counts and dataset size are **not disclosed** in the paper.

### TRELLIS ([2412.01506](https://arxiv.org/abs/2412.01506), CVPR 2025)

- **SLAT.** Local latents on a sparse 3D grid at **N = 64**, ~**20K active voxels** on average; each
  active voxel stores `z_i ∈ ℝ^C` plus index `p_i`. Built from DINOv2 features aggregated over dense
  multiview renders and compressed by a sparse VAE.
- **Two stages, both rectified flow**, loss
  `L_CFM(θ) = E‖v_θ(x,t) − (ε − x₀)‖²`. Stage 1 compresses the binary 64³ occupancy grid to a
  low-resolution feature grid and a transformer `G_S` denoises it to the structure `{p_i}`.
  Stage 2's transformer `G_L` generates `{z_i}` given that structure. Decoders: 3D Gaussians
  (K=32/voxel, `x = p_i + tanh(o_i)`), radiance fields (CP-decomposed local 8³ volumes), or meshes
  (FlexiCubes, upsampled to 256³).
- **How it learns to fill — it doesn't.** Region editing is a **sampler-time adaptation of RePaint**:
  specify a bounding box, regenerate voxels/latents inside it, hold the rest fixed. Detail variation
  keeps the structure and regenerates latents under a new prompt. No mask is seen at training time.
- **Scale.** ~500K assets (Objaverse-XL, ABO, 3D-FUTURE, HSSD), 150 renders each; 342M / 1.1B / 2B
  variants; 64×A100, 400K steps, batch 256.

### TRELLIS.2 / O-Voxel ([2512.14692](https://arxiv.org/abs/2512.14692))

- **What each active cell stores.** Not occupancy — a **dual vertex position** `v_i ∈ [0,1]³`
  (sub-voxel), **edge-intersection flags** `δ_i ∈ {0,1}³` determining quad connectivity across three
  predefined edges, and PBR material attributes (base color, metallic, roughness, opacity).
- **Why.** This "Flexible Dual Grid" represents arbitrary topology — open, non-manifold, and
  fully-enclosed surfaces — free of watertight/manifold constraints. Sharp features survive via QEF
  optimization aligning dual vertices to intersection points, splitting weights for adaptive
  triangulation, and a boundary-distance term penalizing distance from open mesh edges.
- **Model.** Three sequential DiT flow-matching models (sparse structure → geometry → material),
  each ~**1.3B** parameters (width 1536, 30 blocks, 12 heads, MLP width 8192); 512³ → 1024³.
- **Conversion is deterministic and optimization-free**, "within tens of milliseconds".
- The paper does **not** discuss editing or completion.

### Faithful Contouring ([2511.04029](https://arxiv.org/abs/2511.04029))

- **Representation.** Faithful Contour Tokens per surface-intersecting voxel:
  `[voxel index, (anchor position, normal), dual anchor masks and positions, semi-axis intersection
  codes]`, where the semi-axis code records directed crossings in `{-1,0,1}^6` — **connectivity
  without a distance field**.
- **Deterministic GPU encoder, four stages.** SAT over 13 axes for active-voxel detection;
  Sutherland–Hodgman clipping for intersection centroids guaranteed inside the voxel; anchor fitting
  by regularized quadratic error `(MᵀM + λI)x* = Mᵀd + λc̄` via Cholesky, with normals by regularized
  SVD; Möller–Trumbore ray-triangle tests for semi-axis orientation `sign⟨n*, ê⟩`. Fully local, so it
  parallelizes.
- **Then** a VAE over those tokens (cascaded sparse 3D convs + light attention, 8× compression),
  supervising anchor positions (MSE), normals (cosine), semi-axis codes, dual masks, occupancy, KL.
- **Numbers.** Scales to **2048³**. At 1024³: Chamfer 0.01±0.01×10⁻⁴, **F-score 99.71%**. Against
  SparseFlex/SparC: **93% lower CD**, +35% F-score. UDF baselines produced double-layer artifacts;
  flood-fill SDF caused surface thickening and loss of internal structure.

### DOPH ([2407.11272](https://arxiv.org/abs/2407.11272))

Differentiable mesh→occupancy/SDF from solid angles and winding numbers, on GPU, differentiable
w.r.t. mesh vertices. Its own stated tradeoff is the useful part: **accurate near-binary occupancy
concentrates gradients near the surface, while a softer integral improves gradient flow at the cost
of occupancy accuracy.** Relevant only if an experiment needs gradients reaching mesh vertices.

### OCCDiff ([2512.08506](https://arxiv.org/html/2512.08506)) — not in the earlier review

Occupancy diffusion for **building** reconstruction from noisy point clouds. Occupancy as
`o: ℝ³→{0,1}`, resolution **80**, 1000 query positions per shape for the function autoencoder.
**Flow matching**, `L_FM = E‖(z₁−z₀) − h_θ(z_t,t,c)‖²`, conditioned by a DGCNN + transformer point
encoder (CD loss weighted η=1000). Trains on **Building3D** (15,890 labeled buildings, six cities)
and **Building-PCC** (~50,000 NL building models). Headline: Building3D CD_L2 3.20 / F-score 0.6899;
Building-PCC CD_L2 2.86 / F-score 0.6952.

Its relevance is the corpus, not the method: Building-PCC is Dutch, and our BAG/NL corpus is the
same source family. Its input is a point cloud, so it does not answer the editing question.

---

## The trap this project must design around

Every paper above that fills successfully shares one property: **the input genuinely does not
contain the answer.**

- DiffComplete's input is a scan with large holes.
- ArchComplete's block mask erases *half* an unseen grid; its upsampler's coarse patch physically
  cannot encode 9 cm geometry.
- DVD corrupts blocks to uniform noise before the model ever sees them.
- Seed3D 2.0's Stage 1 mesh is genuinely coarser than Stage 2's 1536³ output.

Our proposed input does not have this property. Measured on all 714 held-out buildings
([`massing_arms_eval_ship714.json`](../../execution/artifacts/massing_arms_eval_ship714.json)):

| source | 3D IoU | `vs input` |
|---|---|---|
| footprint envelope (blockout) | **0.9334** | — |
| A2 @ s=0.5 | 0.8756 | **0.9846** |
| codec ceiling | 0.9986 | — |

The envelope already sits at 0.9334 against the target. A model conditioned on it, trained to
reproduce the target, gets ~93% of the objective by **copying**. That is not a hypothetical: it is
what A2 does today at `vs input` 0.9846, and what arm A still does at step 230000 at **0.9951**,
with `beats_envelope_rate` **0.000** at all five scored checkpoints.

Swapping the loss for a diffusion ELBO does not remove this incentive. It removes the *excuse* — a
diffusion model can represent the multi-modality — but the gradient still points at copying.

**The fix is structural, and the papers already name it: destroy the information you want the model
to reinvent.** DVD's BSP and ArchComplete's half-grid block mask are not conveniences. They are what
makes the task well-posed. Applied here: mask a block of the *target*, and require the model to
regenerate it from the surrounding real body plus the footprint. Then `vs input` outside the mask is
1.0 **by construction** (so it stops being a metric to defend), and every claim concentrates on the
masked region, where copying is impossible.

This also flips the evidence value of a negative. A model that cannot rebuild a masked roof from the
rest of a *real* building has failed at something much cleaner than "A2 doesn't carve".

---

## What this repository already has

`models/stage3a_model.py` is a conditional 64³ SDF **latent diffusion** (DiffusionUNet, hybrid
conditioning, DDIM), conditioned on footprint (spatial `c_concat` + a frozen FootprintEmbedNet
global embedding), class, height and style — and it **already implements mechanism (2)**.

[`_build_context()`](../../models/stage3a_model.py#L433-L469) does, per training sample:

- pick a random axis-aligned box, sized `D/4 … D/2` per axis and **upper-biased** on y
  (`y0 ∈ [H/3, H−dy]`, deliberately targeting towers, roofs and dormers);
- `known_body = where(mask, +T, x)` — the target with the region **emptied to the truncation value**;
- `edit_mask` = the region;
- `primitive` = a crude solid bbox of the true body inside the region — the "crude placed mass";
- classify that primitive into a Layer-B element type;
- average-pool all three 64→16 to latent resolution and inject them through a **zero-init context
  conv** (the same trick ControlNet and DiffComplete use).

The docstring states the intent exactly: *"Teaches 'crude mass in region + surrounding body → the
coherent real element'."* That is DVD's BSP with a single scale, plus a coarse hint, written before
DVD published.

**It is gated off, and it has a recorded negative** — ADR 0001 lists "the Layer-A context-snap model"
alongside REPA/adaLN/L1-GAN as detailizers that "produced blurry geometry that could not carve
windows."

**Read that negative precisely.** Windows are *detail*: below `s* = 1.0 m ≈ 3 voxels @64³`
([ADR 0004](../adr/0004-experiment-operating-point.md)). ADR 0001's whole claim is that generation is
"well-posed for coarse massing but ill-posed for fine detail". Layer-A was asked to do the thing the
project's accepted thesis says is impossible, and it failed — **which is consistent with the thesis,
and says nothing about the same machinery above `s*`.**

So the question is not "should we build a masked voxel diffusion editor". One exists. The question
is whether it works in the band it was never aimed at. That is a far cheaper experiment than a new
model, and its result is informative either way:

- if masked massing regeneration works above `s*`, ADR 0001's boundary gains direct positive
  evidence, from the *same* architecture whose sub-`s*` failure is already cited as evidence;
- if it fails above `s*` too, the "well-posed for massing" half of the accepted thesis is in trouble,
  which is a finding worth more than another arm of #92.

---

## Recommended design

Ordered so each step can kill the next.

**Step 0 — do not voxelize a mesh.** A2 already decodes a signed field on a 64³ grid and reads
occupancy as `fld <= 0` before any mesh exists
([`town_generate_service.py`](../../scripts/server/town_generate_service.py#L236-L243)). Branch at the
field. Retain the narrow-band values alongside the binary occupancy: libigl's own comparison shows
serious aliasing when a distance field is clamped to an indicator before contouring, and TRELLIS.2 /
Faithful Contouring exist precisely because binary occupancy discards sub-voxel surface position.

**Step 1 — masked regeneration of *real* buildings, above `s\**`.** Un-gate Layer-A. Train on
`(real building with a block erased) → (real building)`, with the block sized to massing scale, not
detail scale. Adopt DVD's **multi-scale union of blocks** rather than the current single box, and
sample block placement to cover roofs, wings and setbacks. Hard-clamp everything outside the mask at
every sampling step. Report IoU **inside the mask only**; outside it, report the violation count,
which must be zero.

**Step 2 — only if Step 1 passes, introduce A2.** Replace the erased region's content with A2's
actual output there instead of `+T`, so the model learns "crude/wrong mass in region + real
surrounding body → coherent mass". This is the `primitive` channel already in the code, upgraded
from a bbox hint to an authentic wrong answer. This is also the step that depends on #92 choosing a
checkpoint — cache it before then and the pairs go stale.

**Step 3 — gates, fixed in advance.** Reuse the preregistered contract in
[voxel-editor-feasibility-prototype.html](voxel-editor-feasibility-prototype.html), which already
names the four failure modes (no-op, collapse, biased sample, global edit) and their thresholds. Add
the validity checks from [voxelizing-generated-massing.md](voxelizing-generated-massing.md): ground
contact, one connected component, no empty/solid collapse, fringe/spill/uncovered split at fixed
`s*`, watertight extraction. Score on the full n=714, never a prefix, and restore nonzero SNE views —
the watcher passes `--sne 0` ([`watch_checkpoints.py`](../../scripts/foundations/watch_checkpoints.py#L55)),
so surface quality is currently unmeasured, not measured-and-fine.

**What not to do.** Do not train a discrete diffusion model from scratch as step one; DVD is ~400M
parameters over 450K assets and its checkpoint is not reusable for filled LoD2 solids. Do not adopt
O-Voxel or FCT yet — they are surface representations, superb ones, but they are not filled building
solids and they answer a crispness question, not a filling question. Do not use a learned model for
the mesh→voxel conversion itself; that remains deterministic, per the earlier review.

---

## Compatibility with recorded decisions

Masked regeneration conditioned on footprint, height and the surrounding real body is a **transform**,
not sampling from noise, so it sits inside [ADR 0003](../adr/0003-two-claim-thesis.md)'s C1 claim —
the same operator does envelope→massing and user edits. Two variants would conflict and need an
explicit decision first: a pure noise-to-voxel generator replacing the envelope projection, and
treating the voxel mask or baked mesh as the editable building rather than the symbolic recipe.
