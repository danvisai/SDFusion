---
license: apache-2.0
tags:
  - 3d-generation
  - diffusion
  - sdf
  - architecture
  - buildings
  - vecset
library_name: pytorch
pipeline_tag: unconditional-image-generation
---

# Footprint-conditioned building massing — vecset diffusion

**Status: active research. Not a production model.** Every number below is measured on a single
48-building held-out set with one seed. Nothing here has been reproduced independently, and the
project's own hard acceptance criterion is **not yet met**.

This repository holds the weights, the evaluation harness results, and the reasoning trail for a
49M-parameter latent-set ("vecset") diffusion model that turns a **building footprint plus a height**
into a solid 3D mass. It is the geometry stage of a larger hybrid town-generation pipeline.

The interesting content here is not the checkpoint. It is the **chain of measurements that killed
four plausible approaches**, and a set of measurement traps that cost real GPU-hours to learn.

---

## TL;DR

- A **decoded-surface loss** — supervising the diffusion through a frozen mesh decoder instead of on
  latent MSE — is the largest single lever found. It bought **+0.029 3D IoU** for 10 GPU-hours, where
  **tripling training length bought +0.008** for 11.
- Correcting *where* in the noise schedule that loss is applied (t/T 0.40 → 0.55) produced the first
  **selective carve** any model in this project has achieved: on 29 of 48 buildings it removes surplus
  volume without eroding the shape.
- **It is bimodal.** On the other 19 buildings it produces hollow shells. This is a *reliability*
  problem, not a capability one.
- 🔑 **Latent distance does not predict decoded quality.** Spearman ρ = **+0.12** pooled across error
  families — wrong-signed. A latent-MSE objective cannot rank its own candidates. This one result
  invalidated a large amount of earlier evidence.

---

## Results

**Harness:** `eval_massing_arms.py`, 48 pinned held-out ids, all arms scored in one pass, sampling
strength s=0.5. Artifact: `massing_arms_eval_band240000.json`.

3D IoU is split into **`missing`** (shape the model failed to produce) and **`extra`** (surplus volume
it added). That split is the whole reason this project made progress — the aggregate hides which way a
model is failing.

| arm | fp-IoU | missing ↓ | extra ↓ | 3D IoU |
|---|---|---|---|---|
| ground truth | 1.000 | 0.000 | 0.000 | 1.000 |
| codec ceiling (encode→decode GT) | 0.997 | 0.000 | 0.001 | 0.999 |
| **blockout** (extruded footprint — the input) | 1.000 | 0.000 | **0.183** | 0.845 |
| surface-loss model (`v4`) | 0.962 | 0.002 | 0.191 | 0.838 |
| **band-fix model (`v5`, this release)** | 0.954 | 0.051 | **0.158** | 0.737 |

The band-fix row is a **median over a bimodal population**, which is misleading on its own. Split:

| subset | n | extra ↓ | 3D IoU |
|---|---|---|---|
| **solid** (`missing` < 0.15) | **29/48** | **0.149** | **0.833** |
| hollow | 19/48 | 0.162 | 0.353 |

![The bimodal failure — solid and hollow in the same checkpoint](images/band-fix-hollowing-montage.png)

*The two modes, same checkpoint. This is what a median of 0.051 `missing` against a mean of 0.244
actually looks like.*

![Final checkpoint — GT, input, surface-loss model, band fix](images/band-fix-240000-comparison.jpg)

*Final checkpoint @240k. Left to right: ground truth · blockout input · surface-loss model · band fix.
Row 2 is solid; the other two rows are hollow. Same buildings and camera throughout.*

### ⚠️ A selection effect in the headline number

The project's internal write-up reports the solid subset as *"extra 0.149 vs the blockout's 0.183 — a
19% surplus reduction."* **That comparison is not like-for-like.** 0.183 is the blockout's median over
all 48 buildings; 0.149 is the model's median over the 29 it happened to succeed on. Scoring the
blockout on **those same 29 ids** gives **0.169**.

The honest figure is **0.149 vs 0.169 — an 11.8% surplus reduction**, not 19%.

The effect is real and still the first selective carve this project has produced. It is smaller than
previously stated, and the subset is selected by the model's own behaviour, so it is a *conditional*
result: "when it works, it removes ~12% of the surplus."

### The training trajectory is non-monotonic — do not extrapolate it

| checkpoint | fp-IoU | missing | extra | 3D IoU |
|---|---|---|---|---|
| @190k | 0.912 | 0.777 | 0.135 | 0.195 |
| @220k | 0.903 | 0.773 | 0.145 | **0.200** ← renders showed shredded cages |
| @230k | 0.934 | 0.027 | 0.189 | **0.825** ← recovered |
| @240k | 0.954 | 0.051 | 0.158 | 0.737 |

![@220k — the near-stop point](images/band-fix-220000-comparison.jpg)

A stop was recommended at 220k. The very next checkpoint recovered to 0.825. This is the **second**
time a run in this project was nearly killed during a transient collapse — the earlier one went
0.719 → 0.657 → 0.532 → **0.840** across epochs.

🔑 **A 30,000-step window of catastrophic output is not evidence of a dead run in this model.**

---

## Method

**Task.** Footprint polygon (rasterised to a 64×64 mask) + height → solid 3D building mass.
⚠️ **Height is a user input, not inferred.** The contribution as it stands is *footprint + height →
mass*. "Footprint alone" is future work.

**Representation.** A frozen third-party **Dora-VAE** ([CVPR 2025, arXiv 2412.17808](https://arxiv.org/abs/2412.17808),
Apache-2.0) encodes each building surface into a latent **token set**, decoded by cross-attention at
arbitrary query points. Dora was chosen over a dense-grid codec because its **sharp-edge sampling** is a
direct, published fix for the exact failure mode this project was fighting — losing crisp edges at the
autoencoder level.

**Generator.** `VecsetDenoiser` — a DiT-style transformer over the token set:

| | |
|---|---|
| parameters | **49.4 M** (vs ~947 M for the superseded dense-grid model) |
| width / depth / heads | 768 / 12 / 12 |
| latent channels | 64 |
| conditioning | 64×64 footprint mask → 16 tokens, cross-attention; AdaLN for timestep |
| schedule | cosine ᾱ |
| training set | 34,909 LoD2 building meshes |

**Generation is projection.** The model does not synthesise from noise. It receives the extruded
footprint ("blockout") as its starting point and learns to **carve** it. The blockout is therefore a
*stage inside* the method, not a rival baseline — though it remains a valid ablation on whether the
learned step earns its compute.

### The decoded-surface loss (the actual contribution)

Standard latent-diffusion training regresses ε or x₀ in latent space. **Measurement showed that
objective cannot rank its own candidates** (see below), so the loss was changed to supervise through
the frozen decoder:

1. Predict x̂₀ from the noisy latent.
2. Decode x̂₀ at ~8,192 sampled query points via `DoraCodec(differentiable=True).freeze()`.
3. Regress SDF against ground truth **at those points**.

Design notes, both of which were error-corrected mid-project:

- **No eikonal term.** [StEik (NeurIPS 2023)](https://arxiv.org/abs/2305.18414) shows the plain eikonal
  loss becomes an unstable PDE as representation power grows. An earlier attempt in this project
  confirmed it directly: a gradient-TV term at w=0.1 **diverged into rubble**, because ε-error is
  amplified by 1/√ᾱ at high t.
- 🔑 **The noise band matters more than the loss weight.** The first run graded at t/T ≈ 0.40 while
  inference runs at 0.5–0.6. Grading near-clean latents taught the model to **copy its input** —
  `vs input` similarity 0.993. Moving to `--surf_t_center 0.55`, changing nothing else, produced the
  selective carve.

Cost is dominated by `decode`, so it scales with query count, not batch size: 8,192 points = 0.205 s
and 5.04 GB per step, against a 305 ms denoiser step. Freezing the decoder yields **0 parameter
gradients** while still passing gradient to the latent.

![Before and after the decoded-surface loss](images/surface-loss-before-after.png)

*The decoded-surface loss, before and after. Every metric column improved — the only lever in this
project that did.*

### Why a frozen third-party autoencoder

![Deployed dense-grid model vs Dora](images/deployed-vs-dora.png)

*Reconstruction quality: the superseded ~947M dense-grid codec against frozen Dora-VAE on the same
buildings. Measured 0.00328 vs 0.00552 surface roughness.*

---

## 🔑 What was ruled out, and how

This is the most transferable part of the work. Each of these closed a direction that looked
reasonable.

**1. Latent distance is decoupled from decoded quality — the objective cannot rank candidates.**
Spearman ρ of latent distance vs decoded 3D IoU, 144 candidates from two error families:

| pool | n | L2 | cosine |
|---|---|---|---|
| **pooled** | 144 | **+0.120** | −0.113 |
| on-manifold only | 24 | +0.050 | −0.188 |
| off-manifold only | 120 | −0.503 | +0.516 |

A distance metric wants **negative** ρ. Within one error family it works; **pooled — which is the
situation any real training run is in — it is worthless and slightly wrong-signed.**

The model-free version is starker: a latent at **cosine 0.083** decodes to IoU **0.999**, while one at
**cosine 0.995** decodes to IoU **0.053**. The 0.083 case is *the same mesh re-encoded* — furthest-point
sampling simply reorders the tokens.

![Decoder tolerance](images/decoder-tolerance-montage.png)

⚠️ **Consequence: any "the denoiser is working, cosine improved 0.707 → 0.935" evidence is void.**
This retroactively invalidated a substantial body of earlier project evidence.

**2. Post-hoc correction is exhausted — in both spaces.** An SDF-space refiner and a latent-space
corrector (zero-init residual 3D U-Net) both plateau at ~0.0047 roughness against a GT floor of 0.0041
and a codec ceiling of 0.0044. Driving latent L1 down 4× barely moved decoded roughness.

**3. The codec was never the bottleneck.** `decode(encode(GT))` scores 0.999 3D IoU. A crisp building
is fully representable. The diffusion samples latents that decode badly.

**4. Data is not the constraint.** All 35,623 meshes audited. ⚠️ Two findings that will bite anyone
touching the data: the corpus on disk is **inward-wound** (35,602 of 35,623) and only comes out correct
via `load_surfaces` — *never read the h5 directly*. And the meshes are **coarse, median 20 faces**,
which bounds what any sharpness supervision can teach.

**5. Training length is exhausted.** 180k steps / 41 epochs — 3× the scored run — bought +0.008 IoU.

**6. Building size does not explain the bimodality.** *(new, this pass)* Testing the obvious first
hypothesis for what separates the 29 solid from the 19 hollow: median GT volume 50,515 vs 43,554
voxels, **Mann-Whitney p=0.246, point-biserial r=0.186 (p=0.204)**. **Not significant at n=48.** Size is
ruled out; footprint complexity and source corpus remain untested.

---

## ⚠️ Measurement traps

These cost GPU-hours and near-miss false negatives. They are the highest-value content here.

| trap | what happened |
|---|---|
| **Never extrapolate the training curve** | 0.719 → 0.657 → 0.532 → **0.840**. Three monotonic points did not predict the fourth. A stop at the dip would have recorded a false negative. |
| **Always report `vs input`** | The generator scores near the blockout by *declining to act*. At s=0.45 it returned its input at **99.9%** and inherited its score. A model can look excellent while making no edit at all. |
| **Medians lie on bimodal outcomes** | Median `missing` 0.051, mean **0.244**. As a median alone this reads clean. The harness should report a **collapse rate**. |
| **The aggregate can be flat while geometry degrades** | 190k → 220k moved IoU 0.195 → 0.200 — "better" — while a building went from a box to a shredded cage. |
| **n=10 probes are not quotable** | Adjacent-checkpoint swing (0.59–0.78) is as large as any apparent trend. |
| **Quote the outcome, not the peak** | The pre-registered bar was met transiently at 9.5k (`extra` 0.178) and lost by 60k (0.191). Recorded **not-met**. |
| **Compare on the same ids** | See the selection effect above — a subset median against an all-population median inflated a result by 7 points. |

![The no-op failure mode](images/no-op-montage.png)

*The no-op. A model returning 99.9% of its input, scoring near the blockout because it inherited the
blockout's score.*

### Further renders

| image | what it shows |
|---|---|
| [`harness-baseline-montage.png`](images/harness-baseline-montage.png) | the 48-id harness baseline — all arms, one pass |
| [`convergence-run-montage.png`](images/convergence-run-montage.png) | the first convergence run that beat the deployed model |
| [`final-41epoch-montage.png`](images/final-41epoch-montage.png) | 41 epochs / 180k steps — where training length was exhausted |
| [`band-fix-230000-comparison.jpg`](images/band-fix-230000-comparison.jpg) | @230k — the cages filling back in after the collapse |

---

## Acceptance criteria — honest status

| criterion | weight | status |
|---|---|---|
| 1 — visual, human-judged: *"would you take the model's output over the extruded footprint?"* | **primary** | ✅ **passed** (human answered yes) |
| 2 — footprint match | **hard, non-negotiable** | ⚠️ **0.962, needs 1.000** |
| 3 — 3D IoU | diagnostic only | 0.737 median / 0.833 on the solid subset — **not a gate** |

⚠️ **Criterion 2 is the live gap and was under-weighted for an entire work cycle** in favour of
criterion 3, which the project's own specification marks as diagnostic. Anyone continuing this work
should weight footprint fidelity first.

---

## Literature positioning

**Family this belongs to — vecset / latent-set 3D diffusion.** The dominant recipe for open 3D
generative models since 2023.

| work | venue | arXiv | relevance |
|---|---|---|---|
| **3DShape2VecSet** | SIGGRAPH/TOG 2023 | [2301.11445](https://arxiv.org/abs/2301.11445) | Introduced the representation. Base recipe for everything below. |
| **Michelangelo** | NeurIPS 2023 | [2306.17115](https://arxiv.org/abs/2306.17115) | Vecset diffusion conditions cleanly on external modalities. |
| **CLAY** | SIGGRAPH/TOG 2024 | [2406.13897](https://arxiv.org/abs/2406.13897) | Explicit **3D-aware control from primitives** (voxels, boxes, point clouds). Direct precedent that footprint conditioning is a solved pattern, not a research risk. |
| **Direct3D** | NeurIPS 2024 | [2405.14832](https://arxiv.org/abs/2405.14832) | Continuous latent triplane — the taxonomy is not clean-cut. |
| **TRELLIS** | CVPR 2025 Spotlight | [2412.01506](https://arxiv.org/abs/2412.01506) | Sparse structured latents; pays for resolution only where geometry exists. |
| **Hunyuan3D 2.0** | Tencent 2025 | [2501.12202](https://arxiv.org/abs/2501.12202) | Vecset ShapeVAE + flow DiT. Uses **importance sampling on edges and corners** to preserve sharp detail. |
| **Dora** | CVPR 2025 | [2412.17808](https://arxiv.org/abs/2412.17808) | **The closest match to this problem.** Diagnoses that uniform point sampling loses sharp geometry; fixes it with sharp-edge sampling + dual cross-attention. Matches XCube-VAE with an 8× smaller latent. **Used frozen here.** |

**Sharp iso-surface extraction** (orthogonal, applies at export):
[DMTet](https://arxiv.org/abs/2111.04276) (NeurIPS 2021),
[FlexiCubes](https://arxiv.org/abs/2308.05371) (SIGGRAPH 2023),
[GET3D](https://arxiv.org/abs/2209.11163) (NeurIPS 2022, notably demonstrated on a buildings category),
[IGR](https://arxiv.org/abs/2002.10099) (ICML 2020, the eikonal term),
[StEik](https://arxiv.org/abs/2305.18414) (NeurIPS 2023 — the stability caution that shaped this loss design).

**Where this sits.** The vecset recipe, footprint conditioning, and frozen-VAE latent diffusion are all
established. What is *not* standard practice, and is the claim worth testing, is **supervising a latent
diffusion through its frozen decoder at a deliberately chosen noise band**, adopted here because
measurement showed the conventional latent objective could not rank its candidates. The band-placement
finding — that grading near-clean latents teaches input-copying — is the most transferable result.

⚠️ **Scale caveat.** Vecset models in the literature train on hundreds of thousands to millions of
shapes. This is 34,909. The sharp-edge-sampling *training signal* transfers at this scale; the
capacity/scale advantages CLAY and Hunyuan3D-2 report likely do not.

---

## Files

**Start here — the current line of work.** These five are scored on the 48-id harness and are what
every number above refers to.

| file | size | what |
|---|---|---|
| **`vecset_v5_surfband_step240000.pth`** | 189 MB | **the band-fix model** — final, scored (29/48 solid) |
| `vecset_v5_surfband_step230000.pth` | 189 MB | best 3D IoU (0.825); post-recovery |
| `vecset_v5_surfband_step220000.pth` | 189 MB | the collapse checkpoint — kept as evidence |
| `vecset_v4_surf.pth` | 189 MB | surface-loss model, pre-band-fix (+0.029 IoU) |
| `vecset_v3_pair_long_step180000.pth` | 189 MB | 41-epoch control, no surface loss |

**Historical — the latest checkpoint of every earlier run.** Included so no run is lost, *not*
because each is good. None of these are recommended starting points.

| file | size | what |
|---|---|---|
| `stage3a_lod2_deployed.pth` | 7.2 GB | superseded ~947M dense-grid baseline — comparison arm |
| `vecset_v2_pair_step60000.pth` | 189 MB | ⚠️ pre-frame-fix — trained on **transposed** latents |
| `vecset_v2_plain.pth` | 189 MB | ⚠️ pre-frame-fix — same defect |
| `vecset_v1.pth` | 189 MB | first vecset run |
| `vecset_pair_v1.pth` | 189 MB | first aligned-pair run |
| `vqvae_release_res64.pth` | 101 MB | released 64³ VQVAE codec (dense-grid era) |
| `vqvae_clean_ft.pth` | 101 MB | cleaned VQVAE fine-tune |
| `monolith_v1/v2/v3.pth` | 47 MB ea | monolith arms from the composition thesis |

⚠️ **The `v1`/`v2` vecset runs are void, not merely weak.** Their training cache had x and z
transposed, so they learned a **compensating axis swap**. Results from them cannot be compared to
anything after the frame fix. They are here for provenance only.

Optimizer state is stripped (checkpoints are inference/fine-tune ready, **not** resume-ready).
`latent_mu` / `latent_sd` are retained and **load-bearing** — the denoiser trains on globally
normalised latents and decodes to noise without them. Verify with `sha256sum -c SHA256SUMS`.

⚠️ **Keep global latent normalisation.** Per-channel normalisation was measured and is **harmful** —
the 16 low-variance channels are collapsed dimensions the decoder ignores.

## Reproduction

Code: <https://github.com/danvisai/SDFusion> (branch `massing-solid-gate-retrain`).
`REPRODUCING.md` covers clone → environment → data regeneration → verification. The corpus is
regenerated from **25 MB of committed identity + mesh data** rather than shipped: that 25 MB rebuilds a
35 GB SDF field and 17.4 GB of latent caches. Regeneration is **equivalent, not bit-identical**, and
yields 35,623 rows rather than 35,776.

## Attribution

The training corpus derives from three open government datasets, each carrying its own attribution
terms, which **any downstream use must honour**:

- **3DBAG** (Netherlands) — 3D BAG, TU Delft
- **NRW Open Data** (Germany) — Geobasis NRW
- **PLATEAU** (Japan) — MLIT Japan

The frozen autoencoder is **Dora-VAE** (Apache-2.0). Model weights here are released Apache-2.0; the
data terms above are separate and are not superseded by it.

## Limitations

- Bimodal: fails on ~40% of held-out buildings, cause unknown.
- Requires height as an input; does not infer it from footprint.
- Criterion 2 (footprint fidelity 1.000) not met — measured 0.962.
- Single seed, single 48-building held-out set, no independent reproduction.
- Trained on European and Japanese LoD2 building stock; no evidence it generalises beyond that.
- Meshes are coarse (median 20 faces), bounding achievable sharpness.
