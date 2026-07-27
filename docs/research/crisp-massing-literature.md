# Crisp footprint-conditioned massing: literature review

**Research date:** 2026-07-26

**Scope:** the representation-level fix for lumpy/wavy SDF-diffusion massing, following
[map #52 "Crisp clean massing"](https://github.com/danvisai/SDFusion/issues/52) and
[map #58](https://github.com/danvisai/SDFusion/issues/58). This is a literature review to inform
the next architecture decision, not an implementation plan.

**Source policy:** every paper below was checked against its arXiv abstract, official project
page, or the publishing venue (not just a blog/aggregator summary) during this research pass.
Venue/year and arXiv id are given for each; anything I could not independently confirm is flagged
explicitly rather than stated as fact.

---

## Section A — our findings (recap)

Established by [#54](https://github.com/danvisai/SDFusion/issues/54),
[#55](https://github.com/danvisai/SDFusion/issues/55),
[#56](https://github.com/danvisai/SDFusion/issues/56), and
[#59](https://github.com/danvisai/SDFusion/issues/59) — see
[`representation-ceiling-menu.md`](../wayfinding/crisp-massing-model/representation-ceiling-menu.md)
and [`latent-corrector-result.md`](../wayfinding/diffusion-latent-accuracy/latent-corrector-result.md)
for the full numbers. Summary:

- **Stack:** 64³ truncated SDF (±0.2) → VQVAE → **16³×3 latent** (≈21× compression, codebook 8192)
  → latent diffusion (947M-param UNet, footprint-conditioned) → VQVAE decode.
- **The codec is not the bottleneck.** `decode(encode(GT))` = **0.0044** roughness, essentially at
  the GT floor of **0.0041**. A crisp building is fully representable at 64³ in this codec.
- **The diffusion is the bottleneck.** It samples latents that decode to **wavy** surfaces.
- **Post-hoc correction is exhausted, in both spaces it was tried.** An SDF-space refiner (#54)
  and a latent-space corrector (#59, zero-init residual 3D U-Net on the raw 16³×3 latent) both
  plateau at **~0.0047** roughness — visibly lumpy, well short of the 0.0041–0.0044 floor. Driving
  latent L1 down 4× barely moved decoded roughness at all, i.e. **latent distance is decoupled
  from decoded crispness** — nearby-in-latent-space is not the same as nearby-in-surface-quality.
- **Conclusion carried into this doc:** the fix has to change what the diffusion *produces* (or
  remove the requirement that it nail a dense per-voxel field at all), not correct its output
  afterward. Two directions were flagged: (1) cheaper — an x0-sharp/manifold objective finetune of
  the existing diffusion, grid and codec untouched; (2) durable — a query-based implicit
  (vecset/triplane) decoder, which is what Section B evaluates against the frontier.

---

## Section B — literature

### 1. Vecset / latent-set 3D shape diffusion

**What it is.** Instead of a dense voxel/latent grid, the shape is encoded as a **set of latent
tokens** (typically 256–2048 vectors, order-agnostic) via a cross-attention encoder over points
sampled on the surface. A decoder is a small cross-attention/MLP network that takes the token set
plus a **query point** and returns occupancy or signed distance — i.e. the field can be evaluated
at *any* (x, y, z), not just cells of a fixed grid. A transformer/DiT then does diffusion over the
token set. This is the dominant recipe behind essentially every strong open 3D generative model of
2023–2026.

| paper | venue/year | arXiv | verified detail |
|---|---|---|---|
| **3DShape2VecSet** | ACM TOG (SIGGRAPH) 2023 | 2301.11445 | Introduced the vecset representation itself: cross-attention encoder → self-attention latent set → cross-attention query decoder for occupancy/SDF. The base recipe everything below builds on. |
| **Michelangelo** | NeurIPS 2023 | 2306.17115 | SITA-VAE (vecset, aligned to a CLIP-like shape-image-text space) + ASLDM diffusion. Shows vecset diffusion conditions cleanly on external modalities — relevant precedent for injecting our footprint condition into a token-based model. |
| **CLAY** | ACM TOG (SIGGRAPH) 2024 | 2406.13897 | Multi-resolution VAE + "minimalistic" latent DiT on a vecset-like latent; explicitly supports **3D-aware controls from diverse primitives — voxels, bounding boxes, point clouds** as conditioning, not just text/image. Directly relevant: our footprint mask is exactly this kind of geometric control signal. |
| **Direct3D** | NeurIPS 2024 | 2405.14832 | D3D-VAE encodes shapes into a **continuous latent triplane** (not a pure vecset) + D3D-DiT. Sits between the vecset and triplane families — worth noting the taxonomy isn't clean-cut. |
| **TRELLIS** | CVPR 2025 Spotlight (Microsoft Research) | 2412.01506 | Structured LATent (SLAT): sparse 3D grid populated only where geometry exists, fused with dense multiview visual features; rectified-flow transformer backbone, up to 2B params. A **sparse-voxel** variant of the token-set idea — pays for resolution only where the shape actually is. A same-team Dec-2025 follow-up, **TRELLIS.2 / "Native and Compact Structured Latents for 3D Generation"** (arXiv 2512.14692), confirms this family is still the active frontier as of 2026 (O-Voxel representation + Sparse Compression VAE, 4B-param flow models). |
| **Hunyuan3D 2.0** | Tencent, Jan 2025 | 2501.12202 | **Already vendored in this repo (`external/Hunyuan3D-2`).** Hunyuan3D-ShapeVAE is explicitly built on the 3DShape2VecSet vecset recipe, decoding a queried SDF; a flow-based DiT (Hunyuan3D-DiT) does the diffusion. Confirmed via the paper text: it uses an **importance-sampling strategy that samples more points on edges and corners** specifically to preserve high-frequency/sharp detail — i.e. it targets the exact "sharp edges get lost" failure mode we're fighting. |
| **Dora** | CVPR 2025 | 2412.17808 | "Sampling and Benchmarking for 3D Shape Variational Auto-Encoders." Diagnoses that **uniform point sampling when training a vecset VAE loses sharp geometric detail**, and fixes it with a **sharp-edge sampling strategy + dual cross-attention** (separate attention paths for uniformly- and sharp-edge-sampled points, summed for the latent). Matches Dora-VAE reconstruction to the much larger XCube-VAE with an **8× smaller latent** (1,280 vs >10,000 codes). Also introduces Dora-bench, a benchmark keyed on sharp-edge density. **This is the closest literature match to our exact problem** — it is a name for, and a fix for, "the field loses crisp edges" at the *autoencoder* level. |

**Relevance verdict.** This family is real, current (2023 → active Dec-2025 follow-ups), and is the
"durable fix" our own escalation chain (#55 → #58 → #59) already points at. It structurally
removes the thing our own data indicts: a diffusion model being forced to hit an exact value in
every cell of a fixed, low-resolution dense grid. In a vecset model, spatial resolution is a
decode-time query, not a train-time grid size — crispness becomes a property of the query decoder
and its training data/loss (Dora and Hunyuan3D-2 both attack it via *sampling*, not more grid
resolution), which is exactly the lever #56 showed we don't currently have (our codec is already
crisp; our grid is already fine enough).

**Adoption cost, honestly.** This is not "swap in Hunyuan3D-2" — the project's own constraint is to
improve its own model, and a generic foundation model isn't footprint-conditioned or trained on our
LoD2/BuildingNet building corpora anyway. What transfers is the *recipe*, built into our stack:
(a) an encoder that point-samples building surfaces with a **Dora-style sharp-edge-aware sampler**
instead of our current dense-grid SDF encoding; (b) a query decoder replacing the current
transposed-conv VQVAE decoder; (c) a set-transformer/DiT diffusion backbone replacing the 3D UNet,
with the footprint injected as conditioning tokens (CLAY's "3D-aware primitive control" is direct
precedent that this is a solved conditioning pattern, not a research risk); (d) marching cubes (or
FlexiCubes, see family 2) at export time, same as today. This is a **new AE + new diffusion head**,
i.e. genuinely the "high cost" option already named in `representation-ceiling-menu.md`. The
project's own known data bottleneck applies here — though see the **#64 correction** below: the ~1849
figure is DETAIL-era; massing has **35,776** shapes. It still applies here
too or worse — vecset models in the literature are trained on hundreds of thousands to millions of
shapes (CLAY, Hunyuan3D-2). Our own real-data count doesn't grow just because the representation
changes, so the sharp-edge-sampling *training signal* (which needs no extra data, just resampling
existing meshes) is the part of this family most likely to pay off even at our data scale; the
raw model-capacity/scale advantages CLAY and Hunyuan3D-2 report are less likely to transfer.

### 2. Sharp iso-surface extraction

**What it is.** Given *some* scalar/implicit field (SDF, occupancy, UDF), how it gets turned into a
mesh matters for how sharp the result looks — classic Marching Cubes blurs/rounds features even
from a genuinely sharp field. This family is orthogonal to family 1: it's an export-time or
training-time-differentiable meshing step, not a generative-model architecture.

| paper | venue/year | arXiv | verified detail |
|---|---|---|---|
| **DMTet** | NeurIPS 2021 (NVIDIA) | 2111.04276 | Deformable tetrahedral grid encoding an SDF + a **differentiable marching tetrahedra** layer. Key idea: optimize directly for the reconstructed *surface* rather than regressing SDF values pointwise — sharper, fewer artifacts than value-regression alone. |
| **FlexiCubes** | SIGGRAPH 2023 (NVIDIA) | 2308.05371 | Extends Dual Marching Cubes with extra per-cube optimizable parameters (weights, per-vertex/edge offsets) so a *gradient-based* mesh optimization can represent sharp, feature-preserving geometry without the numerical instability plain Marching Cubes/Dual Contouring have in that regime. Shipped in NVIDIA Kaolin ≥0.15. |
| **GET3D** | NeurIPS 2022 (NVIDIA) | 2209.11163 | GAN generating two latent codes (geometry, texture) → DMTet extracts the surface → differentiable rasterization trains from 2D image collections. Notably **demonstrated on a buildings category** among its shape classes, and outputs an explicit textured mesh directly usable in a standard graphics engine (not just a renderable field). |
| **Eikonal / sharp-feature SDF losses** — foundational: **IGR** (Gropp, Yariv, Haim, Atzmon, Lipman), ICML 2020, arXiv 2002.10099 | Established the eikonal term (‖∇f‖=1) as implicit geometric regularization for learning SDFs from raw points, still the standard SDF-training regularizer today. | | |
| **StEik** | NeurIPS 2023 | 2305.18414 | Important caution for any "add an eikonal/sharpness loss and finetune" plan: shows the plain eikonal loss becomes an **unstable PDE as network representation power increases**, causing surface irregularities and sub-optimal minima — not more detail. Proposes a divergence/Laplacian regularizer plus quadratic layers as a stabilized alternative that captures **more** shape detail. |

**Relevance verdict.** This family doesn't fix a wavy field — it fixes how faithfully a field's
*already-present* sharp features survive into the mesh, and (DMTet/FlexiCubes) it can make the
extraction step itself part of a differentiable training loop. It's a genuine, low-to-medium-cost
lever to combine with whichever generative representation is chosen (family 1 or the current
grid), and the current pipeline's marching-cubes export could plausibly be swapped for
FlexiCubes-style extraction independent of the diffusion question. But per #56, our field is
already reconstructable crisply by the codec — the wave is injected by the diffusion sampling a
bad latent, not lost at extraction. **The StEik result is a direct, useful warning for the x0-sharp
finetune currently in flight**: if that finetune adds a naive eikonal-style sharpness term, StEik's
finding says it could destabilize training rather than sharpen it, and a divergence-based
regularizer would be the more defensible choice if an eikonal-family loss is used at all.

### 3. CAD / primitive / mesh-sequence generation

**What it is.** Represent a shape as an explicit construction program — a sequence of 2D sketches
extruded into solids (sketch-and-extrude), or an explicit boundary representation (B-rep) tree of
faces/edges/vertices — rather than any kind of sampled field. Geometry is crisp *by construction*
because edges and faces are explicit primitives, not implicit isosurfaces of a learned function.

| paper | venue/year | arXiv | verified detail |
|---|---|---|---|
| **PolyGen** | ICML 2020 (DeepMind) | 2002.10880 | Autoregressive: a vertex transformer, then a face transformer (pointer networks) conditioned on the generated vertices. Explicit mesh output, no field/marching-cubes step at all. |
| **DeepCAD** | ICCV 2021 | 2105.09492 | First model to generate a **CAD construction-sequence** (sketch + extrude operations) via transformer, trained on a released 178,238-model Onshape dataset. |
| **SkexGen** | ICML 2022 | 2207.04632 | Disentangled codebooks (topology / geometry / extrusion) for autoregressive sketch-and-extrude generation; supports mixing codes across samples for design exploration. |
| **SolidGen** | TMLR 2022 | 2203.13944 | Jayaraman, Lambourne, Desai, Willis, Sanghi, Morris (Autodesk). First model to directly synthesize B-rep CAD (not just a sketch-extrude program) autoregressively via pointer networks over an "Indexed Boundary Representation"; limited to prismatic shapes; can condition on class/image/voxel. |
| **BrepGen** | ACM TOG (SIGGRAPH) 2024 | 2401.15563 | Diffusion directly on a structured B-rep latent (hierarchical tree: solid → faces → edges → vertices), with duplicate-node merging to recover topology. First of this line to go **beyond prismatic** to free-form/doubly-curved surfaces. |

**Relevance verdict.** Buildings — especially the LoD2-scale massing this project targets — are
close to prismatic (footprint polygon + extrusion height, occasionally a pitched-roof extrusion).
This family would in principle give crisp-by-construction massing with zero waviness risk, since
there's no field to sample noisily in the first place. **But this repo already has this idea**, as
a hand/learned-parameter procedural path (`scene/sdf_recipes.py`,
`models/networks/recipe_param_diffusion.py`, per the existing architecture) — a footprint +
extrusion + parametric roof recipe is exactly a restricted sketch-extrude program. What the
DeepCAD/SkexGen/BrepGen line adds beyond that is *learned, data-driven* sequence generation instead
of a hand-authored recipe grammar — which would need a **building-scale CAD/B-rep sequence
dataset**. None of DeepCAD/SkexGen/SolidGen/BrepGen train on buildings (they use mechanical/Onshape
CAD parts); a from-scratch building sketch-extrude corpus doesn't obviously exist at the
hundred-thousand scale these methods were trained at, and this project's own known bottleneck is
scarce real building data generally (for *detail*; massing has 35,776 — see the #64 correction).
So this family is best read as **literature
validation that the project's existing procedural recipe direction is architecturally sound**, not
as a new investment — the marginal cost of building a learned CAD-sequence generator on top of it
is high and directly bounded by data availability, and it caps massing to prismatic forms, forfeiting
the free-form diversity (dormers, curved footprints, irregular massing) the SDF-diffusion prior was
meant to capture from real buildings in the first place.

### 4. Triplane representations

**What it is.** Factor a 3D field into three axis-aligned 2D feature planes (XY, XZ, YZ); a point's
feature is the sum/concat of the three planes' bilinearly-sampled features, decoded by a small MLP.
Standard 2D conv/diffusion architectures apply directly to each plane, giving much higher effective
spatial resolution than a dense 3D voxel grid at the same memory budget.

| paper | venue/year | arXiv | verified detail |
|---|---|---|---|
| **Rodin** | CVPR 2023 (Microsoft Research) | 2212.06135 | "Roll-out diffusion network": rolls the 3-plane NeRF triplane into one 2D feature map, does 3D-aware convolutional diffusion on it, for digital-avatar radiance fields (not SDF/mesh). A CVPR-2024/2024-era follow-up, **RodinHD** (arXiv 2407.06938), specifically targets higher-fidelity detail with a hierarchical triplane — evidence the family scales to fine detail when the training explicitly targets it. |
| **3DGen** | arXiv preprint, Meta AI, 2023 | 2303.05371 | Triplane VAE + conditional diffusion for **textured mesh** generation (SDF-based, not just radiance fields) — closer in spirit to our task than Rodin. *Note: I could not confirm a peer-reviewed venue beyond the arXiv preprint in this pass — treat as an arXiv-only reference.* |

**Relevance verdict.** Triplanes are a cheaper, more conv-friendly way to raise effective resolution
than a dense voxel grid, and are architecturally the smallest step from our current VQVAE/UNet
stack (2D convs instead of 3D, same diffusion-on-a-grid paradigm). But per #56, resolution/grid
coarseness is *not* our bottleneck — the diffusion still has to correctly fill a dense
(planar-projected) field, so triplanes inherit the same class of risk our data already shows: a
diffusion model asked to hit precise values over a dense representation. They're a plausible
**cheaper intermediate experiment** (more headroom than the current 16³×3 latent, less new
machinery than a full vecset rebuild) but not, on the evidence we have, the durable fix — they
raise the resolution ceiling without removing the "diffusion must nail a dense field" constraint
that #59 shows is where the correlation with crispness breaks down.

### Other vendored repos, for completeness

- `external/LGM` (Large Multi-View Gaussian Model, ECCV 2024, arXiv 2402.05054) and
  `external/DiffSplat` (ICLR 2025, arXiv 2501.16764) are multi-view-image-to-3D-Gaussian-splat
  pipelines — appearance/rendering representations, not editable solid-SDF massing generators.
  Not relevant to the crisp-*massing* problem; they matter to this project's separate appearance
  layer (gsplat_compose.py etc.), not to Section A's diffusion-crispness bottleneck.
- `external/MeshUDF` (ECCV 2022, arXiv 2111.14549) is a differentiable Marching-Cubes-for-UDFs
  method — relevant only if the project ever moves to an *unsigned* distance representation (e.g.
  for open/non-watertight surfaces); our current representation is a signed, truncated SDF, so
  ordinary Marching Cubes (or FlexiCubes, family 2) is the applicable extractor today, not MeshUDF.

---

## Section C — recommendation

Mapping back to the #52/#58 conclusion (**the diffusion, not the codec or grid, produces the
waviness; post-hoc correction in either SDF or latent space plateaus at 0.0047**):

1. **The durable fix is family 1 (vecset / query-based implicit decoding), built into our own
   stack.** It is the only family that structurally removes the actual indicted mechanism — a
   diffusion model forced to hit exact values on every cell of a fixed dense grid. Dora and
   Hunyuan3D-2 both independently name and fix *our exact symptom* (sharp edges lost to uniform
   sampling) at the autoencoder level, which is unusually direct literature support. It stays a
   genuinely editable SDF (the decoder still returns a signed distance for an arbitrary query
   point, so SDEdit-style local edits and marching-cubes export are preserved), satisfying the
   project's constraint. Cost is real: a new point-sampled encoder (with sharp-edge-aware
   sampling), a new cross-attention query decoder, and a new set-transformer/DiT diffusion
   backbone with footprint conditioning re-plumbed as tokens — a from-scratch AE + diffusion
   effort, not a finetune, and one that inherits this project's existing scarce-real-data
   constraint (the field's exemplars train on 100K–1M+ shapes; we have **35,776** for massing — see the
   #64 correction below, not the ~1849 detail-era figure).

2. **Family 3 (CAD/extrusion) is not a new recommendation — it's literature confirmation that the
   project's existing procedural-recipe path is the right instinct for the prismatic part of the
   problem.** Extending it with a *learned* sketch-extrude generator (à la SkexGen/BrepGen) would
   need a building-specific CAD-sequence dataset that doesn't appear to exist at scale; skip it as
   a new investment.

3. **Family 2 (FlexiCubes/DMTet, sharp-feature losses) is a cheap, orthogonal accompaniment** to
   whichever generative representation is chosen — swap the export-time meshing step, and heed
   StEik's warning if the in-flight x0-sharp finetune adds an eikonal-style term (prefer a
   divergence/Laplacian-style regularizer over a naive eikonal one, per StEik's own finding that
   plain eikonal loss is unstable at the representation power current networks operate at).

4. **Family 4 (triplane) is a plausible cheaper waypoint but not the durable fix** — it raises
   resolution/memory headroom relative to the current 16³×3 grid without removing the "diffusion
   must nail a dense field" property that #59's latent-corrector result shows breaks the
   correlation with crispness.

5. **Versus the cheaper x0-sharp diffusion finetune currently being tried:** that finetune keeps
   the current grid/codec entirely and only changes the diffusion's training objective
   (decoded-crispness/manifold-aware loss on the *same* dense 16³×3 latent, warm-started from the
   map-#24 weights) — low-medium cost, fast to test, and directly answers the open question of
   whether the grid representation itself caps achievable crispness or whether it was merely
   under-optimized. If it plateaus near the same ~0.0047 ceiling #54/#59 already hit, that is the
   experimental confirmation (beyond the representation argument above) that the dense-grid
   representation is the actual ceiling, and the vecset rebuild (item 1) becomes the well-evidenced
   next step rather than a speculative one.

---

## Section D — checked and rejected

Papers brought to this thread, verified against source, and found **not applicable**. Recorded so
they don't get re-litigated.

### Laplacian-regularized eikonal equation (Hahn, Mikula, Frolkovič) — arXiv 2301.11656, math.NA

**Checked:** 2026-07-27, against the arXiv API record and the ar5iv full text.

**What it is.** A classical cell-centered **finite-volume solver** for computing a distance field:
`−ε·Δu + |∇u| = 1` on `Ω∖Γ`, with Dirichlet `u = 0` on the object `Γ` and the **Soner condition**
`ν·∇u ≥ 0` on the *computational-domain* boundary to force selection of the viscosity solution on
non-convex domains. The regularization parameter is annealed with mesh size
(`ε_n = h_L^(1/(2n))`). Solved by algebraic multigrid with MPI domain decomposition. Stated
applications: wall distance in turbulence modelling, distance from a thin flame, medial-axis
transform for mesh generation, cardiac/seismic wave propagation.

**Why it does not help our crispness problem.**

1. **It takes the surface as input and never moves it.** `Γ` enters as a Dirichlet condition. Our
   indicted failure mode (Section A) is that the **zero level set itself is wavy**. Like any
   redistancing scheme, this fixes eikonality and shock/medial-axis structure *away* from the
   interface while holding the interface fixed — run on a decoded lumpy SDF it yields a cleaner far
   field around the same lumpy building.
2. **Nothing is differentiable or learning-based**, so it cannot be a loss term or a layer in the
   diffusion training loop — the only place Section C says a fix can land.
3. **The step it would improve is not indicted.** #56 measured GT roughness **0.0041** and codec
   round-trip **0.0044**: our SDF *computation* has essentially no headroom. We also use a
   **truncated ±0.2** field on a **regular 64³ grid**, so the paper's headline win — cheap accuracy
   far from the object, on unstructured polyhedra — is for a regime we discard.

**The one real conceptual link, already spent.** `−εΔu + |∇u| = 1` is the vanishing-viscosity idea
that **StEik** (2305.18414, family 2 above) ported into neural SDF training. #60 ran that neural
incarnation: the grad_tv-regularized x0-sharp finetune **diverged at w=0.1** and, with w=0.05 +
grad-clip, was **stable but flat** (0.00547 vs 0.00552 baseline). The only marginally new element
here is the **annealing schedule** (shrink the regularizer as resolution refines → a t-dependent
weight instead of our fixed `w`), but #60's problem was not instability — it was that stability
bought no crispness. That is a micro-tweak on a lever already measured flat, not grounds to reopen
[#58](https://github.com/danvisai/SDFusion/issues/58).

**Note on a near-miss ID:** arXiv **2301.11445** (one family away, same month) is
**3DShape2VecSet** — that one *is* relevant and is the head of family 1 / the Section C
recommendation. Don't confuse the two.


---

## Correction (#64, 2026-07-27) — the corpus figure, and what the literature actually says about scale

**The ~1849-shape figure used above is from the BuildingNet / detail-element era and does not apply to
massing.** [#26](https://github.com/danvisai/SDFusion/issues/26) established
`data/real_massing_v1/real.h5` = **35,776 real LoD2 buildings** (NL 11,776 / DE 12,000 / JP 12,000),
verified 2026-07-27. That is a **19× correction**, and it changes the feasibility argument materially.

**What the exemplars actually trained on** (verified against paper text this pass):

| model | training shapes | source |
|---|---|---|
| **3DShape2VecSet** (SIGGRAPH 2023) | **ShapeNet-v2, ~51K across 55 categories**, trained *jointly* (not per-category) — AE reaches **IoU 0.965** | paper §experiments |
| **Dora** (CVPR 2025) | **~400,000** meshes filtered from Objaverse | *"Our training data consists of approximately 400,000 3D meshes carefully filtered from Objaverse."* |
| **TRELLIS** (CVPR 2025) | **TRELLIS-500K** (Objaverse XL, ABO, 3D-FUTURE, HSSD, Toys4k) | project repo |
| **Hunyuan3D-2** | Objaverse / Objaverse-XL scale; **exact count not published** | paper does not quantify |

**No paper in this family publishes a training-set-size ablation or scaling curve for the AE.** Dora
ablates only its architectural contributions (Sharp Edge Sampling, Dual Cross Attention). So "is N shapes
enough" cannot be answered by citation — only by the two indirect results below.

**The sobering result.** Dora attributes 3DShape2VecSet's weakness directly to data:
*"3DShape2VecSet consistently underperforms across all detail levels, primarily due to its **limited
training data** affecting generalization capability."* 3DShape2VecSet trained at **~40K** — essentially
**our corpus size**. So a vecset AE trained from scratch at our scale is *published* to underperform on
sharp detail.

**The mitigating result, from the same paper.** Fine-tuning a pretrained vecset VAE onto a new corpus is
**standard, published practice**: *"we fine-tune Craftsman-VAE on our dataset (denoted as
Craftsman-VAE†) since both Craftsman-VAE and 3DShape2VecSet were originally trained on smaller
datasets."* This is the remedy for exactly our situation.

**And our own counter-evidence.** Dora's data-hunger finding is about reconstructing **55 diverse
Objaverse categories**. We need **one narrow category**. [#56](https://github.com/danvisai/SDFusion/issues/56)
already showed our *own* VQVAE, trained on our *own* 35,776 buildings, round-trips at **0.0044 ≈ GT
0.0041** — i.e. at our scale, in our domain, an autoencoder already reconstructs crisply. Narrow domain
appears to compensate for raw count on the AE half.

**Consequence for the effort:** the durable fix does not require training a vecset AE from scratch. See
[#64](https://github.com/danvisai/SDFusion/issues/64) and
`docs/wayfinding/crisp-massing-vecset/data-scale-findings.md`.
