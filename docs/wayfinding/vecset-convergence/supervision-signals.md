# #72 — Which supervision signals make implicit generators sharp

**Date:** 2026-07-29 · ticket
[Research: which supervision signals make implicit generators sharp](https://github.com/danvisai/SDFusion/issues/72)
· map [#69](https://github.com/danvisai/SDFusion/issues/69)

**Source policy.** Claims are marked **[code]** (read from the vendored source in this repo — the
strongest evidence, because it is the exact component we adopted), **[paper]** (verified against paper
text), or **[inferred]** (my argument from mechanism, not a citation). One web-sourced claim was checked
against vendored code and found **wrong**; it is flagged below rather than quietly dropped.

---

## The headline

**Almost every sharp-detail supervision technique in this literature belongs to the *autoencoder*
training stage — and we are not running that stage.** We froze the codec. Our diffusion's entire
supervision is `mse_loss(pred, noise)` on latent tokens (`scripts/train_vecset.py:166`) **[code]**, and
the codec's encode/query are both wrapped in `torch.no_grad()` (`models/shape_codec.py:152, 160`)
**[code]** — so no surface, no normal, no query point and no decoded field can reach our training signal
even in principle.

The second headline is better news: **on the encoder-input side we already implement Dora's recipe
faithfully.** `scene/surface_sampling.py` produces exactly what Dora's own preprocessing produces. There
is no missing input signal to add.

So the honest answer to *"what data improves our generation"* is **not a new channel**. It is:

1. a **metric** this literature has and we concluded didn't exist (§5a),
2. a **loss decomposition and model-selection rule** we could adopt at whichever stage we train (§5b),
3. and the recognition that our diffusion's supervision is latent-only, which our own #59 already
   showed is decoupled from surface quality (§5c).

---

## 1. What the exemplars actually consume

| | **Dora** (CVPR 2025) | **Hunyuan3D 2.0** |
|---|---|---|
| encoder input | uniform **+** sharp-edge point streams, **6 channels each: `xyz` + `normal`** **[code]** | 3D coordinates **and normal vectors** **[paper]** |
| target field | **SDF** (`pysdf`) in the released data prep **[code]**; the training system supports **both** SDF (`MSELoss`) and occupancy (`BCEWithLogitsLoss`) as a config switch **[code]** | **SDF**, meshed by marching cubes **[paper]** |
| query points | near-surface (multi-scale jitter) **+** uniform in `[-1.05, 1.05]³` **[code]** | "randomly sampled points in the space and shape surface" **[paper]** |
| loss | **two separately-weighted terms**, `loss_sharp_logits` + `loss_coarse_logits`, plus KL **[code]** | `MSE(D(x\|Z), SDF(x)) + γ·L_KL` **[paper]** |
| latent | 1,280 codes — **8× smaller** than XCube-VAE (>10,000) at comparable quality **[paper]** | max token length 3,072 **[paper]** |

⚠️ **Correction to a web-sourced claim.** A literature-summary read of the Dora paper asserted the
encoder takes *"coordinates only, no surface normals"* and that the decoder predicts *occupancy*. **Both
are wrong for the released implementation.** `shape_autoencoder.py:65-66` carries the comment
`# xyz + normal` on both encoder arguments, and `sharp_sample.py` writes every surface array as 6
channels and computes targets with `pysdf`. This is why the vendored code, not the summary, is the
authority for a component we actually adopted — and it retires the apparent Dora-vs-Hunyuan
disagreement: **both feed normals, both regress a signed field.**

---

## 2. The mechanism, precisely (Dora, from `external/Dora/sharp_edge_sampling/sharp_sample.py`)

All **[code]**:

- **Sharp-edge detection is a dihedral-angle threshold.** Blender's
  `bpy.ops.mesh.edges_select_sharp(sharpness=radians(angle_threshold))`, threshold supplied on the CLI
  (`--angle_threshold`, documented in-source as *"specify dihedral angle threshold"*).
- **The split is 50/50.** `num_target_sharp_vertices = point_number // 2`; each stream is then
  independently reduced by farthest-point sampling (`fpsample.bucket_fps_kdline_sampling`). The paper's
  16,384 salient of 32,768 dense matches this.
- **Sharp edges get densified by interpolation, not just sampled.** If a mesh has fewer unique sharp
  vertices than the target, new points are interpolated *along* each sharp edge until the quota is met.
  Sharp features are guaranteed a fixed share of the budget regardless of tessellation.
- **The normal assigned at a sharp edge is the bisector** — `0.5·n_f1 + 0.5·n_f2`, renormalised. At a
  crease the supervision is deliberately the *average* of the two faces, not either one.
- **🔑 Query points are jittered at multiple scales, and sharp regions get a finer, denser shell:**

  | region | jitter σ | count basis |
  |---|---|---|
  | **sharp** near-surface | **0.001, 0.005, 0.007, 0.01** — four scales | 4 × sharp point count |
  | coarse near-surface | 0.001, 0.005 — two scales | 2 × 200,000 |
  | free space | uniform in `[-1.05, 1.05]³` | 200,000 |

  The extra scales exist **only** around sharp features. This is the actual sharp-detail mechanism, and
  it is a **data-preparation** decision — no architecture change, which makes it cheap.
- **Meshes with no sharp edges are silently skipped entirely.** The pipeline assumes sharp features
  exist. (Harmless for LoD2 massing — flat faces meeting at right angles are nothing but sharp edges —
  but it tells you what the method is built around.)
- **Watertight preprocessing is a required, separate stage** (`to_watertight_mesh.py`): normalise to the
  unit box by bbox centre and `2.0 / max_extent`, evaluate a UDF on a 512³ grid over `[-1.05, 1.05]³`,
  re-extract with `diso` DiffMC/DiffDMC at `eps = 2/resolution`.

**Ablation, Level-4 (most complex) shapes, 1,280 codes** **[paper]**:

| variant | F-score(0.01) | Chamfer | **SNE** |
|---|---|---|---|
| without SES and DCA | 97.890 | 6.432 | 1.828 |
| with SES | 99.170 | 5.265 | **1.579** |

Sharp-Edge Sampling carries the result: **−13.6% SNE, −18% Chamfer**, from a sampling change alone.
⚠️ The Dual-Cross-Attention row extracted identically to the full model, so **DCA's isolated
contribution is not reliably readable** from my extraction — do not quote a number for it. Treat SES as
the evidenced lever and DCA as unquantified.

---

## 3. What we already have

`scene/surface_sampling.py` **[code]** already implements the Dora recipe on the input side:

| Dora | ours |
|---|---|
| dihedral-angle sharp-edge selection | `sample_sharp` via `mesh.face_adjacency_angles`, `deg` threshold |
| normal at a crease = mean of adjacent faces | same — *"normal = mean of the two adjacent face normals"* |
| 6-channel `xyz + normal` surface points | `(n, 6)` as `[x, y, z, nx, ny, nz]`, both streams |
| 50/50 sharp/uniform | `sample_streams(n_coarse=8192, n_sharp=8192)` |
| — | **`ensure_outward`**, which Dora has no equivalent of |

**There is no missing encoder input.** "Do we need normals?" — **we already feed them, on both streams,
and their correctness is safety-critical**: an encoder eats face normals, so inverted winding degrades
it without erroring. That is trap #2 in the handover, and `ensure_outward` is our guard against a class
of bug Dora's own pipeline does not defend.

**What we do not have** is any of the *target*-side machinery, because we never train a decoder:
per-query SDF targets, the multi-scale sharp jitter shell, the split sharp/coarse loss, or a
differentiable path from latent to surface.

---

## 4. Rulings on each signal named in the request

| signal | ruling | why |
|---|---|---|
| **Normals** | ✅ **already in** — nothing to add | 6-channel streams, both exemplars, our sampler matches. Verify orientation, don't add the channel. |
| **Sharp-edge samples** | ✅ **already in** — nothing to add | dihedral selection + bisector normals + 50/50 split all present. |
| **Normal maps** | ❌ **out as an input** · ✅ **in as a metric** (see §5a) | Genuinely a geometry signal — Wonder3D and Era3D generate multi-view normal maps precisely to supply *"additional geometrical supervision"* **[paper]**. But that is a workaround for **image-conditioned** pipelines that have no 3D ground truth. We have full 3D surfaces for all 35,623 buildings; a rendered normal map is a **lossy 2D re-encoding of information we already hold**, so as an input it is strictly worse than the surface itself. **[inferred]** |
| **Height maps** | ❌ **out as the output representation** · ⚠️ **unresolved as auxiliary conditioning** | Map #52's ruling stands and applies to the *output*: a 2.5D height field breaks editable-SDF carving. As auxiliary *conditioning or an auxiliary loss* the ruling does not obviously reach — but the literature that uses a height-map intermediate for buildings is **GeoTexBuild**, which needs a hand-drawn height sketch plus a text prompt, performs **no geometric evaluation at all**, and **concedes direct-3D is better but cost-prohibitive** (21,100 A100-hours). Nothing there recommends adopting it. **[paper]** |
| **UVs** | ❌ **out** | A parametrisation for mapping texture onto a *finished* surface. Carries no information about where the surface is. **[inferred]** |
| **Ambient-occlusion maps** | ❌ **out** | AO is *computed from* geometry — a baked visibility integral. Feeding it back supplies no information the geometry did not already contain, and for a generator it is not even available at input time. **[inferred]** |

---

## 5. The three things not on the list, which matter more than anything on it

### 5a. 🔑 Dora's SNE is the crisp-vs-rough metric we concluded does not exist

**Sharp Normal Error** **[paper]**: render normal maps from **22 viewpoints**, find salient regions by
**Canny edge detection** on those maps, **dilate** the masks, then take **MSE between ground-truth and
reconstructed normal maps inside the masks only**.

This is the direct answer to a problem this project has hit three separate times and written off:

- **#36** — two normal-consistency metrics failed to separate crisp from rough, both ~0.99,
  `separation_ok: False` → gate declared visual-montage-primary.
- **#63** — the teacher scored 0.00295 against GT's 0.00278 while *visibly worse*, heavy rippling.
  *"Our roughness metric is blind to this artifact class."*
- **#68 / deployed-vs-Dora** — `surface_roughness` ranked a melted blob (0.00571) **above** a crisp
  ribbed box (0.00818). Anti-correlated with the goal.

Every one of those failures has the same cause: **the metric averages over the whole surface, so
low-frequency melting — which is what destroys architecture — barely registers.** SNE fixes exactly
that by *masking to the salient regions* before measuring, and it measures **normals**, which respond to
flatness and crease sharpness rather than to field curvature.

It is also cheap: we already render montages, and buildings are the ideal case for Canny-on-normals
because their salient regions are long straight creases.

**This does not replace criterion 1.** The map's rule stands — the human judges the montage. SNE's value
is as a *second* instrument that, unlike `surface_roughness`, is not actively lying, and can therefore
rank checkpoints on the ladder without a human in the loop for every one.

**[inferred]** that it will separate our cases — SNE was validated on Objaverse, not on buildings. It
should be validated on the one pair we know the answer to: the melted blob vs the crisp ribbed box from
`deployed-vs-dora.png`. If SNE ranks those correctly where `surface_roughness` inverted them, it is
trustworthy here.

### 5b. Sharp regions get their own loss term — and their own model-selection rule

Dora does not merely sample sharp regions more densely; it keeps them as a **separate loss with its own
weight** (`loss_sharp_logits` / `lambda_sharp_logits` vs the coarse pair) **[code]**. And
`validation_step` returns **`{"val/loss": out["loss_sharp_logits"]}`** **[code]** — **the model is
selected on the sharp-region loss alone, not the overall loss.**

That second part is nearly free and applies to *any* stage we train, including our diffusion as it
stands. Choosing checkpoints on an aggregate loss lets a model that is smooth-everywhere beat one that is
sharp-where-it-matters. It is the same failure as §5a, moved from evaluation into model selection —
and it may partly explain why "loss still descending" has not translated into visual gains.

### 5c. Our diffusion is supervised only in latent space — and we have already measured that this is decoupled from surface quality

`scripts/train_vecset.py:166` is `mse_loss(pred, noise)` **[code]**. Nothing else. Meanwhile
[#59](https://github.com/danvisai/SDFusion/issues/59) established, in the dense-grid stack, that
**driving latent L1 down 4× barely moved decoded roughness** — *"latent distance is decoupled from
decoded crispness."*

There is no reason that decoupling is specific to the dense grid; it is a property of optimising a
distance in latent space and hoping it maps to a distance on the surface. **[inferred]** If it holds for
the vecset latent too, then **enriching the input data cannot help, because our training signal never
touches the data** — the latents were precomputed once, and the loss compares two noise tensors.

The lever this implies is a **decoded / surface-space loss term for the diffusion**. It is currently not
merely unimplemented but *structurally blocked*: `DoraCodec.query` runs under `torch.no_grad()`
**[code]**, so there is no gradient path from a decoded surface back to the denoiser. Unfreezing that
path is a real cost and a real decision — which is why this ticket surfaces it rather than assuming it.

**Adjacent, worth knowing:** staged VAE training with a **decoder-only second-stage fine-tune** is
reported to improve fine detail and mesh smoothness at far less cost than training at higher resolution
**[paper, snippet-level — not verified in full text]**. If we ever do fine-tune Dora's decoder on our
corpus, that is the shape to copy, and §2's machinery is what would finally become active for us.

---

## 6. Answer

**No new input signal is needed.** Normals and sharp-edge samples — the only two items on the list that
the literature actually supports as geometry inputs — are **already implemented and already fed**. UVs
and AO carry no geometric information. Normal maps are a real geometry signal but only as a substitute
for 3D ground truth we already possess. Height maps remain out as an output representation, and the one
building-domain paper that uses them argues against itself.

**What the literature does offer us, in descending order of value:**

1. **SNE as a second metric** (§5a) — masked, normal-based, salient-region-only. Directly addresses
   three documented failures of our scalar metrics. Cheap. Validate it on the montage pair whose answer
   we already know.
2. **Model selection on a sharp-region loss rather than an aggregate** (§5b) — nearly free, applies to
   our diffusion today.
3. **The sharp/coarse split supervision, multi-scale jitter shell, and watertight preprocessing** (§2) —
   all real and all cheap, but they only become active **if we train a decoder**. Frozen, they do
   nothing for us.
4. **A decoded-surface loss term for the diffusion** (§5c) — the one change that would make any data
   enrichment matter at all, and the one with a genuine cost: unfreezing the query path.

**The reframe worth carrying:** this map asked *"what data improves our generation"*. The evidence says
our data is not the constraint — **what our training signal is allowed to see** is. We hold 35,623
buildings with normals and sharp edges, precompute them into latents once, and then optimise a distance
between two noise tensors.

## Sources

- Dora — [arXiv 2412.17808](https://arxiv.org/abs/2412.17808) · [project page](https://aruichen.github.io/Dora/) · **vendored: `external/Dora/`**
- Hunyuan3D 2.0 — [arXiv 2501.12202](https://arxiv.org/html/2501.12202v1) *(outputs are evidence only — licence §5.b)*
- Wonder3D — [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Long_Wonder3D_Single_Image_to_3D_using_Cross-Domain_Diffusion_CVPR_2024_paper.pdf)
- Era3D — [arXiv 2405.11616](https://arxiv.org/html/2405.11616)
- COD-VAE, *Representing 3D Shapes with 64 Latent Vectors* — [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Cho_Representing_3D_Shapes_with_64_Latent_Vectors_for_3D_Diffusion_ICCV_2025_paper.pdf)
- Prior in-repo survey, not duplicated here: `docs/research/crisp-massing-literature.md`,
  `docs/wayfinding/crisp-massing-vecset/data-scale-findings.md`
