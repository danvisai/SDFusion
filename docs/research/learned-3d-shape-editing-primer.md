# How learned 3D shape editing actually works

**Research date:** 2026-08-22
**Question:** Can a model learn to move, remove, add, or stitch voxelized geometry, and can its
modification process be understood?
**Scope:** a beginner-accessible mechanism primer based on papers, author project pages, and official
code. This note changes no implementation or architectural decision.

## Short answer

Yes, but “the model moves voxels” conflates four genuinely different mechanisms:

1. **Diffusion/denoising changes field values.** A voxel or latent cell is repeatedly re-estimated as
   occupied, empty, or near/far from the surface. There is usually no persistent cube whose path can
   be followed through space. The intermediate denoising frames are computation states, not a learned
   construction explanation.
2. **A deformation field actually moves spatial samples.** It assigns a displacement or velocity to
   a point and transports the point through space. This gives correspondence and meaningful motion,
   but a smooth one-to-one deformation normally preserves topology: it does not naturally split one
   component, cut a hole, or create a new part.
3. **Occupancy/SDF editing changes where matter exists.** Cutting changes occupied to empty, or
   changes an SDF's sign; adding does the reverse. Masked completion can learn which new matter is
   plausible from surrounding context. This changes topology, but it does not say *why* the change is
   a courtyard, roof cut, wing, or passage.
4. **Part-delta and CSG/program models predict an executable explanation.** The learned output is
   “modify this part, delete that part, add this part” or a sequence of union/subtract operations.
   That is the closest match to a model that understands a modification in an editable, inspectable
   sense.

These mechanisms can be combined. For this repository, the strongest combination is a learned
**semantic architectural edit program** (the durable state) with deterministic SDF/CSG
**realization**, while a masked occupancy/SDF prior proposes or repairs local geometry. A final mesh
remains a derived output, not the building.

## First: what representation is being edited?

| Representation | What is stored | What “move/add/remove” means | Main limitation |
|---|---|---|---|
| **Explicit occupancy grid** | One binary state per fixed cell, `V[i,j,k] in {0,1}` | Remove: `1 -> 0`; add: `0 -> 1`. “Move” usually means clear one cell and occupy another, not track an object | Resolution-limited and blocky; cells have locations but not semantic identities |
| **Voxelized SDF/TSDF** | A signed scalar at each grid sample; normally negative inside, positive outside, zero on the surface | The surface moves as scalar values change; addition/removal changes which samples are negative | Still a sampled field; an edit has no name or operation identity |
| **Continuous occupancy/SDF field** | A function queried at arbitrary coordinates, such as `f(p) -> occupancy` or signed distance for `p = (x,y,z)` | Change the latent code, warp query coordinates, or correct the field; extract an isosurface afterward | Smooth geometry is easy, explicit edit history is absent |
| **Polygon mesh** | Vertex coordinates, edges, and faces | Move vertices directly; cutting/stitching must also change connectivity and often remesh | Fixed-connectivity deformation is poor at topology changes |
| **Part tree / CSG / CAD program** | Parts or primitive operations, parameters, relations, and order | Modify/delete/add a part, or execute union/intersection/subtraction | Vocabulary limits the shapes it can express; multiple programs may make the same final solid |

An **occupancy grid** and an **occupancy field** are not synonyms. The first is a finite array. The
second is a learned function that can be queried between grid points. Likewise, SDFusion's input and
output are voxelized T-SDFs, but its diffusion runs on a compressed VQ latent grid rather than on
named mesh vertices or individually tracked voxels. The original paper describes the 3D VQ-VAE and
latent noise-prediction objective explicitly. [SDFusion paper](https://arxiv.org/html/2212.04493)

For an SDF using the negative-inside convention, familiar Boolean operations are deterministic:

```text
union(A, B)        = min(d_A, d_B)
intersection(A,B)  = max(d_A, d_B)
subtract(A, B)     = max(d_A, -d_B)
```

That is why SDF/CSG is attractive here: a learned model can decide *which* operation and parameters
to use, while an exact evaluator decides which material remains.

## The four mechanisms in detail

### 1. Iterative denoising: coherent change, not literal voxel motion

#### SDEdit and SDFusion

[SDEdit](https://sde-image-editing.github.io/) supplies the conceptual mechanism behind this
repository's C1 transform. It adds a chosen amount of noise to a rough guide and runs a pretrained
reverse diffusion/SDE from that intermediate noise level. A low starting noise level preserves more
of the guide; a high level gives the prior more freedom and usually more realism. No edit-specific
training pairs are required by SDEdit itself. Its score model was trained to estimate how noisy data
should change toward the data distribution, not to recover the user's true sequence of actions.

[SDFusion](https://yccyenchicheng.github.io/SDFusion/) transfers the relevant idea to 3D shape
fields. Its learning has two stages:

1. A 3D VQ-VAE compresses a voxelized T-SDF and is trained with reconstruction, codebook, and
   commitment losses.
2. A 3D U-Net receives a noised latent and a timestep and minimizes noise-prediction MSE. Separate
   encoders and cross-attention inject partial-shape, image, and text conditions; classifier-free
   guidance changes their strength. The decoder maps the final latent back to a T-SDF, from which a
   surface is extracted. [Paper, Sections 3.1–3.3](https://arxiv.org/html/2212.04493)

What has the model learned? Repeated noisy examples teach it a vector toward latents typical of its
training shapes. The sequence can make a roof emerge or a hole disappear, but this does **not** mean
one roof voxel travelled there from somewhere else. At every denoising step the network can revise
all values, and structures may appear or vanish.

This is also why a reverse-diffusion animation is not by itself an explanation. It visualizes the
optimizer/sampler trajectory. To claim an understood edit process, the output needs persistent
correspondence, an explicit delta, or executable operations.

#### DVD: direct binary voxel editing

[DVD (Discrete Voxel Diffusion, 2026)](https://arxiv.org/html/2605.07971) is the clearest recent
example for literal `64^3` binary grids. Each cell is a Bernoulli variable. Forward corruption
randomly reassigns cells toward a uniform binary state; the network predicts a categorical
distribution for each clean cell and is trained by a negative diffusion ELBO. This avoids decoding a
continuous value and thresholding it merely to decide occupied versus empty.

For editing, DVD fine-tunes with **block-structured perturbations**: unions of axis-aligned blocks at
several scales are corrupted while the surrounding grid stays largely clean. During inference, the
edit region is sampled and the known region is replaced with its fixed categories after each step.
The important lesson is not merely “use diffusion”; it is “train on the same local corruption the
model will see during editing, and clamp state that must not change.” The authors release generation
and BSP editing checkpoints on their [official model card](https://huggingface.co/Zhengrui/dvd).

Two caveats matter here:

- DVD's voxels form a **sparse surface scaffold** for a later TRELLIS stage, not a filled building
  solid. Its checkpoint is therefore not a drop-in building editor.
- Exact preservation outside a mask does not give semantic understanding inside it. A sampled void
  is still anonymous unless another representation calls it a courtyard, passage, or roof cut.

The repository already has a deeper comparison of masking, completion, and clamping mechanisms in
[How a diffusion model learns to fill voxels](voxel-diffusion-fill-mechanisms.md).

#### Adjacent “voxel editing” methods: useful, but a different state

[Vox-E](https://tau-vailab.github.io/Vox-E/) reconstructs each object from posed views as a grid of
learned rendering features, then optimizes an edited grid per object with 2D diffusion score
distillation. A 3D correlation regularizer preserves the source, and lifted cross-attention masks
localize edits. [Easy3E](https://ustc3dv.github.io/Easy3E/) instead starts from TRELLIS sparse
voxels/structured latents and an edited target view plus a 3D edit mask; its rectified Voxel FlowEdit
changes sparse geometry in one feed-forward pass, followed by SLAT repainting and texture refinement.
These are important mesh/asset editors, and Easy3E is especially relevant to recent flow-based
editing, but neither produces a semantic operation trace or directly trains a filled building-solid
occupancy editor. Their “voxels” carry neural rendering or sparse surface-latent state rather than
this repository's dense absolute filled occupancy.

### 2. Deformation and correspondence: this is actual motion

[DIF-Net](https://github.com/microsoft/DIF-Net) represents a category using a shared continuous
template SDF plus, for each shape, a deformation field `v(x)` and scalar correction `delta_s(x)`.
The deformation maps a query point into template space. Because two shapes map to the same template,
the method obtains dense correspondence without part or correspondence annotations. The correction
field changes the SDF value where deformation alone cannot express an added or missing structure.
[CVPR paper](https://openaccess.thecvf.com/content/CVPR2021/html/Deng_Deformed_Implicit_Field_Modeling_3D_Shapes_With_Learned_Dense_Correspondence_CVPR_2021_paper.html)

Its losses reveal what “learning motion” means:

- SDF regression makes the reconstructed field match sampled ground-truth distances;
- surface-normal and deformation-smoothness terms encourage meaningful alignment;
- a minimal-correction prior asks the deformation to explain shared structure and reserves the
  scalar correction for real structural differences;
- latent regularization organizes the per-shape code.

At edit time the shape latent is optimized so selected points reach user-provided targets while
other terms preserve the original shape. Here a handle really does correspond to spatial locations.
The correction field can also add a structure by changing SDF values rather than stealing and
warping unrelated surface points.

[ShapeFlow](https://research.google/pubs/shapeflow-learnable-deformations-among-3d-shapes/) makes the
motion interpretation even more literal: a network decodes a continuous flow that advects a source
shape toward a target, with options such as bijectivity, no self-intersection, and volume
preservation. That makes it useful for interpolation and rearrangement. Those same guarantees expose
its boundary: a bijection preserves topology, so a clean cut or a newly stitched component requires
an occupancy/SDF correction or a discrete operation in addition to the flow.

### 3. Learned additions, removals, and stitching

#### StructEdit: learn the difference itself

[StructEdit](https://geometry.cs.ucl.ac.uk/projects/2020/structedit/) is the most direct match to the
intuition “show the model how an object changed.” Source and modified shapes are hierarchical part
assemblies. Their **shape delta** contains three explicit kinds of change:

- a delta for a matched/modified part;
- a deleted part;
- an added part.

A conditional VAE recursively encodes and decodes this delta while conditioning corresponding
subtrees on the source shape. Delta reconstruction terms teach the decoder which edit occurred; KL
regularization makes a sampleable edit space. Sampling produces plausible variants of a given
source, and an edit encoded from one source can be transferred to another. Official
[code and data](https://github.com/daerduoCarey/structedit) are available.

This is substantially closer to “understanding the modification” than endpoint-only SDF regression,
because the latent represents a *change*, not just another whole shape. It still has limits:
parts are represented through fine-grained part trees with semantic categories and bounding-box or
point-cloud geometry, and the training deltas are derived between related shapes rather than logged
human construction histories. A building version would need an architecture-specific operation
ontology and canonical matching rules.

#### SALAD: part mixing and completion with diffusion

[SALAD](https://salad3d.github.io/) combines part structure with diffusion. Each implicit part has a
low-dimensional **extrinsic** description (a 3D Gaussian's position, covariance, and blend weight)
and a high-dimensional **intrinsic** latent describing local geometry. An occupancy decoder combines
the parts into a queried field. SALAD first denoises the set of extrinsics, then denoises intrinsic
latents conditioned on those extrinsics; both stages use the usual diffusion noise-prediction MSE.
[ICCV paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Koo_SALAD_Part-Level_Latent_Diffusion_for_3D_Shape_Generation_and_Manipulation_ICCV_2023_paper.pdf)

Because parts, rather than one monolithic field, are the units, reverse-process guidance supports
part completion, part mixing, and refinement without a separate conditionally trained model. It is a
good demonstration of “stitch these parts and make the whole coherent.” The discovered Gaussian
parts are not architecture labels or reversible Boolean operations, however, so SALAD is a useful
part-prior precedent rather than the repository's final state model.

### 4. Programs and CSG: predict an executable explanation

[CSGNet](https://hippogriff.github.io/CSGNet/) takes a 2D or 3D target and recurrently emits a compact
program of primitives and recursive Boolean operations. Executing the program produces the shape;
subtraction is an explicit cut and union is an explicit addition. On synthetic shape/program pairs,
the recurrent decoder learns token prediction from the known programs. For novel shapes without
program labels, the authors also use policy gradients with similarity between the executed result and
target as reward. [CVPR paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Sharma_CSGNet_Neural_Shape_CVPR_2018_paper.html)

This output is inspectable and editable in a way a final voxel grid is not. It also exposes a central
learning problem: many different programs produce the same solid. A high-overlap program is not
necessarily the real or most meaningful construction history. [PLAD](https://openaccess.thecvf.com/content/CVPR2022/html/Jones_PLAD_Learning_To_Infer_Shape_Programs_With_Pseudo-Labels_and_Approximate_CVPR_2022_paper.html)
improves learning without ground-truth programs using executed pseudo-programs, but its authors also
note that reconstruction quality alone does not ensure good editable program structure.

Two companion works show useful ends of the program spectrum:

- [ShapeAssembly](https://rkjones4.github.io/shapeAssembly.html) learns hierarchical programs of
  connected cuboid parts, relations, and symmetries; executing and parameter-editing those programs
  changes structure, but its language is primarily assembly rather than subtractive carving.
- [DI-PCG](https://github.com/TencentARC/DI-PCG) is a recent diffusion version of inverse procedural
  modeling. A small DiT denoises the *parameter vector of a known procedural generator*, conditioned
  on an image, and then runs that generator deterministically. Training pairs come from rendering
  sampled generator parameters. It is especially relevant to recipe-preserving systems, though its
  output can express only what the supplied generator can construct.
  [CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhao_DI-PCG_Diffusion-based_Efficient_Inverse_Procedural_Content_Generation_for_High-quality_3D_CVPR_2025_paper.pdf)

The repository's broader architecture-specific novelty risks and program literature are already
catalogued in [Solid-first subtractive modeling: novelty survey](../wayfinding/solid-first-subtractive-modeling/NOVELTY_SURVEY.md).

## What supervision teaches—and what it cannot teach

| Training signal | What the model can justifiably be said to learn | What remains unknown |
|---|---|---|
| A collection of final shapes | A prior over plausible final geometry | Which edit happened, its order, intent, or locality |
| `(partial shape, complete shape)` | How missing geometry is distributed given visible context | Whether the completion is a named/valid design operation |
| `(source shape, target shape, explicit delta)` | A distribution of modifications conditioned on a source | The human's intent unless the delta carries semantic labels |
| Corresponding source/target points or a learned template | How locations move or correspond | Topology-changing creation/deletion unless separately modeled |
| `(shape, executable program)` | How to predict operations whose execution makes the shape | A unique “true” history; equivalent programs remain possible |
| Ordered authoring histories | The observed command order and dependencies | Whether that history is canonical or transfers to a new domain |

This distinction is crucial for designing data. Pairing a rough and final voxel grid and minimizing
reconstruction loss does not magically reveal “remove this roof block, then add a wing.” Unless the
target includes a displacement, delta, action, or program, the network is rewarded only for the
endpoint. The repository's [Why pair training does not by itself teach carving](why-pair-training-does-not-carve.md)
develops that point for the current building task.

For a model whose process is genuinely inspectable, collect or synthesize records like:

```text
(current recipe, current solid, rough user gesture, site constraints)
    ->
(typed operation, parameters, affected operation IDs, validity result, next recipe)
```

If gradual motion matters, add intermediate displacement fields or action states. If only the final
source and target are available, deformation can be learned through alignment and reconstruction,
but the intermediate path is one regularized plausible path—not observed ground truth.

## Practical reading order

1. **SDEdit**, then **SDFusion** — understand the repository's transform: noise weakens an input guide
   and a learned prior reconstructs a more likely field. Keep asking “which values change?” rather
   than imagining moving cubes.
2. **DIF-Net** — see the missing concept: correspondence and a displacement field make motion
   meaningful, while its correction field handles structures that motion cannot.
3. **StructEdit** — the best first paper for the user's cut/stitch intuition because the learned
   object is explicitly a modified/deleted/added-part delta.
4. **CSGNet** — move from a structural delta to an executable union/subtract explanation, and notice
   the ambiguity of recovering a program from final geometry.
5. **DVD** — return to explicit `64^3` voxels and study how masked corruption plus clamping makes
   localized add/remove editing work without claiming semantic operations.
6. **SALAD** or **DI-PCG**, depending on the goal — SALAD for coherent part mixing via recent
   diffusion; DI-PCG for diffusion whose output remains a procedural, editable parameter state.

Read Vox-E and Easy3E afterward if the goal expands from solid geometry to text/image-guided asset
and appearance editing; keep their feature/SLAT volumes distinct from filled occupancy.

## Best fit for this repository

There are two different “best fits,” because the repository intentionally has two constraints.

### Best fit for C1 transform experiments: masked field correction

SDEdit/SDFusion is already the direct C1 lineage: the same rough-input-to-realistic-output prior can
serve footprint-envelope generation and user sculpt transformation. A DVD-style block corruption
experiment on the settled dense absolute binary `64^3` whole-volume state is the cleanest explicit
voxel test. It can make add/remove behavior local by construction and reveal per-cell uncertainty.
That state is recorded as an in-progress competing empirical route in the repository
[context](../../CONTEXT.md), not as a replacement for the semantic program.

This remains an **experimental transform representation**. It produces absolute occupancy or an SDF,
not recipe closure. It also needs building-specific training and hard evaluation of footprint spill,
uncovered area, connectivity, minimum thickness, and unrelated-region preservation. DVD's general
sparse-surface checkpoint does not answer those questions.

### Best fit for durable editable state: a semantic program delta

The symbolic-recipe constraint makes a StructEdit/CSG/DI-PCG-like interface the stronger long-term
answer:

```text
rough footprint or existing recipe + user add/subtract gesture
                         |
                         v
 learned decision: KEEP / MODIFY / DELETE / ADD typed architectural operations
                         |
                         v
 deterministic SDF/CSG realization + architectural/geometric validators
                         |
                         v
 derived occupancy, SDF, preview mesh, and final mesh
```

The learned model should predict a delta in the repository's **semantic architectural edit program**:
operation type, constrained volume parameters, support/containment relations, and affected operation
IDs. The executor performs exact union/subtraction; validators reject violations. Unrelated
operations retain identity, so an edit is reversible and local.

A field prior can still contribute in three bounded roles:

- propose plausible geometry inside a user-selected region;
- score or rank candidate program realizations for building-likeness;
- supply a residual suggestion that is either converted into a named operation or discarded at the
  explicit recipe-compatibility gate.

This reconciles the mechanisms: diffusion supplies the C1 manifold prior; the program records the
meaning and preserves editability; deterministic realization makes the geometry. It also respects
the repository's accepted doctrine that learned models make **decisions** while exact procedures do
the **realization**.

## Curated primary sources

| Work | Representation | Most useful idea here | Official material |
|---|---|---|---|
| SDEdit, ICLR 2022 | Generic continuous guide/data | Noise level controls faithfulness versus prior-driven realism | [Project/paper/code](https://sde-image-editing.github.io/) |
| SDFusion, CVPR 2023 | Voxelized T-SDF compressed to VQ latents | Conditional latent-field denoising and multimodal completion | [Paper](https://arxiv.org/html/2212.04493), [code](https://github.com/yccyenchicheng/SDFusion) |
| DIF-Net, CVPR 2021 | Continuous template SDF + deformation + correction fields | Literal correspondence/motion plus field-based structural addition/deletion | [Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Deng_Deformed_Implicit_Field_Modeling_3D_Shapes_With_Learned_Dense_Correspondence_CVPR_2021_paper.html), [code](https://github.com/microsoft/DIF-Net) |
| StructEdit, CVPR 2020 | Hierarchical part delta | Modified, deleted, and added parts as the learned object | [Project](https://geometry.cs.ucl.ac.uk/projects/2020/structedit/), [code/data](https://github.com/daerduoCarey/structedit) |
| CSGNet, CVPR 2018 | Executable primitive/Boolean program | Interpretable union/intersection/subtraction prediction | [Project/paper/code](https://hippogriff.github.io/CSGNet/) |
| SALAD, ICCV 2023 | Gaussian part extrinsics + implicit part latents | Cascaded part diffusion for completion, mixing, and refinement | [Project](https://salad3d.github.io/), [paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Koo_SALAD_Part-Level_Latent_Diffusion_for_3D_Shape_Generation_and_Manipulation_ICCV_2023_paper.pdf) |
| DI-PCG, CVPR 2025 | Procedural-generator parameter vector | Denoise editable parameters, then execute a deterministic generator | [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhao_DI-PCG_Diffusion-based_Efficient_Inverse_Procedural_Content_Generation_for_High-quality_3D_CVPR_2025_paper.pdf), [code](https://github.com/TencentARC/DI-PCG) |
| DVD, arXiv 2026 | Explicit binary `64^3` sparse voxel scaffold | Native categorical occupancy, block-perturbation editing, exact clamping | [Paper](https://arxiv.org/html/2605.07971), [official checkpoints](https://huggingface.co/Zhengrui/dvd) |
