# GenerativeTowns / SDFusion: related work and quality-improvement research

**Research date:** 2026-07-13

**Scope:** 3D building and town generation; SDF/implicit generation and editing;
structured building parts; OpenStreetMap conditioning; facade, mesh, texture, and
Gaussian-splat realization; and evaluation.

**Source policy:** external claims use original papers, author project pages, official
repositories, official datasets, or official specifications. The repository observations below
are based on the checked-in code, plans, and generated artifacts, not on external descriptions.

## Executive findings

1. The repository has a coherent and unusually broad technical thesis: use one SDEdit-style
   transform for footprint-to-massing and sculpt repair, then realize learned architectural
   decisions through reversible procedural operations or retrieved real elements. This is
   accurately summarized in [CONTEXT.md](../../CONTEXT.md) and formalized in
   [ADR 0003](../adr/0003-two-claim-thesis.md).

2. The current outputs prove more about **constraint retention and reversible workflows** than
   about visual building quality. The transform experiment improves median footprint IoU from
   0.30 to 0.61 according to [tickets.md](../../tickets.md), but
   [its montage](../../outputs/transform_vs_noise/montage.png) still contains disconnected,
   perforated, or over-smoothed massing. The full-data monolith likewise matches average occupancy
   but remains visibly fragmented in [its montage](../../outputs/monolith_v3/montage.png).
   These are not contradictions: a footprint/occupancy metric can pass while surface topology and
   architectural plausibility fail.

3. The most immediate learned-detail problem is not lack of element vocabulary; it is **set and
   relationship coherence**. The planner output in
   [metrics.json](../../outputs/part_layout_planner_v2/metrics.json) substantially overproduces
   several categories (for example, sampled vs. ground-truth mean windows are 12.87 vs. 2.73 for
   commercial and 13.12 vs. 4.58 for religious buildings), while
   [the montage](../../outputs/part_layout_planner_v2/layout_montage.png) shows cluttered and
   weakly regularized boxes. Structure-aware generation and facade grammars address this failure
   more directly than another independent part classifier.

4. The closest recent systems mean that “learned decisions + procedural realization” or “compose
   assets rather than synthesize everything” are no longer sufficient novelty claims by themselves.
   [BuildingBlock](https://arxiv.org/abs/2505.04051),
   [CityCraft](https://arxiv.org/abs/2406.04983),
   [UrbanWorld](https://arxiv.org/abs/2407.11965), and
   [Proc-GS](https://openaccess.thecvf.com/content/CVPR2025W/USM3D/html/Li_Proc-GS_Procedural_Building_Generation_for_City_Assembly_with_3D_Gaussians_CVPRW_2025_paper.html)
   all combine learned planning with procedural or asset-based construction. GenerativeTowns'
   strongest defensible distinction is the combination of one transform for generation and editing,
   per-building recipe closure, leakage-safe real-element retrieval, and a matched equal-data
   monolith-versus-composition experiment.

5. The best near-term quality path is therefore: strengthen evaluation first; make the part model
   relational and grammar-constrained; consume more of OSM's explicit 3D tags; add block-level town
   context; and treat Gaussian splats as a recipe-linked appearance representation rather than a
   replacement for editable geometry.

## 1. Repository technical map

The project contains several historical and experimental paths, but the active design can be read
as the following stack.

| Layer | Main modules | Role and present design |
|---|---|---|
| Input and map semantics | [`scene/extract_osm.py`](../../scene/extract_osm.py), [`scripts/server/footprint_image.py`](../../scripts/server/footprint_image.py) | Converts OSM or a raster mask into per-building polygon, class, and height. Current OSM use is much narrower than the full Simple 3D Buildings schema. |
| Symbolic massing decisions | [`models/networks/recipe_param_diffusion.py`](../../models/networks/recipe_param_diffusion.py), [`scripts/server/recipe_inference.py`](../../scripts/server/recipe_inference.py) | Conditional diffusion over a compact recipe-parameter vector; deterministic recipes turn parameters into SDF geometry. |
| Learned implicit massing | [`models/vqvae_model.py`](../../models/vqvae_model.py), [`models/stage3a_model.py`](../../models/stage3a_model.py), [`models/sdfusion_model.py`](../../models/sdfusion_model.py) | A VQVAE encodes 64-cubed SDFs; conditional latent diffusion supplies a real-building massing prior. Stage 3a conditions on footprint, class/style, height, and region-related data paths. |
| Transform/edit operator | [`scripts/server/refine.py`](../../scripts/server/refine.py), [`scene/sdf_edit.py`](../../scene/sdf_edit.py) | SDEdit-style partial noising and denoising projects a footprint blockout or user edit toward the learned prior. Local masks preserve untouched regions and the edit list remains recipe state. |
| Structured detail | [`models/networks/part_layout_planner.py`](../../models/networks/part_layout_planner.py), [`models/networks/part_set_refiner.py`](../../models/networks/part_set_refiner.py), [`scripts/server/layout_detail.py`](../../scripts/server/layout_detail.py) | Proposes typed part boxes and attempts to re-cohere them after edits. Learned placement is followed by deterministic snapping, regularization, retrieval, and SDF composition. |
| Detail realization | [`scene/composer_detail.py`](../../scene/composer_detail.py), [`scene/sdf_detail.py`](../../scene/sdf_detail.py), [`scene/element_lib.py`](../../scene/element_lib.py), [`scripts/server/element_fit.py`](../../scripts/server/element_fit.py) | Procedural windows/doors/roofs and real BuildingNet crops are instantiated as editable SDF operations. |
| Geometry and export | [`scene/sdf_primitives.py`](../../scene/sdf_primitives.py), [`scene/mesh_cleanup.py`](../../scene/mesh_cleanup.py), [`scripts/server/town_export.py`](../../scripts/server/town_export.py) | Analytic SDF/CSG, grid sampling, marching cubes, cleanup, placement, and glTF export. |
| Appearance | [`scripts/appearance/texture_bake.py`](../../scripts/appearance/texture_bake.py), [`scripts/server/neural_appearance.py`](../../scripts/server/neural_appearance.py) | Multi-view SDXL/ControlNet appearance, UV back-projection, heuristic PBR maps, and neural town renders. Latest flow tests report about 160 seconds for one textured building and 237 seconds for a two-building textured export. |
| Gaussian experiments | [`models/stage3b_model.py`](../../models/stage3b_model.py), [`models/networks/sdf_to_gs_lifter.py`](../../models/networks/sdf_to_gs_lifter.py), [`scene/gsplat_compose.py`](../../scene/gsplat_compose.py), [`scene/gsplat_guardrail.py`](../../scene/gsplat_guardrail.py) | Lifts SDF plus conditioning into voxel-slot Gaussians, composes sets, renders views, and culls Gaussians outside footprints. This is a promising appearance branch but is not yet the project's strongest validated path. |
| Data and experiments | [`datasets/bag3d_dataset.py`](../../datasets/bag3d_dataset.py), [`datasets/buildingnet_dataset.py`](../../datasets/buildingnet_dataset.py), [`datasets/monolith_pair_dataset.py`](../../datasets/monolith_pair_dataset.py), [`scripts/eval/`](../../scripts/eval) | 3D BAG supplies watertight massing; BuildingNet supplies categorized shapes and parts; sealed splits and neutral evaluation support the C1/C2 thesis. |

This map also explains why a generic text/image-to-mesh foundation model is not a task-matched
replacement. The system's primary contract is a metric footprint plus editable symbolic decisions,
not a single appearance image.

## 2. What the current outputs actually say

### 2.1 Strong results to preserve

- The server and sculpt regression suites validate a real product property: localized edits preserve
  untouched volume, detail operations survive snapping, exports preserve scale, and recipe-driven
  flows complete. The latest reports are
  [branch tests](../../outputs/branch_tests/report_20260710T212500Z.csv) and
  [sculpt flows](../../outputs/sculpt_flows/report_20260710T213819Z.csv).
- The recipe sampler nearly reaches its fitted-recipe footprint ceiling: sampled IoU 0.620 versus
  fitted/ceiling 0.631 in
  [recipe metrics](../../outputs/recipe_param_diffusion_b6/metrics.json). This means a large gain in
  footprint fit probably requires a richer recipe family or a downstream projection, not merely
  longer training of the same parameterization.
- The fixed scale experiment honestly reports only partial semantic/scale coincidence (6 of 11
  categories), rather than moving the preregistered boundary. The distribution is visible in
  [scale_spectrum.png](../../outputs/scale_spectrum/scale_spectrum.png). This strengthens the
  scientific process, but the paper should avoid saying that every semantic detail is necessarily
  below the 1 m boundary.

### 2.2 Quality gaps exposed by the same artifacts

- **Massing topology and surface quality:** blockout SDEdit often preserves the footprint better
  while producing holes, hanging sheets, detached components, or rounded anonymous blocks in
  [transform_vs_noise/montage.png](../../outputs/transform_vs_noise/montage.png). Current connected-
  component cleanup removes small debris, but it does not repair a dominant component whose surface
  is architecturally invalid.
- **Diversity collapse in the recipe head:** raw-parameter standard deviation is only 0.008 for
  `public_civic`, 0.012 for `modern`, and 0.021 for `victorian`, compared with 0.324 for `colonial`.
  The seed/guidance gallery in
  [mesh_gallery_diversity.png](../../outputs/recipe_param_diffusion_b6/mesh_gallery_diversity.png)
  is nearly unchanged for several styles. Increasing classifier-free guidance raises the reported
  diversity but slightly reduces footprint IoU, so the correct target is conditional coverage,
  not diversity at any cost.
- **Part-count and relationship errors:** planner outputs are too dense, and category means do not
  match the training distribution. Independent boxes do not encode “same floor,” “same bay,”
  “supported by wall,” “roof attached to body,” “mutually exclusive,” or repeated-element relations.
- **Regression metrics are not perceptual metrics:** the live tests correctly catch contract leaks,
  but vertex count, occupancy, or endpoint success cannot tell whether a building looks real.
- **Appearance is costly and loosely tied to semantics:** SDXL gives attractive terminal renders,
  but per-view generation and back-projection are slow, and the facade layout is not a hard control
  signal for every generated view. This invites inconsistent windows, seams, and materials whose
  physical scale is not explicit.

These gaps motivate the literature review below.

## 3. Closest related work

### 3.1 SDF and implicit shape priors/editing

[SDFusion](https://arxiv.org/abs/2212.04493) compresses SDFs with an encoder-decoder and learns
diffusion in the latent space, with separate encoders and cross-attention for text, image, partial
shape, and combined conditions. The
[official implementation](https://github.com/yccyenchicheng/SDFusion) is the direct foundation of
this repository. GenerativeTowns' use of a metric footprint and localized sculpt state is a
domain-specific extension rather than a new latent-SDF primitive.

[SDEdit](https://sde-image-editing.github.io/) adds an adjustable amount of noise to a user guide
and denoises it through a pretrained diffusion prior, explicitly trading input faithfulness against
realism without per-task optimization. That tradeoff is the correct conceptual basis for the
project's footprint-blockout and sculpt-strength sweep. The missing piece is a geometry-specific
quality guardrail: SDEdit itself does not guarantee connected, supported, or watertight 3D solids.

[Diffusion-SDF](https://arxiv.org/abs/2212.03293) uses a voxelized SDF autoencoder and a U-in-U-Net
whose inner network focuses on local SDF patches; it also demonstrates forward-noise/reverse-denoise
shape manipulation. [Make-A-Shape](https://proceedings.mlr.press/v235/hui24a.html) instead uses a
wavelet-tree representation and subband-aware diffusion to retain high-resolution SDF structure.
Both suggest a practical massing improvement: keep the global 64-cubed transform for editability,
but add a local or multiscale surface refiner that is forbidden from changing the footprint and
low-frequency mass.

[Neural Dual Contouring](https://arxiv.org/abs/2202.01999) provides an
[official implementation](https://github.com/czq142857/NDC) for extracting feature-preserving
quad meshes from implicit grids. [FlexiCubes](https://research.nvidia.com/labs/toronto-ai/flexicubes/)
adds locally adjustable geometry and connectivity in a differentiable dual-marching-cubes
representation. NDC is the low-risk fixed-SDF A/B test; FlexiCubes is useful only if the extracted
surface will be optimized against render, footprint, or physical constraints.

### 3.2 Structured and part-based building generation

[StructureNet](https://cs.stanford.edu/~kaichun/structurenet/) represents a shape as a hierarchy of
n-ary part graphs and jointly models part geometry and relations. [SPAGHETTI](https://igl.ethz.ch/projects/SPAGHETTI/)
disentangles each implicit part's extrinsic and intrinsic information and mixes parts through a
global neural field. [SALAD](https://salad3d.github.io/) makes that extrinsic/intrinsic split
generative with a cascaded part-level diffusion process and supports completion, mixing, and
refinement. These are the closest conceptual precedents for replacing independent boxes with a
set that knows its relations and can be globally re-cohered.

[BuildingBlock](https://arxiv.org/abs/2505.04051) is the closest recent competitor to the repository's
design doctrine. It uses Transformer diffusion to generate box/component layouts, expands those
layouts into rule-based hierarchical designs with an LLM, and uses PCG for final construction. Its
[official repository](https://github.com/Tencent/BuildingBlock) releases code and layout data.
GenerativeTowns should compare component-layout validity and editing, while distinguishing its
metric footprint/SDEdit operator, real-element retrieval, and recipe closure.

[Roof-GAN](https://openaccess.thecvf.com/content/CVPR2021/html/Qian_Roof-GAN_Learning_To_Generate_Roof_Geometry_and_Relations_for_Residential_CVPR_2021_paper.html)
generates graph-structured roof primitives and explicitly models collinear and coplanar relations;
the authors release [code and data](https://github.com/yi-ming-qian/roofgan). This is directly useful
because current roof selection is mostly categorical/procedural. A roof graph with ridge, plane,
adjacency, and symmetry constraints would improve form without asking the 64-cubed SDF prior to
resolve roof construction detail.

[Building-GAN](https://openaccess.thecvf.com/content/ICCV2021/html/Chang_Building-GAN_Graph-Conditioned_Architectural_Volumetric_Design_Generation_ICCV_2021_paper.html)
conditions volumetric architectural massing on compact program graphs; its
[official code](https://github.com/AutodeskAILab/Building-GAN) provides another structured baseline.
Its main relevance here is not facade fidelity, but graph-conditioned connectivity and program
validity as measurable constraints.

[ShellMaker](https://arxiv.org/abs/2606.31680) is a current scaffold-conditioned building system
that combines parametric roofs, material retrieval, generated parts, PBR assets, and geometry-aware
assembly while preserving a fixed footprint, walls, and openings; its
[author project page](https://ruiqixu37.github.io/ShellMaker_web/) reports footprint and opening-
alignment metrics. It is a close comparator for constraint-preserving facade assembly even though
its fixed scaffold and prompt differ from GenerativeTowns' footprint-to-massing problem. Its
opening-center, opening-size, part-intersection, and footprint-violation measurements are worth
adopting directly.

[BuildingNet](https://buildingnet.org/) contains about 2,000 models, 513,000 annotated mesh
primitives, and 292,000 semantic components. It remains a good source for part supervision and
retrieved real geometry, but building-ID leakage exclusion is essential when crops are evaluated
against held-out buildings.

### 3.3 Facade grammar, repeated detail, and mesh enrichment

[FaçAID](https://arxiv.org/abs/2406.01829) converts segmented facade layouts into editable split-
grammar programs with a neuro-symbolic Transformer. [Pro-DG](https://arxiv.org/abs/2504.01571)
uses such hierarchical facade structure as control for diffusion-based appearance editing, including
large edits such as floor duplication or window rearrangement. These papers address the exact
failure mode visible in the part-layout montage: facade elements are not an unordered cloud; they
are nested floors, bays, repetitions, and exceptions.

[StructuredMesh](https://arxiv.org/abs/2306.04184) detects windows, doors, and balconies in virtual
color/depth views, projects them back to 3D, and uses binary integer programming to regularize their
positions, orientations, and sizes before instance replacement. A lighter version of this idea can
be used as a deterministic projection layer after `PartLayoutPlannerV2`: snap elements into inferred
floor/bay groups while minimizing displacement and rejecting collisions.

[Proc-GS](https://city-super.github.io/procgs/) learns shared Gaussian assets for repeated building
elements and assembles them under procedural code, with instance-specific variation and real/synthetic
building editing. This strongly supports GenerativeTowns' choice to compose repeated detail, but it
also makes composition alone non-novel. Its useful implementation idea is to split a retrieved detail
element into a canonical shared asset plus a small per-placement residual.

[Texture2LoD3](https://wenzhaotang.github.io/Texture2LoD3/) rectifies imagery onto LoD2 facades,
segments openings, and reconstructs facade-rich LoD3 geometry; the authors release the
[ReLoD3 dataset](https://zenodo.org/records/15178144). [TUM2TWIN](https://github.com/tum-gis/tum2twin)
also releases CityGML LoD2/LoD3 buildings, textures, facade-labeled point clouds, and street imagery.
These are stronger sources than generic object meshes for testing real opening placement, facade
regularity, and regional LoD3 element enrichment.

### 3.4 OSM-conditioned town generation and scene context

The official [OpenStreetMap Simple 3D Buildings schema](https://wiki.openstreetmap.org/wiki/Simple_3D_buildings)
supports more than a footprint and approximate height: `building:part`, `height`, `min_height`,
`building:levels`, `roof:height`, `roof:levels`, `roof:shape`, roof orientation/direction, and facade/
roof material and color can all be represented. [OSM2World](https://github.com/tordanik/OSM2World)
is an implemented deterministic OSM-to-3D baseline and should be included as a zero-learning oracle
for footprints that carry rich tags.

[UrbanWorld](https://arxiv.org/abs/2407.11965) accepts OSM or semantic/height maps, designs an urban
scene, renders controllable assets with progressive 3D diffusion, and refines the result. Its
[official repository](https://github.com/Urban-World/UrbanWorld) includes OSM download/division and
a runnable OSM pipeline. It is the most direct end-to-end OSM-conditioned baseline.

[GeoTexBuild](https://arxiv.org/abs/2504.08419) is the nearest direct footprint-to-detailed-building
pipeline found in this review: it starts from map footprints, predicts a height representation,
reconstructs geometry, and stylizes appearance. It should be compared at the task/contract level,
while noting that GenerativeTowns additionally requires reversible recipe state and interactive
sculpt repair.

[CityCraft](https://arxiv.org/abs/2406.04983) combines diffusion-based layout generation, LLM land-use
planning, asset retrieval, and Blender assembly; the
[official repository](https://github.com/djFatNerd/CityCraft) releases the implementation. The
important transferable idea is context-aware retrieval: function, footprint scale, road distance,
district style, and neighbors should influence a building choice instead of querying each footprint
independently.

[CityDreamer](https://openaccess.thecvf.com/content/CVPR2024/html/Xie_CityDreamer_Compositional_Generative_Model_of_Unbounded_3D_Cities_CVPR_2024_paper.html)
separates background “stuff” such as roads/greenery from instance-specific building neural fields,
supports localized building editing, and releases
[code, models, and OSM/Google Earth data](https://github.com/hzxie/CityDreamer).
[CityDreamer4D](https://arxiv.org/abs/2501.08983) extends that composition to traffic and static city
objects with compact BEV layouts, and its [official code](https://github.com/hzxie/CityDreamer4D)
is public. The relevance is the factorization: a town needs a block/district latent and explicit
road/parcel context in addition to independent building recipes.

[MajutsuCity](https://openaccess.thecvf.com/content/CVPR2026/html/Huang_MajutsuCity_Language-driven_Aesthetic-adaptive_City_Generation_with_Controllable_3D_Assets_and_CVPR_2026_paper.html)
is a current system for language-controlled layout, asset, material, and skybox composition, with
a declared [project repository](https://github.com/LongHZ140516/MajutsuCity). [Yo'City](https://openaccess.thecvf.com/content/CVPR2026/html/Lu_YoCity_Personalized_and_Boundless_3D_Realistic_City_Scene_Generation_via_CVPR_2026_paper.html)
uses a City–District–Grid hierarchy and a generate/refine/evaluate loop. These are useful town-level
comparators even though they do not expose GenerativeTowns' same reversible SDF sculpt contract.

For data, [3D BAG's official API](https://api.3dbag.nl/api.html) serves LoD 1.2, 1.3, and 2.2
buildings generated from Dutch building and elevation sources. [Building3D](https://szusic.github.io/Building3D/reconstruction.html)
provides point clouds, meshes, wireframes, and more than 60 roof types. [MatrixCity](https://city-super.github.io/matrixcity/)
provides aerial/street imagery with ground-truth cameras and extra rendered modalities for city-scale
neural rendering. Each covers a different gap: watertight massing, roof structure, and scene-scale
appearance/evaluation respectively.

### 3.5 Gaussian splatting and appearance composition

The original [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
represents a radiance field with optimized anisotropic Gaussians and a visibility-aware rasterizer;
the authors maintain the [reference implementation](https://github.com/graphdeco-inria/gaussian-splatting).
Its output is an appearance representation, not automatically a semantic, collision-ready, or
editable building model.

[GaussianCity](https://arxiv.org/abs/2406.06526) generates unbounded cities with a compact BEV-Point
representation and a spatial-aware Gaussian decoder. Its
[official code, training scripts, pretrained models, and OSM/Google Earth data paths](https://github.com/hzxie/GaussianCity)
make it the best architectural reference for the repository's SDF-to-Gaussian lifter. The most
transferable details are point serialization, separate background/building generators, and compact
intermediate state rather than a dense fixed slot grid everywhere.

[Proc-GS](https://city-super.github.io/procgs/) is more aligned with recipe closure: its Gaussians are
grouped into reusable procedural building elements. A GenerativeTowns Gaussian should likewise carry
`building_id`, `recipe_op_id`, and semantic-part identity, so a window re-roll replaces only that
window's Gaussians.

[CityGaussian](https://arxiv.org/abs/2404.01133) uses divide-and-conquer training and level-of-detail
selection for large reconstructed Gaussian scenes; the
[official repository](https://github.com/Linketic/CityGaussian) releases the series. LOD is relevant
after individual building appearance is acceptable, not before.

[DiffSplat](https://github.com/chenguolin/DiffSplat) repurposes image diffusion for text/single-image
Gaussian generation and reports feed-forward output in seconds. [LGM](https://github.com/3DTopia/LGM)
uses multi-view Gaussian features and an asymmetric U-Net for high-resolution object generation.
Both are useful appearance baselines or teachers, but neither natively consumes a metric footprint
plus reversible building recipe.

[SuGaR](https://imagine.enpc.fr/~guedona/sugar/) aligns Gaussians to surfaces, extracts a Poisson
mesh, and can bind refined Gaussians to that mesh. It provides a route for renderable splats and
surface export to share a representation, though the recipe/SDF should remain the source of truth.

For mesh-first PBR appearance, [Hunyuan3D 2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)
releases shape and PBR texture training/inference code and explicitly predicts physically based
materials. It is not a footprint-conditioned baseline, but its paint stage is a relevant terminal
texture comparator. [MatSynth](https://www.gvecchio.com/matsynth) provides more than 4,000 CC0 PBR
materials with physical scale and relit renders, making retrieval of scale-consistent materials a
strong, cheap baseline against SDXL-derived heuristic normal/roughness maps.

## 4. Compact relevance and gap table

| Work / resource | Public implementation or data | Most relevant capability | Gap relative to GenerativeTowns | Concrete use here |
|---|---|---|---|---|
| [SDFusion](https://github.com/yccyenchicheng/SDFusion) + [SDEdit](https://sde-image-editing.github.io/) | Official code | Latent SDF prior; realism/faithfulness editing | No building relations or topology guarantee | Keep as C1 foundation; add geometry validity guardrails and strength calibration |
| [BuildingBlock](https://github.com/Tencent/BuildingBlock) | Official code/data | Learned component layouts + rules + PCG | Not one metric SDF transform for both generation and sculpting | Layout/coherence baseline; adapt a set representation with hierarchical rules |
| [SALAD](https://salad3d.github.io/) / [StructureNet](https://cs.stanford.edu/~kaichun/structurenet/) | Code links on author pages | Part extrinsics, intrinsics, and relations | Generic categories rather than architecture | Separate layout diffusion from element appearance; add relation edges and global mixing |
| [Roof-GAN](https://github.com/yi-ming-qian/roofgan) | Official code/data | Relation-aware roof primitives | Residential-roof scope | Add ridge/plane/adjacency graph instead of one roof enum |
| [ShellMaker](https://ruiqixu37.github.io/ShellMaker_web/) | Project page; no public code found in this review | Constraint-preserving roof/opening/material assembly | Begins from a fixed scaffold | Adopt footprint, opening-center/size, and part-intersection metrics |
| [FaçAID](https://arxiv.org/abs/2406.01829) / [StructuredMesh](https://arxiv.org/abs/2306.04184) | Paper/method details | Procedural facade programs; constraint optimization | Starts from observed facade layouts | Project sampled part boxes into floors, bays, repeats, and non-collision constraints |
| [Proc-GS](https://city-super.github.io/procgs/) | Author project/code link | Shared repeated Gaussian assets under procedural code | Appearance-focused; no SDF sculpt operator | Recipe-linked canonical element appearance plus small instance residuals |
| [UrbanWorld](https://github.com/Urban-World/UrbanWorld) | Official code | OSM-conditioned end-to-end urban generation | Less explicit reversible per-building state | OSM baseline and town-level visual comparator |
| [GeoTexBuild](https://arxiv.org/abs/2504.08419) | Paper; no official code found in this review | Footprint-to-geometry-to-texture pipeline | No demonstrated recipe-closure/sculpt contract | Direct task-level comparator for final building fidelity |
| [CityCraft](https://github.com/djFatNerd/CityCraft) | Official code | Context-aware layout and asset retrieval | Asset assembly rather than SDF editing | Add district/road/neighborhood features to style and retrieval decisions |
| [CityDreamer](https://github.com/hzxie/CityDreamer) / [CityDreamer4D](https://github.com/hzxie/CityDreamer4D) | Official code/models/data | Instance-vs-background composition; unbounded BEV layouts | Neural fields are not exportable recipes | Add block/district latent and explicit stuff/instance factorization |
| [OSM Simple 3D](https://wiki.openstreetmap.org/wiki/Simple_3D_buildings) / [OSM2World](https://github.com/tordanik/OSM2World) | Official spec + code | Heights, parts, roofs, materials; deterministic realization | Limited generative diversity | Consume known tags as hard constraints and use OSM2World as a zero-learning oracle |
| [Texture2LoD3](https://wenzhaotang.github.io/Texture2LoD3/) / [TUM2TWIN](https://github.com/tum-gis/tum2twin) | Data and code links | Real openings, textures, LoD3 geometry | Mostly reconstruction | Regional facade supervision and held-out detail evaluation |
| [GaussianCity](https://github.com/hzxie/GaussianCity) | Official code/models | Compact feed-forward city Gaussians | Appearance representation lacks recipe semantics | Reference for Stage 3b; carry building/part/op identity through every Gaussian |
| [CityGaussian](https://github.com/Linketic/CityGaussian) | Official code | Large-scene partitioning and LOD | Reconstruction, not generation | Scale town rendering after per-building quality is solved |
| [NDC](https://github.com/czq142857/NDC) / [FlexiCubes](https://research.nvidia.com/labs/toronto-ai/flexicubes/) | Official code | Sharp feature-preserving SDF meshing | Cannot invent missing architectural structure | Low-risk replacement/optimization A/B against marching cubes |
| [Hunyuan3D 2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) / [MatSynth](https://www.gvecchio.com/matsynth) | Training code/weights; CC0 data | PBR texture synthesis and physically scaled materials | Not symbolic-footprint conditioned | Terminal paint baseline and material retrieval source |

## 5. Concrete improvement program

### Priority 0 — make quality measurable before changing models

Build one canonical evaluation harness that renders the same held-out IDs, cameras, lighting,
resolution, and neutral material for every path. Keep the existing endpoint/regression suite, but
add the following distinct panels.

1. **Constraint and massing panel:** footprint IoU, 3D IoU, symmetric surface Chamfer, target-height
   error, silhouette IoU over fixed views, occupied connected-component count, largest-component
   fraction, surface genus or cavity count, and self-intersection/non-manifold failures.
2. **Part-structure panel:** type-count error, box precision/recall/F1, opening-center and opening-size
   error, support/collision violations, floor/bay regularity residual, facade coverage, symmetry/
   repetition consistency, and semantic part-aware Chamfer.
3. **Distribution panel:** multi-view facade KID/CMMD, precision, recall, 3D COV/MMD/1-NNA, and
   per-class confidence intervals. Do not collapse quality and coverage into one number.
4. **Edit panel:** mask IoU, untouched-region delta, structural validity after edit, recipe closure,
   and the realism-versus-faithfulness curve across SDEdit strengths.
5. **Production panel:** connected/manifold/watertight status, degenerates, triangle aspect ratio,
   UV coverage, glTF validation, runtime, peak VRAM, and artifact size.

FID was introduced as a generated-versus-real image distribution distance in the
[original paper](https://proceedings.neurips.cc/paper/2017/hash/8a1d694707eb0fefe65871369074926d-Abstract.html),
but finite-sample FID has model-dependent bias, as shown by
[Chong and Forsyth](https://openaccess.thecvf.com/content_CVPR_2020/html/Chong_Effectively_Unbiased_FID_and_Inception_Score_and_Where_to_Find_CVPR_2020_paper.html).
With only hundreds of held-out buildings, add KID, whose estimator is unbiased in the
[original MMD-GAN work](https://openreview.net/pdf?id=r1lUOzWCW), or
[CMMD](https://openaccess.thecvf.com/content/CVPR2024/papers/Jayasumana_Rethinking_FID_Towards_a_Better_Evaluation_Metric_for_Image_Generation_CVPR_2024_paper.pdf),
which uses CLIP embeddings and unbiased MMD with an
[official implementation](https://github.com/google-research/google-research/tree/master/cmmd).
Use [Clean-FID](https://github.com/GaParmar/clean-fid) if retaining FID so resizing and quantization
are fixed.

For 3D distributions, [Achlioptas et al.](https://proceedings.mlr.press/v80/achlioptas18a.html)
introduced matching-based fidelity and coverage measures for point-cloud generators. The standard
bundle of COV, MMD, and 1-NNA is used in later diffusion work; part-aware Chamfer, as proposed by
[SeaLion](https://openaccess.thecvf.com/content/CVPR2025/html/Zhu_SeaLion_Semantic_Part-Aware_Latent_Point_Diffusion_Models_for_3D_Generation_CVPR_2025_paper.html),
is particularly relevant because an ordinary Chamfer distance can ignore whether the correct
architectural parts cohere.

### Priority 1 — repair structured detail before expanding the element library

Replace “sample many independent typed boxes, then heuristically regularize” with a two-level model:

1. Sample a bounded set of **extrinsics**: part type, presence, center, extent, wall/roof anchor,
   floor index, bay index, and parent mass.
2. Predict explicit **relations**: supported-by, attached-to, same-row, same-column, repeated-with,
   symmetric-with, excludes, and roof-plane adjacency.
3. Project the noisy set through a deterministic facade/roof constraint solver.
4. Only then retrieve or procedurally instantiate intrinsic geometry.

Train presence/count explicitly rather than deriving it from a dense field. Use Hungarian matching
for unordered parts, focal or calibrated BCE for presence, box losses only for matched parts, and
relation losses on matched pairs. Report per-class calibration and count histograms, not only a
single box loss. SALAD's extrinsic-before-intrinsic cascade and StructureNet's relation graph supply
the architectural pattern; FaçAID and StructuredMesh supply the grammar/constraint projection.

The first minimal experiment does not need a new large generator: run the current planner, infer
facade planes and floor/bay spacing, solve for the nearest valid repeated layout, and compare the
existing montage, counts, collision rate, and KID. This directly tests whether structure, rather
than element appearance, is the bottleneck.

### Priority 1 — make the massing transform topology-aware

Keep SDEdit as the core transform, but separate candidate generation from candidate acceptance.
For every footprint/edit, generate a small fixed set of candidates and score them with hard or
learned guardrails:

- exact footprint and height retention;
- largest-component fraction and maximum component count;
- no floating mass below a support threshold;
- bounded cavity/perforation count at the massing scale;
- neutral-view discriminator or precision score;
- edit-mask fidelity and untouched-region preservation.

Reranking is preferable to silently repairing every failure because it leaves the model claim
honest and produces a measurable rejection/failure rate. Then test one local multiscale refiner
(Diffusion-SDF-style local U-Net or wavelet residual) whose loss is masked to a surface band and
whose low-frequency/footprint projections are hard constraints. Finally A/B marching cubes against
NDC on the same SDFs; meshing can sharpen corners but cannot fix missing mass, so keep those effects
separate.

### Priority 1 — use known OSM facts as hard recipe constraints

Expand `scene/extract_osm.py` to preserve, parse, and attach confidence/provenance for:

- `height`, `min_height`, `building:levels`, `building:min_level`;
- `building:part` hierarchy;
- `roof:shape`, `roof:height`, `roof:levels`, `roof:orientation`, `roof:direction`;
- facade and roof material/color;
- use/amenity/land-use context and adjacent road class.

Known tags should override sampling; missing tags become learned decisions. Compare three arms on
richly tagged areas: OSM2World, current minimal conditioning, and full-tag GenerativeTowns. This is
a clean way to improve real-place fidelity without violating the doctrine that learned models make
unknown decisions and procedures realize them.

Add a shared block/district latent inferred from road orientation, density, land use, region, and
neighboring height/style. Condition each building on the shared latent plus its individual recipe.
Measure within-block roof/material coherence and between-sample diversity. CityCraft and the
CityDreamer family show why independent per-footprint decisions are insufficient for a convincing
town.

### Priority 2 — make appearance semantic, reusable, and cheaper

Before generating RGB views, render facade semantic controls from the recipe: wall, window, door,
roof, ornament, material region, depth, normals, and stable instance IDs. Condition every view on
the same maps and down-weight or reject cross-view inconsistent images. Bake albedo/normal/roughness
from semantically aligned views, not a free image whose windows can move.

Add two cheap baselines:

- retrieve physically scaled PBR materials from MatSynth by style, region, era, and semantic part;
- run an external mesh-paint method such as Hunyuan3D-Paint only as a terminal comparator, never as
  the editable source of truth.

For repeated windows/doors, cache canonical appearance assets and generate only small per-instance
variation, following Proc-GS. This should reduce inference time and improve within-building
consistency.

### Priority 2 — constrain the Gaussian branch to preserve recipe closure

Do not emit one anonymous `GaussianSet` for a whole building. Every Gaussian should retain:

```text
building_id, recipe_op_id, semantic_part, canonical_asset_id, local_transform, lod_level
```

Generate or retrieve splats per mass/element, compose them, cull against the footprint/SDF, and
train with RGB plus depth, normal, silhouette, footprint, and semantic-mask render losses. A semantic
edit should replace only the splats owned by that recipe operation. Use GaussianCity for compact
decoding, Proc-GS for reusable procedural assets, and CityGaussian only when a town requires LOD.
If collision or export is needed, derive it from the recipe/SDF or a SuGaR-style bound surface,
not opacity blobs.

## 6. Suggested experiment order

1. Freeze 24–48 representative held-out buildings and 4–6 OSM blocks, stratified by class,
   footprint complexity, region, and current failure type.
2. Run the new measurement panel on existing artifacts without retraining. Publish failure rates,
   not only averages.
3. Add candidate generation plus topology/constraint reranking to SDEdit. This is the quickest
   likely massing gain and produces the missing sculpt strength curve.
4. Add facade/roof constraint projection to current planner outputs. If count, collision, and KID
   improve materially, then replace the planner with a relation-aware set model.
5. Parse full OSM Simple 3D tags and add OSM2World as a baseline before training a town-context model.
6. Evaluate NDC versus marching cubes on identical grids; adopt only if sharpness/manifold metrics
   improve without changing the underlying SDF comparison.
7. Add semantic PBR retrieval and consistent multi-view controls; compare time, seams, and CMMD
   against the current SDXL bake.
8. Resume the Gaussian branch only after semantic ownership and evaluation are part of its data
   contract.
9. Complete the preregistered equal-data C2 curve and blinded two-AFC. The literature now makes
   this evidence more important, not less: it is the cleanest way to distinguish the project's
   scientific contribution from other hybrid procedural/generative systems.

## 7. Novelty positioning and limitations

The paper should not claim that hybrid procedural/neural building generation, OSM-conditioned city
generation, part composition, or Gaussian asset reuse is new in isolation. BuildingBlock, CityCraft,
UrbanWorld, Proc-GS, and current city systems cover those ideas directly.

A narrower and stronger claim is supportable if the experiments hold:

> A single footprint/edit transform recovers massing under small real-data budgets, while a
> leakage-safe library plus learned relational placement recovers facade detail more faithfully
> than a matched monolithic generator, and every decision remains reversible in a symbolic recipe.

Important limitations should remain explicit:

- current 64-cubed SDF outputs can be topologically and visually poor even when footprint metrics
  improve;
- the fixed massing/detail scale has only partial, not universal, semantic coincidence;
- BuildingNet's small and uneven categories limit learned part relationships and regional style;
- retrieved elements can improve fidelity while reducing coverage, so precision and recall must be
  reported separately;
- terminal neural appearance can hide geometry failures and should not be used to score geometry;
- Gaussian splats improve rendering but do not automatically provide editable surfaces, collision,
  or production meshes;
- OSM tag completeness varies geographically, so every hard condition needs provenance and a
  missing-data policy.

That positioning makes the local negative results useful evidence and gives the next implementation
work clear, falsifiable targets.
