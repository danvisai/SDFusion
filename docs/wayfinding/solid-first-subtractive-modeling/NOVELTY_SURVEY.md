# Novelty survey: solid-first semantic architectural carving

**Research cutoff:** 2026-07-15  
**Question:** Does a system already exist that starts from a site-contextualized set of metric building footprints, generates and edits buildings through architecturally typed additive/subtractive programs, and deterministically realizes those programs as valid solids?

## Verdict

**No exact published match was found in the primary sources surveyed.** The proposed system appears novel as an exact end-to-end contract, but not because any one of its ingredients is new. Footprint-driven procedural buildings, learned architectural programs, neural CSG/CAD program induction, editable program representations, block-level programs, and deterministic procedural realization all have strong precedents.

The closest whole-system prior is [CityGenAgent](https://arxiv.org/abs/2602.05362): it learns editable, executable Block and Building Programs, represents footprints as metric polygons, generates coordinated blocks, and updates programs through natural-language edits. The closest footprint-and-massing prior is [CoMa](https://arxiv.org/abs/2601.08464): it conditions on site contours and context and emits separate polygonal extrusion sequences for multiple identified buildings. The closest architecture-specific program/compiler prior is [ArcPro](https://openaccess.thecvf.com/content/CVPR2025/html/Huang_ArcPro_Architectural_Programs_for_Structured_3D_Abstraction_of_Sparse_Points_CVPR_2025_paper.html): it predicts a hierarchical architectural DSL that a learning-free interpreter converts to a mesh. The closest learned sequential interaction prior is [Representation Learning for Sequential Volumetric Design Tasks](https://arxiv.org/abs/2309.02583): its Building-Gym representation supports typed volumetric action sequences and partial-sequence autocompletion.

None of those systems demonstrates all of the following together:

1. user-supplied, fixed metric footprints as hard per-building envelopes;
2. first-class architectural **void** operations such as courtyard, passage, arcade, light well, setback, and roof cut;
3. a learned ordered program mixing constrained addition and subtraction;
4. deterministic SDF/CSG realization with geometry-validity gates;
5. the same representation for autonomous generation and interpretation/completion of a rough user carve; and
6. coordinated programs for a selected block of footprints while preserving separate, non-fused building solids.

This is a negative search result, not proof of nonexistence. The defensible claim is therefore **“no exact match found through the stated cutoff,”** not “the first ever,” until a formal systematic review and patent/product search are also completed.

## Precise proposed contract

The primary input is not text. It is a georeferenced or locally metric **footprint set** with optional hard and soft context:

```text
footprint polygons + ground/eaves heights + building/site attributes
                         |
                         v
        deterministic watertight footprint envelopes
                         |
                         v
      learned per-footprint architectural edit programs
                         |
                         v
        deterministic SDF/CSG evaluation and validation
                         |
                         v
              derived meshes and visual traces
```

Text may optionally condition intent or style, but it does not replace the footprint set or weaken metric constraints.

Each footprint receives its own filled, watertight base envelope. A program is an ordered list whose operations carry at least:

```text
architectural type       courtyard | passage | arcade | setback | roof_cut | wing | roof_volume | ...
Boolean mode             subtract | add
constrained geometry     polygon + height range | sweep/profile | repeated primitive set
relations                parent, inside, attached_to, open_to, repeated_with, symmetric_with
constraints              minimum wall/support thickness, clearance, connectivity, footprint containment
identity                 building_id, operation_id, seed, provenance
```

The same representation supports two learned modes:

- **Autonomous generation:** footprint set and context to complete per-building programs.
- **Guided interpretation:** a user supplies a rough local add/subtract region; the model assigns an architectural type, completes parameters/relations, and changes only the selected building or footprint block.

Selecting several footprints coordinates height rhythm, roof family, setbacks, courtyard patterns, or other shared decisions, but each footprint keeps its own envelope, program, validity state, and mesh. Cross-parcel Boolean fusion is out of scope. Fine windows, doors, cornices, ornament, and materials remain procedural or retrieved downstream. Full interiors, circulation, structural engineering, and code compliance are out of scope.

Validation and final evaluation must emit a **visual carving trace** with fixed isometric, plan, elevation, and section/cutaway views; operation coloring; constraint overlays; and representative/best/worst contact sheets. Training-time screenshot logging is optional, not required.

## Comparison matrix

Legend: **Y** = explicitly demonstrated; **P** = partial/adjacent; **N** = not demonstrated. “Typed +/-” means architecturally semantic additive and subtractive massing operations, not merely generic Boolean primitives.

| Primary-source system | Fixed metric footprint input | Learned executable program | Architectural typed +/- | Deterministic realization | Autonomous generation | Guided edit in same representation | Coordinated multiple buildings | Main gap from the contract |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| [CoMa (2026)](https://arxiv.org/abs/2601.08464) | P | P | N | Y | Y | N | Y | Site contour and context produce per-building polygon extrusion lists, but not fixed input footprints, semantic carve operations, or guided editing. |
| [CityGenAgent (2026)](https://arxiv.org/abs/2602.05362) | P | Y | P | Y | Y | Y | Y | Metric footprint polygons are generated inside a Block Program; Building Programs assemble components/assets rather than execute an SDF/CSG carve sequence over supplied envelopes. |
| [ArcPro (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Huang_ArcPro_Architectural_Programs_for_Structured_3D_Abstraction_of_Sparse_Points_CVPR_2025_paper.html) | P | Y | P | Y | N | P | N | Predicts `CreateLayer` trees from sparse point clouds; real footprints seed synthetic training, but void semantics, autonomous design, and block coordination are absent. |
| [Building-Gym sequential volumetric design (2023)](https://arxiv.org/abs/2309.02583) | P | P | N | Y | P | Y | N | Learns ordered typed room-volume actions and partial-sequence completion, but its mapping prevents deletion and it does not expose a clean architectural CSG DSL. |
| [3D Synthesis for Architectural Design (WACV 2025)](https://openaccess.thecvf.com/content/WACV2025/papers/Tsai_3D_Synthesis_for_Architectural_Design_WACV_2025_paper.pdf) | N | P | P | P | Y | P | N | Autonomous cuboid unions are random, while user intrusion/extrusion edits deform detected mesh regions rather than append learned executable operations. |
| [BuildingBlock (SIGGRAPH 2025)](https://arxiv.org/abs/2505.04051) | N | Y | P | Y | Y | P | N | Learned box layouts, LLM-expanded rules, and PCG are highly relevant, but the representation is component assembly rather than fixed-footprint semantic carving. |
| [Building-GAN (ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/html/Chang_Building-GAN_Graph-Conditioned_Architectural_Volumetric_Design_Generation_ICCV_2021_paper.html) | P | P | N | P | Y | N | N | Generates a typed voxel graph inside a design-space envelope from a room/program graph; it does not generate an ordered Boolean construction history. |
| [Neural Procedural Reconstruction for Residential Buildings (ECCV 2018)](https://openaccess.thecvf.com/content_ECCV_2018/html/Huayi_Zeng_Neural_Procedural_Reconstruction_ECCV_2018_paper.html) | P | Y | P | Y | N | N | N | Learns rule branches and parameters for a fixed residential grammar from aerial LiDAR, but is reconstruction-oriented and mostly additive. |
| [GeoTexBuild (2025)](https://arxiv.org/abs/2504.08419) | Y | N | N | P | Y | N | P | Directly addresses map-footprint-to-building generation, but predicts height/geometry/appearance rather than an editable semantic operation program. |
| [ShapeAssembly (SIGGRAPH Asia 2020)](https://rkjones4.github.io/shapeAssembly.html) | N | Y | N | Y | Y | P | N | Establishes learned, executable, editable hierarchical programs, but uses generic cuboid attachment without architecture, footprints, or subtraction. |
| [CSGNet (CVPR 2018)](https://hippogriff.github.io/CSGNet/) and [PLAD (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Jones_PLAD_Learning_To_Infer_Shape_Programs_With_Pseudo-Labels_and_Approximate_CVPR_2022_paper.html) | N | Y | N | Y | N | P | N | Learn compact union/intersection/subtraction programs from target shapes, but primitives are not architectural and the task is inverse parsing rather than footprint-conditioned design. |
| [CGA Shape (SIGGRAPH 2006)](https://doi.org/10.1145/1179352.1141931) and [Interactive Architectural Modelling with Procedural Extrusions (2011)](https://eprints.gla.ac.uk/48707/) | Y | N | P | Y | P | Y | P | Establish footprint-driven, scalable, editable procedural architecture and two-manifold extrusion, but rules are authored rather than learned and no shared rough-carve predictor exists. |

Two additional 2026 systems narrow adjacent claims. [ShellMaker](https://arxiv.org/abs/2606.31680) preserves a supplied scaffold's footprint, walls, and openings while completing roofs, materials, and exterior parts; this makes fixed-constraint exterior completion and downstream procedural/retrieval detail non-novel. [MajutsuCity](https://openaccess.thecvf.com/content/CVPR2026/html/Huang_MajutsuCity_Language-driven_Aesthetic-adaptive_City_Generation_with_Controllable_3D_Assets_and_CVPR_2026_paper.html) represents cities as controllable layouts, assets, and materials and adds object-level interactive editing; city-scale editability is therefore also not new by itself.

## Closest precedents in more detail

### 1. CityGenAgent is the strongest whole-system novelty risk

CityGenAgent decomposes a city into a **Block Program** and **Building Program**, trains agents to generate schema-valid programs, executes them into meshes, and edits blocks or buildings by updating those same programs. Its Block Program stores non-self-intersecting polygons in meters, building type, floor count, and facade descriptions; it explicitly evaluates collision and program validity. These properties overlap the proposed hierarchy, metric state, coordinated generation, and program-preserving interaction very closely. [Primary paper](https://arxiv.org/abs/2602.05362)

The defensible distinction is not “hierarchical editable city programs.” It is that this project begins with an immutable external footprint set and learns **architecture-specific solid edits**, especially named voids, whose exact Boolean realization and locality are validated. CityGenAgent generates its block polygons from language and realizes building descriptions through prepared/retrieved/generated assets rather than a typed CSG/SDF history.

### 2. CoMa is the strongest multi-building massing risk

CoMa accepts a polygonal site contour, urban context, and a separate requirement record per building, then autoregressively emits each building as polygonal horizontal extrusions with bottom/top elevations and stable IDs. That is very close to coordinated separate per-building massing programs. [Primary paper](https://arxiv.org/abs/2601.08464)

Its gap is equally important: final extrusion lists are not semantic edit traces; the input does not hard-fix each building footprint; there is no subtraction, learned rough-carve interpretation, deterministic SDF/CSG evaluator, or edit-locality contract. Its reported geometric failure modes also make hard validation a meaningful research target rather than implementation polish.

### 3. ArcPro is the strongest architecture-specific program/compiler risk

ArcPro defines a DSL of ground-setting and hierarchical polygonal `CreateLayer` statements, predicts tokenized programs from sparse point clouds, uses a finite-state mask for syntactic validity, and compiles them through a learning-free interpreter. It generates synthetic program/point pairs using **872,487 cleaned Bing Maps footprints** as root contours, then constrains child contours by contraction or planar subdivision. [Primary paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Huang_ArcPro_Architectural_Programs_for_Structured_3D_Abstraction_of_Sparse_Points_CVPR_2025_paper.pdf)

Therefore, “synthetic architectural programs built on real footprints, predicted by a Transformer, then deterministically compiled” is already established. The new DSL must do substantially more than relabel ArcPro layers: it must make void type, Boolean mode, parent/support relations, validity, and interactive completion first-class.

[Synthesizing 3D Abstractions by Inverting Procedural Buildings with Transformers](https://research.google/pubs/synthesizing-3d-abstractions-by-inverting-procedural-buildings-with-transformers/) independently reinforces the same direction: synthetic procedural buildings plus simulated point clouds train a Transformer to recover editable programmatic building abstractions and support structurally consistent inpainting.

### 4. Building-Gym and sketch-guided grammar work threaten the interaction claim

The Building-Gym work represents designs as ordered actions with location, size, and one of seven room types, learns sequential volumetric design representations, and autocompletes a partial user sequence under site/design constraints. Its current action mapping is additive, explicitly preventing deletion. [Primary paper](https://arxiv.org/abs/2309.02583)

[Interactive Sketching of Urban Procedural Models](https://cgvlab.github.io/cgvlab/www/publications/nishida2016interactive/) uses CNNs to interpret user strokes by choosing procedural grammar snippets and predicting their parameters; successive sketches become an executable building grammar. This means learned interpretation of rough user input into architectural procedural state is also known. What was not found is a learned interpreter that turns a **rough 3D add/subtract region** into a semantic architectural void/volume operation inside the same distribution used for autonomous footprint-to-program generation.

### 5. CSG/CAD generation makes the Boolean sequence itself non-novel

[CSGNet](https://hippogriff.github.io/CSGNet/) predicts recursive union, intersection, and subtraction programs over 2D/3D primitives. [D2CSG](https://proceedings.neurips.cc/paper_files/paper/2023/file/4732d425125832887f6c5a9675d49ead-Paper-Conference.pdf) learns compact CSG trees without program annotations and exports editable OpenSCAD scripts. [DeepCAD](https://openaccess.thecvf.com/content/ICCV2021/html/Wu_DeepCAD_A_Deep_Generative_Network_for_Computer-Aided_Design_Models_ICCV_2021_paper.html) generates editable CAD operation sequences and released 178,238 construction histories. [DI-PCG](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhao_DI-PCG_Diffusion-based_Efficient_Inverse_Procedural_Content_Generation_for_High-quality_3D_CVPR_2025_paper.pdf) treats procedural-generator parameters as the diffusion target and deterministically runs the generator after sampling. [Text2CSG](https://www.sciencedirect.com/science/article/pii/S0010448526000977) extends executable CSG-tree generation to text conditions.

The contribution cannot be “a network generates editable Boolean/CAD instructions.” It must be an architectural domain model, conditioning/evaluation contract, data method, or interaction result that these generic systems do not supply.

### 6. Traditional procedural architecture already covers much of the geometry

[CGA Shape](https://doi.org/10.1145/1179352.1141931) procedurally creates consistent mass models and detailed shells at urban scale using context-sensitive volumetric rules. [Interactive Architectural Modelling with Procedural Extrusions](https://eprints.gla.ac.uk/48707/) starts from footprints and supports difficult exterior forms such as curved/overhanging roofs, dormers, bay windows, columns, and alcoves while computing a two-manifold surface. [EvoMass](https://journal.hep.com.cn/foar/EN/1159778666085409149) is especially relevant to the “solid-first” idea because it explores architectural massing with parameterized cuboids and explicit Boolean union/difference inside a maximal boundary.

Thus, the deterministic engine should reuse established geometric ideas. Novelty should be sought in learned semantic decisions and edit behavior, not in the act of subtracting primitives from a building envelope.

## Novelty decomposition

### Established ingredients

- Footprint or site polygon to procedural 3D building: CGA, procedural extrusions, GeoTexBuild, UrbanWorld, and commercial CityEngine-style workflows.
- Neural architectural DSL prediction plus deterministic compilation: ArcPro and neural procedural reconstruction.
- Learned editable 3D construction programs: ShapeAssembly, DeepCAD, CSGNet/PLAD, D2CSG, and DI-PCG.
- Hierarchical block/building programs and program-level editing: CityGenAgent.
- Coordinated multi-building massing conditioned on site context: CoMa.
- Partial architectural action completion: Building-Gym.
- Learned interpretation of user sketches into procedural grammar: Interactive Sketching of Urban Procedural Models.
- Constrained scaffold completion with procedural/retrieved exterior detail: ShellMaker.
- Explicit addition/subtraction for architectural mass exploration: EvoMass.

### Necessary adaptations, unlikely to be novel alone

- Replacing generic CSG primitives with polygon extrusions, sweeps, roof profiles, arches, and repeated bays.
- Adding architectural labels to operations.
- Evaluating a generated program through the repository's existing SDF/CSG stack.
- Conditioning independent per-building generators on a shared block latent.
- Using synthetic programs and fitting approximate programs to real LoD2/LoD3 geometry.
- Keeping facade detail in a separate procedural/retrieval layer.
- Rendering validation contact sheets and operation traces.

### Potentially novel hypotheses

These are the parts worth treating as falsifiable research claims rather than assumed novelty:

1. **One semantic edit distribution for two modes.** A single architecture-specific model can both generate a complete footprint-to-program sequence and interpret/complete a rough local add/subtract gesture into the same typed operation space.
2. **Architectural voids as first-class generative objects.** Courtyards, passages, arcades, light wells, setbacks, and roof cuts can be learned as relational operations with topology/support constraints, not inferred afterward from an anonymous negative mask.
3. **Hard-envelope coordinated block generation.** A selected set of immutable metric footprints can share block-level decisions while retaining independently editable, non-fused, valid per-footprint solids.
4. **Edit locality under program regeneration.** Re-typing, resizing, deleting, or resampling one operation can preserve unrelated program state and geometry while maintaining global validity.
5. **A real-geometry benchmark for semantic carving programs.** Synthetic exact programs, inverse-fitted real pseudo-programs, a small human-audited semantic set, and held-out real geometry could establish a task not covered by current building or CAD datasets.

The first four together form the strongest paper-shaped contribution. Any one alone is vulnerable to being viewed as a straightforward adaptation of the closest systems.

## Strongest novelty risks

1. **Novelty by integration.** Reviewers can reasonably describe the system as ArcPro/ShapeAssembly plus CSGNet plus CityGenAgent under footprint constraints. A new network objective, data formulation, or demonstrated interaction property is needed beyond system assembly.
2. **Superficial typing.** If `courtyard` is only a label attached to an otherwise generic negative box, the method will look like semantic post-labeling. Types need distinct parameterizations, relational constraints, priors, failure rules, or downstream behavior.
3. **CityGenAgent and CoMa erase broad claims.** “Editable city programs,” “coordinated building generation,” “metric polygon programs,” and “context-aware block massing” are no longer safe novelty statements.
4. **Program non-identifiability.** Many different Boolean sequences produce the same final solid. Without a canonical normal form or equivalence-aware loss, supervision can be arbitrary and evaluation of predicted order misleading.
5. **Synthetic-generator ceiling.** Exact synthetic programs may teach only the hand-authored generator's biases. Inverse fitting can compound this by declaring the fitter's vocabulary to be ground truth.
6. **Missing real semantic supervision.** LoD2/LoD3 meshes reveal final geometry, not whether a void is a courtyard, arcade, passage, or accidental gap, nor the operation order that formed it.
7. **Validity is not automatic.** SDF evaluation helps watertight extraction but does not by itself guarantee adequate wall thickness, support, connectedness, or robust Boolean topology. Those must be explicit program validators and evaluation gates.
8. **Interactive mode may become a separate model.** If rough-carve interpretation requires unrelated training data, architecture, or latent state, the “one representation/two modes” claim can collapse into two loosely connected tools.
9. **Architectural scope drift.** Adding rooms, stairs, code compliance, structural simulation, or facade synthesis would blur the massing/void contribution and collide with much denser prior literatures.

## Available code and data

### Code and learned-program baselines

| Resource | What is available | Best use here |
|---|---|---|
| [BuildingBlock repository](https://github.com/Tencent/BuildingBlock) | Training code, Docker workflow, box-layout preprocessing, and linked layout/condition data | Learned architectural component-layout baseline and data-schema reference |
| [Building-GAN repository](https://github.com/AutodeskAILab/Building-GAN) | Training/inference code, pretrained checkpoints, and a downloadable dataset with global/local program graphs and voxel graphs | Architecture-specific structured generation baseline |
| [ShapeAssembly project](https://rkjones4.github.io/shapeAssembly.html) | Paper, source code, program extraction, interpreter, and editing examples | Executable hierarchical-program design and relation vocabulary |
| [CSGNet project](https://hippogriff.github.io/CSGNet/) | 2D and 3D program-induction code | Generic CSG parser and synthetic-program baseline |
| [DeepCAD repository](https://github.com/rundiwu/DeepCAD) | Code, parsed CAD sequences, pretrained models, and STEP export; the paper reports 178,238 models | Sequence validity, command tokenization, and program metrics |
| [UrbanWorld repository](https://github.com/Urban-World/UrbanWorld) | Runnable OSM pipeline and urban layout/appearance modules | End-to-end OSM-conditioned city comparator, not a carve-program baseline |

No official code release was located during this search for ArcPro, CoMa, or CityGenAgent. Their papers/project pages remain usable for task and representation comparisons, but reproducible implementation claims should be marked unavailable until rechecked. Text2CSG's publisher page indicated code/data release intent; availability was not independently confirmed by the cutoff.

### Geometry and context data

| Resource | Relevant contents | Limitation for this task |
|---|---|---|
| [Microsoft Global ML Building Footprints](https://github.com/microsoft/GlobalMLBuildingFootprints) | 1.4B footprint polygons, with height estimates for a subset, under CDLA-Permissive-2.0 | No 3D form, semantic void, or operation history; local quality varies |
| [3D BAG downloads](https://www.3dbag.nl/download) and [schema](https://docs.3dbag.nl/en/schema/concepts/) | Open CC BY 4.0 Dutch buildings at LoD1.2, LoD1.3, and LoD2.2; roof/wall semantics, metric coordinates, and geometry-validity attributes | Strong real massing/roof target, but not LoD3 facade detail or construction programs |
| [BuildingNet](https://buildingnet.org/) | 2,000 buildings, 513K labeled mesh primitives, and 292K semantic components | Exterior part labels are useful for downstream detail, but coverage is small and not carve-sequence supervision |
| [Texture2LoD3 / ReLoD3](https://zenodo.org/records/15178144) | Reconstructed LoD3 facade-rich buildings and associated imagery/data | Final openings can supervise geometry, but not high-level massing operation order |
| [TUM2TWIN](https://github.com/tum-gis/tum2twin) | CityGML LoD2/LoD3 campus buildings, textures, labeled point clouds, and street imagery | Small geographic/style scope; primarily reconstruction data |
| [OpenStreetMap Simple 3D Buildings](https://wiki.openstreetmap.org/wiki/Simple_3D_Buildings) | Footprints, building parts, heights, levels, roofs, materials, and related site context | Tags are incomplete and do not encode a general semantic carving history |
| [IFC `IfcOpeningElement`](https://ifc43-docs.standards.buildingsmart.org/IFC/RELEASE/IFC4x3/HTML/lexical/IfcOpeningElement.htm) | Standard semantics for openings/recesses and an explicit void relationship whose body implies Boolean difference; IFC also defines additive projection relationships | Useful schema inspiration and possible pseudo-label source, but public IFC corpora and high-level courtyard/passage labels are limited |

The largest data gap is therefore not footprints or final meshes. It is paired supervision of:

```text
(footprint set, context, optional rough carve)
                    ->
(canonical typed architectural operation program)
```

## Data strategy implied by the literature

1. **Synthetic exact programs.** Generate valid programs from real footprint distributions, following ArcPro's successful use of cleaned real footprints to ground procedural synthesis.
2. **Canonicalize before learning.** Define equivalence and a normal form for commuting operations, redundant unions, nested subtractions, and geometrically identical sequences.
3. **Fit candidate programs to real LoD2/LoD3.** Treat inverse fitting as latent/pseudo supervision, not unquestioned ground truth. Keep several near-equivalent candidates when the operation history is ambiguous; PLAD and CSGNet provide relevant pseudo-label/inverse-program precedents.
4. **Create a small audited semantic set.** Human annotators should identify void type and relationships on a deliberately small, high-quality validation/test corpus. This is needed to determine whether the model learned “courtyard” rather than merely “negative polygon.”
5. **Synthesize guided-edit pairs from canonical programs.** Perturb or partially erase one operation to make rough carve inputs, then require recovery of its type, geometry, relations, and unchanged sibling operations.
6. **Hold out real regions and buildings.** Final geometry and interaction evaluation must use real held-out targets, with geographic/building identity separation to avoid memorizing footprint/form pairs or retrieved detail.

## Implications for the Wayfinder specification

The literature search changes the likely specification in six ways:

1. **Position the representation narrowly.** Call it a *semantic architectural edit program* or *solid-first architectural CSG program*, not merely an architectural program, procedural building generator, or “degenerative mesh model.”
2. **Make the dual-mode task central.** Autonomous generation plus rough-carve interpretation in one operation space is the clearest unoccupied seam.
3. **Specify a canonical program algebra.** The spec must define operation equivalence, ordering, invalid references, deletion, resampling, and local regeneration before choosing a neural model.
4. **Separate three constraint layers.** Syntax validity, program/architectural validity, and final geometric validity must be measured independently.
5. **Use direct baselines.** At minimum compare against CoMa-style extrusion generation, CityGenAgent-style hierarchical programs, ArcPro-style layer programs, Building-Gym partial completion, generic CSG induction, and deterministic procedural/random sampling.
6. **Do not claim broad firsts.** Avoid claiming the first learned procedural building system, first editable city program, first footprint-conditioned building generator, first neural CSG generator, or first hybrid learned/procedural architecture system.

## Recommended next investigations

The novelty result is strong enough to continue wayfinding, but not yet to write an implementation spec. The next decisions should be resolved in this order:

1. Define the minimal architectural operation ontology and canonical normal form.
2. Test whether real LoD2/LoD3 buildings can be fit adequately with that ontology without free-form residuals.
3. Define the rough-carve interaction and edit-locality benchmark precisely.
4. Decide whether block coordination is a shared latent, explicit block program, relational graph, or constrained joint decoder.
5. Specify real/synthetic splits and the small audited semantic annotation set.
6. Only then select the learned sequence model and training objective.

The current novelty estimate is **medium-high for the exact system contract, medium for a publishable method, and low if presented only as an integration of known components**. A defensible paper would need to demonstrate that semantic typed carving and shared dual-mode generation/editing provide measurable benefits over unlabeled CSG, final extrusion prediction, and separate generation/edit models.
