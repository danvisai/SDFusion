# SDFusion output-quality and research audit

**Date:** 2026-07-13

**Repository:** `SDFusion`

**Purpose:** Decide what the current outputs actually prove, identify the highest-leverage improvements, and turn the repository's plans into a defensible experiment sequence.

**Companion source review:** [research/SDFUSION_RELATED_WORK.md](research/SDFUSION_RELATED_WORK.md) contains the fuller primary-source survey, public implementation/data links, and method-by-method gap analysis behind the research recommendations in this document.

## Executive verdict

The repository contains a broad and promising system: footprint-conditioned SDF generation, editable symbolic recipes, learned part planning, real-part retrieval, sculpt repair, OSM town assembly, and several appearance paths. The engineering breadth is real. The current visual and quantitative outputs, however, do **not yet support the full research claims**.

The immediate problem is evidence integrity rather than a lack of additional models:

1. **The transform claim is partially supported.** On 27 held-out buildings, SDEdit improves mean footprint IoU from `0.356` to `0.592`, but full-volume IoU remains low (`0.065` to `0.090`). The images show that the model preserves a coarse footprint more reliably while still producing hollow, fragmented, or over-smoothed geometry.
2. **The sculpt claim has only preliminary evidence.** The new strength montage is a useful smoke test, but contains three simple edits on the same box-like base, has no run manifest or tabular metrics, and shows the added forms fading as strength rises.
3. **The composition claim is currently blocked by semantic-label quality.** Several BuildingNet label assignments were inferred heuristically. The resulting element montages visibly mix slabs, railings, facade fragments, crosses, domes, and other geometry into the advertised balcony, stairs, and tower pools.
4. **The `monolith_v3` result is not a successful baseline.** Its mean occupancy matches the real data, but most generated examples are empty or tiny fragments. The low-pass conditioning pairs often erase thin BuildingNet shell geometry before training, so occupancy alone gives a misleading conclusion.
5. **Recipe diffusion learns coarse fit but little architectural language.** Sampled IoU retains about 98% of the fitted-recipe ceiling, yet most styles yield nearly identical boxes with superficial caps or towers. The planner also overproduces windows, roofs, doors, and towers.
6. **Town and appearance outputs remain demo-stage.** They demonstrate end-to-end wiring, but show height outliers, overlaps, thin spikes, repeated building forms, weak contextual fit, and expensive appearance operations.

The recommended order is therefore:

> repair labels and experiment provenance → validate monolith pairs → finish the transform/sculpt evidence → rerun the equal-data composition study → improve town context and appearance.

Training another large model before the first two gates would make the comparison more expensive without making it more trustworthy.

## Audit scope and method

This audit covered:

- all **272 first-party Python files** under the repository's source, model, dataset, script, scene, tool, and root-level code paths;
- the current thesis and planning documents: `CONTEXT.md`, `tickets.md`, all ADRs, the three `execution/` plans, the proof PRD/map/issues, the hybrid pipeline plan, professor report, and Claude plan/memory files available in the workspace;
- every top-level family under `outputs/`, plus the JSON experiment artifacts in `execution/artifacts/` and relevant server logs;
- quantitative CSV/JSON/NPZ summaries and visual montages for massing, monoliths, element retrieval, part layouts, recipe diffusion, sculpt flows, town generation, and appearance;
- current primary research and official implementations relevant to building generation, procedural/learned composition, OSM-conditioned cities, mesh extraction, appearance, and generative evaluation.

All 272 Python files were parsed for imports, classes, functions, and module documentation; zero syntax failures were found. High-impact training, inference, data, service, edit, and evaluation modules were then read in detail. “All modules” below therefore means an exhaustive static inventory plus close reading of the paths that determine the present outputs—not a claim that every branch of every dependency was dynamically exercised.

### Code inventory

| Area | Files | Approx. LOC | Main responsibility |
|---|---:|---:|---|
| `scripts/` | 175 | 33,949 | preparation, training, evaluation, serving, OSM and appearance experiments |
| `models/` | 46 | 12,428 | VQ-VAE/diffusion, recipe, part layout/refinement, monolith, retrieval, ARAP and GS models |
| `scene/` | 16 | 3,303 | SDF recipes/editing, element placement, OSM, Gaussian composition and rendering |
| `datasets/` | 17 | 2,367 | BuildingNet, 3DBAG, Stage 3, recipe, monolith, hybrid and retrieval datasets |
| `utils/` | 8 | 1,547 | shared geometry, rendering and experiment support |
| `tools/` | 7 | 663 | Blender-side integration |
| repository root | 2 | 436 | top-level training entry points |

The largest coordination modules are `scripts/server/inference_service.py` (1,173 lines), `models/sdfusion_model_img2shape.py` (938), `models/networks/diff_recipe.py` (913), `scripts/server/refine.py` (896), `scripts/server/layout_detail.py` (890), and `models/stage3a_model.py` (856). These are also the areas where interfaces and failure provenance matter most.

### Verification performed

The following lightweight suites were run without changing checkpoints or outputs:

```bash
PYTHONDONTWRITEBYTECODE=1 env -u LD_PRELOAD -u LD_LIBRARY_PATH \
  ./sdfusion/bin/python -m unittest \
  datasets.test_monolith_pair_dataset \
  models.networks.test_monolith_unet \
  models.test_monolith_diffusion \
  scripts.eval.test_fid \
  scripts.eval.test_measure_scale_spectrum \
  scripts.eval.test_transform_vs_noise \
  scripts.server.test_element_retrieval_baseline
```

Result: **85 tests passed in 12.626 seconds**.

Warnings worth fixing were exposed: unclosed JSON handles in evaluation/library utilities, a tensor-with-gradient conversion in a monolith test, Matplotlib cache fallback, and—in existing server logs—a Stage 3a divide-by-zero warning, deprecated diffusion arguments, checkpoint warnings during inference, and slow image preprocessing.

## End-to-end system map

```text
BuildingNet / 3DBAG / OSM / procedural recipes
                    │
                    ▼
       normalized SDFs, footprints, labels
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
 Stage 3a latent SDEdit   Recipe parameter diffusion
 massing/from-noise       + deterministic SDF recipes
          │                    │
          └─────────┬──────────┘
                    ▼
     editable building recipe + SDF operations
                    │
       ┌────────────┼───────────────┐
       ▼            ▼               ▼
 part planner   real-part      procedural detail
 + refiner      retrieval       and ornaments
       └────────────┼───────────────┘
                    ▼
          mesh / town composition
                    │
        ┌───────────┴────────────┐
        ▼                        ▼
 PBR / relief / texture     SDF-to-GS / neural views
        └───────────┬────────────┘
                    ▼
        FastAPI service + Blender tooling
```

### Module-family findings

| Family | What it does | Audit finding |
|---|---|---|
| Dataset and split modules | Load BuildingNet/3DBAG, derive SDF/footprint/recipe/part/monolith pairs, and enforce building-ID splits | Split discipline is one of the strongest parts of the experiment design. Semantic part identity and low-pass pair validity are weaker than the split logic. |
| VQ-VAE and Stage 3a | Compress 64³ SDFs and sample either from noise or by SDEdit from a blockout | The shared transform is the clearest research direction. It needs topology-aware evaluation and a real strength curve, not just footprint IoU. |
| Recipe diffusion | Predict fixed-length recipe parameters conditioned on footprint/class/style/height, then realize them procedurally | The procedural realization is stable and editable. The learned distribution collapses toward boxes and weakly separates styles. |
| Part layout planner/refiner | Predict semantic boxes and repair part sets under edit operations | Counts are badly miscalibrated and placements are cluttered. Constraint checks exist, but are not yet the primary training/evaluation signal. |
| Element library/retrieval | Extract leakage-safe BuildingNet parts and place them into recipes | Leakage control is good. Taxonomy, crop quality, attachment semantics, and per-type filtering are not yet reliable enough for a research comparison. |
| Monolith diffusion | Generate detailed SDFs from a low-pass coarse input under an equal-data budget | The conditioning representation destroys much of the target signal, and the selected v3 checkpoint collapses structurally despite occupancy matching. |
| SDF recipes and editing | Maintain editable building state, apply ordered operations, regenerate and re-cohere detail | This is an important differentiator. Recipe reversibility and untouched-region preservation need explicit tests and serialized provenance. |
| OSM/town modules | Parse footprints/tags, choose height/style, place buildings and compose towns | Functional but context-light. Full Simple 3D Buildings tags, road/neighbour context, collision/height policies, and deterministic fallback logging are needed. |
| Appearance and GS | Lift/compose Gaussian representations, neural-render views, bake textures, paint relief, add ornaments | Multiple experimental routes are wired, but none yet has a consistent multiview/PBR evaluation protocol. Runtime is too high for interactive claims. |
| FastAPI/Blender delivery | Expose generation, edit, snap, detail, town, render, relief, export and ornament endpoints | Integration coverage is useful, but a 1,173-line service and global engine singletons make provenance, isolation and operational testing harder. |
| Evaluation scripts | FID, scale spectrum, transform comparison, diversity, branch/sculpt gates, visual sheets | Useful building blocks, but artifacts are scattered and several headline metrics are sample-limited or semantically contaminated. |

### Cross-cutting code risks

1. **Vocabulary drift.** OSM/retrieval paths recognize five top-level classes including `MILITARY`; recipe, planner and refiner paths use four and silently map unknown classes to residential. Stage 3a separately uses 53 subtypes. A single versioned ontology should own every mapping.
2. **Silent fallback changes experiment identity.** Recipe inference can fall back from the learned composer to random procedural detail. Broad `except Exception` blocks in service/refinement/layout paths make a “learned” output indistinguishable from a fallback unless provenance is recorded.
3. **Packaging is brittle.** There are 173 `sys.path.insert`/`append` occurrences. This makes invocation-dependent imports and duplicate module identities likely.
4. **Training inputs are not self-describing.** `scripts/train_part_composer.py` expects `outputs/part_layouts/layouts.npz`, while the retained output is `outputs/part_layouts_full/part_instances.npz`. The exact training source for the reported composer is not recoverable from the output tree alone.
5. **Experiment and service code are intertwined.** The largest server modules coordinate model loading, fallback, geometry, appearance and transport in the same files. Splitting typed application services from FastAPI routes would make behavior testable without starting the server.

## Plans and documentation reconciliation

| Source | Current role | Reconciliation |
|---|---|---|
| `CONTEXT.md` and ADR 0003 | Current two-claim thesis | Keep as the top-level framing, but narrow the novelty language using the related-work section below. |
| ADR 0001/0004 | Fixed 96³ representation and `s*=5` voxel massing/detail boundary | Keep as a reproducible operating point, not as universal proof of the semantic scale boundary. |
| ADR 0002 | Equal-data monolith versus decomposition experiment | Correct design principle; current monolith input construction and semantic labels must pass validity gates first. |
| `execution/*PLAN_2026-07-10.md` | Current experiment order | The prescribed sculpt sweep should be completed before resuming the C2 decomposition arm. |
| `tickets.md` and proof map | Best record of actual status | Update after the new strength montage is converted into a manifested, quantitative run. C2 is correctly treated as paused. |
| `docs/HYBRID_PIPELINE_PLAN.md` | Earlier retrieval/residual/ARAP direction | Mark explicitly as historical or extract only still-active pieces; it otherwise conflicts with the current SDEdit/recipe architecture. |
| `docs/professor_report/REPORT.md` | Narrative snapshot | Revise claims that SDEdit and the part composer are already proven. The current evidence supports “promising prototype” and “partial transform evidence.” |
| `README.md` | User-facing entry point | It is stale: it calls the branch serving-only, refers to missing handoff/build-plan/compendium files, and reports 13 branch tests where the latest CSV contains 11. |

## Output audit

The `outputs/` directory is about 216 MB and contains 123 PNGs, 89 CSVs, 8 JSONs, 6 checkpoints, 5 NPZ files, 4 videos and 4 GLBs. The wider repository also contains roughly 124 GB under `logs_building`, so output lifecycle and indexing matter.

| Output family | What is demonstrated | What limits it | Recommended next action |
|---|---|---|---|
| `branch_tests/` | Eleven current integration branches pass | README/professor-facing counts have drifted; pass/fail does not measure output quality | Generate a single versioned gate manifest containing git SHA, checkpoint hashes, command, cases, timings and artifacts |
| `sculpt_flows/` | Nineteen functional sculpt flows pass | Mostly a cube and repeated-window facade; not a representative quality set | Add held-out real buildings, diverse edits and geometric preservation metrics |
| `sculpt_strength_sweep/` | Three edits are rendered at strengths 0.1–0.9 | One base, no seeds, no CSV/manifest or realism metric; tower/dome disappear with strength | Run the paired experiment specified below and plot realism versus edit preservation |
| `transform_vs_noise/` | SDEdit improves footprint adherence on 27 held-out examples | Full IoU remains low; FID is undersampled; visible fragments remain | Expand by class/style, add topology/Chamfer/height metrics and paired multi-seed confidence intervals |
| `massing_diversity/` | Sampling paths produce different shapes | High diversity is dominated by pathological fragments; SDEdit can collapse to near-identical boxes | Report quality-filtered diversity and a fidelity–coverage Pareto frontier |
| `scale_spectrum/` | A reproducible `s*=5` operating point is plotted | Only 6/11 categories agree, and categories inherit inferred semantic labels; max extent is a weak scale proxy | Recompute after label audit using per-axis extent, surface area and attachment scale |
| `monolith_pairs_v1/` | Real/coarse pairs and building-ID split exist | Many coarse inputs are nearly empty because thin shell geometry vanishes under 19³ down/up sampling | Reject invalid pairs; reconstruct a massing solid or distance-field coarse channel |
| `monolith_v1/`, `v2/`, `v3/` | Training iterations and occupancy statistics exist | v1/v2 overfill; v3 occupancy-matches but visually collapses to empty/tiny components | Revoke the “strong baseline” designation until structural gates pass |
| `part_labels_full/` | A label-name mapping and montages make assumptions inspectable | Several names were best guesses; visible semantics are mixed even within advertised categories | Create an annotation audit and versioned mapping with confidence, exclusions and confusion matrix |
| `element_library_v1/`, `element_library_train100_v1/` | 2,744 train-only elements from 1,133 buildings with zero split leakage | Usable pools are tiny for stairs/balconies/columns and a universal solidity threshold rejects valid thin parts while admitting wrong bulky fragments | Use per-type geometry/attachment filters and human-verified precision gates |
| `part_layouts_full/` | Part instances were extracted | Extraction inherits the label uncertainty and does not retain enough provenance for the composer checkpoint | Re-extract only audited classes and store source building/component/transform/split for every item |
| `part_layout_planner_v2/` | Class-conditioned layouts can be sampled | Counts strongly overshoot GT: e.g. commercial windows `12.87` vs `2.73`, public roofs `10.12` vs `2.12` | Calibrate a hierarchical count/support model and penalize unsupported/colliding/off-facade boxes |
| `part_set_refiner/` | Learned/deterministic recoherence paths move inserted parts | Simple synthetic boxes; extra junk is often moved instead of removed; no held-out edit benchmark | Evaluate detection, deletion, support and untouched-part preservation separately |
| `part_composer/` | A negative/prototype composer path exists | Training-data provenance is incomplete and semantic inputs are unreliable | Freeze this claim until labels/layouts are rebuilt; retain the current output only as a diagnostic |
| `recipe_param_diffusion_b6*` | Sampled IoU `0.620` approaches fitted-recipe ceiling `0.631` | Most styles are visually indistinguishable boxes; diversity std is very low outside colonial | Increase recipe expressivity and train/evaluate style separability, hierarchy and topology—not just IoU |
| `sample_renders/` and loose previews | Rendering/mesh plumbing works | Hand-selected examples have no manifest or comparison protocol | Move into versioned experiment runs or mark explicitly as illustrations |
| `demo_video/` | Town, sculpt, relief and appearance paths are connected | Repeated towers/boxes, implausible heights, relief noise and low interactive frame rates | Use a fixed demo tile with per-stage timing, collision/height checks and representative failures |
| `server_*.log` | Operational behavior is observable | Warnings and fallback decisions are not summarized beside outputs | Emit structured run provenance and a warning/error summary per request |

### Quantitative evidence snapshot

| Experiment | Current result | Interpretation |
|---|---:|---|
| Transform vs noise, mean footprint IoU | `0.356 → 0.592` | Meaningful footprint-conditioning gain |
| Transform vs noise, median footprint IoU | `0.304 → 0.607` | Gain is not only from a few outliers |
| Transform vs noise, mean full IoU | `0.065 → 0.090` | Overall 3D reconstruction remains weak |
| Render FID, noise vs SDEdit | `225 → 213` | Directionally better, but explicitly undersampled and not conclusive |
| FID sanity, real vs real | point `132.96`; reported CI `[152.6, 183.45]` | Point estimate outside its own interval indicates an inconsistent estimator/bootstrap artifact that must be fixed |
| Monolith v1/v2/v3 generated occupancy | `.325 / .515 / .0157` | v3 matches real mean `.0166` numerically but fails visibly |
| Element library, train-only | 2,744 elements / 1,133 buildings | Good leakage discipline, poor semantic/pool quality |
| Usable pool at solidity `.12` | stairs `0`, upper balcony `0`, balcony `5`, column `8` | Current retrieval cannot support a diverse headline comparison |
| Recipe sampled/fitted IoU | `.620 / .631` | Learner reproduces a limited recipe family; it does not establish realism/style diversity |
| Integration/sculpt flow gates | `11/11`, `19/19` | Functional wiring passes; quality remains an independent question |

## Root causes and concrete improvements

### P0 — repair evidence integrity

#### 1. Establish a trustworthy part ontology

Create one versioned ontology module used by BuildingNet extraction, Stage 3a, recipes, planners, OSM, retrieval, evaluation and the API. Each source label mapping should include `source`, `target`, `confidence`, `annotation_version`, and `excluded_reason`.

Before rebuilding C2 data:

- draw a stratified sample of at least 25 components per claimed element type;
- have two annotators label semantic identity, completeness and attachment surface;
- report agreement and a confusion matrix;
- exclude classes below 90% precision from headline experiments;
- split broad categories such as `roof_structure` into functional subtypes;
- preserve unknown/uncertain rather than mapping them to a convenient class.

The element-quality filter must be type-specific. A thin balcony or railing can be valid despite low solidity; a bulky facade chunk can have high solidity and still be invalid. Useful features include connected-component count, oriented aspect ratio, boundary completeness, facade-normal alignment, attachment-plane coverage, symmetry/repetition cues and source-neighbour context.

#### 2. Rebuild the monolith task before retraining it

The current low-pass pipeline is unsuitable for sparse shell surfaces. Build the coarse condition from a filled massing solid, a robust signed/unsigned distance transform, or a multi-channel representation containing footprint, height envelope and surface distance.

Every pair should pass pre-training gates:

- coarse/target footprint IoU;
- retained height ratio;
- non-empty occupancy;
- largest-component ratio;
- surface-distance and sign-convention checks;
- visual audit of random accepted and rejected pairs.

Every generated result should then report structural metrics in addition to occupancy: Chamfer distance, silhouette/footprint IoU, height error, connected components, largest-component ratio, watertightness, self-intersections and degenerate faces. A checkpoint that matches one scalar moment while failing these gates is not eligible for model selection.

#### 3. Make every output self-identifying

Adopt one immutable run directory per experiment:

```text
outputs/runs/<experiment>/<YYYYMMDD-HHMMSS>-<gitsha>/
  manifest.json
  metrics.csv
  summary.json
  samples/
  montage.png
  stdout.log
  warnings.json
```

The manifest should record commands, configuration, git SHA and dirty state, dataset/split/ontology versions, checkpoint hashes, random seeds, dependency snapshot, renderer/cameras, hardware, runtime, and every fallback actually taken. A learned path that falls back to random procedural detail must be labeled `fallback`, never counted under the learned arm.

### P1 — strengthen the transform and edit claims

#### 4. Measure the SDEdit fidelity–realism curve

Run a paired strength × seed evaluation using at least 25 held-out buildings, four edit types (add, carve, move/resize, delete), five strengths and three seeds. Preserve the same edited input and seed pairing across methods.

Report:

- edit-mask IoU and edit magnitude retained;
- untouched-region IoU/Chamfer;
- footprint and full-volume IoU;
- height error, components and largest-component ratio;
- neutral-render CMMD/KID and generative precision/recall;
- latency and failure rate;
- blinded 2AFC realism and edit-faithfulness preference on a power-analyzed subset.

Select an operating point from the Pareto curve instead of choosing strength from a single montage. The current `0.1–0.9` image suggests that rising strength improves prior dominance by erasing the user's tower/dome rather than coherently integrating it.

#### 5. Improve massing without confusing defects for diversity

- Sample multiple SDEdit candidates and use a transparent constraint/quality ranker.
- Condition on separate footprint, height envelope and optional roofline channels.
- Train on solid/watertight massing or use losses that distinguish interior, surface and topology.
- Add connectivity and footprint-violation penalties or rejection gates.
- Report diversity only among valid samples; use precision/recall or coverage after quality filtering.
- A/B test Neural Dual Contouring against marching cubes for sharper architectural edges. Treat this as extraction improvement, not as evidence of better generation.

### P1 — make composition architectural

#### 6. Replace flat part boxes with a hierarchy and relations

The current planner independently overproduces parts. A stronger representation should model:

```text
building
  ├── mass/wing
  ├── roof planes and roof relations
  ├── facade bays/floors
  │     └── repeated opening groups
  └── supported ornaments and vertical elements
```

Predict counts at their natural level, then sizes/placements conditional on facade/roof support. Enforce containment, support, repetition, symmetry, opening margins, non-intersection and accessibility as differentiable costs or deterministic projection. Include retrieval-only and procedural-only baselines so the learned planner's value is isolated.

#### 7. Expand recipe expressivity before scaling recipe diffusion

The `.620` IoU score is near the ceiling of the current fitted family, so simply training longer is unlikely to create architectural variety. Add wings/courtyards/setbacks, relation-aware roof primitives, facade bay grids, floor schedules, opening groups and explicit attachment frames. Evaluate fitted-recipe ceiling again before retraining. Style accuracy should be measured with held-out classifiers or human judgments plus per-style diversity, not inferred from condition labels alone.

### P1 — improve town coherence

#### 8. Use the full OSM/building context

Parse and retain `building:part`, levels, height/min-height, roof shape/height/direction, material/color, use, road distance and neighbour statistics. Compare against OSM2World as a deterministic zero-learning baseline. Add:

- robust height priors and hard outlier caps by building use/context;
- parcel/footprint containment and inter-building collision checks;
- landmark-aware scale handling rather than one normalization policy;
- retrieval/generation conditions for road orientation, density, local style and neighbouring heights;
- tile-level metrics for coverage, collisions, height consistency, repetition and semantic fit.

### P2 — make appearance measurable and usable

#### 9. Prefer scale-aware PBR retrieval before unconstrained synthesis

Use a tagged, physically scaled PBR library as the default, with generation only for residual customization. For texture/GS paths, freeze cameras and lighting; measure multiview consistency, depth, semantic-part IoU, relighting stability, UV coverage and glTF validation. Keep building/part identity through Gaussian lifting so edits can update only affected appearance elements.

The existing appearance paths taking 160–275 seconds should be exposed as asynchronous “quality” jobs. Interactive mode should cache geometry buffers/material features and use a bounded-resolution preview.

### P2 — reduce architecture and maintenance risk

#### 10. Split service orchestration and centralize contracts

- Move routes into thin FastAPI adapters.
- Put generation, editing, composition, appearance and export behind typed application services.
- Replace global engine singletons with an explicit model registry/lifecycle.
- Replace broad exception/fallback logic with typed failures and structured provenance.
- Package the repository and eliminate path mutation incrementally.
- Close JSON files with context managers and resolve existing numerical/deprecation warnings.

## Related work and what it changes

The project's broad doctrine—learned decisions with procedural realization and composition instead of monolithic synthesis—is now shared by several active systems. The defensible contribution should be narrower and experimentally stronger:

1. one SDEdit transform used for both footprint-conditioned generation and sculpt repair;
2. hard, reversible per-building recipe state shared by town generation and editing;
3. a controlled equal-data scaling comparison of monolith and composition;
4. real architectural-element retrieval with building-level leakage exclusion.

| Work | Closest overlap | What to reuse or compare |
|---|---|---|
| [BuildingBlock](https://arxiv.org/abs/2505.04051) and [official code](https://github.com/Tencent/BuildingBlock) | Transformer component layouts + LLM hierarchy + procedural realization of editable buildings | Point-set/box hierarchy, component validity and coherence baselines; this is the closest challenge to “models choose, procedures build” |
| [ShellMaker](https://arxiv.org/abs/2606.31680) and [project](https://ruiqixu37.github.io/ShellMaker_web/) | Fixed scaffold with parametric roofs, retrieved/generated parts, PBR materials and geometry-aware assembly | Footprint violation, opening-center/size and part-intersection metrics; direct 2026 facade/editability comparator |
| [Building-GAN](https://openaccess.thecvf.com/content/ICCV2021/html/Chang_Building-GAN_Graph-Conditioned_Architectural_Volumetric_Design_Generation_ICCV_2021_paper.html) and [code](https://github.com/AutodeskAILab/Building-GAN) | Graph-conditioned volumetric building massing | Compact program graph and connectivity/program-validity baseline |
| [Roof-GAN](https://openaccess.thecvf.com/content/CVPR2021/html/Qian_Roof-GAN_Learning_To_Generate_Roof_Geometry_and_Relations_for_Residential_CVPR_2021_paper.html) and [code](https://github.com/yi-ming-qian/roofgan) | Relation-aware roof primitives | Replace isolated roof categories with coplanar/collinear relations |
| [GeoTexBuild](https://arxiv.org/abs/2504.08419) | Footprint-to-detailed-building geometry and appearance | Direct pipeline comparison for footprint-conditioned detailed output |
| [UrbanWorld](https://arxiv.org/abs/2407.11965) and [code](https://github.com/Urban-World/UrbanWorld) | OSM layouts, controllable asset generation and refinement | Most important end-to-end OSM-conditioned baseline |
| [CityCraft](https://arxiv.org/abs/2406.04983) and [code](https://github.com/djFatNerd/CityCraft) | Learned city layout, land-use planning, asset retrieval and assembly | Contextual retrieval and large OSM patch data |
| [CityDreamer](https://openaccess.thecvf.com/content/CVPR2024/html/Xie_CityDreamer_Compositional_Generative_Model_of_Unbounded_3D_Cities_CVPR_2024_paper.html) and [code/data](https://github.com/hzxie/CityDreamer) | Compositional, editable unbounded cities | Building-instance vs background factorization and city-scale evaluation |
| [MajutsuCity](https://openaccess.thecvf.com/content/CVPR2026/html/Huang_MajutsuCity_Language-driven_Aesthetic-adaptive_City_Generation_with_Controllable_3D_Assets_and_CVPR_2026_paper.html) and [project/code](https://github.com/LongHZ140516/MajutsuCity) | Controllable layouts/assets/materials and language edits | Current comparator for city-level editability, structure, material and lighting |
| [Proc-GS](https://openaccess.thecvf.com/content/CVPR2025W/USM3D/papers/Li_Proc-GS_Procedural_Building_Generation_for_City_Assembly_with_3D_Gaussians_CVPRW_2025_paper.pdf) | Procedural composition of repeated Gaussian facade elements | Canonical shared element appearance plus per-instance residual |
| [GaussianCity](https://openaccess.thecvf.com/content/CVPR2025/papers/Xie_Generative_Gaussian_Splatting_for_Unbounded_3D_City_Generation_CVPR_2025_paper.pdf) and [code](https://github.com/hzxie/GaussianCity) | Feed-forward city-scale Gaussian generation | BEV-point serialization and spatial Gaussian decoding reference |
| [Texture2LoD3/ReLoD3](https://wenzhaotang.github.io/Texture2LoD3/) and [TUM2TWIN](https://github.com/tum-gis/tum2twin) | Real facade/opening enrichment and LoD3 supervision | Better real facade/detail data and evaluation than heuristically named fragments alone |
| [BuildingNet](https://buildingnet.org/) | Semantic building component data | Continue using with building-ID leakage control, but audit the local inferred mapping against source semantics |
| [Neural Dual Contouring](https://arxiv.org/abs/2202.01999) and [code](https://github.com/czq142857/NDC) | Sharp-feature mesh extraction from SDF/voxels | Low-risk marching-cubes A/B test |
| [FlexiCubes](https://research.nvidia.com/labs/toronto-ai/flexicubes/) | Differentiable geometry/connectivity extraction | Use only if mesh geometry will be optimized downstream |
| [MatSynth](https://www.gvecchio.com/matsynth) | Large physically scaled CC0 PBR material collection | Scale-consistent material retrieval and relighting data |

### Evaluation sources

- Use [Clean-FID](https://github.com/GaParmar/clean-fid) with one frozen renderer/camera/resolution pipeline.
- Do not headline the current small-sample FID. [Finite-sample FID has model-dependent bias](https://openaccess.thecvf.com/content_CVPR_2020/html/Chong_Effectively_Unbiased_FID_and_Inception_Score_and_Where_to_Find_CVPR_2020_paper.html).
- Add unbiased, sample-friendlier [KID](https://arxiv.org/abs/1801.01401) and [CMMD](https://openaccess.thecvf.com/content/CVPR2024/papers/Jayasumana_Rethinking_FID_Towards_a_Better_Evaluation_Metric_for_Image_Generation_CVPR_2024_paper.pdf).
- Add [generative precision and recall](https://proceedings.neurips.cc/paper_files/paper/2018/hash/f7696a9b362ac5a51c3dc8f098b73923-Abstract.html) so a small element library cannot appear “realistic” merely by collapsing.
- Evaluate OSM semantics against the [Simple 3D Buildings schema](https://wiki.openstreetmap.org/wiki/Simple_3D_buildings) and [OSM2World](https://github.com/tordanik/OSM2World).

## Required experiment protocol

| Claim | Dataset and split | Metrics | Baselines | Exit gate |
|---|---|---|---|---|
| C1a: transform beats noise | Building-ID-held-out, stratified class/style/footprint complexity; paired seeds | footprint/full IoU, Chamfer, height, silhouette, topology, KID/CMMD, P/R, failure rate | from-noise, deterministic extrusion/recipe, SDEdit | Statistically reported gain without worse topology/failure; representative montage includes worst/median/best |
| C1b: sculpt repair is the same transform | Held-out buildings × four edits × five strengths × three seeds | edit retention, untouched preservation, realism, collisions, latency | no repair, deterministic snap, SDEdit | A visible Pareto region that improves realism while preserving a declared fraction of the edit |
| C2: composition scales better | Same train buildings, paired test buildings, matched renderer and declared compute/parameter budget | geometry/detail validity, KID/CMMD/P/R, human 2AFC, runtime/memory/editability | monolith, retrieval-only, procedural-only, learned composition | Pair and label gates pass first; result holds under equal data and multiple seeds |
| Recipe contribution | Held-out fitted recipes and real buildings | fit ceiling, style accuracy, per-style diversity, constraint validity, edit round-trip | fitted optimizer, unconditional/majority recipe, learned recipe | Improved expressivity ceiling and separable styles without validity loss |
| Town coherence | Fixed real OSM tiles with unseen geography | coverage, containment, collision, height/context consistency, repetition, human preference | extrusion, OSM2World, retrieval-only | No catastrophic scale/collision outliers; improvement is consistent across tiles |
| Appearance/GS | Fixed geometries, cameras, HDR lights | CMMD/LPIPS, depth, multiview and semantic IoU, relighting, glTF/UV, runtime/VRAM | neutral PBR, retrieval, each learned path | Quality gain survives multiview/relighting and meets declared preview/quality latency tier |

For all learned comparisons, report at least three seeds, building-level bootstrap confidence intervals, class/style-stratified results, and exact sample counts. Automatically publish failures and the worst decile, not only curated montages.

## Prioritized execution sequence

### Gate 0 — make experiments trustworthy

1. Introduce the run manifest/layout and central ontology.
2. Audit semantic labels and rebuild the accepted element library.
3. Rebuild and gate monolith coarse/detail pairs.
4. Fix the real-real FID point/interval inconsistency and add KID/CMMD/P/R.
5. Mark the historical hybrid plan and stale README claims explicitly.

**Stop condition:** do not start the equal-data training comparison until label precision and pair validity reports are checked into `execution/artifacts/`.

### Gate 1 — close C1

1. Convert the current sculpt montage into the paired strength experiment.
2. Expand transform-vs-noise beyond 27 buildings and report topology plus quality-filtered diversity.
3. A/B test multi-channel conditioning, candidate reranking and Neural Dual Contouring independently.
4. Freeze the chosen operating point and checkpoint hashes.

### Gate 2 — run the C2 experiment once

1. Retrain a valid monolith baseline under the declared data budget.
2. Train/evaluate retrieval-only, procedural-only and learned-composition arms on the same split.
3. Use identical neutral renders/cameras and blinded sample presentation.
4. Publish every arm even if the central hypothesis is false.

### Gate 3 — improve the product-facing output

1. Add hierarchy/relations to recipes and part planning.
2. Add contextual OSM conditioning and town guardrails.
3. Stabilize PBR retrieval and identity-preserving appearance/GS lifting.
4. Split the server into typed services and add request-level provenance.

## Claim-status matrix

| Claim | Status on 2026-07-13 | Evidence needed to change status |
|---|---|---|
| One SDEdit transform can condition massing better than from-noise sampling | **Partially supported** | Larger stratified paired test with geometry/topology and sample-efficient perceptual metrics |
| The same transform coherently repairs user sculpt edits | **Preliminary only** | Manifested, quantitative strength curve on diverse held-out buildings and edits |
| Learned recipe parameters preserve procedural fit | **Supported within the limited recipe family** | Current `.620/.631` result is adequate for this narrow statement |
| Recipe model generates diverse, style-specific architecture | **Not supported** | Higher recipe ceiling plus style-separability and valid-diversity results |
| Real-part composition produces coherent architectural detail | **Not currently supported** | Audited taxonomy/library, calibrated hierarchy and held-out validity/realism study |
| Composition beats a monolith under equal data | **Not tested validly yet** | Valid monolith pairs/checkpoint and audited composition data under matched protocol |
| End-to-end town and sculpt workflows are connected | **Functionally supported** | Existing 11 branch and 19 sculpt-flow gates |
| Town/appearance output is production-quality or photoreal | **Not supported** | Context, multiview, material, export, latency and human-evaluation gates |

## Immediate decision list

- **Keep:** the two-claim experimental framing, building-level split discipline, reversible recipe state, transform-vs-noise harness, deterministic realization, and branch/sculpt functional gates.
- **Pause:** new C2 model training, claims based on current part labels, and model selection by occupancy alone.
- **Repair now:** ontology, monolith pair construction, run manifests, FID sanity calculation, fallback provenance, README/status drift.
- **Run next:** the full sculpt strength curve, then the expanded transform-vs-noise evaluation.
- **Research positioning:** claim the combination of unified SDEdit transform, reversible recipe state, equal-data scaling test, and leakage-safe real retrieval—not learned/procedural composition by itself.

## Reproducibility appendix

### High-value current artifacts

- `execution/artifacts/splits_v1_manifest.json`: building-ID split evidence.
- `execution/artifacts/transform_vs_noise.json`: current C1a measurements.
- `execution/artifacts/fid_sanity.json`: exposes the point/CI inconsistency.
- `execution/artifacts/scale_spectrum.json`: fixed operating-point evidence.
- `execution/artifacts/monolith_pairs_v1_manifest.json`: current pair construction.
- `execution/artifacts/monolith_v*_eval.json`: occupancy-based monolith selection history.
- `execution/artifacts/element_library_train100_v1_manifest.json`: leakage-safe extraction counts.
- `outputs/part_labels_full/label_names.json`: local semantic assumptions and confidence.
- `outputs/transform_vs_noise/montage.png`: qualitative C1a comparison.
- `outputs/monolith_pairs_v1/montage.png` and `outputs/monolith_v3/montage.png`: clearest evidence of the invalid condition/collapse problem.
- `outputs/element_library_train100_v1/montage_*.png`: clearest evidence of semantic contamination.
- `outputs/sculpt_strength_sweep/montage.png`: preliminary C1b smoke test only.

### Artifact hygiene rules

1. Do not overwrite `latest/` without retaining an immutable run directory.
2. Do not place training datasets only under `outputs/`; data manifests should point to immutable source artifacts.
3. Do not allow a result table to exist without its command/config/checkpoint/split/seed metadata.
4. Do not allow a montage to omit failures, rejected samples, or per-sample identifiers.
5. Do not count fallback outputs in a learned-method arm.
6. Do not use a scalar distribution match as a substitute for per-shape validity.

## Bottom line

SDFusion already has enough machinery to produce a strong research result. The shortest path is not to expand the system further; it is to make the existing comparison valid. A trustworthy semantic library, a structurally valid monolith task, and a complete transform/edit evaluation would turn the current collection of demos into a coherent research argument. Until then, the most honest summary is: **the unified transform is promising, the editable architecture is valuable, and the composition advantage remains an open hypothesis.**
