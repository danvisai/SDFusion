# GenerativeTowns

A generative, **sculptable** 3D town generator (built on SDFusion).

**Research thesis — two claims: generality comes from *transform* + *composition*, not data scale.**
You can never enumerate every building, so instead of scaling data:
- **C1 (transform, not generate).** Never sample a building from noise (degenerate at achievable data
  scale). Instead **project** a rough input onto the manifold of real buildings via **SDEdit**. The
  *same* projection does **generation** (from a footprint blockout) and **editing** (from a user sculpt)
  — so **editability is core, not a wrapper**. Transform recovers *massing*, not detail (evidence: the
  residual-correction net aligns footprint to ~0.999 IoU while detailed-shape IoU stays ~0.2).
- **C2 (compose, not synthesize).** Detail is ill-posed to *generate* at achievable data scale, so it is
  **composed/retrieved from understood, real architectural elements** — which beats monolithic
  detail-generation at equal data (the data-scaling curve). The failed detailizers (REPA/adaLN/L1-GAN,
  Layer-A) are the evidence detail-generation is the wrong tool.

Learned models make the *decisions*; deterministic procedure + retrieval do the *realization*.

**Core vs wrapper.** The SDEdit **transform** (snap/sculpt) is **C1 — core**. The *peripheral* edit
features (weathering, ornaments, sketch-relief, recipe-closure round-trip) remain the **demo wrapper**:
they make the artifact impressive but are not what the paper proves.

## Project status (updated 2026-07-26)

The living status lives in the **wayfinding maps** under `docs/wayfinding/` (each mirrors a GitHub
issue map and carries its own tables + montages); this section is the index into them. The current
active thread is **massing-surface crispness** — a *massing-fidelity* (C1) sub-problem, distinct from
the C2 detail-composition thesis above.

- **Solid massing — DONE & shipped.** `docs/wayfinding/solid-massing-generation/` (map #24): the
  LoD2-only from-scratch retrain passes the #27 acceptance gate (footprint-IoU 0.43 → ~0.89, solid
  footprint-matching blocks). This checkpoint is the accepted massing generator. "Breaking apart" was
  a BuildingNet thin-shell artifact, not a model failure.
- **Surface crispness, first pass — CLOSED negative.** `docs/wayfinding/massing-surface-fidelity/`
  (map #34, closed 2026-07-23): the sampled massing is *solid but wavy*; the roughness is **prior-side**
  (`35-roughness-diagnosis.md`), and the map's cheapest-first levers (sampling knobs, then a decoded-x0
  smoothness fine-tune) all fell short (`phase1-result.md`, `phase2-result.md`). Originally deferred; the
  crispness pursuit was then **reopened** by maps #52/#58 below.
- **Crisp clean massing — COMPLETE (locates the ceiling).** `docs/wayfinding/crisp-massing-model/`
  (map #52, commit `c459564`). Key finding: **the VQVAE codec is NOT the crispness bottleneck** —
  `decode(encode(GT))` ≈ **0.0044** roughness ≈ GT floor **0.0041**, so a crisp building *is*
  representable at 64³; **the diffusion is what produces lumpy/wavy massing.** Two fixes ruled out
  cheaply: composite-over-extrusion (#56 — SDF-combine on the 64³ grid corrupts crispness;
  `residual-retrain-design.md` is therefore **superseded**) and a post-decode SDF refiner (#54 —
  bounded residual + sharpness losses plateau at ~**0.0047**, cannot reach GT). The forward menu is
  `representation-ceiling-menu.md`; comparison figures are `refiner-v3-vs-v1.png`, `gate56-*.png`,
  `residual-decomp-*.png`.
- **Diffusion latent accuracy — IN PROGRESS.** `docs/wayfinding/diffusion-latent-accuracy/`
  (map #58). #59 (latent-space corrector, commit `40e9c55`) **CLOSED NEGATIVE**: correcting the
  diffusion's *latent* also plateaus at the same ~0.0047 wall (`latent-corrector-result.md`, table +
  `latent-corrector-montage.png`) — so **post-hoc correction is doubly ruled out** (SDF #54 *and*
  latent #59). #60 (x0-sharp diffusion fine-tune — warm-start the map-#24 prior with the decoded-x0
  smoothness regularizer already in `stage3a_model.forward()`) is **currently running**. If it
  over-smooths or plateaus, the durable fix is a **query-based implicit / vecset decoder** (menu
  option 2) — moving crispness off the dense-grid diffusion and into the decode.

## Language

**Symbolic recipe**:
The compact, reversible description of a building — footprint, recipe parameters, edit/op
list, per-building style image, weather seed, element/ornament ops — from which geometry is
deterministically realized. The building *is* this recipe, not the mesh it produces.
_Avoid_: mesh, model, asset (those are outputs of a recipe, not the building itself)

**Decision**:
A choice about what a building looks like, made by a *learned model* (massing params,
part typing, layout, retrieval ranking, texture/appearance). Decisions are re-rollable.
_Avoid_: generation (overloaded — reserve for the literal act of a net synthesizing pixels/voxels)

**Realization**:
The deterministic construction of geometry from decisions — recipe→SDF, CSG sculpting,
retrieval-fit crop sampling, marching cubes, weathering, UV/PBR. No learned net emits the
final surface.
_Avoid_: rendering (that's the appearance/texture step specifically)

**Editable / Reversible**:
The property that any single decision can be changed or re-rolled without destroying the
rest of the building. The load-bearing claim of the project.
_Avoid_: parametric (too generic), non-destructive

**Frozen mesh**:
The baseline being argued against — a single baked mesh emitted by an end-to-end generator,
where any edit requires regenerating the whole object and cannot preserve unrelated parts.
_Avoid_: static mesh, output mesh

**Massing**:
The generatable part of a building: low-spatial-frequency geometry *above* the detail scale s*
— base mass, wings, overall roof form. Produced by recipe-param diffusion + Stage 3a. The claim:
this is well-posed to generate from {footprint, class, height, style}.
_Avoid_: base shape, blockout (blockout is the crude user primitive, not the generated mass)

**Detail**:
The non-generatable part: high-spatial-frequency geometry *below* s* — windows, doors, balconies,
cornices, ornament, facade articulation. The claim: ill-posed to *generate* at achievable data
scale, so it is *composed* (procedural) or *retrieved* (real element library). The finding is that
the semantic detail set and the sub-s* scale band **coincide**.
_Avoid_: fine geometry, decoration (ornament is one kind of detail, not all of it)

**Detail scale (s\*)**:
The spatial scale separating massing from detail, **fixed a priori** at **1.0 m = 5 voxels @96³**
(≈3 voxels @64³) — tied to the 64³ massing generator's resolution limit, *not* chosen from data
(ADR 0004). The massing/detail **coincidence is then a TEST** against this fixed `s*` (and can fail),
never a line drawn to fit the result.
_Avoid_: cutoff, threshold (name it s* everywhere)

**Massing fidelity**:
How well generated *massing* matches the target — measured **paired** (Chamfer / IoU to the
specific held-out real building), because massing is determined by footprint + height.
_Avoid_: shape accuracy

**Detail fidelity**:
How well generated *detail* matches real — measured **distributionally** (never paired, because
the constraint underdetermines detail). Primary metric: **rendered-facade FID** vs real facade
renders under an identical neutral shader; supported by a 2AFC human-preference study.
Representation-agnostic (both arms render to images), so it compares the decomposition and the
from-scratch monolith on equal footing.
_Avoid_: detail accuracy, reconstruction error (paired error is wrong for detail)

**Monolith (baseline)**:
The end-to-end footprint→detailed-SDF generator the decomposition is compared against — one SDF net
trained from scratch on **real** (coarse-massing → BuildingNet-detail) pairs at 25/50/100 % of
BuildingNet, equal compute. NOT the synthetic `detail_pairs_v1` detailizer (that imitates the
composer — a separate documented negative). No external web-scale giant (an image-conditioned model
cannot take a footprint).
_Avoid_: end-to-end model (ambiguous), detailizer (reserve for the synthetic composer-imitator)

**Snap / SDEdit transform**:
The **C1 core mechanism**: SDEdit projection of a crude input (footprint blockout or user edit) onto the
real-building manifold via the Stage 3a prior (`/snap_sdf`). Localized snap makes an added mass coherent;
the *same* operator, applied to a footprint blockout, is **generation**. Never sampled from noise.
_Avoid_: fit, blend, "inpainting" (it is the generation+editing operator, not a peripheral tool)

**Transform (manifold projection)**:
The **C1** operation — moving an off-manifold rough input toward the distribution of real buildings.
Realized by SDEdit (Stage 3a, live) or, in an earlier tried-not-live variant, by k-NN retrieval + a
residual-correction UNet. Recovers massing/footprint, **not** detail.
_Avoid_: sculpt (that's the UI verb; transform is the operation)

**Composition**:
The **C2** operation — building detail from understood, real architectural *elements* (retrieval +
learned placement + procedural instantiation) rather than synthesizing it with a net.
_Avoid_: detailing, generation (of detail)

**Make-it-architecture**:
Interpreting crude placed primitives (a box on a roof, a subtracted region) as *contextual*
architectural elements (tower, dormer, arcade) via learned typing + retrieval/procedural
realization.
_Avoid_: detailing, interpret

**Solid-first architectural carving**:
A modeling strategy that begins with valid, footprint-constrained solid massing, then has learned
models decide an ordered, editable program of structural-scale additions and subtractions: courtyards,
setbacks, terraces, passages/arcades, roof cuts, wings, and large bays. Deterministic SDF/CSG realizes
the program as part of the symbolic recipe; any mesh is a derived terminal output, not the generated
state. Windows, doors, cornices, and ornament remain separate facade/detail decisions realized by
exact procedural carves or retrieved elements.
The same representation supports autonomous footprint-to-program generation and user-guided carve
interpretation, where a rough subtraction is typed, completed, and constrained without changing
unrelated operations.
_Avoid_: degenerative mesh modeling (suggests degradation or decimation), mesh generation

**Semantic architectural edit program**:
The per-building structured object predicted from a metric footprint, height, building class, and
site context.
It is an ordered graph of architecturally named operations (for example courtyard, entrance passage,
arcade, terrace setback, roof cut, light well, wing, or roof volume), each with geometry, parameters,
support/containment relations, and validity constraints. Text may modify intent, but it is optional
conditioning; the program rather than a mesh or dense final SDF is what the model generates.
One shared operation space supports both autonomous footprint-to-program generation and learned
interpretation/completion of a rough user add/subtract gesture.
_Avoid_: text-to-3D (not the primary contract), carve mask (lacks architectural semantics)

**Footprint set / edit selection**:
The system input is a site-contextualized set of building footprints. Each footprint owns its own
footprint envelope and semantic architectural edit program. An edit selection is one footprint or a user-selected
subset; subset operations may coordinate height rhythm, courtyard/setback logic, roof family, or style
across the selected buildings, but they produce separate per-footprint programs and never silently
merge solids across footprint/property boundaries.
_Avoid_: single-building input, fused block mesh

**Footprint envelope**:
The deterministic, filled, valid starting solid for architectural carving: an exact metric footprint
polygon extruded from ground elevation to a target/eaves height. It contains no facade detail or thin
shells. Roof form, upper profile, and other massing variation enter later as typed operations, so the
base guarantees footprint adherence and a watertight interior before learned decisions begin.
_Avoid_: massing (the envelope precedes generated massing decisions), blockout (a crude user guide)

**Constrained architectural volume**:
The primary geometry carried by a semantic operation: an editable 2D polygon plus height interval, swept
arch/profile, roof profile, repeated bay, or other parameterized volume with explicit architectural
constraints. Learned free-form residual fields are a later optional refinement and may not violate the
footprint, minimum wall thickness, connectivity, or watertightness.
_Avoid_: arbitrary voxel mask, free-form mesh edit

**Recovered carving program**:
An approximate semantic architectural edit program fitted to a real LoD2/LoD3, CityGML/BIM, or
segmented-mesh
building for supervision. It is a pseudo-label whose reconstruction error and ambiguities must remain
visible; it is not the building's known authoring history. Synthetic procedural buildings may supply
exact programs, but success is evaluated on held-out real geometry.
_Avoid_: ground-truth recipe, true construction sequence

**Visual carving trace**:
The required visual QA artifact for validation and final evaluation: fixed plan, facade, isometric,
and sectional views of the footprint envelope, intermediate typed operations, and final solid, with
add/subtract colors, constraint overlays, recipe/seed metadata, metrics, and failure reasons. It is
versioned evidence, not a required per-checkpoint training visualization.
_Avoid_: screenshot (underspecified), beauty render (hides geometric validity)

**Preview/finalization split**:
The interaction contract for user-guided carving. A rough add/subtract operation receives an immediate
deterministic preview; learned typing, completion, and selected-block coordination return within a few
seconds; higher-resolution validation, visual carving traces, and final mesh extraction may complete
asynchronously.
_Avoid_: blocking every edit on final extraction, calling a rough preview the final solid

**Recipe closure**:
A property of the *demo wrapper* (NOT the research claim): every pipeline *stage* consumes the
recipe and emits a recipe (never a frozen mesh), so a decision stays editable after later stages.
Supports the "editable" selling point; it is not what the paper proves.
_Avoid_: composability (too generic), non-destructive pipeline

**Stage**:
One transform in the pipeline (massing, snap, make-it-architecture, weathering, texture,
photoreal, export). The unit the closure claim quantifies over ("for every stage S…").
_Avoid_: step, layer (layer is used for the 3-layer architecture split, not pipeline stages)

**Leak**:
A stage that violates closure — it emits baked geometry that cannot be recovered/re-edited
from recipe fields (as opposed to a stage whose output is re-derivable from stored
parameters + seeds).
_Avoid_: bake (bake is a legitimate terminal realization; a leak is the *loss of editability*)
