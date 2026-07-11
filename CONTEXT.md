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
The spatial scale separating massing from detail, **fixed a priori** as `k` voxels at the working
resolution (≈0.5 m) — *not* chosen from data. The massing/detail **coincidence is then a TEST**
against this fixed `s*` (and can fail), never a line drawn to fit the result.
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
