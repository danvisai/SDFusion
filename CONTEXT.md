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

## Project status (updated 2026-08-21)

The living status lives in the **wayfinding maps** under `docs/wayfinding/` (each mirrors a GitHub
issue map and carries its own tables + montages); this section is only the index into them. Massing
fidelity (C1) has been the active problem since 2026-07. The C2 detail-composition thesis above is
unchanged and is **not** currently being worked — its evidence-package effort (map #11) was closed
stale on 2026-08-21 with its record kept in `.scratch/transform-composition-proof/` and `tickets.md`.

### Active

- **Latent token order — IN PROGRESS.** `docs/wayfinding/latent-token-order/` (map #87). The pair
  training target was corrupted by token ordering; #88–#91 captured the codec's query positions and
  rebuilt the aligned cache. #92's arms train against `v4_surf@240k` as the control — arm A closed
  at step 240000 (its best checkpoint and its first non-zero), arm N (NL/DE only, PLATEAU excluded
  as LoD1) is still running. Checkpoints are scored continuously by
  `scripts/foundations/watch_checkpoints.py` into `execution/artifacts/`.
- **Whole-volume voxel transform — IN PROGRESS (planning + throwaway prototypes only).** Map #113.
  Decides whether an A2-only whole-volume voxel correction can satisfy hard footprint/validity
  invariants *and* preserve editability. #114–#116 are settled (dense absolute binary 64³ state;
  authentic replay supervision; recipe posture deferred to an explicit gate); #117–#125 are open.
  This is a **competing empirical route beside** solid-first semantic carving (#1), not a silent
  replacement — the semantic architectural edit program remains authoritative, and this map may not
  rewrite `CONTEXT.md` or an ADR without the explicit recipe-compatibility decision.
- **Footprint-drawn town demo — IN PROGRESS.** `docs/wayfinding/footprint-town-demo/` (map #97).
  The standalone town editor generating from A2, streamed into the viewport. #102/#104/#105 open.
- **Bitmagic-inspired town experience — IN PROGRESS (Codex).** Map #106: a recipe-preserving town
  interaction and presentation exploration. #107–#112 open.

### Settled

- **Solid massing — DONE & shipped.** `docs/wayfinding/solid-massing-generation/` (map #24):
  footprint-IoU 0.43 → ~0.89. "Breaking apart" was a BuildingNet thin-shell artifact, not a model
  failure. This checkpoint is the accepted dense-grid massing generator.
- **Surface crispness, first pass — CLOSED NEGATIVE.** `docs/wayfinding/massing-surface-fidelity/`
  (map #34): the roughness is prior-side, and the map's cheapest-first levers all fell short.
- **Crisp clean massing — COMPLETE (locates the ceiling).** `docs/wayfinding/crisp-massing-model/`
  (map #52): the VQVAE codec is **not** the crispness bottleneck — `decode(encode(GT))` ≈ 0.0044 vs
  a GT floor of 0.0041 — **the diffusion is**. Composite-over-extrusion (#56) and a post-decode SDF
  refiner (#54) were both ruled out cheaply.
- **Diffusion latent accuracy — CLOSED.** `docs/wayfinding/diffusion-latent-accuracy/` (map #58):
  #59 (latent-space corrector) and #60 (x0-sharp finetune) both hit the same ~0.0047 wall, so
  post-hoc correction is ruled out in **both** the SDF and latent domains. The durable fix was to
  move crispness into the decode — taken up by map #61.
- **Crisp massing via a query-based decoder — model shipped, map still open.**
  `docs/wayfinding/crisp-massing-vecset/` (map #61): the A2 vecset massing diffusion is trained,
  published, and is the current research line (see README). #66/#67 remain open as specs, and the
  map's own "not yet specified" fog — whether editing survives a token-set latent — is exactly what
  map #87 is now burning off.
- **Vecset convergence — COMPLETE.** `docs/wayfinding/vecset-convergence/` (#69–#85, all closed):
  the evaluation harness, the decoded-surface loss, the height-input decision, and the band-fix
  findings that map #87 inherited.

### Known gaps in this record

- Maps #106 and #113 have no `docs/wayfinding/` folder yet; #113 names
  `docs/wayfinding/whole-volume-voxel-transform/` as its required home.
- `effort:solid-first-carving` (#1–#9) is specified but unstarted; it is deliberately kept open.
  #10, #126, #127 and #128 are done — see `docs/wayfinding/solid-first-subtractive-modeling/`.
- 🔑 **#127 broke the no-op.** A 3.4M-parameter footprint→height-map generator carves: `extra`
  0.2308 → **0.0603** with `vs_input` 0.8432, against the shipped 49M model's 0.2357 at 0.9852
  vs-input. The pattern that closed #69–#92 was a property of the output space, not of the task.
  ⚠️ Its **pre-registered arm missed the 1-NN bar** (0.1178 against 0.1031); the arms that clear it
  were run after seeing that. ⚠️ **The montage disagrees with the scorecard** — every trained arm
  returns a rounded mound where the real roof is planes meeting at a ridge, and three amplitude
  statistics failed to separate them. The open problem has moved from *amount* to *form*.
  ✅ **The human reviewed the montages on 2026-08-28 and accepted them**: this meets the scope
  *"input a shape, get a blockout that looks like a building"* where earlier approaches did not.
  Recorded as their judgement on criterion 1; it does not change the scalar record above.
  **Served in the demo** by `town_generate_service.py` behind an `arm` knob (default still `a2`),
  with a `/arms` comparison page — ~0.1 s/building against A2's ~7 s, because a height map needs no
  codec. ⚠️ It is **deterministic**: identical footprints give identical buildings, so a town needs
  the `roof_variation` knob (default 0 = the arm that was scored). Weights: `weights/massing-heightmap/`.

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
— base mass, wings, overall roof form. Produced by a diffusion-based massing generator conditioned
on {footprint, class, height, style} — Stage 3a's dense-grid diffusion, or A2's vecset/Dora-latent
diffusion (map #61); both realize the same C1 transform (ADR 0003), differing in representation.
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
How well generated *massing* matches the target — measured **paired** (Chamfer / IoU to the specific
held-out real building).
⚠️ **Massing is NOT determined by footprint + height.** That was this entry's stated justification
until #126 measured it: two real held-out buildings whose footprints agree to IoU ≥ 0.90 and whose
heights agree within 5% still differ by a median 3D IoU of **0.886** over all matched pairs (0.829
on the carve-needing subset), one re-rendered on the other's exact footprint at its exact height.
The conditioning leaves real architectural freedom and the held-out row is one valid answer among
several.
Paired scoring **survives on the C1 transform reading instead** — "was *this* blockout or sculpt
projected correctly" is well-posed however many valid buildings share the footprint.
🔑 #126 further decided that for **new** massing work the `missing`/`extra` split leads the
scorecard and the aggregate 3D IoU is a diagnostic, because on the **median** a real building and
the envelope are indistinguishable (0.8295 both) while the split separates them unanimously
(`extra` 0.097 against 0.206, winning every decided offer).
⚠️ This is in **tension with map #87's pre-registered gate 4** ("3D IoU split into missing vs extra
— diagnostic only, never pass/fail"), which was fixed before #92's run and is **not** overridden
here: #92 is judged on the gates it pre-registered. See
`docs/wayfinding/solid-first-subtractive-modeling/126-massing-scoring.md`.
Scored on the **carve-needing subset** wherever a generator's carving is the question: 303 of the
714 held-out buildings need no carve, and that no-op majority flatters every aggregate.
⚠️ **The split is blind to roof form.** #127 measured an arm scoring `extra` 0.000 on a building
while looking worse than the blockout it started from — `extra` charges only volume *above* GT, so
a rough or wrongly-shaped surface underneath it is free. Three amplitude statistics (mean height
step, second difference, local extrema) were tried and **none separates a mound from a roof**,
because GT is itself terraced at 64³. Until one exists, the **montage decides form** and the split
decides surplus. See `docs/wayfinding/solid-first-subtractive-modeling/127-height-map-generator.md`.
_Avoid_: shape accuracy, 3D IoU as a lone number, "determined by footprint + height"

**Footprint fidelity — fringe / spill / uncovered**:
The three-way split of footprint error, never reported as one number. Measured on the vertical
projection of the generated massing against the conditioning footprint.
**Fringe** is disagreement within *s\** of the footprint boundary — a discretisation effect of the
64³ grid, present even when the model is right, so it is **reported and ignored**.
**Spill** is massing built *outside* the footprint. **Uncovered** is footprint left unfilled. Both
count. Splitting them exists because a single footprint-IoU conflates the harmless with the real:
their ratio varies from 21% to 100% between buildings, so the aggregate disagrees with what a human
sees in a plan view.
_Avoid_: footprint-IoU as a lone number, footprint error (says which, not what kind)

**Allowance**:
The tolerated fraction of footprint area for *spill* and *uncovered* before a building fails
footprint fidelity. A **decision**, not a measurement — distinct from *s\**, which is fixed a priori
by ADR 0004. Recorded in one place in code so it cannot drift.
_Avoid_: threshold, tolerance (tolerance is s\*, which is not negotiable)

**Footprint solidity**:
Footprint area divided by its convex-hull area. 1.0 is convex; lower means re-entrant — courtyards,
L-plans, terraced party walls.
_Avoid_: complexity, concavity (unquantified)

**vs input**:
Overlap of a projection with the **footprint envelope** it started from. 1.0 means the model returned
its input unchanged. Since generation *is* projection (ADR 0003), a quality score without this is
unattributable: a near-no-op inherits the envelope's perfect footprint and is scored for it.
_Avoid_: no-op rate, self-similarity

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
