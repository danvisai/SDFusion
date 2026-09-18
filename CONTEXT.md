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

## Project status (reconciled 2026-09-09)

Current implementation, evidence, and tracker relationships are reconciled in
[docs/PROJECT_STATE.md](docs/PROJECT_STATE.md). Read it alongside
[docs/INTEGRATION_STATE.md](docs/INTEGRATION_STATE.md). Dated experiment reports retain their
original populations and gates; a closed investigation is not necessarily a successful model.

### Current work

- **BuildingWorld corpus extension — map #156, execution-carrying.** Latest committed work
  (`e48b9e9`, 2026-09-06) records the CRS/units policy (#165) and mesh-acceptance standard (#166).
  Audits #157/#158 have reports but remain open for review. Ingestion, derived surfaces,
  pseudo-labels, split, and new baseline (#174–#178) are not implemented. Safeguards and decisions
  (#159–#173) precede them. New retrain-arm execution tickets are not yet specified.
- **Solid-first semantic carving — map #1.** The representation, recovery fitter, bidirectional
  core operations, stable operation IDs, grid snapping, locality tests, validity helpers, carving
  traces, and block-coordination mechanism exist. #8 completed the proof-package design on
  2026-09-06. Open work includes integration #2, provenance #152, split #153, annotations #154,
  guided-edit proxy #179, coordination evaluation #180, and five-arm scorecard #181.
  The map is a planning charter with explicitly scoped implementation/evaluation child tickets;
  closing a decision ticket does not imply its spawned work is finished.
- **Footprint-town demo — map #97.** A separate town service streams massing from A2 or optional
  height-map/retrieval arms. #102/#104/#105/#135 remain open. It is separate from the older
  recipe/sculpt service, whose hidden controls must not be described as visible features.
- **Whole-volume voxel transform — map #113.** Planning/prototype alternative, not a replacement
  for the semantic program. #114–#117 are settled; #118–#124 and companion spec #125 remain open.
  Production implementation and any change to the recipe contract require the map's explicit gate.
- **Town experience — map #106.** Identity/history, Evidence/Showcase, prompt patches, staged
  realization, and sharing work (#107–#112) remains open. Backend operation IDs alone do not
  complete a cross-page recipe-history contract.

### Established results and limits

- Dense-grid solid massing (#24) shipped; surface-refinement and latent-correction investigations
  (#34/#52/#58) closed with documented limitations or negative results.
- A2's codec, token-set denoiser, training, and town serving exist. The representation-choice map
  #61 and PRD #66 have completed their planning purpose; implementation spec #67 retains
  unresolved quality/localized-edit acceptance, not an unbuilt codec.
- Vecset convergence (#69) and token alignment (#87) are closed. Alignment was repaired and
  remeasured, but did not open a usable transform strength band. Do not restart those completed
  runs from an old handover's “next” paragraph.
- #127's small height-map generator breaks the near-identity failure. On the historical 411
  carve-needing cases, CE+median has `extra=0.0603`, `vs_input=0.8432`, and collapse 0.0268.
  The human accepted its blockout appearance on 2026-08-28. That judgment does not establish
  coherent pitched roof form or operation-level editability.
- #155's unbiased, unsmoothed `fit_decode` is implemented and measured **in evaluation only**:
  collapse 0.0268 → 0.0049, `extra` 0.0603 → 0.0807, planar fraction 0.20 → 0.00.
  The decision to ship it has not been wired into the town service. Its fitter also discards the
  recovered operations at the return boundary.
- The assignment/type 2×2 was run, including an accidental combined arm; no cell passes
  `PROGRAM_BAR`. The combined arm is documented by checkpoint name, not as GitHub #140:
  actual #140 is bidirectional Layer/Ramp editing.
- #131 measured the lossless polygon budget; #134's direct few-vertex fitter was tested and killed.
  Its naive floor control reduced spikes at a substantial over-carving cost. Neither is unstarted.
- C2's transform/composition proof effort (#11) is closed stale. The thesis/ADRs remain recorded
  hypotheses, not a completed comparative proof. The old demo-cleanup effort #47 is closed; its
  sibling #50 (fixing the *old* `inference_service.py` recipe generation) is still **open**, not
  closed or formally marked superseded — #50's own record already points at A2 (#61) as the eventual
  replacement generation, and #97's newer town service sidesteps #50's problem with a separate A2
  path rather than resolving it. Don't cite #50 as done.

### Integration and evidence boundaries

Generated height maps and generated slot programs currently reach the town API as **meshes**,
without an editable operation list. Recovered programs can replay through `EditableBuilding`,
but recovery from GT is not autonomous generation. Syntax validation, program finalization, and
compiled geometric containment are separate checks; the current coordinated commit calls the
program check, not the containment check. Mixed ordered add/subtract programs are executable but
are rejected by the finalize helper's commutativity requirement. These unresolved integration
tensions belong in #2/#179/#180; they are not evidence of an end-to-end valid editing product.

The historical pinned-714 set is a fixed regression control for BuildingWorld (#162/#177).
The new proof benchmark (#153/#181) needs a separately versioned region/tile-stratified split.
Those are distinct datasets and must not silently overwrite one another.

## Reading the numbers

Every massing result on this project is quoted in the same handful of measures. This is what each
one means, which direction is good, and what to compare it against. ⚠️ **No number here is
meaningful alone** — #126 exists because a single figure hid the fact that an arm had done nothing,
and #127 exists because three scalars could not tell a mound from a roof.

**The surplus pair — `missing` and `extra`.** Both are fractions of the real building's volume.
`missing` is GT the arm failed to fill: it **cut into the building**. `extra` is volume the arm
added outside GT: **surplus it failed to carve away**. Lower is better for both. ⚠️ They are **not
symmetric in consequence** — surplus is a building that looks unfinished, `missing` is a building
with a trench through it, and a plane slightly too steep is charged the whole trench while one
slightly too shallow only leaves surplus.

**`vs_input`** — IoU against the blockout the arm started from. **1.0 means it did nothing.** An arm
at 0.99 has not been measured as a generator however good its other numbers look (#75: a model
scored 3D IoU 0.857 while being 99.9% its own input). Lower means it acted. The guard is < 0.98.

**`collapse_rate`** — the fraction of buildings whose `missing` ≥ 0.15, i.e. that were eaten rather
than carved. Lower is better, and the bar is **1-NN retrieval's 0.1582**: a generator that destroys
more buildings than naive retrieval is not servable whatever else it scores.

**`dl_ops` and `dl_planar_fraction` — the FORM pair.** #10's `Layer`/`Ramp`/`CutRoof` fitter is run
on the arm's *own* surface and asked how many operations explain it. `dl_ops` is that count (lower =
simpler); `dl_planar_fraction` is the share of them that are **planes** rather than flat terraces
(higher = more roof-like). 🔑 **Read them together or not at all.** A real building is 2.0 ops at
0.50 planar; a mound is many `Layer`s at 0.00 planar; an arm reaching 3.0 ops at 0.00 planar has
simplified without becoming architecture. ⚠️ The pair is **not carve-aware by design** — a bare
envelope scores 1 op, correctly — so it is always read beside `extra`.

**3D IoU** — demoted to a diagnostic by #126 and printed to the right of the bar. A blockout that
over-fills by 22% and a generator that ate 22% of the building land on the same IoU while wanting
opposite responses.

**Programme-arm numbers.** `slots_used` is how many of its K typed regions the arm actually uses
(the label uses 3.06); an arm at 1.0 is drawing one region and cannot express a gable whatever its
planes do. `used_slots_typed_ramp` is the share of those typed as pitched. ⚠️ **`realised_rise` has
two meanings and both are published**: `..._all_slots_voxels` covers every used slot, so a `Layer` —
flat *by definition* — drags it down as soon as an arm uses more slots; `..._ramp_typed_voxels` is
the one that says a pitch was actually drawn. #6 typed 46% of its slots `Ramp` and drew them with a
median 0.00-voxel rise, which is why "it predicted a ramp" is never evidence that it drew one.

**Vertex-budget numbers (#131).** `verts` and `tokens` are the DSL cost of a program — 98.5% of it
is polygons, not operations. `contained` is the fraction of regions that gained no cell, so #10's
containment guarantee still holds. `spike` is the worst column's surplus in voxels and `spiked` the
fraction of buildings with one over `s*` = 3 — ⚠️ that pair is where the median `extra` lies to you:
a budget can sit inside the allowance while 90% of buildings grow a visible fin.

**The bar itself** is machine-checked in `verdict()`, never in prose, and has three parts. **PASS**
is what the arm must achieve. **GUARD** is what it may not break on the way (collapse and
`vs_input`). **KILL** is a pre-registered clause that answers the ticket "no" — it exists so a
disappointing result cannot be re-narrated as partial success.

**Reference values on the pinned 411 carve-needing buildings**, quote new results against these:

| | `extra` | collapse | ops | planar |
|---|---|---|---|---|
| the real building | — | — | **2.0** | **0.50** |
| compiled label *(sees GT — the ceiling)* | 0.0035 | 0.0000 | 2.0 | 0.50 |
| blockout *(doing nothing)* | 0.2308 | 0.0000 | 0.0 | 0.00 |
| 1-NN retrieval *(the guard)* | 0.1031 | **0.1582** | 2.0 | 0.17 |
| #127 CE+median *(served today)* | **0.0603** | 0.0268 | 6.0 | 0.20 |


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
The coarse architectural solid above the detail scale s*: the main volume, wings, and roof form.
It is the result of massing decisions, independent of whether those decisions are expressed through
a latent transform, a height map, or a semantic architectural edit program.
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
