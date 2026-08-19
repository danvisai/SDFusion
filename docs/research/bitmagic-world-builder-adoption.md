# Bitmagic “World Builder” / World-Forger: adoption review for GenerativeTowns

**Sources accessed:** 2026-08-17

**Scope:** First-party Bitmagic pages and founder/company announcements, compared with this repository.
Bitmagic capability and performance statements below are vendor claims, not independent benchmarks.

## Executive verdict

Do **not** make Bitmagic a dependency yet. I found no public first-party API, SDK, source repository,
self-hosting guide, or documented general project import/export contract. Creator access is currently
limited to approved Game Lab members or invite-code holders.
[Creator access](https://bitmagic.ai/creator/), [Game Lab](https://bitmagic.ai/lab/)

Use Bitmagic as a **product/UX benchmark**. Its best lesson is a short loop—generate, experience,
target an edit, regenerate, share—not one-shot world generation. Bitmagic's own evidence includes a
racing game made with 91 prompts and a school cohort that used 2,532 prompts to ship eight games.
[91-prompt account](https://www.linkedin.com/posts/janipenttinen_i-didnt-participate-in-supercell-ai-game-activity-7426756361444855808-tOK7),
[school case study](https://bitmagic.ai/news/cleverlike-bitmagic/)

Recommended adoption order:

1. Separate **Evidence** and **Showcase** views; make the latter visually polished without hiding the
   former's geometry and metrics.
2. Add **prompt-to-typed-recipe-patch**, targeted by building/operation ID, with diff preview and undo.
3. Extend the streamed town build through explicit envelope, massing, detail, appearance, and export
   stages, with a small candidate tray.
4. Optimize for **urban/architectural coherence** across a footprint set, not only independent solids.
5. Export a **semantic town bundle**: named GLB nodes plus recipes, footprint/operation IDs,
   entrance/storey/site-graph metadata, cameras, and provenance.
6. Keep Gaussian splats optional and outside editable buildings.

These changes preserve the accepted project contract: SDEdit is the C1 transform, detail is composed
or retrieved under C2, and deterministic realization keeps recipe decisions reversible.
[`CONTEXT.md`](../../CONTEXT.md), [`ADR 0003`](../adr/0003-two-claim-thesis.md)

## Naming and current availability

### Verified

- The current official names I found are **Bitmagic Creator**, **Bitmagic V3**, and **Game Lab**, not a
  standalone product named “Bitmagic World Builder.”
  [Product history](https://bitmagic.ai/about/), [Creator](https://bitmagic.ai/creator/),
  [Game Lab](https://bitmagic.ai/lab/)
- A recent CEO announcement calls the level-generation feature **“world-forger.”** Its stated scope is
  prompt → complete game level, with examples including a race track, ski mountain, platformer, and
  open-world city. Depending on the game, it may include NPC spawn points, puzzles/challenges, and a
  preprocessed navmesh. It emits polygonal GLB editable in Blender, after which a Bitmagic agent builds
  gameplay. The post claims 3–5 minutes for showcased levels.
  [World-forger announcement](https://www.linkedin.com/posts/janipenttinen_heres-a-glimpse-of-what-the-new-bitmagic-activity-7477595150723104768-vehY)
- Bitmagic says V3 launched in September 2025 on a custom WebGL engine built with TypeScript and
  Three.js, replacing its earlier Unity web build so AI could change rules, environments, mechanics,
  and assets during creation. [Official product history](https://bitmagic.ai/about/)
- Game Lab says text/image prompts describe rules, environment, and objective; the system builds code,
  physics, animations, and assets, and also exposes a standard 3D scene editor.
  [Game Lab](https://bitmagic.ai/lab/)
- Bitmagic markets sentence-to-playable-game turnaround in minutes, hosted browser publication, and
  share links. [Current home page](https://bitmagic.ai/), [Game Lab](https://bitmagic.ai/lab/)

### Observation / unknowns

The user's “World Builder” most likely means Bitmagic's general creation workflow or the newer
**world-forger**. No examined official page establishes world-forger as a separately callable product,
states that every Creator account has it, or documents its configuration, API, quotas, pricing,
version pinning, deterministic replay, or full-project export.

## Differentiating patterns worth adapting

| Verified Bitmagic pattern | SDFusion translation | Why it matters |
|---|---|---|
| Prompt → play → prompt/manual edit → play/share ([source](https://janipenttinen.com/post/yesterday-several-people-shared-a-post-with-me-where-ollie-scheers-showed-result)) | Prompt → typed `RecipePatch` → diff preview → apply/undo → rebuild affected stages | Conversational iteration without giving up precision or edit locality |
| Prompting plus a conventional scene editor ([source](https://bitmagic.ai/lab/)) | Keep direct footprint/primitive/gizmo editing; make manual and prompted edits share one operation history | Users can repair a miss locally instead of fighting regeneration |
| Historical visible object IDs and object-targeted commands; exact V3 command support is unverified ([source](https://www.bitmagic.ai/blog/closed-alpha-tips-n-tricks/)) | Show stable building and operation IDs; target one, several, or an edit selection | Prevents accidental town-wide changes |
| World-forger adds gameplay metadata and exports GLB ([source](https://www.linkedin.com/posts/janipenttinen_heres-a-glimpse-of-what-the-new-bitmagic-activity-7477595150723104768-vehY)) | Named GLB nodes plus recipe, footprint, entrance/storey/operation metadata and site graph | Makes output useful to Blender/Unreal/GIS instead of only visible |
| World-forger is designed around the level's use, not only scenery ([source](https://www.linkedin.com/posts/janipenttinen_heres-a-glimpse-of-what-the-new-bitmagic-activity-7477595150723104768-vehY)) | Design for road-facing entrances, access, height/roof rhythm, setbacks, courtyards, and selected-block coordination | Makes a coherent town rather than a collection of unrelated footprint solids |
| Prompt-controlled environment, lighting, and time-of-day precedents ([Mini source](https://company.bitmagic.ai/blog/create-with-bitmagic-mini/), [V3 editor source](https://janipenttinen.com/post/yesterday-several-people-shared-a-post-with-me-where-ollie-scheers-showed-result)) | A scene-level visual director for sun, sky, fog, ground/material palette, greenery, and cameras | Large perceptual gain without retraining massing |
| Hosted result and share link ([source](https://bitmagic.ai/)) | Immutable recipe snapshot URL with cached GLB, camera/mode, seeds, and checkpoint hashes | Faster, reproducible supervisor/reviewer feedback |

The strongest inference is that “designed for gameplay” should map to **designed for urban and
architectural coherence**. NPCs and puzzles are outside this thesis; their useful analogue is semantic
context and downstream-ready state.

## What SDFusion already has

- A symbolic recipe stores footprint, parameters, operations, style reference, weather seed, and
  element/ornament decisions; geometry is deterministically derived.
  [`CONTEXT.md`](../../CONTEXT.md)
- The preview/finalization contract already asks for immediate deterministic feedback, learned
  completion within seconds, and asynchronous validation/final extraction.
  [`CONTEXT.md`](../../CONTEXT.md)
- The repository has capabilities/endpoints and older integrated-client code for footprint-image town
  creation, per-building re-style/re-height/re-roll, weathering, ornament, style reference, photoreal
  render, texture bake, glTF export, and sculptor round trips. The current supervisor-facing client
  hides some of those editor/export sections rather than exposing the whole inventory.
  [`feature inventory`](../wayfinding/clean-supervisor-demo/feature-inventory.md),
  [`integrated town client`](../../scripts/server/web/index.html)
- The sculptor already has add/subtract primitives, gizmos, undo, “Make it architecture,” re-roll,
  detail preview, GLB bake, style images, prompted appearance, and sketch relief.
  [`sculptor`](../../scripts/server/web/sculpt.html)
- The newer A2 town client already imports/draws footprint sets, shows the envelope immediately,
  streams buildings as NDJSON, estimates progress, permits cancellation, keeps completed results,
  rejects stale in-flight results after an edit, and exposes `vs input`.
  [`A2 town client`](../../scripts/server/web/town.html)

The A2 client is intentionally coded as a **dark, plain shaded massing view with no textures**; it uses
a flat-shaded material and lacks the enabled shadow/tone-mapping setup already present in the integrated
client. [`A2 renderer`](../../scripts/server/web/town.html),
[`integrated renderer`](../../scripts/server/web/index.html)

That neutral view is valuable evidence. The project record says dense-grid massing is lumpy/wavy, and
a prior demo comparison found that distant town views can hide defects exposed in close-ups.
[`CONTEXT.md status`](../../CONTEXT.md),
[`demo comparison`](../wayfinding/clean-structural-generation/51-prototype-result.md)
Presentation should therefore improve the demo **without replacing** neutral validation views.

## Prioritized adoption plan

Effort ranges are rough repository-specific engineering inferences.

### P0 — Evidence / Showcase toggle (1–3 days)

- Preserve the current neutral material, grid, stable research camera, metrics, and failure labels in
  Evidence mode.
- Reuse the integrated client's existing soft shadows, ACES tone mapping, warm sky/fog/ground, and
  selected-building treatment in Showcase mode.
- Add plan, research-oblique, and street/hero camera presets plus Save PNG with recipe ID, checkpoint,
  seed, and view mode in a sidecar/caption.
- Add clear-day, overcast, and golden-hour presets over identical geometry; never present them as new
  generated samples.

This is the fastest credible visual improvement because most renderer ingredients already exist in
[`index.html`](../../scripts/server/web/index.html) and can be reused in
[`town.html`](../../scripts/server/web/town.html).

### P0 — stable target identity and history (2–4 days)

- Show building ID, seed, height, `vs input`, and generation/failure state in the inspector.
- Show a stable operation stack with ID, semantic type, add/subtract mode, and provenance.
- Make manual changes append reversible recipe patches; provide multi-step undo/redo across town and
  sculptor.

This is a prerequisite for safe prompting and directly implements the existing editable/reversible
contract. [`CONTEXT.md`](../../CONTEXT.md)

### P1 — prompt-to-recipe patch (1–2 weeks)

Start with prompt-like preset chips—“lower selected block,” “coordinate roof family,” “weather the
selected buildings,” “reroll only facade decisions”—compiled to a closed patch schema. Show the exact
diff and preserved fields before apply.

Only then add an LLM as a parser from free text to that schema. Required guardrails: selection
confirmation, schema/constraint validation, typed failures, preview, undo, provenance, and **no code
execution**. This keeps natural language inside the symbolic recipe instead of creating a leak.
[`CONTEXT.md`](../../CONTEXT.md)

### P1 — progressive build and candidate tray (~1 week)

Extend the existing stream rather than replacing it:

1. show the footprint envelope immediately;
2. stream massing per footprint;
3. offer 2–3 candidates only for uncertain/failed cases, with transparent scores;
4. add deterministic detail and cheap materials asynchronously;
5. queue neural appearance only as an optional quality job;
6. invalidate only affected downstream stages after an edit.

Multiple candidates multiply denoising cost, so make them selective. The client already supplies
streaming/cancellation/stale-result primitives, and the audit already recommends transparent quality
reranking. [`town client`](../../scripts/server/web/town.html),
[`output audit`](../SDFUSION_OUTPUT_IMPROVEMENT_AUDIT_2026-07-13.md)

### P1 — urban coherence and semantic export (2–4 weeks)

Produce `town.glb`, `town.recipe.json`, `town.site-graph.json`, `town.provenance.json`, and fixed plan/
oblique/street previews. Give every GLB node a stable building ID and link it to footprint, recipe,
operations, class/style, height/storeys, entrance candidates with confidence, checkpoint/seed, and
failure state. Mark unknown semantics unknown rather than inventing them.

The site graph should capture frontage, roads, adjacency, and selection relations so later decisions
can coordinate access, roof/height rhythm, setbacks, and courtyards. The repository already extracts
OSM roads and exports town geometry; the new work is a stable manifest, inference/constraint logic,
and round-trip validation. [`OSM extractor`](../../scene/extract_osm.py),
[`README export`](../../README.md), [`footprint-set domain model`](../../CONTEXT.md)

### P2 — shareable snapshots; P3 — scale/runtime experiments

Persist immutable recipe snapshots and cached derived artifacts behind a stable review URL. Pin git
SHA, checkpoint hashes, seeds, renderer preset, and camera; never serialize only a frozen mesh.

For much larger towns, investigate tiled/multires derived geometry, spatial indexing, cached LODs, and
view-based streaming. Bitmagic's CEO reports a multi-km² voxel world loading in 2–3 seconds at 60 fps,
but does not disclose the implementation; the proposed mechanisms are inference, not copied facts.
[Performance post](https://www.linkedin.com/posts/janipenttinen_one-of-the-most-exciting-things-in-life-is-activity-7461557037592723456-xQqV)
This is lower priority than current visual quality and interaction.

## Gaussian splatting: context only

Bitmagic says V3 supports splat scenes, but its CEO also identifies immature authoring/editing and
proxy physics as current limitations; the described workaround voxelizes splats and builds polygon
colliders. [Bitmagic GS note](https://janipenttinen.com/gaussian-splatting)

If evaluated here, use splats only for far-field streetscape, landscape, or a photoreal presentation
shell. Keep every editable building as a symbolic recipe. Replacing buildings with splats would create
a recipe-closure leak. [`CONTEXT.md`](../../CONTEXT.md)

## What not to adopt

1. **Unconstrained text-to-code in the geometry path.** Bitmagic built a general AI-native game engine;
   SDFusion has a narrower, testable domain model. Arbitrary code weakens determinism, provenance,
   safety, and recipe closure. [Bitmagic architecture](https://bitmagic.ai/about/),
   [`CONTEXT.md`](../../CONTEXT.md)
2. **A self-modifying production engine.** Capture failed requests and corrective work as telemetry,
   then convert repeated patterns into GitHub issues/tests and reviewed changes.
   [Bitmagic recursive-engine announcement](https://janipenttinen.com/post/bitmagic-s-recursive-ai-engine-improves-itself),
   [`issue policy`](../agents/issue-tracker.md)
3. **Splat-based editable buildings.** Bitmagic's own note identifies the editing gap, and this would
   abandon the project's strongest representation. [GS note](https://janipenttinen.com/gaussian-splatting)
4. **Gameplay scope.** NPCs, puzzles, combat, and multiplayer demonstrate Bitmagic's breadth but are
   not requirements for this architecture thesis.
5. **One-shot quality claims.** Bitmagic's own usage evidence supports human-led iteration.
   [91-prompt account](https://www.linkedin.com/posts/janipenttinen_i-didnt-participate-in-supercell-ai-game-activity-7426756361444855808-tOK7),
   [school case study](https://bitmagic.ai/news/cleverlike-bitmagic/)

## Licensing, privacy, and direct-use risks

### Verified

- Current terms say creators own and may monetize their games, while granting Bitmagic a license to
  host, display, distribute, and showcase their content; users warrant they hold necessary rights.
  [Terms](https://bitmagic.ai/terms-of-service/)
- The terms provide the service as-is, disclaim uninterrupted access, and permit modification,
  suspension, or discontinuation. [Terms](https://bitmagic.ai/terms-of-service/)
- The privacy policy says Bitmagic collects account, usage, technical, and user-generated content for
  operation, enhancement, security, and compliance. [Privacy policy](https://bitmagic.ai/privacy-policy/)
- Bitmagic's case study records requests for custom asset imports, better large-project performance,
  more direct animation control, and editor basics such as copy/paste, showing an evolving product.
  [Case study](https://bitmagic.ai/news/cleverlike-bitmagic/)

### Confirm in writing before uploading project material

- Does game ownership cover every standalone GLB, texture, splat, code file, and intermediate outside
  the hosted game?
- What are the training/source-asset provenance, third-party licenses, infringement handling, and
  indemnification terms?
- Are private prompts, reference images, imported geometry, or outputs used for model training? The
  policy's broad “enhancement” purpose does not answer this.
- What are retention, subprocessors/model providers, enterprise opt-out, confidentiality, deletion,
  export-completeness, versioning, API, pricing, and service-level terms?

Do not upload unreleased data, checkpoints, unpublished results, private site data, or licensed
references until those questions are answered.

## Sensible direct trial

If desired, apply to Game Lab only for a benchmark: use one non-confidential representative town,
record every prompt/manual edit/time/credit/failure/export, and test whether Bitmagic (a) improves
context/presentation beyond Showcase mode, (b) preserves exact footprints and building identity
through edits, and (c) exports a legally and technically usable result. Score architectural
plausibility, consistency, edit locality, browser performance, export completeness, and reproducible
rebuilds. Keep it out of thesis evidence unless versions and methods can be controlled.

## Bottom line

Bitmagic should influence **how the demo is experienced**, not replace its scientific representation.
The immediate win is a polished but separate Showcase view. The durable product win is stable target
identity plus prompt-to-recipe patches, followed by block-level architectural coherence and semantic
export. Direct integration, splat-based buildings, general code-generating agents, and large-world
runtime work should wait.
