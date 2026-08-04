# Deployment Plan — Option B+ → City-Scale Production

**Last updated:** 2026-06-02
**Owner:** Danvi Simhadri
**Status:** specification only — implementation gated on Phase B+.6 (recipe-param diffusion training)
**Related:** `README.md`, `memory/project_option_b_plus_phase1.md`, `memory/project_option_b_plus_phase7.md`

---

## 1. Deployment goal

End target: **generate large cities at production scale** from OSM data with interactive editing. The user (a 3D artist, urban planner, game designer) opens an OSM tile, the system populates it with diverse generated buildings, and the user can click any building to refine it via parameter sliders or drag-drop primitive edits.

Concrete scale targets:
- **City-scale**: 1,000–100,000 buildings in a single scene
- **Interactive edit latency**: < 200 ms for slider tweaks (no model call); ~1–2 s for AI-refinement requests
- **Asset quality**: per-building mesh ≥ Path A quality (Hunyuan-refined), per-scene framerate ≥ 30 fps on RTX 4070+ class hardware

---

## 2. Host environments evaluated

| Criterion | **Unreal Engine** | **Unity** | **Blender** |
|---|---|---|---|
| City-scale geometry | ✅ Nanite + Lumen, 100k+ buildings (City Sample baseline) | ⚠ HDRP scales to ~10k | ❌ Eevee struggles past few thousand |
| OSM tooling | ✅ Cesium-for-Unreal, StreetMap, MapBox SDK | ✅ BuildR2, MapBox, OSMSharp | ⚠ Blender-OSM (import only) |
| 3DGS native support | ✅ Cesium + third-party (KIRI, etc.) | ⚠ Community packages only | ⚠ KIRI Engine plugin |
| Plugin language | C++ (perf) + Python (editor scripting) | C# + Python via socket | Python (native) |
| Embedding our torch model | ❌ Run via REST service (recommended) | ❌ Run via REST service | ✅ Direct import (we already use torch) |
| Industry fit for smart-city | ✅ standard for digital-twin research + production | ⚠ indie / educational | ❌ visualization tool, not realtime |
| Plugin distribution | Unreal Marketplace + GitHub | Unity Asset Store + GitHub | Blender Add-on Registry + GitHub |
| Commercial license | Royalty after $1M gross | Subscription tiered | Free (GPL) |
| Effort to first usable demo | 6–10 weeks (C++ + asset pipeline) | 4–6 weeks (C#) | 2–3 weeks (Python) |

**Decision: Unreal is the production target.** It is the only host that natively handles city-scale geometry and is what production digital-twin / smart-city teams actually use.

---

## 3. Architecture overview

```
┌────────────────────────────────────────────────────────────────┐
│  HOST ENGINE (Unreal in prod; Blender for prototyping)         │
│                                                                │
│  Procedural Content Generation node OR editor widget:          │
│    "Generate City from OSM Tile"                               │
│             │                                                  │
│             │ HTTPS / gRPC                                     │
│             ▼                                                  │
└─────────────┼──────────────────────────────────────────────────┘
              │
┌─────────────▼──────────────────────────────────────────────────┐
│  INFERENCE SERVICE (Python, runs on GPU box)                   │
│                                                                │
│  FastAPI app, stateless:                                       │
│    POST /generate_tile                                         │
│       in:  { osm_geojson, default_style?, seed? }              │
│       out: [                                                   │
│              { footprint, class, height, style,                │
│                recipe_params,            ← editable params     │
│                mesh_glb_b64,             ← rendered mesh       │
│                gs_ply_b64?               ← optional 3DGS       │
│              }, ...                                            │
│            ]                                                   │
│                                                                │
│    POST /regenerate_building                                   │
│       in:  { osm_polygon, class, height, style, seed }         │
│       out: { recipe_params, mesh_glb_b64, gs_ply_b64? }        │
│                                                                │
│    POST /refine_with_edit                                      │
│       in:  { base_params, edit_primitives:[box,cyl,...],       │
│             style, footprint }                                 │
│       out: { recipe_params_refined, mesh_glb_b64 }             │
│                                                                │
│    POST /params_to_mesh   (FAST — no model call)               │
│       in:  { recipe_params, style, footprint, height }         │
│       out: { mesh_glb_b64 }                                    │
│                                                                │
│  Models loaded once at startup:                                │
│    - Recipe-param diffusion (Phase B+.6 weights)               │
│    - Style classifier (Phase 1b.2 — optional CLIP-based)       │
│    - DiffRecipe modules (deterministic, no weights)            │
│    - VQVAE v1 (legacy; only needed if Path D is also served)   │
└────────────────────────────────────────────────────────────────┘
```

### Why service-first

1. **Decouples model from host engine.** Same service feeds Blender prototype, Unity build, Unreal production. No torch in C++.
2. **Latency budget control.** Service handles GPU work; engine handles the 16ms render frame. Slider edits never hit the network — they're local.
3. **Independent scaling.** Generate-tile is GPU-bound (~1-2 s × N buildings); place-mesh is CPU-bound. They scale on different machines if needed.
4. **API stability.** Engine plugins change; the JSON contract above can stay stable across hosts.

### Stateful vs stateless tradeoffs

- **Service is stateless.** Each request carries all conditioning and (for refinement) the current `recipe_params`. No session state on the GPU box.
- **Host engine holds the state.** Each building actor stores its `recipe_params` as a property. Sliders mutate that property locally and call the fast `params_to_mesh` endpoint (10-50 ms). Only "Refine with AI" calls the slow `refine_with_edit` endpoint (~1 s).

---

## 4. Staged delivery plan

### Stage A — Inference service  🟡 SCAFFOLDED 2026-06-03

`scripts/server/recipe_inference.py` (engine) + `scripts/server/inference_service.py` (FastAPI)

Status (validated via `fastapi.testclient`, no live server yet):
- ✅ `RecipeInferenceEngine` loads the B+.6 ckpt + scalers + DiffRecipe registry once.
  - `sample_params` (generative: conditioning → diffusion → raw params)
  - `params_to_mesh` (fast, no model: recipe → world-meter grid → marching cubes → glb) — **~20–150 ms**, meets the <200 ms slider budget
  - `generate_building` (end-to-end → glb + world `position_xz`)
- ✅ FastAPI endpoints: `GET /health`, `POST /params_to_mesh`, `POST /regenerate_building`, `POST /generate_tile`, **`POST /refine_with_edit`** (cleanup / re-style a sculpt — `scripts/server/refine.py`; modes: `fast` snap = trained head ~115 ms; `displacement` = base recipe + learned residual field that PRESERVES sculpted detail, IoU→edit 0.95 vs snap 0.59, and re-styles while keeping detail 0.88, ~3 s. validated over live HTTP). glb base64; footprints/heights in meters; sampling default `guidance=2.0, eta=1.0` with a `diversity` knob. Bad style → HTTP 400.
- ✅ Round-trip verified: sampled params reproduce the mesh; glb decodes + loads in trimesh; batch tile of mixed styles (modern/mediterranean/victorian) generates.
- ✅ **Live `uvicorn` server validated over real HTTP** (not just TestClient): starts ~6 s (loads B+.6 ckpt), `/health` + `/regenerate_building` (10760-vert modern, 388 KB glb) + `/generate_tile` all return; clean shutdown. `... -m uvicorn scripts.server.inference_service:app --port 8077`.
- ✅ **Full real-OSM→town pipeline validated** (`scripts/server/demo_osm_town.py`): 62-building Lafayette tile (osmnx/Overpass) → generated 3D town.

Still TODO for a production Stage A:
- `/refine_with_edit` (Stage 4 interactive sculpt) — not yet wired
- Real GeoJSON ingestion + CRS→meter projection (currently takes footprints as meter polygons)
- Docker container (CUDA base + our env) + a live `uvicorn` daemon on Gilbreth
- Batched diffusion sampling for large tiles (currently one building per call)

Run (dev): `... -m uvicorn scripts.server.inference_service:app --port 8000`

### Stage B — Blender add-on (~2-3 weeks, runs concurrent with Stage C planning)

`tools/blender_addon/`

Why: validates the editing UX before committing 8 weeks to Unreal C++. If sliders + drag-drop primitives don't feel right in Blender, we change them before sinking effort into the production host.

Tasks:
- Add-on with N-panel UI: OSM file picker, "Generate City" button
- HTTP client to inference service
- Per-building object with custom properties = `recipe_params`
- Slider UI in object properties panel
- Drag-drop primitive panel (palette of box/cylinder/etc.)
- "AI Refine" operator calling `/refine_with_edit`
- Local re-execution of DiffRecipe in Python on slider tick (fast, no service call)

Outputs are .blend files; can export to OBJ / glTF for downstream use.

### Stage C — Unreal plugin (~6-10 weeks after Stage A)

`tools/unreal_plugin/`

Tasks:
- C++ plugin module `GenerativeTownsPCG`
- PCG (Procedural Content Generation) graph node "GenerativeTownsBuildingPlacer"
  - Input: OSM polygon set + per-building conditioning
  - Output: spawned building actors with mesh component + recipe_params property
- Editor utility widget for OSM tile loading
- `ABuildingActor` UE class:
  - Stored `recipe_params` (FStructProperty)
  - Details panel exposes them as sliders
  - On slider change: call service `/params_to_mesh`, update mesh component
  - "Refine" button: call `/refine_with_edit`
- Drag-drop editor widget for primitive sculpting
- Cesium-for-Unreal integration for OSM tile loading + georeferencing
- glb importer wrapper (Unreal has native glTF support)
- Optional: 3DGS support via Cesium or a custom plugin

Performance considerations:
- Use Nanite-enabled meshes (auto-LOD for distant buildings)
- Lumen for global illumination at city scale
- World Partition for streaming large tiles
- Object pooling for repeated regeneration

### Stage D (optional) — Unity package (~3-4 weeks)

Only if there's demand for indie / educational users. Mirrors Stage C structure in C#.

---

## 5. Latency budget

| Operation | Budget | How |
|---|---|---|
| OSM tile load → city generation | ~30 s per 1000 buildings | Service `/generate_tile`, GPU-bound |
| Slider tweak on selected building | < 200 ms | Local `DiffRecipe.forward` call (no service) |
| Drag-drop primitive added | < 500 ms | Local primitive composition + DiffRecipe |
| "AI Refine" button | < 2 s | Service `/refine_with_edit` with Stage 4 model |
| Scene render | 60 fps target | Nanite + Lumen handles |
| Save / commit | instant | Local state mutation |

Implication: **slider edits must not hit the service.** The DiffRecipe forward pass is cheap (~10 ms on GPU, < 100 ms on CPU). The host engine ships with a minimal torch wheel just to run forward passes on edited parameters, OR we embed a small ONNX export of the recipe forward + marching cubes.

---

## 6. Model export strategy

For Stage B and C, the host plugins need a way to run `DiffRecipe(params) → mesh` locally without calling the service.

Options:
1. **Embed Python in plugin** — Blender does this natively; Unreal has PythonScriptPlugin. Easiest for prototyping.
2. **TorchScript export** — `torch.jit.script(DiffRecipe)` → loadable via libtorch in C++. ~50 MB binary.
3. **ONNX export** — exporter for each `DiffRecipeX` class → ONNX runtime in C++ / C#. Smallest binary, fastest.
4. **C++ port** — hand-port the recipe primitives to C++. Best perf, most maintenance burden.

**Recommendation:** ONNX for Stage C (Unreal). TorchScript for Stage B (Blender, easier). Embedded Python for first prototype.

---

## 7. Data flow at scale

For a 10,000-building city:

```
1. OSM tile loaded → 10k footprints extracted
2. POST /generate_tile (one batched request)
   - GPU server runs:
     - Diffusion sampling: ~30 s for 10k buildings (batched on A100)
     - DiffRecipe forward: ~30 s for 10k buildings
     - Marching cubes: ~5 s (CPU multithreaded)
   - Response: 10k glb files, total ~500 MB
3. Host engine receives, spawns actors with mesh + params
4. Nanite + Lumen handle render
5. User clicks building → ~10 ms local DiffRecipe call regenerates mesh
6. User adds chimney via drag → ~50 ms local primitive composition
7. User hits "Refine" → ~2 s server round-trip
```

Memory budget per building: ~50 KB recipe_params + ~50 KB low-poly mesh (Nanite-friendly) = ~500 MB for 10k buildings in scene memory.

---

## 8. Open questions to resolve before plugin work starts

1. **Will users want per-building edits or per-tile bulk edits?** Affects whether we expose params at building level (B+ direct) or at tile level (need a tile-aware model).
2. **Mesh vs 3DGS as default output?** Mesh integrates better with game engines; 3DGS gives photorealistic appearance. Probably ship both and let user toggle.
3. **Latency tolerance for first-time generation.** Is 30 s for 10k buildings acceptable, or do we need to precompute neighborhoods and stream?
4. **Service hosting model.** On user's local GPU? Cloud-hosted with rate limits? Both?
5. **Edit history / versioning.** Do we need undo / "save snapshot" / regenerate-from-snapshot? Probably yes — UE editor users expect it.

---

## 9. Dependencies & gating

```
Phase B+.7    (curating data, NOW)  ───┐
Phase B+.4    (synthetic data)         ├──► Phase B+.5/B+.6 (train the model)
                                       │              │
                                       │              ▼
                                       │      [model.pt ready]
                                       │              │
                                       ▼              ▼
                                  Stage A — Inference Service (~1 week)
                                              │
                                              ▼
                                  Stage B — Blender add-on (~2-3 weeks, iterative)
                                              │
                                              ▼     (use Blender prototype to validate UX)
                                  Stage C — Unreal plugin (~6-10 weeks)
                                              │
                                              ▼
                                  Stage D — Unity (optional, ~3-4 weeks)
```

Critical path: **B+.6 training (~2 weeks) → Stage A (~1 week) → Stage B (~3 weeks) → Stage C (~10 weeks)** = roughly **4 months** from today to first usable Unreal demo of a city. Could compress to ~3 months if Stage B's UX validation goes smoothly and we skip iteration.

---

## 10. Risks

| Risk | Mitigation |
|---|---|
| B+.6 model doesn't converge (mode collapse, too few real fits, etc.) | Fall back to Path E (Stage 3b GS lifter) as the inference target; service shape stays the same |
| Recipe library doesn't express 30-50% of buildings | Add HF-NeuS-style displacement layer (Option D) on top; same parameter format + extra delta field |
| Service latency too high for interactive use | Batch precomputation, edge caching, smaller diffusion model |
| Unreal plugin too complex | Ship Blender add-on only first; revisit Unreal after paper |
| Model file too large for plugin embed | Service-only for the diffusion; only DiffRecipe forward is embedded (small) |

---

## 11. What to do TODAY

Nothing in this plan executes until Phase B+.6 trains. For now:
1. Let B+.7 finish curating real-data fits (currently ~88% done)
2. Run B+.4 to log synthetic params (CPU job, ~6h)
3. Run B+.5 deterministic head sanity check (1-2 days)
4. Run B+.6 diffusion training (5-7 days)
5. Then this deployment plan activates.

This document is here so future agents (and your future self) don't have to re-derive the staging decision when the model is ready to ship.
