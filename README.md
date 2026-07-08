# GenerativeTowns

Generate an editable 3D town from a footprint image, then sculpt, detail, age, texture,
and photoreal-render individual buildings — every step reversible, every building a
compact symbolic recipe rather than a frozen mesh.

**Maintainer:** Danvi Simhadri (danvisai03@gmail.com).
Built on top of [SDFusion](https://github.com/yccyenchicheng/SDFusion) — original README
preserved at `README_UPSTREAM.md`; the training/research surface lives on the
`upstream-training` branch, `main` is the demo/serving code. Agents: see
`docs/DEMO_BUILD_PLAN_2026-07-07.md` (current roadmap) and `docs/HANDOFF_*.md`
(session chronicles).

## The design doctrine

**Every *decision* about what a building looks like is made by a learned model; every
*realization* into geometry is deterministic procedure.** That split is what keeps
everything editable: models choose, procedures build, and any choice can be re-rolled
without destroying the rest of the building.

| learned component | checkpoint | decides |
|---|---|---|
| recipe-parameter diffusion | `outputs/recipe_param_diffusion_b6` | massing: proportions/roof/wings from footprint + class + style |
| Stage 3a SDF diffusion (947M) + VQVAE | `logs_building/continue-stage3a-xcultural-warmstart-ft*/ckpt` | "AI massing" from real NL/DE/JP buildings; the localized snap (`/snap_sdf`) |
| PartLayoutPlannerV2 | `outputs/part_layout_planner_v2` | window/door/balcony layouts, ornament slots, Make-it-architecture typing |
| CoherentPartRefiner | `outputs/part_set_refiner` | integrating a sculpted mass into the building's part set |
| PartComposer | `outputs/part_composer` | the always-on statistical facade detail |
| SDXL + depth/canny/scribble ControlNets + IP-Adapter | HF cache (auto-downloads) | texture bake, photoreal renders, sketch-relief art |
| Depth Anything V2 | HF cache | relief height from generated art |

Not learned (by design): footprint extraction (classical CV), recipe→SDF realization,
CSG sculpting, weathering (pure noise functions), ornament retrieval/fit (real heritage
scans; placement IS learned), marching cubes / UV / PBR heuristics, and the sketch-relief
rectification + fusion math.

## Quickstart

Requirements: a CUDA GPU (~24 GB VRAM comfortable), ~10 GB disk for checkpoints,
~46 GB for the Hugging Face model cache (downloaded on first texture/render call).

```bash
python -m venv venv && venv/bin/pip install -r requirements.txt
# checkpoints + data must sit at the paths in the table above (the demo bundle ships them)
CUDA_VISIBLE_DEVICES=0 ./venv/bin/python -m uvicorn scripts.server.inference_service:app \
    --host 0.0.0.0 --port 8099 --log-level warning
```

Then open:

- **`http://localhost:8099/`** — the town page: drop an OSM map / footprint-mask image
  (samples under `scripts/server/web/samples/`), get a town; select a building to
  restyle / re-height / re-roll it, age it with the weathering slider, mount a heritage
  ornament, texture-bake or photoreal-render the whole town, export glTF for Unreal.
- **`http://localhost:8099/sculpt.html`** — the SDF Sculptor: raymarched live editing.
  Place primitives and let *Make it architecture* interpret them (tower/wing/dormer with
  windows and roofs), ask the planner for *AI details*, carve, bake textures, or sketch a
  rough shape on a wall and have it sculpted into real bas-relief geometry (*Sketch
  relief*; reliefs stack, and a prompt like "a lion head" steers the motif).

Round trips between the two pages are lossless (a building opened in the Sculptor comes
back to the town with its edits as first-class state).

## Tests

Both suites run against a live server and assert metrics per flow:

```bash
python scripts/server/test_branches.py      # 13 branch tests (API surface)
python scripts/server/test_sculpt_flows.py  # 19 UI-flow tests (incl. relief, textures)
```

## Repo layout

- `scripts/server/` — FastAPI service (`inference_service.py`), refiner/recipe engines,
  the two web pages (`web/`), both gate suites.
- `scripts/appearance/` — texture bake, neural (SDXL) rendering.
- `scene/` — SDF primitives/CSG, composer detail, weathering, mesh cleanup.
- `models/` — Stage 3a diffusion, VQVAE, planner/refiner/composer networks.
- `docs/` — dated handoffs and build plans.
- Training code, dataset tooling, original SDFusion research surface: `upstream-training`.
