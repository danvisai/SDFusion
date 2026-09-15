# GenerativeTowns

Generate building massing from footprint polygons and height, with an architectural editing and
detail-composition research stack built on [SDFusion](https://github.com/yccyenchicheng/SDFusion).
The intended building representation is a symbolic recipe; the newer town-generation service
currently returns meshes and has not yet connected generated programs to recipe editing.

**Start with [the reconciled project state](docs/PROJECT_STATE.md)** for what is implemented,
measured, served, and still open. [CONTEXT.md](CONTEXT.md) records the thesis and vocabulary;
[INTEGRATION_STATE.md](docs/INTEGRATION_STATE.md) maps the actual code boundaries.
Original upstream documentation is preserved in [README_UPSTREAM.md](README_UPSTREAM.md).

This checkout's active work is on `massing-solid-gate-retrain`. It contains both research and
serving code; do not infer its contents from the historical `main`/`upstream-training` split.

## The three serving surfaces

| Surface | Entry point | Current behavior |
|---|---|---|
| Footprint-town editor | `scripts.server.town_generate_service:app`, port 8767 | Draw/import footprints, set height, stream per-building meshes; compare alternatives at `/arms`. Default generator is A2. |
| Image footprint extraction | `scripts.server.footprint_extract_service:app`, port 8766 | Classical image processing used by the town editor. Separate process, no generator. |
| Original recipe/sculpt demo | `scripts.server.inference_service:app`, port 8099 | Recipe generation, SDEdit, detail, appearance and export APIs; `/sculpt.html` opens the sculptor. The index page's single-building, selected-building and export panels are hidden by the earlier demo-cull decision. |

The town editor uses the extraction and generation ports above (overridable with its `extract`
and `gen` query parameters). The original service can also serve `/town.html`, but that page
still calls the separate town/extraction services. The old index/sculpt local-storage bridge
does not establish an editable round trip for the new town generator.

## Run the footprint-town demo

Set up the environment and model/data prerequisites using [REPRODUCING.md](REPRODUCING.md),
noting its documented corpus-rebuild gap. In separate terminals:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python -m uvicorn \
    scripts.server.footprint_extract_service:app --host 0.0.0.0 --port 8766

env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python -m uvicorn \
    scripts.server.town_generate_service:app --host 0.0.0.0 --port 8767
```

Open `http://localhost:8767/`, or `http://localhost:8767/arms` for model/decode comparisons.
For remote use, forward both ports.

Startup currently loads Dora and `weights/massing-vecset/vecset_v4_surf.pth` unconditionally.
Height-map inference itself needs no codec, but **the service does not yet support a
height-map-only startup**. Optional height-map checkpoints resolve first from
`weights/massing-heightmap/`, then `outputs/height_map_generator/`; retrieval additionally
needs the height-field cache. `/health` reports available arms.

The seven supported arms are `envelope`, `heightmap_mode`, `heightmap_median`,
`heightmap_slope`, `heightmap_program`, `retrieval`, and `a2`.
Mode and median share one checkpoint and differ only in decoding.
The measured `fit_decode` fusion is **not a served arm**.
Height-map generation is deterministic at the scored setting; `roof_variation` is an
unvalidated demonstration control, not a measured diversity result.

## Run the original demo

Serving weights are documented under
[transfer/huggingface](transfer/huggingface/README.md); the original demo uses the
`demo-serving/` bundle, separate from the vecset checkpoints.

```bash
./scripts/server/run_web_demo.sh 8099
```

Open `http://localhost:8099/` or `http://localhost:8099/sculpt.html`.
The original recipe, planner/refiner/composer, and snap-prior weights differ from the newer
town generator's weights. Texture/render features load additional appearance models.
API availability is not a claim that every feature is currently visible or freshly validated.

## Research and architecture

Learned models choose massing or architectural decisions; deterministic SDF/CSG and retrieval
realize them. The thesis separates massing transformation (C1) from detail composition (C2).

Current massing work uses a small footprint-conditioned height-map generator and a
`Layer`/`Ramp`/`CutRoof` program fitter. The latest committed work (2026-09-06) concerns
BuildingWorld ingestion policy, not a completed new training run. Roof form and generated-program
integration remain open. Historical vecset alignment/convergence investigations are complete;
C2's comparative evidence effort is inactive.

| Code | Responsibility |
|---|---|
| `scripts/server/` | The separate APIs, web pages, recipe/refinement and appearance adapters |
| `scene/sdf_edit.py`, `scene/sdf_primitives.py` | Editable operations, SDF realization and validity helpers |
| `models/shape_codec.py`, `models/networks/vecset_*.py` | Codec adapters and A2 token-set projection |
| `scripts/train_vecset.py` | A2 training on cached real/blockout latents |
| `scripts/foundations/train_height_map_generator.py` | Height-map/program models, training, decoding and experimental scorecards |
| `scripts/foundations/recover_massing_programs.py` | Program fitting, replay and block coordination |
| `scripts/foundations/eval_massing_arms.py` | Shared massing measurements and pinned evaluation |
| `docs/wayfinding/`, `execution/artifacts/` | Dated decisions/results and numerical evidence |
| `legacy/`, dated handovers, `tickets.md` | Historical material, not the current queue |

## Verification

CPU/headless checks cover geometry, program replay, validity helpers and model contracts:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python -m unittest \
    scene.test_sdf_edit scripts.foundations.test_recover_massing_programs \
    scripts.foundations.test_train_height_map_generator
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
    scripts/server/test_town_generate.py --geometry-only
```

The older demo's `test_branches.py` and `test_sculpt_flows.py` require its live service and
weights. The town test without `--geometry-only` loads real A2/Dora models.
Passing headless tests does not prove trained-model quality, visual acceptance, or live GPU serving.

## Documentation and work tracking

Use [PROJECT_STATE.md](docs/PROJECT_STATE.md) → [INTEGRATION_STATE.md](docs/INTEGRATION_STATE.md)
→ the relevant dated result and GitHub issue. [AGENTS.md](AGENTS.md) links tracker conventions.
A closed research ticket may record a negative finding; an implemented backend may still lack
a serving caller. Check each claim at that level.
