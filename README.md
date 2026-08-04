# GenerativeTowns

Generate an editable 3D town from a footprint image, then sculpt, detail, age, texture,
and photoreal-render individual buildings — every step reversible, every building a
compact symbolic recipe rather than a frozen mesh.

**Maintainer:** Danvi Simhadri (danvisai03@gmail.com).
Built on top of [SDFusion](https://github.com/yccyenchicheng/SDFusion) — original README
preserved at `README_UPSTREAM.md`; the training/research surface lives on the
`upstream-training` branch, `main` is the demo/serving code. Agents: start at `CONTEXT.md` (research thesis, ubiquitous
language, and current **Project status**), then `docs/wayfinding/` (the living per-map status
with tables + montages) and `docs/adr/` (architecture decisions).

## The design doctrine

**Every *decision* about what a building looks like is made by a learned model; every
*realization* into geometry is deterministic procedure.** That split is what keeps
everything editable: models choose, procedures build, and any choice can be re-rolled
without destroying the rest of the building.

| learned component | checkpoint | decides |
|---|---|---|
| recipe-parameter diffusion | `outputs/recipe_param_diffusion_b6` ⚠️ cluster-only | massing: proportions/roof/wings from footprint + class + style |
| Stage 3a SDF diffusion (947M) + VQVAE | `logs_building/continue-stage3a-xcultural-warmstart-ft*/ckpt` ⚠️ cluster-only | "AI massing" from real NL/DE/JP buildings; the localized snap (`/snap_sdf`) |
| **vecset massing diffusion (49M)** — the current research line | **published**, HF `massing-vecset/` | footprint + height → solid mass; see the [model card](https://huggingface.co/danvisimhadri/SDFUSION) |
| PartLayoutPlannerV2 | `outputs/part_layout_planner_v2` | window/door/balcony layouts, ornament slots, Make-it-architecture typing |
| CoherentPartRefiner | `outputs/part_set_refiner` | integrating a sculpted mass into the building's part set |
| PartComposer | `outputs/part_composer` | the always-on statistical facade detail |
| SDXL + depth/canny/scribble ControlNets + IP-Adapter | HF cache (auto-downloads) | texture bake, photoreal renders, sketch-relief art |
| Depth Anything V2 | HF cache | relief height from generated art |

Not learned (by design): footprint extraction (classical CV), recipe→SDF realization,
CSG sculpting, weathering (pure noise functions), ornament retrieval/fit (real heritage
scans; placement IS learned), marching cubes / UV / PBR heuristics, and the sketch-relief
rectification + fusion math.

## Setting up on a new machine

📄 **[`REPRODUCING.md`](REPRODUCING.md) is the full guide** — clone, environment, third-party
checkpoint, data regeneration with runtimes, and verification against a committed baseline. Start
there. This section is the short version.

| what | where it comes from |
|---|---|
| code, docs, results | this repo |
| **model weights** | **<https://huggingface.co/danvisimhadri/SDFUSION>** → `massing-vecset/` |
| corpus (35 GB SDF + 17.4 GB latents) | **regenerated** from 25 MB committed here — `REPRODUCING.md` §4 |
| `external/dora_vae_1_1.ckpt` (2.1 GB) | Hugging Face, third-party |
| SDXL / ControlNet / Depth-Anything (~46 GB) | HF cache, auto-downloads on first texture/render call |

```bash
git clone https://github.com/danvisai/SDFusion.git && cd SDFusion
git checkout massing-solid-gate-retrain      # the active branch; `main` is far behind

python -m venv venv && venv/bin/pip install -r requirements-frozen.txt
hf download danvisimhadri/SDFUSION --include 'massing-vecset/*' --local-dir weights/

CUDA_VISIBLE_DEVICES=0 ./venv/bin/python -m uvicorn scripts.server.inference_service:app \
    --host 0.0.0.0 --port 8099 --log-level warning
```

⚠️ **`requirements-frozen.txt` is CUDA-pinned** (14 `nvidia-*-cu12`, `torch==2.8.0+cu126`). On an
**AMD/ROCm** box do *not* install it verbatim — see
[`transfer/huggingface/AGENT-HANDOFF.md`](transfer/huggingface/AGENT-HANDOFF.md) §1, which covers the
port: training needs no pytorch3d, the ~40 `.cuda()` sites are fine under ROCm's HIP mapping, and
pytorch3d is the one real blocker (build it CPU-only).

⚠️ **The demo's serving weights are NOT published.** `outputs/recipe_param_diffusion_b6` (80 MB),
`outputs/part_layout_planner_v2` (18 MB), `outputs/part_set_refiner`, `outputs/part_composer`, and the
snap prior `logs_building/continue-stage3a-xcultural-warmstart-ft*` exist **only on the cluster**. The
HF repo carries the massing research checkpoints, not the demo stack — so a fresh clone runs the
server but fails on first request. Copy those paths before decommissioning the cluster.

Hardware: ~24 GB VRAM comfortable, ~10 GB disk for checkpoints, ~46 GB more for the HF model cache.

Then open:

- **`http://localhost:8099/`** — the town page: drop an OSM map / footprint-mask image
  (samples under `scripts/server/web/samples/`), get a town; select a building to
  restyle / re-height / re-roll it, age it with the weathering slider, mount a heritage
  ornament, texture-bake or photoreal-render the whole town, export glTF for Unreal.
- **`http://localhost:8099/sculpt.html`** — the SDF Sculptor: raymarched live editing.
  Place primitives and let *Make it architecture* interpret them (tower/wing/dormer with
  windows and roofs), carve, bake textures, or sketch a
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
- `docs/` — dated handoffs and build plans. `docs/wayfinding/` is the living per-map status
  (tables + montages); `docs/adr/` the architecture decisions; `docs/professor_report/` the thesis.
- `legacy/archive_2026-07-10/` — the superseded research record (28 historical docs + 217 MB of
  renders/metrics from dead experiments), kept for comparison. `RESTORE.md` there explains what was
  archived and how to restore it. ⚠️ Its 356 GB of dead training checkpoints are cluster-only.
- `transfer/huggingface/` — the published model card, agent handoff, and weight staging.
- Training code, dataset tooling, original SDFusion research surface: `upstream-training`.

## Where to start reading

| you are | start at |
|---|---|
| setting up on a new machine | [`REPRODUCING.md`](REPRODUCING.md) |
| continuing the massing research | [`transfer/huggingface/AGENT-HANDOFF.md`](transfer/huggingface/AGENT-HANDOFF.md), then [`docs/SESSION-HANDOVER-2026-08-03.md`](docs/SESSION-HANDOVER-2026-08-03.md) |
| an agent picking up any thread | `CONTEXT.md` → `docs/wayfinding/` → `docs/adr/` |
| judging the research claims | the [model card](https://huggingface.co/danvisimhadri/SDFUSION) — results, what was ruled out, and the measurement traps |
