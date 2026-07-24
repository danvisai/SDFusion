# #48 — Web demo feature inventory (latest weights + smoke)

Execution record for **#48 "Run the web demo on latest weights + inventory its features"**, a TASK
under map **#47 "Clean supervisor-ready demo"**. Goal: get the web demo running, inventory every
endpoint (what it does / which weights / does it work), and investigate (without forcing) putting
the demo's core geometry on the latest massing weights.

## 1. Service: does it run?

`bash scripts/server/run_web_demo.sh 8099` starts clean. `GET /health` on a cold start:

```json
{"status": "ok", "device": "cuda",
 "styles": ["modern","colonial","victorian","industrial","craftsman","mediterranean","contemporary","public_civic"],
 "classes": ["COMMERCIAL","PUBLIC","RELIGIOUS","RESIDENTIAL"],
 "diversity_presets": {"low": 1.0, "medium": 2.0, "high": 3.0},
 "sdedit_ready": false}
```

`sdedit_ready` flips to `true` **~8 minutes** after startup (the snap prior's two ~11-15GB
checkpoints pre-warm in a background thread per the module docstring — confirmed real, not a
placeholder: GPU memory climbed from 0.5GB -> 8.2GB over that window). Everything not on the
SDEdit path is usable immediately (recipe-param engine loads synchronously at startup, <5s).

**Launch + tunnel** (for the human to actually open it in a browser):
```
# on the GPU box:
bash scripts/server/run_web_demo.sh 8099
# from your laptop:
ssh -L 8099:<gpu-host>:8099 dsimhadr@gilbreth.rcac.purdue.edu
# then open http://localhost:8099/  (town page)  or  http://localhost:8099/sculpt.html  (single-building sculptor)
```
`<gpu-host>` for this session was `gilbreth-k010.rcac.purdue.edu` — whichever node your job lands
on; `hostname` on the GPU box prints it, and `run_web_demo.sh` echoes it on startup too.

## 2. Structural finding: the demo ships TWO independent massing generators

This matters for reading the table below and for §4. `inference_service.py` wires two generative
paths that **do not share weights, architecture, or code**:

1. **Recipe-param engine** (`recipe_inference.RecipeInferenceEngine`, "Option B+") — a small
   `ConditionalDenoiser` diffuses a 3-12 dim **parameter vector** (`outputs/recipe_param_diffusion_b6/
   denoiser.pth`, 2.8MB), which a hand-coded differentiable procedural recipe (`models/networks/
   diff_recipe.py`, deterministic `nn.Module`s, **no learned weights**) turns into an SDF via
   marching cubes. No VQVAE, no Stage3a, anywhere in this path. Powers `/params_to_mesh`,
   `/regenerate_building`, `/generate_tile`, `/generate_from_image`, and the `massing_source:
   "recipe"` branch of `/building_sdf`.
2. **Stage3a SDEdit prior** (`refine.Refiner`, the CONTEXT.md **C1 "transform" mechanism**) —
   the real learned-SDF-diffusion + VQVAE massing prior. Only reachable via `/snap_sdf`,
   `/refine_with_edit` (`mode: "sdedit"`), and `/building_sdf` when `sdedit_strength > 0`.

**This is why "latest massing weights" (map #24's LoD2 checkpoint) can only ever matter to path
(2)** — path (1) has no Stage3a/VQVAE dependency to swap in the first place. See §4.

## 3. Feature inventory

Legend — **smoke**: 🟢 green (HTTP 200, real payload) / 🔴 broken (reproducible error) / ⚪
not re-smoked this session (see note). **cluster**: the CONTEXT.md core/wrapper split plus the
redundancy groupings named in the ticket.

| endpoint | what it does | weights / model | smoke | cluster |
|---|---|---|---|---|
| `GET /health` | engine status, style/class enums, `sdedit_ready` | — | 🟢 | infra |
| `POST /params_to_mesh` | FAST path: recipe params → differentiable-recipe SDF → mesh, no model call (<200ms) | none (deterministic recipe) | 🟢 0.3s | **core-adjacent** (Option B+ realization) |
| `POST /regenerate_building` | generative: sample recipe params (diffusion) + compose detail → mesh | `recipe_param_diffusion_b6/denoiser.pth` + `part_composer/part_composer.pth` (detail) | 🟢 (first call 35.6s incl. warm-up, later fast) | **core-adjacent** (Option B+ decision) |
| `POST /generate_tile` | batch wrapper over `regenerate_building` for a footprint set | same as above | 🟢 0.76s (2 buildings) | wrapper (batching) |
| `POST /generate_from_image` | OSM map / footprint-mask image → extracted polygons → a generated town | `footprint_image.py` (OpenCV, no model) + `regenerate_building`'s weights | 🟢 4.0s, 3 buildings from a synthetic test image | wrapper (OSM→town demo goal) |
| `POST /building_sdf` | generate ONE building as a 64³ SDF cube for the raymarched sculptor; `massing_source: recipe\|bag`; `sdedit_strength>0` additionally snaps via Stage3a | recipe engine, or nearest-footprint retrieval from `data/bag3d_v1/bag3d.h5` (`massing_source=bag`, no model), + Stage3a if `sdedit_strength>0` | 🟢 0.19-0.44s (recipe, `sdedit_strength=0`) | **core** input to the sculptor |
| `POST /snap_sdf` | **the C1 SDEdit transform** — base 64³ volume + primitive edits → Stage3a partial-noise projection → a new coherent 64³ volume | `continue-stage3a-xcultural-warmstart-ft-final/ckpt/stage3a_steps-latest.pth` (main) + `continue-stage3a-xcultural-warmstart-ft/ckpt/stage3a_steps-1000.pth` (autoguidance) + VQVAE embedded in that checkpoint | 🟢 0.5s, `iou_to_edit=0.916`, mesh returned | **CORE (C1)** |
| `POST /refine_with_edit` (`mode=fast\|quality\|displacement\|sdedit`) | project a sculpted edit onto a clean recipe building (cleanup/re-style); `sdedit` mode = same C1 transform as `/snap_sdf` but on a recipe-based edited mesh | `fast/quality/displacement`: recipe engine only; `sdedit`: Stage3a (as above) | 🟢 `mode=sdedit` 16.8s (incl. compose overhead) | `sdedit` mode = **CORE (C1)**; other 3 modes = wrapper (optimization/preservation variants of the same idea) |
| `POST /detail_volume` | live "bake-quality" composer detail (windows/bands/plinth/roof/landmarks) preview on a massing volume | `part_composer/part_composer.pth`, `part_layout_planner_v2/planner.pth` | 🟢 0.22s | wrapper; **shared core of the detail cluster** (also runs inside `neural_render`, `bake_texture`, `paint_relief`, `export_town`, `volume_to_world_mesh`) |
| `POST /neural_render` | photoreal single-building render: bake-quality detail volume → G-buffers → SDXL + multi-ControlNet | `diffusers/controlnet-{depth,canny}-sdxl-1.0` + `stabilityai/stable-diffusion-xl-base-1.0` (external pretrained, cached at `external/hf_cache`, 46GB) | 🟢 **but first call took 5m54s** (docstring claims "~30-60s") — see §5 finding | **redundancy cluster: neural-render x3** (with `neural_render_town`, `bake_texture`) |
| `POST /neural_render_town` | same SDXL pipeline, multi-building, per-building style-ref instance masking + weathering | same as `neural_render` | ⚪ not smoked (time budget; same SDXL dependency proven live by `neural_render`) | **neural-render x3** |
| `POST /bake_texture` | UV-unwrap + multi-view SDXL diffusion (TEXTure-style seam-free) → textured .glb | same SDXL pipe + inpaint pipe | ⚪ not smoked (time budget) | **ornament+paint_relief+detail x3** (appearance-bake family) |
| `POST /paint_relief` | paint a patch → SDXL bas-relief art → a real geometric SDF relief (not just texture) | same SDXL pipe + inpaint pipe | ⚪ not smoked (time budget) | **ornament+paint_relief+detail x3** |
| `POST /export_town` (`textures=false`) | export the whole town as one glTF for Unreal/Blender, geometry-only, procedural rebuild | `town_export.py` (procedural; reuses recipe/composer weights, no diffusion) | 🟢 0.31s, 1 building, 30800 verts | wrapper (recipe-closure demo feature) |
| `POST /export_town` (`textures=true`) | same, v2 per-building albedo bake | + SDXL pipe | ⚪ not smoked (time budget; same SDXL family) | wrapper |
| `POST /ornament_building` | places a heritage-relief instance via the trained part-layout planner + CoherentPartRefiner deconfliction; retrieval from a 3-item local library | `part_layout_planner_v2/planner.pth`, `part_set_refiner/coherent_refiner.pth`, `data/ornaments_v1/` (3 GLBs) | 🟢 0.72s, returned `ox_relief_romanesque` | **ornament+paint_relief+detail x3** |
| `POST /interpret_mass` | "smart add": classify a placed cube-frame primitive as an architectural part (tower/dormer/window/…) + optional CoherentPartRefiner row-alignment | `part_layout_planner_v2/planner.pth` (+ retrieval library) | 🟢 1.8s, `kind: raw` for an unclassifiable test box (correct fallback, not an error) | **interpret_mass x2** (cube frame, for `sculpt.html`) |
| `POST /interpret_mass_world` | same classify→construct pipeline, bridged to world-meter frame | same as above | 🟢 0.46s, `kind: tower`, `source: rules+library` | **interpret_mass x2** (world frame, for the town page `index.html`) — genuinely two calling conventions for the same core logic, not pure duplication |
| `POST /rebuild_building` | PROCEDURAL-ONLY rebuild (recipe + CSG edits + composer detail), the exact path `/export_town` uses per-building; town page's undo/Make-it-architecture go through this so placed constructions are never remolded by the diffusion prior | recipe engine + composer (no diffusion) | 🟢 0.18s | wrapper (recipe-closure guarantee) |
| `POST /recohere_details` | LEARNED re-coherence of a **detail-op SET** (windows/balconies/…) against the massing — drops implausible parts, adjusts poses | `part_set_refiner/refiner.pth` | 🟢 0.09s (trivial empty-ops smoke; needs a real op list for a meaningful test) | **facade recohere x2** (per-instance granularity) |
| `POST /recohere_facade` | GENERATIVE **facade-program** re-coherence (12-dim `DetailParams` vector) via a trained detail head + SDEdit | `outputs/detail_generator/detail_gen.pth` | 🔴 **BROKEN** — `FileNotFoundError`, see §6 | **facade recohere x2** (program-space granularity) |

## 4. Latest-weights wiring — investigation + what was done

Per §2, "latest massing weights" (map #24's `logs_building/2026-07-16-stage3a-lod2-fromscratch-
region/ckpt/stage3a_steps-latest.pth`) is only even a candidate for the Stage3a/SDEdit path
(`Refiner`), never the recipe-param engine (no shared architecture at all).

### (a) Clean VQVAE swap — DONE, but it's a no-op (documented, not a real lever)

Investigated whether pointing `Refiner._mk_stage3a`'s fallback `vq_ckpt` at
`logs_building/vqvae_clean_ft/vqvae_clean.pth` (the clean VQVAE `baseline_gate_eval.py`'s
deployed-model config uses) would matter. Traced `Stage3aModel.initialize` →
`load_vqvae(vq_ckpt=opt.vq_ckpt)` constructs `self.vqvae` from that path, but the very next call
(`load_ckpt`, `models/stage3a_model.py:898-899`) does:
```python
if "vqvae" in state:
    self.vqvae.load_state_dict(state["vqvae"])
```
unconditionally — and **every** relevant Stage3a checkpoint embeds its own `"vqvae"` key.
Confirmed via zip-archive/pickle-string inspection (no full `torch.load` needed) on all three:
`continue-stage3a-xcultural-warmstart-ft-final` (deployed snap prior), `2026-07-16-stage3a-
lod2-fromscratch-region` (map #24), and `2026-06-08T11-50-42-stage3a-hybrid-clean` (its parent) —
all three have a `vqvae` top-level key, so whatever `vq_ckpt` path `_mk_stage3a` passes is always
discarded and replaced by the checkpoint's own embedded VQVAE immediately after.
**Made the edit anyway** (`scripts/server/refine.py::_mk_stage3a`, now points at
`vqvae_clean_ft/vqvae_clean.pth` instead of the stale `2025-05-19T...` path) for hygiene/
consistency — but documented in-code that it is **provably behavior-neutral**: the demo's snap
prior was already running on whatever VQVAE its own checkpoint carries, before and after this
edit. Re-smoked `/snap_sdf` and `/refine_with_edit` (sdedit mode) post-edit — both still 🟢.

### (b) Map #24 LoD2 generator for the generation endpoints — BLOCKED, confirmed incompatible (not forced)

The ticket's caveat is real and I made it concrete rather than assumed. Loaded both checkpoints'
state dicts with `torch.load(..., map_location="meta")` (shape-only, fast, no Lustre stall) and
compared `global_proj.0.weight`:

| checkpoint | `global_proj.0.weight` shape | `era_emb`/`floors_emb` (`use_extra_cond`) | `region_emb` (`use_region`) |
|---|---|---|---|
| deployed snap prior (`continue-stage3a-xcultural-warmstart-ft-final`) | `(512, 368)` | present | absent |
| map #24 generator (`2026-07-16-stage3a-lod2-fromscratch-region`) | `(512, 352)` | absent | absent* |

`Stage3aModel.load_ckpt` loads `global_proj` with a **plain, unguarded** `load_state_dict` (no
shape-adapter for this case — the only adapter, `_fit_global_proj`, exists for a different flag,
`use_element_type`) — so instantiating either architecture and loading the other's checkpoint
would hit an immediate `RuntimeError: size mismatch` on `global_proj.0.weight` (368 vs 352
in_features). This is a hard architectural incompatibility, not a maybe — **not attempted live**
(would cost a ~15GB Lustre read to reproduce a deterministic tensor-shape crash). **Options, none
executed:**
1. Retrain/warm-start a variant with **both** `use_extra_cond` and `use_region` active so one
   checkpoint serves both roles (real training run, out of scope here).
2. Add a **third, separate** Stage3a instantiation in `Refiner` for a true from-noise "generate"
   mode (map #24 was trained with `.inference()`, not `.sdedit()` — a plain conditional sample,
   not a partial-noise projection) and expose it as a new `massing_source` option alongside
   `recipe`/`bag` on `/building_sdf` — a real feature addition, not a config edit, so left for a
   follow-up ticket.
3. Leave it split as-is (current state): the demo's SDEdit/snap experience stays on the
   cross-cultural warm-start prior; anyone wanting the #27-gate-passing LoD2 massing quality uses
   `scripts/foundations/baseline_gate_eval.py` directly (already wired, already gate-verified).

**Side finding (not in scope to fix):** `Stage3aModel.save()`/`load_ckpt()` never persists
`region_emb`'s own weights (grep confirms no `state["region_emb"] = ...` anywhere, unlike the
symmetric `era_emb`/`floors_emb` handling) — so the map #24 checkpoint's `region_emb` table is
NOT restored on reload; every reload (including the eval runs behind the #27 gate result) is
using a freshly re-initialized region-embedding table. Whether this affects the #27 result depends
on how much the trained model actually leaned on region conditioning vs the unconditional
"unknown" bucket at inference — flagging for the map-#24 owner, not chasing further here.

### (c) v1 surface-crispness refiner as an optional post-process — DONE, wired + smoked

This one *is* a clean, additive, reversible change, so it was made:
- `scripts/server/refine.py`: `Refiner._load_surface_refiner()` (lazy-load
  `outputs/refiner_v1/refiner_unet_v1.pth` via `baseline_gate_eval.load_refiner`, caches a
  sentinel if the checkpoint is missing) + `Refiner.apply_surface_refiner(grid)` (runs the frozen
  net on any `(R,R,R)` cube-frame SDF, `R` divisible by 8).
- `scripts/server/inference_service.py`: new **opt-in, default-`False`** `surface_refine: bool`
  field on `BuildingSdfReq` and `SnapSdfReq`; applied right after the massing/snap grid is
  produced, before mesh baking.
- **Caveat documented in-code and here:** this refiner was trained + validated specifically
  against **map #24's** output distribution (`docs/wayfinding/massing-surface-fidelity/refiner-
  v1-result.md`) — applying it to the demo's *different*, currently-live Stage3a checkpoint is an
  untested cross-model generalization. It is architecturally safe to run (pure grid→grid residual
  CNN, no dependency on which prior produced the grid) but not validated to help *this* prior's
  specific artifacts. Hence opt-in, not default-on.

**Re-smoked after wiring** (post service-restart, both flag states, both endpoints):

| call | HTTP | notes |
|---|---|---|
| `/building_sdf` `surface_refine=false` | 🟢 200, 0.44s | baseline (recipe massing) |
| `/building_sdf` `surface_refine=true` | 🟢 200, 1.42s (first, lazy-load) / 0.17s (subsequent) | mean\|Δsdf\|=0.0020, max\|Δsdf\|=0.0275 vs baseline — genuinely applied, not identical |
| `/snap_sdf` `surface_refine=false` | 🟢 200, 0.71s | baseline (post-SDEdit snap) |
| `/snap_sdf` `surface_refine=true` | 🟢 200, 0.49s | `iou_to_edit` 0.909→0.905 (small, expected); mean\|Δsdf\|=0.0022 |

Visual: `outputs/ticket48_demo_smoke/surface_refine_before_after.png` and `.../
snap_surface_refine_before_after.png` — at this render scale/edit configuration the delta is
subtle (the recipe-massing path is already analytically smooth with no diffusion-injected wobble
for the refiner to remove; the snap-path test used a small localized edit far from most of the
visible silhouette). The refiner's actual roughness-reduction case (map #24's own wavy output) is
already documented with a clearly-visible before/after in `docs/wayfinding/massing-surface-
fidelity/refiner-v1-before-after.png` — not re-litigated here.

## 5. Finding: `/neural_render` first-call latency far exceeds its own docstring

Docstring says "First call loads SDXL (~30-60s); after that ~10-15s per render." Measured: **5m54s**
for the actual first call on this node (`outputs/ticket48_demo_smoke/neural_render_modern.png` is
the real output — it did succeed, image is a legible photoreal render of the modern-style test
building). GPU memory grew slowly and monotonically throughout (not hung/stuck), consistent with
the Lustre single-file-read stall this repo has already documented elsewhere (`run_web_demo.sh`'s
`PYTHONPYCACHEPREFIX` comment references the same class of issue) rather than a code bug — the
46GB SDXL cache at `external/hf_cache` is on the same Lustre-backed scratch. Not fixed (out of
scope; would mean relocating the HF cache to node-local disk the way `SNAP_CKPT_DIR` already does
for the Stage3a checkpoints — a reasonable follow-up for whoever owns the appearance cluster).

## 6. Broken / missing

- **`/recohere_facade` is broken**: `outputs/detail_generator/detail_gen.pth` does not exist (the
  directory `outputs/detail_generator/` exists and is empty). `scripts/train_detail_generator.py`
  is the training script that would produce it; it was apparently never run (or its output was
  since deleted). Not retrained here (out of scope for an inventory pass) — recorded as a genuine
  gap: one of the two facade-recohere endpoints is currently dead code from the API consumer's
  perspective.
- Four SDXL-family endpoints (`neural_render_town`, `bake_texture`, `paint_relief`,
  `export_town(textures=true)`) were not individually smoked this session (time budget after
  `neural_render`'s 6-minute cold load) — they share the exact same `neural_appearance.get_pipe()`/
  `get_inpaint_pipe()` dependency already proven to load and run successfully.

## 7. Visual evidence

All under `outputs/ticket48_demo_smoke/`:
- `core_endpoints_montage.png` — meshes from `regenerate_building`, `snap_sdf`, `refine_with_edit`
  (sdedit mode, shows composer detail/windows/tower), `export_town`.
- `neural_render_modern.png` — the actual SDXL photoreal render output (proves the appearance
  pipeline works end-to-end, not just that the endpoint returns 200).
- `surface_refine_before_after.png`, `snap_surface_refine_before_after.png` — the new §4(c) wiring,
  before/after.
- `regenerate_building.glb`, `snap_sdf.glb`, `refine_sdedit.glb`, `export_town.glb` — raw glTF
  outputs, importable directly into Blender/UE for a closer look.

## 8. Files touched (uncommitted, per orchestrator instruction)

- `scripts/server/refine.py` — `_mk_stage3a`'s `vq_ckpt` path (no-op, see §4a);
  `Refiner._load_surface_refiner` / `Refiner.apply_surface_refiner` (new, §4c).
- `scripts/server/inference_service.py` — `surface_refine: bool = False` field on
  `BuildingSdfReq`/`SnapSdfReq` + the two call sites (§4c).

Service was killed at the end of this session (`kill <uvicorn pid>`) — nothing left running.
