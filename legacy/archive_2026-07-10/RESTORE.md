# Archive 2026-07-10 — reversible cleanup around the finalized transform+composition proof

Everything here was **moved** (not deleted) to focus the repo on the finalized research direction.
Restore any item with `mv legacy/archive_2026-07-10/<...> <original-location>`.

## What was archived
- **docs/** (28 files): historical HANDOFF_*, dated SPEC/REPORT/PLAN, CODEX_* reports, PROJECT_STATUS,
  PROJECT_COMPENDIUM, DEPLOYMENT_PLAN, etc. (superseded chronicle).
- **memory/** (8 files): demo-wrapper / superseded feature memories (coherent-add, demo-build-plan,
  demo-hardening, frontend-coherence, data-sourcing-progress, weathering, layer-a-context-snap,
  sketch-relief). Note: memory files actually live under
  `~/.claude/projects/.../memory/_archive_2026-07-10/`, not here.
- **outputs/** (61 dirs): dead-experiment outputs (detailizer_v*, detail_generator, osm_3dgs_*,
  osm_sdf_*, sdedit_*, layerA_eval, diff_recipe_*, recipe_param_head_b5*, audits, etc.).
- **logs/** (31 entries): dead training runs from `logs_building/` (20: bag3d-prior*, repa, adaln,
  bs32-*, resume*, layerA/AB-context, region-pilot, _validate_*) and `Logs_GT/` (11: retrieval/residual
  smokes + older residual runs).
- **code/** (3 scripts): the only truly-standalone abandoned scripts —
  `train_osm_generation_success_predictor.py`, `bake_buildingnet_to_3dgs.py`, `repreview_gsplat_v2.py`.

## What was KEPT (and why)
- **Finalized set:** `CONTEXT.md`, `docs/adr/0001-0003`, `execution/*` plans,
  `.scratch/transform-composition-proof/` (codex PRD + 18 tickets), `tickets.md`,
  `docs/professor_report/` (thesis source + paper figures), `docs/HYBRID_PIPELINE_PLAN.md`.
- **Live-serving checkpoints:** `outputs/recipe_param_diffusion_b6(_ema)`, `part_layout_planner_v2`,
  `part_set_refiner`, `part_composer`; `logs_building/{...vqvae-building-all-res64..., stage3a-hybrid-clean,
  continue-stage3a-xcultural-warmstart-ft, ...-ft-final}`; `Logs_GT/retrieval_footprint_full`.
- **C1 evidence:** `Logs_GT/sdf_residual_full_v4_aug_topk3`, `train_sdf_residual.py`,
  `models/networks/sdf_residual_net.py`, `datasets/correction_pair_dataset.py`, `train_retrieval.py`.
- **C2 monolith harness:** `scripts/foundations/train_detailizer.py`, `make_detail_pairs.py`,
  `models/networks/repa.py`, `models/stage3b_model.py`.

## Code NOT archived despite being "abandoned pipelines" — entanglement
The osm / hunyuan / gsplat / SDF→GS-lifter code is a **deeply interconnected web**, and the proof's
generation skeleton **`scripts/stage3_generate.py` imports into it** (`osm_hunyuan_pipeline_smoke`,
`models/networks/sdf_to_gs_lifter` → `voxelize_gsplats`; `scene/gsplat_*`). Archiving it would break
proof-referenced code. **Recommended:** decouple as part of building the C2 decomposition arm (write a
clean Stage3a-massing + retrieval generation script for I2.1 that does not import the hunyuan/gsplat
chain), then archive the web in a follow-up.
