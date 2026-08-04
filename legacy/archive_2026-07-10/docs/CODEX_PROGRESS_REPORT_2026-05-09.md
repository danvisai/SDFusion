# Codex Progress Report - 2026-05-09

Prepared by: Codex  
Project: `/scratch/gilbreth/dsimhadr/GenerativeTowns/SDFusion`  
Timestamp: 2026-05-09 23:44 EDT

## Current Status

Claude's latest direction has been implemented far enough to evaluate:

- Keep retrieval-first as the visual-quality backbone.
- Use corrected SDFs as structural guidance, not as the final high-detail asset.
- Train the residual model with geometry-aware losses instead of plain residual L1.
- Select checkpoints by geometry/sign quality, not only by continuous SDF L1.
- Increase residual-pair diversity with top-K retrieval pairs.

The active best checkpoint for corrected-SDF output is:

```text
Logs_GT/sdf_residual_full_v4_aug_topk3/ckpt_best_geom.pth
```

`scene/run_demo.py` now defaults to this v4 geometry checkpoint when `--use_residual` is enabled.

## Claude Approach Captured

Claude's method was an empirical ablation loop:

1. Plain residual L1 improved continuous SDF error but could make sign/occupancy quality worse.
2. Composite loss was added to optimize the geometry that matters for extraction:
   - residual SmoothL1
   - surface-band SmoothL1
   - soft sign BCE
   - soft footprint-projection BCE
3. Checkpointing was expanded:
   - `ckpt_best.pth` for best corrected L1
   - `ckpt_best_iou.pth` for best sign IoU
   - `ckpt_best_geom.pth` for combined geometry score
4. Augmentation was tested separately and was mostly a wash.
5. Top-K retrieval pairs were added because data diversity looked more important than more augmented views.
6. The output is judged by mesh/contact-sheet evaluation, not scalar loss alone.

## Implemented Changes From This Direction

### Composite Residual Loss

`train_sdf_residual.py` now supports geometry-aware terms:

- `--w_residual_l1`, default `0.5`
- `--w_band_l1`, default `1.0`
- `--w_sign_bce`, default `1.0`
- `--w_fp_bce`, default `0.5`
- `--band_sigma`, default `0.1`
- `--sign_tau`, default `0.05`

The trainer logs each component and computes:

```text
geom_score = corrected_iou + corrected_fp_iou - 0.5 * corrected_l1
```

### Training Augmentation

`datasets/correction_pair_dataset.py` supports matched 3D/2D augmentation:

- 90-degree rotations around the vertical axis
- X flip
- Z flip

This is enabled with `--augment`.

### Top-K Correction Pairs

Top-K=3 correction pairs were built:

```text
data/BuildingNet_dataset_v0_1/correction_pairs_topk3
```

Counts:

```text
train metadata rows: 4444 including header, so 4443 pairs
val metadata rows:    562 including header, so 561 pairs
```

This expands training diversity from the top-1 dataset:

```text
top-1 train pairs: 1481
top-3 train pairs: 4443
```

## Residual Training Runs

### v2: Composite Loss, Top-1, No Augmentation

```text
Logs_GT/sdf_residual_full_v2_composite
```

Best checkpoints:

```text
ckpt_best.pth        epoch 27  L1=0.1011  signIoU=0.2389  fpIoU=0.9990  geom=1.1874
ckpt_best_iou.pth    epoch 16  L1=0.1125  signIoU=0.2925  fpIoU=0.9981  geom=1.2344
ckpt_best_geom.pth   epoch 16  L1=0.1125  signIoU=0.2925  fpIoU=0.9981  geom=1.2344
ckpt_latest.pth      epoch 30  L1=0.1071  signIoU=0.2169  fpIoU=0.9979  geom=1.1613
```

### v3: Composite Loss, Top-1, Augmentation

```text
Logs_GT/sdf_residual_full_v3_aug
```

Best checkpoints:

```text
ckpt_best.pth        epoch 29  L1=0.1002  signIoU=0.2394  fpIoU=0.9987  geom=1.1880
ckpt_best_iou.pth    epoch 16  L1=0.1100  signIoU=0.2926  fpIoU=0.9989  geom=1.2364
ckpt_best_geom.pth   epoch 16  L1=0.1100  signIoU=0.2926  fpIoU=0.9989  geom=1.2364
ckpt_latest.pth      epoch 30  L1=0.1058  signIoU=0.2071  fpIoU=0.9979  geom=1.1520
```

Interpretation:

- Augmentation alone was almost neutral.
- It slightly improved L1 but did not materially improve sign IoU.
- Diversity was still the bottleneck.

### v4: Composite Loss, Top-K=3, Augmentation

```text
Logs_GT/sdf_residual_full_v4_aug_topk3
```

Training completed for 30 epochs.

Best checkpoints:

```text
ckpt_best.pth        epoch 24  L1=0.0986  signIoU=0.2108  fpIoU=0.9993  geom=1.1607
ckpt_best_iou.pth    epoch 21  L1=0.1069  signIoU=0.2950  fpIoU=0.9992  geom=1.2408
ckpt_best_geom.pth   epoch 18  L1=0.1015  signIoU=0.2945  fpIoU=0.9992  geom=1.2429
ckpt_latest.pth      epoch 30  L1=0.1015  signIoU=0.1346  fpIoU=0.9989  geom=1.0828
```

Interpretation:

- v4 is the best geometry run so far, but the improvement over v2/v3 is modest.
- `ckpt_latest.pth` is clearly not the best visual candidate.
- Use `ckpt_best_geom.pth` for downstream corrected-SDF demos and mesh evaluation.

## v4 Mesh Evaluation

Existing output:

```text
outputs/sdf_residual_mesh_eval_v4/val
```

Artifacts:

- `mesh_metrics.csv`
- 16 validation contact sheets under `sheets/`
- source/corrected/target OBJs under `meshes/`

Aggregate metrics over the 16 evaluated validation examples:

```text
source_sdf_l1 mean:        0.2990
corrected_sdf_l1 mean:     0.1149
source_fp_iou mean:        0.2006
corrected_fp_iou mean:     0.9943
source_components mean:    9.25
corrected_components mean: 12.06
target_components mean:    29.81
```

Interpretation:

- The residual model strongly improves SDF L1.
- It almost perfectly enforces footprint occupancy in these examples.
- It does not solve the final visual-quality problem by itself.
- Corrected marching-cubes meshes still inherit 64^3 sparsity/hollow-shell limits and can remain fragmented.
- This confirms the corrected SDF should be used as a structural guide or fallback mesh, while the retrieved OBJ remains the main detailed visual asset.

## Demo Output

Two demo town outputs exist:

```text
outputs/demo_town.obj
outputs/demo_town_corrected.obj
```

The corrected demo is much smaller:

```text
demo_town.obj:           about 39 MB
demo_town_corrected.obj: about 5.3 MB
```

This is expected because the corrected path extracts low-resolution SDF meshes instead of preserving retrieved OBJ detail.

## Current Decision

Claude's approach was correct: add geometry-aware losses, checkpoint by geometry, and evaluate visually. The result is useful, but it also confirms a boundary:

- corrected SDFs are good for footprint/massing control;
- retrieved OBJs are still needed for architectural detail;
- 64^3 corrected marching-cubes output is not enough for final-quality buildings.

## Recommended Next Implementation

The next engineering step should be retrieval-guided OBJ adaptation, not more plain residual training.

Priority order:

1. Use v4 `ckpt_best_geom.pth` only as the corrected-SDF guide.
2. For each retrieved OBJ, use corrected SDF/footprint to choose transform and crop/fit the OBJ.
3. Add rotation-candidate alignment before residual correction:
   - try 0, 90, 180, 270 degrees;
   - compare anisotropic vs uniform X/Z scaling;
   - score by filled footprint IoU.
4. Run top-K candidates through the correction/eval path and select the best candidate by:
   - corrected footprint IoU;
   - sign IoU or geom score;
   - mesh component sanity;
   - visual contact sheet.
5. Keep MeshUDF as an extraction experiment, but do not make it the main visual path until it beats retrieved OBJ detail.

## Immediate Commands

Use this checkpoint for corrected-SDF experiments:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python scripts/eval_sdf_residual_meshes.py \
  --ckpt Logs_GT/sdf_residual_full_v4_aug_topk3/ckpt_best_geom.pth \
  --phase val \
  --limit 16 \
  --out_dir outputs/sdf_residual_mesh_eval_v4 \
  --export_obj \
  --device cuda
```

Use this for a corrected demo town:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python scene/run_demo.py \
  --use_residual \
  --out outputs/demo_town_corrected.obj \
  --device cuda
```

## Hunyuan3D Follow-Up - 2026-05-10

Claude's next recommendation after inspecting the poor scene-level OBJ/SDF outputs was to try an image-to-3D path with Hunyuan3D-2. That has now been started.

Added:

```text
scripts/hunyuan_building_mesh_smoke.py
```

Purpose:

- Take existing building images from the legacy Path-Q / ControlNet experiments.
- Run Hunyuan3D-2 or Hunyuan3D-2mini shape generation.
- Export GLBs.
- Render preview PNGs.
- Build a contact sheet for quick visual comparison.

Important environment fix:

- Hunyuan originally tried to download to `/home/dsimhadr/.cache/hy3dgen` and failed with disk quota.
- The script now defaults model/cache locations to scratch/project-local paths:

```text
HF_HOME=/scratch/gilbreth/dsimhadr/GenerativeTowns/SDFusion/external/hf_cache
HUGGINGFACE_HUB_CACHE=/scratch/gilbreth/dsimhadr/GenerativeTowns/SDFusion/external/hf_cache/hub
XDG_CACHE_HOME=/scratch/gilbreth/dsimhadr/GenerativeTowns/SDFusion/external/xdg_cache
HY3DGEN_MODELS=/scratch/gilbreth/dsimhadr/GenerativeTowns/SDFusion/external/hy3dgen_models
```

Smoke command:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/hunyuan_building_mesh_smoke.py \
  --model mini \
  --limit 4 \
  --out_dir outputs/hunyuan_building_smoke_mini_4
```

Output:

```text
outputs/hunyuan_building_smoke_mini_4/hunyuan_building_smoke_sheet.png
outputs/hunyuan_building_smoke_mini_4/metrics.csv
outputs/hunyuan_building_smoke_mini_4/*.glb
outputs/hunyuan_building_smoke_mini_4/*_render.png
```

Metrics:

```text
00_row1_controlnet_gen      1,016,064 verts  2,032,296 faces  14.37s
01_row2_controlnet_gen      1,335,575 verts  2,671,384 faces  14.50s
02_row1_controlnet_gen_neg    527,470 verts  1,055,064 faces  13.73s
03_row2_controlnet_gen_neg    495,395 verts    990,882 faces  13.75s
```

Artifacts are dense: the four generated GLBs total about 97 MB. The scratch Hugging Face cache is about 21 GB after the download.

Current interpretation:

- Hunyuan3D mini runs successfully on CUDA from the project environment.
- The generated meshes are far denser and more detailed than 64^3 corrected-SDF marching-cubes outputs.
- The next decision should be visual inspection of `hunyuan_building_smoke_sheet.png`.
- If visually acceptable, the next implementation is to decimate/simplify these GLBs and wire them into `scene/run_demo.py` as an optional mesh source.

## Retrieval-Render Hunyuan Run - 2026-05-10

The first Hunyuan run used old ControlNet-generated building images. The better inputs are the retrieval render panels from Claude's retrieval contact sheets, because those are based on actual retrieved OBJ geometry.

Created cropped Hunyuan-ready inputs:

```text
outputs/hunyuan_retrieval_inputs/
```

Source sheets:

```text
outputs/retrieval_visual_smoke/*.png
```

Cropping logic:

- each sheet is 8 columns of 256 px panels;
- query OBJ is column 1;
- rank-1 OBJ is column 3;
- rank-2 OBJ is column 5;
- rank-3 OBJ is column 7;
- only the 256x256 rendered image body is cropped, excluding sheet labels.

Ran Hunyuan3D-2mini on the rank-1 retrieval renders:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/hunyuan_building_mesh_smoke.py \
  --model mini \
  --inputs outputs/hunyuan_retrieval_inputs/*_rank1_obj.png \
  --limit 6 \
  --out_dir outputs/hunyuan_retrieval_rank1_mini
```

Output:

```text
outputs/hunyuan_retrieval_rank1_mini/hunyuan_building_smoke_sheet.png
outputs/hunyuan_retrieval_rank1_mini/metrics.csv
outputs/hunyuan_retrieval_rank1_mini/*.glb
outputs/hunyuan_retrieval_rank1_mini/*_render.png
```

Raw Hunyuan rank-1 metrics:

```text
city_hall rank1:       441,626 verts   883,438 faces
hotel rank1:           402,684 verts   805,372 faces
house6823 rank1:       508,023 verts 1,016,788 faces
house7736 rank1:       228,858 verts   457,740 faces
office0274 rank1:      244,084 verts   488,260 faces
office0640 rank1:        4,834 verts     9,668 faces
```

Added mesh simplification:

```text
scripts/simplify_hunyuan_meshes.py
```

Simplification command:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/simplify_hunyuan_meshes.py \
  --input_dir outputs/hunyuan_retrieval_rank1_mini \
  --out_dir outputs/hunyuan_retrieval_rank1_mini_simplified \
  --target_faces 50000 \
  --out_ext .obj
```

Simplified outputs:

```text
outputs/hunyuan_retrieval_rank1_mini_simplified/*.obj
outputs/hunyuan_retrieval_rank1_mini_simplified/simplify_metrics.csv
```

Simplification result:

```text
first five meshes: reduced to 50,000 faces each
small office0640 mesh: kept at 9,668 faces
```

Added scene composition smoke:

```text
scripts/compose_hunyuan_scene_smoke.py
```

Command:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/compose_hunyuan_scene_smoke.py \
  --mesh_dir outputs/hunyuan_retrieval_rank1_mini_simplified \
  --out outputs/demo_town_hunyuan_rank1.obj
```

Scene output:

```text
outputs/demo_town_hunyuan_rank1.obj
outputs/demo_town_hunyuan_rank1.log.json
```

Scene size comparison:

```text
outputs/demo_town.obj                 38 MB
outputs/demo_town_corrected.obj      5.1 MB
outputs/demo_town_hunyuan_rank1.obj  9.7 MB
```

Hunyuan scene mesh count:

```text
129,362 verts
259,668 faces
```

Current interpretation:

- SDF should remain a footprint/massing correction signal only.
- Hunyuan/retrieval OBJ paths are the candidates for final visual meshes.
- Hunyuan needs input images from retrieval renders or future image generation, not the old weak ControlNet PNGs.
- Hunyuan output must be simplified before scene placement.

## 2026-05-10 Codex Update: OSM Input-to-Output Smoke Pipeline

Added a complete OSM-to-output orchestration script:

```text
scripts/osm_hunyuan_pipeline_smoke.py
```

Purpose:

- Input: OSM JSON from `scene/extract_osm.py`.
- Select largest usable OSM building footprints.
- Rasterize each footprint for retrieval.
- Retrieve a BuildingNet OBJ exemplar.
- Render the retrieved OBJ as the image input to Hunyuan3D-2mini.
- Generate raw Hunyuan GLB meshes.
- Simplify each generated mesh to 50k faces for scene composition.
- Place simplified meshes back onto the OSM footprints.
- Export a composed OBJ scene, metrics CSV, log JSON, and input/output contact sheet.

OSM extraction command used:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scene/extract_osm.py \
  --bbox 40.4234 -86.9075 40.4250 -86.9050 \
  --no-roads \
  -o outputs/osm_pipeline_smoke/osm_input.json
```

OSM extraction result:

```text
26 buildings
24 RESIDENTIALhouse
2 COMMERCIALoffice_building
height min/median/max: 7.0 / 7.0 / 14.0 m
area min/median/max: 71.8 / 355.3 / 2475.9 m2
```

Pipeline command used:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/osm_hunyuan_pipeline_smoke.py \
  --osm_json outputs/osm_pipeline_smoke/osm_input.json \
  --out_dir outputs/osm_pipeline_smoke \
  --limit 4 \
  --model mini
```

Pipeline outputs:

```text
outputs/osm_pipeline_smoke/osm_hunyuan_scene.obj
outputs/osm_pipeline_smoke/osm_hunyuan_scene.log.json
outputs/osm_pipeline_smoke/osm_hunyuan_pipeline_metrics.csv
outputs/osm_pipeline_smoke/osm_hunyuan_pipeline_sheet.png
outputs/osm_pipeline_smoke/hunyuan_inputs/*.png
outputs/osm_pipeline_smoke/hunyuan_raw/*.glb
outputs/osm_pipeline_smoke/hunyuan_simplified/*.obj
outputs/osm_pipeline_smoke/renders/*.png
```

Generated sheet:

```text
outputs/osm_pipeline_smoke/osm_hunyuan_pipeline_sheet.png
size: 1536 x 1648
columns: OSM footprint, retrieved OBJ render, raw Hunyuan mesh render, placed mesh render
```

Metrics for the four-building smoke:

```text
OSM_302408405 <- RESIDENTIALhouse_mesh7919:  419,971 verts /   839,940 faces -> 50,000 faces
OSM_860916350 <- RESIDENTIALhouse_mesh9098:  619,850 verts / 1,239,796 faces -> 50,000 faces
OSM_71497515  <- RESIDENTIALhouse_mesh4551:  756,129 verts / 1,512,256 faces -> 50,000 faces
OSM_116427240 <- RESIDENTIALhouse_mesh4551:  753,378 verts / 1,506,756 faces -> 50,000 faces
```

Important observation:

- The raw Hunyuan outputs look visually stronger than the placed renders.
- The placed renders look flattened because the selected OSM buildings have very large footprints but only default/inferred 7 m residential heights.
- This is not a Hunyuan failure; it is an OSM metadata/height-inference issue plus the current `place_mesh` behavior correctly respecting target footprint and height.

Next improvement:

- Add better OSM height inference before scaling final meshes. Area-aware defaults or class-specific floor estimates should prevent large buildings from being compressed into very shallow geometry when OSM lacks explicit height or `building:levels`.

## 2026-05-10 Codex Update: Map-Level Input/Output and Retrieval Choices

The first OSM sheet was too per-building and did not show the map-level input
or the choices made by retrieval. Added:

```text
scripts/osm_pipeline_map_choices.py
```

Command used:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/osm_pipeline_map_choices.py \
  --osm_json outputs/osm_pipeline_smoke/osm_input.json \
  --pipeline_log outputs/osm_pipeline_smoke/osm_hunyuan_scene.log.json \
  --out_dir outputs/osm_pipeline_smoke
```

New outputs:

```text
outputs/osm_pipeline_smoke/osm_map_input.png
outputs/osm_pipeline_smoke/osm_map_selected.png
outputs/osm_pipeline_smoke/osm_map_output_houses.png
outputs/osm_pipeline_smoke/osm_map_choices_sheet.png
outputs/osm_pipeline_smoke/osm_retrieval_choices.csv
outputs/osm_pipeline_smoke/osm_hunyuan_scene_render.png
outputs/osm_pipeline_smoke/retrieval_choice_renders/*.png
```

The new main sheet:

```text
outputs/osm_pipeline_smoke/osm_map_choices_sheet.png
size: 1536 x 1676
top row: OSM map input, selected footprints, output map with placed houses
per-building rows: footprint, top-1 retrieval, top-2 retrieval, top-3 retrieval, generated mesh, placed output
```

Retrieval choices recorded:

```text
OSM_302408405: mesh7919 0.989, mesh6948 0.977, mesh8733 0.976
OSM_860916350: mesh9098 0.992, mesh4188 0.991, mesh6840 0.991
OSM_71497515:  mesh4551 0.977, mesh9657 0.973, mesh5957 0.972
OSM_116427240: mesh4551 0.980, mesh5957 0.975, mesh4709 0.974
```

Interpretation:

- Retrieval top-1 is being chosen consistently and is logged explicitly.
- Some footprints have very close top-k scores, so a future UI or scoring pass should allow choosing among candidate meshes instead of always taking top-1.
- The output-map thumbnail overlay is for visual explanation only; exact geometry is in `osm_hunyuan_scene.obj`.

## 2026-05-10 Codex Update: Geometry-Aware Retrieval Reranking

Started the first practical step toward a more generative pipeline: retrieval is
no longer forced to accept similarity-only top-1. The OSM -> Hunyuan script now
supports top-k retrieval followed by a geometry-aware reranker.

Updated:

```text
scripts/osm_hunyuan_pipeline_smoke.py
scripts/osm_pipeline_map_choices.py
```

New options:

```text
--retrieval_policy top1|rerank
--retrieval_top_k 5
--aspect_weight 0.08
--height_weight 0.02
```

Reranker inputs:

- retrieval embedding similarity
- OSM footprint aspect ratio
- candidate OBJ footprint aspect ratio
- OSM height-to-footprint ratio
- candidate OBJ height-to-footprint ratio

Reranker score:

```text
rerank_score = retrieval_score
               - aspect_weight * abs(log(candidate_aspect / target_aspect))
               - height_weight * abs(log(candidate_height_ratio / target_height_ratio))
```

Reranked smoke command:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/osm_hunyuan_pipeline_smoke.py \
  --osm_json outputs/osm_pipeline_smoke/osm_input.json \
  --out_dir outputs/osm_pipeline_rerank_smoke \
  --limit 4 \
  --model mini \
  --retrieval_policy rerank \
  --retrieval_top_k 5
```

Reranked map/choice sheet command:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/osm_pipeline_map_choices.py \
  --osm_json outputs/osm_pipeline_smoke/osm_input.json \
  --pipeline_log outputs/osm_pipeline_rerank_smoke/osm_hunyuan_scene.log.json \
  --out_dir outputs/osm_pipeline_rerank_smoke \
  --top_k 5
```

Reranked outputs:

```text
outputs/osm_pipeline_rerank_smoke/osm_hunyuan_scene.obj
outputs/osm_pipeline_rerank_smoke/osm_hunyuan_pipeline_metrics.csv
outputs/osm_pipeline_rerank_smoke/osm_retrieval_rerank_choices.csv
outputs/osm_pipeline_rerank_smoke/osm_map_choices_sheet.png
```

Selection changes versus similarity-only top-1:

```text
OSM_302408405: kept mesh7919
OSM_860916350: changed mesh9098 -> mesh4188
OSM_71497515:  changed mesh4551 -> mesh9657
OSM_116427240: changed mesh4551 -> mesh5957
```

Interpretation:

- This is still retrieval-conditioned generation, not a trained generative model.
- But it creates the decision structure we need for true generation: multiple plausible candidates, scored conditioning signals, and explicit logs of what was chosen.
- The next model-training target should learn to generate or deform from this conditional candidate set, rather than blindly retrieving one mesh.
- Remaining issue: final placed meshes are still flattened when OSM lacks real height metadata. Height inference should be fixed before judging final city output quality.

## 2026-05-10 Codex Update: Area-Aware Height Recomposition

Fixed the next blocker without rerunning Hunyuan. The prior scene used raw OSM
height defaults, so large residential footprints were placed at only 7 m and
looked flattened. Added a recomposition script that reuses existing simplified
Hunyuan meshes and applies a height policy during placement.

Added:

```text
scripts/osm_recompose_height_policy.py
```

Command used:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/osm_recompose_height_policy.py \
  --osm_json outputs/osm_pipeline_smoke/osm_input.json \
  --pipeline_log outputs/osm_pipeline_rerank_smoke/osm_hunyuan_scene.log.json \
  --out_dir outputs/osm_pipeline_heightfix_smoke \
  --height_policy area_aware
```

Then regenerated the map/choice sheet:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/osm_pipeline_map_choices.py \
  --osm_json outputs/osm_pipeline_smoke/osm_input.json \
  --pipeline_log outputs/osm_pipeline_heightfix_smoke/osm_hunyuan_scene.log.json \
  --out_dir outputs/osm_pipeline_heightfix_smoke \
  --top_k 5
```

Height policy result:

```text
OSM_302408405: 7.00m -> 17.50m
OSM_860916350: 7.00m -> 17.50m
OSM_71497515:  7.00m -> 17.50m
OSM_116427240: 7.00m -> 14.00m
```

New outputs:

```text
outputs/osm_pipeline_heightfix_smoke/osm_hunyuan_scene.obj
outputs/osm_pipeline_heightfix_smoke/osm_hunyuan_scene.log.json
outputs/osm_pipeline_heightfix_smoke/height_policy_metrics.csv
outputs/osm_pipeline_heightfix_smoke/osm_map_choices_sheet.png
outputs/osm_pipeline_heightfix_smoke/osm_map_output_houses.png
outputs/osm_pipeline_heightfix_smoke/osm_hunyuan_scene_render.png
```

Interpretation:

- This improved final placement without changing retrieval or Hunyuan generation.
- It confirms the previous flattened outputs were mainly a height metadata problem.
- The area-aware policy is heuristic and should eventually be replaced by learned height/massing prediction from map context, but it is a practical baseline for now.
- Next generative step: build a training dataset of `(footprint, class, height/context, top-k candidate descriptors) -> selected/generated building latent`, using these rerank logs as supervision scaffolding.

## 2026-05-10 Codex Update: 12-Building Scaled OSM Run

Scaled the current pipeline from 4 selected buildings to 12 selected buildings
from the same OSM bbox.

Generation command:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/osm_hunyuan_pipeline_smoke.py \
  --osm_json outputs/osm_pipeline_smoke/osm_input.json \
  --out_dir outputs/osm_pipeline_rerank_12 \
  --limit 12 \
  --model mini \
  --retrieval_policy rerank \
  --retrieval_top_k 5
```

Height recomposition command:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/osm_recompose_height_policy.py \
  --osm_json outputs/osm_pipeline_smoke/osm_input.json \
  --pipeline_log outputs/osm_pipeline_rerank_12/osm_hunyuan_scene.log.json \
  --out_dir outputs/osm_pipeline_heightfix_12 \
  --height_policy area_aware
```

Map/choice sheet command:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/osm_pipeline_map_choices.py \
  --osm_json outputs/osm_pipeline_smoke/osm_input.json \
  --pipeline_log outputs/osm_pipeline_heightfix_12/osm_hunyuan_scene.log.json \
  --out_dir outputs/osm_pipeline_heightfix_12 \
  --top_k 5
```

Outputs:

```text
outputs/osm_pipeline_rerank_12/osm_hunyuan_scene.obj
outputs/osm_pipeline_rerank_12/osm_hunyuan_pipeline_metrics.csv
outputs/osm_pipeline_rerank_12/osm_retrieval_rerank_choices.csv
outputs/osm_pipeline_heightfix_12/osm_hunyuan_scene.obj
outputs/osm_pipeline_heightfix_12/height_policy_metrics.csv
outputs/osm_pipeline_heightfix_12/osm_map_choices_sheet.png
outputs/osm_pipeline_heightfix_12/osm_map_output_houses.png
```

Run summary:

```text
buildings generated: 12
classes: 11 RESIDENTIALhouse, 1 COMMERCIALoffice_building
raw Hunyuan faces total: 9,332,706
simplified faces total: 600,000
scene OBJ size: 24 MB
total Hunyuan generation time: 164.13 s
average Hunyuan generation time: 13.68 s/building
map/choice sheet size: 2048 x 3948
```

Selected meshes:

```text
OSM_302408405  RESIDENTIALhouse             -> RESIDENTIALhouse_mesh7919
OSM_860916350  RESIDENTIALhouse             -> RESIDENTIALhouse_mesh4188
OSM_71497515   RESIDENTIALhouse             -> RESIDENTIALhouse_mesh9657
OSM_116427240  RESIDENTIALhouse             -> RESIDENTIALhouse_mesh5957
OSM_159752237  RESIDENTIALhouse             -> RESIDENTIALhouse_mesh7024
OSM_302408404  RESIDENTIALhouse             -> RESIDENTIALhouse_mesh8294
OSM_116427239  RESIDENTIALhouse             -> RESIDENTIALhouse_mesh8733
OSM_955677965  RESIDENTIALhouse             -> RESIDENTIALhouse_mesh2592
OSM_71497499   COMMERCIALoffice_building    -> COMMERCIALoffice_building_mesh1616
OSM_1093265929 RESIDENTIALhouse             -> RESIDENTIALhouse_mesh0230
OSM_955889856  RESIDENTIALhouse             -> RESIDENTIALhouse_mesh4151
OSM_955889852  RESIDENTIALhouse             -> RESIDENTIALhouse_mesh9657
```

Height policy changes:

```text
large residential footprints: 7.0m -> 14.0-17.5m
commercial footprint: 14.0m -> 17.5m
smaller residential footprints: 7.0m -> 8.5-10.5m
```

Current failure modes exposed by the 12-building run:

- The map overlay is still a visual thumbnail overlay, not an exact rendered top-down scene.
- Some Hunyuan outputs remain weak or flat even after height correction, so we need an output-quality filter/reranker.
- The pipeline has no neighborhood style coherence yet; buildings are plausible independently but not controlled as a consistent district.

Next implementation target:

- Add post-generation mesh quality metrics and filtering: vertical extent ratio, face count sanity, bounding-box collapse detection, and render/thumbnail quality flags.

## 2026-05-10 Codex Update: Post-Generation Quality Audit

Added a non-destructive quality audit pass for generated OSM building meshes.
It inspects raw Hunyuan GLBs, simplified OBJs, placed output renders, and final
placement geometry to flag collapsed or weak assets before they are accepted
into the scene or used as future training data.

Added:

```text
scripts/osm_generation_quality_audit.py
```

Command used:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/osm_generation_quality_audit.py \
  --osm_json outputs/osm_pipeline_smoke/osm_input.json \
  --pipeline_log outputs/osm_pipeline_heightfix_12/osm_hunyuan_scene.log.json \
  --out_dir outputs/osm_pipeline_quality_12
```

Outputs:

```text
outputs/osm_pipeline_quality_12/generation_quality_audit.csv
outputs/osm_pipeline_quality_12/generation_quality_audit_sheet.png
outputs/osm_pipeline_quality_12/generation_quality_summary.json
```

Audit summary:

```text
count: 12
pass: 10
warn: 0
fail: 2
```

Failed assets:

```text
OSM_159752237 / RESIDENTIALhouse_mesh7024
  flags: raw_flat | raw_high_flatness
  interpretation: Hunyuan generated a too-flat asset before placement.

OSM_302408404 / RESIDENTIALhouse_mesh8294
  flags: placed_flat
  interpretation: raw mesh is acceptable, but placement into the OSM footprint/height produces an overly flat result.
```

Quality metrics now recorded:

- raw/simplified vertex and face counts
- raw/simplified/placed height-to-footprint ratios
- flatness ratios
- connected component counts
- watertight flags
- raw and placed render ink coverage
- pass/warn/fail status and flags

Interpretation:

- The audit confirms that most outputs are usable after rerank + heightfix.
- It also gives us the first automated rejection signal for weak generated assets.
- We are not wiring this into fallback/regeneration yet; that quality policy will be tuned later.
- Immediate next action is to turn accepted OSM pipeline runs into structured training records so the project can move from retrieval-first generation toward a truly generative model.

## 2026-05-10 Codex Update: OSM Generation Dataset Scaffold

Added a dataset-building step that converts a completed OSM pipeline run into
structured records suitable for future conditional training or fine-tuning.
This keeps the retrieval/Hunyuan outputs, footprint geometry, candidate choices,
height policy, and quality labels together in one repeatable artifact.

Added:

```text
scripts/build_osm_generation_dataset.py
```

Command used:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/build_osm_generation_dataset.py \
  --osm_json outputs/osm_pipeline_smoke/osm_input.json \
  --pipeline_log outputs/osm_pipeline_heightfix_12/osm_hunyuan_scene.log.json \
  --quality_csv outputs/osm_pipeline_quality_12/generation_quality_audit.csv \
  --out_dir outputs/osm_generation_dataset_12 \
  --split smoke12
```

Outputs:

```text
outputs/osm_generation_dataset_12/smoke12_records.jsonl
outputs/osm_generation_dataset_12/smoke12_index.csv
outputs/osm_generation_dataset_12/smoke12_footprint_masks.npz
outputs/osm_generation_dataset_12/smoke12_summary.json
outputs/osm_generation_dataset_12/footprint_masks/*.npy
```

Dataset summary:

```text
dataset_version: osm_generation_scaffold_v1
split: smoke12
records: 12
positive examples: 10
negative/quality-failed examples: 2
classes: 11 RESIDENTIALhouse, 1 COMMERCIALoffice_building
```

Each JSONL record now includes:

- OSM id, class, top-level category, footprint polygon, centroid, area, height, and height source
- normalized footprint mask as `.npy` plus the packed `.npz` collection
- geometry features such as area, perimeter, compactness, aspect ratio, orientation, and vertex count
- retrieval policy, top-k candidates, raw retrieval scores, rerank scores, and selected candidate id
- paths to retrieved mesh, rendered conditioning PNG, Hunyuan raw GLB, simplified OBJ, placed render, and final scene OBJ
- generation metrics such as raw/simplified face counts, generation time, simplification ratio, and placement scale
- quality status/flags from the audit pass
- training-use labels: `include_as_positive` and `include_as_negative`

Why this matters:

- This is the first bridge from a retrieval-first demo into a trainable data pipeline.
- Good outputs can become positive examples for a conditional generator.
- Failed outputs remain useful as negative examples or filtering data instead of being thrown away.
- The artifact is repeatable across more OSM bounding boxes, so scaling now means collecting more neighborhoods and appending compatible records.

Recommended next pipeline step:

- Run the same OSM-to-dataset sequence across multiple nearby bounding boxes.
- Build a larger mixed-quality corpus before starting model training.
- Keep SDF only in the footprint/massing correction role for now; the training scaffold should target footprint-conditioned building generation and placement metadata first.

## 2026-05-10 Codex Update: First Multi-Tile Dataset Expansion

Expanded the dataset beyond the original 12-building smoke run by extracting an
adjacent north OSM tile, generating the largest 8 buildings, applying the same
area-aware height recomposition, auditing outputs, exporting a dataset shard,
and merging both shards into one corpus.

North tile extraction:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scene/extract_osm.py \
  --bbox 40.4250 -86.9075 40.4266 -86.9050 \
  -o outputs/osm_pipeline_tile_north/osm_input.json
```

OSM extraction result:

```text
buildings: 25
roads: 4
classes: 25 RESIDENTIALhouse
height stats: min=7.0m, median=7.0m, max=10.5m
area stats: min=23.8m2, median=232.5m2, max=1978.3m2
```

North tile generation command:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/osm_hunyuan_pipeline_smoke.py \
  --osm_json outputs/osm_pipeline_tile_north/osm_input.json \
  --out_dir outputs/osm_pipeline_tile_north_gen8 \
  --limit 8 \
  --retrieval_policy rerank \
  --retrieval_top_k 5 \
  --model mini \
  --steps 50 \
  --target_faces 50000 \
  --device cuda
```

North tile generation outputs:

```text
outputs/osm_pipeline_tile_north_gen8/osm_hunyuan_scene.obj
outputs/osm_pipeline_tile_north_gen8/osm_hunyuan_scene.log.json
outputs/osm_pipeline_tile_north_gen8/osm_hunyuan_pipeline_metrics.csv
outputs/osm_pipeline_tile_north_gen8/osm_retrieval_rerank_choices.csv
outputs/osm_pipeline_tile_north_gen8/osm_hunyuan_pipeline_sheet.png
```

Area-aware height recomposition:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/osm_recompose_height_policy.py \
  --osm_json outputs/osm_pipeline_tile_north/osm_input.json \
  --pipeline_log outputs/osm_pipeline_tile_north_gen8/osm_hunyuan_scene.log.json \
  --out_dir outputs/osm_pipeline_tile_north_heightfix8 \
  --height_policy area_aware
```

Height changes:

```text
OSM_71497515: 7.00m -> 17.50m
OSM_159752237: 7.00m -> 14.00m
OSM_764679346: 7.00m -> 14.00m
OSM_955889856: 7.00m -> 10.50m
OSM_860910546: 7.00m -> 10.50m
OSM_860910542: 7.00m -> 10.50m
OSM_860910545: 7.00m -> 10.50m
OSM_860910541: 7.00m -> 8.50m
```

Quality audit result:

```text
count: 8
pass: 7
fail: 1
failed asset: OSM_159752237 / RESIDENTIALhouse_mesh7024
flags: raw_flat | raw_high_flatness
```

North tile visualization outputs:

```text
outputs/osm_pipeline_tile_north_heightfix8/osm_map_input.png
outputs/osm_pipeline_tile_north_heightfix8/osm_map_selected.png
outputs/osm_pipeline_tile_north_heightfix8/osm_map_output_houses.png
outputs/osm_pipeline_tile_north_heightfix8/osm_map_choices_sheet.png
```

North dataset shard:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/build_osm_generation_dataset.py \
  --osm_json outputs/osm_pipeline_tile_north/osm_input.json \
  --pipeline_log outputs/osm_pipeline_tile_north_heightfix8/osm_hunyuan_scene.log.json \
  --quality_csv outputs/osm_pipeline_tile_north_quality8/generation_quality_audit.csv \
  --out_dir outputs/osm_generation_dataset_tile_north8 \
  --split north8
```

North shard summary:

```text
records: 8
positive examples: 7
negative/quality-failed examples: 1
classes: 8 RESIDENTIALhouse
```

Added:

```text
scripts/merge_osm_generation_datasets.py
```

Corpus merge command:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/merge_osm_generation_datasets.py \
  --dataset outputs/osm_generation_dataset_12/smoke12_records.jsonl \
  --dataset outputs/osm_generation_dataset_tile_north8/north8_records.jsonl \
  --out_dir outputs/osm_generation_dataset_corpus_v1 \
  --name campus_lafayette_v1
```

Merged corpus outputs:

```text
outputs/osm_generation_dataset_corpus_v1/campus_lafayette_v1_records.jsonl
outputs/osm_generation_dataset_corpus_v1/campus_lafayette_v1_index.csv
outputs/osm_generation_dataset_corpus_v1/campus_lafayette_v1_footprint_masks.npz
outputs/osm_generation_dataset_corpus_v1/campus_lafayette_v1_summary.json
```

Merged corpus summary:

```text
records: 20
positive examples: 17
negative/quality-failed examples: 3
classes: 19 RESIDENTIALhouse, 1 COMMERCIALoffice_building
splits: smoke12=12, north8=8
```

Current interpretation:

- The corpus is still too small for real model training, but the data machinery is now multi-tile and repeatable.
- We have positive and negative labels that can support filtering, ranking, or later preference-style training.
- The next useful scale target is 5-10 nearby OSM tiles with a controlled per-tile limit, then a first lightweight conditional baseline over footprint masks and class/height metadata.

## 2026-05-10 Codex Update: Batch Runner, Four More Tiles, Corpus Diagnostics

Implemented the next three pipeline steps:

1. Added a repeatable multi-tile batch runner.
2. Ran four additional nearby OSM tiles with a 6-building cap per tile.
3. Added corpus diagnostics and ran them on the expanded corpus.

Added:

```text
scripts/run_osm_generation_batch.py
scripts/osm_generation_corpus_diagnostics.py
```

Batch command:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/run_osm_generation_batch.py \
  --tile east:40.4234,-86.9050,40.4250,-86.9025 \
  --tile west:40.4234,-86.9100,40.4250,-86.9075 \
  --tile south:40.4218,-86.9075,40.4234,-86.9050 \
  --tile northeast:40.4250,-86.9050,40.4266,-86.9025 \
  --limit 6 \
  --base_out outputs/osm_batch_lafayette_v2 \
  --existing_dataset outputs/osm_generation_dataset_12/smoke12_records.jsonl \
  --existing_dataset outputs/osm_generation_dataset_tile_north8/north8_records.jsonl \
  --corpus_out outputs/osm_generation_dataset_corpus_v2 \
  --corpus_name campus_lafayette_v2
```

The batch runner executed this sequence for each new tile:

```text
scene/extract_osm.py
scripts/osm_hunyuan_pipeline_smoke.py
scripts/osm_recompose_height_policy.py
scripts/osm_generation_quality_audit.py
scripts/osm_pipeline_map_choices.py
scripts/build_osm_generation_dataset.py
scripts/merge_osm_generation_datasets.py
```

New tile results:

```text
east6:
  OSM extract: 13 buildings, 9 road segments, all RESIDENTIALhouse
  dataset: 6 records, 6 positive, 0 negative

west6:
  OSM extract: 28 buildings, 4 roads
  classes in extract: 23 RESIDENTIALhouse, 4 COMMERCIALoffice_building, 1 PUBLICschool_building
  dataset: 6 records, 5 positive, 1 negative
  added one PUBLICschool_building and one COMMERCIALoffice_building example

south6:
  OSM extract: 17 buildings, 8 roads
  classes in extract: 15 RESIDENTIALhouse, 2 COMMERCIALoffice_building
  dataset: 6 records, 4 positive, 2 negative

northeast6:
  OSM extract: 11 buildings, 2 roads, all RESIDENTIALhouse
  dataset: 6 records, 5 positive, 1 negative
```

Expanded merged corpus:

```text
outputs/osm_generation_dataset_corpus_v2/campus_lafayette_v2_records.jsonl
outputs/osm_generation_dataset_corpus_v2/campus_lafayette_v2_index.csv
outputs/osm_generation_dataset_corpus_v2/campus_lafayette_v2_footprint_masks.npz
outputs/osm_generation_dataset_corpus_v2/campus_lafayette_v2_summary.json
```

Corpus v2 summary:

```text
records: 44
positive examples: 37
negative/quality-failed examples: 7
positive rate: 0.841
classes:
  RESIDENTIALhouse: 41
  COMMERCIALoffice_building: 2
  PUBLICschool_building: 1
splits:
  smoke12: 12
  north8: 8
  east6: 6
  west6: 6
  south6: 6
  northeast6: 6
```

Diagnostics command:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/osm_generation_corpus_diagnostics.py \
  --records outputs/osm_generation_dataset_corpus_v2/campus_lafayette_v2_records.jsonl \
  --out_dir outputs/osm_generation_dataset_corpus_v2/diagnostics
```

Diagnostics outputs:

```text
outputs/osm_generation_dataset_corpus_v2/diagnostics/diagnostics_summary.json
outputs/osm_generation_dataset_corpus_v2/diagnostics/diagnostics_report.md
outputs/osm_generation_dataset_corpus_v2/diagnostics/class_counts.csv
outputs/osm_generation_dataset_corpus_v2/diagnostics/split_counts.csv
outputs/osm_generation_dataset_corpus_v2/diagnostics/candidate_reuse.csv
outputs/osm_generation_dataset_corpus_v2/diagnostics/candidate_reuse_by_class.csv
outputs/osm_generation_dataset_corpus_v2/diagnostics/failure_by_candidate.csv
outputs/osm_generation_dataset_corpus_v2/diagnostics/failure_flags.csv
```

Key diagnostics:

```text
unique selected candidates: 22
most reused candidates:
  RESIDENTIALhouse_mesh7919: 5
  RESIDENTIALhouse_mesh9657: 5
  RESIDENTIALhouse_mesh7024: 4
  RESIDENTIALhouse_mesh4151: 3
  RESIDENTIALhouse_mesh4858: 3

failed candidates:
  RESIDENTIALhouse_mesh7024: 4
  RESIDENTIALhouse_mesh8294: 1
  RESIDENTIALhouse_mesh9657: 1
  RESIDENTIALhouse_mesh0166: 1

failure flags:
  raw_flat: 4
  raw_high_flatness: 4
  placed_flat: 3
```

Distribution snapshot:

```text
area_m2 median: 1095.33
area_m2 range: 345.87 - 6773.37
height_m median: 14.0
height_m range: 8.5 - 56.0
bbox_aspect median: 1.50
raw_faces median: 864,783
simplified_faces: fixed at 50,000
```

Interpretation:

- The data pipeline is now repeatable and multi-tile.
- Corpus size increased from 20 to 44 records.
- Quality labels are useful: `RESIDENTIALhouse_mesh7024` is repeatedly associated with raw-flat failures.
- Class balance is still heavily residential; future OSM tiles should intentionally target commercial/public areas.
- Before model training, the next engineering improvement should be corpus-aware sampling/filtering so repeated bad candidate ids can be downweighted or excluded from positive training examples.

## 2026-05-10 Codex Update: Corpus-Aware Selector Baseline

Implemented the first measurable model baseline over the expanded corpus.
This is not a mesh generator yet. It is a lightweight candidate selector that
learns to score each retrieved top-k candidate for a footprint using footprint,
class, height, rerank, and candidate geometry features.

Added:

```text
scripts/train_osm_candidate_selector_baseline.py
```

Command:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/train_osm_candidate_selector_baseline.py \
  --records outputs/osm_generation_dataset_corpus_v2/campus_lafayette_v2_records.jsonl \
  --out_dir outputs/osm_candidate_selector_baseline_v1
```

Outputs:

```text
outputs/osm_candidate_selector_baseline_v1/filtered_train_records.jsonl
outputs/osm_candidate_selector_baseline_v1/filtered_val_records.jsonl
outputs/osm_candidate_selector_baseline_v1/negative_records.jsonl
outputs/osm_candidate_selector_baseline_v1/bad_candidates.json
outputs/osm_candidate_selector_baseline_v1/candidate_selector_model.pkl
outputs/osm_candidate_selector_baseline_v1/candidate_selector_metrics.json
outputs/osm_candidate_selector_baseline_v1/candidate_selector_predictions.csv
outputs/osm_candidate_selector_baseline_v1/feature_importance.csv
outputs/osm_candidate_selector_baseline_v1/candidate_selector_report.md
```

Filtering result:

```text
positive records before bad-candidate filter: 37
positive records after bad-candidate filter: 37
negative records retained: 7
train records: 22
validation records: 15
bad candidate ids:
  RESIDENTIALhouse_mesh7024: 4 failures
```

`RESIDENTIALhouse_mesh7024` is flagged as a corpus-level bad candidate because
it has repeated raw-flat failures. The positive set did not currently contain
that candidate after the quality filter, so no positive records were removed.

Validation metrics:

```text
rerank top-1 accuracy: 1.000
filtered rerank top-1 accuracy: 1.000
learned selector top-1 accuracy: 0.800
filtered learned selector top-1 accuracy: 0.800

candidate-row validation average precision: 0.841
candidate-row validation ROC AUC: 0.954
candidate-row validation accuracy at 0.5: 0.893
```

Top learned features:

```text
candidate_rank: 0.1336
candidate_height_ratio: 0.1187
rerank_score: 0.0861
height_ratio_log_error: 0.0662
retrieval_score: 0.0647
height_penalty: 0.0603
aspect_log_error: 0.0581
aspect_penalty: 0.0565
candidate_aspect: 0.0521
log_candidate_verts: 0.0516
```

Interpretation:

- The baseline proves the corpus can now be exported into filtered train/val/negative splits.
- The learned model is behaving sensibly at the candidate-row level, but it does not beat rerank top-1 on validation.
- That is expected: the current positive labels are generated by the rerank policy, so the supervised target mostly teaches the model to imitate rerank.
- To improve beyond rerank, the next target should be post-generation success or preference labels, not only the rerank-selected candidate id.
- The most practical next step is to train a quality/success predictor over generated outputs and use it to rerank candidates before expensive Hunyuan generation.

## 2026-05-11 Codex Update: Quality-Aware Generation Success Predictor

Added the first model that uses post-generation audit labels directly. This is
the next research-relevant pipeline addition: instead of only learning to
imitate retrieval/rerank, it predicts whether a retrieved candidate is likely
to produce a passing Hunyuan output for a given OSM footprint.

Added:

```text
scripts/osm_candidate_quality_features.py
scripts/train_osm_generation_success_predictor.py
```

Updated:

```text
scripts/osm_hunyuan_pipeline_smoke.py
```

The OSM pipeline now supports quality-aware candidate selection:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/osm_hunyuan_pipeline_smoke.py \
  --osm_json <tile>/osm_input.json \
  --out_dir <tile>/quality_gen \
  --limit 6 \
  --retrieval_policy quality \
  --quality_model outputs/osm_generation_success_predictor_v1/generation_success_model.pkl \
  --quality_weight 0.20 \
  --quality_bad_candidate_penalty 1.0 \
  --model mini \
  --device cuda
```

Training command:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/train_osm_generation_success_predictor.py \
  --records outputs/osm_generation_dataset_corpus_v2/campus_lafayette_v2_records.jsonl \
  --out_dir outputs/osm_generation_success_predictor_v1 \
  --quality_weight 0.20
```

Outputs:

```text
outputs/osm_generation_success_predictor_v1/generation_success_model.pkl
outputs/osm_generation_success_predictor_v1/generation_success_metrics.json
outputs/osm_generation_success_predictor_v1/generation_success_report.md
outputs/osm_generation_success_predictor_v1/quality_aware_counterfactual.csv
outputs/osm_generation_success_predictor_v1/feature_importance.csv
```

Training data:

```text
records: 44
pass: 37
fail: 7
train records: 27
validation records: 17
bad candidate ids:
  RESIDENTIALhouse_mesh7024: 4 failures
```

Validation metrics:

```text
average precision: 0.996
ROC AUC: 0.967
accuracy at 0.5: 0.941
```

Counterfactual quality-aware rerank on the existing corpus:

```text
changed choices: 12 / 44
bad selected candidates in corpus: 4
bad candidates selected by quality-aware policy: 0
```

Top features:

```text
bbox_depth_m: 0.1303
height_to_bbox_max: 0.1176
candidate_aspect: 0.0875
target_height_ratio: 0.0806
candidate_height_ratio: 0.0748
height_to_sqrt_area: 0.0603
retrieval_score: 0.0567
compactness: 0.0483
log_area_m2: 0.0470
candidate_rank: 0.0424
```

Interpretation:

- This is the first quality-aware learned component in the pipeline.
- It uses Hunyuan/audit outcomes, not just retrieval labels.
- It gives a measurable claim to test in the next generation batch: quality-aware reranking should reduce flat/collapsed outputs.
- The validation score is optimistic because the corpus is still small, but the counterfactual result is useful: it would avoid all known repeated bad `RESIDENTIALhouse_mesh7024` selections.
- Next empirical step: run a fresh OSM tile twice, once with geometry rerank and once with quality-aware rerank, then compare audit pass rate and choice changes.

## 2026-05-11 Codex Update: Fresh Quality-Rerank A/B Generation Test

Added an empirical A/B runner for the next pipeline step:

```text
scripts/run_quality_rerank_ab_test.py
```

The script runs a complete paired comparison:

```text
OSM extract -> geometry rerank generation -> height policy -> audit -> map sheets
OSM extract -> quality rerank generation -> height policy -> audit -> map sheets
comparison CSV -> summary JSON -> markdown report
```

Executed on a fresh northwest Lafayette tile with four buildings:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/run_quality_rerank_ab_test.py \
  --bbox 40.4250 -86.9100 40.4266 -86.9075 \
  --out_dir outputs/quality_rerank_ab_northwest4 \
  --limit 4 \
  --quality_model outputs/osm_generation_success_predictor_v1/generation_success_model.pkl \
  --quality_weight 0.20 \
  --quality_bad_candidate_penalty 1.0 \
  --model mini \
  --steps 50 \
  --device cuda
```

Fresh OSM extraction summary:

```text
buildings: 20
roads: 4
classes:
  RESIDENTIALhouse: 17
  RELIGIOUSchurch: 1
  PUBLICoffice_building: 1
  COMMERCIALoffice_building: 1
height min/median/max: 7.0 / 7.0 / 14.0 m
area min/median/max: 59.7 / 507.5 / 6773.2 m2
```

Outputs:

```text
outputs/quality_rerank_ab_northwest4/ab_report.md
outputs/quality_rerank_ab_northwest4/ab_summary.json
outputs/quality_rerank_ab_northwest4/ab_comparison.csv
outputs/quality_rerank_ab_northwest4/geometry_rerank/
outputs/quality_rerank_ab_northwest4/quality_rerank/
```

A/B result:

```text
geometry rerank: 3 pass, 1 fail
quality rerank: 3 pass, 1 fail
choice changes: 1 / 4
shared failure flag: placed_flat
```

Changed choice:

```text
OSM_71497515:
  geometry candidate: RESIDENTIALhouse_mesh9657
  quality candidate:  RESIDENTIALhouse_mesh5957
  geometry status: pass
  quality status:  pass
```

Unchanged failure:

```text
OSM_158307997:
  candidate: RESIDENTIALhouse_mesh9657
  status: fail
  flag: placed_flat
```

Interpretation:

- The quality-aware policy is wired into the full generation path and can change retrieval decisions before Hunyuan generation.
- On this small four-building tile it did not improve the pass rate, because the one changed selection was already a passing geometry-rerank case.
- The remaining failure was unchanged by the quality policy, so it likely needs a placement/height/audit-side fix or more diverse candidate alternatives, not only a learned rerank weight.
- CUDA execution was confirmed during the run: diffusion sampled at roughly 31 steps/sec after model load, while volume decoding and mesh postprocessing remained the main runtime cost.
- This is enough evidence to keep the quality predictor as an optional reranking component, but not enough to claim it improves generation success yet. The next research step should be a larger A/B sweep across multiple OSM tiles and more than four buildings.

Verification:

```text
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python -m py_compile \
  scripts/run_quality_rerank_ab_test.py \
  scripts/osm_hunyuan_pipeline_smoke.py
```

## 2026-05-11 Codex Update: 4-Tile Quality-Rerank A/B Sweep

Added a multi-tile sweep runner around the per-tile A/B script:

```text
scripts/run_quality_rerank_ab_sweep.py
```

The runner executes the same paired comparison over repeated OSM bboxes and
writes aggregate outputs:

```text
outputs/quality_rerank_ab_sweep_lafayette4x4/sweep_report.md
outputs/quality_rerank_ab_sweep_lafayette4x4/sweep_summary.json
outputs/quality_rerank_ab_sweep_lafayette4x4/sweep_tile_summary.csv
outputs/quality_rerank_ab_sweep_lafayette4x4/sweep_ab_comparison.csv
```

Command used:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/run_quality_rerank_ab_sweep.py \
  --tile east:40.4234,-86.9050,40.4250,-86.9025 \
  --tile west:40.4234,-86.9100,40.4250,-86.9075 \
  --tile south:40.4218,-86.9075,40.4234,-86.9050 \
  --tile northeast:40.4250,-86.9050,40.4266,-86.9025 \
  --out_dir outputs/quality_rerank_ab_sweep_lafayette4x4 \
  --limit 4 \
  --quality_model outputs/osm_generation_success_predictor_v1/generation_success_model.pkl \
  --quality_weight 0.20 \
  --quality_bad_candidate_penalty 1.0 \
  --model mini \
  --steps 50 \
  --device cuda
```

Aggregate result:

```text
tiles: 4
buildings generated per arm: 16
geometry rerank: 13 pass, 3 fail, pass rate 81.25%
quality rerank:  14 pass, 2 fail, pass rate 87.50%
choice changes: 5 / 16, change rate 31.25%
net improvement: +1 pass, no observed regressions
```

Per-tile result:

```text
east:      geometry 4/4, quality 4/4, changed 1
west:      geometry 3/4, quality 3/4, changed 1
south:     geometry 3/4, quality 3/4, changed 2
northeast: geometry 3/4, quality 4/4, changed 1
```

Changed choices:

```text
east OSM_955677965:
  RESIDENTIALhouse_mesh2592 -> RESIDENTIALhouse_mesh3714
  pass -> pass

west OSM_71497515:
  RESIDENTIALhouse_mesh9657 -> RESIDENTIALhouse_mesh5957
  pass -> pass

south OSM_1424680532:
  RESIDENTIALhouse_mesh0166 -> RESIDENTIALhouse_mesh0642
  fail -> fail

south OSM_843163261:
  RESIDENTIALhouse_mesh0166 -> RESIDENTIALhouse_mesh6424
  pass -> pass

northeast OSM_581739532:
  RESIDENTIALhouse_mesh7024 -> RESIDENTIALhouse_mesh5581
  fail -> pass
```

Important interpretation:

- The quality-aware reranker now has a real positive result beyond counterfactual scoring: it improved one live Hunyuan generation case in the northeast tile.
- The improved case is exactly the pattern the model was designed to catch: geometry rerank chose known bad `RESIDENTIALhouse_mesh7024`, producing `raw_flat|raw_high_flatness`; quality rerank avoided it and passed.
- No changed choice regressed from pass to fail in this 16-building sweep.
- The remaining two failures are not solved by the current quality reranker:
  - west `OSM_158307997`: unchanged `RESIDENTIALhouse_mesh9657`, `placed_flat`
  - south `OSM_1424680532`: changed candidate, but still `placed_flat`
- That points to the next pipeline target: add fallback or repair logic for `placed_flat` cases. The predictor can avoid known bad candidates, but some failures are footprint/placement/height interactions rather than candidate-id failures.
- CUDA behavior stayed healthy throughout the sweep. Diffusion sampling ran around 31 steps/sec after model load; volume decoding and mesh I/O/postprocessing remained the dominant cost.

Research status after this sweep:

- Retrieval + Hunyuan + heightfix + quality audit is stable enough for small multi-tile experiments.
- Quality-aware candidate selection is empirically useful, but only modestly: +6.25 percentage points on this small sweep.
- The strongest next contribution is not another rerank tweak; it is a post-generation recovery stage:
  1. detect `raw_flat` or `placed_flat`,
  2. try the next quality-ranked candidate,
  3. fall back to the retrieved mesh if Hunyuan remains flat,
  4. record the fallback path in the dataset so training labels distinguish retrieval success, Hunyuan success, and fallback success.

Verification:

```text
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python -m py_compile \
  scripts/run_quality_rerank_ab_sweep.py \
  scripts/run_quality_rerank_ab_test.py
```

## 2026-05-11 Codex Update: Visual Quality-Rerank Sweep Outputs

Added a visual report generator for the 4-tile quality-rerank A/B sweep:

```text
scripts/make_quality_rerank_visual_report.py
```

Command used:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/make_quality_rerank_visual_report.py \
  --sweep_dir outputs/quality_rerank_ab_sweep_lafayette4x4
```

New visual outputs:

```text
outputs/quality_rerank_ab_sweep_lafayette4x4/visual_summary/quality_rerank_sweep_overview.png
outputs/quality_rerank_ab_sweep_lafayette4x4/visual_summary/quality_rerank_changed_choices.png
outputs/quality_rerank_ab_sweep_lafayette4x4/visual_summary/quality_rerank_visual_report.md
```

Image sizes:

```text
quality_rerank_sweep_overview.png: 1472 x 1096
quality_rerank_changed_choices.png: 1402 x 1406
```

What the sheets show:

- `quality_rerank_sweep_overview.png` compares each tile's geometry-rerank output map, quality-rerank output map, and audit sheets side by side.
- `quality_rerank_changed_choices.png` focuses only on the five cases where quality rerank selected a different retrieval candidate, including conditioning inputs and placed outputs.

Interpretation:

- The visual sheets make the result easier to inspect than the CSV alone.
- The same conclusion still holds: quality-aware rerank improved one live case by avoiding `RESIDENTIALhouse_mesh7024`, produced no observed pass-to-fail regressions in this sweep, and left two `placed_flat` cases that require a recovery/fallback stage.

Verification:

```text
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python -m py_compile \
  scripts/make_quality_rerank_visual_report.py
```

## 2026-05-11 Codex Update: Failed-Generation Recovery Stage

Added a post-generation recovery script:

```text
scripts/osm_recover_failed_generations.py
```

Purpose:

- read a completed height-fixed OSM/Hunyuan pipeline log plus quality audit;
- keep passing buildings unchanged;
- for failed rows, try the next ranked retrieval candidates through Hunyuan;
- if retries fail, fall back to direct retrieved-OBJ placement;
- export a recovered scene log, recovered OBJ scene, recovery CSV, summary JSON, and before/after visual sheet.

Verification:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python -m py_compile \
  scripts/osm_recover_failed_generations.py
```

Ran recovery on the two remaining quality-rerank failures from the 4-tile
Lafayette sweep.

West tile command:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/osm_recover_failed_generations.py \
  --osm_json outputs/quality_rerank_ab_sweep_lafayette4x4/west/osm_input.json \
  --pipeline_log outputs/quality_rerank_ab_sweep_lafayette4x4/west/quality_rerank/heightfix/osm_hunyuan_scene.log.json \
  --quality_csv outputs/quality_rerank_ab_sweep_lafayette4x4/west/quality_rerank/quality/generation_quality_audit.csv \
  --out_dir outputs/quality_rerank_ab_sweep_lafayette4x4/west/quality_rerank/recovery \
  --model mini \
  --steps 50 \
  --target_faces 50000 \
  --device cuda \
  --max_retry_candidates 2
```

South tile command:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/osm_recover_failed_generations.py \
  --osm_json outputs/quality_rerank_ab_sweep_lafayette4x4/south/osm_input.json \
  --pipeline_log outputs/quality_rerank_ab_sweep_lafayette4x4/south/quality_rerank/heightfix/osm_hunyuan_scene.log.json \
  --quality_csv outputs/quality_rerank_ab_sweep_lafayette4x4/south/quality_rerank/quality/generation_quality_audit.csv \
  --out_dir outputs/quality_rerank_ab_sweep_lafayette4x4/south/quality_rerank/recovery \
  --model mini \
  --steps 50 \
  --target_faces 50000 \
  --device cuda \
  --max_retry_candidates 2
```

Recovery outputs:

```text
outputs/quality_rerank_ab_sweep_lafayette4x4/west/quality_rerank/recovery/osm_hunyuan_scene.obj
outputs/quality_rerank_ab_sweep_lafayette4x4/west/quality_rerank/recovery/osm_hunyuan_scene.log.json
outputs/quality_rerank_ab_sweep_lafayette4x4/west/quality_rerank/recovery/recovery_report.csv
outputs/quality_rerank_ab_sweep_lafayette4x4/west/quality_rerank/recovery/recovery_summary.json
outputs/quality_rerank_ab_sweep_lafayette4x4/west/quality_rerank/recovery/recovery_sheet.png
outputs/quality_rerank_ab_sweep_lafayette4x4/west/quality_rerank/recovery/osm_map_choices_sheet.png

outputs/quality_rerank_ab_sweep_lafayette4x4/south/quality_rerank/recovery/osm_hunyuan_scene.obj
outputs/quality_rerank_ab_sweep_lafayette4x4/south/quality_rerank/recovery/osm_hunyuan_scene.log.json
outputs/quality_rerank_ab_sweep_lafayette4x4/south/quality_rerank/recovery/recovery_report.csv
outputs/quality_rerank_ab_sweep_lafayette4x4/south/quality_rerank/recovery/recovery_summary.json
outputs/quality_rerank_ab_sweep_lafayette4x4/south/quality_rerank/recovery/recovery_sheet.png
outputs/quality_rerank_ab_sweep_lafayette4x4/south/quality_rerank/recovery/osm_map_choices_sheet.png
```

Post-recovery audit outputs:

```text
outputs/quality_rerank_ab_sweep_lafayette4x4/west/quality_rerank/recovery_quality/generation_quality_audit.csv
outputs/quality_rerank_ab_sweep_lafayette4x4/west/quality_rerank/recovery_quality/generation_quality_audit_sheet.png
outputs/quality_rerank_ab_sweep_lafayette4x4/west/quality_rerank/recovery_quality/generation_quality_summary.json

outputs/quality_rerank_ab_sweep_lafayette4x4/south/quality_rerank/recovery_quality/generation_quality_audit.csv
outputs/quality_rerank_ab_sweep_lafayette4x4/south/quality_rerank/recovery_quality/generation_quality_audit_sheet.png
outputs/quality_rerank_ab_sweep_lafayette4x4/south/quality_rerank/recovery_quality/generation_quality_summary.json
```

Empirical result:

```text
west:
  target failure: OSM_158307997 / RESIDENTIALhouse_mesh9657
  retry candidates: RESIDENTIALhouse_mesh1792, RESIDENTIALhouse_mesh4551
  both retries failed with placed_flat
  direct retrieved-OBJ fallback also failed the current audit
  recovered audit: 3 pass, 1 fail

south:
  target failure: OSM_1424680532 / RESIDENTIALhouse_mesh0642
  retry candidates: RESIDENTIALhouse_mesh2714, RESIDENTIALhouse_mesh9227
  both retries failed with placed_flat
  direct retrieved-OBJ fallback also failed the current audit
  recovered audit: 3 pass, 1 fail
```

Interpretation:

- The recovery script is now implemented and produces reusable scene/log/sheet artifacts.
- For these two cases, retrying candidates did not solve the failure mode.
- Direct retrieved-mesh fallback also did not solve the audit because the failure is still `placed_flat`.
- This means the remaining failures are not primarily "bad Hunyuan candidate" failures. They are footprint/placement/massing failures for large or awkward OSM footprints.
- The next pipeline target should be a placement/massing repair stage:
  1. detect footprints whose target aspect/area makes any single building too flat;
  2. split oversized footprints into multiple volumes, or
  3. use a taller massing proxy/footprint extrusion fallback instead of flattening one mesh across the whole polygon.

## 2026-05-11 Codex Update: Footprint-Conditioned Generative Proposal Branch

Started the next branch toward a more generative system. Instead of feeding
Hunyuan only retrieved-OBJ renders, the pipeline can now feed a generated
footprint/class/height-conditioned proposal image.

Added:

```text
scripts/osm_footprint_proposal_images.py
```

Updated:

```text
scripts/osm_hunyuan_pipeline_smoke.py
```

New pipeline option:

```text
--conditioning_source retrieved|proposal
```

Meaning:

- `retrieved`: existing behavior, render the selected BuildingNet OBJ and feed that image to Hunyuan.
- `proposal`: generate a deterministic footprint-conditioned building concept image and feed that to Hunyuan.

This is not the final learned generator. It is a deterministic baseline and
experiment harness. The important architectural change is that Hunyuan input
can now come from a generative proposal path rather than only from retrieval.

Generated proposal input sheet:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/osm_footprint_proposal_images.py \
  --osm_json outputs/quality_rerank_ab_sweep_lafayette4x4/east/osm_input.json \
  --out_dir outputs/osm_generative_proposal_east4 \
  --limit 4 \
  --image_size 384
```

Proposal visual output:

```text
outputs/osm_generative_proposal_east4/proposal_inputs_sheet.png
outputs/osm_generative_proposal_east4/proposal_inputs.csv
```

Ran a two-building end-to-end proposal-conditioned Hunyuan smoke:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/osm_hunyuan_pipeline_smoke.py \
  --osm_json outputs/quality_rerank_ab_sweep_lafayette4x4/east/osm_input.json \
  --out_dir outputs/osm_generative_proposal_east2 \
  --limit 2 \
  --retrieval_policy quality \
  --quality_model outputs/osm_generation_success_predictor_v1/generation_success_model.pkl \
  --quality_weight 0.20 \
  --quality_bad_candidate_penalty 1.0 \
  --conditioning_source proposal \
  --model mini \
  --steps 50 \
  --target_faces 50000 \
  --device cuda
```

Then applied the normal heightfix, audit, and visualization:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/osm_recompose_height_policy.py \
  --osm_json outputs/quality_rerank_ab_sweep_lafayette4x4/east/osm_input.json \
  --pipeline_log outputs/osm_generative_proposal_east2/osm_hunyuan_scene.log.json \
  --out_dir outputs/osm_generative_proposal_east2_heightfix \
  --height_policy area_aware

env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/osm_generation_quality_audit.py \
  --osm_json outputs/quality_rerank_ab_sweep_lafayette4x4/east/osm_input.json \
  --pipeline_log outputs/osm_generative_proposal_east2_heightfix/osm_hunyuan_scene.log.json \
  --out_dir outputs/osm_generative_proposal_east2_quality

env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/osm_pipeline_map_choices.py \
  --osm_json outputs/quality_rerank_ab_sweep_lafayette4x4/east/osm_input.json \
  --pipeline_log outputs/osm_generative_proposal_east2_heightfix/osm_hunyuan_scene.log.json \
  --out_dir outputs/osm_generative_proposal_east2_heightfix \
  --top_k 5 \
  --device cuda
```

Generative proposal outputs:

```text
outputs/osm_generative_proposal_east2/osm_hunyuan_pipeline_sheet.png
outputs/osm_generative_proposal_east2/osm_hunyuan_scene.obj
outputs/osm_generative_proposal_east2/osm_hunyuan_scene.log.json
outputs/osm_generative_proposal_east2_heightfix/osm_hunyuan_scene.obj
outputs/osm_generative_proposal_east2_heightfix/osm_map_choices_sheet.png
outputs/osm_generative_proposal_east2_quality/generation_quality_audit_sheet.png
outputs/osm_generative_proposal_east2_quality/generation_quality_summary.json
```

Audit result:

```text
count: 2
pass: 0
warn: 2
fail: 0

OSM_955677965: warn, many_raw_components | many_simplified_components
OSM_955677968: warn, many_raw_components | many_simplified_components
```

Important interpretation:

- The proposal-conditioned path runs end to end on CUDA.
- It avoided the previous `placed_flat` failure mode on this small east-sector smoke.
- The output is not clean enough yet: procedural proposal images caused Hunyuan to create many disconnected components.
- This is still a useful research step because the pipeline now has a real generative conditioning branch.
- The next improvement should replace the deterministic proposal drawer with a learned or diffusion-based footprint-to-building-image generator, while keeping retrieval as fallback.

Verification:

```text
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python -m py_compile \
  scripts/osm_footprint_proposal_images.py \
  scripts/osm_hunyuan_pipeline_smoke.py
```

## 2026-05-11 Codex Update: Clean Proposal Image Baseline

Improved the deterministic proposal generator so Hunyuan receives a cleaner
single-building conditioning image.

Updated:

```text
scripts/osm_footprint_proposal_images.py
scripts/osm_hunyuan_pipeline_smoke.py
```

New proposal controls:

```text
scripts/osm_footprint_proposal_images.py:
  --detail clean|detailed
  --include_footprint_inset

scripts/osm_hunyuan_pipeline_smoke.py:
  --proposal_detail clean|detailed
  --proposal_footprint_inset
```

Change:

- `clean` mode is now the default.
- The footprint inset is no longer drawn into Hunyuan's conditioning image by default.
- Window/detail marks are connected facade bands rather than many small isolated rectangles.
- Proposal drawing is anti-aliased before being resized to the Hunyuan input size.

Reason:

- The first proposal-conditioned smoke produced no hard failures, but both
  buildings warned for `many_raw_components|many_simplified_components`.
- The likely cause was over-detailed/disconnected image elements in the
  procedural proposal image.

Generated clean proposal sheet:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/osm_footprint_proposal_images.py \
  --osm_json outputs/quality_rerank_ab_sweep_lafayette4x4/east/osm_input.json \
  --out_dir outputs/osm_generative_proposal_clean_east4 \
  --limit 4 \
  --image_size 384 \
  --detail clean
```

Clean proposal visual output:

```text
outputs/osm_generative_proposal_clean_east4/proposal_inputs_sheet.png
outputs/osm_generative_proposal_clean_east4/proposal_inputs.csv
```

Ran the same two-building east-sector smoke with clean proposal inputs:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/osm_hunyuan_pipeline_smoke.py \
  --osm_json outputs/quality_rerank_ab_sweep_lafayette4x4/east/osm_input.json \
  --out_dir outputs/osm_generative_proposal_clean_east2 \
  --limit 2 \
  --retrieval_policy quality \
  --quality_model outputs/osm_generation_success_predictor_v1/generation_success_model.pkl \
  --quality_weight 0.20 \
  --quality_bad_candidate_penalty 1.0 \
  --conditioning_source proposal \
  --proposal_detail clean \
  --model mini \
  --steps 50 \
  --target_faces 50000 \
  --device cuda
```

Then applied the same heightfix, audit, and map visualization:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/osm_recompose_height_policy.py \
  --osm_json outputs/quality_rerank_ab_sweep_lafayette4x4/east/osm_input.json \
  --pipeline_log outputs/osm_generative_proposal_clean_east2/osm_hunyuan_scene.log.json \
  --out_dir outputs/osm_generative_proposal_clean_east2_heightfix \
  --height_policy area_aware

env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/osm_generation_quality_audit.py \
  --osm_json outputs/quality_rerank_ab_sweep_lafayette4x4/east/osm_input.json \
  --pipeline_log outputs/osm_generative_proposal_clean_east2_heightfix/osm_hunyuan_scene.log.json \
  --out_dir outputs/osm_generative_proposal_clean_east2_quality

env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/osm_pipeline_map_choices.py \
  --osm_json outputs/quality_rerank_ab_sweep_lafayette4x4/east/osm_input.json \
  --pipeline_log outputs/osm_generative_proposal_clean_east2_heightfix/osm_hunyuan_scene.log.json \
  --out_dir outputs/osm_generative_proposal_clean_east2_heightfix \
  --top_k 5 \
  --device cuda
```

Clean proposal outputs:

```text
outputs/osm_generative_proposal_clean_east2/osm_hunyuan_pipeline_sheet.png
outputs/osm_generative_proposal_clean_east2/osm_hunyuan_scene.obj
outputs/osm_generative_proposal_clean_east2/osm_hunyuan_scene.log.json
outputs/osm_generative_proposal_clean_east2_heightfix/osm_hunyuan_scene.obj
outputs/osm_generative_proposal_clean_east2_heightfix/osm_map_choices_sheet.png
outputs/osm_generative_proposal_clean_east2_quality/generation_quality_audit_sheet.png
outputs/osm_generative_proposal_clean_east2_quality/generation_quality_summary.json
```

Audit result improved:

```text
previous proposal baseline:
  count: 2
  pass: 0
  warn: 2
  fail: 0

clean proposal baseline:
  count: 2
  pass: 1
  warn: 1
  fail: 0
```

Per-building clean result:

```text
OSM_955677965: pass
  raw components: 6
  simplified components: 4

OSM_955677968: warn
  flags: many_raw_components | many_simplified_components
  raw components: 122
  simplified components: 79
```

Interpretation:

- Cleaner proposal images materially reduced fragmentation for one of two cases.
- The generative branch is now stronger than the first proposal attempt, but still weaker than retrieval-conditioned inputs.
- The remaining warning is a prompt/input-image quality issue, not a flat placement issue.
- Next research target: learn or synthesize more realistic single-building proposal images from footprint/class/height, then compare against this deterministic clean baseline.

## 2026-05-11 Codex Update: Learned Proposal Image Generator Scaffold

Moved beyond the procedural proposal drawer and added the first trainable
footprint-to-image proposal generator.

Added:

```text
scripts/build_osm_proposal_image_dataset.py
scripts/train_osm_proposal_image_generator.py
```

Updated:

```text
scripts/osm_hunyuan_pipeline_smoke.py
```

New OSM/Hunyuan pipeline mode:

```text
--conditioning_source learned_proposal
--learned_proposal_ckpt <checkpoint>
```

Dataset builder:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/build_osm_proposal_image_dataset.py \
  --records outputs/osm_generation_dataset_corpus_v2/campus_lafayette_v2_records.jsonl \
  --out_dir outputs/osm_proposal_image_dataset_v1 \
  --name campus_lafayette_proposal_v1 \
  --val_fraction 0.2
```

Dataset output:

```text
outputs/osm_proposal_image_dataset_v1/campus_lafayette_proposal_v1_train.jsonl
outputs/osm_proposal_image_dataset_v1/campus_lafayette_proposal_v1_val.jsonl
outputs/osm_proposal_image_dataset_v1/campus_lafayette_proposal_v1_all.jsonl
outputs/osm_proposal_image_dataset_v1/campus_lafayette_proposal_v1_summary.json
```

Dataset summary:

```text
examples: 37
train: 30
val: 7
quality: 37 pass
classes:
  RESIDENTIAL: 34
  COMMERCIAL: 2
  PUBLIC: 1
```

Training command:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/train_osm_proposal_image_generator.py \
  --train_jsonl outputs/osm_proposal_image_dataset_v1/campus_lafayette_proposal_v1_train.jsonl \
  --val_jsonl outputs/osm_proposal_image_dataset_v1/campus_lafayette_proposal_v1_val.jsonl \
  --out_dir outputs/osm_proposal_image_generator_v1_smoke \
  --image_size 128 \
  --batch_size 4 \
  --epochs 5 \
  --base_channels 24 \
  --num_workers 2 \
  --device cuda
```

Training outputs:

```text
outputs/osm_proposal_image_generator_v1_smoke/ckpt_best.pth
outputs/osm_proposal_image_generator_v1_smoke/ckpt_latest.pth
outputs/osm_proposal_image_generator_v1_smoke/metrics.csv
outputs/osm_proposal_image_generator_v1_smoke/summary.json
outputs/osm_proposal_image_generator_v1_smoke/val_preview_epoch_005.png
```

Training result:

```text
epoch 1 val_l1_pixel: 0.3253
epoch 5 val_l1_pixel: 0.1986
best checkpoint: outputs/osm_proposal_image_generator_v1_smoke/ckpt_best.pth
```

Interpretation:

- This is a tiny supervised baseline, not the final diffusion model.
- It learns from existing successful retrieved-render conditioning images.
- The task is now explicit and trainable:
  `footprint mask + class + height/context -> Hunyuan conditioning image`.

Integrated learned-proposal Hunyuan smoke:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/osm_hunyuan_pipeline_smoke.py \
  --osm_json outputs/quality_rerank_ab_sweep_lafayette4x4/east/osm_input.json \
  --out_dir outputs/osm_learned_proposal_east1_smoke \
  --limit 1 \
  --retrieval_policy quality \
  --quality_model outputs/osm_generation_success_predictor_v1/generation_success_model.pkl \
  --quality_weight 0.20 \
  --quality_bad_candidate_penalty 1.0 \
  --conditioning_source learned_proposal \
  --learned_proposal_ckpt outputs/osm_proposal_image_generator_v1_smoke/ckpt_best.pth \
  --model mini \
  --steps 50 \
  --target_faces 50000 \
  --device cuda
```

Then applied normal heightfix, audit, and map visualization:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/osm_recompose_height_policy.py \
  --osm_json outputs/quality_rerank_ab_sweep_lafayette4x4/east/osm_input.json \
  --pipeline_log outputs/osm_learned_proposal_east1_smoke/osm_hunyuan_scene.log.json \
  --out_dir outputs/osm_learned_proposal_east1_smoke_heightfix \
  --height_policy area_aware

env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/osm_generation_quality_audit.py \
  --osm_json outputs/quality_rerank_ab_sweep_lafayette4x4/east/osm_input.json \
  --pipeline_log outputs/osm_learned_proposal_east1_smoke_heightfix/osm_hunyuan_scene.log.json \
  --out_dir outputs/osm_learned_proposal_east1_smoke_quality

env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/osm_pipeline_map_choices.py \
  --osm_json outputs/quality_rerank_ab_sweep_lafayette4x4/east/osm_input.json \
  --pipeline_log outputs/osm_learned_proposal_east1_smoke_heightfix/osm_hunyuan_scene.log.json \
  --out_dir outputs/osm_learned_proposal_east1_smoke_heightfix \
  --top_k 5 \
  --device cuda
```

Learned proposal smoke outputs:

```text
outputs/osm_learned_proposal_east1_smoke/osm_hunyuan_pipeline_sheet.png
outputs/osm_learned_proposal_east1_smoke/osm_hunyuan_scene.obj
outputs/osm_learned_proposal_east1_smoke/osm_hunyuan_scene.log.json
outputs/osm_learned_proposal_east1_smoke_heightfix/osm_map_choices_sheet.png
outputs/osm_learned_proposal_east1_smoke_quality/generation_quality_audit_sheet.png
outputs/osm_learned_proposal_east1_smoke_quality/generation_quality_summary.json
```

Learned proposal smoke audit:

```text
count: 1
pass: 1
warn: 0
fail: 0
```

Why this matters:

- The pipeline is no longer only retrieval-conditioned.
- There is now a learned image proposal module in front of Hunyuan.
- Retrieval still supplies fallback/candidate supervision, but the image fed to
  Hunyuan can be generated from footprint/class/height features.
- This is the first complete learned generative path:
  `OSM footprint -> learned proposal image -> Hunyuan mesh -> placed/audited town output`.

Recommended next research step:

- Train the proposal generator longer and at higher resolution.
- Compare `retrieved`, `clean procedural proposal`, and `learned_proposal` on the same 4-8 buildings.
- Once learned proposal quality is competitive, replace the supervised UNet baseline with a diffusion or latent diffusion image proposal model.

## 2026-05-11 Codex Update: Continued Learned Proposal Training

Added resume support to:

```text
scripts/train_osm_proposal_image_generator.py
```

New option:

```text
--resume_ckpt <checkpoint>
```

Continued training from the 5-epoch smoke checkpoint:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/train_osm_proposal_image_generator.py \
  --train_jsonl outputs/osm_proposal_image_dataset_v1/campus_lafayette_proposal_v1_train.jsonl \
  --val_jsonl outputs/osm_proposal_image_dataset_v1/campus_lafayette_proposal_v1_val.jsonl \
  --out_dir outputs/osm_proposal_image_generator_v1_continued \
  --image_size 128 \
  --batch_size 4 \
  --epochs 30 \
  --base_channels 24 \
  --num_workers 2 \
  --resume_ckpt outputs/osm_proposal_image_generator_v1_smoke/ckpt_best.pth \
  --device cuda
```

Continued training outputs:

```text
outputs/osm_proposal_image_generator_v1_continued/ckpt_best.pth
outputs/osm_proposal_image_generator_v1_continued/ckpt_latest.pth
outputs/osm_proposal_image_generator_v1_continued/metrics.csv
outputs/osm_proposal_image_generator_v1_continued/summary.json
outputs/osm_proposal_image_generator_v1_continued/val_preview_epoch_030.png
```

Result:

```text
resumed from epoch: 5
continued through epoch: 30
previous best val_l1_pixel: 0.1986
new best val_l1_pixel:      0.0517
```

Late-epoch metrics:

```text
epoch 26 val_l1_pixel: 0.0541
epoch 28 val_l1_pixel: 0.0521
epoch 30 val_l1_pixel: 0.0517
```

Interpretation:

- The learned proposal image model is still improving substantially.
- The new checkpoint should replace the 5-epoch smoke checkpoint for the next Hunyuan learned-proposal runs.
- Next command for learned proposal inference should use:

```text
--learned_proposal_ckpt outputs/osm_proposal_image_generator_v1_continued/ckpt_best.pth
```

## 2026-05-11 Codex Update: Sharp-Loss Proposal Fine-Tuning

The epoch-30 learned proposal previews were still blurry, so the trainer was
upgraded with sharper image losses.

Updated:

```text
scripts/train_osm_proposal_image_generator.py
```

New loss weights:

```text
--w_l1
--w_grad
--w_ssim
--w_lap
```

Added losses:

- SSIM loss, to preserve local structure.
- Laplacian edge loss, to preserve sharper contours.
- Configurable gradient loss weight.

Fine-tuning command:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/train_osm_proposal_image_generator.py \
  --train_jsonl outputs/osm_proposal_image_dataset_v1/campus_lafayette_proposal_v1_train.jsonl \
  --val_jsonl outputs/osm_proposal_image_dataset_v1/campus_lafayette_proposal_v1_val.jsonl \
  --out_dir outputs/osm_proposal_image_generator_v1_sharp \
  --image_size 128 \
  --batch_size 4 \
  --epochs 60 \
  --base_channels 24 \
  --num_workers 2 \
  --resume_ckpt outputs/osm_proposal_image_generator_v1_continued/ckpt_best.pth \
  --w_l1 1.0 \
  --w_grad 0.75 \
  --w_ssim 0.35 \
  --w_lap 0.08 \
  --lr 0.0001 \
  --device cuda
```

Sharp fine-tune outputs:

```text
outputs/osm_proposal_image_generator_v1_sharp/ckpt_best.pth
outputs/osm_proposal_image_generator_v1_sharp/ckpt_latest.pth
outputs/osm_proposal_image_generator_v1_sharp/metrics.csv
outputs/osm_proposal_image_generator_v1_sharp/summary.json
outputs/osm_proposal_image_generator_v1_sharp/val_preview_epoch_060.png
```

Result:

```text
previous best val_l1_pixel: 0.0517
sharp fine-tune best:       0.0465
best epoch in sharp run:    49
epoch 60 val_l1_pixel:      0.0468
epoch 60 val_ssim_loss:     0.2178
epoch 60 val_lap_l1:        0.0235
```

Interpretation:

- The sharp-loss objective improved metrics modestly.
- This can help edges, but it will not fully solve blur because the current model is still a deterministic L1-style regressor over a small dataset.
- The best current supervised proposal checkpoint is:

```text
outputs/osm_proposal_image_generator_v1_sharp/ckpt_best.pth
```

Next inference should use:

```text
--learned_proposal_ckpt outputs/osm_proposal_image_generator_v1_sharp/ckpt_best.pth
```

Research implication:

- If the preview remains visibly blurry, the next model step should be a conditional diffusion/latent diffusion proposal model rather than more deterministic UNet training.

## 2026-05-11 Codex Update: Conditional DDPM Proposal Image Baseline

Started the diffusion proposal branch.

Added:

```text
scripts/train_osm_proposal_image_ddpm.py
```

Purpose:

- train a compact conditional DDPM over 128px proposal images;
- condition on the same channels as the supervised proposal model:
  footprint mask, height map, area map, aspect map, and class one-hot maps;
- generate sharper/more diverse Hunyuan conditioning images than deterministic RGB regression.

DDPM smoke command:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/train_osm_proposal_image_ddpm.py \
  --train_jsonl outputs/osm_proposal_image_dataset_v1/campus_lafayette_proposal_v1_train.jsonl \
  --val_jsonl outputs/osm_proposal_image_dataset_v1/campus_lafayette_proposal_v1_val.jsonl \
  --out_dir outputs/osm_proposal_image_ddpm_v1_smoke \
  --image_size 128 \
  --batch_size 4 \
  --epochs 20 \
  --base_channels 32 \
  --timesteps 100 \
  --sample_steps 25 \
  --sample_every 10 \
  --num_workers 2 \
  --device cuda
```

DDPM outputs:

```text
outputs/osm_proposal_image_ddpm_v1_smoke/ckpt_best.pth
outputs/osm_proposal_image_ddpm_v1_smoke/ckpt_latest.pth
outputs/osm_proposal_image_ddpm_v1_smoke/metrics.csv
outputs/osm_proposal_image_ddpm_v1_smoke/summary.json
outputs/osm_proposal_image_ddpm_v1_smoke/sample_epoch_010.png
outputs/osm_proposal_image_ddpm_v1_smoke/sample_epoch_020.png
```

Smoke result:

```text
train examples: 30
val examples: 7
best val_noise_mse: 0.1320
best epoch in smoke: 15
epoch 20 val_noise_mse: 0.1494
```

Interpretation:

- The DDPM training loop works end to end.
- Denoising loss drops strongly in the first 20 epochs, so the model is learning.
- Sample sheets are now available for visual inspection.
- This is the right branch for reducing blur because it samples images rather
  than regressing to the conditional mean.

Next DDPM steps:

1. Inspect `sample_epoch_020.png`.
2. If samples are recognizable, train longer with more timesteps/sample steps.
3. Add a sampler/export script so DDPM proposal images can be fed into the existing Hunyuan pipeline.
4. If samples are still noisy, increase training length first, then consider latent diffusion.

## 2026-05-11 Codex Update: DDPM EMA Overfit Test

The first 20-epoch DDPM samples looked like RGB speckle. Added two stabilizers
to the DDPM trainer:

```text
scripts/train_osm_proposal_image_ddpm.py
```

New options:

```text
--max_train_examples
--max_val_examples
--ema_decay
```

Changes:

- Added EMA model weights for evaluation and sampling.
- Added dataset limiting so we can run an explicit overfit test.
- Sampling now uses EMA weights.

Overfit command:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/train_osm_proposal_image_ddpm.py \
  --train_jsonl outputs/osm_proposal_image_dataset_v1/campus_lafayette_proposal_v1_train.jsonl \
  --val_jsonl outputs/osm_proposal_image_dataset_v1/campus_lafayette_proposal_v1_train.jsonl \
  --out_dir outputs/osm_proposal_image_ddpm_v1_overfit8 \
  --image_size 128 \
  --batch_size 4 \
  --epochs 300 \
  --base_channels 48 \
  --timesteps 200 \
  --sample_steps 100 \
  --sample_every 50 \
  --max_train_examples 8 \
  --max_val_examples 8 \
  --ema_decay 0.995 \
  --num_workers 2 \
  --device cuda
```

Overfit outputs:

```text
outputs/osm_proposal_image_ddpm_v1_overfit8/ckpt_best.pth
outputs/osm_proposal_image_ddpm_v1_overfit8/ckpt_latest.pth
outputs/osm_proposal_image_ddpm_v1_overfit8/metrics.csv
outputs/osm_proposal_image_ddpm_v1_overfit8/summary.json
outputs/osm_proposal_image_ddpm_v1_overfit8/sample_epoch_050.png
outputs/osm_proposal_image_ddpm_v1_overfit8/sample_epoch_150.png
outputs/osm_proposal_image_ddpm_v1_overfit8/sample_epoch_300.png
```

Result:

```text
train examples: 8
val examples: 8
best val_noise_mse: 0.0223
final epoch: 300
```

Interpretation:

- The DDPM can overfit a tiny proposal-image set; the architecture/training loop is not fundamentally broken.
- This is the checkpoint/sample set to inspect next:

```text
outputs/osm_proposal_image_ddpm_v1_overfit8/sample_epoch_300.png
```

Next decision:

- If `sample_epoch_300.png` now shows recognizable buildings, train the DDPM on all 37 examples for longer with EMA.
- If it is still speckled despite overfitting, the sampler or pixel-space DDPM formulation needs revision before full training.

## 2026-05-11 Codex Update: Cleanup, Research Check, DDIM Diagnostic

Cleanup status:

- Checked the active generative output folders.
- The DDPM/proposal-image artifacts are small and useful for comparison, so no
  destructive cleanup was performed.
- Existing repo changes include Claude work, Codex work, generated outputs, and
  external dependencies. These were left intact.

Research note saved here:

```text
docs/CODEX_RESEARCH_NOTES_GENERATIVE_PROPOSALS_2026-05-11.md
```

Papers reviewed for the current issue:

- Nichol and Dhariwal, "Improved Denoising Diffusion Probabilistic Models"
- Liu et al., "One-2-3-45++"
- Wei, Vosselman, and Yang, "BuilDiff"
- Zhou et al., "ControlCity"
- Tze et al., "PrITTI"
- "Human-guided urban form generation using multimodal diffusion models"

Implementation added:

```text
scripts/train_osm_proposal_image_ddpm.py
```

New DDPM diagnostic options:

```text
--sampler ddpm|ddim
--recon_every
--recon_timesteps
--resume_ckpt
```

Diagnostic output folder:

```text
outputs/osm_proposal_image_ddpm_v1_overfit8_ddim_diag/
```

Important outputs:

```text
sample_ddim_resume_epoch_300.png
recon_t025_epoch_300.png
recon_t050_epoch_300.png
recon_t100_epoch_300.png
recon_t150_epoch_300.png
```

Result:

- Pure DDIM sampling from noise still produces RGB speckle.
- Reconstruction from partially noised real targets recovers building
  mass/silhouette but remains soft/noisy.

Current conclusion:

- The model learned some conditional denoising.
- It is not yet a usable pure generative image model.
- Continuing the same tiny pixel-space DDPM training is unlikely to solve the
  core problem.
- Best next research move: keep the retrieval/supervised image path for
  end-to-end pipeline quality, and move the generative branch toward
  pretrained/latent/control-conditioned diffusion or a much larger proposal
  image dataset.

## 2026-05-12 Codex Update: Pretrained Image Prior Branch

Decision:

- Use the ControlCity idea as an image-prior strategy, not as a directly
  installable ControlCity model.
- Practical local implementation uses cached SD1.5 plus the existing
  footprint-to-view ControlNet checkpoint:

```text
external/hf_cache/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/
legacy/Logs_GT/CN-2026-05-05T22-55-39-footprint2view-15k-bf16/ckpt/controlnet-015000/
```

Added:

```text
scripts/generate_osm_image_prior_proposals.py
```

Purpose:

```text
OSM footprint/class/height
-> SD1.5 pretrained image prior + footprint ControlNet
-> Hunyuan conditioning image
```

Also extended:

```text
scripts/osm_hunyuan_pipeline_smoke.py
```

New Hunyuan conditioning source:

```text
--conditioning_source image_prior
--image_prior_manifest <csv>
```

Smoke generation command:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/generate_osm_image_prior_proposals.py \
  --osm_json outputs/osm_pipeline_smoke/osm_input.json \
  --out_dir outputs/osm_image_prior_proposals_v1_smoke \
  --limit 1 \
  --image_size 512 \
  --steps 10 \
  --guidance_scale 7.5 \
  --device cuda
```

Image-prior outputs:

```text
outputs/osm_image_prior_proposals_v1_smoke/image_prior_manifest.csv
outputs/osm_image_prior_proposals_v1_smoke/image_prior_sheet.png
outputs/osm_image_prior_proposals_v1_smoke/image_prior_inputs/00_OSM_302408405_image_prior.png
```

End-to-end Hunyuan smoke command:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
  scripts/osm_hunyuan_pipeline_smoke.py \
  --osm_json outputs/osm_pipeline_smoke/osm_input.json \
  --out_dir outputs/osm_image_prior_hunyuan_smoke \
  --limit 1 \
  --model mini \
  --steps 30 \
  --octree_resolution 256 \
  --num_chunks 12000 \
  --conditioning_source image_prior \
  --image_prior_manifest outputs/osm_image_prior_proposals_v1_smoke/image_prior_manifest.csv \
  --retrieval_policy quality \
  --quality_model outputs/osm_generation_success_predictor_v1/generation_success_model.pkl \
  --device cuda
```

End-to-end outputs:

```text
outputs/osm_image_prior_hunyuan_smoke/osm_hunyuan_pipeline_sheet.png
outputs/osm_image_prior_hunyuan_smoke/osm_hunyuan_scene.obj
outputs/osm_image_prior_hunyuan_smoke/osm_hunyuan_scene.log.json
outputs/osm_image_prior_hunyuan_smoke/osm_hunyuan_pipeline_metrics.csv
```

Observed result:

- The pretrained image-prior branch runs and is now wired into the Hunyuan
  pipeline.
- The generated image is building-like and footprint-driven, but currently too
  dark/cropped.
- Hunyuan accepts the image-prior input and generates a mesh.
- The placed render still shows the known flat/height-placement artifact; that
  is separate from the image-prior wiring.

Next tuning target:

- Improve prompt/control framing so the SD/ControlNet prior produces centered,
  fully visible, light-gray building massing images:
  - add more white margin around the footprint control,
  - lower ControlNet conditioning scale if it over-constrains the silhouette,
  - use a stronger negative prompt for cropped/black/filled silhouettes,
  - compare image-prior input against procedural and retrieval inputs through
    the same Hunyuan/audit sheet.
