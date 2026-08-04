# Codex Progress Report - 2026-05-08

Prepared by: Codex  
Project: `/scratch/gilbreth/dsimhadr/GenerativeTowns/SDFusion`  
Timestamp: Fri May 8 21:24 EDT 2026  

## Current Position

We are in the SDFusion project under the broader GenerativeTowns work. The current direction is retrieval-first:

1. Use footprint/map-derived inputs to retrieve detailed original BuildingNet OBJ assets.
2. Preserve OBJ surface detail instead of depending on 64^3 SDF extraction for final visual quality.
3. Add SDF or latent correction later to adapt the retrieved massing to the desired footprint.
4. Defer higher-resolution fields and a truly generative model until the retrieval/correction baseline is working.

This is a shift from trying to make MeshUDF the main output path. MeshUDF is now best treated as an optional/fallback extraction experiment, not the main visual-quality path.

## Context Reviewed

Read and incorporated:

- `README.md`
- `docs/PROJECT_STATUS.md`
- `docs/HYBRID_PIPELINE_PLAN.md`
- Claude chat history under `/home/dsimhadr/.claude/projects/-scratch-gilbreth-dsimhadr-GenerativeTowns-SDFusion/`
- Existing dataset structure and split files
- Existing SDF, footprint, OBJ, and renderer utilities

Important context from the Claude planning history:

- Existing VQVAE round-trips signed SDFs well, with roughly 0.83-0.99 silhouette IoU in empirical tests.
- Feeding UDFs into the same VQVAE performs poorly because the decoder was trained on signed fields.
- The selected near-term plan was to keep SDFs as the representation and avoid a full UDF retrain.
- MeshUDF was explored because marching cubes fragments hollow SDFs.
- Watertight repair was rejected because it erases exterior architectural detail and fills hollow structures.

## Dataset State

Active full v1 split sizes, counted by loader logic:

- train: 1481
- val: 187
- test: 181

Each active split item has:

- SDF H5: `data/BuildingNet_dataset_v0_1/resolution_64/<id>/ori_sample_grid.h5`
- Footprint PNG: `data/BuildingNet_dataset_v0_1/footprints_png/{train,val,test}/<id>.png`
- Original OBJ: `data/BuildingNet_dataset_v0_1/OBJ_MODELS/<id>.obj`

Note: the split text files do not end with trailing newlines, so `wc -l` undercounts them by one.

Occupancy stats from the BuildingNet SDF set:

- Median inside percentage: about 0.5314%
- >= 0.2% inside: 1361
- >= 0.5% inside: 957
- >= 1.0% inside: 658
- Empty fields: 0

The SDFs are often sparse or hollow, but not empty.

## CUDA / Environment Status

CUDA works on the host outside the default sandbox. The machine observed was `gilbreth-k026.rcac.purdue.edu` with an A100 80GB GPU.

Inside the default sandbox, `/dev/nvidia*` is hidden and `torch.cuda.is_available()` reports false. GPU commands need to be run outside the sandbox with the environment cleaned:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python ...
```

This was used successfully for retrieval training, index building, and visual smoke tests.

## MeshUDF Work Completed

Added official MeshUDF code under:

- `external/MeshUDF/`

Installed/compiled local Python extension:

- `external/MeshUDF/custom_mc/_marching_cubes_lewiner_cy.cpython-39-x86_64-linux-gnu.so`

Updated/added:

- `scripts/extraction_strategy_smoke.py`
- `models/arap_deformer.py`

The smoke script compares:

- Signed marching cubes at `SDF = 0`
- Crude `abs(SDF)` plus marching cubes
- Real MeshUDF on `abs(SDF)` with gradients
- ARAP deformation where feasible

Clean output directory:

- `outputs/extraction_smoke_meshudf/`

Observed metrics:

| Building | Signed MC components / IoU | abs+MC components / IoU | MeshUDF components / IoU | ARAP |
| --- | ---: | ---: | ---: | --- |
| house8443 | 109 / 0.9164 | 2 / 0.7246 | 15 / 0.7795 | ok |
| house4208 | 1 / 0.9346 | 2 / 0.8231 | 6 / 0.6597 | skipped, too many faces |
| villa3202 | 25 / 0.7967 | 2 / 0.5717 | 7 / 0.5302 | ok |
| villa5927 | 23 / 0.6295 | 7 / 0.1334 | 9 / 0.2038 | skipped, too many faces |

Conclusion:

- MeshUDF reduces some marching-cubes fragmentation.
- MeshUDF does not solve the core visual-quality problem from 64^3 sparse/hollow SDFs.
- `abs(SDF)` looks shell-like because it removes sign and extracts a distance shell, not a solid volume.
- Missing or weak wall definition is expected from coarse 64^3 fields and hollow BuildingNet SDFs.
- The stronger near-term path is to retrieve original OBJ detail, then adapt it.

## Retrieval Model Work Completed

Implemented a footprint retrieval baseline.

New files:

- `models/networks/retrieval/footprint_embed.py`
- `models/networks/retrieval/__init__.py`
- `datasets/buildingnet_retrieval_dataset.py`
- `train_retrieval.py`
- `scripts/build_retrieval_index.py`
- `scripts/retrieval_smoke.py`
- `scripts/retrieval_visual_smoke.py`

Model:

- Small CNN footprint encoder
- Class embedding
- 256-dimensional L2-normalized embedding
- Classifier head
- NT-Xent contrastive loss between augmented footprint views
- Cross-entropy class loss

Training command used:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python train_retrieval.py \
  --out_dir Logs_GT/retrieval_footprint_full \
  --epochs 30 \
  --batch_size 128 \
  --num_workers 4 \
  --device cuda
```

Training artifacts:

- `Logs_GT/retrieval_footprint_full/ckpt_best.pth`
- `Logs_GT/retrieval_footprint_full/ckpt_latest.pth`
- `Logs_GT/retrieval_footprint_full/label_maps.json`
- `Logs_GT/retrieval_footprint_full/loss_log.txt`

Best checkpoint:

- `ckpt_best.pth`
- Selected at epoch 1 by validation nearest-neighbor top-class score.
- Epoch 1 validation: `val_nn_same_top = 0.9465`, `val_nn_same_subtype = 0.9465`
- Training loss continued decreasing through epoch 30, but retrieval validation was strongest early.

This suggests the model quickly learned class-compatible footprint retrieval and then may have overfit or optimized augmentation details that did not improve nearest-neighbor retrieval.

## Retrieval Index Completed

Index build command:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python scripts/build_retrieval_index.py \
  --ckpt Logs_GT/retrieval_footprint_full/ckpt_best.pth \
  --out_dir data/BuildingNet_dataset_v0_1/retrieval_index \
  --device cuda \
  --num_workers 4 \
  --batch_size 256
```

Artifacts:

- `data/BuildingNet_dataset_v0_1/retrieval_index/train_embeddings.npz`
- `data/BuildingNet_dataset_v0_1/retrieval_index/val_embeddings.npz`
- `data/BuildingNet_dataset_v0_1/retrieval_index/test_embeddings.npz`
- `data/BuildingNet_dataset_v0_1/retrieval_index/metadata.json`

Embedding counts:

- train: 1481
- val: 187
- test: 181

Text smoke examples:

- `COMMERCIALcity_hall_mesh0106 -> COMMERCIALcity_hall_mesh0527, COMMERCIALcity_hall_mesh0258, COMMERCIALcastle_mesh0904`
- `COMMERCIALhotel_building_mesh0504 -> COMMERCIALhotel_building_mesh0461, COMMERCIALhotel_building_mesh0520, COMMERCIALhotel_building_mesh0302`
- `COMMERCIALhouse_mesh6823 -> COMMERCIALhouse_mesh5334, COMMERCIALhouse_mesh7536, COMMERCIALhouse_mesh3602`
- `COMMERCIALhouse_mesh7736 -> COMMERCIALhouse_mesh7536, COMMERCIALhouse_mesh2682, COMMERCIALhouse_mesh3602`
- `COMMERCIALoffice_building_mesh0274 -> COMMERCIALoffice_building_mesh3616, COMMERCIALoffice_building_mesh0246, COMMERCIALoffice_building_mesh0126`
- `COMMERCIALoffice_building_mesh0640 -> COMMERCIALoffice_building_mesh3273, COMMERCIALoffice_building_mesh3311, COMMERCIALoffice_building_mesh1861`

## Visual Smoke Completed

Visual retrieval smoke command:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python scripts/retrieval_visual_smoke.py \
  --phase val \
  --limit 6 \
  --top_k 3 \
  --out_dir outputs/retrieval_visual_smoke \
  --device cuda
```

Output contact sheets:

- `outputs/retrieval_visual_smoke/val_000_COMMERCIALcity_hall_mesh0106.png`
- `outputs/retrieval_visual_smoke/val_001_COMMERCIALhotel_building_mesh0504.png`
- `outputs/retrieval_visual_smoke/val_002_COMMERCIALhouse_mesh6823.png`
- `outputs/retrieval_visual_smoke/val_003_COMMERCIALhouse_mesh7736.png`
- `outputs/retrieval_visual_smoke/val_004_COMMERCIALoffice_building_mesh0274.png`
- `outputs/retrieval_visual_smoke/val_005_COMMERCIALoffice_building_mesh0640.png`

These should be inspected visually before committing to correction-pair training, because nearest-neighbor class metrics are not enough to prove useful architectural similarity.

## Retrieval Alignment Smoke Completed

Added:

- `scripts/retrieval_alignment_smoke.py`

Purpose:

- Query a footprint embedding.
- Retrieve top-k train OBJs from the retrieval index.
- Normalize the retrieved OBJ into its SDF frame using `norm_params`.
- Fit the retrieved OBJ's X/Z bbox to the query footprint bbox.
- Preserve vertical proportion using the geometric mean of the X/Z scales.
- Export aligned OBJs and contact sheets for inspection.

Smoke command used:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python scripts/retrieval_alignment_smoke.py \
  --phase val \
  --limit 3 \
  --top_k 2 \
  --out_dir outputs/retrieval_alignment_smoke \
  --device cuda
```

Artifacts:

- `outputs/retrieval_alignment_smoke/metrics.csv`
- `outputs/retrieval_alignment_smoke/val_000_COMMERCIALcity_hall_mesh0106/alignment_sheet.png`
- `outputs/retrieval_alignment_smoke/val_001_COMMERCIALhotel_building_mesh0504/alignment_sheet.png`
- `outputs/retrieval_alignment_smoke/val_002_COMMERCIALhouse_mesh6823/alignment_sheet.png`
- Aligned OBJ files under each query subdirectory.

Observed footprint IoU values from the first smoke:

| Query | Rank | Retrieved OBJ | Footprint IoU |
| --- | ---: | --- | ---: |
| `COMMERCIALcity_hall_mesh0106` | 1 | `COMMERCIALcity_hall_mesh0527` | 0.3073 |
| `COMMERCIALcity_hall_mesh0106` | 2 | `COMMERCIALcity_hall_mesh0258` | 0.3076 |
| `COMMERCIALhotel_building_mesh0504` | 1 | `COMMERCIALhotel_building_mesh0461` | 0.1944 |
| `COMMERCIALhotel_building_mesh0504` | 2 | `COMMERCIALhotel_building_mesh0520` | 0.0886 |
| `COMMERCIALhouse_mesh6823` | 1 | `COMMERCIALhouse_mesh5334` | 0.3743 |
| `COMMERCIALhouse_mesh6823` | 2 | `COMMERCIALhouse_mesh7536` | 0.5153 |

Interpretation:

- The alignment exports usable fitted OBJs.
- Raw footprint IoU is not a complete quality metric because many query footprints are wall-line or hollow SDF silhouettes, while OBJ top projections include solid roof/surface coverage.
- Contact sheets should drive the next decision more than the scalar IoU alone.
- The next improvement is to add a better shape metric, such as filled-outline IoU or bbox/contour IoU, before training correction pairs.

## Correction Pair Construction Completed

Added:

- `scripts/build_correction_pairs.py`

Purpose:

- Use the retrieval index to select top-k source buildings for each query.
- Load source and target 64^3 SDF fields.
- Align the source SDF into the target/query SDF coordinate frame.
- Save `source_aligned_sdf`, `target_sdf`, `residual_sdf`, source/target footprints, and the alignment transform.
- Write metadata CSVs with retrieval similarity, raw footprint IoU, filled-footprint IoU, residual L1, and residual L2.

Smoke command:

```bash
./sdfusion/bin/python scripts/build_correction_pairs.py \
  --phase val \
  --limit 3 \
  --top_k 2 \
  --out_dir outputs/correction_pairs_smoke
```

Full train command:

```bash
./sdfusion/bin/python scripts/build_correction_pairs.py \
  --phase train \
  --top_k 1 \
  --out_dir data/BuildingNet_dataset_v0_1/correction_pairs
```

Full val command:

```bash
./sdfusion/bin/python scripts/build_correction_pairs.py \
  --phase val \
  --top_k 1 \
  --out_dir data/BuildingNet_dataset_v0_1/correction_pairs
```

Artifacts:

- `outputs/correction_pairs_smoke/val/pair_metadata.csv`
- `outputs/correction_pairs_smoke/val/pairs/*.npz`
- `data/BuildingNet_dataset_v0_1/correction_pairs/train/pair_metadata.csv`
- `data/BuildingNet_dataset_v0_1/correction_pairs/train/pairs/*.npz`
- `data/BuildingNet_dataset_v0_1/correction_pairs/val/pair_metadata.csv`
- `data/BuildingNet_dataset_v0_1/correction_pairs/val/pairs/*.npz`

Counts:

- train pairs: 1481
- val pairs: 187
- total size: about 1.2 GB

Metric summary:

| Phase | Rows | Similarity mean / median | Footprint IoU mean / median | Filled IoU mean / median | Residual L1 mean / median | Residual L2 mean / median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 1481 | 0.9951 / 0.9990 | 0.3082 / 0.2389 | 0.3199 / 0.2574 | 0.3174 / 0.2694 | 0.4058 / 0.3574 |
| val | 187 | 0.9951 / 0.9991 | 0.3137 / 0.2518 | 0.3308 / 0.2924 | 0.3015 / 0.2495 | 0.3835 / 0.3337 |

Interpretation:

- Correction-pair data is now available for residual training.
- Retrieval embedding similarity is high, matching the accepted retrieval direction.
- Raw and filled footprint IoU are still modest because the underlying SDF footprints are often sparse/hollow wall traces, so they are diagnostics rather than hard pass/fail metrics.
- Residual magnitudes are reasonable for a first residual-learning smoke, but train outliers exist (`residual_l1` max about 7.44), so the residual trainer should use robust losses or clipping.

## SDF Residual Training Scaffold Completed

Added:

- `datasets/correction_pair_dataset.py`
- `models/networks/sdf_residual_net.py`
- `train_sdf_residual.py`

Model:

- Small 3D U-Net.
- Input channels:
  - aligned source SDF
  - target footprint repeated as a volume along the vertical axis
- Output:
  - predicted residual SDF
- Final residual head is zero-initialized so the model starts as a no-op correction instead of degrading the retrieved source.
- Output residual is bounded with `tanh * residual_clip`.

Loss and metrics:

- Training loss: SmoothL1 on clipped residuals.
- Validation metrics:
  - residual L1
  - corrected SDF L1
  - aligned-source SDF L1 baseline
  - corrected sign IoU
  - source sign IoU baseline

Initial smoke command before zero-init:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python train_sdf_residual.py \
  --pair_root data/BuildingNet_dataset_v0_1/correction_pairs \
  --out_dir Logs_GT/sdf_residual_smoke \
  --epochs 2 \
  --batch_size 4 \
  --max_train_samples 32 \
  --max_val_samples 16 \
  --num_workers 2 \
  --base_channels 8 \
  --device cuda
```

Result:

- Pipeline worked and checkpoints were written.
- But the random residual head degraded the source baseline on the tiny smoke subset.

Zero-init smoke command:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python train_sdf_residual.py \
  --pair_root data/BuildingNet_dataset_v0_1/correction_pairs \
  --out_dir Logs_GT/sdf_residual_smoke_zeroinit \
  --epochs 2 \
  --batch_size 4 \
  --max_train_samples 32 \
  --max_val_samples 16 \
  --num_workers 2 \
  --base_channels 8 \
  --device cuda
```

Zero-init smoke result:

| Epoch | Train loss | Val corrected L1 | Val source L1 | Val corrected IoU | Val source IoU |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.290164 | 0.297591 | 0.298963 | 0.128806 | 0.126798 |
| 2 | 0.288031 | 0.295364 | 0.298963 | 0.128509 | 0.126798 |

Artifacts:

- `Logs_GT/sdf_residual_smoke/`
- `Logs_GT/sdf_residual_smoke_zeroinit/`

Interpretation:

- The residual training path is wired correctly.
- Zero-init is important and should stay.
- The tiny smoke subset shows a small improvement over the aligned-source baseline, but this is not yet a meaningful model-quality result.
- The next real test is a longer run on the full correction-pair train/val set.

## Full SDF Residual Training Completed

Command:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python train_sdf_residual.py \
  --pair_root data/BuildingNet_dataset_v0_1/correction_pairs \
  --out_dir Logs_GT/sdf_residual_full \
  --epochs 30 \
  --batch_size 8 \
  --num_workers 4 \
  --base_channels 16 \
  --residual_clip 1.0 \
  --device cuda
```

Artifacts:

- `Logs_GT/sdf_residual_full/args.json`
- `Logs_GT/sdf_residual_full/ckpt_best.pth`
- `Logs_GT/sdf_residual_full/ckpt_latest.pth`
- `Logs_GT/sdf_residual_full/loss_log.txt`

Best checkpoints by metric:

| Selection | Epoch | Val corrected L1 | Val source L1 | Val corrected IoU | Val source IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| best corrected L1 / saved `ckpt_best.pth` | 27 | 0.096571 | 0.300991 | 0.130927 | 0.154759 |
| best corrected sign IoU | 19 | 0.103357 | 0.300991 | 0.252189 | 0.154759 |
| final epoch | 30 | 0.101781 | 0.300991 | 0.178829 | 0.154759 |

Interpretation:

- The residual model strongly improves continuous SDF L1 versus the aligned source baseline.
- Sign/occupancy IoU is less stable. Some epochs improve it substantially, but the checkpoint selected by L1 is not the best sign-IoU checkpoint.
- For mesh output, we should not rely on L1 alone. We should either select by sign/footprint/mesh metrics or use a mixed validation criterion.

## SDF Residual Visual Evaluation Completed

Added:

- `scripts/eval_sdf_residual.py`

Command:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python scripts/eval_sdf_residual.py \
  --ckpt Logs_GT/sdf_residual_full/ckpt_best.pth \
  --phase val \
  --limit 16 \
  --out_dir outputs/sdf_residual_eval_best_l1 \
  --device cuda
```

Artifacts:

- `outputs/sdf_residual_eval_best_l1/val/metrics.csv`
- `outputs/sdf_residual_eval_best_l1/val/sheets/*.png`

16-example visual-eval summary for `ckpt_best.pth`:

| Metric | Source mean / median | Corrected mean / median |
| --- | ---: | ---: |
| SDF L1 | 0.2990 / 0.2238 | 0.1157 / 0.1026 |
| Sign IoU | 0.0475 / 0.0062 | 0.0673 / 0.0070 |
| Footprint IoU | 0.2006 / 0.1050 | 0.2630 / 0.1812 |

Counts:

- SDF L1 improved: 15 / 16
- Sign IoU improved: 6 / 16
- Footprint IoU improved: 7 / 16

Interpretation:

- The residual model is useful as a continuous-field corrector.
- It is not yet reliably improving binary occupancy for every example.
- The next step should be checkpoint selection and visualization around occupancy/mesh quality, not just more L1 training.

## SDF Residual Mesh Evaluation Completed

Updated:

- `train_sdf_residual.py`

Change:

- Future full runs now also save `ckpt_best_iou.pth` based on validation corrected sign IoU.
- The already-completed `Logs_GT/sdf_residual_full` run only has `ckpt_best.pth` and `ckpt_latest.pth`, so the first mesh comparison used those two.

Added:

- `scripts/eval_sdf_residual_meshes.py`

Purpose:

- Load a residual checkpoint.
- Predict corrected SDFs on correction-pair examples.
- Extract source/corrected/target meshes using marching cubes.
- Export OBJ files when requested.
- Measure footprint IoU, connected component count, largest-component fraction, and SDF L1.
- Write contact sheets and CSV metrics.

Commands:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python scripts/eval_sdf_residual_meshes.py \
  --ckpt Logs_GT/sdf_residual_full/ckpt_best.pth \
  --phase val \
  --limit 16 \
  --out_dir outputs/sdf_residual_mesh_eval_best_l1 \
  --export_obj \
  --device cuda
```

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python scripts/eval_sdf_residual_meshes.py \
  --ckpt Logs_GT/sdf_residual_full/ckpt_latest.pth \
  --phase val \
  --limit 16 \
  --out_dir outputs/sdf_residual_mesh_eval_latest \
  --export_obj \
  --device cuda
```

Artifacts:

- `outputs/sdf_residual_mesh_eval_best_l1/val/mesh_metrics.csv`
- `outputs/sdf_residual_mesh_eval_best_l1/val/sheets/*.png`
- `outputs/sdf_residual_mesh_eval_best_l1/val/meshes/*.obj`
- `outputs/sdf_residual_mesh_eval_latest/val/mesh_metrics.csv`
- `outputs/sdf_residual_mesh_eval_latest/val/sheets/*.png`
- `outputs/sdf_residual_mesh_eval_latest/val/meshes/*.obj`

16-example mesh-eval summary:

| Checkpoint | Corrected L1 mean / median | Corrected footprint IoU mean / median | Corrected components mean / median | Largest component frac mean / median | FP IoU improved | Components reduced |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ckpt_best.pth` | 0.1157 / 0.1026 | 0.2630 / 0.1812 | 3.5625 / 1.5 | 0.6234 / 0.7635 | 7 / 16 | 10 / 16 |
| `ckpt_latest.pth` | 0.1236 / 0.1133 | 0.2764 / 0.0899 | 4.1250 / 2.5 | 0.6384 / 0.7903 | 8 / 16 | 9 / 16 |

Shared source baseline on the same 16 examples:

- source SDF L1 mean / median: 0.2990 / 0.2238
- source footprint IoU mean / median: 0.2006 / 0.1050
- source components mean / median: 9.25 / 7.0
- source largest-component fraction mean / median: 0.4930 / 0.2653

Interpretation:

- Both residual checkpoints strongly improve continuous SDF L1.
- Both reduce fragmentation versus the aligned-source SDF extraction on most sampled examples.
- `ckpt_best.pth` is better for L1 and average component count.
- `ckpt_latest.pth` has slightly higher mean footprint IoU and largest-component fraction, but worse median footprint IoU.
- The result supports using residual correction as an extraction aid, but not as the final visual-detail source. The retrieved OBJ is still the high-detail asset path.

## Claude Updates Integrated

This section records additional Claude-made changes found in the newer Claude history file:

- `/home/dsimhadr/.claude/projects/-scratch-gilbreth-dsimhadr-GenerativeTowns-SDFusion/b7716954-b3f9-446f-a031-f380759d3f76.jsonl`

### Mesh Evaluation Visualization

Claude updated:

- `scripts/eval_sdf_residual_meshes.py`

Changes:

- Added `sdf_footprint_vis()` for visualization-only footprints.
- Visualization footprints use a wider near-surface threshold, `sdf <= 0.1`, instead of strict `sdf <= 0`.
- Visualization footprints are passed through `scipy.ndimage.binary_fill_holes`.
- Strict `sdf <= 0` footprints are still used for metrics and CSV IoU values.
- Mask sheets were enlarged from 192 px to 256 px.
- Mask rendering now uses anti-aliased colored cells.
- Added an overlay panel:
  - blue = target footprint
  - red = source/retrieved footprint
  - green = corrected footprint
- The overlay makes alignment and correction movement easier to inspect visually.

Reason:

- The strict `sdf <= 0` visualization often looked like most data was missing because BuildingNet SDFs are hollow/thin-shell fields.
- For metric correctness, strict iso=0 is still used.
- For human inspection, the wider near-surface band plus hole filling gives a much more readable footprint.

Validation:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python -m py_compile scripts/eval_sdf_residual_meshes.py
```

Result:

- Syntax check passed.

Claude reran the best-L1 mesh evaluator:

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python scripts/eval_sdf_residual_meshes.py \
  --ckpt Logs_GT/sdf_residual_full/ckpt_best.pth \
  --out_dir outputs/sdf_residual_mesh_eval_best_l1_v3
```

Artifacts:

- `outputs/sdf_residual_mesh_eval_best_l1_v3/val/mesh_metrics.csv`
- `outputs/sdf_residual_mesh_eval_best_l1_v3/val/sheets/*.png`

Metric output was unchanged from the prior best-L1 mesh eval because the visualization changes do not change metric footprints.

### Core SDFusion Training Fixes

Claude-made tracked-file changes also exist in the older SDFusion img2shape training path:

- `configs/sdfusion-img2shape.yaml`
- `datasets/base_dataset.py`
- `datasets/dataloader.py`
- `models/sdfusion_model_img2shape.py`
- `preprocess/create_sdf.py`
- `train.py`

Key changes:

- Replaced the Stable-Diffusion image-VAE scale factor `0.18215` with BuildingNet VQVAE latent scale factor `2.380615`.
- Added comments explaining the measured latent std: approximately `0.4201`, so `scale_factor = 1 / std`.
- Added validation split support for the BuildingNet dataset path:
  - `CreateDataset()` now returns train, val, and test datasets.
  - `CreateDataLoader()` now returns `train_dl`, `val_dl`, `test_dl`, and `test_dl_for_eval`.
  - `train.py` prints val dataset count when available.
  - `train.py` computes a simple val loss during scheduled eval.
- Fixed the footprint-to-latent-volume axis convention in `SDFusionImageFPShapeModel._build_fp3d_for()`:
  - Source SDF layout is `(z, y, x)`.
  - Footprint is `(z, x)` from top-down projection.
  - The footprint should be resized to `(D, W)` and repeated along latent `H`/Y.
  - Earlier code effectively treated it as `(H, W)` and repeated along depth, rotating/misplacing the ground plan.
- Applied latent rescaling during diffusion training:
  - VQVAE latent `z` is multiplied by `self.scale_factor` before diffusion loss.
  - DDIM samples are divided by `self.scale_factor` before VQVAE decoding.
- Added loss reporting through `get_current_errors()`.
- Simplified `eval_metrics()` for that path to return a placeholder while val loss is handled in `train.py`.
- Fixed `preprocess/create_sdf.py::check_insideout()` indexing:
  - The distance-field flat array is from an `(sdf_res + 1)^3` grid.
  - The old stride used `sdf_res`, which read the wrong center voxel.
  - This could spuriously flag hollow meshes as inside-out and cause reprocessing churn.

Status:

- These core SDFusion changes are documented but were not further modified here.
- They are separate from the retrieval/residual pipeline that is currently the main path.
- They may matter later if we resume true diffusion training.

## Current Improvement Plan

The current bottleneck is no longer retrieval. Retrievals are acceptable. The bottleneck is that the residual model is mainly optimized for continuous SDF L1, while final quality depends on occupancy, footprint conformity, surface quality, and mesh fragmentation.

Recommended improvements:

1. Add occupancy-aware residual losses.
   - Add sign/occupancy BCE for `sdf <= 0`.
   - Add higher weighting near the target surface band, for example `abs(target_sdf) < threshold`.
   - Add footprint projection loss comparing top-down occupancy projections.
   - Keep SmoothL1, but do not let it be the only objective.

2. Select checkpoints by geometry, not only L1.
   - The full run showed best L1 and best sign IoU occur at different epochs.
   - Future training now saves `ckpt_best_iou.pth`, but the next trainer pass should also consider footprint IoU and component count.
   - For output, prefer a combined validation score over pure corrected L1.

3. Improve retrieval source selection with top-k pairs.
   - Current correction pairs are top-1 only.
   - Build top-3 or top-5 correction candidates.
   - Choose the best candidate by footprint/filled-footprint IoU before residual correction, or run correction on multiple candidates and select the best corrected result.

4. Improve alignment before residual correction.
   - Current alignment is bbox-only.
   - Add 90-degree rotation candidates.
   - Compare uniform vs anisotropic scaling.
   - Score candidate transforms by filled footprint IoU or contour IoU.
   - This should reduce the burden on the residual model.

5. Keep corrected SDF as a structural guide, not the final visual source.
   - 64^3 SDFs cannot recover high-frequency OBJ detail such as windows, trim, roofs, and wall articulation.
   - The retrieved OBJ remains the high-detail visual asset.
   - Corrected SDF should guide placement, scaling, footprint conformance, mesh fallback extraction, or later deformation/cropping.

Most practical next implementation step:

- Update `train_sdf_residual.py` with occupancy/surface/footprint losses and geometry-aware checkpoint selection.

## Known Issues / Risks

- `docs/PROJECT_STATUS.md` is now stale relative to the retrieval-first path.
- Existing SDF extraction remains limited by 64^3 resolution.
- MeshUDF from `abs(SDF)` can reduce fragmentation but often worsens footprint fidelity.
- `abs(SDF)` output looks hollow because sign information is removed and the method extracts a shell around distance minima.
- ARAP currently skips large detailed OBJs above the configured face threshold.
- The retrieval visual smoke has only been run on six validation examples so far.
- Retrieval alignment has only been smoke-tested on three validation queries and two neighbors each.
- Current alignment is bbox-based only; it does not yet rotate, contour-match, or deform the retrieved mesh.
- Correction pairs are top-1 only; harder top-k pair mining is not implemented yet.
- Correction-pair residuals have outliers, so residual training should not start with plain unbounded MSE only.
- Residual model improves continuous SDF L1 strongly, but binary occupancy and footprint metrics are mixed.
- `ckpt_best.pth` is selected by corrected L1, not by sign IoU or mesh quality.
- Current mesh evaluation uses marching cubes only; MeshUDF extraction of corrected SDFs has not been compared yet.
- The repository already contains unrelated modified/deleted/untracked files. These were not reverted or cleaned.

## Recommended Next Steps

1. Inspect the six retrieval visual contact sheets in `outputs/retrieval_visual_smoke/`.
2. Inspect the three retrieval alignment sheets in `outputs/retrieval_alignment_smoke/`.
3. Evaluate whether simple alignment gives usable town/building placement before training correction.
4. Run a short follow-up residual training run with `ckpt_best_iou.pth` saving enabled, or manually preserve the epoch-19 checkpoint in the next run.
5. Compare marching cubes and MeshUDF extraction on corrected SDFs.
6. If corrected meshes improve over aligned-source extraction, connect correction to output extraction or retrieved-OBJ adaptation.
7. Only after that, revisit higher resolution and a truly generative model.

## Bottom Line

We are past environment setup and basic retrieval implementation. The strongest current status is:

- CUDA is working outside the sandbox.
- MeshUDF is integrated and tested, but not good enough as the main path.
- Retrieval training completed on the full split.
- Retrieval index exists for train/val/test.
- Visual retrieval smoke outputs exist and are ready for inspection.
- Retrieval alignment smoke now exports fitted OBJs and contact sheets.
- Full top-1 train/val correction-pair datasets now exist.
- SDF residual training code exists and has passed a CUDA smoke test.
- Full 30-epoch residual training is complete.
- Residual visual evaluation sheets exist for the best-L1 checkpoint.
- Mesh extraction evaluation exists for `ckpt_best.pth` and `ckpt_latest.pth`.

The next engineering move should be corrected-SDF extraction comparison, especially marching cubes vs MeshUDF and best-L1 vs best-IoU checkpoint selection.

Update: the follow-up Claude/Codex residual work from 2026-05-09 is captured in:

- `docs/CODEX_PROGRESS_REPORT_2026-05-09.md`

That update documents composite geometry loss, augmentation, top-K=3 correction pairs, v2/v3/v4 residual training, and the v4 mesh evaluation.
