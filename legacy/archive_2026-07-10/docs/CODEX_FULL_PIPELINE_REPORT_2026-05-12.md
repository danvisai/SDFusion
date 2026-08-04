# Codex Full Pipeline Report - GenerativeTowns / SDFusion

Prepared by: Codex  
Date: 2026-05-12  
Repo: `/scratch/gilbreth/dsimhadr/GenerativeTowns/SDFusion`

## Executive Summary

The project has a working OSM-to-3D-building pipeline and a separate
generative research branch. The strongest current output path is not pure SDF
generation. It is a hybrid path:

```text
OSM footprint/map
-> footprint/context dataset
-> retrieval and quality rerank or learned proposal image
-> Hunyuan3D image-to-3D
-> mesh simplify
-> world placement and area-aware height correction
-> visual sheet and quality audit
```

SDF remains part of the research pipeline as a geometry representation,
constraint field, and possible refinement/audit layer. It does not need to be
the final mesh generator for the research story to stay SDF-relevant.

## Current Pipeline

1. OSM/vector map ingestion creates building records with polygon footprint,
   class, area, aspect, and height metadata.
2. Dataset builders produce footprint masks and proposal-image records.
3. Retrieval models select BuildingNet-like candidates using footprint/class
   similarity plus quality-aware reranking.
4. Proposal-image branches produce Hunyuan conditioning images:
   - retrieval/procedural proposal image path,
   - supervised learned proposal image path,
   - experimental DDPM proposal image path.
5. Hunyuan3D generates raw meshes from the selected/proposed image.
6. Simplification reduces huge raw Hunyuan meshes to usable scene meshes.
7. Heightfix places meshes at map coordinates and applies area-aware height
   correction.
8. Quality audit flags pass/warn/fail outputs.
9. SDF remains available for footprint correction, fit scoring, residual
   refinement, and MeshUDF-style extraction experiments.

## Dataset / Artifact Status

OSM corpus v2 summary:

```json
{
  "corpus_name": "campus_lafayette_v2",
  "dataset_version": "osm_generation_scaffold_v1",
  "count": 44,
  "positive_count": 37,
  "negative_count": 7,
  "classes": {
    "RESIDENTIALhouse": 41,
    "COMMERCIALoffice_building": 2,
    "PUBLICschool_building": 1
  },
  "splits": {
    "smoke12": 12,
    "north8": 8,
    "east6": 6,
    "west6": 6,
    "south6": 6,
    "northeast6": 6
  },
  "jsonl": "outputs/osm_generation_dataset_corpus_v2/campus_lafayette_v2_records.jsonl",
  "csv": "outputs/osm_generation_dataset_corpus_v2/campus_lafayette_v2_index.csv",
  "masks_npz": "outputs/osm_generation_dataset_corpus_v2/campus_lafayette_v2_footprint_masks.npz"
}
```

Proposal image dataset summary:

```json
{
  "dataset_version": "osm_proposal_image_v1",
  "name": "campus_lafayette_proposal_v1",
  "source_records": "outputs/osm_generation_dataset_corpus_v2/campus_lafayette_v2_records.jsonl",
  "count": 37,
  "train_count": 30,
  "val_count": 7,
  "classes": {
    "PUBLIC": 1,
    "RESIDENTIAL": 34,
    "COMMERCIAL": 2
  },
  "quality_statuses": {
    "pass": 37
  },
  "train_jsonl": "outputs/osm_proposal_image_dataset_v1/campus_lafayette_proposal_v1_train.jsonl",
  "val_jsonl": "outputs/osm_proposal_image_dataset_v1/campus_lafayette_proposal_v1_val.jsonl",
  "all_jsonl": "outputs/osm_proposal_image_dataset_v1/campus_lafayette_proposal_v1_all.jsonl"
}
```

Quality summary for the clean east2 run:

```json
{
  "count": 2,
  "pass": 1,
  "warn": 1,
  "fail": 0
}
```

DDPM overfit summary:

```json
{
  "train_count": 8,
  "val_count": 8,
  "best_val_noise_mse": 0.022264983505010605,
  "ckpt_best": "outputs/osm_proposal_image_ddpm_v1_overfit8/ckpt_best.pth",
  "metrics_csv": "outputs/osm_proposal_image_ddpm_v1_overfit8/metrics.csv"
}
```

## SDF Role

The current high-quality output path uses Hunyuan/retrieval for visible mesh
generation, but SDF is still useful and defensible in the pipeline:

- footprint correction as a signed distance field over the target polygon;
- fit scoring between target footprint and candidate/generated geometry;
- geometry audit for overhang, underfill, and footprint mismatch;
- residual SDF refinement of coarse/retrieved geometry;
- `abs(SDF)`/UDF-style extraction for hollow or open-shell buildings;
- paper framing: SDF is the intermediate geometric constraint representation,
  while image/mesh priors carry high-frequency visual detail.

## Generative Branch Status

The supervised proposal image generator produces cleaner structure than the
first diffusion attempt, but it is still blurry because direct RGB regression
leans toward conditional means.

The pixel-space DDPM branch trains and lowers noise MSE. An 8-example overfit
run reached best validation noise MSE 0.022264983505010605,
but pure samples still show RGB speckle. The new DDIM/reconstruction diagnostic
shows that the model can denoise partially corrupted real targets and recover
building mass/silhouette, but cannot yet generate clean proposal images from
pure noise.

Conclusion: do not spend more time simply extending the same tiny pixel-space
DDPM. The generative branch should move toward pretrained/latent/control-
conditioned diffusion or a much larger proposal-image dataset.

## Paper Direction

Best-fitting research direction:

```text
ControlCity-style geospatial conditioning
+ pretrained/latent diffusion backbone
+ Hunyuan/image-to-3D downstream
+ SDF geometry constraints/audits
```

Most relevant papers already noted:

- Nichol and Dhariwal, Improved Denoising Diffusion Probabilistic Models.
- Liu et al., One-2-3-45++.
- Wei, Vosselman, and Yang, BuilDiff.
- Zhou et al., ControlCity.
- Tze et al., PrITTI.
- Human-guided urban form generation using multimodal diffusion models.

## Current Recommendation

For near-term visible results, keep using:

```text
OSM -> retrieval/proposal image -> Hunyuan -> simplify -> heightfix -> audit
```

For the research novelty, develop:

```text
OSM/context/SDF constraints -> controlled latent proposal generation
```

The next implementation step should be to package a stronger controlled
proposal-image generator and keep SDF as the footprint/geometry constraint
layer, instead of forcing the weak scratch-trained DDPM to carry the whole
generative claim.
