"""Generate the current Codex pipeline report as Markdown and PDF."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
OUT_MD = DOCS / "CODEX_FULL_PIPELINE_REPORT_2026-05-12.md"
OUT_PDF = DOCS / "CODEX_FULL_PIPELINE_REPORT_2026-05-12.pdf"


def rel(path: Path) -> str:
    return str(path.relative_to(REPO))


def read_json(path: str) -> dict:
    p = REPO / path
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def wrap(text: str, width: int = 96) -> str:
    lines = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")
        elif paragraph.startswith("  ") or paragraph.startswith("- ") or paragraph.startswith("```"):
            lines.append(paragraph)
        else:
            lines.extend(textwrap.wrap(paragraph, width=width))
    return "\n".join(lines)


def add_text_page(pdf: PdfPages, title: str, body: str) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.07, 0.05, 0.86, 0.9])
    ax.axis("off")
    ax.text(0, 1.02, title, fontsize=18, fontweight="bold", va="top")
    ax.text(0, 0.96, wrap(body, 92), fontsize=9.7, va="top", family="monospace", linespacing=1.28)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_image_page(pdf: PdfPages, title: str, items: list[tuple[str, str]]) -> None:
    existing = [(label, REPO / path) for label, path in items if (REPO / path).exists()]
    if not existing:
        return
    rows = len(existing)
    fig, axes = plt.subplots(rows, 1, figsize=(8.5, 11))
    if rows == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.99)
    for ax, (label, path) in zip(axes, existing):
        ax.axis("off")
        img = Image.open(path).convert("RGB")
        ax.imshow(img)
        ax.set_title(f"{label}\n{rel(path)}", fontsize=9, loc="left")
    plt.tight_layout(rect=[0, 0, 1, 0.965])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def make_markdown() -> str:
    corpus = read_json("outputs/osm_generation_dataset_corpus_v2/campus_lafayette_v2_summary.json")
    proposals = read_json("outputs/osm_proposal_image_dataset_v1/campus_lafayette_proposal_v1_summary.json")
    quality = read_json("outputs/osm_generative_proposal_clean_east2_quality/generation_quality_summary.json")
    ddpm = read_json("outputs/osm_proposal_image_ddpm_v1_overfit8/summary.json")

    return f"""# Codex Full Pipeline Report - GenerativeTowns / SDFusion

Prepared by: Codex  
Date: 2026-05-12  
Repo: `{REPO}`

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
{json.dumps(corpus, indent=2)}
```

Proposal image dataset summary:

```json
{json.dumps(proposals, indent=2)}
```

Quality summary for the clean east2 run:

```json
{json.dumps(quality, indent=2)}
```

DDPM overfit summary:

```json
{json.dumps(ddpm, indent=2)}
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
run reached best validation noise MSE {ddpm.get("best_val_noise_mse", "unknown")},
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
"""


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(make_markdown())

    with PdfPages(OUT_PDF) as pdf:
        add_text_page(
            pdf,
            "GenerativeTowns / SDFusion - Current Pipeline Report",
            "Prepared by Codex on 2026-05-12.\n\n"
            "The project now has a working OSM-to-3D pipeline and a separate generative research branch. "
            "The strongest current production path is OSM footprint/context -> retrieval or proposal image -> "
            "Hunyuan3D -> simplification -> height/placement correction -> quality audit. SDF remains in the "
            "pipeline as a geometric constraint, fit/audit representation, and future residual refinement layer.",
        )
        add_text_page(
            pdf,
            "Pipeline Overview",
            "Current path:\n\n"
            "OSM vector/map input -> footprint records -> retrieval/quality rerank or proposal image -> "
            "Hunyuan3D raw mesh -> simplify -> world placement -> area-aware heightfix -> quality audit.\n\n"
            "SDF role:\n"
            "- footprint correction and target signed-distance field\n"
            "- fit scoring against target footprint\n"
            "- geometry audit for mismatch and overhang\n"
            "- possible residual SDF refinement branch\n"
            "- possible abs(SDF)-as-UDF extraction for hollow/open shells\n\n"
            "Current judgment: Hunyuan/retrieval carries visual mesh detail; SDF carries geometric constraint value.",
        )
        add_text_page(
            pdf,
            "Progress Snapshot",
            make_markdown().split("## Dataset / Artifact Status", 1)[1].split("## SDF Role", 1)[0],
        )
        add_image_page(
            pdf,
            "OSM Input to Placed Output",
            [
                ("Map-only OSM input", "outputs/osm_generative_proposal_clean_east2_heightfix/osm_map_input.png"),
                ("Selected footprints", "outputs/osm_generative_proposal_clean_east2_heightfix/osm_map_selected.png"),
                ("Map output with houses", "outputs/osm_generative_proposal_clean_east2_heightfix/osm_map_output_houses.png"),
                ("Rendered placed scene", "outputs/osm_generative_proposal_clean_east2_heightfix/osm_hunyuan_scene_render.png"),
            ],
        )
        add_image_page(
            pdf,
            "Candidate Choices and Quality Audit",
            [
                ("Retrieval/proposal choices", "outputs/osm_generative_proposal_clean_east2_heightfix/osm_map_choices_sheet.png"),
                ("Generation quality audit", "outputs/osm_generative_proposal_clean_east2_quality/generation_quality_audit_sheet.png"),
            ],
        )
        add_image_page(
            pdf,
            "Hunyuan and Retrieval Evidence",
            [
                ("Hunyuan retrieval rank-1 mini sheet", "outputs/hunyuan_retrieval_rank1_mini/hunyuan_building_smoke_sheet.png"),
                ("Clean east2 Hunyuan pipeline sheet", "outputs/osm_generative_proposal_clean_east2/osm_hunyuan_pipeline_sheet.png"),
            ],
        )
        add_image_page(
            pdf,
            "Proposal Image Models",
            [
                ("Proposal input sheet", "outputs/osm_generative_proposal_clean_east4/proposal_inputs_sheet.png"),
                ("Supervised proposal preview, sharp loss", "outputs/osm_proposal_image_generator_v1_sharp/val_preview_epoch_060.png"),
                ("Supervised continued preview", "outputs/osm_proposal_image_generator_v1_continued/val_preview_epoch_030.png"),
            ],
        )
        add_image_page(
            pdf,
            "DDPM Diagnostic",
            [
                ("DDPM overfit epoch 300", "outputs/osm_proposal_image_ddpm_v1_overfit8/sample_epoch_300.png"),
                ("DDIM pure sample from resumed checkpoint", "outputs/osm_proposal_image_ddpm_v1_overfit8_ddim_diag/sample_ddim_resume_epoch_300.png"),
                ("DDIM reconstruction from t=25", "outputs/osm_proposal_image_ddpm_v1_overfit8_ddim_diag/recon_t025_epoch_300.png"),
                ("DDIM reconstruction from t=100", "outputs/osm_proposal_image_ddpm_v1_overfit8_ddim_diag/recon_t100_epoch_300.png"),
            ],
        )
        add_text_page(
            pdf,
            "Recommendation",
            "Which paper direction suits our purpose?\n\n"
            "ControlCity-style conditioning is the closest match, but implemented through a practical pretrained/"
            "latent diffusion backbone rather than a tiny scratch-trained pixel DDPM. The pipeline should use "
            "OSM footprint/context/height/class channels, keep SDF as a geometric constraint and audit layer, "
            "generate cleaner proposal images, and pass those images to Hunyuan/image-to-3D.\n\n"
            "Immediate path: keep retrieval/supervised proposal generation for usable output quality. "
            "Research path: controlled latent diffusion with SDF/footprint constraints.",
        )

    print(json.dumps({"markdown": str(OUT_MD), "pdf": str(OUT_PDF)}, indent=2))


if __name__ == "__main__":
    main()
