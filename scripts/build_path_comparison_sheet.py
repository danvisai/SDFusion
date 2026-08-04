"""Build a 3-column visual comparison sheet across the asset paths.

Columns are:
  GT             - ground-truth SDF -> marching cubes mesh render
                   (reference; what any SDF-based path can at best replicate)
  Path D ceiling - v1 VQVAE round-trip (encode-decode of GT SDF) -> MC mesh render
                   (this is the best Stage 3a can ever match: its decoder is v1)
  Path B (3DGS)  - baked Gaussian Splats from gaussian_splats_v2/, single view

The first two columns are sliced from outputs/vqvae_ab_diagnostic_t03/visual_sheet.png
(which already rendered them via render_sdf at a consistent angle). The third column
crops one view from each asset's existing 4-view preview tile.

CPU-only; safe to run while Stage 3a is using the GPU.
"""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO = Path("/scratch/gilbreth/dsimhadr/GenerativeTowns/SDFusion")
DIAG = REPO / "outputs/vqvae_ab_diagnostic_t03"
GS_DIR = REPO / "data/BuildingNet_dataset_v0_1/gaussian_splats_v2"
OUT_DIR = REPO / "outputs/path_comparison_2026_05_27"

# Layout constants must match scripts/eval_vqvae_ab.py
CELL_W = 160
CELL_H = 160
MARGIN = 8
LABEL_W = 200
HEADER_H = 24

# Source visual sheet column indices
SRC_COL_GT = 0
SRC_COL_V1 = 1


def src_col_x(ci: int) -> int:
    return LABEL_W + ci * (CELL_W + MARGIN) + MARGIN


def row_y(ri: int) -> int:
    return HEADER_H + ri * (CELL_H + MARGIN) + MARGIN


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with (DIAG / "per_asset.csv").open() as f:
        ids = [row["id"] for row in csv.DictReader(f)]

    src = Image.open(DIAG / "visual_sheet.png").convert("RGB")
    n_rows = len(ids)
    out_n_cols = 3
    W = LABEL_W + out_n_cols * (CELL_W + MARGIN) + MARGIN
    H = HEADER_H + n_rows * (CELL_H + MARGIN) + MARGIN

    out = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    out_col_labels = ["GT (SDF->MC)", "Path D ceiling (v1 + MC)", "Path B (baked 3DGS)"]
    for ci, cl in enumerate(out_col_labels):
        x = LABEL_W + ci * (CELL_W + MARGIN) + MARGIN
        draw.text((x + 4, 4), cl, fill="black", font=font)

    for ri, asset_id in enumerate(ids):
        y = row_y(ri)
        draw.text((4, y + CELL_H // 2 - 6), asset_id[:24], fill="black", font=font)

        # Col 0 GT, col 1 v1 -> sliced from source sheet
        for out_ci, src_ci in [(0, SRC_COL_GT), (1, SRC_COL_V1)]:
            sx = src_col_x(src_ci)
            crop = src.crop((sx, y, sx + CELL_W, y + CELL_H))
            ox = LABEL_W + out_ci * (CELL_W + MARGIN) + MARGIN
            out.paste(crop, (ox, y))

        # Col 2: one view from the gsplat preview tile.
        # _preview.png is 768x768, 4 views in quadrants:
        #   top-left azim=0, top-right azim=90, bottom-left azim=180, bottom-right azim=270.
        # Take the top-left view (azim=0) for consistency with the SDF render.
        preview_path = GS_DIR / f"{asset_id}_preview.png"
        ox = LABEL_W + 2 * (CELL_W + MARGIN) + MARGIN
        if preview_path.exists():
            tile = Image.open(preview_path).convert("RGB")
            # crop top-left quadrant 0..384, 0..384
            view = tile.crop((0, 0, 384, 384)).resize((CELL_W, CELL_H), Image.LANCZOS)
            out.paste(view, (ox, y))
        else:
            draw.rectangle([ox, y, ox + CELL_W, y + CELL_H], outline="red", width=2)
            draw.text((ox + 8, y + CELL_H // 2 - 6), "missing", fill="red", font=font)

    out_path = OUT_DIR / "path_comparison.png"
    out.save(out_path)
    print(f"[*] wrote {out_path} ({W}x{H})")


if __name__ == "__main__":
    main()
