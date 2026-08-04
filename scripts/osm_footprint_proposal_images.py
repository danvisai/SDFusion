"""Create footprint-conditioned building proposal images for Hunyuan.

This is a deterministic baseline for the generative branch. It does not use a
retrieved mesh as the image input; it draws a simple class/height/footprint
conditioned building concept that can be fed to image-to-3D. Later this module
can be replaced by a trained footprint-to-image generator while keeping the
same pipeline interface.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scene.run_demo import rasterize_polygon
from scripts.hunyuan_building_mesh_smoke import title_cell


PALETTES = {
    "RESIDENTIAL": {
        "wall": (198, 188, 171),
        "side": (159, 151, 140),
        "roof": (129, 62, 52),
        "trim": (238, 235, 226),
        "window": (72, 112, 132),
    },
    "COMMERCIAL": {
        "wall": (175, 182, 188),
        "side": (128, 139, 148),
        "roof": (70, 78, 86),
        "trim": (226, 230, 232),
        "window": (62, 105, 135),
    },
    "PUBLIC": {
        "wall": (188, 176, 154),
        "side": (146, 136, 119),
        "roof": (87, 94, 102),
        "trim": (235, 230, 214),
        "window": (63, 101, 129),
    },
    "RELIGIOUS": {
        "wall": (190, 181, 162),
        "side": (148, 140, 124),
        "roof": (92, 85, 78),
        "trim": (238, 232, 215),
        "window": (58, 96, 126),
    },
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--osm_json", required=True)
    ap.add_argument("--out_dir", default="outputs/osm_generative_proposals")
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--image_size", type=int, default=384)
    ap.add_argument("--detail", choices=["clean", "detailed"], default="clean")
    ap.add_argument("--include_footprint_inset", action="store_true")
    return ap.parse_args()


def choose_buildings(payload: dict, limit: int) -> list[dict]:
    buildings = [
        b for b in payload.get("buildings", [])
        if len(b.get("polygon", [])) >= 3 and float(b.get("area", 0.0)) > 1.0
    ]
    buildings.sort(key=lambda b: float(b.get("area", 0.0)), reverse=True)
    return buildings[:limit] if limit > 0 else buildings


def safe_stem(value: str) -> str:
    keep = []
    for ch in value:
        keep.append(ch if ch.isalnum() or ch in ("-", "_") else "_")
    return "".join(keep).strip("_")[:96] or "building"


def class_top(building_class: str) -> str:
    for top in ("RESIDENTIAL", "COMMERCIAL", "PUBLIC", "RELIGIOUS"):
        if building_class.startswith(top):
            return top
    return "RESIDENTIAL"


def polygon_features(building: dict) -> dict[str, float]:
    poly = np.asarray(building["polygon"], dtype=np.float64)
    ext = poly.max(axis=0) - poly.min(axis=0)
    area = float(building.get("area", 0.0) or 0.0)
    height = float(building.get("height", 8.0) or 8.0)
    return {
        "width": float(ext[0]),
        "depth": float(ext[1]),
        "aspect": float(max(ext[0], ext[1]) / max(min(ext[0], ext[1]), 1e-6)),
        "area": area,
        "height": height,
        "floors": float(max(1, round(height / 3.5))),
    }


def draw_shadow(draw: ImageDraw.ImageDraw, pts: list[tuple[float, float]], offset: tuple[int, int]) -> None:
    ox, oy = offset
    draw.polygon([(x + ox, y + oy) for x, y in pts], fill=(210, 214, 216))


def rect_points(cx: float, cy: float, w: float, d: float) -> list[tuple[float, float]]:
    return [
        (cx - w / 2, cy - d / 2),
        (cx + w / 2, cy - d / 2),
        (cx + w / 2, cy + d / 2),
        (cx - w / 2, cy + d / 2),
    ]


def draw_windows(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    floors: int,
    cols: int,
    color,
    detail: str,
) -> None:
    x0, y0, x1, y1 = box
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    if floors <= 0 or cols <= 0:
        return
    if (x1 - x0) < 16 or (y1 - y0) < 16:
        return
    if detail == "clean":
        for r in range(floors):
            y = y0 + (r + 0.5) * ((y1 - y0) / max(floors, 1))
            draw.rectangle((x0 + 6, y - 2, x1 - 6, y + 2), fill=tuple(max(0, c - 8) for c in color))
        return
    gap_x = (x1 - x0) / max(cols, 1)
    gap_y = (y1 - y0) / max(floors, 1)
    for r in range(floors):
        for c in range(cols):
            wx = x0 + c * gap_x + gap_x * 0.30
            wy = y0 + r * gap_y + gap_y * 0.30
            ww = max(4, gap_x * 0.34)
            wh = max(5, gap_y * 0.28)
            draw.rounded_rectangle((wx, wy, wx + ww, wy + wh), radius=1, fill=color)


def proposal_image(
    building: dict,
    image_size: int = 384,
    detail: str = "clean",
    include_footprint_inset: bool = False,
) -> Image.Image:
    top = class_top(str(building.get("class", "")))
    palette = PALETTES.get(top, PALETTES["RESIDENTIAL"])
    feats = polygon_features(building)
    floors = int(np.clip(feats["floors"], 1, 8))
    aspect = float(np.clip(feats["aspect"], 0.65, 3.0))

    scale = 3
    work_size = image_size * scale
    img = Image.new("RGB", (work_size, work_size), (245, 247, 248))
    draw = ImageDraw.Draw(img)
    cx = work_size * 0.50
    cy = work_size * 0.58
    base_w = work_size * (0.42 if aspect < 1.4 else 0.52)
    base_d = base_w / aspect
    base_d = float(np.clip(base_d, work_size * 0.18, work_size * 0.34))
    height_px = float(np.clip(scale * (34 + floors * 15), work_size * 0.16, work_size * 0.48))
    skew = work_size * 0.12
    rise = work_size * 0.075

    roof = rect_points(cx, cy - height_px, base_w, base_d)
    roof_iso = [(x + (skew if i in (1, 2) else 0), y - (rise if i in (0, 1) else 0)) for i, (x, y) in enumerate(roof)]
    front = [
        roof_iso[3],
        roof_iso[2],
        (roof_iso[2][0], roof_iso[2][1] + height_px),
        (roof_iso[3][0], roof_iso[3][1] + height_px),
    ]
    side = [
        roof_iso[2],
        roof_iso[1],
        (roof_iso[1][0], roof_iso[1][1] + height_px),
        (roof_iso[2][0], roof_iso[2][1] + height_px),
    ]

    draw_shadow(draw, front, (16 * scale, 16 * scale))
    draw.polygon(front, fill=palette["wall"], outline=(92, 96, 98))
    draw.polygon(side, fill=palette["side"], outline=(92, 96, 98))
    draw.polygon(roof_iso, fill=palette["roof"], outline=(82, 72, 68))

    front_box = (
        min(front[0][0], front[1][0]) + 16 * scale,
        min(front[0][1], front[1][1]) + 14 * scale,
        max(front[2][0], front[3][0]) - 16 * scale,
        max(front[2][1], front[3][1]) - 16 * scale,
    )
    side_box = (
        min(side[0][0], side[1][0]) + 10 * scale,
        min(side[0][1], side[1][1]) + 14 * scale,
        max(side[2][0], side[3][0]) - 10 * scale,
        max(side[2][1], side[3][1]) - 16 * scale,
    )
    cols = int(np.clip(round(base_w / (48 * scale)), 2, 6))
    draw_windows(draw, front_box, floors, cols, palette["window"], detail)
    draw_windows(draw, side_box, floors, max(1, cols - 2), tuple(max(0, c - 20) for c in palette["window"]), detail)

    if top == "RESIDENTIAL":
        door_w = max(12, base_w * 0.08)
        door_h = max(22, height_px * 0.18)
        if detail == "detailed":
            draw.rectangle((cx - door_w / 2, cy + height_px * 0.20 - door_h, cx + door_w / 2, cy + height_px * 0.20), fill=(105, 75, 58))
    elif top == "RELIGIOUS":
        spire_x = cx - base_w * 0.28
        spire = [(spire_x - 18 * scale, roof_iso[0][1]), (spire_x + 18 * scale, roof_iso[0][1]), (spire_x, roof_iso[0][1] - 58 * scale)]
        draw.polygon(spire, fill=palette["roof"], outline=(82, 72, 68))
    elif top in {"COMMERCIAL", "PUBLIC"}:
        band_y = front_box[3] - 24 * scale
        draw.rectangle((front_box[0], band_y, front_box[2], band_y + 12 * scale), fill=palette["trim"])

    img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
    if include_footprint_inset:
        mask = rasterize_polygon(building["polygon"], res=64)
        mask_img = Image.fromarray((mask > 0).astype(np.uint8) * 190, "L").resize((86, 86), Image.Resampling.NEAREST)
        inset = Image.new("RGB", (96, 96), (255, 255, 255))
        inset_draw = ImageDraw.Draw(inset)
        inset_draw.rectangle((0, 0, 95, 95), outline=(160, 166, 172))
        tint = Image.new("RGB", (86, 86), (42, 106, 150))
        inset.paste(tint, (5, 5), mask_img)
        img.paste(inset, (image_size - 110, image_size - 110))
    return img


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    proposal_dir = out_dir / "proposal_inputs"
    proposal_dir.mkdir(parents=True, exist_ok=True)

    payload = json.load(open(args.osm_json))
    buildings = choose_buildings(payload, args.limit)
    rows = []
    cells = []
    for i, building in enumerate(buildings):
        osm_id = safe_stem(str(building.get("id", f"osm_{i}")))
        img = proposal_image(
            building,
            args.image_size,
            detail=args.detail,
            include_footprint_inset=args.include_footprint_inset,
        )
        path = proposal_dir / f"{i:02d}_{osm_id}_proposal.png"
        img.save(path, optimize=True)
        feats = polygon_features(building)
        rows.append({
            "index": i,
            "osm_id": building.get("id"),
            "class": building.get("class"),
            "area_m2": f"{feats['area']:.2f}",
            "height_m": f"{feats['height']:.2f}",
            "bbox_aspect": f"{feats['aspect']:.3f}",
            "floors": int(feats["floors"]),
            "proposal_png": str(path),
        })
        cells.append(title_cell(img, f"{i} {building.get('class')}", args.image_size))

    if not rows:
        raise SystemExit("No usable OSM buildings found.")

    csv_path = out_dir / "proposal_inputs.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    cell_h = args.image_size + 28
    cols = min(4, len(cells))
    sheet = Image.new("RGB", (cols * args.image_size, int(np.ceil(len(cells) / cols)) * cell_h), "white")
    for i, cell in enumerate(cells):
        sheet.paste(cell, ((i % cols) * args.image_size, (i // cols) * cell_h))
    sheet_path = out_dir / "proposal_inputs_sheet.png"
    sheet.save(sheet_path, optimize=True)
    print(f"[proposal] csv:   {csv_path}", flush=True)
    print(f"[proposal] sheet: {sheet_path}", flush=True)


if __name__ == "__main__":
    main()
