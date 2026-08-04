"""Build PNG visual reports for a quality-rerank A/B sweep."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


STATUS_COLORS = {
    "pass": (218, 246, 224),
    "warn": (255, 242, 204),
    "fail": (255, 224, 224),
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_dir", required=True)
    ap.add_argument("--out_dir", default=None)
    return ap.parse_args()


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


FONT_14 = font(14)
FONT_16 = font(16)
FONT_18 = font(18, bold=True)
FONT_22 = font(22, bold=True)
FONT_30 = font(30, bold=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def wrap_text(text: str, draw: ImageDraw.ImageDraw, width: int, text_font: ImageFont.ImageFont) -> list[str]:
    lines: list[str] = []
    for paragraph in str(text).splitlines() or [""]:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        line = words[0]
        for word in words[1:]:
            trial = f"{line} {word}"
            if draw.textbbox((0, 0), trial, font=text_font)[2] <= width:
                line = trial
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return lines


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    max_width: int,
    text_font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = (30, 33, 38),
    line_gap: int = 4,
) -> int:
    x, y = xy
    for line in wrap_text(text, draw, max_width, text_font):
        draw.text((x, y), line, font=text_font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=text_font)
        y += (bbox[3] - bbox[1]) + line_gap
    return y


def image_cell(
    path: Path,
    title: str,
    subtitle: str,
    size: tuple[int, int],
    border: tuple[int, int, int] = (190, 196, 205),
) -> Image.Image:
    w, h = size
    header_h = 58
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, w - 1, h - 1), outline=border, width=2)
    draw.rectangle((1, 1, w - 2, header_h), fill=(246, 248, 251))
    draw_wrapped(draw, (12, 8), title, w - 24, FONT_16)
    if subtitle:
        draw_wrapped(draw, (12, 32), subtitle, w - 24, FONT_14, fill=(86, 96, 110))

    body = (w - 20, h - header_h - 16)
    if path.exists():
        img = Image.open(path).convert("RGB")
        img.thumbnail(body, Image.Resampling.LANCZOS)
        x = (w - img.width) // 2
        y = header_h + 8 + (body[1] - img.height) // 2
        canvas.paste(img, (x, y))
    else:
        draw_wrapped(draw, (16, header_h + 24), f"Missing: {path}", w - 32, FONT_14, fill=(160, 45, 45))
    return canvas


def text_cell(
    title: str,
    lines: list[str],
    size: tuple[int, int],
    fill: tuple[int, int, int] = (255, 255, 255),
    border: tuple[int, int, int] = (190, 196, 205),
) -> Image.Image:
    w, h = size
    canvas = Image.new("RGB", (w, h), fill)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, w - 1, h - 1), outline=border, width=2)
    y = draw_wrapped(draw, (14, 12), title, w - 28, FONT_18)
    y += 8
    for line in lines:
        y = draw_wrapped(draw, (14, y), line, w - 28, FONT_14, fill=(50, 56, 65))
        y += 6
    return canvas


def paste_grid(canvas: Image.Image, cells: list[list[Image.Image]], start_y: int, gap: int = 12) -> None:
    y = start_y
    for row in cells:
        x = gap
        row_h = max(cell.height for cell in row)
        for cell in row:
            canvas.paste(cell, (x, y))
            x += cell.width + gap
        y += row_h + gap


def make_overview(sweep_dir: Path, out_dir: Path) -> Path:
    summary = read_json(sweep_dir / "sweep_summary.json")
    tile_rows = read_csv(sweep_dir / "sweep_tile_summary.csv")

    label_size = (260, 230)
    map_size = (260, 230)
    audit_size = (310, 230)
    gap = 12
    row_w = label_size[0] + (map_size[0] * 2) + (audit_size[0] * 2) + gap * 6
    header_h = 116
    row_h = label_size[1]
    canvas = Image.new("RGB", (row_w, header_h + len(tile_rows) * (row_h + gap) + gap), (242, 244, 247))
    draw = ImageDraw.Draw(canvas)

    geom = summary["geometry"]
    qual = summary["quality"]
    title = "Quality Rerank A/B Sweep Overview"
    subtitle = (
        f"Geometry: {geom['pass']}/{summary['building_count']} pass "
        f"({geom['pass_rate'] * 100:.1f}%) | "
        f"Quality: {qual['pass']}/{summary['building_count']} pass "
        f"({qual['pass_rate'] * 100:.1f}%) | "
        f"choice changes: {summary['choice_changed_count']}/{summary['building_count']}"
    )
    draw.text((18, 18), title, font=FONT_30, fill=(24, 28, 35))
    draw.text((20, 62), subtitle, font=FONT_18, fill=(55, 64, 75))

    rows: list[list[Image.Image]] = []
    for row in tile_rows:
        tile = row["tile"]
        geom_pass = int(row["geometry_pass"])
        geom_fail = int(row["geometry_fail"])
        qual_pass = int(row["quality_pass"])
        qual_fail = int(row["quality_fail"])
        delta = qual_pass - geom_pass
        fill = (224, 246, 229) if delta > 0 else (255, 255, 255)
        if delta < 0:
            fill = (255, 226, 226)
        label = text_cell(
            tile,
            [
                f"bbox: {row['bbox']}",
                f"geometry: {geom_pass} pass, {geom_fail} fail",
                f"quality: {qual_pass} pass, {qual_fail} fail",
                f"changed: {row['choice_changed_count']}",
                f"delta: {delta:+d} pass",
            ],
            label_size,
            fill=fill,
        )
        tile_dir = sweep_dir / tile
        rows.append(
            [
                label,
                image_cell(tile_dir / "geometry_rerank" / "heightfix" / "osm_map_output_houses.png", "Geometry", "output map", map_size),
                image_cell(tile_dir / "quality_rerank" / "heightfix" / "osm_map_output_houses.png", "Quality", "output map", map_size),
                image_cell(tile_dir / "geometry_rerank" / "quality" / "generation_quality_audit_sheet.png", "Geometry", "audit sheet", audit_size),
                image_cell(tile_dir / "quality_rerank" / "quality" / "generation_quality_audit_sheet.png", "Quality", "audit sheet", audit_size),
            ]
        )

    paste_grid(canvas, rows, header_h, gap=gap)
    out_path = out_dir / "quality_rerank_sweep_overview.png"
    canvas.save(out_path)
    return out_path


def load_log_by_osm(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open() as f:
        rows = json.load(f)
    return {str(row.get("osm_id")): row for row in rows}


def render_path_from_log(sweep_dir: Path, tile: str, arm: str, osm_id: str) -> Path | None:
    log_path = sweep_dir / tile / arm / "heightfix" / "osm_hunyuan_scene.log.json"
    row = load_log_by_osm(log_path).get(osm_id)
    if not row:
        return None
    placed = row.get("placed_render_png")
    if not placed:
        return None
    return Path(placed)


def candidate_input_path(sweep_dir: Path, tile: str, arm: str, osm_id: str, candidate: str) -> Path:
    root = sweep_dir / tile / arm / "gen" / "hunyuan_inputs"
    matches = sorted(root.glob(f"*{osm_id}_{candidate}_retrieved_input.png"))
    return matches[0] if matches else root / f"{osm_id}_{candidate}_retrieved_input.png"


def make_changed_choices(sweep_dir: Path, out_dir: Path) -> Path:
    rows = [row for row in read_csv(sweep_dir / "sweep_ab_comparison.csv") if row.get("choice_changed") == "1"]
    cell_h = 246
    info_size = (330, cell_h)
    img_size = (250, cell_h)
    gap = 12
    row_w = info_size[0] + img_size[0] * 4 + gap * 6
    header_h = 104
    canvas = Image.new("RGB", (row_w, header_h + len(rows) * (cell_h + gap) + gap), (242, 244, 247))
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 18), "Changed Candidate Choices", font=FONT_30, fill=(24, 28, 35))
    draw.text(
        (20, 62),
        "Rows where the quality-aware policy selected a different candidate than geometry rerank.",
        font=FONT_18,
        fill=(55, 64, 75),
    )

    grid: list[list[Image.Image]] = []
    for row in rows:
        tile = row["tile"]
        osm_id = row["osm_id"]
        geom_status = row["geometry_status"]
        qual_status = row["quality_status"]
        geom_candidate = row["geometry_candidate"]
        qual_candidate = row["quality_candidate"]
        geom_fill = STATUS_COLORS.get(geom_status, (255, 255, 255))
        qual_fill = STATUS_COLORS.get(qual_status, (255, 255, 255))
        if geom_status == "fail" and qual_status == "pass":
            verdict = "fixed by quality rerank"
        elif geom_status == "pass" and qual_status == "fail":
            verdict = "regressed"
        elif geom_status == qual_status:
            verdict = f"status unchanged: {qual_status}"
        else:
            verdict = f"{geom_status} -> {qual_status}"
        info = text_cell(
            f"{tile} / {osm_id}",
            [
                row["class"],
                f"geometry: {geom_candidate}",
                f"quality: {qual_candidate}",
                f"status: {geom_status} -> {qual_status}",
                f"flags: {row.get('geometry_flags') or '-'} -> {row.get('quality_flags') or '-'}",
                verdict,
            ],
            info_size,
            fill=(224, 246, 229) if verdict.startswith("fixed") else (255, 255, 255),
        )
        geom_input = candidate_input_path(sweep_dir, tile, "geometry_rerank", osm_id, geom_candidate)
        qual_input = candidate_input_path(sweep_dir, tile, "quality_rerank", osm_id, qual_candidate)
        geom_render = render_path_from_log(sweep_dir, tile, "geometry_rerank", osm_id)
        qual_render = render_path_from_log(sweep_dir, tile, "quality_rerank", osm_id)
        grid.append(
            [
                info,
                image_cell(geom_input, "Geometry", "conditioning input", img_size, border=geom_fill),
                image_cell(geom_render or Path("__missing__"), "Geometry", "placed output", img_size, border=geom_fill),
                image_cell(qual_input, "Quality", "conditioning input", img_size, border=qual_fill),
                image_cell(qual_render or Path("__missing__"), "Quality", "placed output", img_size, border=qual_fill),
            ]
        )

    if not rows:
        grid.append([text_cell("No changed choices", ["All quality choices matched geometry rerank."], (row_w - gap * 2, cell_h))])

    paste_grid(canvas, grid, header_h, gap=gap)
    out_path = out_dir / "quality_rerank_changed_choices.png"
    canvas.save(out_path)
    return out_path


def make_markdown(sweep_dir: Path, out_dir: Path, overview: Path, changed: Path) -> Path:
    summary = read_json(sweep_dir / "sweep_summary.json")
    geom = summary["geometry"]
    qual = summary["quality"]
    report = out_dir / "quality_rerank_visual_report.md"
    lines = [
        "# Quality Rerank Visual Report",
        "",
        "## Summary",
        f"- geometry rerank: {geom['pass']}/{summary['building_count']} pass",
        f"- quality rerank: {qual['pass']}/{summary['building_count']} pass",
        f"- choice changes: {summary['choice_changed_count']}/{summary['building_count']}",
        f"- net pass delta: {qual['pass'] - geom['pass']:+d}",
        "",
        "## Visual Outputs",
        f"- overview: `{overview.relative_to(sweep_dir)}`",
        f"- changed choices: `{changed.relative_to(sweep_dir)}`",
        "",
        "## Reading Notes",
        "- The overview image compares output maps and audit sheets tile by tile.",
        "- The changed-choice image shows only cases where the quality-aware policy selected a different candidate.",
        "- Green/red cell borders indicate pass/fail audit status for each arm.",
    ]
    report.write_text("\n".join(lines) + "\n")
    return report


def main() -> None:
    args = parse_args()
    sweep_dir = Path(args.sweep_dir)
    out_dir = Path(args.out_dir) if args.out_dir else sweep_dir / "visual_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    overview = make_overview(sweep_dir, out_dir)
    changed = make_changed_choices(sweep_dir, out_dir)
    report = make_markdown(sweep_dir, out_dir, overview, changed)

    print(f"[visual-report] overview: {overview}")
    print(f"[visual-report] changed choices: {changed}")
    print(f"[visual-report] report: {report}")


if __name__ == "__main__":
    main()
