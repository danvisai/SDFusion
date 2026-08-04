"""Visualize OSM map input, selected footprints, and retrieval choices.

This complements scripts/osm_hunyuan_pipeline_smoke.py. It does not rerun
Hunyuan. It reads the OSM JSON and pipeline log, recomputes top-k retrieval
candidates, renders candidate OBJ views, and writes map-level visualizations.

Example:
    env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
        scripts/osm_pipeline_map_choices.py \
        --osm_json outputs/osm_pipeline_smoke/osm_input.json \
        --pipeline_log outputs/osm_pipeline_smoke/osm_hunyuan_scene.log.json \
        --out_dir outputs/osm_pipeline_smoke
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scene.run_demo import _resolve_subtype_idx, _resolve_top, load_retrieval, rasterize_polygon
from scene.gsplat_common import load_inria_ply
from scene.gsplat_renderer import render_gsplat_topdown
from scripts.hunyuan_building_mesh_smoke import render_mesh_png, title_cell
from scripts.render_buildingnet_objfiles import load_obj_as_trimesh, make_renderer, render_one


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--osm_json", required=True)
    ap.add_argument("--pipeline_log", required=True)
    ap.add_argument("--out_dir", default="outputs/osm_pipeline_smoke")
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--map_size", type=int, default=512)
    ap.add_argument("--cell_size", type=int, default=256)
    ap.add_argument("--index_dir", default="data/BuildingNet_dataset_v0_1/retrieval_index")
    ap.add_argument("--retrieval_ckpt", default="Logs_GT/retrieval_footprint_full/ckpt_best.pth")
    ap.add_argument("--obj_dir", default="data/BuildingNet_dataset_v0_1/OBJ_MODELS")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def selected_buildings(payload: dict, log_rows: list[dict]) -> list[dict]:
    by_id = {str(b.get("id")): b for b in payload.get("buildings", [])}
    out = []
    for row in log_rows:
        b = dict(by_id[str(row["osm_id"])])
        b["_pipeline"] = row
        out.append(b)
    return out


def bounds_for(buildings: list[dict]) -> tuple[float, float, float, float]:
    pts = []
    for b in buildings:
        pts.extend(b.get("polygon", []))
    arr = np.asarray(pts, dtype=np.float64)
    return float(arr[:, 0].min()), float(arr[:, 1].min()), float(arr[:, 0].max()), float(arr[:, 1].max())


def mapper(buildings: list[dict], size: int, pad: int = 26):
    xmin, ymin, xmax, ymax = bounds_for(buildings)
    w = max(xmax - xmin, 1e-6)
    h = max(ymax - ymin, 1e-6)
    scale = (size - 2 * pad) / max(w, h)
    ox = (size - w * scale) / 2.0
    oy = (size - h * scale) / 2.0

    def to_px(point):
        x, y = point
        return (ox + (x - xmin) * scale, size - (oy + (y - ymin) * scale))

    return to_px


def draw_map(
    buildings: list[dict],
    selected: list[dict],
    size: int,
    title: str,
    mode: str,
) -> Image.Image:
    selected_ids = {str(b["id"]) for b in selected}
    idx_by_id = {str(b["id"]): i for i, b in enumerate(selected)}
    to_px = mapper(buildings, size)
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    for b in buildings:
        pts = [to_px(p) for p in b["polygon"]]
        is_sel = str(b["id"]) in selected_ids
        if mode == "input":
            fill = (220, 226, 232)
            outline = (128, 143, 156)
        elif mode == "selected":
            fill = (43, 132, 86) if is_sel else (224, 224, 224)
            outline = (14, 92, 55) if is_sel else (170, 170, 170)
        else:
            fill = (221, 236, 225) if is_sel else (232, 232, 232)
            outline = (42, 122, 78) if is_sel else (176, 176, 176)
        draw.polygon(pts, fill=fill, outline=outline)
        if is_sel:
            arr = np.asarray(pts)
            cx, cy = arr[:, 0].mean(), arr[:, 1].mean()
            label = str(idx_by_id[str(b["id"])])
            r = 12
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(20, 66, 122), outline="white", width=2)
            draw.text((cx - 4, cy - 7), label, fill="white")
    return title_cell(img, title, size)


def draw_output_map_with_houses(buildings: list[dict], selected: list[dict], size: int) -> Image.Image:
    cell = draw_map(buildings, selected, size, "output map + placed houses", "output")
    body = cell.crop((0, 28, size, size + 28)).convert("RGB")
    to_px = mapper(buildings, size)
    for i, b in enumerate(selected):
        row = b["_pipeline"]
        thumb_key = (
            row.get("placed_front_png")
            if row.get("asset_format") == "gsplat"
            else row.get("placed_render_png")
        )
        if not thumb_key:
            continue
        thumb_path = Path(thumb_key)
        if not thumb_path.exists():
            continue
        pts = np.asarray([to_px(p) for p in b["polygon"]], dtype=np.float64)
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        footprint_extent = max(pts[:, 0].max() - pts[:, 0].min(), pts[:, 1].max() - pts[:, 1].min())
        thumb_size = int(np.clip(footprint_extent * 0.9, 56, 120))
        thumb = Image.open(thumb_path).convert("RGB").resize((thumb_size, thumb_size), Image.Resampling.BICUBIC)
        x = int(cx - thumb_size / 2)
        y = int(cy - thumb_size / 2)
        body.paste(thumb, (x, y))
        ImageDraw.Draw(body).text((x + 3, y + 3), str(i), fill=(20, 66, 122))
    return title_cell(body, "output map + placed houses", size)


@torch.no_grad()
def retrieve_topk(building: dict, fp_mask: np.ndarray, retrieval: dict, device, top_k: int) -> list[tuple[str, float]]:
    full_class = building["class"]
    sub_idx = _resolve_subtype_idx(full_class, retrieval["subtype_to_idx"])
    top = _resolve_top(full_class)
    top_idx = retrieval["top_to_idx"].get(top, 0)
    fp = torch.from_numpy(fp_mask)[None, None].to(device).float()
    cls = torch.tensor([sub_idx], dtype=torch.long, device=device)
    emb, _ = retrieval["model"](fp, cls)
    emb_np = emb.cpu().numpy()[0]
    sims = retrieval["train_emb"] @ emb_np
    same_top = retrieval["train_top_ids"] == top_idx
    sims = np.where(same_top, sims, -1e9)
    order = np.argsort(-sims)[:top_k]
    return [(str(retrieval["train_ids"][j]), float(sims[j])) for j in order]


def footprint_cell(mask: np.ndarray, title: str, size: int) -> Image.Image:
    img = Image.new("RGB", (size, size), "white")
    mask_img = Image.fromarray(((mask > 0) * 255).astype(np.uint8), "L").resize((size, size), Image.Resampling.NEAREST)
    tint = Image.new("RGB", (size, size), (36, 112, 172))
    img.paste(tint, mask=mask_img)
    ImageDraw.Draw(img).rectangle((0, 0, size - 1, size - 1), outline=(205, 205, 205))
    return title_cell(img, title, size)


def render_obj_cell(obj_id: str, title: str, obj_dir: Path, renderer, device, size: int, out_dir: Path) -> Image.Image:
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{obj_id}.png"
    if png_path.exists():
        img = Image.open(png_path).convert("RGB")
    else:
        obj_path = obj_dir / f"{obj_id}.obj"
        mesh = load_obj_as_trimesh(str(obj_path))
        if mesh is None:
            img = Image.new("RGB", (size, size), "white")
            ImageDraw.Draw(img).text((8, 8), "missing OBJ", fill=(160, 0, 0))
        else:
            rgb = render_one(mesh, renderer, device)
            img = Image.fromarray(rgb, "RGB")
            img.save(png_path, optimize=True)
    return title_cell(img, title, size)


def logged_or_computed_candidates(
    building: dict,
    mask: np.ndarray,
    retrieval: dict,
    device,
    top_k: int,
) -> list[dict[str, object]]:
    logged = building["_pipeline"].get("retrieval_candidates")
    if logged:
        logged = sorted(logged, key=lambda r: int(r["rank"]))[:top_k]
        return [dict(r) for r in logged]
    return [
        {
            "rank": rank,
            "candidate_id": obj_id,
            "retrieval_score": score,
            "rerank_score": score,
            "chosen": int(obj_id == building["_pipeline"]["retrieved_id"]),
        }
        for rank, (obj_id, score) in enumerate(retrieve_topk(building, mask, retrieval, device, top_k), start=1)
    ]


def candidate_title(candidate: dict[str, object]) -> str:
    obj_id = str(candidate["candidate_id"])
    retrieval_score = float(candidate.get("retrieval_score", candidate.get("score", 0.0)))
    rerank_score = float(candidate.get("rerank_score", retrieval_score))
    mark = "*" if int(candidate.get("chosen", 0)) else ""
    return f"{mark}{obj_id} r={retrieval_score:.3f} rr={rerank_score:.3f}"


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir = out_dir / "retrieval_choice_renders"

    payload = json.load(open(args.osm_json))
    log_rows = json.load(open(args.pipeline_log))
    all_buildings = payload.get("buildings", [])
    chosen = selected_buildings(payload, log_rows)

    input_map = draw_map(all_buildings, chosen, args.map_size, "OSM map input: all footprints", "input")
    selected_map = draw_map(all_buildings, chosen, args.map_size, "selected footprints for pipeline", "selected")
    output_map = draw_output_map_with_houses(all_buildings, chosen, args.map_size)
    input_map.save(out_dir / "osm_map_input.png", optimize=True)
    selected_map.save(out_dir / "osm_map_selected.png", optimize=True)
    output_map.save(out_dir / "osm_map_output_houses.png", optimize=True)

    # Detect asset format from the pipeline log; default to mesh if not present.
    asset_format = next(
        (str(r.get("asset_format", "mesh")) for r in log_rows if r.get("asset_format")),
        "mesh",
    )
    if asset_format == "gsplat":
        scene_path = out_dir / "osm_3dgs_scene.ply"
        scene_render = None
        if scene_path.exists():
            scene_g = load_inria_ply(str(scene_path), device=args.device)
            top_img = render_gsplat_topdown(scene_g, image_size=args.map_size)
            scene_render = title_cell(top_img, "3DGS scene (top-down)", args.map_size)
            scene_render.crop((0, 0, args.map_size, args.map_size + 28)).save(
                out_dir / "osm_3dgs_scene_render.png", optimize=True,
            )
    else:
        scene_path = out_dir / "osm_hunyuan_scene.obj"
        scene_render = None
        if scene_path.exists():
            scene_mesh = trimesh.load(scene_path, force="mesh", process=False)
            scene_render = title_cell(render_mesh_png(scene_mesh, args.map_size), "oblique output scene OBJ", args.map_size)
            scene_render.crop((0, 0, args.map_size, args.map_size + 28)).save(
                out_dir / "osm_hunyuan_scene_render.png", optimize=True,
            )

    device = torch.device(args.device)
    retrieval = load_retrieval(Path(args.index_dir), args.retrieval_ckpt, device)
    obj_renderer = make_renderer(device, image_size=args.cell_size)
    obj_dir = Path(args.obj_dir)

    rows = []
    sheet_rows = []
    for i, b in enumerate(chosen):
        mask = rasterize_polygon(b["polygon"], res=64)
        candidates = logged_or_computed_candidates(b, mask, retrieval, device, args.top_k)
        row_cells = [footprint_cell(mask, f"OSM {i} footprint", args.cell_size)]
        for candidate in candidates:
            obj_id = str(candidate["candidate_id"])
            row_cells.append(render_obj_cell(
                obj_id,
                candidate_title(candidate),
                obj_dir,
                obj_renderer,
                device,
                args.cell_size,
                candidate_dir,
            ))
            rows.append({
                "index": i,
                "osm_id": b["id"],
                "rank": candidate["rank"],
                "candidate_id": obj_id,
                "retrieval_score": f"{float(candidate.get('retrieval_score', 0.0)):.6f}",
                "rerank_score": f"{float(candidate.get('rerank_score', candidate.get('retrieval_score', 0.0))):.6f}",
                "chosen_by_pipeline": int(candidate.get("chosen", obj_id == b["_pipeline"]["retrieved_id"])),
            })
        if asset_format == "gsplat":
            raw_key = b["_pipeline"].get("canonical_view_png")
            placed_key = b["_pipeline"].get("placed_front_png")
            raw_label = "baked 3DGS"
            placed_label = "placed 3DGS"
        else:
            raw_key = b["_pipeline"].get("hunyuan_raw_render_png")
            placed_key = b["_pipeline"].get("placed_render_png")
            raw_label = "generated mesh"
            placed_label = "placed output"
        raw_img = Image.open(raw_key).convert("RGB")
        placed_img = Image.open(placed_key).convert("RGB")
        row_cells.append(title_cell(raw_img, raw_label, args.cell_size))
        row_cells.append(title_cell(placed_img, placed_label, args.cell_size))
        sheet_rows.append(row_cells)

    csv_path = out_dir / "osm_retrieval_choices.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    cols = 1 + args.top_k + 2
    width = cols * args.cell_size
    top_h = args.map_size + 28
    row_h = args.cell_size + 28
    sheet = Image.new("RGB", (width, top_h + len(sheet_rows) * row_h), "white")
    top_cells = [input_map, selected_map, output_map]
    map_w = width // 3
    for c, cell in enumerate(top_cells):
        sheet.paste(cell.resize((map_w, top_h), Image.Resampling.BICUBIC), (c * map_w, 0))
    y = top_h
    for cells in sheet_rows:
        for c, cell in enumerate(cells):
            sheet.paste(cell, (c * args.cell_size, y))
        y += row_h
    sheet_path = out_dir / "osm_map_choices_sheet.png"
    sheet.save(sheet_path, optimize=True)

    print(f"[viz] input map:    {out_dir / 'osm_map_input.png'}")
    print(f"[viz] selected map: {out_dir / 'osm_map_selected.png'}")
    print(f"[viz] output map:   {out_dir / 'osm_map_output_houses.png'}")
    print(f"[viz] choices csv:  {csv_path}")
    print(f"[viz] sheet:        {sheet_path}")


if __name__ == "__main__":
    main()
