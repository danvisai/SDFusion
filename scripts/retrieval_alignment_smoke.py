from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import h5py
import numpy as np
from PIL import Image, ImageDraw
import torch
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.render_buildingnet_objfiles import load_obj_as_trimesh, make_renderer, render_one


def load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def load_h5_frame(data_root: Path, model_id: str, res: int = 64) -> dict[str, np.ndarray]:
    h5_path = data_root / f"resolution_{res}" / model_id / "ori_sample_grid.h5"
    with h5py.File(h5_path, "r") as f:
        sdf = np.asarray(f["pc_sdf_sample"], dtype=np.float32).reshape(res, res, res)
        footprint = np.asarray(f["footprint"], dtype=np.uint8)[0] > 0
        sdf_params = np.asarray(f["sdf_params"], dtype=np.float32)
        norm_params = np.asarray(f["norm_params"], dtype=np.float32)
    return {
        "sdf": sdf,
        "footprint": footprint,
        "sdf_params": sdf_params,
        "norm_params": norm_params,
    }


def load_normalized_obj(data_root: Path, model_id: str, norm_params: np.ndarray) -> trimesh.Trimesh:
    obj_path = data_root / "OBJ_MODELS" / f"{model_id}.obj"
    mesh = load_obj_as_trimesh(str(obj_path))
    if mesh is None or len(mesh.vertices) == 0:
        raise ValueError(f"empty OBJ: {obj_path}")
    mesh = mesh.copy()
    centroid = norm_params[:3].astype(np.float64)
    scale = float(norm_params[3])
    mesh.vertices = (np.asarray(mesh.vertices, dtype=np.float64) - centroid) / scale
    return trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)


def grid_bbox_from_mask(mask: np.ndarray, sdf_params: np.ndarray) -> dict[str, float]:
    rows, cols = np.where(mask)
    if len(rows) == 0:
        raise ValueError("empty target footprint")
    res = mask.shape[0]
    xmin, ymin, zmin, xmax, ymax, zmax = sdf_params.tolist()
    col0 = float(cols.min())
    col1 = float(cols.max() + 1)
    row0 = float(rows.min())
    row1 = float(rows.max() + 1)
    return {
        "xmin": xmin + (col0 / res) * (xmax - xmin),
        "xmax": xmin + (col1 / res) * (xmax - xmin),
        "zmin": zmin + (row0 / res) * (zmax - zmin),
        "zmax": zmin + (row1 / res) * (zmax - zmin),
        "ymin": ymin,
        "ymax": ymax,
    }


def sdf_inside_bbox(sdf: np.ndarray, sdf_params: np.ndarray, iso: float = 0.0) -> dict[str, float]:
    inside = sdf <= iso
    zz, yy, xx = np.where(inside)
    if len(xx) == 0:
        return grid_bbox_from_mask(inside.any(axis=1), sdf_params)
    d, h, w = sdf.shape
    xmin, ymin, zmin, xmax, ymax, zmax = sdf_params.tolist()
    return {
        "xmin": xmin + (float(xx.min()) / max(w - 1, 1)) * (xmax - xmin),
        "xmax": xmin + (float(xx.max()) / max(w - 1, 1)) * (xmax - xmin),
        "ymin": ymin + (float(yy.min()) / max(h - 1, 1)) * (ymax - ymin),
        "ymax": ymin + (float(yy.max()) / max(h - 1, 1)) * (ymax - ymin),
        "zmin": zmin + (float(zz.min()) / max(d - 1, 1)) * (zmax - zmin),
        "zmax": zmin + (float(zz.max()) / max(d - 1, 1)) * (zmax - zmin),
    }


def bbox_from_mesh(mesh: trimesh.Trimesh) -> dict[str, float]:
    v = np.asarray(mesh.vertices)
    return {
        "xmin": float(v[:, 0].min()),
        "xmax": float(v[:, 0].max()),
        "ymin": float(v[:, 1].min()),
        "ymax": float(v[:, 1].max()),
        "zmin": float(v[:, 2].min()),
        "zmax": float(v[:, 2].max()),
    }


def center(bbox: dict[str, float], axis: str) -> float:
    return 0.5 * (bbox[f"{axis}min"] + bbox[f"{axis}max"])


def extent(bbox: dict[str, float], axis: str) -> float:
    return max(bbox[f"{axis}max"] - bbox[f"{axis}min"], 1e-8)


def align_mesh_to_target(
    source: trimesh.Trimesh,
    target_fp_bbox: dict[str, float],
    target_sdf_bbox: dict[str, float],
    scale_mode: str,
) -> tuple[trimesh.Trimesh, dict[str, float]]:
    src_bbox = bbox_from_mesh(source)
    sx = extent(target_fp_bbox, "x") / extent(src_bbox, "x")
    sz = extent(target_fp_bbox, "z") / extent(src_bbox, "z")
    if scale_mode == "uniform":
        sx = sz = min(sx, sz)
    sy = float(np.sqrt(sx * sz))

    verts = np.asarray(source.vertices, dtype=np.float64).copy()
    src_cx = center(src_bbox, "x")
    src_cz = center(src_bbox, "z")
    tgt_cx = center(target_fp_bbox, "x")
    tgt_cz = center(target_fp_bbox, "z")

    verts[:, 0] = (verts[:, 0] - src_cx) * sx + tgt_cx
    verts[:, 2] = (verts[:, 2] - src_cz) * sz + tgt_cz
    verts[:, 1] = (verts[:, 1] - src_bbox["ymin"]) * sy + target_sdf_bbox["ymin"]

    aligned = trimesh.Trimesh(vertices=verts, faces=source.faces, process=False)
    transform = {
        "scale_x": float(sx),
        "scale_y": float(sy),
        "scale_z": float(sz),
        "target_cx": float(tgt_cx),
        "target_cz": float(tgt_cz),
    }
    return aligned, transform


def footprint_from_mesh(mesh: trimesh.Trimesh, sdf_params: np.ndarray, res: int = 64) -> np.ndarray:
    mask = np.zeros((res, res), dtype=bool)
    if len(mesh.vertices) == 0:
        return mask
    pts = np.asarray(mesh.vertices)
    if len(mesh.faces) > 0:
        try:
            pts, _ = trimesh.sample.sample_surface(mesh, min(160_000, max(8000, len(mesh.faces) * 4)))
        except Exception:
            pts = np.asarray(mesh.vertices)

    xmin, _, zmin, xmax, _, zmax = sdf_params.tolist()
    cols = np.floor((pts[:, 0] - xmin) / max(xmax - xmin, 1e-9) * res).astype(np.int32)
    rows = np.floor((pts[:, 2] - zmin) / max(zmax - zmin, 1e-9) * res).astype(np.int32)
    ok = (rows >= 0) & (rows < res) & (cols >= 0) & (cols < res)
    mask[rows[ok], cols[ok]] = True
    return mask


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def mask_cell(mask: np.ndarray, title: str, size: int = 256) -> Image.Image:
    img = Image.fromarray((mask.astype(np.uint8) * 255), "L").resize(
        (size, size), Image.Resampling.NEAREST
    ).convert("RGB")
    canvas = Image.new("RGB", (size, size + 28), "white")
    canvas.paste(img, (0, 28))
    ImageDraw.Draw(canvas).text((6, 7), title, fill=(0, 0, 0))
    return canvas


def text_cell(lines: list[str], size: int = 256) -> Image.Image:
    img = Image.new("RGB", (size, size + 28), "white")
    draw = ImageDraw.Draw(img)
    y = 8
    for line in lines:
        draw.text((8, y), line, fill=(0, 0, 0))
        y += 18
    return img


def render_cell(mesh: trimesh.Trimesh, renderer, device, title: str, size: int = 256) -> Image.Image:
    try:
        arr = render_one(mesh, renderer, device)
        img = Image.fromarray(arr, "RGB").resize((size, size), Image.Resampling.LANCZOS)
        return mask_title(img, title)
    except Exception as exc:
        return text_cell([title, "render failed", type(exc).__name__, str(exc)[:28]], size=size)


def mask_title(img: Image.Image, title: str) -> Image.Image:
    canvas = Image.new("RGB", (img.width, img.height + 28), "white")
    canvas.paste(img, (0, 28))
    ImageDraw.Draw(canvas).text((6, 7), title, fill=(0, 0, 0))
    return canvas


def save_sheet(cells: list[Image.Image], title: str, path: Path) -> None:
    width = sum(c.width for c in cells)
    height = max(c.height for c in cells) + 34
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title, fill=(180, 0, 0))
    x = 0
    for cell in cells:
        sheet.paste(cell, (x, 34))
        x += cell.width
    sheet.save(path)


def nearest_neighbors(train: dict[str, np.ndarray], query: dict[str, np.ndarray], query_i: int, top_k: int, phase: str) -> list[tuple[int, str, float]]:
    sims = query["embeddings"][query_i] @ train["embeddings"].T
    masked = sims.copy()
    same_top = train["top_ids"] == query["top_ids"][query_i]
    masked[~same_top] = -1e9
    if phase == "train":
        masked[train["ids"] == query["ids"][query_i]] = -1e9
    nn = np.argsort(-masked)[:top_k]
    return [(rank, str(train["ids"][j]), float(masked[j])) for rank, j in enumerate(nn, 1)]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/BuildingNet_dataset_v0_1")
    ap.add_argument("--index_dir", default="data/BuildingNet_dataset_v0_1/retrieval_index")
    ap.add_argument("--out_dir", default="outputs/retrieval_alignment_smoke")
    ap.add_argument("--phase", default="val", choices=["val", "test", "train"])
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--scale_mode", default="anisotropic", choices=["anisotropic", "uniform"])
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    index_dir = Path(args.index_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = load_npz(index_dir / "train_embeddings.npz")
    query = load_npz(index_dir / f"{args.phase}_embeddings.npz")

    device = torch.device(args.device)
    renderer = make_renderer(device, image_size=args.image_size)

    rows = []
    for i in range(min(args.limit, len(query["ids"]))):
        query_id = str(query["ids"][i])
        q_frame = load_h5_frame(data_root, query_id, args.res)
        target_fp_bbox = grid_bbox_from_mask(q_frame["footprint"], q_frame["sdf_params"])
        target_sdf_bbox = sdf_inside_bbox(q_frame["sdf"], q_frame["sdf_params"])
        neighbors = nearest_neighbors(train, query, i, args.top_k, args.phase)

        q_out = out_dir / f"{args.phase}_{i:03d}_{query_id}"
        q_out.mkdir(parents=True, exist_ok=True)

        cells = [mask_cell(q_frame["footprint"], "query footprint", args.image_size)]
        try:
            query_mesh = load_normalized_obj(data_root, query_id, q_frame["norm_params"])
            cells.append(render_cell(query_mesh, renderer, device, "query OBJ", args.image_size))
        except Exception as exc:
            cells.append(text_cell(["query OBJ failed", type(exc).__name__, str(exc)[:32]], args.image_size))

        for rank, source_id, sim in neighbors:
            s_frame = load_h5_frame(data_root, source_id, args.res)
            source_mesh = load_normalized_obj(data_root, source_id, s_frame["norm_params"])
            aligned, transform = align_mesh_to_target(
                source_mesh,
                target_fp_bbox=target_fp_bbox,
                target_sdf_bbox=target_sdf_bbox,
                scale_mode=args.scale_mode,
            )
            aligned_path = q_out / f"rank{rank:02d}_{source_id}_aligned.obj"
            aligned.export(aligned_path)

            aligned_fp = footprint_from_mesh(aligned, q_frame["sdf_params"], args.res)
            score = iou(q_frame["footprint"], aligned_fp)
            cells.append(mask_cell(aligned_fp, f"#{rank} aligned fp {score:.3f}", args.image_size))
            cells.append(render_cell(aligned, renderer, device, f"#{rank} aligned OBJ", args.image_size))

            rows.append({
                "query_phase": args.phase,
                "query_id": query_id,
                "rank": rank,
                "source_id": source_id,
                "similarity": f"{sim:.6f}",
                "footprint_iou": f"{score:.6f}",
                "scale_x": f"{transform['scale_x']:.6f}",
                "scale_y": f"{transform['scale_y']:.6f}",
                "scale_z": f"{transform['scale_z']:.6f}",
                "aligned_obj": str(aligned_path),
            })

        sheet_path = q_out / "alignment_sheet.png"
        save_sheet(cells, query_id, sheet_path)
        print(f"{query_id}: wrote {sheet_path}")

    metrics_path = out_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "query_phase", "query_id", "rank", "source_id", "similarity",
            "footprint_iou", "scale_x", "scale_y", "scale_z", "aligned_obj",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()
