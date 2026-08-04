from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.render_buildingnet_objfiles import load_obj_as_trimesh, make_renderer, render_one


def load_npz(path: Path):
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def text_cell(lines: list[str], size=(256, 256)) -> Image.Image:
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    y = 8
    for line in lines:
        draw.text((8, y), line, fill=(0, 0, 0))
        y += 18
    return img


def load_fp(data_root: Path, phase: str, mid: str, size=256) -> Image.Image:
    p = data_root / "footprints_png" / phase / f"{mid}.png"
    return Image.open(p).convert("L").resize((size, size), Image.Resampling.NEAREST).convert("RGB")


def render_obj(data_root: Path, mid: str, renderer, device, size=256) -> Image.Image:
    p = data_root / "OBJ_MODELS" / f"{mid}.obj"
    mesh = load_obj_as_trimesh(str(p))
    if mesh is None or len(mesh.faces) < 4:
        return text_cell(["render failed", mid], size=(size, size))
    try:
        arr = render_one(mesh, renderer, device)
    except Exception as exc:
        return text_cell(["render failed", type(exc).__name__, str(exc)[:24]], size=(size, size))
    return Image.fromarray(arr, "RGB").resize((size, size), Image.Resampling.LANCZOS)


def add_title(img: Image.Image, title: str) -> Image.Image:
    canvas = Image.new("RGB", (img.width, img.height + 28), "white")
    canvas.paste(img, (0, 28))
    ImageDraw.Draw(canvas).text((6, 7), title, fill=(0, 0, 0))
    return canvas


def make_sheet(query_id: str, query_phase: str, neighbors, data_root: Path, renderer, device) -> Image.Image:
    cell_w = 256
    cells = [
        add_title(load_fp(data_root, query_phase, query_id, cell_w), "query footprint"),
        add_title(render_obj(data_root, query_id, renderer, device, cell_w), "query OBJ"),
    ]
    for rank, train_id, sim in neighbors:
        fp = load_fp(data_root, "train", train_id, cell_w)
        obj = render_obj(data_root, train_id, renderer, device, cell_w)
        cells.append(add_title(fp, f"#{rank} fp {sim:.3f}"))
        cells.append(add_title(obj, f"#{rank} OBJ"))

    h = max(c.height for c in cells)
    sheet = Image.new("RGB", (cell_w * len(cells), h + 36), "white")
    ImageDraw.Draw(sheet).text((8, 8), query_id, fill=(180, 0, 0))
    for i, cell in enumerate(cells):
        sheet.paste(cell, (i * cell_w, 36))
    return sheet


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/BuildingNet_dataset_v0_1")
    ap.add_argument("--index_dir", default="data/BuildingNet_dataset_v0_1/retrieval_index")
    ap.add_argument("--out_dir", default="outputs/retrieval_visual_smoke")
    ap.add_argument("--phase", default="val", choices=["val", "test", "train"])
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    index_dir = Path(args.index_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = load_npz(index_dir / "train_embeddings.npz")
    query = load_npz(index_dir / f"{args.phase}_embeddings.npz")
    sims = query["embeddings"] @ train["embeddings"].T

    device = torch.device(args.device)
    renderer = make_renderer(device, image_size=args.image_size)

    summary = []
    for i in range(min(args.limit, len(query["ids"]))):
        same_top = train["top_ids"] == query["top_ids"][i]
        masked = sims[i].copy()
        if args.phase == "train":
            same_id = train["ids"] == query["ids"][i]
            masked[same_id] = -1e9
        masked[~same_top] = -1e9
        nn = np.argsort(-masked)[: args.top_k]
        neighbors = [(rank, str(train["ids"][j]), float(masked[j])) for rank, j in enumerate(nn, 1)]
        qid = str(query["ids"][i])
        sheet = make_sheet(qid, args.phase, neighbors, data_root, renderer, device)
        out_path = out_dir / f"{args.phase}_{i:03d}_{qid}.png"
        sheet.save(out_path)
        summary.append((qid, neighbors, out_path))
        print(f"{qid} -> {[n[1] for n in neighbors]}  {out_path}")

    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()

