from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_fill_holes
from skimage import measure
import torch
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.correction_pair_dataset import CorrectionPairDataset
from models.networks.sdf_residual_net import SDFResidualUNet


def sdf_footprint(sdf: np.ndarray, iso: float = 0.0) -> np.ndarray:
    """Metric footprint: strict iso=0 threshold, used for IoU computation."""
    return (sdf <= iso).any(axis=1)


def sdf_footprint_vis(sdf: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """Visualization footprint: wider threshold + fill holes.

    BuildingNet meshes are hollow shells so sdf<=0 captures only thin wall
    voxels (~0.1% of grid). threshold=0.1 expands to the near-surface band;
    binary_fill_holes converts outlines to solid shapes for readability.
    """
    fp = (sdf <= threshold).any(axis=1)
    return binary_fill_holes(fp)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def mesh_from_sdf(sdf: np.ndarray, level: float = 0.0) -> trimesh.Trimesh | None:
    try:
        verts_zyx, faces, _, _ = measure.marching_cubes(sdf, level=level)
    except Exception:
        return None
    if len(verts_zyx) == 0 or len(faces) == 0:
        return None
    res = sdf.shape[0]
    verts = np.empty((len(verts_zyx), 3), dtype=np.float32)
    verts[:, 0] = (verts_zyx[:, 2] / max(res - 1, 1)) * 2.0 - 1.0
    verts[:, 1] = (verts_zyx[:, 1] / max(res - 1, 1)) * 2.0 - 1.0
    verts[:, 2] = (verts_zyx[:, 0] / max(res - 1, 1)) * 2.0 - 1.0
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def mesh_stats(mesh: trimesh.Trimesh | None) -> dict[str, float]:
    if mesh is None:
        return {
            "verts": 0,
            "faces": 0,
            "components": 0,
            "largest_component_frac": 0.0,
        }
    try:
        comps = mesh.split(only_watertight=False)
    except Exception:
        comps = []
    if not comps:
        components = 1 if len(mesh.faces) else 0
        largest_frac = 1.0 if components else 0.0
    else:
        components = len(comps)
        largest_frac = max(len(c.faces) for c in comps) / max(len(mesh.faces), 1)
    return {
        "verts": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "components": int(components),
        "largest_component_frac": float(largest_frac),
    }


def mask_cell(mask: np.ndarray, title: str, size: int = 256,
              color: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """Render a binary mask as a smooth colored cell with anti-aliased edges."""
    # Upsample 4× with bilinear then threshold for smooth edges
    raw = Image.fromarray((mask.astype(np.uint8) * 255), "L")
    smooth = raw.resize((size * 4, size * 4), Image.Resampling.BILINEAR)
    smooth = smooth.resize((size, size), Image.Resampling.LANCZOS)
    alpha = np.array(smooth, dtype=np.float32) / 255.0
    canvas_arr = np.ones((size, size, 3), dtype=np.uint8) * 255
    for c, col in enumerate(color):
        canvas_arr[:, :, c] = (255 * (1 - alpha) + col * alpha).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(canvas_arr)
    canvas = Image.new("RGB", (size, size + 28), "white")
    canvas.paste(img, (0, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 7), title, fill=color)
    return canvas


def overlay_cell(target: np.ndarray, source: np.ndarray, corrected: np.ndarray,
                 title: str, size: int = 256) -> Image.Image:
    """Color overlay: target=blue, source=red, corrected=green. Overlaps blend."""
    colors = {
        "target":    np.array([40,  100, 220], dtype=np.float32),
        "source":    np.array([220,  60,  60], dtype=np.float32),
        "corrected": np.array([40,  180,  60], dtype=np.float32),
    }
    canvas_arr = np.ones((size, size, 3), dtype=np.float32) * 255.0
    for name, mask in [("target", target), ("source", source), ("corrected", corrected)]:
        raw = Image.fromarray((mask.astype(np.uint8) * 255), "L")
        smooth = raw.resize((size * 4, size * 4), Image.Resampling.BILINEAR)
        alpha = np.array(smooth.resize((size, size), Image.Resampling.LANCZOS),
                         dtype=np.float32) / 255.0 * 0.55
        for c, col in enumerate(colors[name]):
            canvas_arr[:, :, c] = canvas_arr[:, :, c] * (1 - alpha) + col * alpha
    img = Image.fromarray(canvas_arr.clip(0, 255).astype(np.uint8))
    canvas = Image.new("RGB", (size, size + 28), "white")
    canvas.paste(img, (0, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 7), title, fill=(60, 60, 60))
    draw.text((size - 130, 7), "■tgt ■src ■cor",
              fill=(100, 100, 100))
    # Color legend squares
    for x_off, col in [(size - 130, (40, 100, 220)),
                       (size - 90,  (220, 60,  60)),
                       (size - 50,  (40, 180,  60))]:
        draw.rectangle([x_off, 8, x_off + 12, 20], fill=col)
    return canvas


def save_sheet(cells: list[Image.Image], title: str, path: Path) -> None:
    w = sum(c.width for c in cells)
    h = max(c.height for c in cells) + 32
    sheet = Image.new("RGB", (w, h), "white")
    ImageDraw.Draw(sheet).text((8, 8), title, fill=(180, 0, 0))
    x = 0
    for cell in cells:
        sheet.paste(cell, (x, 32))
        x += cell.width
    sheet.save(path)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pair_root", default="data/BuildingNet_dataset_v0_1/correction_pairs")
    ap.add_argument("--phase", default="val", choices=["train", "val"])
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument("--out_dir", default="outputs/sdf_residual_mesh_eval")
    ap.add_argument("--export_obj", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    sheet_dir = out_dir / args.phase / "sheets"
    mesh_dir = out_dir / args.phase / "meshes"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    if args.export_obj:
        mesh_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model_args = ckpt.get("args", {})
    model = SDFResidualUNet(
        in_channels=2,
        base_channels=int(model_args.get("base_channels", 16)),
        residual_clip=float(model_args.get("residual_clip", 1.0)),
    )
    model.load_state_dict(ckpt["model"])
    device = torch.device(args.device)
    model.to(device).eval()

    ds = CorrectionPairDataset(
        args.pair_root,
        args.phase,
        max_samples=args.limit,
        residual_clip=float(model_args.get("residual_clip", 1.0)),
    )

    rows = []
    with torch.no_grad():
        for i in range(len(ds)):
            item = ds[i]
            query_id = str(item["query_id"])
            source_id = str(item["source_id"])
            source = item["source_sdf"][0].numpy()
            target = item["target_sdf"][0].numpy()
            pred = model(item["input"][None].to(device))[0, 0].cpu().numpy()
            corrected = source + pred

            # Metric footprints: strict iso=0, used for IoU numbers in CSV
            target_fp = sdf_footprint(target)
            source_fp = sdf_footprint(source)
            corrected_fp = sdf_footprint(corrected)
            # Visualization footprints: wider threshold + fill holes for readability
            target_fp_vis = sdf_footprint_vis(target)
            source_fp_vis = sdf_footprint_vis(source)
            corrected_fp_vis = sdf_footprint_vis(corrected)
            meshes = {
                "source": mesh_from_sdf(source),
                "corrected": mesh_from_sdf(corrected),
                "target": mesh_from_sdf(target),
            }
            stats = {name: mesh_stats(mesh) for name, mesh in meshes.items()}

            if args.export_obj:
                for name, mesh in meshes.items():
                    if mesh is not None:
                        mesh.export(mesh_dir / f"{i:03d}_{query_id}_{name}.obj")

            row = {
                "query_id": query_id,
                "source_id": source_id,
                "source_sdf_l1": f"{float(np.mean(np.abs(source - target))):.6f}",
                "corrected_sdf_l1": f"{float(np.mean(np.abs(corrected - target))):.6f}",
                "source_fp_iou": f"{iou(source_fp, target_fp):.6f}",
                "corrected_fp_iou": f"{iou(corrected_fp, target_fp):.6f}",
            }
            for name in ("source", "corrected", "target"):
                for key, value in stats[name].items():
                    row[f"{name}_{key}"] = f"{value:.6f}" if isinstance(value, float) else value
            rows.append(row)

            cells = [
                mask_cell(target_fp_vis, "target",
                          color=(40, 100, 220)),
                mask_cell(source_fp_vis, f"source  iou={row['source_fp_iou']}",
                          color=(220, 60, 60)),
                mask_cell(corrected_fp_vis, f"corrected  iou={row['corrected_fp_iou']}",
                          color=(40, 180, 60)),
                overlay_cell(target_fp_vis, source_fp_vis, corrected_fp_vis,
                             "overlay"),
            ]
            save_sheet(cells, f"{i:03d} {query_id} <- {source_id}", sheet_dir / f"{i:03d}_{query_id}.png")
            print(
                f"{i:03d} {query_id}: fp {row['source_fp_iou']} -> {row['corrected_fp_iou']} "
                f"components {stats['source']['components']} -> {stats['corrected']['components']}"
            )

    metrics_path = out_dir / args.phase / "mesh_metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()
