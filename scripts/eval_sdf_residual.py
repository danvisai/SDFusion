from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_fill_holes
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.correction_pair_dataset import CorrectionPairDataset
from models.networks.sdf_residual_net import SDFResidualUNet


def sign_iou_np(a: np.ndarray, b: np.ndarray) -> float:
    a_in = a <= 0
    b_in = b <= 0
    inter = np.logical_and(a_in, b_in).sum()
    union = np.logical_or(a_in, b_in).sum()
    return float(inter / union) if union else 0.0


def footprint(sdf: np.ndarray) -> np.ndarray:
    """Metric footprint: strict iso=0, used for IoU computation."""
    return (sdf <= 0).any(axis=1)


def footprint_vis(sdf: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """Visualization footprint: wider threshold + fill holes for readability."""
    fp = (sdf <= threshold).any(axis=1)
    return binary_fill_holes(fp)


def iou_mask(a: np.ndarray, b: np.ndarray) -> float:
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


def sdf_slice_cell(sdf: np.ndarray, title: str, size: int = 256) -> Image.Image:
    mid = sdf.shape[1] // 2
    sl = sdf[:, mid, :]
    lo, hi = np.percentile(sl, [2, 98])
    arr = ((sl - lo) / max(hi - lo, 1e-6)).clip(0, 1)
    img = Image.fromarray((arr * 255).astype(np.uint8), "L").resize(
        (size, size), Image.Resampling.BILINEAR
    ).convert("RGB")
    canvas = Image.new("RGB", (size, size + 28), "white")
    canvas.paste(img, (0, 28))
    ImageDraw.Draw(canvas).text((6, 7), title, fill=(0, 0, 0))
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
    ap.add_argument("--out_dir", default="outputs/sdf_residual_eval")
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    sheet_dir = out_dir / args.phase / "sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)

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
            x = item["input"][None].to(device)
            source = item["source_sdf"][0].numpy()
            target = item["target_sdf"][0].numpy()
            pred = model(x)[0, 0].cpu().numpy()
            corrected = source + pred

            source_l1 = float(np.mean(np.abs(source - target)))
            corrected_l1 = float(np.mean(np.abs(corrected - target)))
            source_iou = sign_iou_np(source, target)
            corrected_iou = sign_iou_np(corrected, target)
            # Metric footprints (iso=0) for CSV numbers
            source_fp = footprint(source)
            corrected_fp = footprint(corrected)
            target_fp = footprint(target)
            # Visualization footprints (wider threshold + fill) for images
            source_fp_vis = footprint_vis(source)
            corrected_fp_vis = footprint_vis(corrected)
            target_fp_vis = footprint_vis(target)

            query_id = str(item["query_id"])
            source_id = str(item["source_id"])
            rows.append({
                "query_id": query_id,
                "source_id": source_id,
                "source_l1": f"{source_l1:.6f}",
                "corrected_l1": f"{corrected_l1:.6f}",
                "source_iou": f"{source_iou:.6f}",
                "corrected_iou": f"{corrected_iou:.6f}",
                "source_fp_iou": f"{iou_mask(source_fp, target_fp):.6f}",
                "corrected_fp_iou": f"{iou_mask(corrected_fp, target_fp):.6f}",
            })

            cells = [
                mask_cell(target_fp_vis, "target fp"),
                mask_cell(source_fp_vis, f"source fp iou={rows[-1]['source_fp_iou']}"),
                mask_cell(corrected_fp_vis, f"corrected fp iou={rows[-1]['corrected_fp_iou']}"),
                sdf_slice_cell(target, "target sdf slice"),
                sdf_slice_cell(source, "source sdf slice"),
                sdf_slice_cell(corrected, "corrected sdf slice"),
            ]
            save_sheet(cells, f"{i:03d} {query_id} <- {source_id}", sheet_dir / f"{i:03d}_{query_id}.png")
            print(f"{i:03d} {query_id}: source_l1={source_l1:.4f} corrected_l1={corrected_l1:.4f}")

    metrics_path = out_dir / args.phase / "metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()
