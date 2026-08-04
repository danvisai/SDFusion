"""Phase B+.7 — fit each style's recipe parameters to real BuildingNet GT SDFs.

This is the CRITICAL experiment for Option B+. It answers:
  "Can the current differentiable recipe library express real building shapes?"

For each asset:
  1. Load GT 64^3 SDF + footprint mask + bounding box (norm_params, sdf_params)
  2. Extract polygon from the footprint mask
  3. For each of 8 styles, fit recipe params via gradient descent on L1(diff_recipe, GT_SDF)
  4. Pick the style with lowest residual
  5. Report sign IoU (iso=0 occupancy IoU) for the best fit

Output: per-asset CSV with (best_style, best_iou, best_l1, per_style_l1) + summary stats
        + visual sheet of the worst / median / best fits.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path("/scratch/gilbreth/dsimhadr/GenerativeTowns/SDFusion")
sys.path.insert(0, str(REPO))

from models.networks.diff_recipe import DIFF_RECIPE_REGISTRY, build_diff_recipe, make_grid_points


BN_ROOT = REPO / "data/BuildingNet_dataset_v0_1"
SDF_DIR = BN_ROOT / "resolution_64"
ASSET_CSV = REPO / "outputs/stage3_metadata/asset_dimensions.csv"
OUT_DIR = REPO / "outputs/fit_recipes_buildingnet"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_asset(asset_id: str):
    """Returns (sdf_64x64x64, footprint_64x64, bbox_xyz).

    NOTE: BuildingNet meshes are non-watertight, so the "SDF" is effectively a
    UDF with a thin (a few voxels) negative shell near the surface. Don't expect
    sdf <= 0 to identify the *interior volume* — it identifies the surface band.
    """
    h5_path = SDF_DIR / asset_id / "ori_sample_grid.h5"
    if not h5_path.exists():
        return None, None, None
    with h5py.File(h5_path, "r") as f:
        sdf = f["pc_sdf_sample"][:].reshape(64, 64, 64).astype(np.float32)  # (D=z, H=y, W=x)
        footprint = f["footprint"][0].astype(np.uint8)                      # (D, W)
        sdf_params = f["sdf_params"][:].astype(np.float32)                  # (x0,y0,z0, x1,y1,z1)
    return sdf, footprint, sdf_params


def mask_to_polygon(mask: np.ndarray, x_min: float, x_max: float,
                    z_min: float, z_max: float, n_target: int = 16) -> np.ndarray:
    """Find largest contour in `mask`, simplify to ~n_target vertices, return (P, 2) XZ."""
    import skimage.measure as skm
    if mask.sum() == 0:
        return None
    contours = skm.find_contours(mask.astype(float), 0.5)
    if not contours:
        return None
    largest = max(contours, key=lambda c: len(c))  # (N, 2) = (row=D, col=W)
    if len(largest) < 3:
        return None
    # Simplify: take ~n_target evenly-spaced vertices
    if len(largest) > n_target:
        idx = np.linspace(0, len(largest) - 1, n_target, dtype=int)
        largest = largest[idx]
    D, W = mask.shape
    xs = largest[:, 1] / (W - 1) * (x_max - x_min) + x_min
    zs = largest[:, 0] / (D - 1) * (z_max - z_min) + z_min
    poly = np.stack([xs, zs], axis=-1)
    # Ensure CCW: signed area > 0
    area = 0.5 * np.sum((poly[1:, 0] - poly[:-1, 0]) * (poly[1:, 1] + poly[:-1, 1]))
    if area > 0:  # signed area uses shoelace; CCW is negative in our (x, z) convention
        poly = poly[::-1]
    return poly.astype(np.float32)


def make_query_grid(bbox: np.ndarray, device: str) -> torch.Tensor:
    """Grid that matches the BuildingNet SDF sampling (D, H, W) in 64^3."""
    x0, y0, z0, x1, y1, z1 = bbox.tolist()
    xs = torch.linspace(x0, x1, 64, device=device)
    ys = torch.linspace(y0, y1, 64, device=device)
    zs = torch.linspace(z0, z1, 64, device=device)
    Z, Y, X = torch.meshgrid(zs, ys, xs, indexing="ij")
    return torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)


@torch.no_grad()
def footprint_iou(pred_sdf_grid: torch.Tensor, gt_footprint: torch.Tensor) -> float:
    """IoU between the recipe's top-down silhouette and the GT footprint mask.

    BuildingNet meshes are non-watertight so iso=0 of the GT SDF is unreliable
    as a volume measure. The footprint (top-down Y-collapse) IS a reliable
    structural measure and is what our recipes are designed to match.
    """
    # Recipe: collapse Y (axis 1 of (D, H, W)) — count voxels where SDF <= 0 anywhere along H
    pred_fp = (pred_sdf_grid <= 0).any(dim=1)         # (D, W) bool
    gt_fp = (gt_footprint > 0)                          # (D, W) bool
    inter = (pred_fp & gt_fp).sum().float()
    union = (pred_fp | gt_fp).sum().float()
    if union.item() == 0:
        return 0.0
    return float((inter / union).cpu())


def fit_style(style: str, gt_sdf: torch.Tensor, gt_footprint: torch.Tensor,
              polygon_xz: torch.Tensor, height: torch.Tensor,
              query_pts: torch.Tensor, y_min: float,
              steps: int = 200, lr: float = 0.05,
              surface_band: float = 0.08) -> dict:
    """Fit one style's recipe params to GT SDF via Adam.

    Loss: SURFACE-BAND L1 — only voxels where the GT SDF is within `surface_band`
    of zero contribute. Outside that band the GT signal is unreliable
    (BuildingNet's truncated/UDF-ish range), so we'd just be fitting noise.
    """
    module, default_fn, n_params = build_diff_recipe(style)
    module = module.to(query_pts.device)
    params = default_fn(query_pts.device).clone().detach().requires_grad_(True)

    # Shift queries so y=0 in recipe maps to y_min in GT
    pts_shifted = query_pts.clone()
    pts_shifted[:, 1] = pts_shifted[:, 1] - y_min

    # Surface band mask: voxels near the GT iso=0
    gt_flat = gt_sdf.reshape(-1)
    band_mask = (gt_flat.abs() < surface_band).float()
    band_n = band_mask.sum().clamp_min(1.0)

    opt = torch.optim.Adam([params], lr=lr)
    final_loss = None
    for step in range(steps):
        opt.zero_grad()
        pred = module(params, polygon_xz, height, pts_shifted)
        # Surface-band weighted L1 + a tiny full-grid L1 to anchor scale.
        band_loss = (((pred - gt_flat).abs()) * band_mask).sum() / band_n
        anchor_loss = (pred.clamp(min=-surface_band) - gt_flat.clamp(min=-surface_band)).abs().mean()
        loss = band_loss + 0.1 * anchor_loss
        loss.backward()
        opt.step()
        final_loss = loss.item()

    with torch.no_grad():
        pred = module(params, polygon_xz, height, pts_shifted)
        pred_grid = pred.reshape(64, 64, 64)
        fp_iou = footprint_iou(pred_grid, gt_footprint)
        # Banded L1 — what the loss measures
        band_l1 = ((pred - gt_flat).abs() * band_mask).sum().item() / band_n.item()
    return {
        "style": style, "iou": fp_iou, "band_l1": band_l1, "final_loss": final_loss,
        "params": params.detach().cpu().numpy(),
    }


def fit_asset(asset_id: str, device: str, styles: list, steps: int = 600,
              n_poly_verts: int = 16, fp_min_cells: int = 100,
              iso_min_voxels: int = 300) -> dict:
    """Fit all `styles` to one asset; return best fit + per-style numbers.

    Sparsity filter: skips assets where the GT footprint mask has fewer than
    fp_min_cells occupied cells (~3% of 4096), or where the GT SDF has fewer
    than iso_min_voxels with sdf<=0. These are non-watertight meshes whose
    iso=0 contour is just fragments — fitting them is meaningless and they'd
    contribute spurious low IoUs to the dataset.
    """
    sdf_np, fp_np, sdf_params = load_asset(asset_id)
    if sdf_np is None:
        return {"asset_id": asset_id, "ok": False, "error": "missing h5"}
    n_fp = int((fp_np > 0).sum())
    n_iso = int((sdf_np <= 0).sum())
    if n_fp < fp_min_cells:
        return {"asset_id": asset_id, "ok": False,
                "error": f"sparse footprint n_fp={n_fp} < {fp_min_cells}"}
    if n_iso < iso_min_voxels:
        return {"asset_id": asset_id, "ok": False,
                "error": f"sparse SDF iso n={n_iso} < {iso_min_voxels}"}
    x0, y0, z0, x1, y1, z1 = sdf_params.tolist()
    polygon = mask_to_polygon(fp_np, x0, x1, z0, z1, n_target=n_poly_verts)
    if polygon is None:
        return {"asset_id": asset_id, "ok": False, "error": "polygon extraction failed"}

    gt_sdf = torch.tensor(sdf_np, device=device)
    gt_fp = torch.tensor(fp_np, device=device)
    poly_t = torch.tensor(polygon, device=device)
    height = torch.tensor(y1 - y0, dtype=torch.float32, device=device)
    query_pts = make_query_grid(sdf_params, device)

    per_style = []
    for style in styles:
        try:
            r = fit_style(style, gt_sdf, gt_fp, poly_t, height, query_pts,
                          y_min=y0, steps=steps)
            per_style.append(r)
        except Exception as exc:
            per_style.append({"style": style, "iou": 0.0, "band_l1": float("inf"),
                              "error": str(exc)})
    best = max(per_style, key=lambda r: r.get("iou", 0.0))
    return {
        "asset_id": asset_id, "ok": True,
        "best_style": best["style"], "best_iou": best["iou"],
        "best_l1": best.get("band_l1", 0.0),
        "best_params": best.get("params"),
        "polygon": polygon,
        "bbox": sdf_params,
        "per_style": {r["style"]: {"iou": r.get("iou", 0.0),
                                    "band_l1": r.get("band_l1", float("inf"))}
                      for r in per_style},
    }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_assets", type=int, default=16,
                    help="How many assets to fit. Use 0 for ALL.")
    ap.add_argument("--steps", type=int, default=200, help="Adam steps per (asset, style)")
    ap.add_argument("--styles", type=str, default=None,
                    help="Comma-separated style names. Default: all 8.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_poly_verts", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def sample_asset_ids(n: int, seed: int) -> list:
    """Pick a diverse subset across classes and splits."""
    import random
    random.seed(seed)
    rows = list(csv.DictReader(open(ASSET_CSV)))
    rows = [r for r in rows if (SDF_DIR / r["id"] / "ori_sample_grid.h5").exists()]
    if n == 0 or n >= len(rows):
        return [r["id"] for r in rows]
    # Stratify by top-level class (first word in id, e.g. RESIDENTIAL/COMMERCIAL/...)
    by_class = {}
    for r in rows:
        cls = ""
        for prefix in ("RESIDENTIAL", "COMMERCIAL", "RELIGIOUS", "PUBLIC", "CIVIC", "INDUSTRIAL", "AGRICULTURAL"):
            if r["id"].startswith(prefix):
                cls = prefix; break
        by_class.setdefault(cls or "OTHER", []).append(r["id"])
    per_class = max(1, n // max(1, len(by_class)))
    picked = []
    for cls, ids in by_class.items():
        random.shuffle(ids)
        picked.extend(ids[:per_class])
    return picked[:n]


def main():
    args = parse_args()
    styles = args.styles.split(",") if args.styles else list(DIFF_RECIPE_REGISTRY.keys())
    print(f"[*] device={args.device}  steps={args.steps}  styles={styles}")

    asset_ids = sample_asset_ids(args.n_assets, args.seed)
    print(f"[*] fitting {len(asset_ids)} assets")

    t0 = time.time()
    results = []
    for i, aid in enumerate(asset_ids):
        ts = time.time()
        r = fit_asset(aid, args.device, styles, steps=args.steps, n_poly_verts=args.n_poly_verts)
        dt = time.time() - ts
        if not r["ok"]:
            print(f"  [{i+1:4d}/{len(asset_ids)}]  {aid:40s}  SKIPPED  ({r['error']})")
            results.append(r)
            continue
        print(f"  [{i+1:4d}/{len(asset_ids)}]  {aid:40s}  best={r['best_style']:13s}  IoU={r['best_iou']:.3f}  L1={r['best_l1']:.4f}  ({dt:.1f}s)")
        results.append(r)

    # Aggregate
    ok = [r for r in results if r["ok"]]
    print(f"\n=== AGGREGATE ===")
    print(f"  n_assets attempted: {len(results)}")
    print(f"  n_assets succeeded: {len(ok)}")
    if ok:
        ious = np.array([r["best_iou"] for r in ok])
        print(f"  IoU mean = {ious.mean():.3f}, median = {np.median(ious):.3f}, std = {ious.std():.3f}")
        print(f"  IoU > 0.7 fraction: {(ious > 0.7).mean():.2%}")
        print(f"  IoU > 0.5 fraction: {(ious > 0.5).mean():.2%}")
        # Style distribution among best fits
        from collections import Counter
        style_counter = Counter(r["best_style"] for r in ok)
        print(f"  Best-style counts: {dict(style_counter)}")
    print(f"  Total elapsed: {time.time() - t0:.1f}s")

    # Save per-asset CSV
    csv_path = OUT_DIR / "per_asset_fits.csv"
    with open(csv_path, "w") as f:
        w = csv.writer(f)
        header = ["asset_id", "ok", "best_style", "best_iou", "best_l1"] + \
                 [f"iou_{s}" for s in styles] + [f"l1_{s}" for s in styles]
        w.writerow(header)
        for r in results:
            if not r["ok"]:
                w.writerow([r["asset_id"], 0, "", 0, 0] + [0] * len(styles) * 2)
                continue
            row = [r["asset_id"], 1, r["best_style"], r["best_iou"], r["best_l1"]]
            row += [r["per_style"].get(s, {}).get("iou", 0.0) for s in styles]
            row += [r["per_style"].get(s, {}).get("band_l1", 0.0) for s in styles]
            w.writerow(row)
    print(f"  wrote {csv_path}")

    # Save successful best-params for downstream use
    if ok:
        params_path = OUT_DIR / "best_params.npz"
        save_dict = {}
        for r in ok:
            save_dict[r["asset_id"]] = {
                "style": r["best_style"],
                "params": r["best_params"],
                "iou": r["best_iou"],
                "bbox": r["bbox"],
                "polygon": r["polygon"],
            }
        np.savez_compressed(params_path, fits=save_dict)
        print(f"  wrote {params_path} ({len(save_dict)} fits)")


if __name__ == "__main__":
    main()
