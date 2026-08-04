"""Single-asset debug: visualize GT SDF vs diff_recipe output BEFORE fitting, then AFTER."""
import sys, h5py
from pathlib import Path
import numpy as np
import torch

REPO = Path("/scratch/gilbreth/dsimhadr/GenerativeTowns/SDFusion")
sys.path.insert(0, str(REPO))

from models.networks.diff_recipe import build_diff_recipe, make_grid_points
import scripts.fit_recipes_to_buildingnet as fitter

ASSET = "RESIDENTIALhouse_mesh1238"
device = "cuda"

sdf_np, fp_np, bbox = fitter.load_asset(ASSET)
print(f"GT SDF: shape={sdf_np.shape}, range=[{sdf_np.min():.4f}, {sdf_np.max():.4f}]")
print(f"GT occupancy (sdf <= 0): {(sdf_np <= 0).sum()} voxels, frac={(sdf_np <= 0).mean():.4f}")
print(f"sdf_params bbox: {bbox.tolist()}")
x0, y0, z0, x1, y1, z1 = bbox.tolist()
print(f"  X extent: {x1-x0:.3f}, Y extent: {y1-y0:.3f}, Z extent: {z1-z0:.3f}")

poly = fitter.mask_to_polygon(fp_np, x0, x1, z0, z1, n_target=16)
print(f"polygon: shape={poly.shape}, X range=[{poly[:,0].min():.3f}, {poly[:,0].max():.3f}], Z range=[{poly[:,1].min():.3f}, {poly[:,1].max():.3f}]")

# Run all styles with DEFAULT params (no fitting) to see baseline
gt_t = torch.tensor(sdf_np, device=device)
poly_t = torch.tensor(poly, device=device)
height = torch.tensor(y1 - y0, device=device)
query_pts = fitter.make_query_grid(bbox, device)
# Shift y so y=0 in recipe = y_min in GT
pts_shifted = query_pts.clone()
pts_shifted[:, 1] = pts_shifted[:, 1] - y0
print(f"\nQuery grid: shape={query_pts.shape}, y range=[{query_pts[:,1].min():.3f}, {query_pts[:,1].max():.3f}]")
print(f"After shift:  y range=[{pts_shifted[:,1].min():.3f}, {pts_shifted[:,1].max():.3f}]  (expected 0..{(y1-y0):.3f})")

print("\nPer-style baseline (defaults, no fitting):")
for style in ["modern", "colonial", "victorian", "industrial",
              "craftsman", "mediterranean", "contemporary", "public_civic"]:
    module, default_fn, n = build_diff_recipe(style)
    module = module.to(device)
    p = default_fn(device)
    with torch.no_grad():
        pred = module(p, poly_t, height, pts_shifted)
    pred_g = pred.reshape(64, 64, 64)
    iou = fitter.sign_iou(pred_g, gt_t)
    l1 = (pred - gt_t.reshape(-1)).abs().mean().item()
    print(f"  {style:14s}  iou={iou:.3f}  l1={l1:.4f}  pred_range=[{pred.min().item():.3f}, {pred.max().item():.3f}]  pred_occ_frac={(pred_g <= 0).float().mean().item():.4f}")
