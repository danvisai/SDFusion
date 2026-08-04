"""Quick sparsity audit of BuildingNet GT SDFs.

For each asset:
  - n_voxels with sdf <= 0  (interior + thin surface band, since meshes are non-watertight)
  - n footprint cells       (top-down Y-collapse occupancy)

Goal: pick a threshold below which an asset is "too sparse to fit" — its iso=0
contour is just fragments and the footprint IoU metric is meaningless.
"""

from __future__ import annotations
import sys, csv
from pathlib import Path
import h5py
import numpy as np

REPO = Path("/scratch/gilbreth/dsimhadr/GenerativeTowns/SDFusion")
BN_ROOT = REPO / "data/BuildingNet_dataset_v0_1"
SDF_DIR = BN_ROOT / "resolution_64"
ASSET_CSV = REPO / "outputs/stage3_metadata/asset_dimensions.csv"
OUT_DIR = REPO / "outputs/buildingnet_sparsity_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    rows = list(csv.DictReader(open(ASSET_CSV)))
    print(f"Auditing {len(rows)} assets...")

    stats = []
    for i, r in enumerate(rows):
        aid = r["id"]
        path = SDF_DIR / aid / "ori_sample_grid.h5"
        if not path.exists():
            continue
        try:
            with h5py.File(path, "r") as f:
                sdf = f["pc_sdf_sample"][:].reshape(64, 64, 64).astype(np.float32)
                fp = f["footprint"][0]
            n_iso = int((sdf <= 0).sum())
            n_fp = int((fp > 0).sum())
            sdf_min = float(sdf.min())
            stats.append({"id": aid, "n_iso": n_iso, "n_fp": n_fp, "sdf_min": sdf_min})
        except Exception as e:
            continue
        if (i + 1) % 200 == 0:
            print(f"  audited {i+1}/{len(rows)}")

    # Save CSV
    csv_path = OUT_DIR / "sparsity.csv"
    with open(csv_path, "w") as f:
        w = csv.writer(f)
        w.writerow(["id", "n_iso", "n_fp", "sdf_min"])
        for s in stats:
            w.writerow([s["id"], s["n_iso"], s["n_fp"], f"{s['sdf_min']:.4f}"])
    print(f"wrote {csv_path}")

    # Quantiles
    n_iso = np.array([s["n_iso"] for s in stats])
    n_fp = np.array([s["n_fp"] for s in stats])
    print(f"\n=== iso=0 voxel count (out of 262144) ===")
    for q in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75]:
        print(f"  q{q*100:4.0f}% = {int(np.quantile(n_iso, q)):>6d}")
    print(f"  max    = {n_iso.max()}")
    print(f"\n=== footprint cell count (out of 4096) ===")
    for q in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75]:
        print(f"  q{q*100:4.0f}% = {int(np.quantile(n_fp, q)):>4d}")
    print(f"  max    = {n_fp.max()}")

    # Suggested thresholds
    print(f"\n=== suggested thresholds ===")
    for fp_min in [20, 40, 60, 100, 200]:
        n_keep = int((n_fp >= fp_min).sum())
        print(f"  fp_min={fp_min:4d}: {n_keep:4d}/{len(n_fp)} kept ({n_keep/len(n_fp)*100:.1f}%)")


if __name__ == "__main__":
    main()
