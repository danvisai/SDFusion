"""Derive the RELATIONAL structure a coherent layout has — the targets the coherence losses
measure against (recohere plan §2 / coherent-add-primitive spec §4.2).

Per building, from [type, axis-bbox] instances, compute (all DERIVED, noisy — used softly):
  - principal horizontal axis (PCA on centers) + mirror plane (perpendicular, through centroid)
  - height-BANDS: gap-cluster cy of row-forming parts (window/door/balcony) -> band centroids +
    per-instance band_id  (the rhythm grid; ~3/bldg, BuildingNet 78% / LoD3 99% co-planar)
  - side_id: nearest of 4 facade sides (bearing bin) per instance  (position-derived, NOT pose)
  - symmetry score per building (mirror-chamfer/spread) -> GATE L_sym to the ~10% symmetric
  - band cleanliness per building -> drop confetti buildings from rhythm supervision

Output cache: data/part_structure/<source>_structure.npz (per-instance band_id/side_id aligned to
the source rows; per-building sym_score / cleanliness / principal_axis / band centroids).

  env -u LD_PRELOAD -u LD_LIBRARY_PATH PYTHONPATH=. \
    /tmp/sdfusion_venv/bin/python scripts/foundations/derive_part_structure.py [--source lod3|buildingnet]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from models.networks.part_layout_planner import RAW_TYPES  # noqa: E402

RAW2IDX = {r: i for i, r in enumerate(RAW_TYPES)}
ROW_TYPES = {RAW2IDX[t] for t in (2, 6, 14, 16) if t in RAW2IDX}   # window/door/balcony(_up)
SOURCES = {"lod3": REPO / "data/lod3_tum/lod3_part_instances.npz",
           "buildingnet": REPO / "outputs/part_layouts_full/part_instances.npz"}
OUT = REPO / "data/part_structure"
TAU = 0.05                                                  # band/co-planarity tolerance (cube units)
MAX_BANDS = 12


def gap_cluster(vals, gap=0.06):
    order = np.argsort(vals); v = vals[order]
    lab = np.zeros(len(v), int); k = 0
    for i in range(1, len(v)):
        if v[i] - v[i - 1] >= gap:
            k += 1
        lab[i] = k
    out = np.empty(len(v), int); out[order] = lab
    return out, k + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="lod3", choices=list(SOURCES))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    d = np.load(SOURCES[a.source], allow_pickle=True)
    rows = d["rows"]
    # map raw type -> planner idx; keep row index alignment with the source npz
    tidx = np.array([RAW2IDX.get(int(t), -1) for t in rows[:, 1]])
    band_id = np.full(len(rows), -1, np.int32)
    side_id = np.full(len(rows), -1, np.int32)
    pb_sym, pb_clean, pb_axis, pb_bands, pb_id = [], [], [], [], []

    for b in np.unique(rows[:, 0]).astype(int):
        m = np.where(rows[:, 0] == b)[0]
        C = rows[m, 2:5].astype(np.float32)                # centers (x,y,z)
        xz = C[:, [0, 2]] - C[:, [0, 2]].mean(0)
        # principal horizontal axis via PCA
        if len(xz) >= 2:
            _, _, V = np.linalg.svd(xz, full_matrices=False)
            axis = V[0]
        else:
            axis = np.array([1.0, 0.0], np.float32)
        # side_id: 4-bin bearing of each instance from centroid
        ang = (np.degrees(np.arctan2(xz[:, 1], xz[:, 0])) + 360) % 360
        side_id[m] = (((ang + 45) // 90) % 4).astype(np.int32)
        # height-bands from row-forming parts
        is_row = np.array([tidx[i] in ROW_TYPES for i in m])
        cy = C[:, 1]
        if is_row.sum() >= 2:
            lab_r, nb = gap_cluster(cy[is_row], gap=0.06)
            cents = np.array([cy[is_row][lab_r == k].mean() for k in range(nb)], np.float32)
            band_id[m[is_row]] = lab_r.astype(np.int32)
            # non-row parts -> nearest band
            for i in m[~is_row]:
                band_id[i] = int(np.argmin(np.abs(cents - rows[i, 3])))
            clean = float(np.mean([abs(cy[is_row][j] - cents[lab_r[j]]) < 0.03
                                   for j in range(is_row.sum())]))
        else:
            cents = np.array([cy.mean()], np.float32); nb = 1; clean = 0.0
            band_id[m] = 0
        # mirror symmetry score: reflect centers across the principal plane through centroid
        n = np.array([-axis[1], axis[0]])                  # plane normal (horizontal)
        proj = xz @ n
        mirror = xz - 2 * proj[:, None] * n[None]
        from scipy.spatial import cKDTree
        dd, _ = cKDTree(mirror).query(xz)
        sym = float(np.median(dd) / (xz.std() + 1e-6)) if len(xz) >= 4 else 1.0
        pb_id.append(b); pb_sym.append(sym); pb_clean.append(clean)
        pb_axis.append(axis.astype(np.float32))
        cents = np.pad(cents[:MAX_BANDS], (0, MAX_BANDS - min(len(cents), MAX_BANDS)),
                       constant_values=np.nan)
        pb_bands.append(cents.astype(np.float32))

    pb_sym = np.array(pb_sym, np.float32)
    np.savez(OUT / f"{a.source}_structure.npz",
             building=np.array(pb_id, np.int32), band_id=band_id, side_id=side_id,
             sym_score=pb_sym, cleanliness=np.array(pb_clean, np.float32),
             principal_axis=np.array(pb_axis, np.float32), band_centroids=np.array(pb_bands),
             row_type_idx=np.array(sorted(ROW_TYPES), np.int32), tau=TAU)
    print(f"[{a.source}] {len(pb_id)} buildings | bands/bldg "
          f"mean {np.mean([np.sum(~np.isnan(x)) for x in pb_bands]):.1f} | "
          f"symmetric(<0.5) {(pb_sym<0.5).mean():.0%} (gate L_sym to these) | "
          f"mean cleanliness {np.mean(pb_clean):.2f}")
    print(f"[save] {OUT/(a.source+'_structure.npz')}")


if __name__ == "__main__":
    main()
