"""Ticket #27 gate calibration: measure the solidity / anti-fragmentation distribution of the REAL
solidified targets so the gate floors are set where real data passes (not arbitrary).

Solidify = #28 (hybrid per-column occupancy extrusion + stored-footprint fallback, keep-centered
lowest-to-top). Per #32 we exclude label-contaminated meshes (non-building face share > 10%) so the
distribution reflects the building-only pipeline. Metrics per building:
  lcc_frac  = largest 6-connected component / total occupancy   (anti-fragmentation; solid block ~1.0)
  fill_ratio= occupancy / (footprint_area * vertical_extent)     (solidity; hollow shell is low)
CPU only. Run: env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/calibrate_massing_gate.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, h5py
from scipy import ndimage

REPO = Path(__file__).resolve().parents[2]
BN = REPO / "data/BuildingNet_dataset_v0_1"
DATA, FL = BN / "resolution_64", BN / "model_data/obj/face_labels"
RES, UP = 64, 1
NONBLD = {5, 9, 13, 19, 23}


def load(bid):
    with h5py.File(DATA / bid / "ori_sample_grid.h5", "r") as f:
        sdf = np.asarray(f["pc_sdf_sample"]).reshape(RES, RES, RES).astype(np.float32)
        fp = np.asarray(f["footprint"]); fp = fp[0] if fp.ndim == 3 else fp
    return sdf, fp.astype(bool)


def nonbld_share(bid):
    p = FL / f"{bid}.json"
    if not p.exists():
        return 1.0
    v = np.array(list(json.load(open(p)).values()))
    return float(np.isin(v, list(NONBLD)).mean())


def solidify(occ, fp):  # #28 hybrid, keep-centered lowest-to-top
    m = np.moveaxis(occ, UP, 0); H = m.shape[0]
    colocc = m.any(axis=0); hh = np.arange(H)[:, None, None]
    low = np.where(colocc, np.argmax(m, axis=0), H)
    high = np.where(colocc, H - 1 - np.argmax(m[::-1], axis=0), -1)
    out = (hh >= low[None]) & (hh <= high[None]) & colocc[None]
    if m.any():
        lv = np.where(m.any(axis=(1, 2)))[0]; blo, bhi = int(lv.min()), int(lv.max())
        if bhi - blo < 4:
            c = H // 2; blo, bhi = c - H // 5, c + H // 5
    else:
        c = H // 2; blo, bhi = c - H // 5, c + H // 5
    need = fp & (~colocc); band = (hh >= blo) & (hh <= bhi)
    return np.moveaxis(out | (band & need[None]), 0, UP)


def metrics(occ):
    n = int(occ.sum())
    if n == 0:
        return None
    lab, k = ndimage.label(occ)  # 6-conn
    lcc = np.bincount(lab.ravel())[1:].max() / n if k else 0
    fp = occ.any(axis=UP); area = int(fp.sum())
    m = np.moveaxis(occ, UP, 0); lv = np.where(m.any(axis=(1, 2)))[0]
    vext = int(lv.max() - lv.min() + 1) if len(lv) else 1
    fill = n / max(area * vext, 1)
    return dict(lcc_frac=float(lcc), fill_ratio=float(fill), n_vox=n)


def main():
    ids = json.load(open(REPO / "data/splits_v1/test.json"))
    rng = np.random.default_rng(1)
    rows = []
    for bid in rng.choice(ids, size=110, replace=False):
        if nonbld_share(bid) > 0.10:      # #32: exclude contaminated meshes
            continue
        sdf, fp = load(bid)
        mm = metrics(solidify(sdf <= 0, fp))
        if mm:
            rows.append(mm)
        if len(rows) >= 70:
            break
    lcc = np.array([r["lcc_frac"] for r in rows]); fill = np.array([r["fill_ratio"] for r in rows])

    def pcts(x):
        return {p: float(np.percentile(x, p)) for p in (5, 10, 25, 50)}
    summary = dict(n_clean_targets=len(rows), lcc_frac=pcts(lcc), fill_ratio=pcts(fill),
                   recommend_floor=dict(lcc_frac_p10=float(np.percentile(lcc, 10)),
                                        fill_ratio_p10=float(np.percentile(fill, 10))))
    (REPO / "execution/artifacts/massing_gate_calibration.json").write_text(json.dumps(summary, indent=2))
    print("REAL solidified-target distribution (clean, building-only proxy):", flush=True)
    print(f"  n = {len(rows)}", flush=True)
    print(f"  largest-connected-component fraction: p5={pcts(lcc)[5]:.3f} p10={pcts(lcc)[10]:.3f} "
          f"p25={pcts(lcc)[25]:.3f} median={pcts(lcc)[50]:.3f}", flush=True)
    print(f"  fill ratio (occ / footprint-envelope):  p5={pcts(fill)[5]:.3f} p10={pcts(fill)[10]:.3f} "
          f"p25={pcts(fill)[25]:.3f} median={pcts(fill)[50]:.3f}", flush=True)
    print(f"\n  => floors at real 10th percentile:  LCC >= {np.percentile(lcc,10):.2f}   "
          f"fill >= {np.percentile(fill,10):.2f}", flush=True)


if __name__ == "__main__":
    main()
