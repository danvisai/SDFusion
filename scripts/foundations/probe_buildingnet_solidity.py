"""Ticket #25 probe: is BuildingNet's low massing occupancy a hollow-mesh reality or a
surface-vs-filled artifact, and does solidify-in-place recover a solid footprint-matching block?

For a sample of the sealed held-out set we compare, per building (64^3 native field):
  raw       : (sdf<=0).mean()                        -- what the model trains on today
  fillholes : binary_fill_holes(raw)                 -- recovers WATERTIGHT enclosed voids
  colfill   : ground-anchored footprint extrusion    -- fill each (x,z) column floor->top-occupied
              along the H-up axis (axis=1); this is the SOLID MASSING target we actually want
  boundary_frac : fraction of raw-occupied voxels on the surface (a shell indicator; ~1.0 = pure shell)

Writes execution/artifacts/buildingnet_solidity_probe.json and a raw-vs-colfill montage.
CPU only. Run: env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/probe_buildingnet_solidity.py
"""
from __future__ import annotations
import json, time, sys
from pathlib import Path
import numpy as np
import h5py
from scipy import ndimage

REPO = Path(__file__).resolve().parents[2]
RES = 64
UP = 1  # H-up axis (render_facades: footprint = occ.any(axis=1))
DATA = REPO / "data/BuildingNet_dataset_v0_1/resolution_64"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 80


def load_sdf(bid):
    with h5py.File(DATA / bid / "ori_sample_grid.h5", "r") as f:
        return np.asarray(f["pc_sdf_sample"]).reshape(RES, RES, RES).astype(np.float32)


def column_fill(mask, up=UP):
    """Ground-anchored footprint extrusion: for each column along `up`, fill from index 0 up to
    the highest occupied voxel. Turns a shell into a solid massing block sitting on the floor."""
    m = np.moveaxis(mask, up, 0)            # (H, A, B)
    H = m.shape[0]
    any_occ = m.any(axis=0)                 # (A,B) footprint
    top = np.where(any_occ, (m * np.arange(H)[:, None, None]).max(axis=0), -1)  # highest occ per col
    hh = np.arange(H)[:, None, None]
    out = (hh <= top[None]) & any_occ[None]  # floor..top for occupied columns
    return np.moveaxis(out, 0, up)


def boundary_fraction(mask):
    if mask.sum() == 0:
        return float("nan")
    er = ndimage.binary_erosion(mask)
    interior = er.sum()
    return float((mask.sum() - interior) / mask.sum())


def main():
    ids = json.load(open(REPO / "data/splits_v1/test.json"))
    rng = np.random.default_rng(0)
    sample = list(rng.choice(ids, size=min(N, len(ids)), replace=False))
    rows = []
    t0 = time.time()
    for i, bid in enumerate(sample):
        try:
            sdf = load_sdf(bid)
        except Exception as e:
            print(f"  skip {bid}: {e}", flush=True); continue
        occ = sdf <= 0
        fh = ndimage.binary_fill_holes(occ)
        cf = column_fill(occ)
        rows.append(dict(
            bid=bid,
            raw=float(occ.mean()),
            fillholes=float(fh.mean()),
            colfill=float(cf.mean()),
            boundary_frac=boundary_fraction(occ),
            n_vox=int(occ.sum()),
        ))
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(sample)}] {time.time()-t0:.0f}s", flush=True)

    a = {k: np.array([r[k] for r in rows], float) for k in ("raw", "fillholes", "colfill", "boundary_frac")}
    def pct(x, p): return float(np.nanpercentile(x, p))
    summary = dict(
        n=len(rows),
        raw_occ_median=pct(a["raw"], 50), raw_occ_p10=pct(a["raw"], 10), raw_occ_p90=pct(a["raw"], 90),
        frac_below_0p5pct_raw=float((a["raw"] < 0.005).mean()),
        fillholes_occ_median=pct(a["fillholes"], 50),
        fillholes_gain_median=float(np.nanmedian(a["fillholes"] - a["raw"])),
        colfill_occ_median=pct(a["colfill"], 50), colfill_occ_p10=pct(a["colfill"], 10),
        colfill_gain_x_median=float(np.nanmedian(a["colfill"] / np.maximum(a["raw"], 1e-9))),
        frac_colfill_above_5pct=float((a["colfill"] >= 0.05).mean()),
        frac_colfill_below_0p5pct=float((a["colfill"] < 0.005).mean()),
        boundary_frac_median=pct(a["boundary_frac"], 50),
    )
    out = dict(summary=summary, per_building=rows,
               note="raw=trained-on today; colfill=ground-anchored footprint extrusion (solid massing)")
    (REPO / "execution/artifacts/buildingnet_solidity_probe.json").write_text(json.dumps(out, indent=2))
    print("\n=== SUMMARY ===", flush=True)
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}", flush=True)

    # montage: 6 buildings spanning raw occupancy, cols = raw | fillholes | colfill
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from skimage import measure
        order = np.argsort(a["raw"])
        picks = [rows[order[j]] for j in np.linspace(0, len(rows) - 1, 6).astype(int)]
        fig = plt.figure(figsize=(9, 18))
        for ri, r in enumerate(picks):
            sdf = load_sdf(r["bid"]); occ = sdf <= 0
            variants = [("raw", occ), ("fillholes", ndimage.binary_fill_holes(occ)), ("colfill", column_fill(occ))]
            for ci, (title, m) in enumerate(variants):
                ax = fig.add_subplot(6, 3, ri * 3 + ci + 1, projection="3d")
                ax.set_axis_off()
                if m.sum() > 8:
                    try:
                        v, f, *_ = measure.marching_cubes(m.astype(np.float32), 0.5)
                        ax.plot_trisurf(v[:, 0], v[:, 2], f, v[:, 1], color=(0.72, 0.68, 0.55), lw=0)
                        ax.set_xlim(0, RES); ax.set_ylim(0, RES); ax.set_zlim(0, RES)
                    except Exception:
                        pass
                if ri == 0: ax.set_title(title)
                if ci == 0: ax.text2D(-0.1, 0.5, f"{r['bid'][:22]}\nraw={100*r['raw']:.2f}%", transform=ax.transAxes, fontsize=7)
        mp = REPO / "outputs/buildingnet_solidity/montage.png"
        mp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(mp, dpi=90, bbox_inches="tight"); plt.close(fig)
        print(f"montage: {mp}", flush=True)
    except Exception as e:
        print(f"montage skipped: {e}", flush=True)


if __name__ == "__main__":
    main()
