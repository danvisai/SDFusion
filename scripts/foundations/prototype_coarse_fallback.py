"""Ticket #29 prototype: does #28's target solidification alone kill the empty-coarse-input
collapse, or is an explicit footprint-extrusion fallback on the COARSE input still needed?

The collapse: coarse = low_pass_sdf(target); thin-shell targets band-limit to an EMPTY coarse at
the s* grid (~19^3), so the model sees a blank conditioning signal and emits a byte-identical grid.
#28 makes the TARGET a solid (hybrid footprint-extrusion, keep-centered lowest-to-top). This probe
asks, on the buildings whose OLD coarse is empty: is the NEW coarse (low-pass of the solidified
target) now non-empty and distinct per building?

CPU only, faithful to ADR 0004 (WORKING_RES=96, s*=5 vox -> coarse res ~19). Run:
  env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/prototype_coarse_fallback.py
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np, h5py, torch
import torch.nn.functional as F
from scipy import ndimage

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data/BuildingNet_dataset_v0_1/resolution_64"
NATIVE, WRES, SVOX, UP = 64, 96, 5, 1
CRES = round(WRES / SVOX)  # ~19, ADR 0004 coarse resolution


def _resample(g, out):
    t = torch.as_tensor(g, dtype=torch.float32)[None, None]
    return F.interpolate(t, size=(out,) * 3, mode="trilinear", align_corners=True)[0, 0].numpy()


def low_pass(sdf):  # ADR 0004 monolith coarse primary: down to CRES then back up
    return _resample(_resample(sdf, CRES), WRES)


def signed_edt(mask):  # solid occupancy mask -> SDF (outside +, inside -), voxel units
    inside = ndimage.distance_transform_edt(mask)
    outside = ndimage.distance_transform_edt(~mask)
    return (outside - inside).astype(np.float32)


def solidify(occ, footprint96):
    """#28 hybrid, keep-centered lowest-to-top. Per-column occupancy extrusion; stored-footprint
    fallback fills empty footprint columns across the building's global occupied band (centered
    default band if the building is essentially empty)."""
    m = np.moveaxis(occ, UP, 0)          # (H, A, B)
    H = m.shape[0]
    colocc = m.any(axis=0)               # columns with real geometry
    hh = np.arange(H)[:, None, None]
    # per-column lowest..highest occupied
    low = np.where(colocc, np.argmax(m, axis=0), H)
    high = np.where(colocc, H - 1 - np.argmax(m[::-1], axis=0), -1)
    out = (hh >= low[None]) & (hh <= high[None]) & colocc[None]
    # stored-footprint fallback for footprint columns lacking occupancy
    fp2d = footprint96
    if m.any():
        occ_levels = np.where(m.any(axis=(1, 2)))[0]
        band_lo, band_hi = int(occ_levels.min()), int(occ_levels.max())
        if band_hi - band_lo < 4:  # essentially empty -> centered default band
            c = H // 2; band_lo, band_hi = c - H // 5, c + H // 5
    else:
        c = H // 2; band_lo, band_hi = c - H // 5, c + H // 5
    need_fallback = fp2d & (~colocc)
    band = (hh >= band_lo) & (hh <= band_hi)
    out = out | (band & need_fallback[None])
    return np.moveaxis(out, 0, UP)


def load(bid):
    with h5py.File(DATA / bid / "ori_sample_grid.h5", "r") as f:
        sdf = np.asarray(f["pc_sdf_sample"]).reshape(NATIVE, NATIVE, NATIVE).astype(np.float32)
        fp = np.asarray(f["footprint"]); fp = fp[0] if fp.ndim == 3 else fp
    return sdf, fp.astype(bool)


def fp_to_96(fp):
    t = torch.as_tensor(fp.astype(np.float32))[None, None]
    return (F.interpolate(t, size=(WRES, WRES), mode="nearest")[0, 0].numpy() > 0.5)


def main():
    ids = json.load(open(REPO / "data/splits_v1/test.json"))
    rng = np.random.default_rng(0)
    sample = list(rng.choice(ids, size=60, replace=False))
    rows, collapsing = [], []
    for bid in sample:
        sdf64, fp64 = load(bid)
        tgt = _resample(sdf64, WRES)
        old_coarse = low_pass(tgt)
        occ_old = float((old_coarse <= 0).mean())
        solid = solidify(tgt <= 0, fp_to_96(fp64))
        new_coarse = low_pass(signed_edt(solid))
        occ_new = float((new_coarse <= 0).mean())
        h = hashlib.md5((new_coarse <= 0).tobytes()).hexdigest()[:8]
        r = dict(bid=bid, occ_old_coarse=occ_old, occ_new_coarse=occ_new, solid_occ=float(solid.mean()), hash=h)
        rows.append(r)
        if occ_old < 1e-4:  # the collapse condition
            collapsing.append(r)

    print(f"sample={len(rows)}  collapsing(old coarse empty)={len(collapsing)}", flush=True)
    if collapsing:
        occ_new = np.array([r["occ_new_coarse"] for r in collapsing])
        hashes = [r["hash"] for r in collapsing]
        print(f"  after #28 solidification, NEW coarse on the collapsing set:", flush=True)
        print(f"    non-empty (occ>1e-4): {int((occ_new>1e-4).sum())}/{len(collapsing)}", flush=True)
        print(f"    new coarse occ: median {np.median(occ_new)*100:.2f}%  min {occ_new.min()*100:.2f}%", flush=True)
        print(f"    distinct grids: {len(set(hashes))}/{len(hashes)}  (byte-identical collapse => 1)", flush=True)
    out = REPO / "execution/artifacts/coarse_fallback_prototype.json"
    out.write_text(json.dumps(dict(n=len(rows), n_collapsing=len(collapsing), collapsing=collapsing, all=rows), indent=2))
    print(f"  artifact: {out}", flush=True)

    # montage: collapsing buildings, cols = old coarse | solid target | new coarse
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        from skimage import measure
        picks = collapsing[:5] if len(collapsing) >= 5 else collapsing
        if picks:
            fig = plt.figure(figsize=(9, 3 * len(picks)))
            for ri, r in enumerate(picks):
                sdf64, fp64 = load(r["bid"]); tgt = _resample(sdf64, WRES)
                solid = solidify(tgt <= 0, fp_to_96(fp64))
                grids = [("old coarse (empty)", low_pass(tgt) <= 0),
                         ("solid target (#28)", solid),
                         ("new coarse", low_pass(signed_edt(solid)) <= 0)]
                for ci, (title, m) in enumerate(grids):
                    ax = fig.add_subplot(len(picks), 3, ri * 3 + ci + 1, projection="3d"); ax.set_axis_off()
                    if m.sum() > 8:
                        try:
                            v, f, *_ = measure.marching_cubes(m.astype(np.float32), 0.5)
                            ax.plot_trisurf(v[:, 0], v[:, 2], f, v[:, 1], color=(0.72, 0.68, 0.55), lw=0)
                            ax.set_xlim(0, WRES); ax.set_ylim(0, WRES); ax.set_zlim(0, WRES)
                        except Exception: pass
                    if ri == 0: ax.set_title(title)
                    if ci == 0: ax.text2D(-0.1, 0.5, r["bid"][:20], transform=ax.transAxes, fontsize=7)
            mp = REPO / "outputs/coarse_fallback_prototype/montage.png"; mp.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(mp, dpi=90, bbox_inches="tight"); plt.close(fig)
            print(f"  montage: {mp}", flush=True)
    except Exception as e:
        print(f"  montage skipped: {e}", flush=True)


if __name__ == "__main__":
    main()
