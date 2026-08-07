"""#82: can height be inferred from the footprint, and what does inferring it cost?

Answer: only to R^2 ~ 0.55, and it costs the blockout baseline -0.074 mean 3D IoU with a 37.5% rate
of under-building by >10% of GT volume. Footprint-only is a genuinely WEAKER task definition, not a
free alternative -- which reinforces #81's decision to treat height as a user input.

Run: env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/probe_height_inference.py

Step 2 found buildings are per-instance normalised: corr(voxel extent, height_m) = 0.43 and
voxels-per-metre spans 3.8-6.8. So `height_m` is not what `blockout_sdf` eats -- it eats (y0, y1),
and the footprint mask lives in that same normalised frame. Predicting the extent is the well-posed
version of "infer height from the footprint".

Then the re-score the ticket demands: blockout with the SPECIFIED extent (today's baseline) against
blockout with the INFERRED extent, on the pinned ids, side by side.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path("/scratch/gilbreth/dsimhadr/GenerativeTowns/SDFusion")
sys.path.insert(0, str(ROOT))

from scripts.foundations.eval_massing_arms import (_vertical_extent, blockout_sdf,  # noqa: E402
                                                   pick_ids, score_arm, summarise)


def feats(fp):
    """Hand features from a 64x64 binary footprint: area, perimeter, compactness, bbox, elongation."""
    from scipy import ndimage
    m = fp > 0
    area = m.sum()
    if area == 0:
        return np.zeros(7)
    per = int((m ^ ndimage.binary_erosion(m)).sum())
    ys, xs = np.nonzero(m)
    h, w = np.ptp(ys) + 1, np.ptp(xs) + 1
    bbox = h * w
    return np.array([np.log1p(area), np.log1p(per), 4 * np.pi * area / max(per ** 2, 1),
                     np.log1p(bbox), area / max(bbox, 1),
                     max(h, w) / max(min(h, w), 1), np.log1p(max(h, w))], float)


def r2(y, p):
    return 1.0 - float(((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())

H5 = ROOT / "data/real_massing_v1/real.h5"
LAT = ROOT / "data/real_massing_v1/vecset_latents.h5"


def main():
    import h5py
    from sklearn.ensemble import HistGradientBoostingRegressor

    with h5py.File(LAT, "r") as f:
        held = f["held_out"][:] == 1
        reg = f["region"][:].astype(int)
        fps = f["footprint"][:]
        rows = f["row"][:]

    # ---- targets: the extent the harness actually consumes --------------------------------------
    cache = Path(__file__).resolve().parents[2] / "outputs/height_inference_extents.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        d = np.load(cache); y0, y1, ok = d["y0"], d["y1"], d["ok"]
        print("[extract] extents from cache", flush=True)
    else:
        print("[extract] reading GT vertical extents ...", flush=True)
        y0s, y1s, ok = [], [], []
        with h5py.File(H5, "r") as g:
            for i, r in enumerate(rows):
                ext = _vertical_extent(np.asarray(g["sdf"][int(r)], np.float32) <= 0)
                if ext is None:
                    y0s.append(0); y1s.append(0); ok.append(False)
                else:
                    y0s.append(ext[0]); y1s.append(ext[1]); ok.append(True)
                if (i + 1) % 8000 == 0:
                    print(f"   {i+1}/{len(rows)}", flush=True)
        y0 = np.array(y0s); y1 = np.array(y1s); ok = np.array(ok)
        np.savez(cache, y0=y0, y1=y1, ok=ok)
    span = (y1 - y0 + 1).astype(float)

    print(f"\n=== the target the harness consumes ===")
    print(f"  y0 (ground): mean {y0[ok].mean():.2f}  sd {y0[ok].std():.2f}  "
          f"unique<=3: {len(np.unique(y0[ok]))<=3}")
    print(f"  span (voxels): mean {span[ok].mean():.1f}  sd {span[ok].std():.1f}")

    X = np.stack([feats(fp) for fp in fps])
    Xf = np.c_[X, reg]
    tr, te = ok & ~held, ok & held

    g1 = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                       categorical_features=[X.shape[1]], random_state=0)
    g1.fit(Xf[tr], span[tr])
    pspan = g1.predict(Xf[te])
    print(f"\n=== predicting the VOXEL SPAN from footprint+region (n_test={te.sum()}) ===")
    print(f"  B0 mean span      R2 {r2(span[te], np.full(te.sum(), span[tr].mean())):>7.3f}")
    print(f"  GBM               R2 {r2(span[te], pspan):>7.3f}   MAE {np.abs(span[te]-pspan).mean():.2f} vox"
          f"   median |err| {np.median(np.abs(span[te]-pspan)):.2f} vox")
    print(f"  (for reference, predicting metric height_m gave R2 0.537)")

    g0 = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06,
                                       categorical_features=[X.shape[1]], random_state=0)
    g0.fit(Xf[tr], y0[tr].astype(float))
    py0 = g0.predict(Xf[te])
    print(f"  y0 (ground) GBM   R2 {r2(y0[te].astype(float), py0):>7.3f}   "
          f"MAE {np.abs(y0[te]-py0).mean():.2f} vox")

    # ---- the re-score, on the pinned ids ---------------------------------------------------------
    cand, lat_of = pick_ids(LAT, None)
    ids = []
    scores = {"blockout_specified": {}, "blockout_inferred": {}}
    idx_of = {int(r): i for i, r in enumerate(rows)}
    print(f"\n[rescore] on the pinned held-out ids ...", flush=True)
    with h5py.File(H5, "r") as g:
        for bid in cand:
            if len(ids) >= 48:
                break
            i = idx_of[bid]
            if not ok[i]:
                continue
            fp = fps[i]
            if not fp.any():
                continue
            gocc = np.asarray(g["sdf"][bid], np.float32) <= 0
            ext = _vertical_extent(gocc)
            bo_spec = blockout_sdf(fp, *ext)
            # inferred: same ground level, predicted span (clamped into the grid)
            ps = float(g1.predict(np.c_[X[i:i+1], reg[i:i+1]])[0])
            p0 = int(round(float(g0.predict(np.c_[X[i:i+1], reg[i:i+1]])[0])))
            p0 = max(0, min(63, p0))
            p1 = max(p0, min(63, p0 + int(round(ps)) - 1))
            bo_inf = blockout_sdf(fp, p0, p1)
            if bo_spec is None or bo_inf is None:
                continue
            scores["blockout_specified"][bid] = score_arm(bo_spec, gocc, fp)
            scores["blockout_inferred"][bid] = score_arm(bo_inf, gocc, fp)
            ids.append(bid)

    print(f"\n=== #82 RE-SCORE: specified height vs inferred height (n={len(ids)} pinned ids) ===")
    print("\n-- as the harness reports it (PER-COLUMN MEDIANS -- the four numbers in a row are NOT")
    print("   the same building, and the map already recorded that medians lie on bimodal outcomes) --")
    print(f"{'arm':<24}{'fp-IoU':>9}{'missing':>10}{'extra':>9}{'3D IoU':>9}")
    for a in ("blockout_specified", "blockout_inferred"):
        s = summarise(scores[a].values())
        print(f"{a:<24}{s['fp_iou']:>9.3f}{s['missing']:>10.3f}{s['extra']:>9.3f}{s['vol_iou']:>9.3f}")

    print("\n-- the distribution, which is what actually changed --")
    print(f"{'arm':<24}{'IoU mean':>10}{'IoU med':>9}{'IoU p10':>9}"
          f"{'miss mean':>11}{'miss med':>10}{'>10% miss':>11}")
    for a in ("blockout_specified", "blockout_inferred"):
        v = list(scores[a].values())
        iou = np.array([r["vol_iou"] for r in v]); mis = np.array([r["missing"] for r in v])
        print(f"{a:<24}{iou.mean():>10.3f}{np.median(iou):>9.3f}{np.percentile(iou,10):>9.3f}"
              f"{mis.mean():>11.3f}{np.median(mis):>10.3f}{(mis>0.10).mean()*100:>10.1f}%")

    sp = np.array([scores['blockout_specified'][b]["vol_iou"] for b in ids])
    inf = np.array([scores['blockout_inferred'][b]["vol_iou"] for b in ids])
    print(f"\n  per-building 3D IoU change: mean {np.mean(inf-sp):+.3f}  median {np.median(inf-sp):+.3f}"
          f"  worst {np.min(inf-sp):+.3f}  buildings hurt {(inf<sp).mean()*100:.0f}%")
    print("\n  ⚠️  A2's published 0.962/0.002/0.191/0.838 is against the SPECIFIED-height task.")
    print("      It is NOT comparable to the inferred row -- that is a different task definition.")


if __name__ == "__main__":
    main()
