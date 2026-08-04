"""Unsupervised MASSING/ROOF-FORM style discovery on 3D BAG (the Dutch corpus).

3D BAG is real LoD2.2 massing — UNLABELED (style_id all 8, class all BAG_real) and facade-less.
So "style" is discovered, not read: extract scale-invariant massing/roof features + absolute
height, cluster (KMeans), and report the cluster proportions = the corpus's style COMPOSITION.
Complements the BuildingNet facade fit (BAG -> massing-style; BuildingNet -> facade-style).

Reads SDFs CONTIGUOUSLY (random access stalls on Lustre).

  env -u LD_PRELOAD -u LD_LIBRARY_PATH PYTHONPATH=. \
    /tmp/sdfusion_venv/bin/python scripts/foundations/bag3d_style_clusters.py [N] [K]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes
from sklearn.cluster import KMeans

H5 = REPO / "data/bag3d_v1/bag3d.h5"
OUT = REPO / "outputs/bag3d_style_clusters"
FEATS = ["height_m", "elongation", "fill_ratio", "roof_pitch", "mass_top_heavy", "footprint_area"]


def massing_features(sdf):
    """Scale-invariant roof/footprint shape + absolute height-independent descriptors.
    sdf axes = (z, y, x); y is height."""
    occ = sdf <= 0
    ys = np.where(occ.any(axis=(0, 2)))[0]
    if len(ys) < 3 or occ.sum() < 20:
        return None
    y0, y1 = ys.min(), ys.max()
    prof = np.array([occ[:, y, :].sum() for y in range(y0, y1 + 1)], np.float32)  # area per level
    prof = prof / (prof.max() + 1e-6)
    mid = prof[len(prof) // 2] + 1e-6
    top = prof[int(len(prof) * 0.85)]
    roof_pitch = float(np.clip(1.0 - top / mid, 0, 1))           # high = pitched/gabled, low = flat
    com_y = float((np.arange(len(prof)) * prof).sum() / (prof.sum() + 1e-6) / max(len(prof) - 1, 1))
    fp = occ.any(axis=1)                                          # XZ silhouette
    zz, xx = np.where(fp)
    dz, dx = np.ptp(zz) + 1, np.ptp(xx) + 1
    elongation = float(max(dz, dx) / max(min(dz, dx), 1))        # 1 = square, >1 = elongated
    fill_ratio = float(fp.sum() / max(dz * dx, 1))               # 1 = rectangular block, low = L/complex
    fp_area = float(fp.sum()) / (64 * 64)                        # relative plan footprint
    return [elongation, fill_ratio, roof_pitch, com_y, fp_area]


def render(ax, sdf, title):
    try:
        v, f, *_ = marching_cubes(sdf, level=0.0)
    except Exception:
        ax.set_axis_off(); ax.set_title(title, fontsize=6); return
    tris = v[f][:, :, [2, 0, 1]]                                 # (z,y,x)->(x,z,y) for plotting
    fy = tris[:, :, 2].mean(1)
    c = plt.cm.bone(0.25 + 0.6 * (fy - fy.min()) / (np.ptp(fy) + 1e-9))
    ax.add_collection3d(Poly3DCollection(tris, facecolors=c, edgecolors="k", linewidths=0.03))
    for setlim in (ax.set_xlim, ax.set_ylim): setlim(0, 63)
    ax.set_zlim(0, 63); ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=16, azim=-60)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.set_title(title, fontsize=6)


def label_cluster(centroid_z):
    """Short descriptive name from standardized centroid (z-scores over FEATS)."""
    h, el, fill, pitch, toph, area = centroid_z
    parts = []
    parts.append("tall" if h > 0.6 else ("low" if h < -0.6 else "mid-rise"))
    parts.append("pitched" if pitch > 0.3 else ("flat-roof" if pitch < -0.5 else "mixed-roof"))
    if el > 0.7: parts.append("elongated")
    elif fill < -0.7: parts.append("complex-plan")
    elif area > 0.7: parts.append("large-plan")
    return " ".join(parts[:3])


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    with h5py.File(H5, "r") as f:
        N = min(N, f["sdf"].shape[0])
        heights = f["height_m"][:N].astype(np.float32)
        sdfs, feats, keep = [], [], []
        for a in range(0, N, 250):                               # contiguous chunked read
            blk = f["sdf"][a:min(a + 250, N)]
            for j in range(blk.shape[0]):
                mf = massing_features(blk[j])
                if mf is None:
                    continue
                feats.append([heights[a + j]] + mf); keep.append(a + j)
                sdfs.append(blk[j] if len(sdfs) < 4000 else None)  # keep volumes for exemplars
    feats = np.asarray(feats, np.float32)
    # X cols: [height_m, elongation, fill, roof_pitch, com_y(top-heavy), fp_area]
    print(f"[features] {len(feats)} buildings in {time.time()-t0:.0f}s")

    mean, std = feats.mean(0), feats.std(0) + 1e-6
    Z = (feats - mean) / std
    km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(Z)
    lab = km.labels_
    order = np.argsort(-np.bincount(lab, minlength=K))           # biggest cluster first
    remap = {c: i for i, c in enumerate(order)}
    names, comp = {}, {}
    for c in order:
        names[c] = label_cluster(km.cluster_centers_[c])
        comp[names[c]] = round(float((lab == c).mean()), 3)

    # ---- sheet: composition bar (left) + K rows x 3 exemplars ----
    fig = plt.figure(figsize=(3.1 * 4, 2.7 * K))
    gs = fig.add_gridspec(K, 4)
    axb = fig.add_subplot(gs[:, 0])
    sizes = [(lab == c).mean() for c in order]
    axb.barh(range(K), sizes[::-1], color=plt.cm.viridis(np.linspace(0.15, 0.85, K))[::-1])
    axb.set_yticks(range(K)); axb.set_yticklabels([f"{names[c]}\n{comp[names[c]]*100:.0f}%"
                                                   for c in order][::-1], fontsize=7)
    axb.set_xlabel("share of corpus"); axb.set_title("3D BAG style composition\n(discovered)", fontsize=9)

    feat_med = {}
    for ri, c in enumerate(order):
        idx = np.where(lab == c)[0]
        med = feats[idx].mean(0)
        feat_med[names[c]] = {k: round(float(v), 2) for k, v in zip(FEATS, med)}
        ex = idx[np.argsort(((Z[idx] - km.cluster_centers_[c]) ** 2).sum(1))][:3]  # near-centroid
        for cj, e in enumerate(ex):
            ax = fig.add_subplot(gs[ri, cj + 1], projection="3d")
            sd = sdfs[e]
            ttl = (f"{names[c]}" if cj == 1 else
                   f"h={feats[e,0]:.0f}m pitch={feats[e,3]:.2f}")
            render(ax, sd, ttl) if sd is not None else ax.set_axis_off()
    fig.suptitle(f"3D BAG (Dutch LoD2.2) — unsupervised massing/roof-form style clusters "
                 f"(N={len(feats)}, K={K}). Labels are RELATIVE to the corpus "
                 f"(flat = lower-pitch than average)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "bag3d_style_sheet.png", dpi=120); plt.close(fig)

    json.dump({"n": len(feats), "k": K, "composition": comp,
               "feature_means_per_cluster": feat_med, "feats": FEATS},
              open(OUT / "clusters.json", "w"), indent=2)
    print("[composition]", json.dumps(comp))
    print(f"[save] {OUT/'bag3d_style_sheet.png'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
