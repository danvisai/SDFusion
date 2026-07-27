"""#53 — de-risk the residual-over-extrusion reframing on real LoD2 (GT decomposition, NO training).

Decompose each real.h5 GT SDF as:  GT = extrusion_prior + residual
where extrusion_prior extrudes GT's OWN mid-height cross-section vertically over [y0,y1]. Using the
mid-height slice (a genuinely crisp 2D SDF taken straight from the GT mesh, NOT an EDT of the
rasterized 64^2 mask) makes the prior as crisp as the data — the earlier EDT-of-mask prior
staircased and was falsely "rough".

Three questions:
  1) Is the extrusion prior crisp & footprint-faithful?  -> rough(ext) vs rough(GT); fp-IoU(ext, GT).
  2) Is the residual SMALL?  -> RMS(residual)/RMS(GT).
  3) Is the residual STRUCTURED / off the walls (so the walls are 100% the crisp analytic prior and
     immune to whatever a residual-predictor does)?  -> mean|residual| on the vertical-wall band vs
     the roof band, and the per-height |residual| profile.

If ext is crisp + footprint-exact AND the residual is small & concentrated at the roof/setbacks, then
a model that predicts only the residual keeps crisp walls by construction -> greenlight lever (1).

Deliverables: outputs/residual_decomp/{montage.png, residual_structure.png, metrics.json}.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

REPO = Path("/scratch/gilbreth/dsimhadr/GenerativeTowns/SDFusion")
sys.path.insert(0, str(REPO))
from scripts.foundations.baseline_gate_eval import mesh_sdf_surface           # noqa: E402
from scripts.foundations.refiner_prototype import surface_roughness           # noqa: E402

H5 = REPO / "data/real_massing_v1/real.h5"
OUT = REPO / "outputs/residual_decomp"
TRUNC = 0.2
VOX = 2.0 / 63.0
N = 8
MONTAGE_N = 6
TAN = (0.82, 0.75, 0.60)


def rough(vol):
    return surface_roughness(torch.from_numpy(np.asarray(vol, np.float32)))


def extrusion_from_midslice(gt, y0, y1):
    """Extrude GT's mid-height cross-section over [y0,y1]. d2d = crisp 2D SDF straight from GT."""
    ymid = (y0 + y1) // 2
    d2d = gt[:, ymid, :].astype(np.float64)                       # (z,x), smooth SDF from the mesh
    yi = np.arange(64)
    dy = np.maximum(y0 - yi, yi - y1).astype(np.float64) * VOX    # (y,), <0 inside slab
    w0 = d2d[:, None, :]; w1 = dy[None, :, None]
    ext = np.minimum(np.maximum(w0, w1), 0.0) + np.sqrt(np.maximum(w0, 0) ** 2 + np.maximum(w1, 0) ** 2)
    return np.clip(ext, -TRUNC, TRUNC).astype(np.float32)


def orient_fp(fp, gt):
    gt_fp = (gt <= 0).any(axis=1)
    def iou(a, b):
        a = a > 0.5; b = b > 0.5; u = (a | b).sum(); return float((a & b).sum() / u) if u else 0.0
    return fp if iou(fp, gt_fp) >= iou(fp.T, gt_fp) else fp.T


def fp_iou(occ_zyx, ref_zx):
    g = occ_zyx.any(axis=1); r = ref_zx > 0.5
    u = (g | r).sum(); return float((g & r).sum() / u) if u else 0.0


def wall_roof_residual(residual, fp_zx, y0, y1):
    mask = fp_zx > 0.5
    db = np.abs(distance_transform_edt(~mask) - distance_transform_edt(mask))     # voxels from boundary
    near_wall = (db <= 2.0)[:, None, :]
    yi = np.arange(64)[None, :, None]
    mid = (yi >= y0 + 3) & (yi <= y1 - 3)
    roof = (yi >= y1 - 2) & (yi <= y1 + 1)
    a = np.abs(residual)
    ws = np.broadcast_to(near_wall & mid, residual.shape); rs = np.broadcast_to(roof, residual.shape)
    return (float(a[ws].mean()) if ws.any() else 0.0), (float(a[rs].mean()) if rs.any() else 0.0)


def main():
    import h5py
    OUT.mkdir(parents=True, exist_ok=True)
    f = h5py.File(H5, "r")
    pick = np.random.default_rng(1).choice(f["sdf"].shape[0], size=N, replace=False)
    rows, samples = [], []
    for gi in sorted(pick.tolist()):
        gt = np.clip(f["sdf"][gi].astype(np.float32), -TRUNC, TRUNC)
        fp = orient_fp(f["footprint"][gi].astype(np.float32), gt)
        occ = gt <= 0
        ys = np.where(occ.any(axis=(0, 2)))[0]
        if len(ys) < 6:
            continue
        y0, y1 = int(ys.min()), int(ys.max())
        ext = extrusion_from_midslice(gt, y0, y1)
        residual = gt - ext
        wr, rr = wall_roof_residual(residual, fp, y0, y1)
        rec = dict(gi=gi, y0=y0, y1=y1,
                   resid_rms_ratio=float(np.sqrt((residual ** 2).mean()) / (np.sqrt((gt ** 2).mean()) + 1e-9)),
                   resid_absmax=float(np.abs(residual).max()),
                   wall_resid_mean=wr, roof_resid_mean=rr, roof_over_wall=float(rr / (wr + 1e-9)),
                   fp_iou_ext=fp_iou(ext <= 0, fp),
                   rough_gt=rough(gt), rough_ext=rough(ext))
        rows.append(rec); samples.append((gi, gt, ext, residual, fp, y0, y1))
        print(f"  gi={gi}: resid_rms_ratio={rec['resid_rms_ratio']:.3f} "
              f"wall|r|={wr:.4f} roof|r|={rr:.4f} roof/wall={rec['roof_over_wall']:.2f} "
              f"fp_iou_ext={rec['fp_iou_ext']:.3f} rough gt={rec['rough_gt']:.5f} ext={rec['rough_ext']:.5f}",
              flush=True)
    f.close()

    agg = {k: float(np.mean([r[k] for r in rows])) for k in
           ("resid_rms_ratio", "wall_resid_mean", "roof_resid_mean", "roof_over_wall",
            "fp_iou_ext", "rough_gt", "rough_ext")}
    (OUT / "metrics.json").write_text(json.dumps(dict(n=len(rows), aggregate=agg, per_building=rows), indent=2))
    print("\n=== AGGREGATE ===")
    for k, v in agg.items():
        print(f"  {k}: {v:.5f}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    ls = LightSource(azdeg=315, altdeg=50)
    ms = samples[:MONTAGE_N]
    # montage: GT | extrusion prior (the crisp "clean simple base" a residual-model would start from)
    fig = plt.figure(figsize=(7, 3.3 * len(ms)))
    for ri, (gi, gt, ext, residual, fp, y0, y1) in enumerate(ms):
        for ci, (title, vol) in enumerate([("GT (real LoD2)", gt), ("extrusion prior (crisp base)", ext)]):
            ax = fig.add_subplot(len(ms), 2, ri * 2 + ci + 1, projection="3d"); ax.set_axis_off()
            v, fc = mesh_sdf_surface(vol)
            if v is not None:
                ax.plot_trisurf(v[:, 2], v[:, 0], fc, v[:, 1], color=TAN, shade=True, lightsource=ls,
                                edgecolor="none", linewidth=0, antialiased=False)
                ax.set_xlim(0, 64); ax.set_ylim(0, 64); ax.set_zlim(0, 64)
                try: ax.set_box_aspect((1, 1, 1))
                except Exception: pass
            ax.view_init(elev=24, azim=-58)
            if ri == 0: ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "montage.png", dpi=112, bbox_inches="tight"); plt.close(fig)

    # residual structure: mid-x |residual| slice (+ GT/ext 0-contours) | per-height profile
    fig = plt.figure(figsize=(9, 3.0 * len(ms)))
    for ri, (gi, gt, ext, residual, fp, y0, y1) in enumerate(ms):
        xm = gt.shape[2] // 2
        ax = fig.add_subplot(len(ms), 2, ri * 2 + 1)
        ax.imshow(np.abs(residual[:, :, xm]).T, origin="lower", cmap="magma", vmin=0, vmax=TRUNC * 0.5)
        ax.contour(gt[:, :, xm].T, levels=[0], colors="cyan", linewidths=1.0)
        ax.contour(ext[:, :, xm].T, levels=[0], colors="lime", linewidths=0.8, linestyles="--")
        ax.set_title(f"gi={gi}  |residual| mid-slice  (cyan=GT-0, green=ext-0)", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        ax2 = fig.add_subplot(len(ms), 2, ri * 2 + 2)
        ax2.plot(np.arange(64), np.abs(residual).mean(axis=(0, 2)), color="crimson")
        ax2.axvspan(y0, y1, color="0.85", zorder=0)
        ax2.set_title("mean|residual| per height (grey=building)", fontsize=8)
        ax2.tick_params(labelsize=6)
    fig.tight_layout(); fig.savefig(OUT / "residual_structure.png", dpi=112, bbox_inches="tight"); plt.close(fig)
    print(f"\nSAVED: {OUT}/montage.png , {OUT}/residual_structure.png , {OUT}/metrics.json")


if __name__ == "__main__":
    main()
