"""Map #34 ticket #35 — diagnose WHERE the generated massing's surface roughness enters.

Ladder (per audit_stage3a_inference.py), on the from-scratch LoD2 checkpoint:
  A  real GT SDF                              -> reference (crisp real LoD2)
  B  VQVAE no-quant round-trip of the GT SDF  -> the DECODE CEILING. If B is already rough,
                                                 the VQVAE caps crispness and no prior change helps.
  C  full prior sample                        -> what we currently see.
Plus a render-vs-geometry check: a mid-height SDF slice of each, to see whether striations live in
the field itself or are a marching-cubes/threshold artifact.

CPU-metrics + GPU generation. Run:
  env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/diagnose_surface_roughness.py --n 6
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy import ndimage

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts" / "foundations"))
from baseline_gate_eval import build_opt  # reuse the tested model-loading opt

CKPT = "logs_building/2026-07-16-stage3a-lod2-fromscratch-region/ckpt/stage3a_steps-latest.pth"


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def surf_ratio(occ):
    """boundary voxels / occupied voxels -- higher = rougher/thinner surface, lower = crisp solid."""
    n = int(occ.sum())
    if n == 0:
        return float("nan")
    boundary = occ & ~ndimage.binary_erosion(occ)
    return float(boundary.sum() / n)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=6); a = ap.parse_args()
    import torch
    from datasets.bag3d_dataset import Bag3dDataset
    from models.stage3a_model import Stage3aModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    opt = build_opt(dev, ckpt=CKPT, use_region=True, use_extra_cond=False)
    print(f"[load] {CKPT}", flush=True)
    model = Stage3aModel(); model.initialize(opt)
    ds = Bag3dDataset(); ds.initialize(opt, phase="test")
    pick = np.random.default_rng(0).choice(len(ds), size=a.n, replace=False)

    rows, tri, sl = [], [], []
    for j, idx in enumerate(pick):
        item = ds[int(idx)]
        gt = item["sdf"].numpy()[0]
        x = item["sdf"].unsqueeze(0).to(dev)
        data = {k: (v.unsqueeze(0).to(dev) if torch.is_tensor(v) else v)
                for k, v in item.items() if torch.is_tensor(v)}
        with torch.no_grad():
            z = model.vqvae(x, forward_no_quant=True, encode_only=True)   # B: encode
            rt = model.vqvae.decode_no_quant(z).detach().cpu().numpy()[0, 0]  # B: decode ceiling
            samp = model.inference(data, ddim_steps=100).detach().cpu().numpy()[0, 0]  # C
        og, orr, os_ = gt <= 0, rt <= 0, samp <= 0
        rows.append(dict(idx=int(idx),
                         iou_roundtrip_gt=iou(orr, og), iou_sample_gt=iou(os_, og),
                         surf_gt=surf_ratio(og), surf_roundtrip=surf_ratio(orr), surf_sample=surf_ratio(os_)))
        print(f"  [{j+1}/{a.n}] iou(B,A)={rows[-1]['iou_roundtrip_gt']:.3f} iou(C,A)={rows[-1]['iou_sample_gt']:.3f}"
              f"  surf A={rows[-1]['surf_gt']:.3f} B={rows[-1]['surf_roundtrip']:.3f} C={rows[-1]['surf_sample']:.3f}", flush=True)
        if len(tri) < 6:
            # store the CONTINUOUS SDF fields (not binary occ) so the geometry montage meshes
            # at 0.0 -> true surfaces, not a binary marching-cubes staircase artifact (#39).
            tri.append((gt.copy(), rt.copy(), samp.copy()))
            h = gt.shape[1] // 2
            sl.append((gt[:, h, :].copy(), rt[:, h, :].copy(), samp[:, h, :].copy()))

    arr = lambda k: np.array([r[k] for r in rows], float)
    summary = dict(
        n=len(rows),
        iou_roundtrip_gt_median=float(np.median(arr("iou_roundtrip_gt"))),
        iou_sample_gt_median=float(np.median(arr("iou_sample_gt"))),
        surf_gt_median=float(np.median(arr("surf_gt"))),
        surf_roundtrip_median=float(np.median(arr("surf_roundtrip"))),
        surf_sample_median=float(np.median(arr("surf_sample"))),
    )
    (REPO / "execution/artifacts/surface_roughness_diagnosis.json").write_text(
        json.dumps(dict(summary=summary, per_building=rows), indent=2))
    print("\n=== SUMMARY ===", flush=True)
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}", flush=True)

    # geometry montage: A GT | B roundtrip | C sample
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        from skimage import measure
        outdir = REPO / "outputs/surface_roughness"; outdir.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(9, 3 * len(tri)))
        for ri, (og, orr, os_) in enumerate(tri):
            for ci, (title, m) in enumerate([("A: real GT", og), ("B: VQVAE round-trip", orr), ("C: prior sample", os_)]):
                ax = fig.add_subplot(len(tri), 3, ri * 3 + ci + 1, projection="3d"); ax.set_axis_off()
                if (m <= 0).sum() > 8 and (m > 0).any():
                    try:
                        v, f, *_ = measure.marching_cubes(m.astype(np.float32), 0.0)
                        ax.plot_trisurf(v[:, 0], v[:, 2], f, v[:, 1], color=(0.72, 0.68, 0.55), lw=0)
                        ax.set_xlim(0, 64); ax.set_ylim(0, 64); ax.set_zlim(0, 64)
                    except Exception: pass
                if ri == 0: ax.set_title(title)
        fig.savefig(outdir / "ladder_montage.png", dpi=90, bbox_inches="tight"); plt.close(fig)
        # slice montage: mid-height SDF field (field noise vs render artifact)
        fig2, axes = plt.subplots(len(sl), 3, figsize=(9, 3 * len(sl)))
        for ri, (sg, sr, ss) in enumerate(sl):
            for ci, (title, s) in enumerate([("A field", sg), ("B field", sr), ("C field", ss)]):
                ax = axes[ri, ci] if len(sl) > 1 else axes[ci]
                ax.imshow(np.clip(s, -0.2, 0.2), cmap="coolwarm"); ax.axis("off")
                ax.contour(s, levels=[0.0], colors="k", linewidths=0.6)
                if ri == 0: ax.set_title(title)
        fig2.savefig(outdir / "slice_montage.png", dpi=90, bbox_inches="tight"); plt.close(fig2)
        print(f"montages: {outdir}/ladder_montage.png , slice_montage.png", flush=True)
    except Exception as e:
        print(f"montage skipped: {e}", flush=True)


if __name__ == "__main__":
    main()
