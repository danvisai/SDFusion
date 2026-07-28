"""Foundations task #1 — benchmark VQVAE reconstruction on CLEAN SDFs (quantify gap #6).

The v1 VQVAE was trained on BuildingNet (broken thin-shell SDFs); it hallucinates surface texture
when reconstructing clean watertight SDFs (the panel artifacts we worked around in the snap with
margin=1.5). This measures that precisely on (a) real 3D BAG held-out buildings and (b) analytic
procedural boxes at varying cube-fill, via the prior's path: encode_only -> decode_no_quant.

Metrics: occ-IoU (recon vs input), surface-band L1, and MC vert-ratio (recon/input = artifact proxy;
>1.5 means hallucinated detail). Baselines the problem; the finetune (task #2) must beat these.

Usage: ... bench_vqvae_recon.py --device cuda [--vq_ckpt <path>]
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
from types import SimpleNamespace
import h5py, numpy as np, torch
from omegaconf import OmegaConf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from models.model_utils import load_vqvae
from scene.sdf_primitives import grid_to_mesh

BBOX = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0); TRUNC = 0.2
V1 = "logs_building/2025-05-19T19-58-28-vqvae-building-all-res64-LR1e-4-T0.2-release/ckpt/vqvae_steps-latest.pth"


def box_sdf(hx, hy, hz, R=64, dev="cpu"):
    g = torch.linspace(-1, 1, R, device=dev)
    ZZ, YY, XX = torch.meshgrid(g, g, g, indexing="ij")          # (D=z,H=y,W=x)
    q = torch.stack([ZZ.abs() - hz, YY.abs() - hy, XX.abs() - hx], 0)
    d = torch.linalg.vector_norm(q.clamp(min=0), dim=0) + q.amax(0).clamp(max=0)
    return d.clamp(-TRUNC, TRUNC).view(1, 1, R, R, R)


def n_verts(sdf):
    m = grid_to_mesh(sdf.detach().cpu()[0, 0], BBOX, iso=0.0)
    return 0 if m is None else len(m.vertices), m


def metrics(x, r):
    ox, orc = (x <= 0), (r <= 0)
    u = (ox | orc).sum().item()
    iou = (ox & orc).sum().item() / u if u else 0.0
    band = (x.abs() < 0.1)
    l1 = float(((r - x).abs() * band).sum() / band.sum().clamp_min(1)) if band.any() else 0.0
    return iou, l1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--vq_ckpt", default=str(REPO / V1))
    ap.add_argument("--bag_h5", default="/dev/shm/bag3d_fast.h5")
    ap.add_argument("--out", default="outputs/foundations/bench_vqvae_v1.png")
    args = ap.parse_args()
    dev = args.device

    vq_conf = OmegaConf.load(REPO / "configs/vqvae_bnet.yaml")
    vqvae = load_vqvae(vq_conf, vq_ckpt=args.vq_ckpt, opt=SimpleNamespace(device=dev, gpu_ids=[0] if dev == "cuda" else []))

    samples = []
    h5p = args.bag_h5 if Path(args.bag_h5).exists() else str(REPO / "data/bag3d_v1/bag3d.h5")
    with h5py.File(h5p, "r") as f:
        for idx in (0, 250, 1000, 4000):
            s = torch.from_numpy(f["sdf"][idx].astype(np.float32)).clamp(-TRUNC, TRUNC).view(1, 1, 64, 64, 64)
            samples.append((f"BAG#{idx}", s.to(dev)))
    for tag, (hx, hy, hz) in [("box fill0.90", (0.9, 0.9, 0.9)), ("box fill0.75", (0.75, 0.75, 0.75)),
                              ("box fill0.60", (0.6, 0.6, 0.6)), ("box tall", (0.5, 0.9, 0.5))]:
        samples.append((tag, box_sdf(hx, hy, hz, dev=dev)))

    print(f"\n=== v1 VQVAE reconstruction on clean SDFs ({dev}) ===")
    print(f"{'sample':14} {'occ-IoU':>8} {'surfL1':>8} {'vIn':>7} {'vOut':>7} {'vRatio':>7}")
    rows = []
    for tag, x in samples:
        with torch.no_grad():
            z = vqvae(x, forward_no_quant=True, encode_only=True)
            r = vqvae.decode_no_quant(z)
        iou, l1 = metrics(x, r)
        vin, mi = n_verts(x); vout, mo = n_verts(r)
        ratio = vout / max(vin, 1)
        print(f"{tag:14} {iou:8.3f} {l1:8.4f} {vin:7d} {vout:7d} {ratio:7.2f}")
        rows.append((tag, x, r, iou, ratio, mi, mo))

    # sheet: rows = samples, cols = input | recon
    n = len(rows); fig = plt.figure(figsize=(6.6, 2.7 * n))
    for i, (tag, x, r, iou, ratio, mi, mo) in enumerate(rows):
        for j, (lab, m) in enumerate([(f"{tag} input", mi), (f"recon iou={iou:.2f} vR={ratio:.1f}", mo)]):
            ax = fig.add_subplot(n, 2, i * 2 + j + 1, projection="3d"); ax.set_title(lab, fontsize=8); ax.set_axis_off()
            if m is not None and len(m.vertices):
                v, fc = np.asarray(m.vertices), np.asarray(m.faces)
                ax.plot_trisurf(v[:, 0], v[:, 2], fc, v[:, 1], color="#b9c4cf", edgecolor="none", shade=True)
                lim = [v.min(), v.max()]; ax.set_xlim(lim); ax.set_ylim(lim); ax.set_zlim(lim)
            ax.view_init(elev=18, azim=-60); ax.set_box_aspect((1, 1, 1))
    fig.suptitle("v1 VQVAE reconstruction of CLEAN SDFs — gap #6 baseline (vRatio>>1 = hallucinated detail)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    outp = REPO / args.out; outp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outp, dpi=90); print(f"[saved] {outp}")


if __name__ == "__main__":
    main()
