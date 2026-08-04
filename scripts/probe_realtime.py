"""Two cheap probes for the real-time + generative question (no commitment, just evidence).

PROBE 1 — latency: how fast is SDEdit vs the number of DDIM steps? Times the full sdedit() call
(encode + denoise + decode) at several step budgets, plus the fixed encode/decode cost. Answers:
is generation cheap enough to run IN the edit loop (placement #1), or must we go async (#2)?
Uses uc_scale=1.0 (pure conditional, 1 UNet pass/step — the fastest, cleanest config; autoguidance
≈ 2x these numbers).

PROBE 2 — latent geometry: encode two real buildings, interpolate + perturb the VQVAE latent, decode.
If intermediates are PLAUSIBLE buildings, then DualSDF-style real-time handle-dragging (placement #3,
no diffusion at edit time) is viable; if they're garbage, that route is out.

GPU recommended: --device cuda.
"""
from __future__ import annotations
import argparse, sys, time, statistics
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from datasets.bag3d_dataset import Bag3dDataset
from models.stage3a_model import Stage3aModel
from scene.sdf_primitives import grid_to_mesh

BBOX = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0); TRUNC = 0.2
CKPT_DIR = REPO / "logs_building/2026-06-05T15-02-24-bag3d-prior-fast/ckpt"


def mesh(sdf):
    return grid_to_mesh(sdf.detach().cpu()[0, 0], BBOX, iso=0.0)


def load_model(ckpt, dev):
    opt = SimpleNamespace(isTrain=False, device=dev, df_cfg=str(REPO/"configs/stage3a_sdf_diffusion.yaml"),
                          vq_cfg=str(REPO/"configs/vqvae_bnet.yaml"),
                          vq_ckpt=str(REPO/"logs_building/2025-05-19T19-58-28-vqvae-building-all-res64-LR1e-4-T0.2-release/ckpt/vqvae_steps-latest.pth"),
                          ckpt=str(ckpt), ddim_steps=50, debug="0", gpu_ids=[0] if dev=="cuda" else [],
                          ckpt_dir="/tmp", latent_size_HW=(16,16), latent_size_D=16)
    print(f"[load] {Path(ckpt).name} on {dev}")
    m = Stage3aModel(); m.initialize(opt); return m


def tower_edit(sdf, dev):
    s = sdf[0, 0].clone()
    g = torch.linspace(-1, 1, 64, device=dev); Z, Y, X = torch.meshgrid(g, g, g, indexing="ij")
    q = torch.stack([(Z-0.45).abs()-0.13, (Y-0.1).abs()-0.95, (X-0.45).abs()-0.13], 0)
    box = torch.linalg.vector_norm(q.clamp(min=0), dim=0) + q.max(0).values.clamp(max=0)
    return torch.minimum(s, box).clamp(-TRUNC, TRUNC)[None, None]


def add_panel(fig, nrow, ncol, idx, title, sdf):
    ax = fig.add_subplot(nrow, ncol, idx, projection="3d")
    ax.set_title(title, fontsize=9); ax.set_axis_off()
    if sdf is None: return
    mm = mesh(sdf)
    if mm is not None:
        v, fc = np.asarray(mm.vertices), np.asarray(mm.faces)
        ax.plot_trisurf(v[:, 0], v[:, 2], fc, v[:, 1], color="#b9c4cf", edgecolor="none", shade=True)
        lim = [v.min(), v.max()]; ax.set_xlim(lim); ax.set_ylim(lim); ax.set_zlim(lim)
    ax.view_init(elev=18, azim=-60); ax.set_box_aspect((1, 1, 1))


def sync(dev):
    if dev == "cuda": torch.cuda.synchronize()


def get_item(ds, i, dev):
    it = ds[i]
    return {"sdf": it["sdf"].view(1,1,64,64,64).to(dev), "fp": it["fp"].view(1,1,64,64).to(dev),
            "class_id": it["class_id"].view(1).to(dev), "style_id": it["style_id"].view(1).to(dev),
            "height": it["height"].view(1).to(dev)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(CKPT_DIR / "stage3a_steps-30000.pth"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sample", type=int, default=5)
    ap.add_argument("--sampleB", type=int, default=20)
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--outdir", default="outputs/probe_realtime")
    args = ap.parse_args()
    dev = args.device
    outdir = REPO / args.outdir; outdir.mkdir(parents=True, exist_ok=True)

    ds = Bag3dDataset(); ds.initialize(SimpleNamespace(bag3d_h5="/dev/shm/bag3d_fast.h5", trunc_thres=TRUNC, augment=False), "train")
    m = load_model(args.ckpt, dev)

    # ---------------- PROBE 1: latency ----------------
    A = get_item(ds, args.sample, dev)
    edit = dict(A); edit["sdf"] = tower_edit(A["sdf"], dev)
    m.sdedit(edit, strength=args.strength, ddim_steps=8, uc_scale=1.0)  # warmup
    sync(dev)

    # fixed costs: encode + decode (done once per edit regardless of steps)
    x = edit["sdf"]
    sync(dev); t0 = time.perf_counter()
    z = m.vqvae(x, forward_no_quant=True, encode_only=True)
    sync(dev); t_enc = (time.perf_counter() - t0) * 1000
    sync(dev); t0 = time.perf_counter()
    _ = m.vqvae.decode_no_quant(z)
    sync(dev); t_dec = (time.perf_counter() - t0) * 1000

    print(f"\n=== PROBE 1: SDEdit latency (strength={args.strength}, uc=1.0, {dev}) ===")
    print(f"fixed: vqvae encode {t_enc:.1f} ms   decode {t_dec:.1f} ms")
    print(f"{'ddim_S':>7} {'steps':>6} {'ms (median of 3)':>18} {'occ':>7}")
    rows, panels = [], []
    for S in [50, 25, 16, 8, 4]:
        ts = []
        out = None
        for _ in range(3):
            torch.manual_seed(0)
            sync(dev); t0 = time.perf_counter()
            out = m.sdedit(edit, strength=args.strength, ddim_steps=S, uc_scale=1.0)
            sync(dev); ts.append((time.perf_counter() - t0) * 1000)
        ms = statistics.median(ts); steps = int(round(args.strength * S))
        occ = float((out <= 0).float().mean())
        print(f"{S:>7} {steps:>6} {ms:>18.1f} {occ:>7.3f}")
        rows.append((S, steps, ms, occ)); panels.append((f"S={S} ({steps} st)\n{ms:.0f} ms", out))

    fig = plt.figure(figsize=(3.2 * (len(panels)+1), 3.4))
    add_panel(fig, 1, len(panels)+1, 1, "edited (+tower)", edit["sdf"])
    for j, (t, sd) in enumerate(panels):
        add_panel(fig, 1, len(panels)+1, j+2, t, sd)
    fig.suptitle(f"PROBE 1 — SDEdit quality vs steps & latency (uc=1.0; autoguidance ~2x)", fontsize=12)
    fig.tight_layout(rect=(0,0,1,0.92)); fig.savefig(outdir/"probe1_latency.png", dpi=88)
    print(f"[saved] {outdir/'probe1_latency.png'}")

    # ---------------- PROBE 2: latent geometry ----------------
    B = get_item(ds, args.sampleB, dev)
    zA = m.vqvae(A["sdf"], forward_no_quant=True, encode_only=True)
    zB = m.vqvae(B["sdf"], forward_no_quant=True, encode_only=True)
    std = float(zA.std())
    print(f"\n=== PROBE 2: latent geometry (A={args.sample}, B={args.sampleB}, latent std={std:.3f}) ===")

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    sigmas = [0.0, 0.5, 1.0, 2.0]
    ncol = max(len(alphas), len(sigmas))
    fig = plt.figure(figsize=(3.2 * ncol, 6.8))
    for j, a in enumerate(alphas):
        zi = (1 - a) * zA + a * zB
        sdf = m.vqvae.decode_no_quant(zi)
        occ = float((sdf <= 0).float().mean())
        print(f"  interp a={a:.2f}  occ={occ:.3f}")
        add_panel(fig, 2, ncol, j+1, f"interp a={a:.2f}", sdf)
    for j, sg in enumerate(sigmas):
        torch.manual_seed(j)
        zi = zA + sg * std * torch.randn_like(zA)
        sdf = m.vqvae.decode_no_quant(zi)
        occ = float((sdf <= 0).float().mean())
        print(f"  perturb sigma={sg:.1f}  occ={occ:.3f}")
        add_panel(fig, 2, ncol, ncol + j+1, f"A + {sg:.1f}*std noise", sdf)
    fig.suptitle("PROBE 2 — latent interpolation (top: A->B) & perturbation (bottom: A+noise). Plausible => handles viable", fontsize=11)
    fig.tight_layout(rect=(0,0,1,0.94)); fig.savefig(outdir/"probe2_latent.png", dpi=88)
    print(f"[saved] {outdir/'probe2_latent.png'}")


if __name__ == "__main__":
    main()
