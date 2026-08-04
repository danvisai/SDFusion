"""Does SDEdit (the massing prior) turn a *hole* into a door, or a *box on top* into a chimney?

Honest reality-check: carves a door-shaped hole (SDF subtraction) and adds a roof box (SDF union)
on a real 3D BAG building, then runs sdedit (autoguided) at low->high strength. Shows whether the
prior PRESERVES the edit (low s), REFINES it into a recognizable element (the hope), or ERASES it
(high s). The massing prior has no "door"/"chimney" concept and trains on solid blocks with no
openings, so the expected result is: edits survive only as crude bumps/recesses and wash out with
strength — motivating layer ② (detail) + Track 2 (learned part mixing) for true element semantics.

GPU recommended: --device cuda.
"""
from __future__ import annotations
import argparse, sys
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


def box_sdf(cz, cy, cx, hz, hy, hx, dev):
    """Analytic box SDF on the 64^3 (D=z,H=y,W=x) grid in [-1,1]^3."""
    g = torch.linspace(-1, 1, 64, device=dev); Z, Y, X = torch.meshgrid(g, g, g, indexing="ij")
    q = torch.stack([(Z-cz).abs()-hz, (Y-cy).abs()-hy, (X-cx).abs()-hx], 0)
    return torch.linalg.vector_norm(q.clamp(min=0), dim=0) + q.max(0).values.clamp(max=0)


def occ_bounds(sdf):
    occ = (sdf[0, 0] <= 0)
    def c(i): return -1.0 + 2.0 * float(i) / 63.0
    zs = torch.where(occ.any(dim=(1, 2)))[0]; ys = torch.where(occ.any(dim=(0, 2)))[0]
    xs = torch.where(occ.any(dim=(0, 1)))[0]
    return dict(z=(c(zs.min()), c(zs.max())), y=(c(ys.min()), c(ys.max())), x=(c(xs.min()), c(xs.max())))


def add_panel(fig, nrow, ncol, r, c, title, sdf):
    ax = fig.add_subplot(nrow, ncol, r * ncol + c + 1, projection="3d")
    ax.set_title(title, fontsize=9); ax.set_axis_off()
    if sdf is None: return
    mm = mesh(sdf)
    if mm is not None:
        v, fc = np.asarray(mm.vertices), np.asarray(mm.faces)
        ax.plot_trisurf(v[:, 0], v[:, 2], fc, v[:, 1], color="#b9c4cf", edgecolor="none", shade=True)
        lim = [v.min(), v.max()]; ax.set_xlim(lim); ax.set_ylim(lim); ax.set_zlim(lim)
    ax.view_init(elev=18, azim=-60); ax.set_box_aspect((1, 1, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(CKPT_DIR / "stage3a_steps-30000.pth"))
    ap.add_argument("--guide_ckpt", default=str(CKPT_DIR / "stage3a_steps-10000.pth"))
    ap.add_argument("--auto_scale", type=float, default=2.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sample", type=int, default=5)
    ap.add_argument("--strengths", default="0.2,0.4,0.6")
    ap.add_argument("--out", default="outputs/sdedit_bag3d/sdedit_semantic_edits.png")
    args = ap.parse_args()
    dev = args.device

    ds = Bag3dDataset(); ds.initialize(SimpleNamespace(bag3d_h5="/dev/shm/bag3d_fast.h5", trunc_thres=TRUNC, augment=False), "train")
    it = ds[args.sample]
    sdf0 = it["sdf"].view(1, 1, 64, 64, 64).to(dev)
    base = {"sdf": sdf0, "fp": it["fp"].view(1, 1, 64, 64).to(dev),
            "class_id": it["class_id"].view(1).to(dev), "style_id": it["style_id"].view(1).to(dev),
            "height": it["height"].view(1).to(dev)}
    b = occ_bounds(sdf0)
    cz = (b["z"][0] + b["z"][1]) / 2; cx = (b["x"][0] + b["x"][1]) / 2
    print(f"[bounds] z={b['z']} y={b['y']} x={b['x']}  occ={float((sdf0<=0).float().mean()):.3f}")

    # Edit 1 — DOOR: subtract a tall narrow box into the +x facade at ground level.
    door = box_sdf(cz, b["y"][0] + 0.30, b["x"][1], 0.12, 0.30, 0.22, dev)
    sdf_door = torch.maximum(sdf0[0, 0], -door).clamp(-TRUNC, TRUNC)[None, None]
    # Edit 2 — CHIMNEY: union a small box straddling the roof at the centre.
    chim = box_sdf(cz, b["y"][1], cx, 0.10, 0.18, 0.10, dev)
    sdf_chim = torch.minimum(sdf0[0, 0], chim).clamp(-TRUNC, TRUNC)[None, None]

    main_m = load_model(args.ckpt, dev); guide_m = load_model(args.guide_ckpt, dev)
    strengths = [float(x) for x in args.strengths.split(",")]
    ncol = 2 + len(strengths); nrow = 2
    fig = plt.figure(figsize=(3.2 * ncol, 3.4 * nrow))

    for r, (name, edited) in enumerate([("DOOR (carve hole)", sdf_door), ("CHIMNEY (box on roof)", sdf_chim)]):
        edata = dict(base); edata["sdf"] = edited
        add_panel(fig, nrow, ncol, r, 0, f"{name}\nbefore", sdf0)
        add_panel(fig, nrow, ncol, r, 1, "edited", edited)
        print(f"[{name}] edited occ={float((edited<=0).float().mean()):.3f}")
        for j, s in enumerate(strengths):
            torch.manual_seed(0)
            out = main_m.sdedit(edata, strength=s, ddim_steps=50, guide_model=guide_m, auto_scale=args.auto_scale)
            print(f"    s={s:.2f}  occ={float((out<=0).float().mean()):.3f}")
            add_panel(fig, nrow, ncol, r, 2 + j, f"sdedit s={s:.1f}", out)

    out_p = REPO / args.out; out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("Does the massing prior turn a HOLE->door / BOX->chimney? (autoguided SDEdit)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(out_p, dpi=88)
    print(f"[saved] {out_p}")


if __name__ == "__main__":
    main()
