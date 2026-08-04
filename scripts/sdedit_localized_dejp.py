"""Sharper localized (inpainting) before/after: PRONOUNCED tower edit on DE + JP inputs.

Tests the two open questions: (1) does the broadened prior turn a clearly-protruding placed mass
into a COHERENT element (vs blob / vs flatten), and (2) does it generalize to German/Japanese
geometry the deployed prior never saw. Body held bit-exact via the locality mask; only the edit
region is regenerated (refine.py:snap_volume local=True).

  env -u LD_PRELOAD -u LD_LIBRARY_PATH PYTHONPATH=. ./sdfusion/bin/python scripts/sdedit_localized_dejp.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import h5py, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "server"))
from sdedit_bag3d_test import load_model, mesh, TRUNC
from refine import _edit_locality_mask
from scene.sdf_edit import EditOp, _primitive

REAL = REPO / "data/real_massing_v1/real.h5"
# pronounced corner tower: thin, tall, rises clearly above the body
TOWER = {"kind": "box", "center": [0.5, 0.35, 0.5], "size": [0.12, 0.7, 0.12], "mode": "add"}
DEP = REPO / "logs_building/2026-06-08T11-50-42-stage3a-hybrid-clean/ckpt/stage3a_steps-latest.pth"
FT  = REPO / "logs_building/continue-stage3a-xcultural-warmstart-ft-final/ckpt/stage3a_steps-latest.pth"
STRENGTHS = [0.3, 0.5, 0.7]
INNER, BAND = 0.2, 0.5
# (label, source_id, global row in real.h5)  NL 0-11775 | DE 11776-23775 | JP 23776-35775
BUILDINGS = [("DE (NRW)", 12500), ("JP (PLATEAU)", 25000)]


def cube_pts(dev):
    g = torch.linspace(-1, 1, 64, device=dev)
    Z, Y, X = torch.meshgrid(g, g, g, indexing="ij")
    return torch.stack([X, Y, Z], -1).reshape(-1, 3)


def load_building(row, dev):
    with h5py.File(REAL, "r") as f:
        sdf = torch.from_numpy(f["sdf"][row].astype(np.float32))      # (D,H,W)
        fp = torch.from_numpy(f["footprint"][row].astype(np.uint8)).float()
        sid = int(f["source_id"][row])
    occ_y = (sdf.numpy() <= 0).any(axis=(0, 2))
    ys = np.where(occ_y)[0]
    h_n = float((ys.max() - ys.min() + 1) * (2.0 / 63.0)) if ys.size else 0.0
    return sdf.view(1,1,64,64,64).to(dev), fp.view(1,1,64,64).to(dev), h_n, sid


def panel(fig, nr, nc, r, c, title, sdf):
    ax = fig.add_subplot(nr, nc, r * nc + c + 1, projection="3d")
    ax.set_title(title, fontsize=8); ax.set_axis_off(); ax.view_init(elev=18, azim=-55)
    ax.set_box_aspect((1,1,1))
    if sdf is None: return
    mm = mesh(sdf if sdf.dim()==5 else sdf[None,None])
    if mm is None: return
    v, fc = np.asarray(mm.vertices), np.asarray(mm.faces)
    ax.plot_trisurf(v[:,0], v[:,2], fc, v[:,1], color="#c9b790", edgecolor="none", shade=True)
    lim=[v.min(), v.max()]; ax.set_xlim(lim); ax.set_ylim(lim); ax.set_zlim(lim)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pts = cube_pts(dev)
    box = _primitive(EditOp.from_dict(TOWER))(pts).reshape(64,64,64)
    w = _edit_locality_mask([TOWER], pts, INNER, BAND).reshape(64,64,64).to(dev)
    print(f"[dejp AB] mask>0.5 frac={float((w>0.5).float().mean()):.3f}")
    models = {"BEFORE deployed 20k": load_model(str(DEP), dev),
              "AFTER x-cultural 6k": load_model(str(FT), dev)}

    for blabel, row in BUILDINGS:
        sdf0, fp, h_n, sid = load_building(row, dev)
        edited = torch.minimum(sdf0[0,0], box).clamp(-TRUNC, TRUNC)[None,None]
        base = {"fp": fp, "class_id": torch.zeros(1,dtype=torch.long,device=dev),
                "style_id": torch.full((1,),8,dtype=torch.long,device=dev),
                "height": torch.tensor([h_n],dtype=torch.float32,device=dev)}
        edit_data = dict(base); edit_data["sdf"] = edited
        print(f"\n=== {blabel} row{row} src{sid} occ={float((sdf0<=0).float().mean()):.3f} h_n={h_n:.2f} ===")
        nc = 2 + len(STRENGTHS); nr = 2
        fig = plt.figure(figsize=(3.0*nc, 6.4))
        for r, (label, m) in enumerate(models.items()):
            panel(fig, nr, nc, r, 0, f"{label}\n{blabel} body", sdf0)
            panel(fig, nr, nc, r, 1, "edited (+tower)", edited)
            for j, s in enumerate(STRENGTHS):
                torch.manual_seed(0)
                out = m.sdedit(edit_data, strength=s, ddim_steps=50, uc_scale=1.0)
                local = edited[0,0]*(1-w) + out[0,0]*w
                print(f"  {label}  s={s:.1f}  occ={float((local<=0).float().mean()):.3f}")
                panel(fig, nr, nc, r, 2+j, f"localized snap s={s:.1f}", local[None,None])
        tag = blabel.split()[0].lower()
        out_p = REPO / f"outputs/sdedit_xcultural/localized_{tag}.png"
        fig.suptitle(f"LOCALIZED snap on {blabel} input (pronounced tower) — body bit-exact\n"
                     "row0: deployed 20k prior   row1: cross-cultural finetuned 6k prior", fontsize=11)
        fig.tight_layout(rect=(0,0,1,0.93)); fig.savefig(out_p, dpi=88)
        print(f"[saved] {out_p}")


if __name__ == "__main__":
    main()
