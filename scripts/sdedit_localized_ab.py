"""LOCALIZED (inpainting) SDEdit before/after — deployed prior vs cross-cultural finetuned prior.

Production-faithful: applies the tower edit, runs sdedit, then blends with the edit-locality mask
(out = edited*(1-w) + snapped*w) so the building BODY stays bit-exact and ONLY the edit region is
regenerated — exactly refine.py:snap_volume(local=True). Judges the EDIT-REGION snap, not the body.

  env -u LD_PRELOAD -u LD_LIBRARY_PATH PYTHONPATH=. ./sdfusion/bin/python scripts/sdedit_localized_ab.py
"""
from __future__ import annotations
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "server"))
from datasets.bag3d_dataset import Bag3dDataset
from sdedit_bag3d_test import load_model, mesh, BBOX, TRUNC
from refine import _edit_locality_mask
from scene.sdf_edit import EditOp, _primitive

TOWER = {"kind": "box", "center": [0.45, 0.25, 0.0], "size": [0.14, 0.5, 0.14], "mode": "add"}
DEP = REPO / "logs_building/2026-06-08T11-50-42-stage3a-hybrid-clean/ckpt"
FT  = REPO / "logs_building/continue-stage3a-xcultural-warmstart-ft-final/ckpt"
PRIORS = [("BEFORE: deployed 20k", DEP / "stage3a_steps-latest.pth"),
          ("AFTER: x-cultural 6k", FT / "stage3a_steps-latest.pth")]
STRENGTHS = [0.3, 0.5, 0.7]
INNER, BAND = 0.2, 0.5            # cube units: snap region = tower + ~0.2 dilation, fade over 0.5


def cube_pts(dev):
    g = torch.linspace(-1, 1, 64, device=dev)
    Z, Y, X = torch.meshgrid(g, g, g, indexing="ij")        # (D,H,W)
    return torch.stack([X, Y, Z], -1).reshape(-1, 3)         # (x,y,z), flat in (D,H,W) order


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = Bag3dDataset(); ds.initialize(SimpleNamespace(bag3d_h5=str(REPO/"data/real_massing_v1/real.h5"),
                                                       trunc_thres=TRUNC, augment=False), "train")
    it = ds[5]
    sdf0 = it["sdf"].view(1, 1, 64, 64, 64).to(dev)
    pts = cube_pts(dev)
    box = _primitive(EditOp.from_dict(TOWER))(pts).reshape(64, 64, 64)        # (D,H,W) box SDF
    edited = torch.minimum(sdf0[0, 0], box).clamp(-TRUNC, TRUNC)[None, None]  # union = "add"
    w = _edit_locality_mask([TOWER], pts, INNER, BAND).reshape(64, 64, 64).to(dev)
    print(f"[localized AB] sample 5  body occ={float((sdf0<=0).float().mean()):.3f}  "
          f"mask>0.5 frac={float((w>0.5).float().mean()):.3f}")

    data = {"fp": it["fp"].view(1,1,64,64).to(dev), "class_id": it["class_id"].view(1).to(dev),
            "style_id": it["style_id"].view(1).to(dev), "height": it["height"].view(1).to(dev)}
    edit_data = dict(data); edit_data["sdf"] = edited

    ncol = 2 + len(STRENGTHS)
    fig = plt.figure(figsize=(3.0 * ncol, 6.4))
    def panel(r, c, title, sdf):
        ax = fig.add_subplot(2, ncol, r * ncol + c + 1, projection="3d")
        ax.set_title(title, fontsize=9); ax.set_axis_off(); ax.view_init(elev=18, azim=-55)
        ax.set_box_aspect((1,1,1))
        if sdf is None: return
        mm = mesh(sdf if sdf.dim()==5 else sdf[None,None])
        if mm is None: return
        v, fc = np.asarray(mm.vertices), np.asarray(mm.faces)
        ax.plot_trisurf(v[:,0], v[:,2], fc, v[:,1], color="#c9b790", edgecolor="none", shade=True)
        lim=[v.min(), v.max()]; ax.set_xlim(lim); ax.set_ylim(lim); ax.set_zlim(lim)

    for r, (label, ckpt) in enumerate(PRIORS):
        m = load_model(str(ckpt), dev)
        panel(r, 0, f"{label}\nbody (real)", sdf0)
        panel(r, 1, "edited (+tower)", edited)
        for j, s in enumerate(STRENGTHS):
            torch.manual_seed(0)
            out = m.sdedit(edit_data, strength=s, ddim_steps=50, uc_scale=1.0)   # production: uc=1
            local = edited[0,0] * (1 - w) + out[0,0] * w                          # INPAINTING blend
            occ = float((local <= 0).float().mean())
            print(f"  {label}  s={s:.1f}  localized occ={occ:.3f}")
            panel(r, 2 + j, f"localized snap s={s:.1f}", local[None,None])

    out_p = REPO / "outputs/sdedit_xcultural/localized_ab.png"; out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("LOCALIZED (inpainting) snap — body bit-exact, only tower region regenerated\n"
                 "row0: deployed 20k prior   row1: cross-cultural finetuned 6k prior", fontsize=12)
    fig.tight_layout(rect=(0,0,1,0.93)); fig.savefig(out_p, dpi=88)
    print(f"[saved] {out_p}")


if __name__ == "__main__":
    main()
