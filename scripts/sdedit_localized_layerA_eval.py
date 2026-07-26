"""LAYER-A/AB CONTEXT-SNAP EVAL (the pending eval from 2026-06-30, run 2026-07-08 for the
report record before parking the line — superseded in-product by coherent-add).

Question: does conditioning the snap on the building's OWN structure (known_body +
edit_mask + primitive channels, Layer A; + element-type id, Layer B) make the added mass
more coherent than the deployed cross-cultural prior under the PRODUCTION-FAITHFUL
localized snap (body bit-exact, only the edit region regenerated)?

Rows: deployed xcultural 6k (null ctx — production) | layerA ckpt (real ctx) |
layerAB ckpt (real ctx + elem type). Same NL body, same tower edit, s in {0.3,0.5,0.7}.
Metrics per cell: localized occupancy + tower-region solidity (occupied fraction of the
placed box) + IoU of the snapped edit region vs the placed box.

Run: env -u LD_PRELOAD -u LD_LIBRARY_PATH PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
       ./sdfusion/bin/python scripts/sdedit_localized_layerA_eval.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "server"))

from datasets.bag3d_dataset import Bag3dDataset
from models.stage3a_model import Stage3aModel
from refine import _edit_locality_mask
from scene.sdf_edit import EditOp, _primitive
from sdedit_bag3d_test import TRUNC, mesh

TOWER = {"kind": "box", "center": [0.45, 0.25, 0.0], "size": [0.14, 0.5, 0.14], "mode": "add"}
STRENGTHS = [0.3, 0.5, 0.7]
INNER, BAND = 0.2, 0.5
T = 0.2
XC = REPO / "logs_building/continue-stage3a-xcultural-warmstart-ft-final/ckpt/stage3a_steps-latest.pth"
LA = REPO / "logs_building/continue-stage3a-layerA-context/ckpt/stage3a_steps-latest.pth"
LAB = REPO / "logs_building/continue-stage3a-layerAB-context-elemtype/ckpt/stage3a_steps-latest.pth"


def load_model(ckpt, dev, use_context=False, use_element_type=False):
    opt = SimpleNamespace(
        isTrain=False, device=dev, df_cfg=str(REPO / "configs/stage3a_sdf_diffusion.yaml"),
        vq_cfg=str(REPO / "configs/vqvae_bnet.yaml"),
        vq_ckpt=str(REPO / "logs_building/2025-05-19T19-58-28-vqvae-building-all-res64-LR1e-4-T0.2-release/ckpt/vqvae_steps-latest.pth"),
        ckpt=str(ckpt), ddim_steps=50, debug="0", gpu_ids=[0] if dev == "cuda" else [],
        ckpt_dir="/tmp", latent_size_HW=(16, 16), latent_size_D=16,
        use_context=use_context, use_element_type=use_element_type)
    print(f"[load] {Path(ckpt).parent.parent.name} (use_context={use_context}, "
          f"elem_type={use_element_type})")
    m = Stage3aModel()
    m.initialize(opt)
    return m


def cube_pts(dev):
    g = torch.linspace(-1, 1, 64, device=dev)
    Z, Y, X = torch.meshgrid(g, g, g, indexing="ij")
    return torch.stack([X, Y, Z], -1).reshape(-1, 3)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = Bag3dDataset()
    ds.initialize(SimpleNamespace(bag3d_h5=str(REPO / "data/real_massing_v1/real.h5"),
                                  trunc_thres=TRUNC, augment=False), "train")
    it = ds[40]                       # ds[5] in real.h5 is a degenerate near-full slab
    sdf0 = it["sdf"].view(1, 1, 64, 64, 64).to(dev)
    pts = cube_pts(dev)
    box_sdf = _primitive(EditOp.from_dict(TOWER))(pts).reshape(64, 64, 64)
    box_occ = (box_sdf <= 0)
    edited = torch.minimum(sdf0[0, 0], box_sdf).clamp(-TRUNC, TRUNC)[None, None]
    w = _edit_locality_mask([TOWER], pts, INNER, BAND).reshape(64, 64, 64).to(dev)

    # REAL Layer-A context from the actual edit (mirrors _build_context's semantics):
    # region = the placed primitive dilated a bit; known_body = body with region emptied;
    # primitive = the crude placed mass itself. avg-pool 64 -> 16 (latent res).
    region = (box_sdf <= 0.15).float()[None, None]
    known = torch.where(region > 0.5, torch.full_like(sdf0, T), sdf0)
    prim = torch.where(box_occ[None, None], torch.full_like(sdf0, -T),
                       torch.full_like(sdf0, T))
    ds4 = lambda v: F.avg_pool3d(v, kernel_size=4, stride=4)
    ctx = [ds4(known), ds4(region), ds4(prim)]

    # Layer-B element type id for the placed box (same classifier the training used)
    idx = torch.nonzero(box_occ, as_tuple=False)
    lo, hi = idx.min(0).values.tolist(), (idx.max(0).values + 1).tolist()
    hz, hy, hw = (hi[0] - lo[0]) / 2.0, (hi[1] - lo[1]) / 2.0, (hi[2] - lo[2]) / 2.0
    y_center = (lo[1] + hi[1]) / 2.0

    data = {"fp": it["fp"].view(1, 1, 64, 64).to(dev), "class_id": it["class_id"].view(1).to(dev),
            "style_id": it["style_id"].view(1).to(dev), "height": it["height"].view(1).to(dev)}
    edit_data = dict(data)
    edit_data["sdf"] = edited

    rows = [("PROD: xcultural 6k (no ctx)", XC, False, None)]
    if LA.exists():
        rows.append(("Layer-A ctx", LA, True, None))
    if LAB.exists():
        rows.append(("Layer-AB ctx+type", LAB, True, "auto"))

    ncol = 2 + len(STRENGTHS)
    fig = plt.figure(figsize=(3.0 * ncol, 3.2 * len(rows)))

    def panel(r, c, title, sdf):
        ax = fig.add_subplot(len(rows), ncol, r * ncol + c + 1, projection="3d")
        ax.set_title(title, fontsize=8)
        ax.set_axis_off()
        ax.view_init(elev=18, azim=-55)
        ax.set_box_aspect((1, 1, 1))
        if sdf is None:
            return
        mm = mesh(sdf if sdf.dim() == 5 else sdf[None, None])
        if mm is None:
            return
        v, fc = np.asarray(mm.vertices), np.asarray(mm.faces)
        ax.plot_trisurf(v[:, 0], v[:, 2], fc, v[:, 1], color="#c9b790", edgecolor="none",
                        shade=True)
        lim = [v.min(), v.max()]
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_zlim(lim)

    results = []
    for r, (label, ckpt, use_ctx, elem) in enumerate(rows):
        m = load_model(str(ckpt), dev, use_context=use_ctx,
                       use_element_type=(elem == "auto"))
        elem_id = None
        if elem == "auto":
            elem_id = torch.tensor([m._classify_element_type(hz, hy, hw, y_center, 64.0)],
                                   device=dev)
            print(f"  elem_type_id={int(elem_id)}")
        panel(r, 0, f"{label}\nbody (real)", sdf0)
        panel(r, 1, "edited (+tower)", edited)
        for j, s in enumerate(STRENGTHS):
            torch.manual_seed(0)
            kw = dict(strength=s, ddim_steps=50, uc_scale=1.0)
            if use_ctx:
                kw["ctx_channels"] = ctx
            if elem_id is not None:
                kw["elem_type_id"] = elem_id
            out = m.sdedit(edit_data, **kw)
            local = edited[0, 0] * (1 - w) + out[0, 0] * w
            occ = float((local <= 0).float().mean())
            solid = float((local <= 0)[box_occ].float().mean())
            snap_occ = (local <= 0) & (w > 0.5)
            uni = (snap_occ | (box_occ & (w > 0.5))).sum()
            iou = float((snap_occ & box_occ).sum() / uni) if uni else 0.0
            results.append((label, s, occ, solid, iou))
            print(f"  {label}  s={s:.1f}  occ={occ:.3f}  tower solidity={solid:.2f}  iou={iou:.2f}")
            panel(r, 2 + j, f"s={s:.1f} solid={solid:.2f}\niou={iou:.2f}", local[None, None])
        del m
        torch.cuda.empty_cache()

    out_p = REPO / "outputs/layerA_eval/localized_layerA_eval.png"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("Localized snap: production xcultural prior (null ctx) vs Layer-A/AB "
                 "context conditioning (real edit ctx)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_p, dpi=90)
    print(f"[saved] {out_p}")
    with open(out_p.with_suffix(".csv"), "w") as f:
        f.write("model,strength,occ,tower_solidity,edit_iou\n")
        for row in results:
            f.write(",".join(str(x) for x in row) + "\n")


if __name__ == "__main__":
    main()
