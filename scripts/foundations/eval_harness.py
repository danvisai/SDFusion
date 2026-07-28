"""Prior eval harness — training-gaps plan §0. Makes model change MEASURABLE.

Fixed, deterministic eval set (4 real 3D-BAG buildings + 1 recipe building + 3 canonical
sculpt edits) evaluated the way the model is actually USED (SDEdit at partial noise, the
deployed inference path — NOT the train loss, which is decoupled from generation quality).

Per checkpoint:
  - faithfulness curve:  iou(out, in) at strength {0.3, 0.5, 0.7}
  - footprint IoU        (top-down silhouette kept?) at strength 0.5
  - roughness            mean |3D Laplacian| in the surface band — the wavy-wobble metric
  - style divergence     same input, styles 0..7 -> 1 - mean pairwise occupancy IoU
                         (0 = conditioning is dead, the audit's gap #1)
  - montage PNG          inputs vs outputs at each strength

Appends one row to outputs/eval_prior/metrics.csv -> compare checkpoints over time.

Run (hybrid prior, node-local ckpts):
  env -u LD_PRELOAD -u LD_LIBRARY_PATH PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
    ./sdfusion/bin/python scripts/foundations/eval_harness.py \
      --ckpt /tmp/hybrid_ckpts/stage3a_steps-latest.pth \
      --guide /tmp/hybrid_ckpts/stage3a_steps-5000.pth --label hybrid20k
Old prior baseline: add --extra_cond off.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import os
import sys

import h5py
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "server"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from types import SimpleNamespace

from refine import Refiner, volume_to_sdf                      # noqa: E402
from scene.sdf_edit import EditableBuilding, EditOp            # noqa: E402
from scene.sdf_primitives import grid_to_mesh                  # noqa: E402

SMOKE_H5 = os.path.join(REPO, "data/bag3d_v1/bag3d_smoke.h5")
OUT = os.path.join(REPO, "outputs", "eval_prior")
REAL_IDX = [3, 50, 80, 140]            # fixed, spread over the smoke corpus
STRENGTHS = [0.3, 0.5, 0.7]
EDITS = [  # canonical sculpt edits (cube coords), applied to the recipe building
    ("tower", dict(kind="box", center=[0.45, 0.25, 0.0], size=[0.14, 0.5, 0.14], mode="add", smooth=0.0)),
    ("dome", dict(kind="sphere", center=[0.0, 0.55, 0.0], size=[0.3], mode="add", smooth=0.3)),
    ("carve", dict(kind="box", center=[0.0, -0.2, 0.62], size=[0.18, 0.4, 0.2], mode="subtract", smooth=0.0)),
]


def frame_n_input(occ, device, trunc=0.2):
    """occupancy (D,H,W) -> the prior's input contract: EDT-true truncated SDF + fp + height_n."""
    from scipy.ndimage import distance_transform_edt
    R = occ.shape[0]
    vox = 2.0 / (R - 1)
    sdf = np.clip((distance_transform_edt(~occ) - distance_transform_edt(occ)) * vox,
                  -trunc, trunc).astype(np.float32)
    ys = np.where(occ.any((0, 2)))[0]
    height_n = float((ys.max() - ys.min() + 1) * vox) if len(ys) else 1.0
    return (torch.from_numpy(sdf).view(1, 1, R, R, R).to(device),
            torch.from_numpy(occ.any(1).astype(np.float32)).view(1, 1, R, R).to(device),
            height_n)


def sdedit(model, guide, sdf_t, fp_t, height_n, strength, device, style=8, auto_scale=2.0):
    data = {"sdf": sdf_t, "fp": fp_t,
            "class_id": torch.zeros(1, dtype=torch.long, device=device),
            "style_id": torch.full((1,), int(style), dtype=torch.long, device=device),
            "height": torch.tensor([height_n], dtype=torch.float32, device=device)}
    out = model.sdedit(data, strength=strength, ddim_steps=8, uc_scale=1.0,
                       guide_model=guide, auto_scale=auto_scale)
    return out[0, 0].detach().cpu().numpy().astype(np.float32)


def iou(a_occ, b_occ):
    u = (a_occ | b_occ).sum()
    return float((a_occ & b_occ).sum() / u) if u else 0.0


def fp_iou(a_occ, b_occ):
    return iou(a_occ.any(1), b_occ.any(1))


def roughness(sdf):
    """Mean |3D Laplacian| in the surface band — high = wavy/lumpy surface."""
    lap = (np.roll(sdf, 1, 0) + np.roll(sdf, -1, 0) + np.roll(sdf, 1, 1) + np.roll(sdf, -1, 1)
           + np.roll(sdf, 1, 2) + np.roll(sdf, -1, 2) - 6 * sdf)
    band = np.abs(sdf) < 0.06
    return float(np.abs(lap[band]).mean()) if band.any() else 0.0


def draw(ax, sdf, title):
    m = grid_to_mesh(torch.from_numpy(np.ascontiguousarray(sdf)), (-1, -1, -1, 1, 1, 1), 0.0)
    ax.set_title(title, fontsize=6)
    if m is not None and len(m.faces):
        pc = Poly3DCollection(m.vertices[m.faces], alpha=1.0)
        pc.set_facecolor((0.80, 0.78, 0.72)); pc.set_edgecolor("none")
        ax.add_collection3d(pc)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(elev=16, azim=-60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--guide", default=None, help="weaker ckpt for autoguidance (optional)")
    ap.add_argument("--label", default=None)
    ap.add_argument("--extra_cond", choices=["on", "off"], default="on")
    ap.add_argument("--adaln", choices=["on", "off"], default="off")
    args = ap.parse_args()
    label = args.label or os.path.basename(os.path.dirname(os.path.dirname(args.ckpt)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT, exist_ok=True)

    r = Refiner(SimpleNamespace(device=device))
    xc = args.extra_cond == "on"
    al = args.adaln == "on"
    model = r._mk_stage3a(args.ckpt, use_extra_cond=xc, use_adaln=al)
    guide = r._mk_stage3a(args.guide, use_extra_cond=xc, use_adaln=al) if args.guide else None

    # ---- eval inputs (all as Frame-N occupancy) -------------------------------------
    cases = []
    with h5py.File(SMOKE_H5, "r") as h:
        for i in REAL_IDX:
            cases.append((f"bag{i}", h["sdf"][i] <= 0))
    from models.networks.diff_recipe import build_diff_recipe
    _, default_fn, _ = build_diff_recipe("modern")
    params = default_fn(device).detach().cpu().numpy()
    poly = np.array([[-7, -9], [7, -9], [7, 9], [-7, 9]], np.float32)
    rec_grid, _, _, _ = r.building_volume(poly, "modern", params, 16.0, margin=1.05)
    cases.append(("recipe", rec_grid <= 0))
    base_sdf_fn = volume_to_sdf(rec_grid, device)
    g1 = torch.linspace(-1, 1, 64, device=device)
    Z, Y, X = torch.meshgrid(g1, g1, g1, indexing="ij")
    pts = torch.stack([X, Y, Z], -1).reshape(-1, 3)
    for name, e in EDITS:
        comp = EditableBuilding(base_sdf_fn, [EditOp.from_dict(e)]).composed()
        with torch.no_grad():
            cases.append((f"edit_{name}", (comp(pts).reshape(64, 64, 64) <= 0).cpu().numpy()))

    # ---- faithfulness curve + roughness + fp ----------------------------------------
    n_rows, n_cols = len(cases), 1 + len(STRENGTHS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.6 * n_rows),
                             subplot_kw={"projection": "3d"})
    curve = {s: [] for s in STRENGTHS}
    fps, roughs = [], []
    for ri, (name, occ_in) in enumerate(cases):
        sdf_t, fp_t, hn = frame_n_input(occ_in, device)
        draw(axes[ri, 0], sdf_t[0, 0].cpu().numpy(), f"{name}\ninput")
        for ci, s in enumerate(STRENGTHS):
            out = sdedit(model, guide, sdf_t, fp_t, hn, s, device)
            occ_o = out <= 0
            curve[s].append(iou(occ_o, occ_in))
            if s == 0.5:
                fps.append(fp_iou(occ_o, occ_in))
                roughs.append(roughness(out))
            draw(axes[ri, 1 + ci], out, f"s={s} iou={curve[s][-1]:.2f}")
        print(f"{name:12s} " + " ".join(f"s{s}={curve[s][-1]:.3f}" for s in STRENGTHS))

    # ---- style-divergence probe ------------------------------------------------------
    sdf_t, fp_t, hn = frame_n_input(rec_grid <= 0, device)
    style_occ = []
    for st in range(8):
        torch.manual_seed(0)
        style_occ.append(sdedit(model, guide, sdf_t, fp_t, hn, 0.6, device, style=st) <= 0)
    pair = [iou(style_occ[a], style_occ[b]) for a in range(8) for b in range(a + 1, 8)]
    style_div = 1.0 - float(np.mean(pair))

    stamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    png = os.path.join(OUT, f"{label}_{stamp}.png")
    plt.tight_layout(); plt.savefig(png, dpi=100); plt.close()

    row = {"utc": stamp, "label": label, "ckpt": args.ckpt,
           **{f"iou_s{str(s).replace('.', '')}": round(float(np.mean(curve[s])), 4) for s in STRENGTHS},
           "fp_iou_s05": round(float(np.mean(fps)), 4),
           "roughness_s05": round(float(np.mean(roughs)), 5),
           "style_divergence": round(style_div, 4)}
    csvp = os.path.join(OUT, "metrics.csv")
    new = not os.path.exists(csvp)
    with open(csvp, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)
    print("\n== summary ==")
    for k, v in row.items():
        print(f"  {k}: {v}")
    print("montage:", png)


if __name__ == "__main__":
    main()
