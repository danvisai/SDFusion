"""Headless sculpt regression — makes the sculpt loop trackable WITHOUT the browser.

Runs canonical EditOp sessions against a RUNNING inference server (reuses its warm SDEdit
prior, so this is fast): /building_sdf -> compose edits locally -> /snap_sdf. Meshes
base / edited / snapped, renders a montage to outputs/sculpt_regression/<UTC>.png, and
appends metrics.csv (iou_to_edit, n_verts, snap_ms) so results are diffable across runs.

Run (server already up on :8099):
  env -u LD_PRELOAD -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="" \
    ./sdfusion/bin/python scripts/server/sculpt_regression.py [--url http://127.0.0.1:8099]
"""
from __future__ import annotations
import argparse, base64, csv, datetime, json, os, sys, time, urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch
from refine import volume_to_sdf
from scene.sdf_edit import EditableBuilding, EditOp
from scene.sdf_primitives import grid_to_mesh

OUT = os.path.join(REPO, "outputs", "sculpt_regression")

# (name, footprint [x0,z0,x1,z1] meters, height, style, class, edits in CUBE coords)
CASES = [
    ("res_tower", [-7, -9, 7, 9], 16, "modern", "RESIDENTIAL",
     [dict(kind="box", center=[0.45, 0.25, 0.0], size=[0.14, 0.5, 0.14], mode="add", smooth=0.0)]),
    ("civic_wing_dome", [-10, -7, 10, 7], 14, "public_civic", "PUBLIC",
     [dict(kind="box", center=[-0.6, 0.0, 0.0], size=[0.25, 0.35, 0.4], mode="add", smooth=0.1),
      dict(kind="sphere", center=[0.0, 0.5, 0.0], size=[0.30], mode="add", smooth=0.3)]),
    ("res_carve", [-8, -8, 8, 8], 18, "victorian", "RESIDENTIAL",
     [dict(kind="box", center=[0.0, -0.2, 0.62], size=[0.18, 0.4, 0.2], mode="subtract", smooth=0.0)]),
]


def post(url, path, body):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())


def vol_from_b64(b64, res):
    return np.frombuffer(base64.b64decode(b64), dtype="<f4").reshape(res, res, res).copy()


def mesh_of(grid):
    return grid_to_mesh(torch.from_numpy(np.asarray(grid, np.float32)),
                        (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0), iso=0.0)


def draw(ax, mesh, title):
    ax.set_title(title, fontsize=8)
    if mesh is not None and len(mesh.faces):
        pc = Poly3DCollection(mesh.vertices[mesh.faces], alpha=1.0)
        pc.set_facecolor((0.80, 0.78, 0.72)); pc.set_edgecolor("none")
        ax.add_collection3d(pc)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(elev=16, azim=-60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8099")
    ap.add_argument("--strength", type=float, default=0.5)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    fig, axes = plt.subplots(len(CASES), 3, figsize=(9, 3 * len(CASES)),
                             subplot_kw={"projection": "3d"})
    if len(CASES) == 1:
        axes = axes[None, :]
    rows = []
    for i, (name, fp, h, style, cls, edits) in enumerate(CASES):
        poly = [[fp[0], fp[1]], [fp[2], fp[1]], [fp[2], fp[3]], [fp[0], fp[3]]]
        b = post(args.url, "/building_sdf",
                 {"footprint": poly, "style": style, "building_class": cls, "height": h, "seed": 0})
        base_grid = vol_from_b64(b["sdf_b64"], b["res"])
        # edited shape (base + edits), composed locally for the montage's middle column
        base = volume_to_sdf(base_grid, "cpu")
        edited = EditableBuilding(base, [EditOp.from_dict(d) for d in edits]).composed()
        g1 = torch.linspace(-1, 1, b["res"])
        Z, Y, X = torch.meshgrid(g1, g1, g1, indexing="ij")
        eg = edited(torch.stack([X, Y, Z], -1).reshape(-1, 3)).reshape(b["res"], b["res"], b["res"]).numpy()

        t = time.time()
        s = post(args.url, "/snap_sdf",
                 {"base_sdf_b64": b["sdf_b64"], "res": b["res"], "edits": edits, "strength": args.strength})
        ms = int((time.time() - t) * 1000)
        snapped = vol_from_b64(s["sdf_b64"], s["res"])
        msn = mesh_of(snapped); nv = 0 if msn is None else len(msn.vertices)

        draw(axes[i, 0], mesh_of(base_grid), f"{name}\nbase")
        draw(axes[i, 1], mesh_of(eg), "edited")
        draw(axes[i, 2], msn, f"snapped  iou={s['iou_to_edit']:.2f}")
        rows.append([name, round(float(s["iou_to_edit"]), 3), nv, ms])
        print(f"{name:16s} iou={s['iou_to_edit']:.3f} nverts={nv} snap_ms={ms}")

    stamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    png = os.path.join(OUT, f"{stamp}.png")
    plt.tight_layout(); plt.savefig(png, dpi=110); plt.close()
    csvp = os.path.join(OUT, "metrics.csv"); new = not os.path.exists(csvp)
    with open(csvp, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["utc", "case", "iou_to_edit", "n_verts", "snap_ms"])
        for r in rows:
            w.writerow([stamp] + r)
    print("wrote", png)


if __name__ == "__main__":
    main()
