"""GENERATIVE smart-add photo: the SAME placed box, different seeds/classes ->
different sampled architecture (the proof it's not a fixed procedural mapping).

Row 1: same tall corner box on a RELIGIOUS building, seeds 0..3 (expect dome/spire
variation at the real 38% dome rate, varying window rhythm/faces, spire heights).
Row 2: same box on the roof, seeds 0..3 (gable/hip dormers, proportions vary).
Run (server warm): env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python scripts/server/ab_generative_add.py
"""
import base64
import datetime
import json
import os
import sys
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

URL = os.environ.get("SCULPT_URL", "http://127.0.0.1:8099")
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
RECT = [[-7, -9], [7, -9], [7, 9], [-7, 9]]
SEEDS = [0, 1, 2, 3]


def post(path, body):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=900).read())


def draw(ax, g, title):
    from skimage import measure
    ax.set_title(title, fontsize=8)
    if (g <= 0).sum() > 8:
        v, fc, _, _ = measure.marching_cubes(g.astype(np.float32), level=0.0)
        v = v[:, [2, 1, 0]]
        ax.plot_trisurf(v[:, 0], v[:, 2], fc, v[:, 1], color="#cdb892",
                        edgecolor="none", antialiased=True, shade=True)
        lo, hi = v.min(), v.max()
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_zlim(lo, hi)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(elev=16, azim=-55)


def main():
    import torch
    from refine import volume_to_sdf
    from scene.sdf_edit import EditableBuilding, EditOp
    from scene.sdf_primitives import sample_grid

    plain = post("/building_sdf", {"footprint": RECT, "style": "modern",
                                   "building_class": "RELIGIOUS", "height": 16})
    grid = np.frombuffer(base64.b64decode(plain["sdf_b64"]),
                         dtype="<f4").reshape(64, 64, 64).copy()
    base = volume_to_sdf(grid, "cpu")
    occ = grid <= 0
    g = np.linspace(-1, 1, 64)
    to_i = lambda v: int(np.clip((v + 1) * 0.5 * 63, 0, 63))
    colY = occ[to_i(-0.2), :, to_i(0.0)]
    y_roof = g[np.where(colY)[0].max()] if colY.any() else 0.4

    tower_box = {"kind": "box", "center": [0.45, 0.3, 0.0], "size": [0.1, 0.6, 0.1],
                 "mode": "add", "smooth": 0.0}
    roof_box = {"kind": "box", "center": [0.0, float(y_roof + 0.11), -0.2],
                "size": [0.16, 0.12, 0.12], "mode": "add", "smooth": 0.0}

    def compose(ops, res=110):
        comp = EditableBuilding(base, [EditOp.from_dict(d) for d in ops]).composed()
        return sample_grid(comp, res, (-1, -1, -1, 1, 1, 1), device="cpu").numpy()

    fig, axes = plt.subplots(2, len(SEEDS), figsize=(3.1 * len(SEEDS), 6.6),
                             subplot_kw={"projection": "3d"})
    for row, (label, box) in enumerate([("tall corner box", tower_box),
                                        ("box on the roof", roof_box)]):
        for j, sd in enumerate(SEEDS):
            r = post("/interpret_mass", {"base_sdf_b64": plain["sdf_b64"], "res": 64,
                                         "op": box, "building_class": "RELIGIOUS",
                                         "seed": sd})
            draw(axes[row, j], compose(r["ops"]),
                 f"{label} · seed {sd}\n{r['kind']} ({r['n']} ops, {r['source']})")
            print(f"  {label} seed {sd}: {r['kind']} ({r['n']} ops, {r['source']})")

    stamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out = os.path.join(REPO, "outputs", "eval_visual", f"generative_add_{stamp}.png")
    fig.suptitle("GENERATIVE smart add — same box, sampled architecture (RELIGIOUS)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(out, dpi=120)
    print(f"[generative-add] -> {out}")


if __name__ == "__main__":
    main()
