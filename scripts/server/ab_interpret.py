"""SMART ADD photo: the same raw boxes, dumb-unioned vs INTERPRETED as architecture.

Four placements on one building: tall corner box / box on the roof / slab on a wall /
box on the ground beside the building. Row 1 = raw CSG union (what sculpting alone
gives). Row 2 = /interpret_mass construction (tower / dormer / balcony / wing).
Run (server warm): env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python scripts/server/ab_interpret.py
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


def make_cases(grid):
    """Place the raw boxes RELATIVE to the actual massing (like a user would visually):
    on the roof means bottom at the local roofline; on the ground means bottom at grade."""
    occ = grid <= 0
    g = np.linspace(-1, 1, 64)
    ys = np.where(occ.any(axis=(0, 2)))[0]
    y_ground, y_top = g[ys.min()], g[ys.max()]
    to_i = lambda v: int(np.clip((v + 1) * 0.5 * 63, 0, 63))
    colY = occ[to_i(-0.2), :, to_i(0.0)]
    y_roof = g[np.where(colY)[0].max()] if colY.any() else y_top
    return [
        ("tall corner box", {"kind": "box", "center": [0.45, 0.3, 0.0],
                             "size": [0.1, 0.6, 0.1], "mode": "add", "smooth": 0.0}),
        ("box on the roof", {"kind": "box",
                             "center": [0.0, float(y_roof + 0.11), -0.2],
                             "size": [0.16, 0.12, 0.12], "mode": "add", "smooth": 0.0}),
        ("slab on a wall", {"kind": "box", "center": [0.0, 0.05, 0.78],
                            "size": [0.16, 0.05, 0.07], "mode": "add", "smooth": 0.0}),
        ("box on the ground", {"kind": "box",
                               "center": [0.85, float(y_ground + 0.17), 0.0],
                               "size": [0.18, 0.18, 0.2], "mode": "add", "smooth": 0.0}),
    ]


def post(path, body):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=900).read())


def draw(ax, g, title):
    from skimage import measure
    ax.set_title(title, fontsize=9)
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
                                   "building_class": "RESIDENTIAL", "height": 16})
    grid = np.frombuffer(base64.b64decode(plain["sdf_b64"]),
                         dtype="<f4").reshape(64, 64, 64).copy()
    base = volume_to_sdf(grid, "cpu")

    def compose(ops, res=96):
        comp = EditableBuilding(base, [EditOp.from_dict(d) for d in ops]).composed()
        return sample_grid(comp, res, (-1, -1, -1, 1, 1, 1), device="cpu").numpy()

    cases = make_cases(grid)
    fig, axes = plt.subplots(2, len(cases), figsize=(3.2 * len(cases), 7),
                             subplot_kw={"projection": "3d"})
    for j, (name, op) in enumerate(cases):
        r = post("/interpret_mass", {"base_sdf_b64": plain["sdf_b64"], "res": 64, "op": op})
        draw(axes[0, j], compose([op]), f"{name}\nraw union")
        draw(axes[1, j], compose(r["ops"]),
             f"INTERPRETED: {r['kind'].upper()}\n({r['n']} ops)")
        print(f"  {name:20s} -> {r['kind']} ({r['n']} ops)")

    stamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out = os.path.join(REPO, "outputs", "eval_visual", f"smart_add_ab_{stamp}.png")
    fig.suptitle("SMART ADD — the system makes sense of what you place", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(out, dpi=120)
    print(f"[smart-add] -> {out}")


if __name__ == "__main__":
    main()
