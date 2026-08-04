"""A/B photo: GLOBAL vs LOCALIZED generative snap on a tower edit.

Panels: crisp edited input · global snap (old: whole building remolded) ·
localized snap (new: base crisp, generative only at the placed mass).
Run (server warm): env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python scripts/server/ab_local_snap.py
"""
import base64
import datetime
import json
import os
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

URL = os.environ.get("SCULPT_URL", "http://127.0.0.1:8099")
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RECT = [[-7, -9], [7, -9], [7, 9], [-7, 9]]
TOWER = {"kind": "box", "center": [0.45, 0.25, 0.0], "size": [0.14, 0.5, 0.14],
         "mode": "add", "smooth": 0.0}


def post(path, body):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=900).read())


def vol(b64):
    return np.frombuffer(base64.b64decode(b64), dtype="<f4").reshape(64, 64, 64).copy()


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
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(elev=18, azim=-55)


def main():
    plain = post("/building_sdf", {"footprint": RECT, "style": "modern",
                                   "building_class": "RESIDENTIAL", "height": 16})
    base = plain["sdf_b64"]
    body = {"base_sdf_b64": base, "res": 64, "edits": [TOWER], "strength": 0.5}
    g_glob = vol(post("/snap_sdf", {**body, "local": False})["sdf_b64"])
    g_loc = vol(post("/snap_sdf", {**body, "local": True})["sdf_b64"])

    # crisp edited input for reference
    import torch
    import sys
    sys.path.insert(0, REPO)
    sys.path.insert(0, os.path.dirname(__file__))
    from refine import volume_to_sdf
    from scene.sdf_edit import EditableBuilding, EditOp
    g1 = torch.linspace(-1, 1, 64)
    Z, Y, X = torch.meshgrid(g1, g1, g1, indexing="ij")
    pts = torch.stack([X, Y, Z], -1).reshape(-1, 3)
    comp = EditableBuilding(volume_to_sdf(vol(base), "cpu"),
                            [EditOp.from_dict(TOWER)]).composed()
    with torch.no_grad():
        g_edit = comp(pts).reshape(64, 64, 64).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), subplot_kw={"projection": "3d"})
    draw(axes[0], g_edit, "crisp edited input\n(base + placed tower)")
    draw(axes[1], g_glob, "GLOBAL snap (old)\nwhole building remolded")
    draw(axes[2], g_loc, "LOCALIZED snap (new)\nbase crisp · generative at the mass")
    stamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out = os.path.join(REPO, "outputs", "eval_visual", f"local_snap_ab_{stamp}.png")
    fig.tight_layout(); fig.savefig(out, dpi=120)
    print(f"[ab] -> {out}")


if __name__ == "__main__":
    main()
