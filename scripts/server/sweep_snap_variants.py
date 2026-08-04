"""Quick warm sweep of snap_volume sampling variants on one sculpt case.

{autoguidance on/off} x {renorm margin 1.05 / 1.5} on the res_tower case, rendered into one
row-montage -> outputs/sculpt_regression/variants_<UTC>.png. Picks the snap defaults.
"""
import datetime
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from refine import Refiner
from models.networks.diff_recipe import build_diff_recipe
from scene.sdf_primitives import grid_to_mesh

dev = "cuda" if torch.cuda.is_available() else "cpu"
r = Refiner(SimpleNamespace(device=dev))
_, default_fn, _ = build_diff_recipe("modern")
params = default_fn(dev).detach().cpu().numpy()
poly = np.array([[-7, -9], [7, -9], [7, 9], [-7, 9]], np.float32)
grid, c, s, hn = r.building_volume(poly, "modern", params, 16.0)
edit = dict(kind="box", center=[0.45, 0.25, 0.0], size=[0.14, 0.5, 0.14], mode="add", smooth=0.0)

variants = [
    ("AG2.0 m1.05 (current)", dict(autoguidance=True, auto_scale=2.0, margin=1.05)),
    ("AG off m1.05",          dict(autoguidance=False, margin=1.05)),
    ("AG2.0 m1.5",            dict(autoguidance=True, auto_scale=2.0, margin=1.5)),
    ("AG off m1.5",           dict(autoguidance=False, margin=1.5)),
]

fig, axes = plt.subplots(1, len(variants) + 1, figsize=(3 * (len(variants) + 1), 3.2),
                         subplot_kw={"projection": "3d"})

def draw(ax, g, title):
    m = grid_to_mesh(torch.from_numpy(np.asarray(g, np.float32)), (-1, -1, -1, 1, 1, 1), 0.0)
    ax.set_title(title, fontsize=7)
    if m is not None and len(m.faces):
        pc = Poly3DCollection(m.vertices[m.faces], alpha=1.0)
        pc.set_facecolor((0.80, 0.78, 0.72)); pc.set_edgecolor("none")
        ax.add_collection3d(pc)
        nv = len(m.vertices)
    else:
        nv = 0
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(elev=16, azim=-60)
    return nv

draw(axes[0], grid, "base+edit (input)")
for ax, (name, kw) in zip(axes[1:], variants):
    snapped, iou = r.snap_volume(grid, [edit], strength=0.5, **kw)
    nv = draw(ax, snapped, "")
    ax.set_title(f"{name}\niou={iou:.2f} v={nv}", fontsize=7)
    print(f"{name:24s} iou={iou:.3f} verts={nv}")

stamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
out = os.path.join(REPO, "outputs", "sculpt_regression", f"variants_{stamp}.png")
plt.tight_layout(); plt.savefig(out, dpi=120)
print("wrote", out)
