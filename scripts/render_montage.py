"""Render a montage of OBJ meshes to one PNG for quick visual eval (headless)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh

files = sys.argv[1:-1]
out = sys.argv[-1]
n = len(files)
fig = plt.figure(figsize=(4 * n, 4.2))
for i, f in enumerate(files):
    ax = fig.add_subplot(1, n, i + 1, projection="3d")
    m = trimesh.load(f, process=False)
    v, fc = np.asarray(m.vertices), np.asarray(m.faces)
    # x=width, y=depth, z=height(up)
    ax.plot_trisurf(v[:, 0], v[:, 2], fc, v[:, 1], color="#cdb892",
                    edgecolor="none", linewidth=0, antialiased=True, shade=True)
    ax.set_title(Path(f).stem, fontsize=10)
    ax.view_init(elev=22, azim=-58)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
    try:
        lim = [v.min(0).min(), v.max(0).max()]
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_zlim(lim)
    except Exception:
        pass
fig.tight_layout()
fig.savefig(out, dpi=90, bbox_inches="tight")
print(f"[montage] -> {out}")
