"""Validate /refine_with_edit: sculpt a building, then refine (fast/quality) + re-style.

Renders: crude edited building | fast refine | quality refine | re-styled refine,
and reports iou_to_edit (how well the refine kept the sculpted massing) + latency.
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from recipe_inference import RecipeInferenceEngine
from refine import Refiner
from scene.sdf_edit import EditableBuilding, EditOp, recipe_base_sdf

OUT = REPO / "outputs/refine_demo"


def render(ax, mesh, title):
    if mesh is None or len(mesh.faces) == 0:
        ax.set_title(title + "\n(empty)", fontsize=8); return
    V, F = mesh.vertices, mesh.faces; tris = V[F]
    fy = tris[:, :, 1].mean(1); col = plt.cm.viridis(0.15 + 0.7 * (fy - fy.min()) / (np.ptp(fy) + 1e-9))
    ax.add_collection3d(Poly3DCollection(tris[:, :, [0, 2, 1]], facecolors=col, edgecolors="none"))
    x, z, y = V[:, 0], V[:, 2], V[:, 1]
    ax.set_xlim(x.min(), x.max()); ax.set_ylim(z.min(), z.max()); ax.set_zlim(0, max(y.max(), 1))
    try: ax.set_box_aspect((np.ptp(x), np.ptp(z), max(np.ptp(y), 1)))
    except Exception: pass
    ax.view_init(elev=22, azim=-58); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title(title, fontsize=8)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    eng = RecipeInferenceEngine()
    ref = Refiner(eng, res=64)

    fp = [[-8, -10], [8, -10], [8, 10], [-8, 10]]
    H = 12.0
    params = eng.sample_params(fp, H, "RESIDENTIAL", "modern", seed=3)
    base_state = {"style": "modern", "recipe_params": [float(x) for x in params],
                  "footprint": fp, "height": H}

    # Crude sculpt: add a tower + a side wing, carve an entrance.
    edits = [
        EditOp("box", center=(5, H + 3, 5), size=(3, 4, 3), smooth=0.6).to_dict(),
        EditOp("box", center=(-11, 4, 0), size=(4, 4, 6), mode="add", smooth=0.8).to_dict(),
        EditOp("box", center=(0, 2.5, -10), size=(2, 3, 2), mode="subtract", smooth=0.4).to_dict(),
    ]

    # Crude edited mesh (the input to refine)
    base_sdf = recipe_base_sdf("modern", params, fp, H, device=eng.device)
    crude = EditableBuilding(base_sdf, [EditOp.from_dict(d) for d in edits])
    from refine import _bbox
    bbox = _bbox(fp, H, edits)
    crude_mesh = crude.to_mesh(bbox, 72, device=eng.device)

    jobs = [("crude edit", None), ("fast (modern)", ("fast", "modern")),
            ("quality (modern)", ("quality", "modern")), ("restyle (victorian)", ("fast", "victorian"))]
    rows = []
    print(f"[refine] device={eng.device}")
    for label, spec in jobs:
        if spec is None:
            rows.append((label, crude_mesh, None, None)); continue
        mode, ts = spec
        t = time.time()
        out = ref.refine(base_state, edits, target_style=ts, mode=mode,
                         building_class="RESIDENTIAL", seed=3, steps=250)
        dt = 1000 * (time.time() - t)
        rows.append((label, out["mesh"], out["iou_to_edit"], dt))
        print(f"  {label:20s} style={out['style']:10s} iou_to_edit={out['iou_to_edit']:.3f} "
              f"params={len(out['recipe_params'])} | {dt:.0f}ms")

    fig = plt.figure(figsize=(3.4 * len(rows), 3.4))
    for j, (label, mesh, iou, dt) in enumerate(rows):
        ax = fig.add_subplot(1, len(rows), j + 1, projection="3d")
        t = label if iou is None else f"{label}\nIoU→edit {iou:.2f}, {dt:.0f}ms"
        render(ax, mesh, t)
    fig.suptitle("AI Refine: crude sculpt -> clean styled building (keeps massing)", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "refine_compare.png", dpi=105); plt.close(fig)
    print(f"[save] {OUT/'refine_compare.png'}")


if __name__ == "__main__":
    main()
