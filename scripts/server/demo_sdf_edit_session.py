"""Headless validation of the interactive SDF edit engine (scene/sdf_edit.py).

Simulates a sculpt session on a B+.6-generated base building: add a rooftop tower, add a
gable roof, carve an entrance, add a chimney, add a corner turret. After each edit it
re-meshes and records latency at PREVIEW resolution (fast, for live drag) and COMMIT
resolution (crisp, on mouse-up), then renders a storyboard.

Validates: (1) edits compose into coherent buildings; (2) the SDF stays valid/watertight;
(3) latency meets the docs/DEPLOYMENT_PLAN budget (<200 ms slider, <500 ms drag-drop).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts" / "server"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from scene.sdf_edit import EditableBuilding, EditOp, recipe_base_sdf
from recipe_inference import RecipeInferenceEngine

OUT = REPO / "outputs/sdf_edit_demo"


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
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # Base building: a B+.6-generated modern block, 16 x 20 m, ~13 m tall.
    eng = RecipeInferenceEngine()
    poly = np.array([[-8, -10], [8, -10], [8, 10], [-8, 10]], dtype=np.float32)
    H = 13.0
    params = eng.sample_params(poly, H, "COMMERCIAL", "modern", seed=2)
    base = recipe_base_sdf("modern", params, poly, H, device=dev)
    bldg = EditableBuilding(base)

    # World bbox generous enough for the body + edits (roof, turret).
    bbox = (-12, 0.0, -14, 12, 24.0, 14)

    # A sequence of palette edits (what a user would drag in).
    session = [
        ("base building",        None),
        ("+ rooftop tower",      EditOp("box",     center=(4, H + 3, 4),  size=(2.5, 3.5, 2.5), smooth=0.6)),
        ("+ gable roof",         EditOp("gable",   center=(0, 0, 0),      size=(16, 20, H, 4.0), mode="add", smooth=0.8)),
        ("- entrance notch",     EditOp("box",     center=(0, 2.0, -10),  size=(2.0, 2.5, 2.0), mode="subtract", smooth=0.4)),
        ("+ chimney",            EditOp("cylinder",center=(-5, H + 2, -5),size=(0.8, 4.0))),
        ("+ corner turret",      EditOp("cylinder",center=(-8, H * 0.5, 10), size=(2.0, H + 4))),
    ]

    PREVIEW, COMMIT = 40, 72
    rows = []
    print(f"[edit session] device={dev} | preview={PREVIEW}^3  commit={COMMIT}^3")
    print(f"{'step':22s} {'preview ms':>10} {'commit ms':>10} {'verts':>7} {'faces':>7} watertight")
    for label, op in session:
        if op is not None:
            bldg.add(op)
        # warm + time preview
        if dev == "cuda":
            _ = bldg.to_mesh(bbox, PREVIEW, device=dev); torch.cuda.synchronize()
        t0 = time.time(); _ = bldg.to_mesh(bbox, PREVIEW, device=dev)
        if dev == "cuda": torch.cuda.synchronize()
        prev_ms = 1000 * (time.time() - t0)
        t0 = time.time(); mesh = bldg.to_mesh(bbox, COMMIT, device=dev)
        if dev == "cuda": torch.cuda.synchronize()
        commit_ms = 1000 * (time.time() - t0)
        nv = 0 if mesh is None else len(mesh.vertices)
        nf = 0 if mesh is None else len(mesh.faces)
        wt = "-" if mesh is None else str(mesh.is_watertight)
        print(f"{label:22s} {prev_ms:10.1f} {commit_ms:10.1f} {nv:7d} {nf:7d}  {wt}")
        rows.append((label, mesh, prev_ms, commit_ms))

    # Storyboard
    fig = plt.figure(figsize=(3 * len(rows), 3.2))
    for j, (label, mesh, pms, cms) in enumerate(rows):
        ax = fig.add_subplot(1, len(rows), j + 1, projection="3d")
        render(ax, mesh, f"{label}\n{pms:.0f}ms preview")
    fig.suptitle("Interactive SDF sculpting on a generated building (palette edits)", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "edit_storyboard.png", dpi=105); plt.close(fig)
    print(f"[save] {OUT/'edit_storyboard.png'}")

    # Export final edited building + its serializable edit state (host stores this).
    rows[-1][1].export(OUT / "edited_building.glb")
    import json
    json.dump(bldg.edit_state(), open(OUT / "edit_state.json", "w"), indent=2)
    print(f"[save] {OUT/'edited_building.glb'}  + edit_state.json ({len(bldg.ops)} ops)")


if __name__ == "__main__":
    main()
