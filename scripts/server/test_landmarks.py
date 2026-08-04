"""Ground landmark occurrence in the BuildingNet labels, then render the rich primitives
(dome=22, tower=7, steps=17) composed onto a building.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from recipe_inference import RecipeInferenceEngine
from scene.sdf_edit import recipe_base_sdf
from scene import sdf_detail as det
from scene.sdf_primitives import sample_grid, grid_to_mesh

PL = REPO / "data/BuildingNet_dataset_v0_1/model_data/point_cloud/point_labels"
DOME, TOWER, STAIRS = 22, 7, 17


def ground_occurrence(n_per_class=120):
    by = defaultdict(lambda: defaultdict(int))
    tot = defaultdict(int)
    files = sorted(os.listdir(PL)); np.random.RandomState(0).shuffle(files)
    cnt = defaultdict(int)
    for fn in files:
        cls = re.match(r"^([A-Z]+)", fn).group(1)
        if cnt[cls] >= n_per_class:
            continue
        d = np.fromiter(json.load(open(PL / fn)).values(), dtype=np.int32)
        u = set(np.unique(d).tolist())
        frac = lambda i: (d == i).mean()
        if DOME in u and frac(DOME) > 0.01: by[cls]["dome"] += 1
        if TOWER in u and frac(TOWER) > 0.02: by[cls]["tower"] += 1
        if STAIRS in u and frac(STAIRS) > 0.03: by[cls]["steps"] += 1
        tot[cls] += 1; cnt[cls] += 1
    print("[real per-class landmark occurrence from BuildingNet labels]")
    prob = {}
    for cls in ["RELIGIOUS", "PUBLIC", "COMMERCIAL", "RESIDENTIAL"]:
        if not tot[cls]:
            continue
        p = {k: round(by[cls][k] / tot[cls], 2) for k in ["dome", "tower", "steps"]}
        prob[cls] = p
        print(f"  {cls:12s} n={tot[cls]:3d}  dome={p['dome']:.2f} tower={p['tower']:.2f} steps={p['steps']:.2f}")
    return prob


def render(ax, mesh, title):
    if mesh is None or len(mesh.faces) == 0:
        ax.set_title(title, fontsize=8); return
    V, F = mesh.vertices, mesh.faces; tris = V[F]
    fy = tris[:, :, 1].mean(1); c = plt.cm.bone(0.25 + 0.6 * (fy - fy.min()) / (np.ptp(fy) + 1e-9))
    ax.add_collection3d(Poly3DCollection(tris[:, :, [0, 2, 1]], facecolors=c, edgecolors="k", linewidths=0.04))
    x, z, y = V[:, 0], V[:, 2], V[:, 1]
    ax.set_xlim(x.min(), x.max()); ax.set_ylim(z.min(), z.max()); ax.set_zlim(0, max(y.max(), 1))
    try: ax.set_box_aspect((np.ptp(x), np.ptp(z), max(np.ptp(y), 1)))
    except Exception: pass
    ax.view_init(elev=12, azim=-58); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title(title, fontsize=8)


def main():
    out = REPO / "outputs/landmark_detail"; out.mkdir(parents=True, exist_ok=True)
    prob = ground_occurrence()
    eng = RecipeInferenceEngine(); dev = eng.device

    fp = [[-9, -9], [9, -9], [9, 9], [-9, 9]]; H = 12.0
    params = eng.sample_params(fp, H, "PUBLIC", "public_civic", seed=1)
    base = recipe_base_sdf("public_civic", params, fp, H, device=dev)
    base_det = det.add_facade_detail(base, fp, H, det.vector_to_params(det.sample_detail_vector("public_civic", np.random.default_rng(0))))

    variants = [
        ("recipe base", base),
        ("+ facade detail", base_det),
        ("+ dome", det.add_landmarks(base_det, fp, H, dome=True)),
        ("+ dome + 4 towers", det.add_landmarks(base_det, fp, H, dome=True, n_towers=4)),
        ("+ dome + towers + steps", det.add_landmarks(base_det, fp, H, dome=True, n_towers=4, steps=True)),
    ]
    pad = 3.0
    bbox = (-9 - pad, 0.0, -9 - pad, 9 + pad, H * 1.7, 9 + pad)
    fig = plt.figure(figsize=(3.3 * len(variants), 3.6))
    for j, (label, sdf) in enumerate(variants):
        mesh = grid_to_mesh(sample_grid(sdf, 120, bbox, device=dev), bbox, 0.0)
        nv = 0 if mesh is None else len(mesh.vertices)
        print(f"  {label:26s} verts={nv}")
        ax = fig.add_subplot(1, len(variants), j + 1, projection="3d")
        render(ax, mesh, label)
    fig.suptitle("Rich landmark primitives (dome=22, tower=7, steps=17) — grounded in BuildingNet labels",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(out / "landmark_detail.png", dpi=115); plt.close(fig)
    print(f"[save] {out/'landmark_detail.png'}")
    json.dump(prob, open(out / "occurrence.json", "w"), indent=2)


if __name__ == "__main__":
    main()
