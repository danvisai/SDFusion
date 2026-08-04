"""Validate the SDEdit refine bridge (recipe edit -> Frame-N -> prior -> world mesh) headlessly.

Builds a recipe building, adds a crude tower edit, runs Refiner.refine(mode='sdedit') at several
strengths, and renders recipe / edited / snapped so we can confirm the world<->Frame-N bridge is
correct and the snap output is a clean building (not garbage from an off-distribution input).
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from recipe_inference import RecipeInferenceEngine
from refine import Refiner, _bbox
from scene.sdf_edit import EditableBuilding, EditOp, recipe_base_sdf

rect = lambda w, d: [[-w/2, -d/2], [w/2, -d/2], [w/2, d/2], [-w/2, d/2]]


def panel(fig, n, i, title, mesh):
    ax = fig.add_subplot(1, n, i, projection="3d"); ax.set_title(title, fontsize=9); ax.set_axis_off()
    if mesh is None or not len(mesh.vertices):
        return
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    ax.plot_trisurf(v[:, 0], v[:, 2], f, v[:, 1], color="#b9c4cf", edgecolor="none", shade=True)
    lim = [v.min(), v.max()]; ax.set_xlim(lim); ax.set_ylim(lim); ax.set_zlim(lim)
    ax.view_init(elev=18, azim=-60); ax.set_box_aspect((1, 1, 1))


def main():
    eng = RecipeInferenceEngine(); ref = Refiner(eng); dev = eng.device
    print(f"[device] {dev}")
    fp, h, style, cls = rect(14, 18), 16.0, "modern", "RESIDENTIAL"
    b = eng.generate_building(fp, cls, h, style, seed=1, detail=False)
    base = {"style": b.style, "recipe_params": b.recipe_params, "footprint": fp, "height": h}
    print(f"[recipe] style={b.style} verts={b.n_vertices}")

    # crude tall box at a footprint corner (meters; size = half-extents)
    tower = EditOp(kind="box", center=(5.0, 9.0, 6.0), size=(1.6, 9.0, 1.6), mode="add").to_dict()

    recipe_mesh, _ = eng.params_to_mesh(b.recipe_params, b.style, fp, h)
    bb = _bbox(fp, h, [tower])
    edited_mesh = EditableBuilding(recipe_base_sdf(style, b.recipe_params, fp, h, device=dev),
                                   [EditOp.from_dict(tower)]).to_mesh(bb, res=64, device=dev)

    panels = [("recipe base", recipe_mesh), ("edited (+tower)", edited_mesh)]
    for s in (0.3, 0.5, 0.7):
        t0 = time.perf_counter()
        out = ref.refine(base, [tower], mode="sdedit", strength=s, sdedit_steps=8,
                         autoguidance=True, auto_scale=2.0)
        dt = (time.perf_counter() - t0) * 1000
        m = out["mesh"]
        print(f"  sdedit s={s:.1f}  {dt:7.0f} ms  iou_to_edit={out['iou_to_edit']:.3f}  "
              f"verts={0 if m is None else len(m.vertices)}")
        panels.append((f"snap s={s:.1f}\niou={out['iou_to_edit']:.2f}", m))

    fig = plt.figure(figsize=(3.2 * len(panels), 3.6))
    for i, (t, m) in enumerate(panels):
        panel(fig, len(panels), i + 1, t, m)
    fig.suptitle("SDEdit refine (mode='sdedit'): recipe -> crude edit -> snapped to BAG manifold", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    outp = REPO / "outputs/sdedit_refine/test_sdedit_refine.png"; outp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outp, dpi=88); print(f"[saved] {outp}")


if __name__ == "__main__":
    main()
