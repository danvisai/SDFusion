"""Diagnose the surface-artifact source in SDEdit refine on a recipe box.
Isolates: VQVAE round-trip (no diffusion) vs single-model SDEdit vs autoguidance (w=1,2)."""
from __future__ import annotations
import sys
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
    ax = fig.add_subplot(1, n, i, projection="3d"); ax.set_title(title, fontsize=8); ax.set_axis_off()
    if mesh is None or not len(mesh.vertices): return
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    ax.plot_trisurf(v[:, 0], v[:, 2], f, v[:, 1], color="#b9c4cf", edgecolor="none", shade=True)
    lim = [v.min(), v.max()]; ax.set_xlim(lim); ax.set_ylim(lim); ax.set_zlim(lim)
    ax.view_init(elev=18, azim=-60); ax.set_box_aspect((1, 1, 1))


def main():
    eng = RecipeInferenceEngine(); ref = Refiner(eng); dev = eng.device
    fp, h, style = rect(14, 18), 16.0, "modern"
    b = eng.generate_building(fp, "RESIDENTIAL", h, style, seed=1, detail=False)
    base = {"style": b.style, "recipe_params": b.recipe_params, "footprint": fp, "height": h}
    tower = EditOp(kind="box", center=(5.0, 9.0, 6.0), size=(1.6, 9.0, 1.6), mode="add").to_dict()
    recipe_mesh, _ = eng.params_to_mesh(b.recipe_params, b.style, fp, h)

    cfgs = [("m1.05 no-smooth", dict(strength=0.5, steps=8, autoguidance=True, margin=1.05, smooth_iters=0)),
            ("m1.05 +taubin12", dict(strength=0.5, steps=8, autoguidance=True, margin=1.05, smooth_iters=12)),
            ("m1.5 +taubin12", dict(strength=0.5, steps=8, autoguidance=True, margin=1.5, smooth_iters=12)),
            ("m1.5 +taubin25", dict(strength=0.5, steps=8, autoguidance=True, margin=1.5, smooth_iters=25)),
            ("m1.5 no-smooth", dict(strength=0.5, steps=8, autoguidance=True, margin=1.5, smooth_iters=0))]
    panels = [("recipe base", recipe_mesh)]
    for name, kw in cfgs:
        out = ref.refine_sdedit(base, [tower], **kw)
        print(f"  {name:22s} iou={out['iou_to_edit']:.3f} verts={0 if out['mesh'] is None else len(out['mesh'].vertices)}")
        panels.append((name, out["mesh"]))

    fig = plt.figure(figsize=(3.0 * len(panels), 3.4))
    for i, (t, m) in enumerate(panels): panel(fig, len(panels), i + 1, t, m)
    fig.suptitle("Diagnose snap artifacts: what injects the surface noise?", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    outp = REPO / "outputs/sdedit_refine/diag_sdedit_refine.png"; outp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outp, dpi=88); print(f"[saved] {outp}")


if __name__ == "__main__":
    main()
