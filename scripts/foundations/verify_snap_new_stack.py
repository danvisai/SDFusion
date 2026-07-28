"""Task #5 — verify the snap on the NEW foundations stack (clean VQVAE + 20k footprint+height prior).

Tests two claims: (1) the snap is sharper than the old prior, and (2) we can DROP the margin=1.5 /
Taubin workaround now that the clean VQVAE reconstructs cube-filling boxes (gap#6 fixed). Injects the
new models into the real Refiner (reuses the deployed recipe->Frame-N->sdedit bridge) and renders
margin1.05-raw vs margin1.05+taubin vs margin1.5+taubin (the old config) for the same tower edit.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts/server"))
from recipe_inference import RecipeInferenceEngine
from refine import Refiner, _bbox
from scene.sdf_edit import EditableBuilding, EditOp, recipe_base_sdf
from models.stage3a_model import Stage3aModel

RUN = REPO / "logs_building/2026-06-08T11-50-42-stage3a-hybrid-clean/ckpt"
VQ_CLEAN = REPO / "logs_building/vqvae_clean_ft/vqvae_clean.pth"
rect = lambda w, d: [[-w/2, -d/2], [w/2, -d/2], [w/2, d/2], [-w/2, d/2]]


def build_prior(ckpt, dev):
    opt = SimpleNamespace(isTrain=False, device=dev, df_cfg=str(REPO/"configs/stage3a_sdf_diffusion.yaml"),
        vq_cfg=str(REPO/"configs/vqvae_bnet.yaml"), vq_ckpt=str(VQ_CLEAN), ckpt=str(ckpt),
        ddim_steps=50, debug="0", gpu_ids=[0] if dev=="cuda" else [], ckpt_dir="/tmp",
        latent_size_HW=(16,16), latent_size_D=16, use_extra_cond=True)
    m = Stage3aModel(); m.initialize(opt)
    st = torch.load(str(ckpt), map_location="cpu", mmap=True)
    if "ema_df" in st:
        m.df.load_state_dict(st["ema_df"]); print("[prior] using EMA weights")
    return m


def panel(fig, n, i, title, mesh):
    ax = fig.add_subplot(1, n, i, projection="3d"); ax.set_title(title, fontsize=9); ax.set_axis_off()
    if mesh is None or not len(mesh.vertices):
        return
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    ax.plot_trisurf(v[:, 0], v[:, 2], f, v[:, 1], color="#b9c4cf", edgecolor="none", shade=True)
    lim = [v.min(), v.max()]; ax.set_xlim(lim); ax.set_ylim(lim); ax.set_zlim(lim)
    ax.view_init(elev=18, azim=-60); ax.set_box_aspect((1, 1, 1))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    eng = RecipeInferenceEngine(); ref = Refiner(eng)
    ref._sd_main = build_prior(RUN / "stage3a_steps-latest.pth", dev)   # inject new prior
    fp, h = rect(14, 18), 16.0
    b = eng.generate_building(fp, "RESIDENTIAL", h, "modern", seed=1, detail=False)
    base = {"style": b.style, "recipe_params": b.recipe_params, "footprint": fp, "height": h}
    tower = EditOp(kind="box", center=(5.0, 9.0, 6.0), size=(1.6, 9.0, 1.6), mode="add").to_dict()
    recipe_mesh, _ = eng.params_to_mesh(b.recipe_params, b.style, fp, h)
    edited_mesh = EditableBuilding(recipe_base_sdf("modern", b.recipe_params, fp, h, device=dev),
                                   [EditOp.from_dict(tower)]).to_mesh(_bbox(fp, h, [tower]), res=64, device=dev)

    panels = [("recipe base", recipe_mesh), ("edited (+tower)", edited_mesh)]
    for label, margin, smooth in [("m1.05 RAW", 1.05, 0), ("m1.05 +taubin", 1.05, 12), ("m1.5 +taubin (old)", 1.5, 12)]:
        t0 = time.time()
        out = ref.refine_sdedit(base, [tower], strength=0.5, steps=24, autoguidance=False,
                                margin=margin, smooth_iters=smooth)
        m = out["mesh"]; nv = 0 if m is None else len(m.vertices)
        print(f"  {label:20s} iou={out['iou_to_edit']:.3f} verts={nv} {1000*(time.time()-t0):.0f}ms")
        panels.append((f"{label}\niou={out['iou_to_edit']:.2f} v={nv}", m))

    fig = plt.figure(figsize=(3.1 * len(panels), 3.6))
    for i, (t, m) in enumerate(panels):
        panel(fig, len(panels), i + 1, t, m)
    fig.suptitle("Snap on NEW stack (clean VQVAE + 20k footprint+height prior) — can we drop margin=1.5?", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    outp = REPO / "outputs/foundations/verify_snap_new_stack.png"; outp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outp, dpi=88); print(f"[saved] {outp}")


if __name__ == "__main__":
    main()
