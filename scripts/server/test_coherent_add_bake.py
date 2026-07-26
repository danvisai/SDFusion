"""Does the PRODUCTION bake path (not the Stage3a snap) give a clean, attached add + good
diffusion-textured detail? Mirrors /bake_texture exactly, in-process (no server):

  base recipe building -> detail_edits (RAW CSG union via EditableBuilding, always exact,
  never suppressed) -> detail_cube_volume (procedural composer: windows/bands/plinth/roof) ->
  bake_glb (SDXL + multi-ControlNet texture diffusion, geometry untouched).

This is a different path from the Stage3a SDEdit massing-snap (which we found suppresses
added mass in scripts/sdedit_layerAB_eval.py) — here the ADD is pure CSG (cannot be
suppressed by construction), and "detail" comes from the appearance/texture-bake layer.

  env -u LD_PRELOAD -u LD_LIBRARY_PATH PYTHONPATH=. ./sdfusion/bin/python scripts/server/test_coherent_add_bake.py
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts" / "server"))
sys.path.insert(0, str(REPO / "scripts" / "appearance"))

import numpy as np
import torch

from recipe_inference import RecipeInferenceEngine
from refine import Refiner, volume_to_sdf
from scene.sdf_edit import EditableBuilding, EditOp
from scene.sdf_primitives import sample_grid

OUT = REPO / "outputs/coherent_add_bake"; OUT.mkdir(parents=True, exist_ok=True)


def world_box_op(center_w, size_w, c, s, mode="add", smooth_w=0.1):
    """EditableBuilding(volume_to_sdf(grid), ...) composes in CUBE [-1,1] space (the grid IS
    the base sdf there) — world-meter centers/sizes must be converted via the SAME (center,
    scale) the massing volume was built with, or the primitive lands outside the sampled cube
    entirely (silently: occ delta stays exactly 0). world = cube*s + c  =>  cube = (world-c)/s."""
    c = np.asarray(c, np.float32)
    cc = ((np.asarray(center_w, np.float32) - c) / s).tolist()
    sz = (np.asarray(size_w, np.float32) / s).tolist()
    return EditOp("box", center=tuple(cc), size=tuple(sz), mode=mode, smooth=smooth_w / s).to_dict()


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    eng = RecipeInferenceEngine()
    r = Refiner(eng, res=64)

    fp = [[-8, -10], [8, -10], [8, 10], [-8, 10]]; H = 12.0
    # "victorian" is in scene.sdf_detail.ARCH_STYLES -> windows/openings get a round top,
    # visually confirming the arch capability (not just the flat-top tower/balcony treatments).
    style, cls = "victorian", "RESIDENTIAL"
    params = eng.sample_params(fp, H, cls, style, seed=3)
    grid, c, s, _hn = r.building_volume(fp, style, params, H, res=64)
    print(f"[base] center={c} scale={s:.3f} occ={float((grid<=0).mean()):.3f}")

    # Five probes in WORLD METERS, converted to cube units via (c, s), one per classify_shape
    # vocabulary entry: size = half-extents (sdf_box convention). Sizes were chosen to (a)
    # classify as intended and (b) survive the 64-res raw CSG union (< ~0.45m half-extent on
    # the thin axis vanishes there — a resolution floor, not a treatment bug).
    CASES = [
        ("tower", world_box_op((6.0, H + 3.5, 8.0), (2.2, 4.5, 2.2), c, s, smooth_w=0.15)),
        ("balcony", world_box_op((9.0, 5.0, 0.0), (1.0, 0.2, 1.8), c, s, smooth_w=0.05)),
        ("bay", world_box_op((0.0, H * 0.25, 10.6), (1.3, 1.3, 1.3), c, s, smooth_w=0.1)),
        ("wall", world_box_op((13.0, 2.5, -3.0), (0.15, 2.5, 4.0), c, s, smooth_w=0.05)),
        ("window", world_box_op((8.5, 6.0, -3.0), (0.28, 0.9, 0.7), c, s, smooth_w=0.05)),
    ]

    import neural_appearance as na
    import texture_bake as tb
    print("[bake] loading SDXL pipe ...")
    pipe = na.get_pipe()

    for tag, edit in CASES:
        # 1) RAW CSG add on the cube-frame massing volume (exactly what /bake_texture does
        #    for detail_edits) -- deterministic union, cannot suppress the added mass.
        comp = EditableBuilding(volume_to_sdf(grid, dev), [EditOp.from_dict(edit)]).composed()
        edited = sample_grid(comp, 64, (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0), device=dev).cpu().numpy()
        add_occ = float((edited <= 0).mean()) - float((grid <= 0).mean())
        print(f"\n=== {tag} === csg-add occ delta={add_occ:.4f} (should be >0: mass was added)")

        # 2) procedural composer detail (windows/bands/plinth/roof + the ADDED element's OWN
        #    detail via detail_edits -> scene.sdf_detail.add_element_detail, 2026-07-02 fix)
        grid96 = r.detail_cube_volume(edited, c, s, building_class=cls, style=style, seed=3,
                                      res_out=96, detail_edits=[edit])
        add_occ96 = float((grid96 <= 0).mean())
        print(f"  detail_cube_volume: occ={add_occ96:.3f} shape={grid96.shape}")

        # 3) SDXL multi-view texture bake (geometry untouched from here on)
        prompt = f"photo of a {style} {cls.lower()} building with a {tag}, architectural photography, high detail"
        print(f"  [bake] {tag}: baking 6-view SDXL texture ...")
        res = tb.bake_building(grid96, pipe, prompt, seed=7, n_views=6, steps=28,
                               style=style, return_views=True)
        print(f"  [bake] coverage={res['coverage']:.3f} n_verts={res['n_verts']}")
        res["atlas"].save(OUT / f"{tag}_atlas.png")
        glb = res["mesh"].export(file_type="glb")
        (OUT / f"{tag}_building.glb").write_bytes(glb)

        # verification sheet: 6 raw diffusion views + atlas, so we can SEE whether the added
        # element got plausible, attached detail (not a floating/ungrounded texture patch).
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        views = res["views"]
        fig = plt.figure(figsize=(3.0 * (len(views) + 1), 3.2))
        for i, v in enumerate(views):
            ax = fig.add_subplot(1, len(views) + 1, i + 1); ax.imshow(v); ax.axis("off")
            ax.set_title(f"view {i}", fontsize=8)
        ax = fig.add_subplot(1, len(views) + 1, len(views) + 1)
        ax.imshow(res["atlas"]); ax.axis("off"); ax.set_title("UV atlas", fontsize=8)
        fig.suptitle(f"coherent-add + SDXL bake: {tag} (CSG-preserved, occ_delta={add_occ:.4f})", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        out_p = OUT / f"{tag}_sheet.png"; fig.savefig(out_p, dpi=100); plt.close(fig)
        print(f"  [saved] {out_p}  {OUT / f'{tag}_building.glb'}")


if __name__ == "__main__":
    main()
