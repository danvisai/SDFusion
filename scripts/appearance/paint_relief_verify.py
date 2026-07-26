"""Standalone verification for paint-to-relief (POST /paint_relief against a locally
running inference_service, no UI needed). Mirrors texture_bake.py's own __main__ demo:
fetch a live detail volume, synthesize a fake painted circle aimed at one facade, bake,
then plot the depth/edge G-buffer + mask overlay + a before/after grid cross-section so a
real geometric bump/carving (not just color) is visually confirmable.

Run (server already up on :8099, matching texture_bake.py's own demo):
    PYTHONPATH=. ./sdfusion/bin/python scripts/appearance/paint_relief_verify.py
"""
from __future__ import annotations

import base64
import json
import os
import sys
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
import texture_bake as tb  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "paint_relief_verify")
os.makedirs(OUT, exist_ok=True)
URL = os.environ.get("VERIFY_URL", "http://127.0.0.1:8099")


def post(p, b):
    r = urllib.request.Request(URL + p, data=json.dumps(b).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=600).read())


def grid_from_resp(resp, key="sdf_b64"):
    res = resp["res"]
    return np.frombuffer(base64.b64decode(resp[key]), dtype="<f4").reshape(res, res, res).copy()


def make_paint(grid, cam, view_res=512, radius_frac=0.15, color=(176, 96, 64, 255)):
    """A filled colored circle (RGBA, transparent elsewhere) centered on the painted-view
    hit centroid, so it's guaranteed to land on the building surface regardless of camera
    framing -- mirrors sculpt.html's paintDab() (a real color brush, not a binary mask)."""
    from PIL import Image, ImageDraw
    _depth, _edge, basis = tb.trace_view(grid, cam, res=view_res)
    hit = basis["hit"]
    rows, cols = np.where(hit)
    cy, cx = (rows.mean(), cols.mean()) if len(rows) else (view_res / 2, view_res / 2)
    r = view_res * radius_frac
    img = Image.new("RGBA", (view_res, view_res), (0, 0, 0, 0))
    ImageDraw.Draw(img).ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    buf = __import__("io").BytesIO()
    img.save(buf, format="PNG")
    return img, base64.b64encode(buf.getvalue()).decode()


def main():
    RECT = [[-7, -9], [7, -9], [7, 9], [-7, 9]]
    style, cls, h = "victorian", "RESIDENTIAL", 14
    g = post("/building_sdf", {"footprint": RECT, "style": style, "building_class": cls, "height": h})
    base_res = g["res"] if "res" in g else 64
    dv = post("/detail_volume", {"base_sdf_b64": g["sdf_b64"], "res": base_res,
                                 "center": g["center"], "scale": g["scale"],
                                 "building_class": cls, "style": style, "res_out": 96})
    grid96 = grid_from_resp(dv)
    print(f"[verify] fetched detail volume {grid96.shape}, occ frac "
         f"{(grid96 <= 0).mean():.3f}")

    center, ext = tb.occ_frame(grid96)
    cam_raw = tb.make_cameras(center, ext, n=1)[0]        # reuse the real orbit-camera math
    cam = {"pos": [float(x) for x in cam_raw["pos"]], "look": [float(x) for x in cam_raw["look"]],
          "fov": float(cam_raw["fov"])}                   # numpy float32 -> JSON-serializable
    paint_pil, paint_b64 = make_paint(grid96, cam)
    paint_pil.save(os.path.join(OUT, "paint.png"))

    prompt = "carved stone rosette relief, victorian architectural ornament, high detail"
    strength = 0.6
    resp = post("/paint_relief", {
        "base_sdf_b64": g["sdf_b64"], "res": base_res, "center": g["center"], "scale": g["scale"],
        "building_class": cls, "style": style, "cam": cam, "paint_png_b64": paint_b64,
        "prompt": prompt, "strength": strength, "relief_depth": 0.12, "band": 0.07, "return_mesh": True,
    })
    out_grid = grid_from_resp(resp)
    print(f"[verify] paint_relief -> grid {out_grid.shape}, "
         f"max |delta| {np.abs(out_grid - grid96).max():.4f}")

    if resp.get("mesh_glb_b64"):
        glb = base64.b64decode(resp["mesh_glb_b64"])
        open(os.path.join(OUT, "relief_building.glb"), "wb").write(glb)
        print(f"[verify] mesh -> {OUT}/relief_building.glb ({len(glb)//1024} KB)")

    art_img = None
    if resp.get("art_png_b64"):
        from PIL import Image
        import io
        art_img = Image.open(io.BytesIO(base64.b64decode(resp["art_png_b64"])))
        art_img.save(os.path.join(OUT, "generated_art.png"))
        print(f"[verify] generated art -> {OUT}/generated_art.png")

    # visual check: depth/edge G-buffer, mask overlay, before/after cross-section slice
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    depth_img, edge_img, basis = tb.trace_view(grid96, cam, res=512)
    hit_rows, _ = np.where(basis["hit"])
    slice_y = int(np.median(hit_rows)) if len(hit_rows) else grid96.shape[1] // 2
    y_vox = int(np.clip(slice_y / 512 * grid96.shape[1], 0, grid96.shape[1] - 1))

    fig, axs = plt.subplots(2, 4, figsize=(19, 9))
    axs[0, 0].imshow(depth_img); axs[0, 0].set_title("depth G-buffer"); axs[0, 0].axis("off")
    axs[0, 1].imshow(edge_img); axs[0, 1].set_title("edge G-buffer"); axs[0, 1].axis("off")
    axs[0, 2].imshow(np.asarray(paint_pil.convert("RGB")))
    axs[0, 2].set_title("painted colors (input)"); axs[0, 2].axis("off")
    if art_img is not None:
        axs[0, 3].imshow(np.asarray(art_img)); axs[0, 3].set_title(f"generated art (strength {strength:.2f})")
    axs[0, 3].axis("off")
    axs[1, 0].imshow(grid96[:, y_vox, :] <= 0, cmap="gray")
    axs[1, 0].set_title(f"BEFORE occupancy (y-slice {y_vox})")
    axs[1, 1].imshow(out_grid[:, y_vox, :] <= 0, cmap="gray")
    axs[1, 1].set_title("AFTER occupancy (same slice)")
    diff = out_grid[:, y_vox, :] - grid96[:, y_vox, :]
    im = axs[1, 2].imshow(diff, cmap="coolwarm", vmin=-0.06, vmax=0.06)
    axs[1, 2].set_title("SDF delta (relief magnitude)")
    fig.colorbar(im, ax=axs[1, 2], fraction=0.046)
    axs[1, 3].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "paint_relief_preview.png"), dpi=110)
    print(f"[verify] preview -> {OUT}/paint_relief_preview.png")


if __name__ == "__main__":
    main()
