"""VISUAL branch eval — every user-facing pipeline path rendered into ONE review photo.

Companion to test_branches.py (which asserts metrics): this renders what each branch
actually PRODUCES so a human can review quality at a glance. Fast: re-uses the running
server (warm calls are sub-second; only sdedit paths take ~1s each).

Panels: plain massing · AI massing (BAG prior) · snap (tower edit) · AI-detail ops ·
bake @96 · town building · town sculpt · learned re-cohere · image->town (mask + scene).

Run (server up on :8099):
  env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python scripts/server/eval_visual.py
Output: outputs/eval_visual/pipeline_<UTC>.png
"""
from __future__ import annotations

import base64
import datetime
import io
import json
import os
import time
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

URL = os.environ.get("SCULPT_URL", "http://127.0.0.1:8099")
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "outputs", "eval_visual")
RECT = [[-7, -9], [7, -9], [7, 9], [-7, 9]]
FACE = "#cdb892"


def post(path, body, timeout=600):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def vol(b64, res=64):
    return np.frombuffer(base64.b64decode(b64), dtype="<f4").reshape(res, res, res).copy()


def sdf_trimesh(grid):
    """64^3 SDF (D=z,H=y,W=x) -> verts/faces in (x, y_up, z) order via marching cubes."""
    from skimage import measure
    if (grid <= 0).sum() < 8:
        return None, None
    v, f, _, _ = measure.marching_cubes(grid, level=0.0)
    return v[:, [2, 1, 0]], f          # (z,y,x) index -> (x,y,z)


def glb_trimesh(b64):
    import trimesh
    m = trimesh.load(io.BytesIO(base64.b64decode(b64)), file_type="glb")
    parts = [g for g in m.geometry.values()] if hasattr(m, "geometry") else [m]
    c = trimesh.util.concatenate(parts)
    return np.asarray(c.vertices), np.asarray(c.faces)


def draw_mesh(ax, v, f, title, color=FACE):
    ax.set_title(title, fontsize=8)
    if v is not None and len(f):
        # world/local is y-up: plot x, z on the ground plane, y as height
        ax.plot_trisurf(v[:, 0], v[:, 2], f, v[:, 1], color=color,
                        edgecolor="none", linewidth=0, antialiased=True, shade=True)
        lo, hi = v.min(), v.max()
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_zlim(lo, hi)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(elev=22, azim=-58)


def cube_to_idx(c, R=64):
    return [(np.clip(x, -1, 1) + 1) * 0.5 * (R - 1) for x in c]


def main():
    os.makedirs(OUT, exist_ok=True)
    t_all = time.time()
    panels = []     # (kind, payload, title)
    timing = {}

    def call(name, fn):
        t = time.time()
        out = fn()
        timing[name] = time.time() - t
        print(f"  {name:28s} {timing[name]:.1f}s")
        return out

    # ---- branch calls (same contracts as test_branches.py) -----------------------
    plain = call("plain massing", lambda: post("/building_sdf", {
        "footprint": RECT, "style": "modern", "building_class": "RESIDENTIAL", "height": 16}))
    g_plain = vol(plain["sdf_b64"])

    ai = call("AI massing (sdedit)", lambda: post("/building_sdf", {
        "footprint": RECT, "style": "modern", "building_class": "RESIDENTIAL", "height": 16,
        "sdedit_strength": 0.45}))
    g_ai = vol(ai["sdf_b64"])
    o, o0 = g_ai <= 0, g_plain <= 0
    iou_ai = (o & o0).sum() / max((o | o0).sum(), 1)

    tower = {"kind": "box", "center": [0.45, 0.25, 0.0], "size": [0.14, 0.5, 0.14],
             "mode": "add", "smooth": 0.0}
    snap = call("snap (tower edit)", lambda: post("/snap_sdf", {
        "base_sdf_b64": plain["sdf_b64"], "res": 64, "edits": [tower], "strength": 0.5}))
    g_snap = vol(snap["sdf_b64"])

    # NOTE: the standalone ops-preview endpoint this panel originally called
    # (`/propose_details`) no longer exists on the live server -- `/detail_volume` is what
    # the current sculpt.html frontend actually calls for the "AI detail preview" button
    # (scripts/server/web/sculpt.html's `detPrev` handler). It composes the bake-quality
    # detail treatment directly into a volume rather than returning a flat ops list, so this
    # panel now renders that composed volume instead of scattering op-type dots.
    det = call("AI details (detail_volume)", lambda: post("/detail_volume", {
        "base_sdf_b64": plain["sdf_b64"], "res": 64, "center": plain["center"],
        "scale": plain["scale"], "building_class": "RELIGIOUS", "style": "modern",
        "detail_edits": []}))
    g_det = vol(det["sdf_b64"], res=det["res"])

    bake = call("bake @96", lambda: post("/snap_sdf", {
        "base_sdf_b64": plain["sdf_b64"], "res": 64, "edits": [], "strength": 0.5,
        "return_mesh": True, "center": plain["center"], "scale": plain["scale"],
        "detail": True, "building_class": "RESIDENTIAL", "style": "modern",
        "detail_edits": []}))

    town = call("town building", lambda: post("/regenerate_building", {
        "footprint": RECT, "style": "victorian", "building_class": "RESIDENTIAL",
        "height": 12, "seed": 3, "detail": True}))

    sculpt_edit = {"kind": "box", "center": [5.5, 8.0, 0.0], "size": [1.5, 8.0, 1.5],
                   "mode": "add", "smooth": 0.0}
    sculpt = call("town sculpt", lambda: post("/refine_with_edit", {
        "base_style": "modern", "base_recipe_params": plain["recipe_params"],
        "footprint": RECT, "height": 16, "edits": [sculpt_edit], "mode": "sdedit",
        "strength": 0.5}))

    junk = {"kind": "box", "center": [0.93, 0.93, 0.93], "size": [0.05, 0.07, 0.05],
            "mode": "subtract", "smooth": 0.0, "det": "window", "grp": "gJ"}
    keep = {"kind": "box", "center": [0.0, 0.0, 0.7], "size": [0.05, 0.06, 0.04],
            "mode": "subtract", "smooth": 0.0, "det": "window", "grp": "gK"}
    reco = call("learned re-cohere", lambda: post("/recohere_details", {
        "base_sdf_b64": plain["sdf_b64"], "res": 64, "ops": [junk, keep]}))

    from PIL import Image, ImageDraw
    im = Image.new("L", (256, 256), 255)
    d = ImageDraw.Draw(im)
    for xy in [(30, 30, 90, 100), (130, 50, 210, 110), (60, 150, 160, 220)]:
        d.rectangle(xy, fill=0)
    buf = io.BytesIO(); im.save(buf, "PNG")
    img_town = call("image -> town", lambda: post("/generate_from_image", {
        "image_b64": base64.b64encode(buf.getvalue()).decode(),
        "meters_across": 120, "max_buildings": 6}))

    # ---- figure -------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 13))
    grid = (3, 4)

    def ax3(i):
        return fig.add_subplot(*grid, i, projection="3d")

    v, f = sdf_trimesh(g_plain)
    draw_mesh(ax3(1), v, f, f"1· plain massing (recipe)\n{timing['plain massing']:.1f}s")

    v, f = sdf_trimesh(g_ai)
    draw_mesh(ax3(2), v, f,
              f"2· AI massing (BAG prior, s=0.45)\niou_vs_plain={iou_ai:.2f} · "
              f"{timing['AI massing (sdedit)']:.1f}s")

    v, f = sdf_trimesh(g_snap)
    ax = ax3(3)
    draw_mesh(ax, v, f, f"3· snap: tower edit (s=0.5)\niou_to_edit={snap['iou_to_edit']:.2f} · "
                        f"{timing['snap (tower edit)']:.1f}s")
    cx, cy, cz = cube_to_idx(tower["center"])
    ax.scatter([cx], [cz], [cy], color="crimson", s=40, marker="^")

    # AI details: the composed bake-quality detail volume (windows/bands/plinth/roof/
    # landmarks), at its own res_out resolution -- what /detail_volume actually returns now.
    v, f = sdf_trimesh(g_det)
    draw_mesh(ax3(4), v, f, f"4· AI details: composed volume ({det['res']}³)\n"
                            f"{timing['AI details (detail_volume)']:.1f}s")

    v, f = glb_trimesh(bake["mesh_glb_b64"])
    draw_mesh(ax3(5), v, f, f"5· bake @96 (snap + details)\nverts={len(v)} · "
                            f"{timing['bake @96']:.1f}s")

    v, f = glb_trimesh(town["mesh_glb_b64"])
    draw_mesh(ax3(6), v, f, f"6· town building (victorian+detail)\nverts={len(v)} · "
                            f"{timing['town building']:.1f}s")

    v, f = glb_trimesh(sculpt["mesh_glb_b64"])
    draw_mesh(ax3(7), v, f, f"7· town sculpt (sdedit, s=0.5)\nverts={len(v)} · "
                            f"{timing['town sculpt']:.1f}s")

    # re-cohere: junk (x) vs re-seated/kept (o)
    ax = ax3(8)
    v, f = sdf_trimesh(g_plain)
    title = (f"8· learned re-cohere\n{reco['dropped']} dropped · {reco['n']} kept · "
             f"{timing['learned re-cohere']:.1f}s")
    draw_mesh(ax, v, f, title)
    for op, mark, col in [(junk, "x", "crimson"), (keep, "s", "#444444")]:
        x, y, z = cube_to_idx(op["center"])
        ax.scatter([x], [z], [y], color=col, s=42, marker=mark)
    for op in reco["ops"]:
        x, y, z = cube_to_idx(op["center"])
        ax.scatter([x], [z], [y], color="#2ca02c", s=42, marker="o")

    ax = fig.add_subplot(*grid, 9)
    ax.imshow(im, cmap="gray")
    ax.set_title("9· image->town: input mask", fontsize=8)
    ax.set_axis_off()

    ax = ax3(10)
    ax.set_title(f"10· image->town: {img_town['n_buildings']} buildings\n"
                 f"{timing['image -> town']:.1f}s", fontsize=8)
    allv = []
    for b in img_town["buildings"]:
        v, f = glb_trimesh(b["glb_b64"])
        v = v + np.array([b["position"][0], 0.0, b["position"][1]])
        ax.plot_trisurf(v[:, 0], v[:, 2], f, v[:, 1], color=FACE,
                        edgecolor="none", linewidth=0, antialiased=True, shade=True)
        allv.append(v)
    if allv:
        allv = np.concatenate(allv)
        lo, hi = allv.min(), allv.max()
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_zlim(lo, hi)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(elev=35, azim=-58)

    stamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    fig.suptitle(f"pipeline visual eval · {stamp} · total {time.time()-t_all:.0f}s",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    png = os.path.join(OUT, f"pipeline_{stamp}.png")
    fig.savefig(png, dpi=110)
    print(f"\n[eval_visual] -> {png}  ({time.time()-t_all:.0f}s total)")


if __name__ == "__main__":
    main()
