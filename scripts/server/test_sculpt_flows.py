"""SCULPT WORKFLOW SUITE — every flow a user can click in sculpt.html, asserted + photographed.

Mirrors the UI exactly (same payloads the JS sends). Quantitative checks per flow:
  F1  generate -> AI details -> snap (no masses)   base volume BIT-EXACT preserved
  F2  generate -> roof detail -> snap (no masses)  base preserved, roof ops untouched
  F3  generate -> place mass -> snap (localized)   base crisp OUTSIDE mask, edit adapted
  F4  place mass -> interpret (architecture) -> snap   constructions survive the snap
  F5  AI details -> re-cohere                       ops stay on the surface
  F6  bake with ONLY details                        massing fidelity kept (fp IoU), detailed mesh
  F7  GLOBAL re-mold opt-in                          actually changes the massing (sane occ)
  F8  town sculpt (refine_with_edit sdedit)          detailed mesh back
  F9  carve subtract -> interpret                    window/door typed
Prints PASS/FAIL table -> outputs/sculpt_flows/report_<UTC>.csv + montage PNG. Exit = #fails.

Run (server on :8099):
  env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python scripts/server/test_sculpt_flows.py
"""
from __future__ import annotations

import base64
import csv
import datetime
import json
import os
import sys
import time
import urllib.request

import numpy as np

URL = os.environ.get("SCULPT_URL", "http://127.0.0.1:8099")
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "outputs", "sculpt_flows")
RECT = [[-7, -9], [7, -9], [7, 9], [-7, 9]]
ROWS, PANELS = [], []


def post(path, body, timeout=900):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def vol(b64, res=64):
    return np.frombuffer(base64.b64decode(b64), dtype="<f4").reshape(res, res, res).copy()


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def case(name, fn, retry=False):
    """retry=True: one re-run on failure, for flows with a KNOWN environmental flake
    (decided 2026-07-07: F16 fails only under GPU pressure from the preceding
    texture-heavy flows in full-suite runs, never in isolation — rerun-once policy
    instead of tracking it as a bug). A retried pass is labeled so it stays visible."""
    t = time.time()
    for attempt in (0, 1):
        try:
            metric = fn()
            note = " (passed on retry)" if attempt else ""
            ROWS.append((name, "PASS", f"{metric}{note}", f"{time.time()-t:.1f}s"))
            print(f"  PASS  {name:44s} {metric}{note}")
            return
        except AssertionError as e:
            if retry and attempt == 0:
                print(f"  retry {name:44s} {e}")
                continue
            ROWS.append((name, "FAIL", str(e), f"{time.time()-t:.1f}s"))
            print(f"  FAIL  {name:44s} {e}")
            return
        except Exception as e:
            if retry and attempt == 0:
                print(f"  retry {name:44s} {type(e).__name__}: {str(e)[:80]}")
                continue
            ROWS.append((name, "ERROR", f"{type(e).__name__}: {str(e)[:90]}", f"{time.time()-t:.1f}s"))
            print(f"  ERROR {name:44s} {type(e).__name__}: {str(e)[:100]}")
            return


def panel(title, grid):
    PANELS.append((title, grid.copy()))


def main():
    os.makedirs(OUT, exist_ok=True)
    S = {}

    def fresh():
        r = post("/building_sdf", {"footprint": RECT, "style": "modern",
                                   "building_class": "RESIDENTIAL", "height": 16})
        return r, vol(r["sdf_b64"])

    # ---- F1: AI details then snap with NO mass edits -> base preserved ----------
    def f1():
        r, g0 = fresh()
        S["plain"] = r
        det = post("/propose_details", {"base_sdf_b64": r["sdf_b64"], "res": 64,
                                        "building_class": "RESIDENTIAL", "seed": 0})
        assert det["n"] >= 3, f"planner gave {det['n']} ops"
        snap = post("/snap_sdf", {"base_sdf_b64": r["sdf_b64"], "res": 64, "edits": [],
                                  "strength": 0.5, "local": True,
                                  "resnap_detail_ops": det["ops"]})
        g1 = vol(snap["sdf_b64"])
        dmax = float(np.abs(g1 - g0).max())
        assert dmax < 1e-5, f"base CHANGED under details-only snap (max dSDF={dmax:.4f})"
        kept = snap.get("resnapped_ops") or det["ops"]
        assert len(kept) == det["n"], f"detail ops lost ({len(kept)}/{det['n']})"
        panel("F1 details-only snap\nbase bit-exact", g1)
        S["det_ops"] = det["ops"]
        return f"base bit-exact (d={dmax:.1e}) · {det['n']} details kept"
    case("F1 AI details -> snap = base preserved", f1)

    # ---- F2: roof detail (UI roof_gable ops) then snap ---------------------------
    def f2():
        r = S["plain"]
        g0 = vol(r["sdf_b64"])
        h, scale, ctr = 16.0, r["scale"], r["center"]
        m = lambda v: v / scale
        cy = lambda c: (c - ctr[1]) / scale
        yCut = cy(0) + m(0.78 * h)
        roof = [
            {"kind": "box", "center": [0, yCut + m(h), 0], "size": [m(7), m(h), m(9)],
             "mode": "subtract", "smooth": 0, "det": "roof", "grp": "gR"},
            {"kind": "gable", "center": [0, yCut, 0], "size": [m(14.8), m(19), m(0.02), m(4.8)],
             "mode": "add", "smooth": 0, "det": "roof", "grp": "gR"},
        ]
        snap = post("/snap_sdf", {"base_sdf_b64": r["sdf_b64"], "res": 64, "edits": [],
                                  "strength": 0.5, "local": True,
                                  "resnap_detail_ops": roof})
        g1 = vol(snap["sdf_b64"])
        assert float(np.abs(g1 - g0).max()) < 1e-5, "base changed under roof+snap"
        ops = snap.get("resnapped_ops") or roof
        assert ops[0]["center"] == roof[0]["center"], "roof op MOVED by resnap"
        return "base bit-exact · roof ops untouched"
    case("F2 roof detail -> snap = base preserved", f2)

    # ---- F3: place a mass -> localized snap --------------------------------------
    def f3():
        r = S["plain"]
        g0 = vol(r["sdf_b64"])
        tower = {"kind": "box", "center": [0.45, 0.25, 0.0], "size": [0.14, 0.5, 0.14],
                 "mode": "add", "smooth": 0.0}
        snap = post("/snap_sdf", {"base_sdf_b64": r["sdf_b64"], "res": 64,
                                  "edits": [tower], "strength": 0.5, "local": True})
        g1 = vol(snap["sdf_b64"])
        # outside a generous box around the edit, the SDF must be (near) identical
        R, g = 64, np.linspace(-1, 1, 64)
        Z, Y, X = np.meshgrid(g, g, g, indexing="ij")
        far = (np.abs(X - 0.45) > 0.45) | (np.abs(Z) > 0.45)
        d_far = float(np.abs((g1 - g0)[far]).max())
        assert d_far < 0.02, f"base remolded outside the edit (d_far={d_far:.3f})"
        assert snap["iou_to_edit"] > 0.7, f"edit not kept (iou={snap['iou_to_edit']:.2f})"
        panel("F3 localized snap\nbase crisp, tower adapted", g1)
        return f"outside-mask d={d_far:.3f} · iou_to_edit={snap['iou_to_edit']:.2f}"
    case("F3 place mass -> localized snap", f3)

    # ---- F4: interpret -> snap (constructions survive) ---------------------------
    def f4():
        r = S["plain"]
        tall = {"kind": "box", "center": [0.45, 0.3, 0.0], "size": [0.1, 0.6, 0.1],
                "mode": "add", "smooth": 0.0}
        it = post("/interpret_mass", {"base_sdf_b64": r["sdf_b64"], "res": 64, "op": tall,
                                      "building_class": "RESIDENTIAL", "seed": 0})
        assert it["kind"] == "tower", f"typed '{it['kind']}'"
        mass_ops = [o for o in it["ops"] if not o.get("det")]
        det_ops = [o for o in it["ops"] if o.get("det")]
        snap = post("/snap_sdf", {"base_sdf_b64": r["sdf_b64"], "res": 64,
                                  "edits": mass_ops, "strength": 0.5, "local": True,
                                  "resnap_detail_ops": det_ops})
        g1 = vol(snap["sdf_b64"])
        # the tower shaft must still stand (occupied above the old roofline at its xz)
        gx = np.linspace(-1, 1, 64)
        xi = int((0.45 + 1) / 2 * 63); zi = int((0.0 + 1) / 2 * 63)
        col = g1[zi, :, xi] <= 0
        top_y = gx[np.where(col)[0].max()] if col.any() else -1
        assert top_y > 0.5, f"tower lost after snap (top_y={top_y:.2f})"
        panel("F4 interpret+snap\ntower survives", g1)
        return f"tower stands to y={top_y:.2f} · {len(det_ops)} windows kept"
    case("F4 make-architecture -> snap survives", f4)

    # ---- F5: AI details -> re-cohere ---------------------------------------------
    def f5():
        r = S["plain"]
        g0 = vol(r["sdf_b64"])
        rc = post("/recohere_details", {"base_sdf_b64": r["sdf_b64"], "res": 64,
                                        "ops": S["det_ops"]})
        vox = 2.0 / 63
        ds = []
        for o in rc["ops"]:
            cx, cy_, cz = o["center"]
            i = lambda v: int(np.clip((v + 1) / 2 * 63, 0, 63))
            ds.append(abs(float(g0[i(cz), i(cy_), i(cx)])) / vox)
        med = float(np.median(ds)) if ds else 0.0
        assert med < 5.0, f"re-cohered ops off surface (med={med:.1f} vox)"
        return f"{rc['n']} kept, {rc['dropped']} dropped · med {med:.1f} vox"
    case("F5 AI details -> re-cohere on-surface", f5)

    # ---- F6: bake with ONLY details — massing fidelity ---------------------------
    def f6():
        r = S["plain"]
        g0 = vol(r["sdf_b64"])
        bake = post("/snap_sdf", {"base_sdf_b64": r["sdf_b64"], "res": 64, "edits": [],
                                  "strength": 0.5, "local": True, "return_mesh": True,
                                  "center": r["center"], "scale": r["scale"],
                                  "detail": True, "building_class": "RESIDENTIAL",
                                  "style": "modern", "detail_edits": S["det_ops"][:8]})
        assert bake["mesh_glb_b64"], "no mesh"
        g1 = vol(bake["sdf_b64"])
        fp_keep = iou((g1 <= 0).any(1), (g0 <= 0).any(1))
        assert fp_keep > 0.97, f"bake changed the footprint (IoU={fp_keep:.2f})"
        import io
        import trimesh
        m = trimesh.load(io.BytesIO(base64.b64decode(bake["mesh_glb_b64"])), file_type="glb")
        nv = sum(len(gg.vertices) for gg in m.geometry.values())
        assert nv > 1000, f"verts={nv}"
        return f"fp IoU={fp_keep:.3f} · mesh verts={nv}"
    case("F6 bake (details only) keeps massing", f6)

    # ---- F7: GLOBAL re-mold opt-in actually remolds -------------------------------
    def f7():
        r = S["plain"]
        g0 = vol(r["sdf_b64"])
        snap = post("/snap_sdf", {"base_sdf_b64": r["sdf_b64"], "res": 64, "edits": [],
                                  "strength": 0.5, "local": False})
        g1 = vol(snap["sdf_b64"])
        ch = iou(g1 <= 0, g0 <= 0)
        occ = float((g1 <= 0).mean())
        assert ch < 0.999, "global remold did nothing"
        assert 0.03 < occ < 0.6, f"global remold degenerate (occ={occ:.2f})"
        panel("F7 global re-mold (opt-in)", g1)
        return f"changed (IoU vs base {ch:.2f}, occ {occ:.2f})"
    case("F7 global re-mold is opt-in + sane", f7)

    # ---- F8: town sculpt path ------------------------------------------------------
    def f8():
        edit = {"kind": "box", "center": [5.5, 8.0, 0.0], "size": [1.5, 8.0, 1.5],
                "mode": "add", "smooth": 0.0}
        r = post("/refine_with_edit", {"base_style": "modern",
                                       "base_recipe_params": S["plain"]["recipe_params"],
                                       "footprint": RECT, "height": 16, "edits": [edit],
                                       "mode": "sdedit", "strength": 0.5})
        import io
        import trimesh
        m = trimesh.load(io.BytesIO(base64.b64decode(r["mesh_glb_b64"])), file_type="glb")
        nv = sum(len(gg.vertices) for gg in m.geometry.values())
        assert nv > 4000, f"town sculpt mesh too plain (verts={nv} — detail layer missing?)"
        return f"detailed mesh verts={nv}"
    case("F8 town sculpt returns detailed mesh", f8)

    # ---- F9: carve -> interpreted as window/door ----------------------------------
    def f9():
        carve = {"kind": "box", "center": [0.0, 0.1, 0.7], "size": [0.05, 0.05, 0.05],
                 "mode": "subtract", "smooth": 0.0}
        it = post("/interpret_mass", {"base_sdf_b64": S["plain"]["sdf_b64"], "res": 64,
                                      "op": carve, "seed": 0})
        assert it["kind"] in ("window", "door"), f"carve typed '{it['kind']}'"
        return f"carve -> {it['kind']}"
    case("F9 carve interprets as window/door", f9)

    # ---- F10: live detail preview = bake-quality volume ----------------------------
    def f10():
        r = S["plain"]
        g0 = vol(r["sdf_b64"])
        pv = post("/detail_volume", {"base_sdf_b64": r["sdf_b64"], "res": 64,
                                     "center": r["center"], "scale": r["scale"],
                                     "building_class": "RESIDENTIAL", "style": "modern"})
        g1 = vol(pv["sdf_b64"], pv["res"])
        assert pv["res"] == 96, f"res={pv['res']}"
        # detail must ADD surface complexity (windows/bands) vs the plain massing
        from skimage import measure
        v0, _f0, _, _ = measure.marching_cubes(g0.astype(np.float32), level=0.0)
        v1, _f1, _, _ = measure.marching_cubes(g1.astype(np.float32), level=0.0)
        density0 = len(v0) / 64 ** 2
        density1 = len(v1) / 96 ** 2
        assert density1 > density0 * 1.15, \
            f"preview no richer than massing (density {density1:.2f} vs {density0:.2f})"
        panel("F10 detail preview\n(bake quality, live)", g1)
        return f"96^3 · surface density x{density1 / density0:.2f}"
    case("F10 live detail preview is bake-quality", f10)

    # ---- F11: details get ADJUSTED (re-cohered) after a wall-moving snap ------------
    def f11():
        r = S["plain"]
        det = post("/propose_details", {"base_sdf_b64": r["sdf_b64"], "res": 64,
                                        "building_class": "RESIDENTIAL", "seed": 1})
        snap = post("/snap_sdf", {"base_sdf_b64": r["sdf_b64"], "res": 64, "edits": [],
                                  "strength": 0.5, "local": False, "adjust": True,
                                  "resnap_detail_ops": det["ops"]})
        g1 = vol(snap["sdf_b64"])
        ops = snap.get("resnapped_ops") or []
        assert ops, "no adjusted ops back"
        vox = 2.0 / 63
        i = lambda v: int(np.clip((v + 1) / 2 * 63, 0, 63))
        wall = [o for o in ops if o.get("det") in ("window", "door", "balcony", "column")]
        ds = [abs(float(g1[i(o["center"][2]), i(o["center"][1]), i(o["center"][0])])) / vox
              for o in wall]
        med = float(np.median(ds)) if ds else 0.0
        assert med < 3.0, f"adjusted details off the NEW surface (med={med:.1f} vox)"
        return f"{len(ops)}/{det['n']} ops adjusted onto new walls · med {med:.1f} vox"
    case("F11 details re-cohere after wall-moving snap", f11)

    # ---- F12: photoreal neural render (style picker backend) ------------------------
    def f12():
        r = S["plain"]
        out = post("/neural_render", {"base_sdf_b64": r["sdf_b64"], "res": 64,
                                      "center": r["center"], "scale": r["scale"],
                                      "building_class": "RESIDENTIAL", "style": "modern",
                                      "steps": 14}, timeout=1200)
        img = base64.b64decode(out["image_b64"])
        assert len(img) > 100_000, f"render too small ({len(img)} bytes)"
        return f"photoreal png {len(img) // 1024}KB (first call incl. SDXL load)"
    case("F12 photoreal render endpoint", f12)

    # ---- F13: photoreal TOWN with a per-building style ref ---------------------------
    def f13():
        from PIL import Image as _Im
        import io as _io
        ref = _Im.new("RGB", (256, 256), (150, 60, 40))     # brick-ish color field
        buf = _io.BytesIO(); ref.save(buf, "PNG")
        ref_b64 = base64.b64encode(buf.getvalue()).decode()
        r = S["plain"]
        bld = {"footprint": RECT, "style": "modern", "building_class": "RESIDENTIAL",
               "height": 16, "recipe_params": r["recipe_params"], "edits": []}
        out = post("/neural_render_town", {"buildings": [
            {**bld, "position": [-12, 0], "style_ref_b64": ref_b64},
            {**bld, "position": [12, 0]}], "steps": 12}, timeout=1200)
        img = base64.b64decode(out["image_b64"])
        assert out["n_buildings"] == 2 and len(img) > 100_000, \
            f"town render thin ({out['n_buildings']} bldgs, {len(img)}B)"
        return f"2 bldgs (1 styled) · png {len(img) // 1024}KB"
    case("F13 photoreal town (per-building style)", f13)

    # ---- F14: export town as glTF scene for Unreal ----------------------------------
    def f14():
        import io as _io
        import trimesh
        r = S["plain"]
        bld = {"footprint": RECT, "style": "modern", "building_class": "RESIDENTIAL",
               "height": 16, "recipe_params": r["recipe_params"], "edits": []}
        out = post("/export_town", {"buildings": [
            {**bld, "position": [-12, 0]}, {**bld, "position": [12, 0]}], "scale": 100.0})
        assert out["manifest"]["n_buildings"] == 2, out["manifest"]["n_buildings"]
        glb = base64.b64decode(out["glb_b64"])
        sc = trimesh.load(_io.BytesIO(glb), file_type="glb")
        geoms = len(sc.geometry)
        assert geoms >= 3, f"expected 2 buildings + ground, got {geoms} nodes"
        ext = (sc.bounds[1] - sc.bounds[0]).max()
        assert ext > 1000, f"export not in cm (max extent {ext:.0f})"
        names = list(sc.geometry.keys())
        assert any("bldg_" in n for n in names), f"no named building nodes: {names}"
        return f"{geoms} nodes ({out['n_vertices'] // 1000}k verts) · {ext:.0f}cm span"
    case("F14 export town glTF for Unreal", f14)

    # ---- F15: v2 texture bake -> textured glb (UV + albedo) -------------------------
    def f15():
        import io as _io
        import trimesh
        r = S["plain"]
        out = post("/bake_texture", {"base_sdf_b64": r["sdf_b64"], "res": 64,
                                     "center": r["center"], "scale": r["scale"],
                                     "building_class": "RESIDENTIAL", "style": "modern",
                                     "n_views": 3, "steps": 10}, timeout=1800)
        assert out["coverage"] > 0.5, f"texture coverage low ({out['coverage']})"
        glb = base64.b64decode(out["glb_b64"])
        m = trimesh.load(_io.BytesIO(glb), file_type="glb", process=False)
        g = m if isinstance(m, trimesh.Trimesh) else list(m.geometry.values())[0]
        has_uv = getattr(g.visual, "uv", None) is not None
        mat = getattr(g.visual, "material", None)
        has_tex = getattr(mat, "baseColorTexture", None) is not None
        has_pbr = getattr(mat, "normalTexture", None) is not None \
            and getattr(mat, "metallicRoughnessTexture", None) is not None
        assert has_uv and has_tex, f"glb missing UV/albedo (uv={has_uv} tex={has_tex})"
        assert has_pbr, "glb missing PBR maps (normal / metallic-roughness)"
        return f"textured PBR glb · cov {out['coverage']} · {out['n_vertices'] // 1000}k verts · albedo+normal+MR"
    case("F15 v2 texture bake -> textured glb", f15)

    # ---- F16: textured TOWN export (per-building albedo in one scene) ----------------
    def f16():
        import io as _io
        import trimesh
        r = S["plain"]
        bld = {"footprint": RECT, "style": "modern", "building_class": "RESIDENTIAL",
               "height": 16, "recipe_params": r["recipe_params"], "edits": []}
        out = post("/export_town", {"buildings": [{**bld, "position": [-12, 0]},
                                                  {**bld, "position": [12, 0]}],
                                    "scale": 100.0, "textures": True,
                                    "n_views": 3, "steps": 10}, timeout=2400)
        assert out["manifest"]["n_buildings"] == 2, out["manifest"]["n_buildings"]
        glb = base64.b64decode(out["glb_b64"])
        sc = trimesh.load(_io.BytesIO(glb), file_type="glb", process=False)
        textured = [n for n, g in sc.geometry.items()
                    if n.startswith("bldg_") and (getattr(g.visual, "uv", None) is not None)]
        assert len(textured) == 2, f"expected 2 textured buildings, got {textured}"
        covs = [b["texture_coverage"] for b in out["manifest"]["buildings"]]
        assert min(covs) > 0.5, f"low coverage {covs}"
        return f"2 textured bldgs in 1 scene · cov {covs} · {out['n_vertices'] // 1000}k verts"
    case("F16 textured town export (per-building)", f16, retry=True)

    # ---- F17: cleanup removes floating debris from a snap ---------------------------
    def f17():
        from scipy.ndimage import label
        r = S["plain"]
        # a tiny mass placed far in the corner -> disconnected speck; cleanup should drop it
        floater = {"kind": "box", "center": [0.92, 0.92, 0.92], "size": [0.04, 0.04, 0.04],
                   "mode": "add", "smooth": 0.0}
        snap = post("/snap_sdf", {"base_sdf_b64": r["sdf_b64"], "res": 64,
                                  "edits": [floater], "strength": 0.5, "local": True})
        g1 = vol(snap["sdf_b64"])
        n = label(g1 <= 0)[1]
        assert n <= 1, f"floating debris not cleaned ({n} components after snap)"
        return f"{n} component (floater removed)"
    case("F17 cleanup removes floating debris", f17)

    # ---- F18: sketch relief -> real, LOCAL sculpted geometry -------------------------
    def f18():
        import io as _io
        from PIL import Image, ImageDraw
        from scipy.ndimage import zoom
        r = S["plain"]
        # the "before": same detailed volume /paint_relief derives internally
        # (composer seed pinned to 0 on both sides so facade decoration is identical)
        dv = post("/detail_volume", {"base_sdf_b64": r["sdf_b64"], "res": 64,
                                     "center": r["center"], "scale": r["scale"],
                                     "building_class": "RESIDENTIAL", "style": "modern",
                                     "seed": 0, "res_out": 96})
        before96 = vol(dv["sdf_b64"], 96)
        # synthetic stroke: a filled circle mid-view, camera facing the +z wall head-on
        img = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        ImageDraw.Draw(img).ellipse([196, 240, 336, 360], fill=(176, 137, 104, 255))
        buf = _io.BytesIO(); img.save(buf, format="PNG")
        out = post("/paint_relief", {
            "base_sdf_b64": r["sdf_b64"], "res": 64, "center": r["center"],
            "scale": r["scale"], "building_class": "RESIDENTIAL", "style": "modern",
            "cam": {"pos": [0.0, 0.1, 2.2], "look": [0.0, 0.0, 0.0], "fov": 40.0},
            "paint_png_b64": base64.b64encode(buf.getvalue()).decode(),
            "seed": 5, "steps": 12, "composer_seed": 0, "return_mesh": False}, timeout=1800)
        assert out.get("art_png_b64"), "no generated art returned"
        g1 = vol(out["sdf_b64"], out["res"])
        assert out["res"] == 128, f"expected out_res default 128, got {out['res']}"
        up = zoom(before96, out["res"] / 96.0, order=1)          # trilinear before at 128
        diff = np.abs(g1 - up)
        assert float(diff.max()) > 0.05, f"no real relief (max dSDF={diff.max():.4f})"
        # locality: change must sit on the painted (+z) wall — volume layout is (z,y,x)
        zz = np.where(diff > 0.04)[0]
        assert len(zz) > 100, f"relief too small ({len(zz)} voxels over threshold)"
        zc = float(zz.mean()) / out["res"]
        assert zc > 0.55, f"relief not on the painted +z wall (z-centroid {zc:.2f})"
        far = float(diff[: out["res"] // 3].max())               # -z half must stay intact
        assert far < 0.03, f"relief leaked to the far side (max dSDF {far:.3f})"
        panel("F18 sketch relief\nlocal sculpted geometry", g1)
        return f"relief on +z wall (zc={zc:.2f}) · {len(zz)} voxels · far side clean ({far:.3f})"
    case("F18 sketch relief -> local sculpted geometry", f18)

    # ---- report -------------------------------------------------------------------
    stamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    with open(os.path.join(OUT, f"report_{stamp}.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["flow", "status", "metric", "secs"]); w.writerows(ROWS)
    fails = sum(1 for r in ROWS if r[1] != "PASS")

    if PANELS:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from skimage import measure
        n = len(PANELS)
        fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.4), subplot_kw={"projection": "3d"})
        axes = np.atleast_1d(axes)
        for ax, (title, gd) in zip(axes, PANELS):
            ax.set_title(title, fontsize=8)
            if (gd <= 0).sum() > 8:
                v, fc, _, _ = measure.marching_cubes(gd.astype(np.float32), level=0.0)
                v = v[:, [2, 1, 0]]
                ax.plot_trisurf(v[:, 0], v[:, 2], fc, v[:, 1], color="#cdb892",
                                edgecolor="none", antialiased=True, shade=True)
                lo, hi = v.min(), v.max()
                ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_zlim(lo, hi)
            ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(elev=16, azim=-55)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"flows_{stamp}.png"), dpi=110)
    print(f"\n== {len(ROWS) - fails}/{len(ROWS)} PASS ==  outputs/sculpt_flows/report_{stamp}.csv")
    sys.exit(fails)


if __name__ == "__main__":
    main()
