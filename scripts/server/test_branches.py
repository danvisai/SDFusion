"""BRANCH TEST SUITE — every user-facing path, asserted, in one command.

The project now has many branches (generate / AI-massing / sculpt / snap / details / resnap /
bake / town / image). This exercises each against a RUNNING server with metric assertions and
prints a PASS/FAIL table -> outputs/branch_tests/report_<UTC>.csv. Exit code = number of fails.

Run (server up on :8099):
  env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python scripts/server/test_branches.py
"""
from __future__ import annotations

import base64
import csv
import datetime
import io
import json
import os
import sys
import time
import urllib.request

import numpy as np

URL = os.environ.get("SCULPT_URL", "http://127.0.0.1:8099")
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "outputs", "branch_tests")
RECT = [[-7, -9], [7, -9], [7, 9], [-7, 9]]
ROWS = []


def post(path, body, timeout=900):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def get(path):
    return json.loads(urllib.request.urlopen(URL + path, timeout=60).read())


def vol(b64, res=64):
    return np.frombuffer(base64.b64decode(b64), dtype="<f4").reshape(res, res, res)


def sdf_at(grid, c):
    """nearest-voxel SDF value at cube coords c=(x,y,z); grid layout (D=z,H=y,W=x)."""
    R = grid.shape[0]
    i = lambda v: int(np.clip((v + 1) * 0.5 * (R - 1), 0, R - 1))
    return float(grid[i(c[2]), i(c[1]), i(c[0])])


def glb_verts(b64):
    import trimesh
    m = trimesh.load(io.BytesIO(base64.b64decode(b64)), file_type="glb")
    return sum(len(g.vertices) for g in m.geometry.values())


def case(name, fn):
    t = time.time()
    try:
        metric = fn()
        ROWS.append((name, "PASS", metric, f"{time.time()-t:.1f}s"))
        print(f"  PASS  {name:34s} {metric}")
    except AssertionError as e:
        ROWS.append((name, "FAIL", str(e), f"{time.time()-t:.1f}s"))
        print(f"  FAIL  {name:34s} {e}")
    except Exception as e:
        ROWS.append((name, "ERROR", f"{type(e).__name__}: {str(e)[:90]}", f"{time.time()-t:.1f}s"))
        print(f"  ERROR {name:34s} {type(e).__name__}: {str(e)[:90]}")


def main():
    os.makedirs(OUT, exist_ok=True)
    S = {}

    def manual_details(r):
        """Hand-built window/door detail ops on the RECT walls — the gates' detail-op
        source since /propose_details (the removed AI-detailing feature) is gone."""
        scale, ctr = r["scale"], r["center"]
        m = lambda v: v / scale
        cy = lambda c: (c - ctr[1]) / scale
        ops = [dict(kind="box", center=[m(7.0), cy(6.0), m(z)],
                    size=[m(0.45), m(0.8), m(0.5)], mode="subtract", smooth=0.0,
                    det="window") for z in (-6.0, -3.0, 0.0, 3.0, 6.0)]
        ops.append(dict(kind="box", center=[m(0.0), cy(1.1), m(9.0)],
                        size=[m(0.6), m(1.1), m(0.4)], mode="subtract", smooth=0.0,
                        det="door"))
        return ops

    def b1():
        h = get("/health")
        assert h["status"] == "ok", h
        return f"device={h['device']}"
    case("B1 health", b1)

    def b2():
        r = post("/building_sdf", {"footprint": RECT, "style": "modern",
                                   "building_class": "RESIDENTIAL", "height": 16})
        g = vol(r["sdf_b64"])
        occ = float((g <= 0).mean())
        assert 0.03 < occ < 0.6, f"occ%={occ:.3f} out of bounds"
        assert r["scale"] > 1, f"scale={r['scale']}"
        S["plain"] = r
        return f"occ={occ:.2f} scale={r['scale']:.1f}"
    case("B2 generate massing (plain)", b2)

    def b3():
        # don't bill the prior's ~15GB x2 ckpt load to this branch: the server pre-warms it
        # at startup; wait for readiness (old servers without the flag fall through at once)
        t0 = time.time()
        while time.time() - t0 < 1500:
            if get("/health").get("sdedit_ready", True):
                break
            print("        ... waiting for snap-prior warmup")
            time.sleep(15)
        r = post("/building_sdf", {"footprint": RECT, "style": "modern",
                                   "building_class": "RESIDENTIAL", "height": 16,
                                   "sdedit_strength": 0.45})
        g, g0 = vol(r["sdf_b64"]) <= 0, vol(S["plain"]["sdf_b64"]) <= 0
        u = (g | g0).sum(); i = (g & g0).sum()
        iou = i / max(u, 1)
        assert 0.03 < g.mean() < 0.6, f"occ={g.mean():.3f}"
        assert iou > 0.5, f"AI massing strayed too far from footprint massing (iou={iou:.2f})"
        S["ai"] = r
        return f"iou_vs_plain={iou:.2f}"
    case("B3 generate + AI massing (BAG prior)", b3)

    def b4():
        tower = {"kind": "box", "center": [0.45, 0.25, 0.0], "size": [0.14, 0.5, 0.14],
                 "mode": "add", "smooth": 0.0}
        r = post("/snap_sdf", {"base_sdf_b64": S["plain"]["sdf_b64"], "res": 64,
                               "edits": [tower], "strength": 0.5})
        assert r["iou_to_edit"] > 0.7, f"iou_to_edit={r['iou_to_edit']:.2f}"
        S["snapped"] = r
        return f"iou_to_edit={r['iou_to_edit']:.2f}"
    case("B4 snap (massing edit)", b4)

    def b5():
        # the balcony bug regression: tagged details must sit ON the NEW surface after a snap,
        # and UNTAGGED ops must come back UNTOUCHED (guessing once teleported balconies to roofs)
        s = S["plain"]["scale"]
        balc = {"kind": "box", "center": [0.0, 0.1, 0.72], "size": [1.6 / s, 0.1 / s, 0.9 / s],
                "mode": "add", "smooth": 0.0, "det": "balcony"}
        win = {"kind": "box", "center": [-0.25, 0.15, 0.7], "size": [0.6 / s, 0.7 / s, 0.3 / s],
               "mode": "subtract", "smooth": 0.0, "det": "window"}
        untagged = {"kind": "box", "center": [0.3, 0.2, 0.6], "size": [0.1, 0.1, 0.1],
                    "mode": "add", "smooth": 0.0}
        # local=False: this branch REGRESSION-TESTS detail re-snap after a wall-moving
        # global snap (with local=True an empty edit list is now a no-op by design)
        r = post("/snap_sdf", {"base_sdf_b64": S["plain"]["sdf_b64"], "res": 64, "edits": [],
                               "strength": 0.5, "local": False, "adjust": False,
                               "resnap_detail_ops": [balc, win, untagged]})
        gN = vol(r["sdf_b64"])
        ops = r.get("resnapped_ops")
        assert ops and len(ops) == 3, "resnapped ops missing"
        vox = 2.0 / 63
        ds = [abs(sdf_at(gN, o["center"])) / vox for o in ops[:2]]
        assert max(ds) < 4.0, f"tagged op {max(ds):.1f} voxels off the new surface"
        assert ops[2]["center"] == untagged["center"], f"untagged op MOVED: {ops[2]['center']}"
        return f"tagged on-surface {['%.1f' % d for d in ds]} vox · untagged untouched"
    case("B5 details RE-SNAP after snap (bug regression)", b5)


    def b7():
        r = post("/snap_sdf", {"base_sdf_b64": S["plain"]["sdf_b64"], "res": 64, "edits": [],
                               "strength": 0.5, "return_mesh": True,
                               "center": S["plain"]["center"], "scale": S["plain"]["scale"],
                               "detail": True, "building_class": "RESIDENTIAL", "style": "modern",
                               "detail_edits": manual_details(S["plain"])})
        assert r["mesh_glb_b64"], "no mesh"
        nv = glb_verts(r["mesh_glb_b64"])
        assert nv > 1000, f"verts={nv}"
        return f"baked mesh verts={nv}"
    case("B7 bake (snap + details @96)", b7)

    def b8():
        r = post("/regenerate_building", {"footprint": RECT, "style": "victorian",
                                          "building_class": "RESIDENTIAL", "height": 12,
                                          "seed": 3, "detail": True})
        nv = glb_verts(r["mesh_glb_b64"])
        assert nv > 2000, f"verts={nv}"
        return f"town-path detailed mesh verts={nv}"
    case("B8 town building (recipe+composer detail)", b8)

    def b9():
        edit = {"kind": "box", "center": [5.5, 8.0, 0.0], "size": [1.5, 8.0, 1.5],
                "mode": "add", "smooth": 0.0}
        r = post("/refine_with_edit", {"base_style": "modern", "base_recipe_params":
                                       S["plain"]["recipe_params"], "footprint": RECT,
                                       "height": 16, "edits": [edit], "mode": "sdedit",
                                       "strength": 0.5})
        nv = glb_verts(r["mesh_glb_b64"])
        assert nv > 1000, f"verts={nv}"
        return f"town sculpt verts={nv}"
    case("B9 town sculpt (refine_with_edit sdedit)", b9)

    def b11():
        # learned re-coherence: a junk floating part should be dropped or pulled to the building
        junk = {"kind": "box", "center": [0.93, 0.93, 0.93], "size": [0.05, 0.07, 0.05],
                "mode": "subtract", "smooth": 0.0, "det": "window", "grp": "gJ"}
        keep = {"kind": "box", "center": [0.0, 0.0, 0.7], "size": [0.05, 0.06, 0.04],
                "mode": "subtract", "smooth": 0.0, "det": "window", "grp": "gK"}
        r = post("/recohere_details", {"base_sdf_b64": S["plain"]["sdf_b64"], "res": 64,
                                       "ops": [junk, keep]})
        g0 = vol(S["plain"]["sdf_b64"])
        vox = 2.0 / 63
        if r["dropped"] > 0:
            return f"junk dropped ({r['n']} kept)"
        ds = [abs(sdf_at(g0, o["center"])) / vox for o in r["ops"]]
        assert max(ds) < 5.0, f"junk neither dropped nor re-seated ({max(ds):.1f} vox off)"
        return f"all re-seated on surface {['%.1f' % d for d in ds]} vox"
    case("B11 learned re-cohere (set refiner)", b11)

    def b12():
        # smart add: a tall slender mass at the corner must be UNDERSTOOD as a tower
        tall = {"kind": "box", "center": [0.45, 0.3, 0.0], "size": [0.1, 0.6, 0.1],
                "mode": "add", "smooth": 0.0}
        r = post("/interpret_mass", {"base_sdf_b64": S["plain"]["sdf_b64"], "res": 64,
                                     "op": tall})
        assert r["kind"] == "tower", f"tall corner mass read as '{r['kind']}'"
        assert r["n"] >= 3, f"tower construction too thin ({r['n']} ops)"
        carve = {"kind": "box", "center": [0.0, 0.1, 0.7], "size": [0.05, 0.05, 0.05],
                 "mode": "subtract", "smooth": 0.0}
        r2 = post("/interpret_mass", {"base_sdf_b64": S["plain"]["sdf_b64"], "res": 64,
                                      "op": carve})
        assert r2["kind"] in ("window", "door"), f"wall carve read as '{r2['kind']}'"
        return f"tower ({r['n']} ops) · carve -> {r2['kind']}"
    case("B12 smart add (interpret placed mass)", b12)

    def b10():
        from PIL import Image, ImageDraw
        im = Image.new("L", (256, 256), 255)
        d = ImageDraw.Draw(im)
        for xy in [(30, 30, 90, 100), (130, 50, 210, 110), (60, 150, 160, 220)]:
            d.rectangle(xy, fill=0)
        buf = io.BytesIO(); im.save(buf, "PNG")
        r = post("/generate_from_image", {"image_b64": base64.b64encode(buf.getvalue()).decode(),
                                          "meters_across": 120, "max_buildings": 6})
        assert r["n_buildings"] >= 2, f"only {r['n_buildings']} buildings"
        return f"{r['n_buildings']} buildings from mask"
    case("B10 image -> town", b10)


    stamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    with open(os.path.join(OUT, f"report_{stamp}.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["branch", "status", "metric", "secs"]); w.writerows(ROWS)
    fails = sum(1 for r in ROWS if r[1] != "PASS")
    print(f"\n== {len(ROWS) - fails}/{len(ROWS)} PASS ==  report_{stamp}.csv")
    sys.exit(fails)


if __name__ == "__main__":
    main()
