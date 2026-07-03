"""Export the generated town as a single glTF scene — the v1 Unreal exporter.

GEOMETRY ONLY (gray, untextured). Each building is rebuilt EXACTLY from its town state
(recipe params + sculpt edits + composer detail), placed at its world position as a NAMED
node, plus a ground plane. A sidecar manifest carries per-building metadata so the
symbolic state survives into the engine.

Coordinate convention: standard glTF — right-handed, +Y up, the file's own unit. Unreal's
glTF importer converts to its +Z-up / left-handed / centimeter space automatically.
`scale` BAKES a unit factor into the geometry (100 = centimeters = the Unreal default so a
16 m building reads as 1600 uu regardless of importer scale settings; 1 = meters for
Blender / other glTF tools).

v2 (later) adds xatlas UVs + diffusion-baked textures so buildings arrive styled.
"""
from __future__ import annotations

import numpy as np


def _orient_mesh_outward(mesh, sdf_fn, device):
    """Flip winding if it points inward (so single-sided renderers like Unreal show the
    OUTSIDE, not the back walls). Face normal should align with the SDF gradient (outward,
    inside<0/outside>0). The grid_to_mesh axis reorder is a reflection that can invert
    marching-cubes winding; trimesh.fix_normals only enforces consistency, not direction."""
    import torch
    v = np.asarray(mesh.vertices, np.float32)
    f = np.asarray(mesh.faces)
    if not len(f):
        return mesh
    tris = v[f]
    fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    fc = tris.mean(1)
    ext = float(np.linalg.norm(v.max(0) - v.min(0)))
    eps = max(ext * 1e-3, 1e-3)
    p = torch.as_tensor(fc, dtype=torch.float32, device=device)

    def s(q):
        out = []
        for i in range(0, q.shape[0], 200000):
            out.append(sdf_fn(q[i:i + 200000]))
        return torch.cat(out)
    ex = torch.tensor([eps, 0, 0], device=device)
    ey = torch.tensor([0, eps, 0], device=device)
    ez = torch.tensor([0, 0, eps], device=device)
    grad = torch.stack([s(p + ex) - s(p - ex), s(p + ey) - s(p - ey),
                        s(p + ez) - s(p - ez)], -1).cpu().numpy()
    if np.nanmean((fn * grad).sum(1)) < 0:
        mesh.invert()
    return mesh


def _op_top_y(op):
    """World y-top of an add-mode EditOp (scene/sdf_primitives conventions: box/sphere
    centered at op center; cylinder size=[r, TOTAL h] centered; cone size=[angle, h] base at
    center y, apex at +h; gable/hip size=[w, d, body_h, roof_h] base at center y)."""
    k = op.get("kind", "box")
    cy = float(op.get("center", (0, 0, 0))[1])
    s = [float(v) for v in op.get("size", (1.0, 1.0, 1.0))]
    if k == "sphere":
        return cy + s[0]
    if k == "cylinder":
        return cy + s[1] / 2.0
    if k == "cone":
        return cy + s[1]
    if k in ("gable", "hip"):
        return cy + s[2] + s[3]
    return cy + (s[1] if len(s) > 1 else s[0])           # box / rounded_box


def _op_xz_r(op):
    """World xz half-extent (radius) of an EditOp, kind-aware — treating cone's
    size=[angle_deg, h] angle as meters would inflate the sample bbox ~10x and coarsen the
    whole mesh (96^3 is spread over the bbox)."""
    import math
    k = op.get("kind", "box")
    s = [abs(float(v)) for v in op.get("size", (1.0, 1.0, 1.0))]
    if k in ("sphere", "cylinder"):
        return s[0]
    if k == "cone":
        return s[1] * math.tan(math.radians(min(max(s[0], 1.0), 89.0)))
    if k in ("gable", "hip"):
        return max(s[0], s[1]) / 2.0                     # full width/depth -> half
    return max(s[0], s[2] if len(s) > 2 else s[0])       # box / rounded_box


def build_building_mesh(refiner, b, res=96):
    """Rebuild ONE town building's final detailed mesh (recipe + sculpt edits + composer
    detail) in local world meters, sitting on y=0, OUTWARD winding. Mirrors the bake path."""
    from scene.sdf_edit import recipe_base_sdf, EditableBuilding, EditOp
    from scene.composer_detail import compose_detail, get_composer
    from scene.sdf_primitives import sample_grid, grid_to_mesh
    poly = np.asarray(b["footprint"], np.float32)
    h = float(b["height"])
    base = recipe_base_sdf(b["style"], b["recipe_params"], poly, h, device=refiner.device)
    if b.get("edits"):
        base = EditableBuilding(base, [EditOp.from_dict(d) for d in b["edits"]]).composed()
    sdf, n_towers = base, 0
    try:
        sdf, _lay, dec = compose_detail(base, poly, h, b.get("building_class", "RESIDENTIAL"),
                                        style=b.get("style", "modern"),
                                        composer=get_composer(refiner.device))
        n_towers = dec["n_towers"]
    except Exception as ex:
        print(f"[export] composer detail unavailable ({ex}); plain massing")
    x0, z0 = float(poly[:, 0].min()), float(poly[:, 1].min())
    x1, z1 = float(poly[:, 0].max()), float(poly[:, 1].max())
    pad = 0.12 * max(x1 - x0, z1 - z0) + 1.0
    head = h * (1.9 if n_towers else 1.5)
    for op in (b.get("edits") or []):
        # user constructions extend past the recipe bbox (an interpret_mass tower + spire
        # tops out above h*1.5; a wing reaches past the footprint) — grow the sample bbox
        # or they clip flat at its faces
        if str(op.get("mode", "add")) != "add":
            continue                                      # subtract can't extend geometry
        head = max(head, _op_top_y(op) * 1.12)
        cx, cz = float(op["center"][0]), float(op["center"][2])
        r = _op_xz_r(op)
        x0, x1 = min(x0, cx - r), max(x1, cx + r)
        z0, z1 = min(z0, cz - r), max(z1, cz + r)
    bbox = (x0 - pad, 0.0, z0 - pad, x1 + pad, head, z1 + pad)
    g = sample_grid(sdf, res, bbox, device=refiner.device)
    wx = float(b.get("weather") or 0.0)
    if wx > 0:
        # Layer 2.5a: procedural aging (cracks/edge wear/erosion) on the FINAL detailed
        # SDF grid — part of the building's symbolic state (weather + weather_seed), so
        # rebuild and export reproduce it exactly
        import torch
        from scene.sdf_weather import weather_grid
        vox = ((bbox[3] - bbox[0]) / (res - 1), (bbox[4] - bbox[1]) / (res - 1),
               (bbox[5] - bbox[2]) / (res - 1))
        g = torch.from_numpy(weather_grid(g.detach().cpu().numpy(), vox,
                                          seed=int(b.get("weather_seed") or 0),
                                          intensity=wx, y0_m=float(bbox[1])))
    mesh = grid_to_mesh(g, bbox, 0.0)
    if mesh is not None and len(mesh.faces):
        mesh = _orient_mesh_outward(mesh, sdf, refiner.device)
        from scene.mesh_cleanup import cleanup_mesh
        mesh = cleanup_mesh(mesh)                         # weld + drop floating fragments
        if b.get("ornaments"):
            # Layer 2.5b: heritage-scan relief instances, merged AFTER cleanup so the
            # fragment-dropper can't remove them; full scan detail, SDF-res independent
            try:
                from ornaments import apply_ornaments
                mesh = apply_ornaments(mesh, b["ornaments"], poly)
            except Exception as ex:
                print(f"[export] ornaments unavailable ({ex})")
    return mesh


def export_town(refiner, buildings, scale=100.0, ground=True, res=96):
    """Returns (glb_bytes, manifest_dict, total_vertices)."""
    import trimesh
    scene = trimesh.Scene()
    bldgs, allx, allz, total_v = [], [], [], 0
    for i, b in enumerate(buildings):
        mesh = build_building_mesh(refiner, b, res=res)      # already oriented outward
        if mesh is None or len(mesh.faces) == 0:
            continue
        mesh.apply_scale(float(scale))                       # m -> chosen unit
        px, pz = float(b["position"][0]) * scale, float(b["position"][1]) * scale
        T = trimesh.transformations.translation_matrix([px, 0.0, pz])
        name = f"bldg_{i:02d}_{b.get('style', 'modern')}_{str(b.get('building_class', 'RES'))[:3]}"
        mesh.metadata.update({"style": b.get("style"), "class": b.get("building_class"),
                              "height_m": b.get("height")})
        scene.add_geometry(mesh, node_name=name, geom_name=name, transform=T)
        total_v += len(mesh.vertices)
        allx.append(px); allz.append(pz)
        bldgs.append({"name": name, "style": b.get("style"),
                      "class": b.get("building_class"), "height_m": b.get("height"),
                      "position": [round(px, 2), round(pz, 2)],
                      "n_vertices": int(len(mesh.vertices))})
    if ground and bldgs:
        m = 30.0 * scale
        ax0, ax1, az0, az1 = min(allx) - m, max(allx) + m, min(allz) - m, max(allz) + m
        g = trimesh.creation.box(extents=[ax1 - ax0, 0.1 * scale, az1 - az0])
        g.apply_translation([(ax0 + ax1) / 2, -0.05 * scale, (az0 + az1) / 2])
        scene.add_geometry(g, node_name="ground", geom_name="ground")
    glb = scene.export(file_type="glb")
    manifest = {
        "units": "centimeters" if scale == 100.0 else ("meters" if scale == 1.0 else f"scale={scale}"),
        "up": "Y (glTF standard; Unreal converts to Z-up on import)",
        "note": "v1 GEOMETRY ONLY (untextured/gray). UE5: File > Import > town.glb, "
                "enable Nanite on the meshes, add a Directional Light + Sky. "
                "If it imports small, set Import Uniform Scale appropriately.",
        "n_buildings": len(bldgs),
        "buildings": bldgs,
    }
    return glb, manifest, total_v


def export_town_textured(refiner, buildings, pipe, unit=100.0, ground=True,
                         n_views=5, steps=22, res96=96, inpaint_pipe=None):
    """TEXTURED town export: each building rebuilt (params + edits + composer detail),
    multi-view diffusion texture-baked (with its own style ref if any), placed at its world
    position as a named TEXTURED node + ground. Returns (glb_bytes, manifest, total_v).
    ~ (n_views+1) SDXL renders per building, so SLOW — cap building count upstream."""
    import base64 as _b
    import io as _io
    import numpy as np
    import trimesh
    from PIL import Image
    from refine import _bbox
    from scene.sdf_edit import recipe_base_sdf, EditableBuilding, EditOp
    import texture_bake as tb

    scene = trimesh.Scene()
    bldgs, allx, allz, total_v = [], [], [], 0
    for i, b in enumerate(buildings):
        poly = np.asarray(b["footprint"], np.float32)
        h = float(b["height"])
        base = recipe_base_sdf(b["style"], b["recipe_params"], poly, h, device=refiner.device)
        if b.get("edits"):
            base = EditableBuilding(base, [EditOp.from_dict(d) for d in b["edits"]]).composed()
        bbox = _bbox(poly, h, b.get("edits", []))
        sdf_t, _fp, _hn, c, s = refiner._recipe_to_frame_n(base, bbox, margin=1.3)
        grid64 = sdf_t[0, 0].detach().cpu().numpy().astype(np.float32)
        grid96 = refiner.detail_cube_volume(grid64, c, s,
                                            building_class=b.get("building_class", "RESIDENTIAL"),
                                            style=b.get("style", "modern"), res_out=res96)
        ref = None
        if b.get("style_ref_b64"):
            ref = Image.open(_io.BytesIO(
                _b.b64decode(b["style_ref_b64"].split(",")[-1]))).convert("RGB")
        prompt = b.get("prompt") or (f"photo of a {b.get('style', 'modern')} "
                                     f"{str(b.get('building_class', 'RESIDENTIAL')).lower()} building, "
                                     "architectural photography, high detail")
        if inpaint_pipe is not None:
            res = tb.bake_building_iterative(grid96, pipe, inpaint_pipe, prompt, style_ref=ref,
                                             seed=7, n_views=n_views, steps=steps,
                                             style=b.get("style", "modern"))
        else:
            res = tb.bake_building(grid96, pipe, prompt, style_ref=ref, seed=7,
                                   n_views=n_views, steps=steps, style=b.get("style", "modern"))
        mesh = res["mesh"]
        mesh.apply_scale(float(s) * float(unit))            # cube[-1,1] -> meters*unit
        v = np.asarray(mesh.vertices, np.float32)
        v[:, 1] -= v[:, 1].min()
        mesh.vertices = v
        px, pz = float(b["position"][0]) * unit, float(b["position"][1]) * unit
        T = trimesh.transformations.translation_matrix([px, 0.0, pz])
        name = f"bldg_{i:02d}_{b.get('style', 'modern')}_{str(b.get('building_class', 'RES'))[:3]}"
        scene.add_geometry(mesh, node_name=name, geom_name=name, transform=T)
        total_v += len(mesh.vertices)
        allx.append(px); allz.append(pz)
        bldgs.append({"name": name, "style": b.get("style"),
                      "class": b.get("building_class"), "height_m": b.get("height"),
                      "position": [round(px, 2), round(pz, 2)],
                      "n_vertices": int(len(mesh.vertices)),
                      "texture_coverage": round(float(res["coverage"]), 3)})
    if ground and bldgs:
        m = 30.0 * unit
        ax0, ax1, az0, az1 = min(allx) - m, max(allx) + m, min(allz) - m, max(allz) + m
        g = trimesh.creation.box(extents=[ax1 - ax0, 0.1 * unit, az1 - az0])
        g.apply_translation([(ax0 + ax1) / 2, -0.05 * unit, (az0 + az1) / 2])
        scene.add_geometry(g, node_name="ground", geom_name="ground")
    glb = scene.export(file_type="glb")
    manifest = {
        "units": "centimeters" if unit == 100.0 else ("meters" if unit == 1.0 else f"scale={unit}"),
        "up": "Y (glTF standard; Unreal converts to Z-up on import)",
        "note": "v2 TEXTURED (per-building albedo atlas). UE5: File > Import > town.glb, "
                "enable Nanite, the materials import automatically. Add a Directional Light + Sky.",
        "n_buildings": len(bldgs),
        "buildings": bldgs,
    }
    return glb, manifest, total_v
