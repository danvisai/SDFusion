"""End-to-end HTTP check of the /refine_with_edit sdedit mode via in-process TestClient."""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts/server"))
from fastapi.testclient import TestClient
from inference_service import app

fp = [[-7, -9], [7, -9], [7, 9], [-7, 9]]
with TestClient(app) as c:
    r = c.post("/regenerate_building", json={"footprint": fp, "style": "modern",
               "building_class": "RESIDENTIAL", "height": 16.0, "seed": 1, "detail": False})
    assert r.status_code == 200, r.text
    b = r.json(); print(f"[regenerate] {r.status_code} style={b['style']} verts={b['n_vertices']}")

    edit = {"kind": "box", "center": [5, 9, 6], "size": [1.6, 9, 1.6], "mode": "add", "smooth": 0.0}
    r2 = c.post("/refine_with_edit", json={"base_style": b["style"], "base_recipe_params": b["recipe_params"],
                "footprint": fp, "height": 16.0, "edits": [edit], "building_class": "RESIDENTIAL",
                "mode": "sdedit", "strength": 0.5})
    print(f"[refine sdedit] {r2.status_code}")
    if r2.status_code == 200:
        s = r2.json(); print(f"  OK  verts={s['n_vertices']}  faces={s['n_faces']}  glb_b64_len={len(s['mesh_glb_b64'])}")
    else:
        print("  FAIL:", r2.text[:400])
