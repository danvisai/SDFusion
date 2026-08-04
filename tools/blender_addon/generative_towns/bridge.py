"""Compute bridge for the GenerativeTowns Blender add-on.

NO `bpy` import here — this is the pure-Python layer that talks to our torch stack, so it
is unit-testable outside Blender (see tools/blender_addon/test_bridge.py).

Blender ships its own Python; this injects the repo + the sdfusion venv site-packages onto
sys.path so Blender's interpreter can import our modules + torch, and runs the B+.6
generative head + `scene/sdf_edit.EditableBuilding` IN-PROCESS (the realtime local-edit
loop). If the local import fails (wrong Python, no venv), `generate()` falls back to the
Stage A HTTP service; edits still require the local engine.

Override locations with env vars: GENTOWNS_REPO, GENTOWNS_VENV_SITE, GENTOWNS_SERVICE_URL.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

DEFAULT_REPO = "/scratch/gilbreth/dsimhadr/GenerativeTowns/SDFusion"
REPO = Path(os.environ.get("GENTOWNS_REPO", DEFAULT_REPO))
VENV_SITE = os.environ.get("GENTOWNS_VENV_SITE", str(REPO / "sdfusion/lib/python3.9/site-packages"))
SERVICE_URL = os.environ.get("GENTOWNS_SERVICE_URL", "http://127.0.0.1:8077")

_engine = None
_refiner = None
_local_ok: Optional[bool] = None
_device = None


def _ensure_paths():
    for p in (str(REPO / "scripts" / "server"), str(REPO), VENV_SITE):
        if p and Path(p).exists() and p not in sys.path:
            sys.path.insert(0, p)


def local_available() -> bool:
    """True if we can run the edit engine in-process (the realtime path)."""
    global _local_ok, _device
    if _local_ok is None:
        _ensure_paths()
        try:
            import torch  # noqa
            import numpy  # noqa
            from scene.sdf_edit import EditableBuilding  # noqa
            _device = "cuda" if torch.cuda.is_available() else "cpu"
            _local_ok = True
        except Exception as e:  # pragma: no cover - environment dependent
            print(f"[gentowns] local engine unavailable ({type(e).__name__}: {e}); "
                  f"will use HTTP service at {SERVICE_URL}")
            _local_ok = False
    return _local_ok


def _engine_obj():
    global _engine
    if _engine is None:
        from recipe_inference import RecipeInferenceEngine
        _engine = RecipeInferenceEngine(device=_device)
    return _engine


def _refiner_obj():
    global _refiner
    if _refiner is None:
        from refine import Refiner
        _refiner = Refiner(_engine_obj())
    return _refiner


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _verts_faces(mesh):
    if mesh is None or len(getattr(mesh, "faces", [])) == 0:
        return [], []
    return mesh.vertices.tolist(), mesh.faces.tolist()


def _bbox_for(footprint, height, edit_ops, pad=0.2):
    import numpy as np
    poly = np.asarray(footprint, dtype=float)
    x0, z0 = float(poly[:, 0].min()), float(poly[:, 1].min())
    x1, z1 = float(poly[:, 0].max()), float(poly[:, 1].max())
    ymax = float(height) * 1.3
    for op in edit_ops:
        cx, cy, cz = op["center"]; sz = op.get("size") or (1.0,)
        r = max(sz)
        x0, x1 = min(x0, cx - r), max(x1, cx + r)
        z0, z1 = min(z0, cz - r), max(z1, cz + r)
        ymax = max(ymax, cy + r)
    px, pz = (x1 - x0) * pad + 1.0, (z1 - z0) * pad + 1.0
    return (x0 - px, 0.0, z0 - pz, x1 + px, ymax * 1.1, z1 + pz)


def _edit_mesh(style, params, footprint, height, edit_ops, res):
    import numpy as np
    from scene.sdf_edit import EditableBuilding, EditOp, recipe_base_sdf
    base = recipe_base_sdf(style, np.asarray(params, np.float32), footprint, height, device=_device)
    bldg = EditableBuilding(base, [EditOp.from_dict(d) for d in edit_ops])
    mesh = bldg.to_mesh(_bbox_for(footprint, height, edit_ops), res, device=_device)
    return _verts_faces(mesh)


# ---------------------------------------------------------------------------
# Public API (called by the bpy operators)
# ---------------------------------------------------------------------------

def generate(footprint: List[List[float]], style: str, building_class: str,
             height: float, seed: Optional[int] = None, guidance: float = 2.0,
             res: int = 64) -> dict:
    """Generate a base building. Returns {recipe_params, verts, faces}."""
    if local_available():
        eng = _engine_obj()
        params = eng.sample_params(footprint, height, building_class, style, seed, guidance)
        verts, faces = _edit_mesh(style, params, footprint, height, [], res)
        return {"recipe_params": [float(x) for x in params], "verts": verts,
                "faces": faces, "backend": "local"}
    # HTTP fallback (generate only) — returns glb for the bpy glTF importer.
    import json
    import urllib.request
    payload = json.dumps({"footprint": footprint, "style": style,
                          "building_class": building_class, "height": height,
                          "seed": seed, "guidance": guidance}).encode()
    req = urllib.request.Request(SERVICE_URL + "/regenerate_building", payload,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.load(r)
    return {"recipe_params": d["recipe_params"], "glb_b64": d["mesh_glb_b64"],
            "backend": "http"}


def apply_edits(style: str, recipe_params: List[float], footprint: List[List[float]],
                height: float, edit_ops: List[dict], res: int = 64) -> dict:
    """Re-mesh a building given its base params + edit stack. LOCAL ONLY (realtime)."""
    if not local_available():
        raise RuntimeError("local engine required for edits; cannot edit over HTTP "
                           "(would not be realtime). Install torch into Blender's Python "
                           "or point GENTOWNS_VENV_SITE at the sdfusion venv.")
    verts, faces = _edit_mesh(style, recipe_params, footprint, height, edit_ops, res)
    return {"verts": verts, "faces": faces, "backend": "local"}


def refine(base_state: dict, edits: List[dict], target_style: Optional[str] = None,
           mode: str = "fast", building_class: str = "RESIDENTIAL",
           seed: Optional[int] = None) -> dict:
    """Project a sculpt onto a clean recipe building (cleanup / re-style). LOCAL ONLY.

    Returns {style, recipe_params, footprint, height, verts, faces, iou_to_edit} — the
    refined building is a fresh clean recipe building, so the caller resets the edit stack.
    """
    if not local_available():
        raise RuntimeError("local engine required for AI refine")
    out = _refiner_obj().refine(base_state, edits, target_style=target_style, mode=mode,
                                building_class=building_class, seed=seed)
    verts, faces = _verts_faces(out["mesh"])
    return {"style": out["style"], "recipe_params": out["recipe_params"],
            "footprint": out["footprint"], "height": out["height"],
            "verts": verts, "faces": faces, "iou_to_edit": out["iou_to_edit"],
            "backend": "local"}


def palette() -> List[str]:
    if local_available():
        from scene.sdf_edit import PALETTE
        return list(PALETTE)
    return ["box", "rounded_box", "sphere", "cylinder", "cone", "gable", "hip"]
