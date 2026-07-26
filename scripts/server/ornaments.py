"""Layer 2.5b — ornament retrieval + fit (the placement half; data half =
scripts/foundations/ingest_ornaments.py -> data/ornaments_v1/).

Retrieval picks a library ornament for the building (style -> culture affinity + slot tags,
seeded) — the only thing decided by a hand-coded rule, since it's a small local table over a
tiny library, not a placement judgment. WHERE it goes is decided by the ALREADY-TRAINED
part-layout planner + CoherentPartRefiner (layout_detail.place_ornament) — no external API
and no hand-coded 'longest wall' heuristic in the placement path; see place_ornament's
docstring for why 'balcony' stands in (BuildingNet's part taxonomy has no ornament label).
Fit scales the retrieved mesh to the model's slot (aspect preserved), yaws it to the slot's
outward normal, and sinks its back into the wall so no seam shows. Ornaments are SYMBOLIC
state ({id, center, normal, w}) resolved to mesh INSTANCES merged after the building mesh is
built — the scan's full carve detail survives (independent of SDF sampling res) and the
diffusion prior never touches it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LIB = REPO / "data/ornaments_v1"

# which scanned cultures suit which recipe styles (v0: tiny library, soft weights)
STYLE_CULTURE = {
    "victorian": ["romanesque"], "colonial": ["romanesque"],
    "mediterranean": ["romanesque"], "public_civic": ["romanesque"],
    "craftsman": ["khmer"], "contemporary": [], "modern": [], "industrial": [],
}

_manifest = None
_meshes = {}


def library():
    global _manifest
    if _manifest is None:
        _manifest = json.load(open(LIB / "manifest.json"))
    return _manifest


def _mesh(entry_id):
    if entry_id not in _meshes:
        import trimesh
        e = next(e for e in library() if e["id"] == entry_id)
        m = trimesh.load(LIB / e["file"], process=False)
        if isinstance(m, trimesh.Scene):
            m = m.to_geometry()
        _meshes[entry_id] = m
    return _meshes[entry_id]


def retrieve(style, slot="wall_panel", seed=None):
    """Seeded, affinity-weighted choice from the library. Returns a manifest entry."""
    entries = library()
    if not entries:
        raise ValueError("ornament library is empty — run ingest_ornaments.py")
    cultures = STYLE_CULTURE.get(style, [])
    w = np.array([(3.0 if e.get("culture") in cultures else 1.0)
                  * (2.0 if slot in e.get("slots", []) else 1.0) for e in entries])
    rng = np.random.default_rng(seed)
    return entries[int(rng.choice(len(entries), p=w / w.sum()))]


def _propose_heuristic(footprint, height, style, seed, e):
    """Fallback ONLY: used when propose_ornament_slot found no candidate (the planner's
    autoregressive sampling is stochastic and can occasionally emit zero 'balcony'
    instances). Centers the panel on the longest footprint wall at door-lintel height —
    the pre-2026-07-06 default, now a safety net rather than the primary path."""
    poly = np.asarray(footprint, np.float64)
    lens = [np.linalg.norm(poly[(i + 1) % len(poly)] - poly[i]) for i in range(len(poly))]
    edge = int(np.argmax(lens))
    p0, p1 = poly[edge], poly[(edge + 1) % len(poly)]
    d = (p1 - p0) / (lens[edge] + 1e-9)
    n = np.array([d[1], -d[0]])
    area2 = float(np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1]))
    if area2 > 0:
        n = -n
    at = p0 + d * 0.5 * lens[edge]
    w = float(np.clip(0.30 * lens[edge], 1.0, 6.0))
    aspect = max(float(e.get("aspect_wh", 1.5)), 0.3)
    h_panel = min(w / aspect, 0.45 * height)
    w = h_panel * aspect
    y = float(np.clip(0.62 * height, h_panel / 2 + 0.5, height - h_panel / 2 - 0.3))
    return {"id": e["id"], "center": [float(at[0]), y, float(at[1])],
            "normal": [float(n[0]), float(n[1])], "w": round(w, 3),
            "name": e["name"], "culture": e.get("culture", "unknown"), "source": "fallback"}


def propose(footprint, height, style, seed=None, slot=None):
    """ONE fitted ornament instance for the building. `slot` is a WORLD-frame placement
    already decided by the trained models (layout_detail.place_ornament, bridged to world
    meters by the caller): {"center":[x,y,z], "normal":[nx,nz], "half_extent":[ex,ey,ez]}.
    Retrieval (WHICH relief) is still a seeded style/culture affinity choice over the small
    local library; slot is None only when the planner sampled no candidate this draw, in
    which case _propose_heuristic is the safety net. Returns the symbolic instance dict the
    mesh composition (instance_mesh/apply_ornaments) consumes."""
    e = retrieve(style, seed=seed)
    if slot is None:
        return _propose_heuristic(footprint, height, style, seed, e)
    ex, ey, ez = slot["half_extent"]
    w = float(np.clip(2.0 * max(ex, ez), 1.0, 6.0))
    aspect = max(float(e.get("aspect_wh", 1.5)), 0.3)
    h_panel = w / aspect
    max_h = max(2.0 * ey, 0.6)
    if h_panel > max_h:
        h_panel, w = max_h, max_h * aspect
    y = float(np.clip(slot["center"][1], h_panel / 2 + 0.5, height - h_panel / 2 - 0.3))
    return {"id": e["id"], "center": [slot["center"][0], y, slot["center"][2]],
            "normal": slot["normal"], "w": round(w, 3), "name": e["name"],
            "culture": e.get("culture", "unknown"), "source": "learned"}


def instance_mesh(inst):
    """Resolve a symbolic instance -> a transformed trimesh in building-local world meters."""
    import trimesh
    m = _mesh(inst["id"]).copy()
    ext = m.vertices.max(0) - m.vertices.min(0)   # normalized panel: max(w,h)=1
    s = float(inst["w"]) / max(float(ext[0]), 1e-6)
    m.apply_scale(s)
    depth_w = float(ext[2]) * s
    nx, nz = inst["normal"]
    yaw = float(np.arctan2(nx, nz))               # rotate panel +Z onto the outward normal
    m.apply_transform(trimesh.transformations.rotation_matrix(yaw, [0, 1, 0]))
    protrude = min(0.30, 0.65 * depth_w)          # sink the back into the wall, no seam
    off = protrude - depth_w / 2.0
    cx, cy, cz = inst["center"]
    m.apply_translation([cx + nx * off, cy, cz + nz * off])
    return m


def apply_ornaments(mesh, ornaments, footprint=None):
    """Merge ornament instances into a built building mesh (building-local frame).
    `footprint` is accepted for call-site stability but unused — placement is now an
    absolute world pose (center + normal), not footprint-edge-relative."""
    import trimesh
    parts = [mesh]
    for inst in ornaments:
        try:
            parts.append(instance_mesh(inst))
        except Exception as ex:
            print(f"[ornaments] skipped {inst.get('id')}: {ex}")
    return trimesh.util.concatenate(parts) if len(parts) > 1 else mesh
