"""Layer 2.5b — ornament retrieval + fit (the placement half; data half =
scripts/foundations/ingest_ornaments.py -> data/ornaments_v1/).

Retrieval picks a library ornament for the building (style -> culture affinity + slot tags,
seeded). Fit scales it to a slot on a footprint wall (aspect preserved), yaws the panel to
the wall's outward normal, and sinks its back into the wall so no seam shows. Ornaments are
SYMBOLIC state ({id, edge, t, y, w}) resolved to mesh INSTANCES merged after the building
mesh is built — the scan's full carve detail survives (independent of SDF sampling res) and
the diffusion prior never touches it. Doctrine: retrieval DECIDES, placement is procedural.
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


def _edge_frame(footprint, edge):
    """(base point p0, edge dir d, outward normal n, edge length) for footprint edge i."""
    poly = np.asarray(footprint, np.float64)
    p0, p1 = poly[edge % len(poly)], poly[(edge + 1) % len(poly)]
    d = p1 - p0
    L = float(np.linalg.norm(d)) + 1e-9
    d = d / L
    n = np.array([d[1], -d[0]])                   # right-hand perp
    area2 = float(np.sum(poly[:, 0] * np.roll(poly[:, 1], -1)
                         - np.roll(poly[:, 0], -1) * poly[:, 1]))
    if area2 > 0:                                 # CCW polygon -> flip to point outward
        n = -n
    return p0, d, n, L


def propose(footprint, height, style, seed=None, slot="wall_panel"):
    """ONE fitted ornament instance for the building (v0): the library piece retrieved for
    the style, centered on the LONGEST footprint wall at door-lintel height. Returns the
    symbolic instance dict the mesh composition consumes."""
    poly = np.asarray(footprint, np.float64)
    lens = [np.linalg.norm(poly[(i + 1) % len(poly)] - poly[i]) for i in range(len(poly))]
    edge = int(np.argmax(lens))
    e = retrieve(style, slot=slot, seed=seed)
    w = float(np.clip(0.30 * lens[edge], 1.0, 6.0))
    aspect = max(float(e.get("aspect_wh", 1.5)), 0.3)
    h_panel = w / aspect
    if h_panel > 0.45 * height:                   # keep the panel wall-scaled
        h_panel = 0.45 * height
        w = h_panel * aspect
    y = float(np.clip(0.62 * height, h_panel / 2 + 0.5, height - h_panel / 2 - 0.3))
    return {"id": e["id"], "edge": edge, "t": 0.5, "y": round(y, 3), "w": round(w, 3),
            "name": e["name"], "culture": e.get("culture", "unknown")}


def instance_mesh(inst, footprint):
    """Resolve a symbolic instance -> a transformed trimesh in building-local world meters."""
    import trimesh
    m = _mesh(inst["id"]).copy()
    ext = m.vertices.max(0) - m.vertices.min(0)   # normalized panel: max(w,h)=1
    s = float(inst["w"]) / max(float(ext[0]), 1e-6)
    m.apply_scale(s)
    depth_w = float(ext[2]) * s
    p0, d, n, L = _edge_frame(footprint, int(inst["edge"]))
    at = p0 + d * (float(inst["t"]) * L)
    yaw = float(np.arctan2(n[0], n[1]))           # rotate panel +Z onto the outward normal
    m.apply_transform(trimesh.transformations.rotation_matrix(yaw, [0, 1, 0]))
    protrude = min(0.30, 0.65 * depth_w)          # sink the back into the wall, no seam
    off = protrude - depth_w / 2.0
    m.apply_translation([at[0] + n[0] * off, float(inst["y"]), at[1] + n[1] * off])
    return m


def apply_ornaments(mesh, ornaments, footprint):
    """Merge ornament instances into a built building mesh (building-local frame)."""
    import trimesh
    parts = [mesh]
    for inst in ornaments:
        try:
            parts.append(instance_mesh(inst, footprint))
        except Exception as ex:
            print(f"[ornaments] skipped {inst.get('id')}: {ex}")
    return trimesh.util.concatenate(parts) if len(parts) > 1 else mesh
