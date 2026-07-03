"""Layer 2.5b — ornament library ingester (retrieval-and-fit source data).

Builds `data/ornaments_v1/`: real heritage-scan ornament meshes (reliefs, panels, friezes)
normalized into a common panel frame + a manifest the retrieval side keys on. This is the
DATA half of ornament retrieval; placement/fit lives in scripts/server/ornaments.py.

Sources (feasibility 2026-07-03):
  - threedscans.com (Oliver Laric) — open direct downloads, no login. License: the project
    publishes museum scans without copyright restriction claims; recorded per-item as
    "no known copyright restrictions" — re-verify before any commercial use.
  - Smithsonian Open Access (CC0) — search API (api.si.edu, key via SI_API_KEY env;
    DEMO_KEY works but is rate-limited ~hourly) resolves 3d-api.si.edu .glb resource URLs,
    which download WITHOUT a key. ~40 ornament-adjacent CC0 items (reliefs/capitals/columns).
  - Scan the World / MyMiniFactory: bot-walled (403) — needs an account + API key; the
    natural growth path once credentials exist. Sketchfab: OAuth, same story.

Normalization: largest component -> quadric decimation (pymeshlab) to ~TARGET_FACES ->
PCA panel frame (width=X >= height=Y >= relief depth=Z, carved face toward +Z via
flat-back heuristic) -> center at origin, max(width,height)=1 -> data/ornaments_v1/<id>.glb.

Run:
  env -u LD_PRELOAD -u LD_LIBRARY_PATH PYTHONPATH=. ./sdfusion/bin/python \
    scripts/foundations/ingest_ornaments.py [--only ID] [--smithsonian]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data/ornaments_v1"
RAW = OUT / "raw"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) GenerativeTowns-research"}
TARGET_FACES = 60000

# curated v0 sources — each is a real, culturally-specific architectural relief
SOURCES = [
    {"id": "ox_relief_romanesque",
     "name": "Ox Relief (Romanesque limestone, 1st half 12th c., Musee de la Romanite)",
     "url": "https://threedscans.com/wp-content/uploads/2023/03/relief_au_taureau_lower_Rez.obj.zip",
     "page": "https://threedscans.com/uncategorized/relief-with-bulls-legs/",
     "license": "threedscans.com — no known copyright restrictions (verify for commercial use)",
     "culture": "romanesque", "period": "12th century",
     "tags": ["relief", "panel", "animal"], "slots": ["wall_panel", "above_door"]},
    {"id": "samudra_manthan_khmer",
     "name": "Angkor Wat bas-relief cast: Samudra Manthan (Churning of the Ocean of Milk)",
     "url": "https://threedscans.com/wp-content/uploads/2016/10/Samudra-Manthan.OBJ.zip",
     "page": "https://threedscans.com/uncategorized/molding-of-a-section-of-the-bas-relief-of-"
             "angkor-wat-depicting-samudra-manthan-churning-of-the-ocean-of-milk/",
     "license": "threedscans.com — no known copyright restrictions (verify for commercial use)",
     "culture": "khmer", "period": "12th century",
     "tags": ["relief", "frieze", "figurative"], "slots": ["wall_panel", "frieze_band"]},
    {"id": "bayon_naval_khmer",
     "name": "Bayon bas-relief cast: Naval Battle",
     "url": "https://threedscans.com/wp-content/uploads/2016/10/Bas-Relief-Bayon.stl.zip",
     "page": "https://threedscans.com/uncategorized/2222/",
     "license": "threedscans.com — no known copyright restrictions (verify for commercial use)",
     "culture": "khmer", "period": "12th-13th century",
     "tags": ["relief", "frieze", "figurative"], "slots": ["wall_panel", "frieze_band"]},
]

SI_TERMS = ("relief", "capital", "column", "ornament")
SI_KEY = os.environ.get("SI_API_KEY", "DEMO_KEY")


def _download(url, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    return dest


def _load_mesh(path: Path):
    """Load a mesh file (or the single mesh inside a zip) as one trimesh.Trimesh."""
    import trimesh
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if not n.startswith("__MACOSX")
                     and n.lower().endswith((".obj", ".stl", ".ply", ".glb"))]
            if not names:
                raise ValueError(f"no mesh inside {path.name}")
            data = z.read(names[0])
            m = trimesh.load(io.BytesIO(data), file_type=names[0].rsplit(".", 1)[-1].lower(),
                             process=False)
    else:
        m = trimesh.load(path, process=False)
    if isinstance(m, trimesh.Scene):
        m = m.to_geometry()
    # STL scans arrive as unindexed triangle soups (verts == 3*faces) — without welding,
    # split() sees every triangle as its own component and "largest component" is 1 face
    m.merge_vertices()
    return m


def _decimate(mesh, target=TARGET_FACES):
    if len(mesh.faces) <= target:
        return mesh
    import pymeshlab
    import trimesh
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(np.asarray(mesh.vertices, np.float64),
                               np.asarray(mesh.faces, np.int32)))
    try:
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target,
                                                    preservenormal=True)
    except AttributeError:   # older pymeshlab API
        ms.simplification_quadric_edge_collapse_decimation(targetfacenum=target,
                                                           preservenormal=True)
    m = ms.current_mesh()
    return trimesh.Trimesh(m.vertex_matrix(), m.face_matrix(), process=True)


def _panel_frame(mesh):
    """Rotate into the panel convention: X=width >= Y=height >= Z=relief depth, carved face
    toward +Z (the flatter side — the mounting back — has its vertices bunched at one plane)."""
    import trimesh
    v = np.asarray(mesh.vertices, np.float64)
    c = v.mean(0)
    _, _, Vt = np.linalg.svd(v - c, full_matrices=False)
    R = Vt                                        # rows: principal axes, variance descending
    if np.linalg.det(R) < 0:
        R[2] *= -1.0                              # keep it a rotation, not a reflection
    q = (v - c) @ R.T                             # x=widest, y=second, z=depth
    # carved front vs flat back: the back concentrates vertices near one depth extreme
    z = q[:, 2]
    zr = z.max() - z.min() + 1e-9
    lo = float((z < z.min() + 0.08 * zr).mean())  # vertex mass near z-min
    hi = float((z > z.max() - 0.08 * zr).mean())
    if hi > lo:                                   # flat side (back) is at +z -> flip around Y
        q[:, 2] *= -1.0
        q[:, 0] *= -1.0
    out = trimesh.Trimesh(q, np.asarray(mesh.faces), process=False)
    return out


def _normalize(mesh):
    """Center, unit-scale (max of width/height = 1). Returns (mesh, aspect_wh, depth_ratio)."""
    v = np.asarray(mesh.vertices, np.float64)
    v -= (v.max(0) + v.min(0)) / 2.0
    ext = v.max(0) - v.min(0)
    s = max(ext[0], ext[1])
    v /= s
    mesh.vertices = v
    ext = ext / s
    return mesh, float(ext[0] / max(ext[1], 1e-6)), float(ext[2] / max(ext[0], ext[1]))


def ingest_item(src) -> dict:
    print(f"[{src['id']}] downloading …")
    raw = _download(src["url"], RAW / src["url"].rsplit("/", 1)[-1])
    print(f"[{src['id']}] loading {raw.name} ({raw.stat().st_size >> 20} MB) …")
    m = _load_mesh(raw)
    print(f"[{src['id']}] raw: {len(m.vertices)} verts / {len(m.faces)} faces")
    # keep the largest connected component (scans carry floating debris)
    parts = m.split(only_watertight=False)
    if len(parts) > 1:
        m = max(parts, key=lambda p: len(p.faces))
    m = _decimate(m)
    m = _panel_frame(m)
    m, aspect, depth = _normalize(m)
    out_f = OUT / f"{src['id']}.glb"
    m.export(out_f)
    print(f"[{src['id']}] -> {out_f.name}: {len(m.faces)} faces, aspect w/h={aspect:.2f}, "
          f"depth ratio={depth:.2f}")
    return {**{k: src[k] for k in ("id", "name", "culture", "period", "tags", "slots",
                                   "license")},
            "source": src["page"], "file": out_f.name, "n_faces": int(len(m.faces)),
            "aspect_wh": round(aspect, 3), "depth_ratio": round(depth, 3)}


def smithsonian_candidates():
    """Resolve CC0 Smithsonian 3D items for SI_TERMS -> source entries (glb resource URLs).
    The search API is rate-limited on DEMO_KEY; the file CDN (3d-api.si.edu) is not."""
    found = []
    for term in SI_TERMS:
        import urllib.parse
        q = urllib.parse.urlencode({"q": f'{term} AND online_media_type:"3D Models"',
                                    "rows": 20, "api_key": SI_KEY})
        try:
            with urllib.request.urlopen(
                    f"https://api.si.edu/openaccess/api/v1.0/search?{q}", timeout=30) as r:
                d = json.load(r)
        except Exception as ex:
            print(f"[smithsonian] '{term}' failed ({ex}) — rate limit? set SI_API_KEY")
            continue
        if "response" not in d:
            print(f"[smithsonian] '{term}': {d.get('error', d)}")
            continue
        for row in d["response"]["rows"]:
            dnr = row.get("content", {}).get("descriptiveNonRepeating", {})
            if dnr.get("metadata_usage", {}).get("access") != "CC0":
                continue
            glbs = [res.get("url") for m in dnr.get("online_media", {}).get("media", [])
                    for res in m.get("resources", [])
                    if str(res.get("url", "")).endswith("_std.glb")
                    and "draco" not in str(res.get("url", ""))]
            if not glbs:
                continue
            sid = "si_" + row["id"].replace(":", "_").replace("-", "_")[-24:]
            found.append({"id": sid, "name": row["title"], "url": glbs[0],
                          "page": dnr.get("record_link", "https://3d.si.edu"),
                          "license": "CC0 (Smithsonian Open Access)",
                          "culture": "unknown", "period": "unknown",
                          "tags": ["relief" if term == "relief" else term],
                          "slots": ["wall_panel"] if term in ("relief", "ornament")
                                   else ["column", "pilaster"]})
        print(f"[smithsonian] '{term}': {len(found)} total candidates so far")
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="ingest a single source id")
    ap.add_argument("--smithsonian", action="store_true",
                    help="also query Smithsonian Open Access (needs API quota)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    sources = list(SOURCES)
    if args.smithsonian:
        sources += smithsonian_candidates()
    if args.only:
        sources = [s for s in sources if s["id"] == args.only]

    manifest_p = OUT / "manifest.json"
    manifest = {e["id"]: e for e in json.load(open(manifest_p))} if manifest_p.exists() else {}
    fails = 0
    for src in sources:
        try:
            manifest[src["id"]] = ingest_item(src)
        except Exception as ex:
            print(f"[{src['id']}] FAILED: {ex}")
            fails += 1
    json.dump(sorted(manifest.values(), key=lambda e: e["id"]),
              open(manifest_p, "w"), indent=1)
    print(f"\nlibrary: {len(manifest)} ornaments -> {manifest_p} ({fails} failed)")
    return fails


if __name__ == "__main__":
    sys.exit(main())
