"""Ingest LoD3 CityGML buildings -> our part-instance frame ([type, axis-bbox] per building).

Source: savenow/lod3-road-space-models (TUM Ingolstadt LoD3, openly licensed) — real buildings
with explicit <bldg:Window>/<bldg:Door> semantic surfaces (1075 windows / 61 doors / 59 bldgs).
This is the REAL, instance-annotated facade-element data BuildingNet (sparse per-point) and 3D BAG
(facade-less LoD2) lack — fuel for the coherent-add-primitive refiner (see
docs/COHERENT_ADD_PRIMITIVE_BUILD_SPEC_2026-06-15.md).

Output mirrors outputs/part_layouts_full/part_instances.npz:
  rows (N,9) = [building_idx, type_id, cx,cy,cz, sx,sy,sz, n_pts]   (per-building normalized to
  [-1,1]; size = half-extent fraction). types dict + building names.

CityGML frame is EPSG:32632 (X=easting, Y=northing, Z=UP, metres). We map to OUR axes
(cy = height): our (x,y,z) = citygml (X, Z, Y).

  env -u LD_PRELOAD -u LD_LIBRARY_PATH PYTHONPATH=. \
    /tmp/sdfusion_venv/bin/python scripts/foundations/ingest_lod3_citygml.py
"""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
GML = REPO / "data/lod3_tum/raw/lod3_combined.gml"
OUT = REPO / "data/lod3_tum"
TYPE_OF = {"Window": 2, "Door": 6, "RoofSurface": 4}     # our part-instance type ids
TYPES = {"2": "window", "6": "door", "4": "roof"}


def _local(tag):
    return tag.split("}")[-1]


def _poslists(el):
    """All (M,3) point arrays under `el` (CityGML X,Y,Z) -> our (X,Z,Y) order."""
    pts = []
    for pl in el.iter():
        if _local(pl.tag) == "posList" and pl.text:
            v = np.fromstring(pl.text.strip(), sep=" ")
            if v.size >= 9 and v.size % 3 == 0:
                p = v.reshape(-1, 3)[:, [0, 2, 1]]           # X, Z(up->our y), Y
                pts.append(p)
    return pts


def _bbox(point_arrays):
    if not point_arrays:
        return None
    allp = np.concatenate(point_arrays, 0)
    return allp.min(0), allp.max(0), len(allp)


def main():
    if not GML.exists():
        sys.exit(f"missing {GML} — download it first (see module docstring).")
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[parse] {GML.name} ({GML.stat().st_size/1e6:.0f} MB) ...", flush=True)
    root = ET.parse(GML).getroot()
    buildings = [el for el in root.iter() if _local(el.tag) == "Building"]
    print(f"[parse] {len(buildings)} buildings", flush=True)

    rows, names = [], []
    for bi, b in enumerate(buildings):
        bb = _bbox(_poslists(b))                              # whole-building bbox (all surfaces)
        if bb is None:
            continue
        mn, mx, _ = bb
        c = (mn + mx) / 2.0
        s = float(max(mx - mn) / 2.0) or 1.0                 # half the largest dimension
        names.append(b.get("{http://www.opengis.net/gml}id", f"b{bi}"))
        n_el = 0
        for el in b.iter():
            t = _local(el.tag)
            if t not in TYPE_OF:
                continue
            eb = _bbox(_poslists(el))
            if eb is None:
                continue
            emn, emx, npts = eb
            ec = ((emn + emx) / 2.0 - c) / s                 # normalized center [-1,1]
            es = np.maximum((emx - emn) / 2.0 / s, 1e-4)     # normalized half-extent [0,1]
            rows.append([len(names) - 1, TYPE_OF[t], *ec.astype(np.float32),
                         *es.astype(np.float32), npts])
            n_el += 1
        if (bi + 1) % 10 == 0:
            print(f"  {bi+1}/{len(buildings)} buildings, {len(rows)} elements", flush=True)

    rows = np.asarray(rows, np.float32)
    import collections
    hist = collections.Counter(rows[:, 1].astype(int).tolist())
    print("[done] elements:", {TYPES[str(k)]: v for k, v in hist.items()},
          f"| {len(names)} buildings, mean {len(rows)/max(len(names),1):.1f}/bldg")
    np.savez(OUT / "lod3_part_instances.npz", rows=rows,
             names=np.array(names, dtype="<U40"), types=json.dumps(TYPES))
    print(f"[save] {OUT/'lod3_part_instances.npz'}")


if __name__ == "__main__":
    main()
