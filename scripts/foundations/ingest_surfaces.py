"""#62 re-ingest: recover the SURFACE geometry that `real.h5` never stored.

`real.h5` holds 64^3 SDFs and 64^2 footprints only. A vecset encoder point-samples the *surface*, so
spec #67 cannot proceed without meshes. Both existing ingests already build a `trimesh` per building
and then throw it away at the `building_to_sdf` step -- this walks the same sources with the same
parsers and keeps it.

Keyed by the SAME id string the ingests wrote to `bag_id`, so rows join by id rather than by order:
re-running upstream need not reproduce the original ordering or filter outcomes for the pairing to
hold. (#62 verified all 35,776 ids are unique; the JP ids are truncated at the 64-char storage width
but still resolve uniquely by prefix.)

Meshes are stored in **Frame-N**, i.e. exactly the normalisation `building_to_sdf` applies before
voxelising: centre on the bbox, divide by (max_extent/2 * 1.05), then reorder CityGML's z-up axes to
y-up. That is what makes a recovered mesh line up with its stored SDF -- verified by
`--verify`, which re-voxelises recovered meshes and compares against the stored field.

Usage:
    ingest_surfaces.py --source plateau            # JP, 3 nested zips, ~45 MB
    ingest_surfaces.py --source nrw                # DE, 41 tiles, ~234 MB
    ingest_surfaces.py --source bag3d              # NL, per-id 3DBAG API calls
    ingest_surfaces.py --verify --source plateau   # alignment check against real.h5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

MARGIN = 1.05           # must match building_to_sdf
H5 = REPO / "data/real_massing_v1/real.h5"
OUT = REPO / "data/real_massing_v1"
BAG_API = "https://api.3dbag.nl/collections/pand/items"


def to_frame_n(m):
    """Mesh -> Frame-N vertices, replicating building_to_sdf's normalisation exactly.

    Also repairs winding: the CityGML rings come out inverted (negative volume), which is invisible
    to the SDF path because that uses fast-winding-number signing, but NOT to a vecset encoder --
    it consumes face normals, and inverted winding would hand it inside-out surfaces.
    """
    if m.volume < 0:
        m.invert()
    v = np.asarray(m.vertices, np.float64)
    ext = m.extents
    c = m.bounds.mean(0)
    s = float(ext.max()) / 2 * MARGIN
    vn = (v - c) / s
    return np.ascontiguousarray(np.stack([vn[:, 0], vn[:, 2], vn[:, 1]], 1), np.float32)


def _wanted_ids(source: str):
    """The ids real.h5 actually holds for this source -> {id: row}."""
    import h5py
    sid = {"bag3d": 0, "nrw": 1, "plateau": 2}[source]
    with h5py.File(H5, "r") as f:
        rows = np.nonzero(f["source_id"][:] == sid)[0]
        ids = [f["bag_id"][int(r)].decode() for r in rows]
    return {i: int(r) for i, r in zip(ids, rows)}


def _iter_citygml(source: str, want: dict, max_tiles: int):
    """Yield (id, mesh) for CityGML sources, reusing the existing ingest's parsers."""
    from scripts.foundations.ingest_citygml_lod2 import (
        _buildings_from_gml, _building_rings, _rings_to_mesh, _nrw_gmls, _plateau_gmls)
    # only fetch tiles our rows actually reference
    tiles = {i.split("#")[0] for i in want}
    if source == "nrw":
        # Their _nrw_gmls STRIDES across the state to sample `max_tiles` tiles -- that would pull
        # thousands of files and still might miss ours. We know exactly which 41 tiles our rows
        # reference (#62), so fetch precisely those.
        from scripts.foundations.ingest_citygml_lod2 import NRW_DIR, _fetch

        def src(_mt, _pg):
            for i, name in enumerate(sorted(tiles), 1):
                print(f"[nrw] {i}/{len(tiles)} {name}", flush=True)
                try:
                    yield name, _fetch(NRW_DIR + name)
                except Exception as e:
                    print(f"  [skip tile] {type(e).__name__}: {str(e)[:70]}", flush=True)
    else:
        src = _plateau_gmls
    for gml_key, gml_bytes in src(max_tiles, None):
        for gid, geographic, b in _buildings_from_gml(gml_bytes):
            key = f"{gml_key}#{gid}"[:64]
            if key not in want:
                continue
            try:
                rings = _building_rings(b)
                if len(rings) < 3:
                    continue
                m = _rings_to_mesh(rings, geographic)
                if m is not None:
                    yield key, m
            except Exception:
                continue


def _iter_bag3d(want: dict, workers: int = 16, chunk: int = 512):
    """Yield (id, mesh) for NL by resolving each stored bag_id against the 3DBAG API (#62).

    The API costs ~2s per building, so sequentially this corpus is ~400 minutes. The calls are
    I/O-bound, so the network is parallelised across a small thread pool while mesh construction
    stays on the main thread (keeping trimesh single-threaded). Requests are issued in chunks so
    fetched JSON does not all accumulate in memory at once.
    """
    from concurrent.futures import ThreadPoolExecutor
    from scripts.ingest_3dbag import lod22_mesh

    def _get(bid):
        try:
            req = urllib.request.Request(f"{BAG_API}/{bid}", headers={"User-Agent": "curl/8"})
            with urllib.request.urlopen(req, timeout=45) as r:
                return bid, json.load(r)
        except Exception:
            return bid, None

    ids = list(want)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i in range(0, len(ids), chunk):
            for bid, d in ex.map(_get, ids[i:i + chunk]):
                done += 1
                if d is None:
                    continue
                feat = d.get("feature", d)
                tr = (d.get("metadata", {}) or {}).get("transform") or d.get("transform") \
                    or feat.get("transform")
                try:
                    m = lod22_mesh(feat, tr)
                except Exception:
                    m = None
                if m is not None:
                    yield bid, m
            print(f"  ...{done}/{len(ids)} fetched", flush=True)


def run(source: str, max_tiles: int, limit: int) -> None:
    import h5py
    want = _wanted_ids(source)
    print(f"[{source}] rows in real.h5: {len(want)}")
    it = _iter_bag3d(want) if source == "bag3d" else _iter_citygml(source, want, max_tiles)

    keys, verts, faces, vo, fo = [], [], [], [0], [0]
    t0 = time.time()
    for key, m in it:
        vn = to_frame_n(m)
        fn = np.asarray(m.faces, np.int32)
        keys.append(key); verts.append(vn); faces.append(fn)
        vo.append(vo[-1] + len(vn)); fo.append(fo[-1] + len(fn))
        if len(keys) % 250 == 0:
            print(f"  recovered {len(keys)}/{len(want)} ({time.time()-t0:.0f}s)", flush=True)
        if limit and len(keys) >= limit:
            break

    if not keys:
        raise SystemExit(f"[{source}] recovered nothing")
    rows = np.array([want[k] for k in keys], np.int32)
    path = OUT / f"surfaces_{source}.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("verts", data=np.concatenate(verts), compression="lzf")
        f.create_dataset("faces", data=np.concatenate(faces), compression="lzf")
        f.create_dataset("vert_offset", data=np.asarray(vo, np.int64))
        f.create_dataset("face_offset", data=np.asarray(fo, np.int64))
        f.create_dataset("row", data=rows)                       # index into real.h5
        f.create_dataset("bag_id", data=np.array(keys, dtype="S64"))
    print(f"[{source}] recovered {len(keys)}/{len(want)} "
          f"({100*len(keys)/len(want):.1f}%)  verts={vo[-1]:,} faces={fo[-1]:,}  "
          f"-> {path}  ({time.time()-t0:.0f}s)")


def verify(source: str, n: int) -> None:
    """Re-voxelise recovered meshes and compare against the stored SDF -- the alignment proof."""
    import h5py, igl
    FWN = igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER
    from scripts.foundations.vecset_ceiling_probe import RES, TRUNC, grid_points
    pts = grid_points()
    with h5py.File(OUT / f"surfaces_{source}.h5", "r") as s, h5py.File(H5, "r") as f:
        k = min(n, len(s["row"]))
        print(f"[verify {source}] n={k}   (IoU of occupancy vs the stored field)")
        ious, errs = [], []
        for i in range(k):
            a, b = int(s["vert_offset"][i]), int(s["vert_offset"][i + 1])
            c, d = int(s["face_offset"][i]), int(s["face_offset"][i + 1])
            v = np.asarray(s["verts"][a:b], np.float64)
            fc = np.ascontiguousarray(np.asarray(s["faces"][c:d]), np.int32)
            got = np.asarray(igl.signed_distance(pts, v, fc, FWN)[0], np.float32).reshape(RES, RES, RES)
            # building_to_sdf grids as meshgrid(ZZ,YY,XX) -> the STORED array is indexed [z,y,x],
            # while grid_points() queries in [x,y,z]. Transpose the reference to compare. (#63 was
            # unaffected: its mesh and query were both in the array's own index frame.)
            ref = np.asarray(f["sdf"][int(s["row"][i])], np.float32).transpose(2, 1, 0)
            ga, ra = got <= 0, ref <= 0
            iou = float((ga & ra).sum() / max((ga | ra).sum(), 1))
            err = float(np.abs(np.clip(got, -TRUNC, TRUNC) - np.clip(ref, -TRUNC, TRUNC)).mean())
            ious.append(iou); errs.append(err)
            print(f"   {s['bag_id'][i].decode()[:44]:44s} IoU={iou:.4f}  L1={err:.5f}")
        print(f"  MEAN IoU={np.mean(ious):.4f}  L1={np.mean(errs):.5f}   "
              f"{'ALIGNED' if np.mean(ious) > 0.95 else 'MISALIGNED — do not use'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["bag3d", "nrw", "plateau"], required=True)
    ap.add_argument("--max_tiles", type=int, default=999)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--verify_n", type=int, default=8)
    a = ap.parse_args()
    verify(a.source, a.verify_n) if a.verify else run(a.source, a.max_tiles, a.limit)
