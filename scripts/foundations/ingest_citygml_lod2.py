"""Generalized LoD2 CityGML -> SDF corpus ingester (cross-cultural massing for Layer 1).

Reuses the proven SDF core from scripts/ingest_3dbag.py (igl fast-winding SDF, orientation/
watertight-robust) and adds CRS-aware CityGML parsing so DIFFERENT national sources flow into
the SAME packed h5 schema as data/bag3d_v1/bag3d.h5 (so HybridDataset/Bag3dDataset consume it
unchanged). The point is DIVERSITY: NL (3D BAG, already have) + DE (NRW) + JP (PLATEAU) so the
Stage-3a diffusion sees real cross-cultural massing instead of plateauing on one culture.

Sources (verified 2026-06-29):
  --source nrw      Germany, opengeodata.nrw.de LoD2 CityGML, per-1km .gml tiles, EPSG:25832
                    (PROJECTED metres, posList = E N H).
  --source plateau  Japan, PLATEAU on gic-plateau S3, {mesh}_2.zip -> bldg.zip -> *_bldg_6697_*.gml,
                    EPSG:6697 (GEOGRAPHIC, posList = lat lon height[m], degrees) -> reprojected to
                    local metres here. Only LoD2 boundedBy surfaces are taken (LoD0/1 also present).

Output: <out>/<source>.h5 with the bag3d schema + a source_id channel for region conditioning:
  sdf (N,R,R,R) f32 | footprint (N,R,R) u8 | height_m (N,) f32 | style_id (N,) i32 (8=unknown/real)
  | source_id (N,) i32 (0=NL 1=DE 2=JP) | class_label (N,) S16 | src_key (N,) S64

  env -u LD_PRELOAD -u LD_LIBRARY_PATH PYTHONPATH=. sdfusion/bin/python \
    scripts/foundations/ingest_citygml_lod2.py --source nrw --max_tiles 1 --limit 200 --smoke
"""
from __future__ import annotations

import argparse
import io
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import h5py
import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.ingest_3dbag import building_to_sdf  # proven mesh -> (sdf, footprint, height_m)

SOURCE_ID = {"nl": 0, "nrw": 1, "plateau": 2}
SOURCE_CLASS = {"nrw": b"DE_real", "plateau": b"JP_real"}

NRW_DIR = "https://www.opengeodata.nrw.de/produkte/geobasis/3dg/lod2_gml/lod2_gml/"
PLATEAU_BASE = "https://gic-plateau.s3-ap-northeast-1.amazonaws.com/2020/tokyo23ku/"
# A spread of Tokyo 2nd-level mesh tiles (varied wards) -> varied JP massing.
PLATEAU_TILES = ["533937_2.zip", "533957_2.zip", "533954_2.zip", "533934_2.zip",
                 "533925_2.zip", "533947_2.zip", "533944_2.zip"]

_loc = lambda t: t.split("}")[-1]


def _fetch(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


# ---- CityGML geometry -------------------------------------------------------

def _building_rings(b):
    """LoD2 surface rings under a Building, as raw (M,3) posList arrays.

    Prefer boundedBy semantic surfaces (= LoD2 in both NRW and PLATEAU) so we do NOT mix in
    LoD0 footprints / LoD1 blocks (PLATEAU ships all three under one Building)."""
    bounded = [el for el in b.iter() if _loc(el.tag) == "boundedBy"]
    targets = bounded or [el for el in b.iter() if _loc(el.tag) in ("lod2Solid", "lod1Solid")] or [b]
    rings = []
    for tgt in targets:
        for pl in tgt.iter():
            if _loc(pl.tag) == "posList" and pl.text:
                v = np.fromstring(pl.text.strip(), sep=" ")
                if v.size >= 9 and v.size % 3 == 0:
                    rings.append(v.reshape(-1, 3))
    return rings


def _to_local_metres(rings, geographic):
    """Raw CityGML rings -> (x=east, y=north, z=up) metres, recentred.

    geographic (EPSG:6697): cols are (lat, lon, h[m]) deg -> equirectangular about the centroid.
    projected  (EPSG:25832): cols are (E, N, H) metres -> used as (x, y, z) directly."""
    allp = np.concatenate(rings, 0)
    if geographic:
        lat0, lon0 = allp[:, 0].mean(), allp[:, 1].mean()
        mlat = 111320.0
        mlon = 111320.0 * np.cos(np.radians(lat0))
        return [np.stack([(r[:, 1] - lon0) * mlon, (r[:, 0] - lat0) * mlat, r[:, 2]], 1) for r in rings]
    return rings  # already (E, N, H) metres


def _rings_to_mesh(rings, geographic):
    local = _to_local_metres(rings, geographic)
    V, F = [], []
    for r in local:
        ring = r[:-1] if len(r) > 1 and np.allclose(r[0], r[-1]) else r
        if len(ring) < 3:
            continue
        base = len(V)
        V.extend(ring.tolist())
        for i in range(1, len(ring) - 1):           # fan triangulation (planar convex polys)
            F.append([base, base + i, base + i + 1])
    if len(F) < 4:
        return None
    m = trimesh.Trimesh(np.asarray(V, float), np.asarray(F, int), process=True)
    m.merge_vertices()
    try:
        m.fix_normals()
    except Exception:
        pass
    return m


def _buildings_from_gml(gml_bytes):
    """Yield (gml_id, geographic, building_element) for each Building."""
    root = ET.fromstring(gml_bytes)
    head = gml_bytes[:6000].decode("utf-8", "replace")
    geographic = "6697" in head or "4326" in head or "CRS84" in head
    for b in root.iter():
        if _loc(b.tag) == "Building":
            gid = b.get("{http://www.opengis.net/gml}id", "")
            yield gid, geographic, b


# ---- source front-ends (yield gml byte-blobs) -------------------------------

def _nrw_gmls(max_tiles, plateau_gml=None):
    listing = _fetch(NRW_DIR).decode("utf-8", "replace")
    names = [m.split('"')[0] for m in listing.split('file name="')[1:]]
    names = [n for n in names if n.endswith(".gml")]
    step = max(1, len(names) // max(max_tiles, 1))          # stride => spread across the state
    chosen = names[::step][:max_tiles]
    print(f"[nrw] {len(names)} tiles available; striding {step} -> {len(chosen)} spread tiles", flush=True)
    for n in chosen:
        print(f"[nrw] download {n}", flush=True)
        yield n, _fetch(NRW_DIR + n)


def _plateau_gmls(max_tiles, plateau_gml=None):
    if plateau_gml:                                  # local smoke shortcut
        yield Path(plateau_gml).name, Path(plateau_gml).read_bytes()
        return
    for tz in PLATEAU_TILES[:max_tiles]:
        print(f"[plateau] download {tz}", flush=True)
        outer = zipfile.ZipFile(io.BytesIO(_fetch(PLATEAU_BASE + tz)))
        if "bldg.zip" not in outer.namelist():
            continue
        inner = zipfile.ZipFile(io.BytesIO(outer.read("bldg.zip")))
        for n in inner.namelist():
            if n.endswith(".gml"):
                yield f"{tz}:{n}", inner.read(n)


# ---- main -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["nrw", "plateau"])
    ap.add_argument("--out", default="data/real_massing_v1")
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--max_tiles", type=int, default=1)
    ap.add_argument("--limit", type=int, default=10_000_000)
    ap.add_argument("--plateau_gml", default=None, help="local *_bldg_6697_*.gml for smoke")
    ap.add_argument("--min_h", type=float, default=2.5)
    ap.add_argument("--max_ext", type=float, default=90.0)
    ap.add_argument("--min_fp", type=int, default=80)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    R = args.res
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    src_iter = {"nrw": _nrw_gmls, "plateau": _plateau_gmls}[args.source]

    sdfs, fps, heights, ids = [], [], [], []
    kept = skipped = seen = 0
    t0 = time.time()
    for gml_key, gml_bytes in src_iter(args.max_tiles, args.plateau_gml):
        for gid, geographic, b in _buildings_from_gml(gml_bytes):
            if kept >= args.limit:
                break
            seen += 1
            try:
                rings = _building_rings(b)
                if len(rings) < 3:
                    skipped += 1; continue
                m = _rings_to_mesh(rings, geographic)
                if m is None:
                    skipped += 1; continue
                if m.extents[2] < args.min_h or float(m.extents.max()) > args.max_ext:
                    skipped += 1; continue
                S, fp, h_m = building_to_sdf(m, R)
                if int(fp.sum()) < args.min_fp or not (0.01 < (S <= 0).mean() < 0.7):
                    skipped += 1; continue
                sdfs.append(S); fps.append(fp); heights.append(h_m)
                ids.append(f"{gml_key}#{gid}"[:64])
                kept += 1
                if kept % 100 == 0:
                    print(f"  kept={kept} skipped={skipped} seen={seen} "
                          f"({time.time()-t0:.0f}s)", flush=True)
            except Exception as e:
                skipped += 1
                if skipped % 200 == 1:
                    print(f"  [skip] {type(e).__name__}: {str(e)[:80]}", flush=True)
        if kept >= args.limit:
            break

    if not sdfs:
        raise SystemExit(f"no buildings ingested from {args.source} (seen={seen})")
    sdfs = np.stack(sdfs); fps = np.stack(fps)
    heights = np.asarray(heights, np.float32)
    sid = SOURCE_ID[args.source]
    fname = out / (f"{args.source}_smoke.h5" if args.smoke else f"{args.source}.h5")
    with h5py.File(fname, "w") as f:
        f.create_dataset("sdf", data=sdfs, chunks=(1,) + sdfs.shape[1:], compression="lzf")
        f.create_dataset("footprint", data=fps, chunks=(1,) + fps.shape[1:], compression="lzf")
        f.create_dataset("height_m", data=heights)
        f.create_dataset("style_id", data=np.full(len(sdfs), 8, np.int32))     # 8 = unknown/real
        f.create_dataset("source_id", data=np.full(len(sdfs), sid, np.int32))  # region token
        f.create_dataset("class_label", data=np.array([SOURCE_CLASS[args.source]] * len(sdfs)))
        f.create_dataset("bag_id", data=np.array(ids, dtype="S64"))
    print(f"[done] {args.source}: kept={kept} skipped={skipped} seen={seen} -> {fname}\n"
          f"       sdf{sdfs.shape} occ_mean={(sdfs<=0).mean():.3f} "
          f"height_m[min/med/max]={heights.min():.1f}/{np.median(heights):.1f}/{heights.max():.1f} "
          f"fp_sum_med={int(np.median(fps.sum((1,2))))} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
