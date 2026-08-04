"""Ingest real watertight LoD2.2 buildings from the 3D BAG (Netherlands) into a clean
SDF corpus for the SDEdit massing prior.

Pipeline per building:
    3DBAG OGC API (bbox) -> CityJSONFeature
      -> LoD2.2 Solid boundaries (BuildingPart) -> trimesh (apply CityJSON transform)
      -> fix_normals, filter degenerate
      -> normalize to Frame-N [-1,1]^3 (CityGML z-up -> our y-up, axes (D=z,H=y,W=x))
      -> igl.signed_distance(FAST_WINDING_NUMBER) on a 64^3 grid  (orientation-robust)
      -> footprint = (sdf<=0).any(H);  height_m from real extents
    -> h5 in the data/recipe_augmentation_v1 schema (so Stage3a can consume it).

Real watertight massing (L-shapes, wings, towers, real roofs) — fixes the broken-GT
curse and raises the SDEdit richness ceiling (see memory project_sdedit_corpus_ceiling).

Run (smoke):  ... scripts/ingest_3dbag.py --smoke
Run (full):   ... scripts/ingest_3dbag.py --per_bbox 3000 --total 12000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import h5py
import igl
import numpy as np
import trimesh

API = "https://api.3dbag.nl/collections/pand/items"
FWN = igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER

# ~1 km^2 RD (EPSG:28992) boxes over varied Dutch city centres -> varied massing.
BBOXES = {
    "rotterdam":  (92000, 436500, 93000, 437500),   # modern towers + varied
    "amsterdam":  (120500, 487000, 121500, 488000),  # canal houses, gables
    "denhaag":    (80000, 454500, 81000, 455500),
    "utrecht":    (136000, 455500, 137000, 456500),
    "delft":      (84000, 447000, 85000, 448000),
}


def fetch(url, retries=3):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.load(r)
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(1.0 + i)


def iter_features(bbox, cap, sleep=0.1):
    """Page through the OGC API for a bbox, yielding (feature, transform)."""
    url = f"{API}?bbox={','.join(map(str, bbox))}&limit=100"
    n = 0
    while url and n < cap:
        d = fetch(url)
        tr = d.get("transform") or d.get("metadata", {}).get("transform")
        for feat in d.get("features", []):
            yield feat, tr
            n += 1
            if n >= cap:
                break
        nxt = [l["href"] for l in d.get("links", []) if l.get("rel") == "next"]
        url = nxt[0] if nxt and n < cap else None
        time.sleep(sleep)


def lod22_mesh(feat, tr):
    scale = np.asarray(tr["scale"]); trans = np.asarray(tr["translate"])
    V = np.asarray(feat["vertices"], float) * scale + trans
    for _cid, obj in feat["CityObjects"].items():
        for g in obj.get("geometry", []):
            if g.get("type") == "Solid" and str(g.get("lod")) == "2.2":
                faces = [[s[0][0], s[0][i], s[0][i + 1]]
                         for s in g["boundaries"][0] for i in range(1, len(s[0]) - 1)]
                if len(faces) < 8:
                    return None
                m = trimesh.Trimesh(V, np.asarray(faces, int), process=True)
                m.fix_normals()
                return m
    return None


def building_to_sdf(m, R=64, margin=1.05):
    """Watertight mesh -> (sdf (R,R,R) float32, footprint (R,R) uint8, height_m)."""
    ext = m.extents
    height_m = float(ext[2])  # CityGML z = up
    c = m.bounds.mean(0)
    s = float(ext.max()) / 2 * margin
    Vn = (np.asarray(m.vertices) - c) / s
    # CityGML (x, y, z-up) -> Frame-N (x, y=up, z); axes used as (D=z, H=y, W=x)
    Vn = np.ascontiguousarray(np.stack([Vn[:, 0], Vn[:, 2], Vn[:, 1]], 1), np.float64)
    Fn = np.ascontiguousarray(m.faces, np.int64)
    g1 = np.linspace(-1, 1, R)
    ZZ, YY, XX = np.meshgrid(g1, g1, g1, indexing="ij")          # (D=z, H=y, W=x)
    P = np.ascontiguousarray(np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], 1), np.float64)
    S = np.asarray(igl.signed_distance(P, Vn, Fn, FWN)[0]).reshape(R, R, R).astype(np.float32)
    fp = (S <= 0).any(axis=1).astype(np.uint8)                   # collapse H -> (D, W)
    return S, fp, height_m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/bag3d_v1")
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--per_bbox", type=int, default=3000)
    ap.add_argument("--total", type=int, default=12000)
    ap.add_argument("--min_h", type=float, default=2.5)     # skip flat slabs/ground
    ap.add_argument("--max_ext", type=float, default=90.0)  # skip merged blocks
    ap.add_argument("--min_fp", type=int, default=80)       # skip near-empty footprints
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.per_bbox, args.total = 150, 150
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    R = args.res

    sdfs, fps, heights, ids = [], [], [], []
    kept = skipped = 0
    t0 = time.time()
    for name, bbox in BBOXES.items():
        if kept >= args.total:
            break
        cap = min(args.per_bbox, args.total - kept)
        got = 0
        for feat, tr in iter_features(bbox, cap):
            try:
                m = lod22_mesh(feat, tr)
                if m is None or not m.is_watertight:
                    skipped += 1; continue
                if m.extents[2] < args.min_h or float(m.extents.max()) > args.max_ext:
                    skipped += 1; continue
                S, fp, h_m = building_to_sdf(m, R)
                if int(fp.sum()) < args.min_fp or not (0.01 < (S <= 0).mean() < 0.7):
                    skipped += 1; continue
                sdfs.append(S); fps.append(fp); heights.append(h_m); ids.append(feat.get("id", ""))
                kept += 1; got += 1
                if kept % 50 == 0:
                    print(f"  kept={kept} skipped={skipped} [{name}] {(time.time()-t0):.0f}s", flush=True)
            except Exception as e:
                skipped += 1
                if skipped % 100 == 1:
                    print(f"  [skip] {type(e).__name__}: {e}", flush=True)
            if kept >= args.total:
                break
        print(f"[bbox] {name}: +{got} (kept={kept}, skipped={skipped})", flush=True)

    if not sdfs:
        raise SystemExit("no buildings ingested — check API reachability / bboxes")
    sdfs = np.stack(sdfs); fps = np.stack(fps); heights = np.asarray(heights, np.float32)
    fname = out / ("bag3d_smoke.h5" if args.smoke else "bag3d.h5")
    with h5py.File(fname, "w") as f:
        # one-building-per-chunk is ESSENTIAL — h5py auto-chunking bundles ~700
        # buildings per chunk, making per-sample random reads ~20s/batch (GPU starves
        # to 0% util). chunks=(1,...) -> reading one building decompresses one chunk.
        f.create_dataset("sdf", data=sdfs, chunks=(1,) + sdfs.shape[1:], compression="lzf")
        f.create_dataset("footprint", data=fps, chunks=(1,) + fps.shape[1:], compression="lzf")
        f.create_dataset("height_m", data=heights)
        f.create_dataset("style_id", data=np.full(len(sdfs), 8, np.int32))     # 8 = "unknown"/real
        f.create_dataset("class_label", data=np.array(["BAG_real"] * len(sdfs), dtype="S16"))
        f.create_dataset("bag_id", data=np.array(ids, dtype="S40"))
    print(f"[done] kept={kept} skipped={skipped} -> {fname}  "
          f"sdf{sdfs.shape} occ_mean={(sdfs<=0).mean():.3f}  {(time.time()-t0):.0f}s")


if __name__ == "__main__":
    main()
