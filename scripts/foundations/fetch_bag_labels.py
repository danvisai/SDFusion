"""Foundations task #3 — fetch real BAG conditioning labels (era / floors / roof) for the corpus.

The 3D BAG API has no gebruiksdoel(function), but per Building it exposes oorspronkelijkbouwjaar
(year -> era), b3_bouwlagen (floors -> class proxy) and b3_dak_type (roof). Re-page the SAME bboxes
the ingest used, build {bag_id -> attrs}, align to the corpus bag_id order, and save labels.npz.
Combined with the procedural 8-style labels, this is the hybrid conditioning for the prior retrain.
"""
from __future__ import annotations
import json, sys, time, urllib.request
from collections import Counter
from pathlib import Path
import h5py, numpy as np

REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO / "scripts"))
from ingest_3dbag import API, BBOXES  # reuse the exact source

ERA_EDGES = [1900, 1945, 1975, 2000]      # -> buckets 0..4 ; 5 = unknown
ROOF_MAP = {"slanted": 0, "horizontal": 1, "multiple horizontal": 2, "no roof": 3}  # 4 = unknown


def era_bucket(y):
    if not y:
        return 5
    return int(np.searchsorted(ERA_EDGES, int(y)))


def fetch(url):
    for i in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "curl/8"}), timeout=40) as r:
                return json.load(r)
        except Exception as e:
            if i == 5:
                print(f"  [fetch giveup] {e}", flush=True); return None
            time.sleep(2 ** i)   # 1,2,4,8,16,32 backoff (API throttles on bursts)


def main():
    h5p = REPO / "data/bag3d_v1/bag3d.h5"
    with h5py.File(h5p, "r") as f:
        corpus_ids = [x.decode() for x in f["bag_id"][:]]
    print(f"[corpus] {len(corpus_ids)} buildings")

    attrs = {}                               # bag_id -> (year, floors, roof_str)
    corpus_set = set(corpus_ids)
    for name, bbox in BBOXES.items():
        url = f"{API}?bbox={','.join(map(str, bbox))}&limit=100"; pages = 0
        while url and pages < 100:
            d = fetch(url)
            if d is None:                    # tolerate throttling: skip the rest of this bbox
                print(f"  [{name}] aborted at page {pages} (network); moving on", flush=True); break
            pages += 1
            for feat in d.get("features", []):
                fid = feat.get("id")
                for _cid, obj in feat.get("CityObjects", {}).items():
                    if obj.get("type") == "Building":
                        a = obj.get("attributes", {})
                        attrs[fid] = (a.get("oorspronkelijkbouwjaar"), a.get("b3_bouwlagen"), a.get("b3_dak_type"))
                        break
            if pages % 10 == 0:
                hit = sum(1 for k in attrs if k in corpus_set)
                print(f"  [{name}] page {pages}, attrs {len(attrs)} (corpus-matched {hit})", flush=True)
            nxt = [l["href"] for l in d.get("links", []) if l.get("rel") == "next"]
            url = nxt[0] if nxt else None
            time.sleep(0.25)
        print(f"  [{name}] done: paged {pages}, total attrs {len(attrs)}", flush=True)

    era = np.full(len(corpus_ids), 5, np.int64)
    floors = np.full(len(corpus_ids), -1, np.int64)
    roof = np.full(len(corpus_ids), 4, np.int64)
    matched = 0
    for i, bid in enumerate(corpus_ids):
        if bid in attrs:
            matched += 1
            y, fl, rk = attrs[bid]
            era[i] = era_bucket(y)
            floors[i] = int(fl) if fl not in (None, "") else -1
            roof[i] = ROOF_MAP.get(rk, 4)
    print(f"\n[match] {matched}/{len(corpus_ids)} corpus buildings got attributes ({100*matched/len(corpus_ids):.1f}%)")
    print("  era buckets (<1900,1900-44,1945-74,1975-99,>=2000,unk):", dict(sorted(Counter(era.tolist()).items())))
    print("  roof (slanted,horiz,multi-horiz,none,unk):", dict(sorted(Counter(roof.tolist()).items())))
    fl_ok = floors[floors > 0]
    print(f"  floors: n={len(fl_ok)} min={fl_ok.min() if len(fl_ok) else 0} "
          f"median={int(np.median(fl_ok)) if len(fl_ok) else 0} max={fl_ok.max() if len(fl_ok) else 0}")

    outp = REPO / "data/bag3d_v1/bag_labels.npz"
    np.savez(outp, era=era, floors=floors, roof=roof, bag_id=np.array(corpus_ids),
             era_edges=np.array(ERA_EDGES), roof_map=json.dumps(ROOF_MAP))
    print(f"[saved] {outp}")


if __name__ == "__main__":
    main()
