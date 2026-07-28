"""Concat the real LoD2 massing corpora (NL + DE + JP) into one packed h5 for Stage 3a.

Stacks data/bag3d_v1/bag3d.h5 (NL, source_id=0) + data/real_massing_v1/nrw.h5 (DE, 1) +
data/real_massing_v1/plateau.h5 (JP, 2) -> data/real_massing_v1/real.h5, in the SAME schema
(so Bag3dDataset/HybridDataset read it unchanged) plus a guaranteed `source_id` region channel.

Memory-safe: pre-sizes the output and stream-copies each source in blocks (never loads a whole
corpus into RAM). Missing inputs are skipped with a warning (run again once ingestion lands).

  env -u LD_PRELOAD -u LD_LIBRARY_PATH PYTHONPATH=. sdfusion/bin/python \
    scripts/foundations/concat_real_massing.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
# (path, default source_id if the file lacks a source_id field)
SOURCES = [
    (REPO / "data/bag3d_v1/bag3d.h5", 0),          # NL
    (REPO / "data/real_massing_v1/nrw.h5", 1),     # DE
    (REPO / "data/real_massing_v1/plateau.h5", 2), # JP
]
BLK = 512


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "data/real_massing_v1/real.h5"))
    ap.add_argument("--smoke", action="store_true", help="use *_smoke.h5 inputs")
    args = ap.parse_args()

    srcs = []
    for p, sid in SOURCES:
        if args.smoke:
            p = p.with_name(p.stem + "_smoke.h5")
        if not p.exists():
            print(f"[skip] missing {p}")
            continue
        with h5py.File(p, "r") as f:
            n = int(f["sdf"].shape[0]); R = f["sdf"].shape[1:]
        srcs.append((p, sid, n, R))
        print(f"[src] {p.name}: n={n} sdf{R} source_id<-{sid}")
    if not srcs:
        raise SystemExit("no source corpora found")

    total = sum(n for _, _, n, _ in srcs)
    R = srcs[0][3]
    assert all(r == R for *_, r in srcs), "SDF resolutions differ across sources"
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[out] {out}  total={total}  sdf{R}")

    with h5py.File(out, "w") as d:
        d.create_dataset("sdf", (total, *R), np.float32, chunks=(1, *R), compression="lzf")
        d.create_dataset("footprint", (total, R[0], R[2]), np.uint8,
                         chunks=(1, R[0], R[2]), compression="lzf")
        d.create_dataset("height_m", (total,), np.float32)
        d.create_dataset("style_id", (total,), np.int32)
        d.create_dataset("source_id", (total,), np.int32)
        d.create_dataset("class_label", (total,), "S16")
        d.create_dataset("bag_id", (total,), "S64")

        off = 0
        for p, sid, n, _ in srcs:
            with h5py.File(p, "r") as s:
                has_src = "source_id" in s
                has_cls = "class_label" in s
                has_bid = "bag_id" in s
                has_sty = "style_id" in s
                for i in range(0, n, BLK):
                    j = min(i + BLK, n)
                    d["sdf"][off + i:off + j] = s["sdf"][i:j]
                    d["footprint"][off + i:off + j] = s["footprint"][i:j]
                    d["height_m"][off + i:off + j] = s["height_m"][i:j]
                    d["style_id"][off + i:off + j] = (s["style_id"][i:j].astype(np.int32)
                                                      if has_sty else np.full(j - i, 8, np.int32))
                    d["source_id"][off + i:off + j] = (s["source_id"][i:j].astype(np.int32)
                                                       if has_src else np.full(j - i, sid, np.int32))
                    if has_cls:
                        d["class_label"][off + i:off + j] = s["class_label"][i:j].astype("S16")
                    if has_bid:
                        d["bag_id"][off + i:off + j] = s["bag_id"][i:j].astype("S64")
                    print(f"  {p.name}: {j}/{n}", end="\r", flush=True)
            off += n
            print(f"\n[copied] {p.name} ({n}) -> offset {off}")

    with h5py.File(out, "r") as d:
        sid = d["source_id"][:]
        hm = d["height_m"][:]
        print(f"[done] {out}  total={len(sid)}  "
              f"by source: " + ", ".join(f"{k}={int((sid==k).sum())}" for k in (0, 1, 2)) +
              f"  height_m med={np.median(hm):.1f} max={hm.max():.1f}")


if __name__ == "__main__":
    main()
