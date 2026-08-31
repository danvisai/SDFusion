"""#90: choose the token alignment -- canonical sort, or explicit matching.

Scores each candidate reordering on the quantity that matters, **token agreement between a real
latent and its blockout partner**, over a region-stratified sample. Position distance is only the
proxy the methods optimise; it is reported so the two can be seen to disagree, never ranked on.

Usage:
    probe_token_alignment.py --real R.h5 --blockout B.h5              # the method comparison
    probe_token_alignment.py --real R.h5 --blockout B.h5 --sweep_k    # where to put the knee

Both caches must carry `query_pos`, i.e. be written after #88.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from models.token_alignment import METHODS, align, report  # noqa: E402
from utils.numeric_guard import check_numpy  # noqa: E402

OUT = REPO / "execution/artifacts/token_alignment_probe.json"

#: what the lost run measured on n=102, kept beside the new numbers so a regression is visible
PUBLISHED = {"as_encoded": 0.0405, "morton": 0.2112, "greedy": 0.5387, "hungarian": 0.5106,
             "nn": 0.7079}
PUBLISHED_K = {16: 0.4783, 64: 0.5141, 256: 0.5329, 2048: 0.5392}


def _rows(real: str, blockout: str, n: int):
    """Rows present in both caches, with their region, refusing caches that predate #88."""
    import h5py
    with h5py.File(real, "r") as a, h5py.File(blockout, "r") as b:
        for f, name in ((a, real), (b, blockout)):
            if "query_pos" not in f:
                raise SystemExit(f"{name} has no query_pos -- rebuild it, positions cannot be "
                                 "recovered after the fact (#88)")
        ra, rb = np.asarray(a["row"]), np.asarray(b["row"])
        reg = {int(r): int(g) for r, g in zip(ra, np.asarray(a["region"]))}
    common = np.intersect1d(ra, rb)
    ia = {int(r): i for i, r in enumerate(ra)}
    ib = {int(r): i for i, r in enumerate(rb)}
    # The cache is already stratified at build time; still report the mix, because a sample that
    # silently collapses to one source has voided figures on this project before.
    pick = [int(r) for r in common[:n]] if n else [int(r) for r in common]
    return pick, ia, ib, reg


def _pair(a, b, i: int, j: int):
    """One building's (za, zb, pa, pb), read fresh from disk."""
    return (np.array(a["latent"][i], np.float32), np.array(b["latent"][j], np.float32),
            np.array(a["query_pos"][i], np.float32), np.array(b["query_pos"][j], np.float32))


def run(real: str, blockout: str, n: int, k: int, methods) -> dict:
    import h5py

    pick, ia, ib, reg = _rows(real, blockout, n)
    mix = {}
    for r in pick:
        mix[reg.get(r, -1)] = mix.get(reg.get(r, -1), 0) + 1
    print(f"[align] {len(pick)} buildings, region mix {dict(sorted(mix.items()))}")

    acc = {m: {"cosine": [], "matched_frac": [], "dist_matched": [], "dist_unmatched": [],
               "seconds": 0.0} for m in methods}
    with h5py.File(real, "r") as a, h5py.File(blockout, "r") as b:
        for r in pick:
            za, zb, pa, pb = _pair(a, b, ia[r], ib[r])
            for m in methods:
                t0 = time.time()
                perm = align(pa, pb, method=m, k=k)
                acc[m]["seconds"] += time.time() - t0
                rep = report(za, zb, pa, pb, perm)
                if m != "nn" and not rep["is_permutation"]:
                    raise SystemExit(f"[align] {m} did not return a permutation on row {r}")
                for key in ("cosine", "matched_frac", "dist_matched", "dist_unmatched"):
                    acc[m][key].append(rep[key])

    out = {"n": len(pick), "k": k, "region_mix": {str(g): c for g, c in sorted(mix.items())},
           "methods": {}}
    for m in methods:
        out["methods"][m] = {
            "cosine": float(np.mean(acc[m]["cosine"])),
            "cosine_std": float(np.std(acc[m]["cosine"])),
            "matched_frac": float(np.mean(acc[m]["matched_frac"])),
            "dist_matched": float(np.nanmean(acc[m]["dist_matched"])),
            "dist_unmatched": float(np.nanmean(acc[m]["dist_unmatched"])),
            "seconds_per_building": acc[m]["seconds"] / max(1, len(pick)),
            "published": PUBLISHED.get(m),
        }
    return out


def sweep_k(real: str, blockout: str, n: int, ks) -> dict:
    import h5py

    pick, ia, ib, _ = _rows(real, blockout, n)
    out = {}
    with h5py.File(real, "r") as a, h5py.File(blockout, "r") as b:
        for k in ks:
            cos = []
            for r in pick:
                za, zb, pa, pb = _pair(a, b, ia[r], ib[r])
                cos.append(report(za, zb, pa, pb, align(pa, pb, "greedy", k))["cosine"])
            out[k] = float(np.mean(cos))
            print(f"  k={k:5d}  cosine {out[k]:.4f}"
                  + (f"   (lost run: {PUBLISHED_K[k]:.4f})" if k in PUBLISHED_K else ""))
    return out


def main() -> None:
    check_numpy()
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--blockout", required=True)
    ap.add_argument("--n", type=int, default=0, help="0 = every row both caches share")
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--methods", nargs="*", default=list(METHODS))
    ap.add_argument("--sweep_k", action="store_true")
    ap.add_argument("--ks", nargs="*", type=int, default=[16, 64, 256, 2048])
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    art = {"real": args.real, "blockout": args.blockout}
    if args.sweep_k:
        print("[align] greedy, by candidate-list size k:")
        art["k_sweep"] = sweep_k(args.real, args.blockout, args.n, args.ks)
    else:
        art.update(run(args.real, args.blockout, args.n, args.k, args.methods))
        print(f"\n{'method':11s} {'cosine':>8s} {'lost run':>9s} {'matched':>8s} "
              f"{'d(matched)':>11s} {'d(rest)':>8s} {'s/bldg':>7s}")
        for m, v in art["methods"].items():
            pub = f"{v['published']:.4f}" if v["published"] is not None else "-"
            print(f"{m:11s} {v['cosine']:8.4f} {pub:>9s} {v['matched_frac']:8.1%} "
                  f"{v['dist_matched']:11.4f} {v['dist_unmatched']:8.4f} "
                  f"{v['seconds_per_building']:7.2f}")
        print("\n⚠️ `nn` is many-to-one -- a bound on what alignment could recover, not a method.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(art, indent=2))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
