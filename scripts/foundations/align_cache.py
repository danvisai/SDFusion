"""#91: write the blockout cache in the real partner's token order, with a write-time guard.

The aligned cache is a **permutation of the same tokens**, so it is derived from an existing cache
with no second encode -- which is what makes #92's two arms *literally the same numbers in a
different order*, differing in token order and nothing else. Encoding a second time instead would
resample the surface draw and reintroduce the confound the map was chartered to remove.

    align_cache.py --real R.h5 --blockout B.h5 --out ALIGNED.h5
    align_cache.py --verify_only ALIGNED.h5 --real R.h5 --blockout B.h5

⚠️ **The guards are the point, and the first version of them proved nothing.** On the lost run,
review found the identity check read a loop variable *after* the loop (so it tested one row) and the
permutation check compared 131,072 *scalars* instead of 2,048 *token vectors* -- which passes for any
per-column shuffle. Both forms are re-implemented here against the review's correction, and
`--verify_only` re-runs them from disk in minutes rather than rebuilding.
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

from models.token_alignment import align  # noqa: E402
from utils.numeric_guard import check_numpy  # noqa: E402

#: what the lost run measured on the full corpus, kept beside the new numbers
PUBLISHED = {"unaligned_pct_random": 99.7, "aligned_pct_random": 13.7,
             "pos_elementwise": 1.1138, "pos_aligned": 0.1904, "pos_matched_floor": 0.0423}


def _pos_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, np.float32) - np.asarray(b, np.float32), axis=-1).mean())


def build(real: str, blockout: str, out: str, k: int, method: str) -> dict:
    import h5py

    t0 = time.time()
    with h5py.File(real, "r") as a, h5py.File(blockout, "r") as b:
        for f, name in ((a, real), (b, blockout)):
            if "query_pos" not in f:
                raise SystemExit(f"{name} has no query_pos -- alignment needs the token positions "
                                 "(#88); they cannot be recovered after the fact")
        ra, rb = np.asarray(a["row"]), np.asarray(b["row"])
        ia = {int(r): i for i, r in enumerate(ra)}
        common = [int(r) for r in rb if int(r) in ia]
        if not common:
            raise SystemExit("the two caches share no rows")
        print(f"[align] {len(common)} rows, {method} k={k} -> {out}")

        keep = [ia[r] for r in common]                      # index into the real cache
        src = [i for i, r in enumerate(rb) if int(r) in ia]  # index into the blockout cache
        lat = np.empty((len(common),) + b["latent"].shape[1:], np.float16)
        pos = np.empty((len(common),) + b["query_pos"].shape[1:], np.float16)
        for n, (i, j) in enumerate(zip(keep, src)):
            pa = np.array(a["query_pos"][i], np.float32)
            pb = np.array(b["query_pos"][j], np.float32)
            perm = align(pa, pb, method=method, k=k)
            lat[n] = np.array(b["latent"][j])[perm]
            pos[n] = np.array(b["query_pos"][j])[perm]
            if (n + 1) % 500 == 0:
                el = time.time() - t0
                print(f"  {n+1}/{len(common)}  {el:.0f}s  eta {el/(n+1)*(len(common)-n-1):.0f}s",
                      flush=True)

        with h5py.File(out, "w") as o:
            o.create_dataset("latent", data=lat, compression="lzf")
            o.create_dataset("query_pos", data=pos, compression="lzf")
            for name in ("footprint", "height_m", "region", "row", "held_out"):
                if name in b:
                    o.create_dataset(name, data=np.asarray(b[name])[src],
                                     compression="lzf" if name == "footprint" else None)
            for key, val in b.attrs.items():
                o.attrs[key] = val
            o.attrs["aligned_to"] = str(real)
            o.attrs["alignment"] = f"{method}@k={k}"
    print(f"[align] built in {time.time()-t0:.0f}s")
    return verify(out, real, blockout)


def verify(aligned: str, real: str, blockout: str, n: int = 0) -> dict:
    """Read the three caches back and prove the aligned one is a reordering, not a rewrite."""
    import h5py

    with h5py.File(aligned, "r") as c, h5py.File(real, "r") as a, h5py.File(blockout, "r") as b:
        rc = [int(r) for r in c["row"]]
        ia = {int(r): i for i, r in enumerate(np.asarray(a["row"]))}
        ib = {int(r): i for i, r in enumerate(np.asarray(b["row"]))}
        rows = rc[:n] if n else rc

        perm_ok = reordered = 0
        d_unaligned, d_aligned, d_floor, followed = [], [], [], []
        for r in rows:
            i, j, kk = ia[r], ib[r], rc.index(r)
            zc = np.array(c["latent"][kk], np.float32)
            zb = np.array(b["latent"][j], np.float32)
            # ⚠️ Compare TOKEN VECTORS, not scalars. Sorting 131,072 scalars passes for any per-column
            # shuffle and proves nothing about token order -- the exact mistake review caught.
            if np.array_equal(zc[np.lexsort(zc.T)], zb[np.lexsort(zb.T)]):
                perm_ok += 1
            if not np.array_equal(zc, zb):
                reordered += 1

            pa = np.array(a["query_pos"][i], np.float32)
            pb = np.array(b["query_pos"][j], np.float32)
            pc = np.array(c["query_pos"][kk], np.float32)
            d_unaligned.append(_pos_dist(pa, pb))
            d_aligned.append(_pos_dist(pa, pc))
            followed.append(_pos_dist(pc, pb[align(pa, pb)]))
            from scipy.spatial import cKDTree
            d_floor.append(float(np.linalg.norm(pa - pb[cKDTree(pb).query(pa)[1]], axis=-1).mean()))

    res = {
        "n": len(rows),
        "token_permutation": f"{perm_ok}/{len(rows)}",
        "reordered": f"{reordered}/{len(rows)}",
        "pos_elementwise": float(np.mean(d_unaligned)),
        "pos_aligned": float(np.mean(d_aligned)),
        "pos_matched_floor": float(np.mean(d_floor)),
        "pos_followed": float(np.mean(followed)),
    }
    span = max(1e-9, res["pos_elementwise"] - res["pos_matched_floor"])
    res["aligned_pct_random"] = 100.0 * (res["pos_aligned"] - res["pos_matched_floor"]) / span
    res["unaligned_pct_random"] = 100.0

    print(f"\n[verify] {res['n']} rows")
    print(f"  token-wise permutations : {res['token_permutation']}   reordered: {res['reordered']}")
    print(f"  query-position distance : elementwise {res['pos_elementwise']:.4f} -> aligned "
          f"{res['pos_aligned']:.4f}   (nn floor {res['pos_matched_floor']:.4f})")
    print(f"  aligned is {res['aligned_pct_random']:.1f}% of the way to a random pairing "
          f"(lost run: {PUBLISHED['aligned_pct_random']}%)")
    print(f"  positions follow the latents to {res['pos_followed']:.4f} (0 = exactly)")

    if perm_ok != len(rows):
        raise SystemExit("[verify] FAILED: some rows are not a permutation of their source")
    if reordered == 0:
        raise SystemExit("[verify] FAILED: nothing was reordered -- this is a copy, not an alignment")
    if res["pos_followed"] > 1e-6:
        raise SystemExit("[verify] FAILED: stored positions do not match the applied permutation")
    return res


def main() -> None:
    check_numpy()
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--blockout", required=True)
    ap.add_argument("--out")
    ap.add_argument("--verify_only", help="an existing aligned cache to re-check from disk")
    ap.add_argument("--verify_n", type=int, default=0, help="0 = every row")
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--method", default="greedy")
    ap.add_argument("--artifact", default=str(REPO / "execution/artifacts/align_cache.json"))
    args = ap.parse_args()

    if args.verify_only:
        res = verify(args.verify_only, args.real, args.blockout, args.verify_n)
    else:
        if not args.out:
            raise SystemExit("--out is required unless --verify_only is given")
        res = build(args.real, args.blockout, args.out, args.k, args.method)

    Path(args.artifact).parent.mkdir(parents=True, exist_ok=True)
    Path(args.artifact).write_text(json.dumps({"published": PUBLISHED, **res}, indent=2))
    print(f"-> {args.artifact}")


if __name__ == "__main__":
    main()
