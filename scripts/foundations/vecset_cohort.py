"""#188: which corpus rows the retrained A2 massing source trains on, and how many per region.

Two decisions live here, separated on purpose so each is testable without a GPU or a corpus:

  * `capped_proportional_quota` -- HOW MANY rows per region bucket.
  * `select_cohort`             -- WHICH rows, outcome-blind.

## Why a cohort exists at all

The height-map family's `corpus_scope="all"` means literally every ledger row, because its cache is
a cheap read of an SDF volume (1,562,401 rows in 18.5 min parallelized, #183). The vecset family
cannot copy that: a vecset latent costs a **Dora encode per row, measured at 207 ms on this box**,
and pair training needs a second encode of the blockout partner. Encoding all 1,526,778
BuildingWorld rows is ~88 GPU-hours for the real pass alone and ~412 GB of cache. So the vecset
retrain folds BuildingWorld in through a *sample*, and this module is where that sample is defined
rather than improvised at the call site.

⚠️ The legacy 35,623 rows are **not** sampled -- their latent and blockout caches already exist and
enter the retrain whole, so they stay the regression anchor. Only the BuildingWorld side is drawn.

## Capped-proportional, and why not the two obvious alternatives

The corpus is severely imbalanced: Canada 640,020 rows against Oceania 13,092, a 49x spread
(#171's buckets). Three policies were on the table, and the owner chose capped-proportional on
2026-09-18:

  * *strictly proportional* -- faithful, but Canada + Germany alone take ~72% of the draw and
    Oceania lands under 1%, so whole style families are present only as rounding.
  * *equal per bucket* -- maximum coverage of a distribution no real city has. It is also
    self-defeating here: the retrain is **region-free**, so the model cannot tell buckets apart and
    equalising them only reweights the geometry it sees toward rare styles.
  * *capped-proportional* -- proportional in the middle, clipped at both ends. Germany/Canada
    cannot swamp the cohort; Tokyo/Oceania stay representable.

⚠️ `cap_frac` and `floor_frac` are **#188's own local call, not a settled policy**.
[#168](https://github.com/danvisai/SDFusion/issues/168) (sampling cap / corpus-balance) is OPEN and
undecided; if it settles differently, this module is the one place to change, and any cohort drawn
here is reproducible from its salt and quota alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

ROLES = ("train", "gate")
MANIFEST = REPO / "execution/artifacts/188_cohort.json"

# #171's BuildingWorld style buckets (`source_provenance.BUILDINGWORLD_CITY_BUCKET`, v2 mapping).
# 0-2 are the legacy BAG/NL, NRW/DE and PLATEAU/JP sources and are never sampled -- see the module
# docstring.
BUILDINGWORLD_REGIONS = (3, 4, 5, 6, 7, 8)

CAP_FRAC = 0.25      # no single bucket may exceed a quarter of the BuildingWorld draw
FLOOR_FRAC = 0.05    # and none may fall under a twentieth of it
COHORT_SALT = "vecset-a2-retrain-188"


def capped_proportional_quota(sizes: dict, total: int, cap_frac: float = CAP_FRAC,
                              floor_frac: float = FLOOR_FRAC) -> dict:
    """Rows to draw per region: proportional to `sizes`, water-filled under a cap and a floor.

    Returns a dict summing to **exactly** `total`. A bucket holding fewer rows than its share gets
    all of them and the remainder is redistributed; a cap or floor that cannot reach `total` is
    refused rather than silently exceeded, because a quota that quietly breaks its own stated cap
    is worse than no cap.
    """
    sizes = {int(k): int(v) for k, v in sorted(sizes.items())}
    total = int(total)
    if not sizes:
        raise ValueError("no region buckets to draw from")
    if total <= 0:
        raise ValueError(f"total must be positive, got {total}")
    if any(v < 0 for v in sizes.values()):
        raise ValueError(f"region sizes must be non-negative: {sizes}")
    corpus = sum(sizes.values())
    if total > corpus:
        raise ValueError(f"asked for {total} rows but the buckets hold {corpus}")
    if not 0 < cap_frac <= 1:
        raise ValueError(f"cap_frac must be in (0, 1], got {cap_frac}")
    if not 0 <= floor_frac <= cap_frac:
        raise ValueError(f"floor_frac must be in [0, cap_frac], got {floor_frac}")

    cap, floor = int(cap_frac * total), int(floor_frac * total)
    hi = {r: min(cap, sizes[r]) for r in sizes}
    lo = {r: min(floor, sizes[r]) for r in sizes}
    if sum(hi.values()) < total:
        raise ValueError(
            f"cap_frac={cap_frac} caps the draw at {sum(hi.values())} rows across "
            f"{len(sizes)} buckets, which cannot reach {total}; raise cap_frac or lower the total")
    if sum(lo.values()) > total:
        raise ValueError(
            f"floor_frac={floor_frac} already demands {sum(lo.values())} rows, more than {total}")

    # Water-fill: allocate proportionally over the still-free buckets, pin whichever ones that
    # allocation pushes past a bound, and repeat. Caps are settled before floors so a bucket cannot
    # be pinned low and then re-opened by the redistribution that pinning caused.
    fixed: dict = {}
    pool = [r for r in sizes]
    remaining = total
    while pool:
        weight = sum(sizes[r] for r in pool)
        if weight <= 0:
            break
        alloc = {r: remaining * sizes[r] / weight for r in pool}
        over = [r for r in pool if alloc[r] > hi[r]]
        under = [r for r in pool if alloc[r] < lo[r]]
        hit = over or under
        if not hit:
            break
        bound = hi if over else lo
        for r in hit:
            fixed[r] = bound[r]
            remaining -= bound[r]
            pool.remove(r)

    # Largest-remainder rounding over whatever stayed free, so the integer quota still sums exactly.
    if pool:
        weight = sum(sizes[r] for r in pool)
        exact = {r: (remaining * sizes[r] / weight if weight else remaining / len(pool))
                 for r in pool}
        base = {r: int(exact[r]) for r in pool}
        short = remaining - sum(base.values())
        # ties broken by region id, so the result cannot depend on dict ordering
        for r in sorted(pool, key=lambda r: (-(exact[r] - base[r]), r))[:max(short, 0)]:
            base[r] += 1
        fixed.update(base)

    return _repair_residual(fixed, hi, lo, total)


def _repair_residual(quota: dict, hi: dict, lo: dict, total: int) -> dict:
    """Push any rounding residue into buckets that have slack, deterministically.

    The water-fill plus largest-remainder pass lands on `total` in every case the tests exercise;
    this exists so that a configuration which does not is corrected rather than returning a quota
    whose sum silently disagrees with what the caller asked for.
    """
    quota = dict(sorted(quota.items()))
    for _ in range(len(quota) + 1):
        residue = total - sum(quota.values())
        if residue == 0:
            return quota
        step = 1 if residue > 0 else -1
        slack = [r for r in quota
                 if (quota[r] < hi[r] if residue > 0 else quota[r] > lo[r])]
        if not slack:
            break
        for r in sorted(slack, key=lambda r: (-hi[r], r)):
            if residue == 0:
                break
            quota[r] += step
            residue -= step
    if sum(quota.values()) != total:
        raise ValueError(f"could not reach {total} rows under the cap/floor: {quota}")
    return quota


def row_hash(row: int, salt: str) -> int:
    """A stable 64-bit hash of a corpus row id under `salt`.

    `hashlib`, not `hash()`: Python salts `hash()` per process, so a cohort drawn with it would be
    a different cohort on every run and no checkpoint could name the rows it trained on.
    """
    digest = hashlib.blake2b(f"{salt}:{int(row)}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def select_cohort(rows, regions, held_out, quota: dict, salt: str = COHORT_SALT) -> np.ndarray:
    """The `quota`-many TRAINABLE rows per region, chosen by salted row hash. Ascending.

    **Outcome-blind, per #115's surviving method.** The only input to the choice is the row id and
    the salt -- never source, city, geometry, difficulty, or any measured outcome. Held-out rows
    (#177's roof-family-stratified, spatially blocked split) are excluded, so the population the
    gate scores on cannot enter training.

    Ordering by hash rather than shuffling buys one property worth naming: **a larger quota is a
    superset of a smaller one**. Topping the cohort up later extends the encode instead of
    invalidating the rows already paid for at ~0.2 s each, and any checkpoint trained on the
    earlier draw stays interpretable.
    """
    return _draw(rows, regions, held_out, quota, salt, held=False)


def select_gate_rows(rows, regions, held_out, quota: dict, salt: str = COHORT_SALT) -> np.ndarray:
    """The `quota`-many HELD-OUT rows per region -- the population #188's bar is scored on.

    The mirror of `select_cohort`, and deliberately a separate name rather than a boolean flag:
    these two draw from complementary halves of the corpus, so the one property that matters most
    -- that a gate row can never be a training row -- is true by construction and readable at every
    call site.
    """
    return _draw(rows, regions, held_out, quota, salt, held=True)


def _draw(rows, regions, held_out, quota: dict, salt: str, held: bool) -> np.ndarray:
    rows = np.asarray(rows).astype(np.int64, copy=False)
    regions = np.asarray(regions).astype(np.int64, copy=False)
    held_out = np.asarray(held_out).astype(np.int64, copy=False)
    if not (len(rows) == len(regions) == len(held_out)):
        raise ValueError(f"ragged ledger columns: {len(rows)}, {len(regions)}, {len(held_out)}")
    eligible = (held_out == 1) if held else (held_out == 0)
    pool = "held-out" if held else "trainable"

    picked = []
    for region in sorted(quota):
        want = int(quota[region])
        if want < 0:
            raise ValueError(f"region {region} asked for {want} rows")
        candidates = rows[(regions == int(region)) & eligible]
        if want > len(candidates):
            raise ValueError(
                f"region {region}: asked for {want} rows but only {len(candidates)} are {pool}")
        if want == 0:
            continue
        h = np.fromiter((row_hash(r, salt) for r in candidates), dtype=np.uint64,
                        count=len(candidates))
        # lexsort's LAST key is primary: hash first, row id as the tie-break, so the result cannot
        # depend on the order the ledger happened to hand these rows over in
        order = np.lexsort((candidates, h))
        picked.append(candidates[order[:want]])

    if not picked:
        return np.empty(0, np.int64)
    return np.sort(np.concatenate(picked))


def row_digest(rows) -> str:
    """SHA-256 over the ascending, deduplicated row stream.

    Order-independent by construction (the stream is sorted first), so a manifest and a cache that
    hold the same rows in different orders agree -- while any change to *which* rows are named
    changes the digest.
    """
    stream = np.unique(np.asarray(rows, np.int64))
    return hashlib.sha256(stream.tobytes()).hexdigest()


def build_manifest(rows, regions, held_out, train_n: int, gate_n: int,
                   salt: str = COHORT_SALT, cap_frac: float = CAP_FRAC,
                   floor_frac: float = FLOOR_FRAC, extra_meta: dict | None = None) -> dict:
    """The committed cohort: quotas, row lists and digests for both roles.

    Both roles draw from #171's six BuildingWorld buckets only. The legacy 35,623 rows are not
    sampled -- they enter the retrain whole from the caches that already exist, and the gate's
    legacy control is the pinned 714, unchanged.
    """
    rows = np.asarray(rows, np.int64)
    regions = np.asarray(regions, np.int64)
    held_out = np.asarray(held_out, np.int64)

    trainable = {r: int(((regions == r) & (held_out == 0)).sum()) for r in BUILDINGWORLD_REGIONS}
    heldout_sizes = {r: int(((regions == r) & (held_out == 1)).sum()) for r in BUILDINGWORLD_REGIONS}

    train_quota = capped_proportional_quota(trainable, train_n, cap_frac, floor_frac)
    gate_quota = capped_proportional_quota(heldout_sizes, gate_n, cap_frac, floor_frac)
    train_rows = select_cohort(rows, regions, held_out, train_quota, salt)
    gate_rows = select_gate_rows(rows, regions, held_out, gate_quota, salt)

    overlap = set(train_rows.tolist()) & set(gate_rows.tolist())
    if overlap:   # impossible by construction; asserted because it is the leakage property
        raise ValueError(f"{len(overlap)} rows are in both the training cohort and the gate")

    meta = {
        "ticket": 188,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_rev": _git_rev(),
        "salt": salt,
        "cap_frac": cap_frac,
        "floor_frac": floor_frac,
        "policy": "capped-proportional over #171's BuildingWorld style buckets; provisional "
                  "pending #168",
        "buildingworld_trainable": trainable,
        "buildingworld_held_out": heldout_sizes,
    }
    meta.update(extra_meta or {})
    return {
        "meta": meta,
        "train": {"n": int(len(train_rows)), "quota": {str(k): v for k, v in train_quota.items()},
                  "digest": row_digest(train_rows), "rows": [int(r) for r in train_rows]},
        "gate": {"n": int(len(gate_rows)), "quota": {str(k): v for k, v in gate_quota.items()},
                 "digest": row_digest(gate_rows), "rows": [int(r) for r in gate_rows]},
    }


def load_manifest(path, role: str) -> np.ndarray:
    """One role's rows from a committed manifest, refusing a row list its digest disowns.

    The digest check is not ceremony. The encode this feeds costs ~8 GPU-hours and the training run
    ~27 more; a manifest edited by hand between those two steps would produce a checkpoint whose
    stated cohort is not the one it saw, and nothing downstream could tell.
    """
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, got {role!r}")
    manifest = json.loads(Path(path).read_text())
    if role not in manifest:
        raise ValueError(f"{path} has no {role!r} role (has {sorted(set(manifest) - {'meta'})})")
    block = manifest[role]
    rows = np.asarray(block["rows"], np.int64)
    got = row_digest(rows)
    if got != block["digest"]:
        raise ValueError(
            f"{path} role {role!r}: the row list does not match its own digest "
            f"({got[:12]}... vs {block['digest'][:12]}...) -- the manifest has been edited")
    if len(rows) != int(block["n"]):
        raise ValueError(f"{path} role {role!r}: claims n={block['n']} but names {len(rows)} rows")
    return rows


def _git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                                       text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description="#188: draw and commit the vecset retrain cohort")
    ap.add_argument("--ledger", default=None, help="corpus ledger (default: the #161 ledger)")
    ap.add_argument("--train_n", type=int, default=60_000,
                    help="BuildingWorld rows to encode for training. The binding constraint is GPU "
                         "time, not the corpus: a vecset latent costs ~207 ms and pair training "
                         "needs a blockout partner too, so 60,000 rows is ~8 GPU-hours of encode.")
    ap.add_argument("--gate_n", type=int, default=900,
                    help="held-out BuildingWorld rows the bar is scored on (never trained on)")
    ap.add_argument("--salt", default=COHORT_SALT)
    ap.add_argument("--cap_frac", type=float, default=CAP_FRAC)
    ap.add_argument("--floor_frac", type=float, default=FLOOR_FRAC)
    ap.add_argument("--out", default=str(MANIFEST))
    args = ap.parse_args()

    from scripts.foundations.corpus_ledger import LEDGER_PATH, read_ledger

    ledger_path = Path(args.ledger) if args.ledger else LEDGER_PATH
    ledger = read_ledger(ledger_path)
    manifest = build_manifest(
        ledger["row"], ledger["region"], ledger["held_out"],
        train_n=args.train_n, gate_n=args.gate_n, salt=args.salt,
        cap_frac=args.cap_frac, floor_frac=args.floor_frac,
        extra_meta={"ledger": str(ledger_path), "ledger_rows": int(len(ledger["row"]))})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=1))
    for role in ROLES:
        block = manifest[role]
        print(f"[{role}] {block['n']:,} rows  digest {block['digest'][:16]}...")
        for region, n in sorted(block["quota"].items(), key=lambda kv: int(kv[0])):
            print(f"    region {region}: {n:,}")
    print(f"[cohort] -> {out}")


if __name__ == "__main__":
    main()
