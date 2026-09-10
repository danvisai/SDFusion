"""#153 -- region- and tile-stratified train/val/test split over real.h5.

#5's audit found `datasets/bag3d_dataset.py`'s existing split (a row-random
`np.random.default_rng(0).permutation(n_total)`) has two faults: it has no
`source_id` stratification (today's NL/DE/JP balance in each split is incidental),
and it holds out individual building ROWS rather than spatial groups, so two
buildings from the same city block/CityGML tile -- not independent draws, since
roof family and height rhythm correlate structurally within a block -- can land on
both sides of the train/held-out boundary.

This module is a NEW, separate split. It deliberately does NOT replace
`Bag3dDataset`'s existing split or `scripts/foundations/vecset_ceiling_probe.py`'s
`test_indices()`: dozens of already-closed massing tickets (#10, #92's four arms,
#126, #127, #129 onward) recorded their numbers against that exact frozen
714/411-id population, and #162 (open) is specifically about pinning that
population's row identity ahead of BuildingWorld ingestion. Retroactively
reshuffling it here would silently invalidate that whole evidentiary record for a
ticket scoped as "a data-generation change, independent of any model." #181 already
names this new split as what IT will run against, prospectively -- see
`docs/wayfinding/solid-first-subtractive-modeling/153-stratified-split.md` for the
full writeup and the achieved numbers.

Tile-key derivation (verified against the live corpus, no re-ingestion needed):
  NL (3D BAG):        bag_id = 'NL.IMBAG.Pand.<gemeentecode><...>' -- the official
                       Dutch cadastral id; the first 4 digits after 'Pand.' are the
                       CBS gemeente (municipality) code. `ingest_3dbag.py` drew from
                       5 distinct ~1km^2 city-centre bboxes, one per gemeente --
                       confirmed exactly 5 distinct codes across all 11,776 NL rows.
  DE (NRW) / JP (PLATEAU) CityGML: `ingest_citygml_lod2.py` already writes
                       `bag_id = f"{gml_key}#{gid}"[:64]`, so the source .gml tile
                       filename is the bag_id prefix before '#' -- read directly off
                       the id, truncation-safe since it comes first in the string.

Run: env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
     scripts/foundations/stratified_split.py [--h5 data/real_massing_v1/real.h5]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.frozen_corpus import open_real_corpus  # noqa: E402

SOURCE_NAMES = {0: "NL", 1: "DE", 2: "JP"}
NL_PREFIX = "NL.IMBAG.Pand."


def tile_key(source_id: int, bag_id: str) -> str:
    """The spatial grouping key a row's own source already encodes in its id."""
    name = SOURCE_NAMES.get(int(source_id), str(source_id))
    if source_id == 0:
        if not bag_id.startswith(NL_PREFIX) or len(bag_id) < len(NL_PREFIX) + 4:
            raise ValueError(f"unrecognized NL bag_id format: {bag_id!r}")
        return f"{name}:{bag_id[len(NL_PREFIX):len(NL_PREFIX) + 4]}"
    if "#" not in bag_id:
        raise ValueError(f"unrecognized DE/JP bag_id format (no '#' tile separator): {bag_id!r}")
    return f"{name}:{bag_id.split('#', 1)[0]}"


def _closest_subset(target: float, counts: np.ndarray, min_items: int = 1) -> list[int]:
    """Indices of the whole-tile subset whose total count is CLOSEST to `target`.

    Exact 0/1 subset-sum DP, not greedy: achievable sums are integers in [0, sum(counts)],
    so the reachable-sum table has at most sum(counts)+1 entries regardless of tile count
    (<=41 tiles here) -- cheap. Ties keep the earliest-found (lowest achievable sum, then
    input order) for determinism.

    `min_items=1` (the default, and what every caller here uses) excludes the empty
    subset from consideration whenever `counts` is nonempty: the empty subset's sum (0)
    is numerically closer to a small `target` than any single real tile whenever every
    tile is individually larger than the target -- true for NL, whose smallest tile
    (858) is 3.6x the naive 2% target. Picking "closest" without this constraint zeroes
    out that region's held-out representation entirely, defeating stratification's own
    point. Forcing >=1 tile trades exact closeness for actually representing the region.
    """
    counts_i = [int(c) for c in counts]
    reachable: dict[int, tuple[int, ...]] = {0: ()}
    for idx, c in enumerate(counts_i):
        for s, path in list(reachable.items()):
            ns = s + c
            if ns not in reachable:
                reachable[ns] = path + (idx,)
    candidates = {s: p for s, p in reachable.items() if len(p) >= min_items} or reachable
    best_sum = min(candidates, key=lambda s: (abs(s - target), s))
    return list(candidates[best_sum])


def _region_tiles(region_keys: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Unique tiles in a region, tie-broken then sorted smallest-count-first."""
    uniq, counts = np.unique(region_keys, return_counts=True)
    tie_order = rng.permutation(len(uniq))
    uniq, counts = uniq[tie_order], counts[tie_order]
    size_order = np.argsort(counts, kind="stable")
    return uniq[size_order], counts[size_order]


def make_split(
    source_id: np.ndarray,
    bag_id: np.ndarray,
    val_frac: float = 0.02,
    test_frac: float = 0.02,
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """Whole-tile, per-region-stratified split. Returns (split labels, report dict).

    Tiles, never individual rows, are assigned to val/test, so no two buildings from
    the same tile ever land on opposite sides of the train/held-out boundary --
    structurally, by construction, not by a post-hoc check.

    Naively targeting `val_frac`/`test_frac` of EACH region independently fails
    acceptance criterion 1 whenever one region's tiles are coarse: NL has only 5
    tiles (the 5 city-centre bboxes `ingest_3dbag.py` drew from), so its smallest
    single tile (858 buildings) is already 7.3% of NL alone -- 3.6x the 2% target --
    and an independent-per-region fill makes val/test end up 50-75% NL, reproducing
    (inverted) exactly the region-imbalance bug #5/#127 already found and patched
    around in `eval_massing_arms.py`'s `pick_ids`.

    Instead: compute each region's naive fraction-targeted achievable count first: this
    is a genuine per-region constraint, since tile size is fixed by whichever source's
    ingestion tiling produced it (5 bboxes for NL, 41 CityGML tiles for DE, 9 for JP).
    The largest such count across regions becomes the shared ABSOLUTE target for val
    (and separately for test) -- i.e. the coarsest region sets the floor, and every
    OTHER region matches that same absolute count as closely as achievable (exact
    closest-subset-sum over its own whole tiles, not a greedy approximation) rather
    than independently chasing its own fraction. This makes NL/DE/JP's contributions
    to val (and to test) comparable in absolute size, at the cost of a larger overall
    held-out fraction than 2%/2% -- an honest trade disclosed in the achieved numbers,
    not hidden.
    """
    bag_id = np.asarray(
        [b.decode() if isinstance(b, (bytes, np.bytes_)) else b for b in bag_id]
    )
    source_id = np.asarray(source_id).astype(int)
    keys = np.array([tile_key(s, b) for s, b in zip(source_id, bag_id)])
    sids = sorted(set(source_id.tolist()))
    rng = np.random.default_rng(seed)

    # Pass 1: each region's own naive fraction-targeted achievable count sets the floor.
    tiles_by_region = {}
    naive_val_n, naive_test_n = {}, {}
    for sid in sids:
        region_keys = keys[source_id == sid]
        region_n = len(region_keys)
        uniq, counts = _region_tiles(region_keys, rng)
        tiles_by_region[sid] = (uniq, counts)
        val_i = _closest_subset(val_frac * region_n, counts)
        v_n = int(counts[val_i].sum()) if val_i else 0
        remaining = [i for i in range(len(uniq)) if i not in val_i]
        test_i = _closest_subset(test_frac * region_n, counts[remaining]) if remaining else []
        t_n = int(counts[[remaining[i] for i in test_i]].sum()) if test_i else 0
        naive_val_n[sid], naive_test_n[sid] = v_n, t_n
    shared_val_target = max(naive_val_n.values())
    shared_test_target = max(naive_test_n.values())

    # Pass 2: every region fills toward the SHARED absolute target (smallest tiles first).
    split = np.full(len(source_id), "train", dtype=object)
    report: dict = {}
    for sid in sids:
        region_mask = source_id == sid
        region_n = int(region_mask.sum())
        region_keys = keys[region_mask]
        uniq, counts = tiles_by_region[sid]

        val_i = _closest_subset(shared_val_target, counts)
        val_n = int(counts[val_i].sum()) if val_i else 0
        remaining = [i for i in range(len(uniq)) if i not in val_i]
        test_i_local = _closest_subset(shared_test_target, counts[remaining]) if remaining else []
        test_i = [remaining[i] for i in test_i_local]
        test_n = int(counts[test_i].sum()) if test_i else 0

        val_keys = set(uniq[val_i].tolist())
        test_keys = set(uniq[test_i].tolist())
        assert val_keys.isdisjoint(test_keys)

        region_split = np.where(
            np.isin(region_keys, list(val_keys)), "val",
            np.where(np.isin(region_keys, list(test_keys)), "test", "train"),
        )
        split[region_mask] = region_split
        name = SOURCE_NAMES.get(sid, str(sid))
        report[name] = {
            "n": region_n,
            "n_tiles": int(len(uniq)),
            "val_n": val_n, "val_frac_achieved": round(val_n / region_n, 4),
            "val_tiles": sorted(val_keys),
            "test_n": test_n, "test_frac_achieved": round(test_n / region_n, 4),
            "test_tiles": sorted(test_keys),
            "train_n": region_n - val_n - test_n,
        }
    report["_shared_targets"] = {"val_n": shared_val_target, "test_n": shared_test_target,
                                  "binding_region": max(naive_val_n, key=naive_val_n.get)}
    return split, report


def assert_no_tile_crosses_boundary(source_id: np.ndarray, bag_id: np.ndarray, split: np.ndarray) -> None:
    """Every tile's rows must fall entirely within one split label."""
    bag_id = np.asarray(
        [b.decode() if isinstance(b, (bytes, np.bytes_)) else b for b in bag_id]
    )
    source_id = np.asarray(source_id).astype(int)
    keys = np.array([tile_key(s, b) for s, b in zip(source_id, bag_id)])
    for k in np.unique(keys):
        labels = set(split[keys == k].tolist())
        assert len(labels) == 1, f"tile {k!r} spans multiple splits: {labels}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="data/real_massing_v1/real.h5")
    ap.add_argument("--val_frac", type=float, default=0.02)
    ap.add_argument("--test_frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the full report json here")
    args = ap.parse_args()

    with open_real_corpus(args.h5) as f:
        source_id = f["source_id"][:]
        bag_id = f["bag_id"][:]

    split, report = make_split(source_id, bag_id, args.val_frac, args.test_frac, args.seed)
    assert_no_tile_crosses_boundary(source_id, bag_id, split)

    pooled = {
        "n_total": int(len(split)),
        "train_n": int((split == "train").sum()),
        "val_n": int((split == "val").sum()),
        "test_n": int((split == "test").sum()),
    }
    print(json.dumps({"pooled": pooled, "per_region": report}, indent=2, default=str))
    if args.out:
        Path(args.out).write_text(json.dumps({"pooled": pooled, "per_region": report}, indent=2, default=str))
        print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
