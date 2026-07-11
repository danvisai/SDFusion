"""Freeze deterministic, class-stratified, nested BuildingNet splits (ticket 03 / I0.1).

Produces a sealed test set and nested `train_25 ⊂ train_50 ⊂ train_100` id lists so every
downstream arm (real monolith pairs, per-fraction retrieval libraries) consumes the SAME detail
data. Split ids come from the same universe the element-library builder enumerates
(`model_data/obj/component_labels/*_label.json`), so a held-out test building can be excluded from
both training pairs and retrieval libraries.

Run from the repo root:
  env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
      scripts/foundations/make_splits.py --seed 0 --test-frac 0.15 --out data/splits_v1
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CLBL = REPO / "data/BuildingNet_dataset_v0_1/model_data/obj/component_labels"

FRACTIONS = ((0.25, "train_25"), (0.50, "train_50"), (1.00, "train_100"))


def parse_class(name: str) -> str:
    """Top-level BuildingNet class = the leading uppercase run.

    `COMMERCIALcastle_mesh0365` -> `COMMERCIAL`; `RESIDENTIALhouse_mesh1234` -> `RESIDENTIAL`.
    """
    m = re.match(r"^[A-Z]+", name)
    return m.group(0) if m else "UNKNOWN"


def enumerate_buildingnet():
    """`(id, top-level class)` for every labeled BuildingNet building, sorted by id."""
    names = sorted(p.stem[: -len("_label")] for p in CLBL.glob("*_label.json"))
    return [(n, parse_class(n)) for n in names]


def make_splits(items, seed: int = 0, test_frac: float = 0.15) -> dict:
    """Class-stratified sealed test set + nested train fractions.

    `items`: iterable of `(id, class)`. Deterministic in `(items, seed)` — the shuffle uses one
    seeded generator over classes and ids taken in sorted order, so input order does not matter.
    Fractions are nested prefixes of each class's post-test train list, so
    `train_25 ⊂ train_50 ⊂ train_100` and every class is represented in each fraction.
    """
    by_class: dict[str, list[str]] = defaultdict(list)
    for iid, cls in sorted(items):
        by_class[cls].append(iid)

    rng = np.random.default_rng(seed)
    out: dict[str, list[str]] = {"test": [], **{name: [] for _, name in FRACTIONS}}
    for cls in sorted(by_class):
        ids = sorted(by_class[cls])
        ids = [ids[i] for i in rng.permutation(len(ids))]
        n_test = round(test_frac * len(ids))
        out["test"].extend(ids[:n_test])
        train = ids[n_test:]
        for frac, name in FRACTIONS:
            out[name].extend(train[: round(frac * len(train))])
    return {k: sorted(v) for k, v in out.items()}


def _class_balance(ids):
    c = Counter(parse_class(i) for i in ids)
    return dict(sorted(c.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--out", default=str(REPO / "data/splits_v1"))
    ap.add_argument("--limit", type=int, default=0, help="debug: cap the universe size")
    a = ap.parse_args()

    items = enumerate_buildingnet()
    if a.limit:
        items = items[: a.limit]
    if not items:
        raise SystemExit(f"no BuildingNet buildings found under {CLBL}")

    splits = make_splits(items, seed=a.seed, test_frac=a.test_frac)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, ids in splits.items():
        (out / f"{name}.json").write_text(json.dumps(ids, indent=0))

    manifest = {
        "seed": a.seed,
        "test_frac": a.test_frac,
        "total_buildings": len(items),
        "counts": {k: len(v) for k, v in splits.items()},
        "class_balance": {k: _class_balance(v) for k, v in splits.items()},
        "universe_class_balance": _class_balance([i for i, _ in items]),
        "invariants": {
            "sealed_test_disjoint": not (set(splits["test"]) & set(splits["train_100"])),
            "nested": set(splits["train_25"]) <= set(splits["train_50"]) <= set(splits["train_100"]),
            "full_coverage": set(splits["test"]) | set(splits["train_100"]) == {i for i, _ in items},
        },
        "command": f"scripts/foundations/make_splits.py --seed {a.seed} "
                   f"--test-frac {a.test_frac} --out {a.out}",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    assert all(manifest["invariants"].values()), "split invariants failed"


if __name__ == "__main__":
    main()
