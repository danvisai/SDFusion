"""#183 arms 2-3: score a trained height-map checkpoint against #178's signed-off BuildingWorld
bar (#170's PASS/GUARD/KILL floors on the 9-city held-out population).

This is the SECOND of #183's two required gates. The first (the pinned-714 regression guard
against arm 1's noise band) is `train_height_map_generator.py`'s own CLI -- exactly what arm 1 and
arm 2 already ran. This script is the other one: it reuses #178's own query population (`#178's
BuildingWorld held-out set) and its own already-signed-off `registration`, but scores the
CHECKPOINT's predictions instead of the 1-NN baseline's transplanted donor.
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

from scripts.foundations.buildingworld_baseline import (  # noqa: E402
    load_population, population_report, buildingworld_verdict,
)
from scripts.foundations.corpus_ledger import LEDGER_PATH, read_ledger  # noqa: E402
from scripts.foundations.eval_massing_arms import RES  # noqa: E402
from scripts.foundations.recover_program_labels_buildingworld import OUT as LABELS_PATH  # noqa: E402
from scripts.foundations.train_height_map_generator import (  # noqa: E402
    _build_cache_batch, cache_batch_size, load_checkpoint, predict, score_arm,
)
from utils.frozen_corpus import REAL_CORPUS_PATH, open_real_corpus  # noqa: E402

REGISTRATION_PATH = REPO / "execution/artifacts/buildingworld_baseline_178.json"


def build_held(h5_path: Path, ledger_path: Path, rows: np.ndarray, workers: int = 0) -> dict:
    """The GT height-field + conditioning population `predict`/`score_arm` need, for exactly
    #178's query rows. Reuses `_build_cache_batch` (parallel, spawn-context, same as `build_cache`)
    rather than a second, drifting copy of the same per-row extraction.
    """
    import multiprocessing as mp

    n = len(rows)
    workers = workers or min(mp.cpu_count(), 48)
    batch = cache_batch_size(n, workers)
    jobs = [(h5_path, rows[i:i + batch]) for i in range(0, n, batch)]
    fps = np.zeros((n, RES, RES), bool)
    targets = np.zeros((n, RES, RES), np.int16)
    y0s = np.zeros(n, np.int32)
    extents = np.zeros(n, np.int32)
    k = 0
    with mp.get_context("spawn").Pool(workers) as pool:
        for results in pool.imap(_build_cache_batch, jobs):
            for r in results:
                if r is None:
                    raise ValueError(f"#183: invalid height field at row {int(rows[k])}; "
                                     f"#178's query rows must all be valid -- no silent skip")
                fp, y0, extent, target = r
                fps[k], targets[k], y0s[k], extents[k] = fp, target, y0, extent
                k += 1
    assert k == n, f"#183: processed {k} rows, expected {n}"
    ledger = read_ledger(ledger_path)
    row_to_region = {int(r): int(reg) for r, reg in zip(ledger["row"], ledger["region"])}
    row_to_height = {int(r): float(h) for r, h in zip(ledger["row"], ledger["height_m"])}
    return dict(row=rows.astype(np.int64), fp=fps, target=targets, y0=y0s, extent=extents,
               region=np.array([row_to_region[int(r)] for r in rows], np.int32),
               height_m=np.array([row_to_height[int(r)] for r in rows], np.float32))


def run(ckpt: Path, h5_path: Path = REAL_CORPUS_PATH, ledger_path: Path = LEDGER_PATH,
        labels_path: Path = LABELS_PATH, registration_path: Path = REGISTRATION_PATH,
        quantile: float | None = 0.5, cpu: bool = False, workers: int = 0) -> dict:
    """Score `ckpt` on #178's BuildingWorld held-out population and verdict it against the
    already-signed-off registration -- never recomputes or relaxes that registration itself.
    """
    population = load_population(h5_path, ledger_path, labels_path)
    query = population["query"]
    held = build_held(h5_path, ledger_path, query, workers=workers)
    d = load_checkpoint(ckpt)  # #163 provenance gate; a stale-mapping checkpoint fails loudly here
    if d["objective"] != "ce":
        raise ValueError(f"#183: {ckpt} is a {d['objective']!r} arm; this gate scores 'ce' arms")
    heights, meta = predict(ckpt, held, cpu=cpu, quantile=quantile)
    rows = score_arm(heights, held)
    for row, city, family in zip(rows, population["city"], population["family"]):
        row.update(city=str(city), family=str(family))
    report = population_report(rows)
    registration = json.loads(registration_path.read_text())["registration"]
    verdict = buildingworld_verdict(report, registration)
    return dict(checkpoint=str(ckpt), decode=meta.get("decode"), report=report, verdict=verdict,
               n_query=len(query))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--quantile", type=float, default=0.5)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    result = run(args.ckpt, quantile=args.quantile, cpu=args.cpu, workers=args.workers)
    print(f"[buildingworld-gate] {args.ckpt}  n={result['n_query']}  "
         f"numeric_pass={result['verdict']['numeric_pass']}", flush=True)
    for name, v in result["verdict"]["floors"].items():
        print(f"  {name:16s} pass={v.get('pass_')}"
             + (" (guard_only)" if v.get("guard_only") else ""), flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(result, open(args.out, "w"), indent=1)
        print(f"[artifact] {args.out}", flush=True)


if __name__ == "__main__":
    main()
