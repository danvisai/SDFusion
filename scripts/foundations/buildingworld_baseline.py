"""#178: a separate BuildingWorld 1-NN baseline and #170's three-floor bar.

The train bank is every held_out=0 ledger row, in ascending corpus-row order, with
no source cap or reweighting (#168). Evaluation is the full held-out population
of #170's nine named cities, including flat roofs. #176's fitter-derived families
are pseudo-labels. The pinned-714 control is copied from its historical artifact,
never evaluated again. Numeric bars are proposals pending project-owner sign-off.

Only footprints are loaded for the full bank; SDFs are read for queries and their
selected donors. Exact duplicate footprints retain the FIRST bank row, just as
retrieve_nn's argmax does; they are not a new corpus deduplication policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.foundations.buildingworld_stratified_split import FAMILIES, city_for_row
from scripts.foundations.corpus_ledger import LEDGER_PATH, read_ledger
from scripts.foundations.recover_program_labels_buildingworld import (
    BUILDINGWORLD_SOURCE_ID, OUT as LABELS_PATH,
)
from scripts.foundations.recover_massing_programs import height_field, occupancy
from scripts.foundations.train_height_map_generator import (
    PROGRAM_BAR, retrieve_nn, score_arm, summarise, transplant_height,
)
from utils.frozen_corpus import (
    FROZEN_SPLIT_N_TOTAL, REAL_CORPUS_PATH, open_real_corpus,
)

BAR_CITIES = ("Berlin", "Boston", "Cambridge", "Cape Town", "Edmonton",
              "Melbourne", "Montreal", "New York", "Tokyo")
HISTORICAL = REPO / "execution/artifacts/height_map_generator_714.json"
OUT = REPO / "execution/artifacts/buildingworld_baseline_178.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_populations(ledger: dict, source_id: np.ndarray,
                       source_key: np.ndarray) -> dict:
    """Validate complete BW assignments and the old split, then select by row ID."""
    rows = ledger["row"]
    if np.any(rows < 0) or np.any(rows >= len(source_id)):
        raise ValueError("#178: ledger contains out-of-range corpus rows")
    bw = np.flatnonzero(source_id == BUILDINGWORLD_SOURCE_ID)
    if not np.array_equal(np.sort(rows[source_id[rows] == BUILDINGWORLD_SOURCE_ID]), bw):
        raise ValueError("#178: every BuildingWorld row needs an explicit #177 assignment")
    frozen_held = np.zeros(FROZEN_SPLIT_N_TOTAL, np.uint8)
    # Same frozen mechanism as Bag3dDataset and #162; checks flags, not scores.
    n_val = max(1, int(FROZEN_SPLIT_N_TOTAL * 0.02))
    frozen_held[np.random.default_rng(0).permutation(FROZEN_SPLIT_N_TOTAL)[n_val:2 * n_val]] = 1
    old = rows < FROZEN_SPLIT_N_TOTAL
    if not np.array_equal(ledger["held_out"][old], frozen_held[rows[old]]):
        raise ValueError("#178: historical ledger held_out flags differ from the frozen split")
    train = np.sort(rows[ledger["held_out"] == 0])
    held = np.sort(rows[(ledger["held_out"] == 1) &
                        (source_id[rows] == BUILDINGWORLD_SOURCE_ID)])
    cities = np.array([city_for_row(source_key[r]) for r in held])
    selected = np.isin(cities, BAR_CITIES)
    if not len(train) or not np.any(source_id[train] == BUILDINGWORLD_SOURCE_ID):
        raise ValueError("#178: expected a nonempty new train bank containing BuildingWorld")
    missing = set(BAR_CITIES) - set(cities[selected])
    if missing:
        raise ValueError(f"#178: missing required held-out cities: {sorted(missing)}")
    return dict(train=train, query=held[selected], city=cities[selected],
                excluded_cities={c: int(np.sum(cities == c))
                                 for c in sorted(set(cities[~selected]))})


def load_population(h5_path: Path, ledger_path: Path, labels_path: Path) -> dict:
    ledger = read_ledger(ledger_path)
    with open_real_corpus(h5_path) as corpus:
        sources = corpus["source_id"][:]
        population = select_populations(ledger, sources, corpus["source_key"][:])
        population["corpus_rows"] = len(sources)
        population["identity_sha256"] = hashlib.sha256(
            corpus["bag_id"][:].tobytes() + corpus["height_m"][:].tobytes()).hexdigest()
        population["train_source_counts"] = {
            str(s): int(np.sum(sources[population["train"]] == s)) for s in np.unique(sources)}
    with h5py.File(labels_path, "r") as labels:
        n = int(labels.attrs["committed_rows"])
        rows, ok = labels["row"][:n], labels["ok"][:n]
        families = labels["family"][:n].astype(str)
    if len(np.unique(rows)) != len(rows):
        raise ValueError("#178: duplicate program-label row IDs")
    lookup = {int(r): f for r, valid, f in zip(rows, ok, families) if valid}
    if any(lookup.get(int(r)) not in FAMILIES for r in population["query"]):
        raise ValueError("#178: every held-out row needs a valid committed family pseudo-label")
    population["family"] = np.array([lookup[int(r)] for r in population["query"]])
    return population


def exact_retrieval(query: np.ndarray, bank: np.ndarray, *, chunk: int = 64,
                    bank_chunk: int = 16384, device: str | None = None) -> np.ndarray:
    """Exact footprint-IoU 1-NN, with duplicate/exact-match shortcuts preserving ties."""
    if len(bank) == 0 or bank.shape[1:] != query.shape[1:]:
        raise ValueError("#178: nonempty bank and matching footprint shapes required")
    if not len(query):
        return np.empty(0, np.int64)
    if not bank.reshape(len(bank), -1).any(1).all() or not query.reshape(len(query), -1).any(1).all():
        raise ValueError("#178: empty footprint in train or held-out population")
    packed = np.packbits(bank.reshape(len(bank), -1), axis=1)
    # Void records compare all bytes (including trailing zero bytes) without Python objects.
    keys = packed.view(np.dtype((np.void, packed.shape[1]))).ravel()
    unique, first = np.unique(keys, return_index=True)
    qpacked = np.packbits(query.reshape(len(query), -1), axis=1)
    qkeys = qpacked.view(unique.dtype).ravel()
    positions = np.searchsorted(unique, qkeys)
    positions = np.minimum(positions, len(unique) - 1)
    matched = unique[positions] == qkeys
    result = first[positions]
    # Order, not only membership, matters for tied IoUs and different roofs on equal plans.
    first = np.sort(first)
    print(f"[1-NN] {len(bank)} bank rows, {len(first)} unique footprints; "
          f"{int(matched.sum())}/{len(query)} exact matches", flush=True)
    if not matched.all():
        result[~matched] = first[retrieve_nn(query[~matched], bank[first], chunk=chunk,
                                            bank_chunk=bank_chunk, device=device)]
    return result.astype(np.int64)


def load_footprints(h5_path: Path, rows: np.ndarray) -> np.ndarray:
    """Read contiguous disk batches, then gather the explicitly selected rows."""
    with open_real_corpus(h5_path) as corpus:
        result = np.empty((len(rows), *corpus["footprint"].shape[1:]), bool)
        for start in range(0, len(rows), 8192):
            batch = rows[start:start + 8192]
            low, high = int(batch[0]), int(batch[-1]) + 1
            result[start:start + len(batch)] = corpus["footprint"][low:high][batch - low] > 0
    return result


def score_pair(corpus: h5py.File, row: int, donor: int) -> dict:
    """The existing transplant + score_arm contract, plus a raw-GT residual audit."""
    fields = []
    for identity in (row, donor):
        fp = corpus["footprint"][identity] > 0
        raw = corpus["sdf"][identity] <= 0
        hf = height_field(raw, fp)
        if hf is None or not fp.any():
            raise ValueError(f"#178: invalid height field at row {identity}; no silent row drop")
        y0, y1, target = hf
        fields.append((fp, raw, y0, y1 - y0 + 1, target))
    fp, raw, y0, extent, target = fields[0]
    donor_fp, _, _, donor_extent, donor_target = fields[1]
    prediction = transplant_height(donor_target, donor_fp, donor_extent, fp, extent)
    held = dict(row=np.array([row]), fp=fp[None], y0=np.array([y0]),
                extent=np.array([extent]), target=target[None])
    metrics = score_arm(prediction[None], held)[0]
    metrics.update(donor_id=int(donor),
                   nn_footprint_iou=float((fp & donor_fp).sum() / (fp | donor_fp).sum()),
                   gt_height_field_mismatch_voxels=int((occupancy(fp, y0, target) ^ raw).sum()),
                   gt_raw_voxels=int(raw.sum()))
    return metrics


def _score_batch(job: tuple) -> list:
    path, pairs = job
    with open_real_corpus(path) as corpus:
        return [score_pair(corpus, int(row), int(donor)) for row, donor in pairs]


def population_report(rows: list[dict]) -> dict:
    """Summaries keep all rows, all nine city floors, and all four family diagnostics."""
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)) or any(r["city"] not in BAR_CITIES or
                                     r["family"] not in FAMILIES for r in rows):
        raise ValueError("#178: duplicate rows or invalid evaluation city/family")
    if not rows or set(BAR_CITIES) - {r["city"] for r in rows}:
        raise ValueError("#178: all nine cities must be measured")
    def summary(selected):
        return summarise(selected) if selected else {"n": 0, "status": "empty"}
    return dict(overall=summary(rows),
                per_city={c: summary([r for r in rows if r["city"] == c]) for c in BAR_CITIES},
                per_family={f: summary([r for r in rows if r["family"] == f]) for f in FAMILIES},
                gable_hip=summary([r for r in rows if r["family"] in ("gable", "hip")]))


def quoted_bar(summary: dict) -> dict:
    """#170: quote this population's 1-NN measurements without rounding or tuning."""
    if summary["n"] == 0:
        raise ValueError("#178: cannot calibrate a hard floor from an empty population")
    keys = ("dl_ops", "dl_planar_fraction", "extra", "collapse_rate")
    if not all(np.isfinite(summary[k]) for k in keys):
        raise ValueError("#178: cannot calibrate a hard floor from non-finite measurements")
    return dict(max_ops=summary["dl_ops"], min_planar=summary["dl_planar_fraction"],
                max_extra=summary["extra"], kill_planar=summary["dl_planar_fraction"],
                max_collapse=summary["collapse_rate"], max_vs_input=0.98)


def preregister(report: dict) -> dict:
    floors = {"overall": report["overall"], **{f"city:{c}": report["per_city"][c]
              for c in BAR_CITIES}, "gable_hip": report["gable_hip"]}
    bars = {name: quoted_bar(summary) for name, summary in floors.items()}
    return dict(status="pending_owner_signoff", owner="danvisai", rule_issue=170,
                threshold_source="1-NN on each floor's own full held-out population",
                floors=bars,
                infeasible_floors=[name for name, bar in bars.items()
                                   if bar["max_extra"] <= 0 or bar["kill_planar"] >= 1],
                infeasibility_note="extra is nonnegative; strict extra < 0 cannot pass. "
                                   "Planar fraction cannot exceed 1. Never relax a floor automatically.")


def buildingworld_verdict(report: dict, registration: dict) -> dict:
    """Evaluate all floors; absent or non-finite measurements cannot earn a pass.

    This is a numerical comparison, not sign-off. Registration status is carried
    separately so an unapproved proposal cannot silently become a binding bar.

    A floor named in `registration["guard_only_floors"]` (owner sign-off override, set by
    `apply_signoff`, always a subset of the honestly-quoted `infeasible_floors`) is graded on
    its two GUARD-shaped bounds only (`collapse`, `moved`) -- its PASS-shaped clauses
    (`form_ops`, `form_planar`, `beats_extra`, `clear_kill`) are analytically unreachable
    (quoted from 1-NN's own `extra<=0` or `planar_fraction>=1` on that population) and are
    reported for visibility but never gate `pass_` or the aggregate `numeric_pass`. The quoted
    bar itself is untouched -- #170 forbids relaxing a floor -- only which of its clauses gate.
    """
    summaries = {"overall": report.get("overall", {}), "gable_hip": report.get("gable_hip", {}),
                 **{f"city:{c}": report.get("per_city", {}).get(c, {}) for c in BAR_CITIES}}
    expected = set(summaries)
    if set(registration["floors"]) != expected:
        raise ValueError("#178: registration must contain exactly all eleven required floors")
    guard_only = set(registration.get("guard_only_floors", []))
    verdicts = {}
    for name, bar in registration["floors"].items():
        s = summaries[name]
        measured = s.get("n", 0) > 0 and all(
            np.isfinite(s.get(k, float("nan"))) for k in
            ("dl_ops", "dl_planar_fraction", "extra", "collapse_rate", "vs_input"))
        if not measured:
            verdicts[name] = dict(pass_=False, status="missing_measurement")
            continue
        clauses = dict(form_ops=s["dl_ops"] <= bar["max_ops"],
                       form_planar=s["dl_planar_fraction"] >= bar["min_planar"],
                       beats_extra=s["extra"] < bar["max_extra"],
                       collapse=s["collapse_rate"] <= bar["max_collapse"],
                       moved=s["vs_input"] < bar["max_vs_input"],
                       clear_kill=s["dl_planar_fraction"] > bar["kill_planar"])
        if name in guard_only:
            gating = dict(collapse=clauses["collapse"], moved=clauses["moved"])
            verdicts[name] = dict(pass_=bool(all(gating.values())), pass_available=False,
                                  guard_only=True, **{k: bool(v) for k, v in clauses.items()})
        else:
            verdicts[name] = dict(pass_=bool(all(clauses.values())), pass_available=True,
                                  **{k: bool(v) for k, v in clauses.items()})
    return dict(numeric_pass=all(v["pass_"] for v in verdicts.values()), floors=verdicts,
                registration_status=registration["status"])


def apply_signoff(registration: dict, guard_only_floors: list | None = None,
                  note: str = "") -> dict:
    """Record the project owner's #170 sign-off on a `preregister()` proposal.

    `guard_only_floors` must be a subset of the already-computed `infeasible_floors` --
    populations whose quoted PASS/KILL clauses are analytically unreachable (1-NN itself scored
    `extra<=0` or `planar_fraction>=1` there, usually from a tiny or degenerate held-out sample).
    Sign-off never edits `floors`' quoted numbers; it only marks which floors' PASS-shaped
    clauses stop gating `buildingworld_verdict`'s aggregate, per the owner's explicit choice
    (2026-09-15: GUARD-only ceiling, no PASS defined, for #178's four infeasible city floors).
    """
    guard_only_floors = list(guard_only_floors or [])
    unknown = set(guard_only_floors) - set(registration["infeasible_floors"])
    if unknown:
        raise ValueError(f"#178: guard_only_floors must be a subset of infeasible_floors: {sorted(unknown)}")
    if registration["status"] != "pending_owner_signoff":
        raise ValueError(f"#178: registration already {registration['status']!r}; refusing a second sign-off")
    signed = dict(registration)
    signed["status"] = "signed_off"
    signed["guard_only_floors"] = guard_only_floors
    signed["signoff_note"] = note
    return signed


def frozen_control(path: Path = HISTORICAL) -> dict:
    """Copy already-published summaries; do not call a scorer on the pinned 714."""
    historical = json.loads(path.read_text())
    return dict(source=str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path),
                sha256=sha256_file(path), n_pinned=historical["meta"]["n_pinned"],
                n_carve=historical["meta"]["n_carve"], program_bar=dict(PROGRAM_BAR),
                nn_retrieval=historical["arms"]["nn_retrieval"],
                served_ce_median=historical["arms"]["heightmap_ce_median"])


def run(h5_path: Path = REAL_CORPUS_PATH, ledger_path: Path = LEDGER_PATH,
        labels_path: Path = LABELS_PATH, output: Path = OUT, workers: int = 8,
        chunk: int = 64, bank_chunk: int = 16384, device: str | None = None) -> dict:
    """Build the complete, separate comparison; inputs are opened read-only."""
    if output.exists():
        raise FileExistsError(f"#178: {output} exists; choose a new versioned --output")
    started = time.monotonic()
    hashes = {"ledger_sha256": sha256_file(ledger_path), "labels_sha256": sha256_file(labels_path)}
    control = frozen_control()
    population = load_population(h5_path, ledger_path, labels_path)
    train, query = population["train"], population["query"]
    print(f"[population] train={len(train)} held-out={len(query)}", flush=True)
    bank_fp = load_footprints(h5_path, train)
    query_fp = load_footprints(h5_path, query)
    donors = train[exact_retrieval(query_fp, bank_fp, chunk=chunk, bank_chunk=bank_chunk,
                                   device=device)]
    del bank_fp, query_fp
    pairs = list(zip(query.tolist(), donors.tolist()))
    jobs = [(h5_path, pairs[i:i + 32]) for i in range(0, len(pairs), 32)]
    records = []
    # Opening the corpus inside each batch avoids inheriting an HDF5 handle across fork.
    with mp.get_context("spawn").Pool(workers) as pool:
        for batch in pool.imap(_score_batch, jobs, chunksize=1):
            records.extend(batch)
            if len(records) % 512 == 0 or len(records) == len(query):
                print(f"[score] {len(records)}/{len(query)} "
                      f"{time.monotonic() - started:.0f}s", flush=True)
    for record, city, family in zip(records, population["city"], population["family"]):
        record.update(city=str(city), family=str(family))
    if hashes != {"ledger_sha256": sha256_file(ledger_path),
                  "labels_sha256": sha256_file(labels_path)} or control["sha256"] != sha256_file(HISTORICAL):
        raise ValueError("#178: an input changed during evaluation; refusing to publish")
    report = population_report(records)
    registration = preregister(report)
    artifact = dict(meta=dict(issue=178, corpus=str(h5_path), **hashes,
                             corpus_rows=population["corpus_rows"],
                             identity_sha256=population["identity_sha256"],
                             train_n=len(train), train_source_counts=population["train_source_counts"],
                             train_rows_sha256=hashlib.sha256(train.astype("<i8").tobytes()).hexdigest(),
                             bank_order="ascending corpus row; first row wins equal footprint IoU",
                             evaluation_cities=list(BAR_CITIES),
                             excluded_held_out_cities=population["excluded_cities"],
                             family_source="committed #176 fitter pseudo-labels, not human ground truth",
                             scoring="score_arm on height-field GT, greedy form fitter max_ops=16",
                             elapsed_seconds=time.monotonic() - started),
                    pinned_714_control=control, buildingworld=report,
                    registration=registration, per_building=records)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(artifact, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"[report] {output}; numeric bar pending owner sign-off", flush=True)
    return artifact


def signoff_file(path: Path, guard_only_floors: list, note: str) -> dict:
    """Apply `apply_signoff` to an already-written artifact's registration, in place.

    Only the `registration` object changes; `buildingworld`/`per_building`/`meta` measurements
    are untouched, so this does not repeat the (expensive) retrieval-and-scoring run.
    """
    artifact = json.loads(path.read_text())
    artifact["registration"] = apply_signoff(artifact["registration"], guard_only_floors, note)
    with path.open("w") as stream:
        json.dump(artifact, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, default=REAL_CORPUS_PATH)
    parser.add_argument("--ledger", type=Path, default=LEDGER_PATH)
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--query-chunk", type=int, default=64)
    parser.add_argument("--bank-chunk", type=int, default=16384)
    parser.add_argument("--device", default=None,
                        help="e.g. cuda: runs only the 1-NN retrieval matmul on this torch "
                             "device; scoring stays on CPU workers regardless")
    parser.add_argument("--signoff", action="store_true",
                        help="apply the owner's #170 sign-off to an existing --output artifact "
                             "instead of running a new measurement")
    parser.add_argument("--guard-only-floors", nargs="*", default=[],
                        help="--signoff only: subset of the artifact's infeasible_floors to ship "
                             "as a GUARD-only ceiling (no PASS clause gates them)")
    parser.add_argument("--signoff-note", default="")
    args = parser.parse_args()
    if args.signoff:
        signoff_file(args.output, args.guard_only_floors, args.signoff_note)
        return
    if min(args.workers, args.query_chunk, args.bank_chunk) < 1:
        parser.error("worker and chunk counts must be positive")
    run(args.h5, args.ledger, args.labels, args.output, args.workers,
        args.query_chunk, args.bank_chunk, args.device)


if __name__ == "__main__":
    main()
