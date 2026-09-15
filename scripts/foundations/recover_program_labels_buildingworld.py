"""#176 -- run the beam-search fitter against every BuildingWorld row #174 appended to `real.h5`
and #175 verified in `surfaces_buildingworld.h5`, producing per-row program pseudo-label
diagnostics: `family` (flat/gable/hip/complex), `dl_ops`, `dl_planar_fraction`, `is_shed`, and the
fitted op-type sequence. #177 (the roof-family-stratified split for these rows) is blocked on this
artifact -- it is the only place `family` is computed for BuildingWorld at corpus scale.

WHICH ROWS, AND WHY REAL.H5 RATHER THAN THE RECOVERED MESH
------------------------------------------------------------
#175's own docstring names this script as a "downstream consumer" of `surfaces_buildingworld.h5`,
but the fitter itself (`recover_massing_programs.height_field` -> `fit_program_beam`) has only ever
read `real.h5`'s own `sdf`/`footprint` occupancy grids -- for the NL/DE/JP rows in
`recover_massing_programs.main()`, for the 35,623-row training cache in
`train_height_map_generator.build_cache`, and for the NL/DE/JP comparison population in #159's own
pilot (`pilot_buildingworld_roof_families._region_building_task`). #174 wrote BuildingWorld's `sdf`/
`footprint` through the identical `building_to_sdf` call the isosurface recovery re-derives its mesh
from, so re-reading it here (rather than re-voxelising #175's mesh a second time) keeps this script
on the exact same input every other population's fit already uses, and costs nothing #175 already
verified: `surfaces_buildingworld.h5`'s `row` column IS the confirmed, deduplicated BuildingWorld
row set (Perth's bag_id collisions resolved), so it is used here purely to KNOW those rows -- not to
re-supply their geometry. `real.h5`'s own `source_id == -1` sentinel (#174) names the identical set
directly, so it is the primary selector; `cross_check_row_count` (below) compares it against
`surfaces_buildingworld.h5`'s own row count, but only ever WARNS on a mismatch -- #175's own `run()`
legitimately drops rows it cannot confidently recover a mesh for, so a smaller surfaces count is an
expected, by-design outcome there, not proof the two files disagree, and this fitter needs nothing
from that file's geometry regardless.

THE FIT RECIPE
---------------
`fit_row` calls #159's own `fit_and_classify` (`pilot_buildingworld_roof_families.py`) directly --
reused, not reimplemented. This is the same recipe (`fit_program_beam`'s own defaults: `max_ops=4`,
`allowance=CARVE_NEEDED`, `beam=6`, `branch=6`, the full `Layer`/`CutRoof`/`Ramp` vocabulary) #159
already used for its own BuildingWorld sample and its NL/DE/JP comparison population -- the same
recipe this ticket's own numbers were measured with ("~0.25s/building"). `fit_and_classify` was
extended (this ticket) to also return the op-type sequence its own return contract previously
omitted, rather than this script re-running the fit a second time just to recover it.

OUTPUT
------
`data/real_massing_v1/program_labels_buildingworld.h5`, one row per BuildingWorld `real.h5` row:
`row` (int32), `ok` (uint8), `family` (S8), `dl_ops` (int16), `dl_planar_fraction` (float32),
`is_shed` (uint8), `ops` (S64, semicolon-joined op-type sequence, e.g. `b"Layer;Ramp;Ramp"`),
`fail_stage` (S32, empty when `ok`). Not the CutRoof-withheld `(assign, types, planes)` slot format
`train_height_map_generator.build_program_cache` writes for training supervision -- that format is
lossless only with CutRoof excluded (no plane), so it cannot carry `family`, which this ticket
needs and that one does not compute. `build_program_cache` is also keyed by #161's ledger, which
does not yet include these rows (#177, blocked on this ticket, is what adds them) -- this artifact
is row-id-keyed against `real.h5` directly instead, so it does not depend on #177 having run.

Chunked, resizable, and resumable (`IncrementalRowWriter`, `committed_rows` boundary), and tasks are
submitted to the worker pool in bounded batches (`POOL_BATCH_SIZE`) rather than all at once -- #174's
own `ingest_buildingworld.py` hit a real OOM from `imap_unordered` buffering an entire city's results
in the main process; this script re-runs the identical class of job (1.5M CPU-bound tasks, one
h5py-writing consumer) and inherits that fix rather than re-discovering it.

Usage:
    recover_program_labels_buildingworld.py                 # every BuildingWorld row, all cores
    recover_program_labels_buildingworld.py --limit 500      # smoke
    recover_program_labels_buildingworld.py --report         # summarize an existing output file
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.foundations.pilot_buildingworld_roof_families import (  # noqa: E402
    fit_and_classify, summarize,
)
from scripts.foundations.recover_massing_programs import height_field  # noqa: E402
from utils.frozen_corpus import open_real_corpus  # noqa: E402

H5 = REPO / "data/real_massing_v1/real.h5"
SURFACES_BUILDINGWORLD_H5 = REPO / "data/real_massing_v1/surfaces_buildingworld.h5"
OUT = REPO / "data/real_massing_v1/program_labels_buildingworld.h5"

# #174: the source_id every BuildingWorld row was stamped with, since no real region id exists for
# it yet (#171). Registered here rather than re-derived, so this script and #174's own ingester
# cannot silently disagree on which rows are BuildingWorld's.
BUILDINGWORLD_SOURCE_ID = -1

SCHEMA_VERSION = 1
# same incident, same fix as `ingest_buildingworld.POOL_BATCH_SIZE` -- see module docstring.
POOL_BATCH_SIZE = 2000

COLUMNS = ("row", "ok", "family", "dl_ops", "dl_planar_fraction", "is_shed", "ops", "fail_stage")
DTYPES = dict(row=np.int32, ok=np.uint8, family="S8", dl_ops=np.int16,
              dl_planar_fraction=np.float32, is_shed=np.uint8, ops="S64", fail_stage="S32")


# --------------------------------------------------------------------------------------------
# row selection
# --------------------------------------------------------------------------------------------

def buildingworld_rows(source_id: np.ndarray) -> np.ndarray:
    """Every `real.h5` row index #174 stamped as BuildingWorld, ascending."""
    return np.nonzero(np.asarray(source_id) == BUILDINGWORLD_SOURCE_ID)[0].astype(np.int32)


def cross_check_row_count(n_source_id: int, surfaces_path: Path = SURFACES_BUILDINGWORLD_H5) -> None:
    """Informational only -- never blocks the fit. `surfaces_buildingworld.h5`'s own `run()`
    (#175) explicitly DROPS rows it cannot confidently recover a mesh for (a `[skip]`ped mesh
    load, or an unresolved Perth-style `bag_id` collision `_resolve_collision` gives up on), so a
    strictly SMALLER surfaces count than `real.h5`'s BuildingWorld row count is a legitimate,
    by-design outcome of #175 -- not evidence the two files disagree. This fitter never reads
    `surfaces_buildingworld.h5`'s geometry (see module docstring), so it has everything it needs
    from `real.h5` alone regardless of what this check finds; a LARGER surfaces count, which #175's
    own logic cannot produce, is the one direction actually worth a loud warning."""
    if not surfaces_path.exists():
        print(f"[warn] {surfaces_path} missing -- skipping the #175 row-count cross-check",
             flush=True)
        return
    with h5py.File(surfaces_path, "r") as f:
        n_surf = f["row"].shape[0]
    if n_surf != n_source_id:
        severity = "warn" if n_surf < n_source_id else "WARN (unexpected direction)"
        print(f"[{severity}] real.h5 has {n_source_id} source_id=={BUILDINGWORLD_SOURCE_ID} rows "
             f"but {surfaces_path.name} has {n_surf} -- proceeding anyway, since this fitter reads "
             f"only real.h5 (see cross_check_row_count's docstring for why this is not fatal)",
             flush=True)


# --------------------------------------------------------------------------------------------
# the fit -- #159's own `fit_and_classify`, reused rather than reimplemented (it now also returns
# `ops`, added for this ticket; see its docstring in pilot_buildingworld_roof_families.py)
# --------------------------------------------------------------------------------------------

_FAILED = dict(ok=False, family="", dl_ops=0, dl_planar_fraction=0.0, is_shed=False, ops="")


def fit_row(gt_occ: np.ndarray, fp_mask: np.ndarray) -> dict:
    """One `real.h5` row's occupancy -> a full pseudo-label record (never raises: a fit failure is
    recorded in `fail_stage`, matching #159's pilot, rather than aborting the whole run)."""
    hf = height_field(gt_occ, fp_mask)
    if hf is None:
        return dict(_FAILED, fail_stage="empty_height_field")
    y0, y1, target = hf
    try:
        return dict(fit_and_classify(fp_mask, y0, y1, target), fail_stage="")
    except Exception as e:
        return dict(_FAILED, fail_stage=f"fit:{type(e).__name__}")


# --------------------------------------------------------------------------------------------
# incremental, resumable output writer -- mirrors ingest_buildingworld.IncrementalCityWriter
# --------------------------------------------------------------------------------------------

class IncrementalRowWriter:
    """Resizable, resumable h5 writer keyed by `real.h5` row id, one flat table."""

    def __init__(self, path: Path, resume: bool = True, flush_every: int = 5000):
        self.path, self.flush_every = Path(path), flush_every
        self.buf: dict = {k: [] for k in COLUMNS}
        self.done: set = set()
        self.closed = False
        mode = "a" if (resume and self.path.exists()) else "w"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = h5py.File(self.path, mode)
        if mode == "w":
            self.f.attrs["schema_version"] = SCHEMA_VERSION
            self.f.attrs["committed_rows"] = 0
            self.f.flush()
        else:
            version = int(self.f.attrs.get("schema_version", 0))
            if version != SCHEMA_VERSION:
                self.f.close()
                raise SystemExit(f"#176: cannot resume {path}: schema {version}, need "
                                f"{SCHEMA_VERSION}")
            committed = int(self.f.attrs.get("committed_rows", 0))
            for k in COLUMNS:
                if k in self.f and self.f[k].shape[0] > committed:
                    self.f[k].resize(committed, axis=0)
            self.f.flush()
            if "row" in self.f:
                self.done = {int(r) for r in self.f["row"][:committed]}

    def already_done(self, row: int) -> bool:
        return int(row) in self.done

    def add(self, **rec) -> None:
        for k in COLUMNS:
            self.buf[k].append(rec[k])
        if len(self.buf["row"]) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        n = len(self.buf["row"])
        if not n:
            return
        committed = int(self.f.attrs["committed_rows"])
        for k in COLUMNS:
            arr = np.asarray(self.buf[k], DTYPES[k])
            if k not in self.f:
                self.f.create_dataset(k, data=arr, maxshape=(None,))
            else:
                d = self.f[k]
                d.resize(committed + n, axis=0)
                d[committed:] = arr
        self.f.flush()
        self.f.attrs.modify("committed_rows", committed + n)
        self.f.flush()
        self.done.update(int(r) for r in self.buf["row"])
        for v in self.buf.values():
            v.clear()

    def close(self) -> None:
        if self.closed:
            return
        self.flush()
        self.f.close()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------------------------------------------------------------------------
# per-worker-process corpus handle (fork pool) + task body
# --------------------------------------------------------------------------------------------

_H5_CACHE: dict = {}


def _corpus_for(h5_path: str):
    g = _H5_CACHE.get(h5_path)
    if g is None:
        g = open_real_corpus(h5_path)
        _H5_CACHE[h5_path] = g
    return g


def _fit_corpus_row(g, row: int) -> dict:
    """One open corpus handle + row id -> `fit_row`'s record, with `row` set. The one place both
    the pool worker (`_row_task`) and `run()`'s sequential path read a row, so they cannot drift."""
    gt = np.asarray(g["sdf"][row], np.float32) <= 0
    fp = np.asarray(g["footprint"][row]) > 0
    rec = fit_row(gt, fp)
    rec["row"] = row
    return rec


def _row_task(task: tuple) -> dict:
    """Picklable, module-level worker body: (h5_path, row) -> `_fit_corpus_row`'s record."""
    h5_path, row = task
    return _fit_corpus_row(_corpus_for(h5_path), row)


# --------------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------------

def run(h5_path: Path, rows, out_path: Path, workers: int = 1,
       pool_batch_size: int = POOL_BATCH_SIZE, resume: bool = True) -> dict:
    """`workers=1`: sequential, single corpus handle -- what every test exercises. `workers>1`: a
    fork pool computes `_row_task` in parallel; the writer itself stays single-process, fed
    sequentially from the pool's results, since h5py writes are not safe to parallelize. Tasks are
    submitted in `pool_batch_size` chunks, each fully drained before the next -- see the module
    docstring's OOM incident."""
    rows = [int(r) for r in rows]
    t0 = time.time()
    counters: dict = {}
    n_seen = 0

    with IncrementalRowWriter(out_path, resume=resume) as writer:
        pending = [r for r in rows if not writer.already_done(r)]
        print(f"[#176] {len(pending)}/{len(rows)} rows pending "
             f"({len(rows) - len(pending)} already committed)", flush=True)

        def handle(rec: dict) -> None:
            nonlocal n_seen
            n_seen += 1
            writer.add(**rec)
            key = rec["family"] if rec["ok"] else f"fail:{rec['fail_stage']}"
            counters[key] = counters.get(key, 0) + 1
            if n_seen % 20000 == 0:
                print(f"  [#176] {n_seen}/{len(pending)}  {time.time() - t0:.0f}s", flush=True)

        if workers > 1 and pending:
            import multiprocessing as mp

            with mp.get_context("fork").Pool(workers) as pool:
                for i in range(0, len(pending), pool_batch_size):
                    batch = pending[i:i + pool_batch_size]
                    tasks = [(str(h5_path), r) for r in batch]
                    for rec in pool.imap_unordered(_row_task, tasks, chunksize=8):
                        handle(rec)
        else:
            g = open_real_corpus(h5_path)
            for r in pending:
                handle(_fit_corpus_row(g, r))

    print(f"[#176] done  n={len(rows)}  {counters}  ({time.time() - t0:.0f}s)", flush=True)
    return dict(n_total=len(rows), counters=counters, out_path=str(out_path))


def report(out_path: Path = OUT) -> None:
    """Reads every column in one bulk h5py call each (not one `f[col][i]` read per row, which does
    not scale to the ~1.5M rows this script targets), then builds `summarize`'s list-of-dicts input
    from the in-memory numpy arrays."""
    with h5py.File(out_path, "r") as f:
        ok, dl_ops = f["ok"][:], f["dl_ops"][:]
        dl_planar_fraction, is_shed = f["dl_planar_fraction"][:], f["is_shed"][:]
        family, fail_stage = f["family"][:], f["fail_stage"][:]
    records = [dict(ok=bool(ok[i]), family=family[i].decode(), dl_ops=int(dl_ops[i]),
                    dl_planar_fraction=float(dl_planar_fraction[i]), is_shed=bool(is_shed[i]),
                    stage=fail_stage[i].decode())
              for i in range(len(ok))]
    s = summarize(records)
    print(f"[#176 report] {out_path}")
    print(f"  n={s['n_sampled']}  ok={s['n_ok']}  failed={s['n_failed']}  "
         f"fail_stages={s['fail_stages']}")
    if "family_fractions" in s:
        ff = s["family_fractions"]
        print(f"  family: flat={ff.get('flat', 0):.3f} gable={ff.get('gable', 0):.3f} "
             f"hip={ff.get('hip', 0):.3f} complex={ff.get('complex', 0):.3f}")
        print(f"  dl_ops median={s['dl_ops_median']:.2f}  "
             f"dl_planar_fraction median={s['dl_planar_fraction_median']:.3f}  "
             f"n_shed={s['n_shed']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", default=str(H5))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--limit", type=int, default=0, help="cap the number of rows fit, 0 = no cap "
                                                          "(smoke testing)")
    ap.add_argument("--workers", type=int, default=0, help="0 = all but two cores")
    ap.add_argument("--no_resume", action="store_true")
    ap.add_argument("--report", action="store_true", help="summarize an existing --out, fit nothing")
    args = ap.parse_args()

    if args.report:
        report(Path(args.out))
        raise SystemExit(0)

    import multiprocessing as mp

    with open_real_corpus(args.h5) as g:
        source_id = np.asarray(g["source_id"])
    rows = buildingworld_rows(source_id)
    cross_check_row_count(len(rows))
    if args.limit:
        rows = rows[:args.limit]
    print(f"[#176] {len(rows)} BuildingWorld rows to fit", flush=True)

    workers = args.workers or max(mp.cpu_count() - 2, 1)
    run(Path(args.h5), rows, Path(args.out), workers=workers, resume=not args.no_resume)
    report(Path(args.out))
