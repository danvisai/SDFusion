"""#177 -- per-row (x_m, y_m) footprint-centroid for every BuildingWorld row in `real.h5`: the
spatial key `buildingworld_stratified_split.py` grids into per-city tiles for its spatially-blocked
held-out split.

WHY THIS EXISTS
---------------
Neither `real.h5` (#174) nor `surfaces_buildingworld.h5` (#175) stores a BuildingWorld row's
absolute position: `real.h5`'s `sdf`/`footprint` are `building_to_sdf`'s self-normalized,
per-building grid (#153 already named this exact gap for NL: "the packed corpus stores no
per-building coordinates... each building is stored as a self-normalized SDF grid"), and #175's
`verts` are `to_frame_n`'s per-building-centered Frame-N -- translation-invariant by construction,
so it carries no more world position than the SDF does. #177's own issue text requires SPATIAL
blocking "within each city's own (correctly-scaled...) coordinate system"; that position has to
come from somewhere, and the only place it still exists is the raw BuildingWorld mesh itself,
before either normalization throws it away.

REUSED, NOT RE-DERIVED
-----------------------
Recovering "which zip member is row R's mesh" is #175's own already-solved, already-verified
problem (Perth's `bag_id`-truncation collisions -- `real.h5` can legitimately hold more than one
row under one truncated `bag_id`, and the zip can hold more candidate members than there are rows
for it). `_wanted_ids` / `_group_members_by_bag_id` / `_load_corrected_mesh` / `_resolve_collision`
are imported directly from `ingest_surfaces_buildingworld.py` rather than reimplemented here, so
this script cannot silently disagree with #175 about which mesh belongs to which row. The one
thing this script adds on top of that resolution is what it actually needs -- the corrected mesh's
`bounds` mean, taken BEFORE `to_frame_n` would throw position away -- and unlike #175/#176 it never
voxelizes a building at all in the common (non-colliding) case, only transiently inside
`_resolve_collision` for the rare truncation-collision candidate, exactly as #175 already does.

OUTPUT
------
`data/real_massing_v1/buildingworld_centroids.h5`, row-id-keyed against `real.h5`: `row` (int32),
`x_m` / `y_m` (float32, corrected per-city metres -- the same corrected frame `building_to_sdf`
was itself handed), `ok` (uint8, 0 for the rare row a mesh reload could not resolve or load -- a
disclosed failure, not a silently-dropped row: `buildingworld_stratified_split.py` refuses to
build a split while any BuildingWorld row lacks an `ok=1` centroid, per #177's own "an unassigned
row must be an error" requirement). Chunked, resizable, resumable (`IncrementalRowWriter`,
mirroring #176's `recover_program_labels_buildingworld.IncrementalRowWriter`).

Usage:
    buildingworld_spatial_keys.py                       # every BuildingWorld row, all cores
    buildingworld_spatial_keys.py --cities Berlin,Tokyo  # a subset
    buildingworld_spatial_keys.py --limit 200 --cities Cambridge   # smoke
    buildingworld_spatial_keys.py --report               # summarize an existing output file
"""
from __future__ import annotations

import argparse
import sys
import time
import zipfile
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.foundations.ingest_buildingworld import (  # noqa: E402
    CITY_SLUG, INGESTABLE_CITIES,
)
from scripts.foundations.ingest_surfaces_buildingworld import (  # noqa: E402
    _group_members_by_bag_id, _load_corrected_mesh, _resolve_collision, _wanted_ids,
)
from scripts.foundations.profile_buildingworld_meshes import canonical_zip  # noqa: E402
from utils.frozen_corpus import open_real_corpus  # noqa: E402

H5 = REPO / "data/real_massing_v1/real.h5"
OUT = REPO / "data/real_massing_v1/buildingworld_centroids.h5"

SLUG_TO_CITY = {slug: city for city, slug in CITY_SLUG.items()}

SCHEMA_VERSION = 1
COLUMNS = ("row", "x_m", "y_m", "ok", "reason")
DTYPES = dict(row=np.int32, x_m=np.float32, y_m=np.float32, ok=np.uint8, reason="S64")
# same OOM-avoidance shape as ingest_buildingworld.POOL_BATCH_SIZE / recover_program_labels_
# buildingworld.POOL_BATCH_SIZE -- this script runs the identical class of job (order-10^6
# CPU-bound tasks, one h5py-writing consumer), so it inherits the fix rather than re-discovering
# it under a fresh OOM.
POOL_BATCH_SIZE = 2000


def centroid_xy(mesh) -> tuple:
    """A corrected mesh's footprint-centroid, in its own city's corrected metres: the mean of its
    axis-aligned bounding box in X/Y. `mesh.bounds` (not a vertex mean) so tessellation density --
    a detailed roof vs. plain walls -- cannot bias the point used to grid the building spatially."""
    b = np.asarray(mesh.bounds, dtype=np.float64)
    return float(b[:, 0].mean()), float(b[:, 1].mean())


# ---- per-worker-process zip handle cache (fork pool), mirroring ingest_buildingworld._zf_for ----
_ZF_CACHE: dict = {}


def _zf_for(zpath: str):
    zf = _ZF_CACHE.get(zpath)
    if zf is None:
        zf = zipfile.ZipFile(zpath)
        _ZF_CACHE[zpath] = zf
    return zf


def _simple_task(task: tuple) -> dict:
    """Picklable, module-level worker body for the common (non-colliding) case: one bag_id, one
    member, one row. (city, zpath, member_name, row) -> a centroid record."""
    city, zpath, member_name, row = task
    try:
        mesh = _load_corrected_mesh(city, member_name, _zf_for(zpath))
        x, y = centroid_xy(mesh)
        return dict(row=row, x_m=np.float32(x), y_m=np.float32(y), ok=np.uint8(1), reason=b"")
    except Exception as e:
        reason = f"{type(e).__name__}: {str(e)}"[:63]
        print(f"  [skip] row {row} ({city}#{member_name}): {reason}", flush=True)
        return dict(row=row, x_m=np.float32("nan"), y_m=np.float32("nan"), ok=np.uint8(0),
                   reason=reason.encode("ascii", errors="replace"))


# ---- incremental, resumable output writer -- mirrors recover_program_labels_buildingworld's ----

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
                raise SystemExit(f"#177: cannot resume {path}: schema {version}, need "
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
# driver
# --------------------------------------------------------------------------------------------

def run(cities: list, out_path: Path = OUT, h5_path: Path = H5, limit: int = 0,
       workers: int = 1, pool_batch_size: int = POOL_BATCH_SIZE, resume: bool = True) -> dict:
    """`workers=1`: sequential, single zip handle per city -- what every test exercises.
    `workers>1`: a fork pool computes the common single-member/single-row case in parallel; any
    `bag_id` group with more than one candidate member or more than one claiming row (#175's
    truncation-collision case) is resolved sequentially in the main process afterward, since
    `_resolve_collision` needs the real, open `real.h5` SDF dataset to disambiguate -- rare enough
    (Perth only, at production scale) that this costs nothing worth parallelizing.

    `real.h5` is deliberately NOT held open while the fork pool is created: a metadata read (this
    function's own `source_key`/`bag_id` columns) happens first and the handle is closed before
    any `Pool(...)` call, reopened only for the collision path afterward -- #176's own `run()`
    follows the identical shape (its pool path never opens the corpus in the parent at all) for
    the same reason: forking a process with an open HDF5 file handle inherits that fd into every
    child regardless of whether the child ever touches it, which is unnecessary risk to carry for
    workers (`_simple_task`) that never read `real.h5` at all.

    Resuming a collision group: `already_done` is row-keyed, but `_resolve_collision`'s greedy
    match must see the group's FULL candidate/row set every time it runs, never a partial one --
    passing only the still-pending rows would let a candidate mesh already claimed by a
    previously-committed row be re-offered to a different row this time (nothing stops the greedy
    match from picking a "better" edge among the rows it can see, even though that mesh is already
    spoken for), corrupting an already-correct prior result. Mirrors
    `ingest_surfaces_buildingworld.run()`'s own resume shape exactly: skip a whole `bag_id` group
    only once EVERY one of its rows is already committed; otherwise recompute the full group fresh
    and simply skip re-`add()`-ing whichever of its rows are already written.
    """
    per_city: dict = {}
    t0 = time.time()
    with open_real_corpus(h5_path) as f:
        source_key_col = f["source_key"][:]
        bag_id_col = f["bag_id"][:]

    with IncrementalRowWriter(out_path, resume=resume) as writer:
        for city in cities:
            want = _wanted_ids(city, source_key_col, bag_id_col)
            n_want = sum(len(v) for v in want.values())
            print(f"[{city}] rows in real.h5: {n_want}", flush=True)
            if not want:
                per_city[city] = 0
                continue

            zpath, _ = canonical_zip(city)
            zpath = str(zpath)
            n_city = 0
            with zipfile.ZipFile(zpath) as zf:
                names = sorted(m for m in zf.namelist() if m.lower().endswith(".obj"))
                groups = _group_members_by_bag_id(city, names, want)

                simple_tasks, collision_groups = [], []
                for bag_id, member_names in groups.items():
                    rows = want[bag_id]
                    if all(writer.already_done(r) for r in rows):
                        continue
                    if len(member_names) == 1 and len(rows) == 1:
                        simple_tasks.append((city, zpath, member_names[0], rows[0]))
                    else:
                        collision_groups.append((bag_id, member_names, rows))  # FULL rows
                    if limit and len(simple_tasks) + len(collision_groups) >= limit:
                        break

                def handle(rec: dict) -> None:
                    nonlocal n_city
                    writer.add(**rec)
                    n_city += 1
                    if n_city % 20000 == 0:
                        print(f"  [{city}] {n_city}/{n_want}  ({time.time() - t0:.0f}s)",
                             flush=True)

                if workers > 1 and simple_tasks:
                    import multiprocessing as mp

                    with mp.get_context("fork").Pool(workers) as pool:
                        for i in range(0, len(simple_tasks), pool_batch_size):
                            batch = simple_tasks[i:i + pool_batch_size]
                            for rec in pool.imap_unordered(_simple_task, batch, chunksize=8):
                                handle(rec)
                else:
                    for task in simple_tasks:
                        handle(_simple_task(task))

                if collision_groups:
                    with open_real_corpus(h5_path) as f2:
                        real_sdf = f2["sdf"]
                        for bag_id, member_names, rows in collision_groups:
                            resolved = _resolve_collision(city, bag_id, member_names, rows, zf,
                                                          real_sdf, real_sdf.shape[-1])
                            for row, mesh in resolved.items():
                                if writer.already_done(row):
                                    continue
                                x, y = centroid_xy(mesh)
                                handle(dict(row=row, x_m=np.float32(x), y_m=np.float32(y),
                                           ok=np.uint8(1), reason=b""))
                            for row in rows:
                                if row not in resolved and not writer.already_done(row):
                                    handle(dict(row=row, x_m=np.float32("nan"),
                                               y_m=np.float32("nan"), ok=np.uint8(0),
                                               reason=b"collision_unresolved"))
            per_city[city] = n_city
            print(f"[{city}] done: {n_city}/{n_want}", flush=True)

    with h5py.File(out_path, "r") as f:
        n_total = int(f.attrs.get("committed_rows", 0))
        n_ok = int(np.asarray(f["ok"][:n_total]).sum()) if n_total else 0
    print(f"[#177 centroids] {n_total} rows written ({n_ok} ok, {n_total - n_ok} failed)  "
         f"-> {out_path}  ({time.time() - t0:.0f}s)", flush=True)
    return dict(per_city=per_city, n_total=n_total, n_ok=n_ok, out_path=str(out_path))


def report(out_path: Path = OUT) -> None:
    with h5py.File(out_path, "r") as f:
        n = int(f.attrs.get("committed_rows", 0))
        ok = np.asarray(f["ok"][:n])
    print(f"[#177 centroids report] {out_path}: {n} rows, {int(ok.sum())} ok, "
         f"{int((1 - ok).sum())} failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cities", default="", help="comma-separated subset of INGESTABLE_CITIES; "
                                                  "default = all 18 (Toronto excluded per #165)")
    ap.add_argument("--h5", default=str(H5))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--limit", type=int, default=0, help="cap tasks PER CITY, 0 = no cap (smoke)")
    ap.add_argument("--workers", type=int, default=0, help="0 = all but two cores")
    ap.add_argument("--no_resume", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    if args.report:
        report(Path(args.out))
        raise SystemExit(0)

    import multiprocessing as mp

    cities = args.cities.split(",") if args.cities else INGESTABLE_CITIES
    unknown = set(cities) - set(INGESTABLE_CITIES)
    if unknown:
        raise SystemExit(f"#177: not ingestable (excluded by #165, or misspelled): {sorted(unknown)}")
    workers = args.workers or max(mp.cpu_count() - 2, 1)

    result = run(cities, Path(args.out), Path(args.h5), limit=args.limit, workers=workers,
                resume=not args.no_resume)
    print(result)
