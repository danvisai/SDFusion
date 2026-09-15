"""#175 -- extract isosurface meshes for the BuildingWorld rows #174 appended to `real.h5`, into
`surfaces_buildingworld.h5`, and register the new source in `dora_frozen_gate.SOURCES` /
`load_surfaces()` so downstream consumers (the beam-search fitter's pseudo-label regeneration,
#176; the vecset-latent pipeline) can find them -- mirroring `ingest_surfaces.py`'s existing
per-source pipeline (recover a mesh, normalise into Frame-N, fix winding, store it).

Unlike `ingest_surfaces.py`'s three sources, BuildingWorld's meshes are already local (#174's own
docstring: "no live network dependency"), so this walks the same canonical zips #174's own
`stage_city` reads, not a fetch. The mesh recovered here is the SAME `trimesh` #174's
`process_member` already built and then discarded before `building_to_sdf` -- re-walked with the
SAME per-city correction (`apply_geometric_correction`) so its Frame-N vertices line up with the
stored `sdf` exactly the way #62's ingests already do.

Which candidates to re-load is read from `real.h5` itself (rows whose `source_key` matches a
city's `bw:<CitySlug>`, joined back to a zip member by recomputing `bag_id_for` the same way #174
did), not by re-running #174's gates -- `--verify` re-voxelises a sample of recovered meshes and
compares against the stored SDF the same way `ingest_surfaces.py --verify` does, so a silent
mismatch would show up as a low IoU rather than a guess.

`bag_id` is not always a unique join key: `bag_id_for` truncates to 64 bytes, and one production
city (Perth) has member names that only differ beyond that cutoff
('3D_Buildings_Level_2_Dec_2024_WSL1/building_<n>_part<k>.obj') -- real.h5 can then legitimately
hold more than one row under the identical truncated bag_id, and the zip can hold more candidate
members than there are rows for it. `_resolve_collision` disambiguates any such group by
recomputing each candidate's SDF (the SAME `building_to_sdf` #174 used) and matching it against
the actual STORED sdf for each contending row -- the one join key truncation cannot destroy.

All 18 ingestable BuildingWorld cities (#165's Toronto exclusion aside) share ONE
`surfaces_buildingworld.h5`: per #167, the per-row `source_key` column (stored here alongside
`row` and `bag_id`) is already the single provenance authority, so a merged file loses no
identity a future, finer-grained consumer (#171/#176/#177) might need -- it filters by
`source_key`, not by which file a row lives in.

Usage:
    ingest_surfaces_buildingworld.py                        # all 18 cities -> one h5
    ingest_surfaces_buildingworld.py --cities Berlin,Tokyo   # a subset
    ingest_surfaces_buildingworld.py --limit 20 --cities Cambridge   # smoke
    ingest_surfaces_buildingworld.py --verify                # alignment check against real.h5
    ingest_surfaces_buildingworld.py --no_resume             # start --out fresh (default resumes)

POST-PRODUCTION FIXES (code review, this ticket)
-------------------------------------------------
Two correctness/robustness gaps surfaced reviewing the committed version, after the full
production run had already completed successfully -- disclosed here rather than silently patched:

  * `run()` used to buffer every city's verts/faces in plain Python lists and write once at the
    end: no resume, unbounded memory over the full ~1.5M-row population. `IncrementalSurfaceWriter`
    (below) replaces that with the periodic, `committed_rows`-gated flush
    `ingest_buildingworld.IncrementalCityWriter` (#174) already established for this exact class of
    job -- same zips, same scale. The already-committed `surfaces_buildingworld.h5` did not need
    rewriting for this (it finished; see `--verify`'s own numbers), but a future re-run (a new
    city, a bug fix) now gets the same robustness #174's own OOM postmortem earned.
  * `_resolve_collision` claimed to keep only "confident" matches but enforced no threshold --
    greedy nearest-pairing would accept whatever candidate was left, however poor the fit. Fixed
    with an occupancy-IoU floor (`COLLISION_MIN_IOU`, the same 0.95 bar `ingest_surfaces.
    revoxelize_iou_l1` uses as "ALIGNED" elsewhere in this pipeline); a pairing below it is now left
    unresolved rather than accepted. Checked directly against the shipped file rather than assumed
    safe: every one of Perth's already-committed collision rows independently re-verified at
    IoU=1.0000 under the new floor, so the production artifact needed no changes either.
"""
from __future__ import annotations

import argparse
import io
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.foundations.ingest_buildingworld import (  # noqa: E402
    INGESTABLE_CITIES, apply_geometric_correction, bag_id_for, source_key_for,
)
from scripts.foundations.ingest_surfaces import fix_winding, to_frame_n  # noqa: E402
from scripts.foundations.profile_buildingworld_meshes import canonical_zip  # noqa: E402
from scripts.ingest_3dbag import building_to_sdf  # noqa: E402
from utils.frozen_corpus import open_real_corpus  # noqa: E402

H5 = REPO / "data/real_massing_v1/real.h5"
OUT = REPO / "data/real_massing_v1"
OUT_NAME = "surfaces_buildingworld.h5"


def _wanted_ids(city: str, source_key_col: np.ndarray, bag_id_col: np.ndarray) -> dict:
    """bag_id (as stored, possibly truncated to 64 bytes) -> [row, ...], for this city's real.h5
    rows. A list, not a single row: truncation can make two DIFFERENT member names collide onto
    the identical bag_id (see module docstring), so real.h5 can legitimately hold more than one
    row under one bag_id key.

    `source_key_col`/`bag_id_col` are read once by the caller and passed in so a multi-city run
    does not re-read real.h5's metadata columns per city.
    """
    key = source_key_for(city).encode("ascii")
    out: dict = {}
    for r in np.nonzero(source_key_col == key)[0]:
        out.setdefault(bytes(bag_id_col[r]), []).append(int(r))
    return out


def _group_members_by_bag_id(city: str, names, want: dict) -> dict:
    """{bag_id: [member_name, ...]} restricted to bag_ids `want` cares about. More than one
    member name per bag_id means truncation collapsed genuinely different candidates onto the
    same identity."""
    groups: dict = {}
    for member_name in names:
        bag_id = bag_id_for(city, member_name)
        if bag_id in want:
            groups.setdefault(bag_id, []).append(member_name)
    return groups


def _load_corrected_mesh(city: str, member_name: str, zf: zipfile.ZipFile):
    """The trimesh #174's own `process_member` built for this candidate: loaded, then given the
    SAME per-city correction, before `building_to_sdf` ever saw it."""
    import trimesh

    mesh = trimesh.load(io.BytesIO(zf.read(member_name)), file_type="obj")
    apply_geometric_correction(mesh, city, member_name)
    return mesh


# The occupancy-IoU bar `_resolve_collision` requires before calling a pairing "confident" --
# the SAME 0.95 "ALIGNED" threshold `ingest_surfaces.revoxelize_iou_l1` uses everywhere else in
# this pipeline (code-review finding on #175: the greedy pairing below had no threshold at all
# before this, so "confident" was this function's own docstring claim, not an enforced property --
# any candidate could pair with any row just for being the least-bad option left).
COLLISION_MIN_IOU = 0.95


def _resolve_collision(city: str, bag_id: bytes, member_names: list, rows: list,
                       zf: zipfile.ZipFile, real_sdf, r: int,
                       min_iou: float = COLLISION_MIN_IOU) -> dict:
    """`member_names` (>1) share `bag_id`'s truncated identity, and/or `rows` (>1) were stamped
    with it. Disambiguate by recomputing each candidate's SDF at resolution `r` -- the SAME
    `building_to_sdf` #174 used to write `real_sdf` -- and greedily pairing the globally closest
    (candidate, row) by L1 error first, so a genuine match (near-zero error, IoU 1.0) is claimed
    before a false one can be. A pairing is only accepted once its occupancy IoU against the
    stored SDF reaches `min_iou`; the closest remaining candidate for a row that never clears it is
    left unpaired rather than accepted as a last resort. Returns {row: mesh} for whichever rows
    cleared the bar; logs and drops the rest rather than risk mis-pairing geometry to the wrong
    row."""
    candidates = []
    for name in member_names:
        try:
            mesh = _load_corrected_mesh(city, name, zf)
            sdf, _, _ = building_to_sdf(mesh, r)
            candidates.append((name, mesh, sdf))
        except Exception as e:
            print(f"  [collision:skip] {city}#{name}: {type(e).__name__}: {str(e)[:70]}",
                 flush=True)

    ref_sdf = {row: np.asarray(real_sdf[row]) for row in rows}  # read each stored row once, not
                                                                # once per candidate
    edges = []
    for ci, (_, _, sdf) in enumerate(candidates):
        for row in rows:
            ref = ref_sdf[row]
            err = float(np.abs(sdf - ref).mean())
            ga, ra = sdf <= 0, ref <= 0
            iou = float((ga & ra).sum() / max((ga | ra).sum(), 1))
            edges.append((err, iou, ci, row))
    edges.sort(key=lambda t: t[0])

    out, used_c, used_r = {}, set(), set()
    for err, iou, ci, row in edges:
        if ci in used_c or row in used_r or iou < min_iou:
            continue
        used_c.add(ci); used_r.add(row)
        out[row] = candidates[ci][1]
        print(f"  [collision:resolved] {bag_id!r} row {row} <- {candidates[ci][0]} "
             f"(IoU={iou:.4f} L1={err:.5f})", flush=True)
    for row in rows:
        if row not in out:
            print(f"  [collision:unresolved] {bag_id!r} row {row}: no candidate reached "
                 f"IoU>={min_iou} among {len(member_names)} candidate(s)", flush=True)
    return out


POOL_BATCH_SIZE = 2000  # unused today (see IncrementalSurfaceWriter's docstring); kept named and
                        # exported so a future worker-pool addition to run() reaches for the same
                        # constant `ingest_buildingworld.POOL_BATCH_SIZE` -- and this ticket's own
                        # fix -- established, rather than picking a fresh number.


class IncrementalSurfaceWriter:
    """Resizable, resumable h5 writer for `surfaces_buildingworld.h5`'s ragged per-row mesh data.

    Code-review finding on #175: `run()` used to accumulate EVERY city's verts/faces in plain
    Python lists and write once at the end -- no flush, no resume, unbounded memory over ~1.5M
    rows. `ingest_buildingworld.IncrementalCityWriter` (#174) established the fix for the identical
    class of job on these same zips (a `committed_rows`-gated periodic flush); this reuses that
    shape, adapted for RAGGED data. `bag_id`/`source_key`/`row`/`vert_offset`/`face_offset` grow one
    entry per row, but `verts`/`faces` grow by a variable count per row, so resuming needs to know
    both boundaries -- `vert_offset[committed_rows]`/`face_offset[committed_rows]` (the last
    committed cumulative offset) gives the second one without a separate persisted counter.

    Parallelism (a worker pool computing meshes, like `ingest_buildingworld.stage_city`'s
    `workers>1` path) is deliberately NOT added here in the same pass as this fix, matching #174's
    own history of landing the sequential/robust version first and adding a pool separately once
    that baseline worked: the concrete, reproduced risk this fixes is unbounded memory + no resume,
    not wall-clock speed -- #175's own production run already completed single-process. A future
    pool would reuse `POOL_BATCH_SIZE` above rather than inventing a second bound.
    """
    SCHEMA_VERSION = 1

    def __init__(self, path: Path, resume: bool = True, flush_every: int = 5000):
        import h5py

        self.path, self.flush_every = Path(path), flush_every
        self.buf: dict = dict(bag_id=[], source_key=[], row=[], verts=[], faces=[])
        self.done: set = set()
        self.closed = False
        mode = "a" if (resume and self.path.exists()) else "w"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = h5py.File(self.path, mode)
        if mode == "w":
            self.f.attrs["schema_version"] = self.SCHEMA_VERSION
            self.f.attrs["committed_rows"] = 0
            self.f.flush()
        else:
            version = int(self.f.attrs.get("schema_version", 0))
            if version != self.SCHEMA_VERSION:
                self.f.close()
                raise SystemExit(f"#175: cannot resume {path}: schema {version}, need "
                                f"{self.SCHEMA_VERSION}")
            cr = int(self.f.attrs.get("committed_rows", 0))
            cv = int(self.f["vert_offset"][cr]) if "vert_offset" in self.f else 0
            cf = int(self.f["face_offset"][cr]) if "face_offset" in self.f else 0
            for k, n in (("bag_id", cr), ("source_key", cr), ("row", cr),
                        ("vert_offset", cr + 1), ("face_offset", cr + 1),
                        ("verts", cv), ("faces", cf)):
                if k in self.f and self.f[k].shape[0] > n:
                    self.f[k].resize(n, axis=0)
            self.f.flush()
            if "bag_id" in self.f:
                self.done = {bytes(b) for b in self.f["bag_id"][:cr]}

    def already_done(self, bag_id: bytes) -> bool:
        return bag_id in self.done

    def add(self, bag_id: bytes, source_key: bytes, row: int, verts: np.ndarray,
           faces: np.ndarray) -> None:
        """Buffers one row. Deliberately does NOT auto-flush here -- `_resolve_collision` can make
        one `bag_id` emit several rows in a row (one call each), and `already_done` is keyed on
        `bag_id` alone. A flush landing BETWEEN two of that bag_id's own rows would mark the whole
        `bag_id` done after committing only the first, so a resume would skip re-deriving the rest
        -- silently dropping a row forever. `maybe_flush()` -- called by a caller once it has
        finished emitting everything for one `bag_id`, never mid-group -- is what makes flush
        boundaries and `bag_id` group boundaries coincide, so `already_done` stays accurate."""
        self.buf["bag_id"].append(bag_id)
        self.buf["source_key"].append(source_key)
        self.buf["row"].append(row)
        self.buf["verts"].append(verts)
        self.buf["faces"].append(faces)

    def maybe_flush(self) -> None:
        """Flush if the buffer has grown past `flush_every`. Call only at a `bag_id` group
        boundary (see `add`'s docstring) -- never between two `add()` calls for the same `bag_id`."""
        if len(self.buf["bag_id"]) >= self.flush_every:
            self.flush()

    def _extend(self, name: str, data: np.ndarray) -> None:
        if name not in self.f:
            maxshape = (None,) + data.shape[1:]
            self.f.create_dataset(name, data=data, maxshape=maxshape,
                                  compression="lzf" if name in ("verts", "faces") else None)
        else:
            d = self.f[name]
            old = d.shape[0]
            d.resize(old + len(data), axis=0)
            d[old:old + len(data)] = data

    def flush(self) -> None:
        n = len(self.buf["bag_id"])
        if not n:
            return
        cr = int(self.f.attrs["committed_rows"])
        cv = int(self.f["vert_offset"][cr]) if "vert_offset" in self.f else 0
        cf = int(self.f["face_offset"][cr]) if "face_offset" in self.f else 0
        vcounts = np.array([len(v) for v in self.buf["verts"]], np.int64)
        fcounts = np.array([len(fc) for fc in self.buf["faces"]], np.int64)

        if "vert_offset" not in self.f:
            self.f.create_dataset("vert_offset", data=np.array([0], np.int64), maxshape=(None,))
        if "face_offset" not in self.f:
            self.f.create_dataset("face_offset", data=np.array([0], np.int64), maxshape=(None,))
        self._extend("vert_offset", cv + np.cumsum(vcounts))
        self._extend("face_offset", cf + np.cumsum(fcounts))
        self._extend("verts", np.concatenate(self.buf["verts"]))
        self._extend("faces", np.concatenate(self.buf["faces"]))
        self._extend("bag_id", np.array(self.buf["bag_id"], dtype="S64"))
        self._extend("source_key", np.array(self.buf["source_key"], dtype="S64"))
        self._extend("row", np.array(self.buf["row"], np.int32))

        self.f.flush()
        self.f.attrs.modify("committed_rows", cr + n)
        self.f.flush()
        self.done.update(self.buf["bag_id"])
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


def run(cities: list, limit: int, out_path: Path = OUT / OUT_NAME, resume: bool = True) -> dict:
    per_city = {}
    t0 = time.time()
    with open_real_corpus(H5) as f, IncrementalSurfaceWriter(out_path, resume=resume) as writer:
        source_key_col = f["source_key"][:]
        bag_id_col = f["bag_id"][:]
        real_sdf = f["sdf"]

        for city in cities:
            want = _wanted_ids(city, source_key_col, bag_id_col)
            n_want = sum(len(v) for v in want.values())
            print(f"[{city}] rows in real.h5: {n_want}", flush=True)
            if not want:
                per_city[city] = 0
                continue

            zpath, _ = canonical_zip(city)
            n_city = 0
            sk = source_key_for(city).encode("ascii")
            with zipfile.ZipFile(zpath) as zf:
                names = sorted(m for m in zf.namelist() if m.lower().endswith(".obj"))
                groups = _group_members_by_bag_id(city, names, want)

                def _emit(bag_id, row, mesh):
                    nonlocal n_city
                    vn = to_frame_n(mesh)
                    fn = fix_winding(vn, mesh.faces)
                    writer.add(bag_id, sk, row, vn, fn)
                    n_city += 1

                for bag_id, member_names in groups.items():
                    rows = want[bag_id]
                    if writer.already_done(bag_id):
                        continue                            # a resumed run's earlier flush
                    if len(member_names) == 1 and len(rows) == 1:
                        try:
                            mesh = _load_corrected_mesh(city, member_names[0], zf)
                        except Exception as e:
                            print(f"  [skip] {bag_id!r}: {type(e).__name__}: {str(e)[:70]}",
                                 flush=True)
                            continue
                        _emit(bag_id, rows[0], mesh)
                    else:
                        for row, mesh in _resolve_collision(city, bag_id, member_names, rows,
                                                            zf, real_sdf,
                                                            real_sdf.shape[-1]).items():
                            _emit(bag_id, row, mesh)
                    writer.maybe_flush()                    # only ever at a bag_id boundary
                    if n_city % 20000 < len(rows):
                        print(f"  [{city}] recovered {n_city}/{n_want} ({time.time()-t0:.0f}s)",
                             flush=True)
                    if limit and n_city >= limit:
                        break
            per_city[city] = n_city
            print(f"[{city}] recovered {n_city}/{n_want} ({100*n_city/max(n_want,1):.1f}%)",
                 flush=True)

    # Read back rather than sum `per_city` (which only counts THIS session's newly-processed rows
    # on a resumed run -- already-`already_done` bag_ids are skipped and never reach `_emit`).
    import h5py

    with h5py.File(out_path, "r") as f:
        n_total = f["row"].shape[0] if "row" in f else 0
    if not n_total:
        raise SystemExit("#175: recovered nothing")
    print(f"recovered {n_total} total  -> {out_path}  ({time.time()-t0:.0f}s)", flush=True)
    return dict(per_city=per_city, n_total=n_total, out_path=str(out_path))


def verify(n: int, path: Path = OUT / OUT_NAME) -> None:
    """Re-voxelise a sample spread across the file (so multiple cities are covered, not just the
    alphabetically-first one) and compare against the stored SDF -- the alignment proof."""
    import h5py

    from scripts.foundations.ingest_surfaces import revoxelize_iou_l1
    from scripts.foundations.vecset_ceiling_probe import grid_points

    pts = grid_points()
    with h5py.File(path, "r") as s, open_real_corpus(H5) as f:
        n_total = len(s["row"])
        k = min(n, n_total)
        idx = np.unique(np.linspace(0, n_total - 1, k).astype(int))
        print(f"[verify buildingworld] n={len(idx)}/{n_total}  "
             "(IoU of occupancy vs the stored field)")
        ious, errs = [], []
        for i in idx:
            i = int(i)
            a, b = int(s["vert_offset"][i]), int(s["vert_offset"][i + 1])
            c, d = int(s["face_offset"][i]), int(s["face_offset"][i + 1])
            v = np.asarray(s["verts"][a:b], np.float64)
            fc = np.ascontiguousarray(np.asarray(s["faces"][c:d]), np.int32)
            ref = np.asarray(f["sdf"][int(s["row"][i])], np.float32).transpose(2, 1, 0)
            iou, err = revoxelize_iou_l1(v, fc, ref, pts)
            ious.append(iou); errs.append(err)
            print(f"   {s['bag_id'][i].decode(errors='replace')[:40]:40s} "
                 f"{s['source_key'][i].decode():16s} IoU={iou:.4f}  L1={err:.5f}")
        print(f"  MEAN IoU={np.mean(ious):.4f}  L1={np.mean(errs):.5f}   "
             f"{'ALIGNED' if np.mean(ious) > 0.95 else 'MISALIGNED - do not use'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cities", default="", help="comma-separated subset of INGESTABLE_CITIES; "
                                                  "default = all 18 (Toronto excluded per #165)")
    ap.add_argument("--limit", type=int, default=0, help="cap recovered meshes PER CITY, 0 = no "
                                                          "cap (smoke testing)")
    ap.add_argument("--out", default=str(OUT / OUT_NAME))
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--verify_n", type=int, default=24)
    ap.add_argument("--no_resume", action="store_true",
                    help="start --out fresh instead of resuming a partially-written file")
    args = ap.parse_args()

    cities = args.cities.split(",") if args.cities else INGESTABLE_CITIES
    unknown = set(cities) - set(INGESTABLE_CITIES)
    if unknown:
        raise SystemExit(f"#175: not ingestable (excluded by #165, or misspelled): {sorted(unknown)}")

    if args.verify:
        verify(args.verify_n, Path(args.out))
    else:
        run(cities, args.limit, Path(args.out), resume=not args.no_resume)
