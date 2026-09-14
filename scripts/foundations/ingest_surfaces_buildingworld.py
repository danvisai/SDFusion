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


def _resolve_collision(city: str, bag_id: bytes, member_names: list, rows: list,
                       zf: zipfile.ZipFile, real_sdf, r: int) -> dict:
    """`member_names` (>1) share `bag_id`'s truncated identity, and/or `rows` (>1) were stamped
    with it. Disambiguate by recomputing each candidate's SDF at resolution `r` -- the SAME
    `building_to_sdf` #174 used to write `real_sdf` -- and greedily pairing the globally closest
    (candidate, row) by L1 error first, so a genuine match (near-zero error) is claimed before a
    false one can be. Returns {row: mesh} for whichever rows found a confident match; logs and
    drops the rest rather than risk mis-pairing geometry to the wrong row."""
    candidates = []
    for name in member_names:
        try:
            mesh = _load_corrected_mesh(city, name, zf)
            sdf, _, _ = building_to_sdf(mesh, r)
            candidates.append((name, mesh, sdf))
        except Exception as e:
            print(f"  [collision:skip] {city}#{name}: {type(e).__name__}: {str(e)[:70]}",
                 flush=True)

    edges = []
    for ci, (_, _, sdf) in enumerate(candidates):
        for row in rows:
            err = float(np.abs(sdf - np.asarray(real_sdf[row])).mean())
            edges.append((err, ci, row))
    edges.sort(key=lambda t: t[0])

    out, used_c, used_r = {}, set(), set()
    for err, ci, row in edges:
        if ci in used_c or row in used_r:
            continue
        used_c.add(ci); used_r.add(row)
        out[row] = candidates[ci][1]
        print(f"  [collision:resolved] {bag_id!r} row {row} <- {candidates[ci][0]} "
             f"(L1={err:.5f})", flush=True)
    for row in rows:
        if row not in out:
            print(f"  [collision:unresolved] {bag_id!r} row {row}: no confident match among "
                 f"{len(member_names)} candidate(s)", flush=True)
    return out


def run(cities: list, limit: int, out_path: Path = OUT / OUT_NAME) -> dict:
    import h5py

    keys, source_keys, rows_out, verts, faces, vo, fo = [], [], [], [], [], [0], [0]
    per_city = {}
    t0 = time.time()
    with open_real_corpus(H5) as f:
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
                    keys.append(bag_id); source_keys.append(sk); rows_out.append(row)
                    verts.append(vn); faces.append(fn)
                    vo.append(vo[-1] + len(vn)); fo.append(fo[-1] + len(fn))
                    n_city += 1

                for bag_id, member_names in groups.items():
                    rows = want[bag_id]
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
                    if n_city % 20000 < len(rows):
                        print(f"  [{city}] recovered {n_city}/{n_want} ({time.time()-t0:.0f}s)",
                             flush=True)
                    if limit and n_city >= limit:
                        break
            per_city[city] = n_city
            print(f"[{city}] recovered {n_city}/{n_want} ({100*n_city/max(n_want,1):.1f}%)",
                 flush=True)

    if not keys:
        raise SystemExit("#175: recovered nothing")
    with h5py.File(out_path, "w") as f:
        f.create_dataset("verts", data=np.concatenate(verts), compression="lzf")
        f.create_dataset("faces", data=np.concatenate(faces), compression="lzf")
        f.create_dataset("vert_offset", data=np.asarray(vo, np.int64))
        f.create_dataset("face_offset", data=np.asarray(fo, np.int64))
        f.create_dataset("row", data=np.asarray(rows_out, np.int32))
        f.create_dataset("bag_id", data=np.array(keys, dtype="S64"))
        f.create_dataset("source_key", data=np.array(source_keys, dtype="S64"))
    print(f"recovered {len(keys)} total  verts={vo[-1]:,} faces={fo[-1]:,}  -> {out_path}  "
         f"({time.time()-t0:.0f}s)", flush=True)
    return dict(per_city=per_city, n_total=len(keys), out_path=str(out_path))


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
    args = ap.parse_args()

    cities = args.cities.split(",") if args.cities else INGESTABLE_CITIES
    unknown = set(cities) - set(INGESTABLE_CITIES)
    if unknown:
        raise SystemExit(f"#175: not ingestable (excluded by #165, or misspelled): {sorted(unknown)}")

    verify(args.verify_n, Path(args.out)) if args.verify else run(cities, args.limit, Path(args.out))
