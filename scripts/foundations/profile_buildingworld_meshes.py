"""#158 -- watertightness, boundary-defect character, and extent/solidity distributions for every
BuildingWorld mesh city, on a properly randomized sample (not the 15-25 mesh spot-checks #156/#166
already flagged as too small to trust).

Per city, per sampled mesh:
    load -> is_watertight, is_winding_consistent
    if not watertight: classify boundary defect as floor_open (all boundary edges sit within 5% of
        the mesh's own z-range above z_min -- a missing floor cap, plausibly benign for igl's
        fast-winding-number SDF) vs scattered (boundary edges spread through the surface -- plausibly
        NOT benign: a real sign-error risk) vs mixed.
    extents, height_m, footprint solidity -> via ingest_3dbag.building_to_sdf, UNMODIFIED (the same
        function BuildingWorld ingestion will actually call), at a smaller grid (default R=32) since
        this is a profiling pass, not the ingest itself.

Also records, per city, which zip was used as the canonical file list (mesh.zip preferred over
obj.zip when both exist) and the OTHER zip's member count, since Adelaide and Wellington have both
with DIFFERENT counts (#158's own subfolder-inconsistency finding) -- this script documents that
mismatch, it does not resolve it.

Run (smoke):  ./venv/bin/python3 scripts/foundations/profile_buildingworld_meshes.py --n_sample 5 --cities Adelaide,Toronto
Run (full):   ./venv/bin/python3 scripts/foundations/profile_buildingworld_meshes.py
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import random
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.ingest_3dbag import building_to_sdf  # noqa: E402

MESH_ROOT = REPO / "data" / "buildingworld_mesh"
FLOOR_OPEN_Z_FRAC = 0.05   # boundary edge counts as "near z_min" if within this fraction of mesh height
FLOOR_OPEN_THRESH = 0.90   # >= this frac of boundary edges near z_min -> "floor_open"
SCATTERED_THRESH = 0.50    # <= this frac -> "scattered"

ALL_CITIES = sorted(
    d.name for d in MESH_ROOT.iterdir()
    if d.is_dir() and d.name not in ("sample_renders",) and not d.name.startswith(".")
)


def find_zips(city: str) -> dict:
    """-> {'mesh.zip': path_or_None, 'obj.zip': path_or_None} for one city, wherever nested."""
    found = {}
    for z in glob.glob(str(MESH_ROOT / city / "**" / "*.zip"), recursive=True):
        found[os.path.basename(z)] = z
    return found


def canonical_zip(city: str) -> tuple[str, dict]:
    """Pick the zip this profiler (and, by convention, ingestion) treats as canonical for a city.

    mesh.zip wins when both exist -- render_meshes.py already made this same choice for Adelaide.
    Returns (path_to_canonical_zip, {other_zip_name: member_count}) so a count mismatch is recorded,
    not silently dropped.
    """
    zips = find_zips(city)
    canonical = zips.get("mesh.zip") or zips.get("obj.zip")
    others = {}
    for name, path in zips.items():
        if path == canonical:
            continue
        with zipfile.ZipFile(path) as zf:
            others[name] = sum(1 for n in zf.namelist() if n.lower().endswith(".obj"))
    return canonical, others


def boundary_defect(m) -> dict:
    """For a non-watertight mesh: is the defect a missing floor cap (benign-ish) or scattered
    non-manifold geometry (not benign)? Classified from where boundary edges sit in z."""
    import trimesh
    edges = m.edges_sorted
    boundary_mask = trimesh.grouping.group_rows(edges, require_count=1)
    n_boundary = int(len(boundary_mask))
    if n_boundary == 0:
        # is_watertight is False for a reason OTHER than open boundary (e.g. non-manifold but closed,
        # or inconsistent winding) -- record it as its own bucket rather than forcing floor_open/scattered.
        return dict(n_boundary_edges=0, frac_near_zmin=None, defect_class="non_boundary_defect")
    z = np.asarray(m.vertices)[:, 2]
    zlo, zhi = float(z.min()), float(z.max())
    thresh = zlo + FLOOR_OPEN_Z_FRAC * (zhi - zlo) if zhi > zlo else zlo + 1e-6
    be = edges[boundary_mask]
    mid_z = z[be].mean(axis=1)
    frac_near_zmin = float((mid_z <= thresh).mean())
    if frac_near_zmin >= FLOOR_OPEN_THRESH:
        cls = "floor_open"
    elif frac_near_zmin <= SCATTERED_THRESH:
        cls = "scattered"
    else:
        cls = "mixed"
    return dict(n_boundary_edges=n_boundary, frac_near_zmin=frac_near_zmin, defect_class=cls)


def profile_one(data: bytes, r: int) -> dict:
    import trimesh
    try:
        m = trimesh.load(io.BytesIO(data), file_type="obj", process=False)
    except Exception as e:
        return dict(load_ok=False, error=f"load: {e!r}")
    if not hasattr(m, "faces") or len(m.faces) == 0 or len(m.vertices) == 0:
        return dict(load_ok=False, error="no faces/vertices")

    rec = dict(load_ok=True, n_verts=int(len(m.vertices)), n_faces=int(len(m.faces)))
    try:
        rec["watertight"] = bool(m.is_watertight)
        rec["winding_consistent"] = bool(m.is_winding_consistent)
    except Exception as e:
        rec["watertight"] = False
        rec["winding_consistent"] = False
        rec["topology_error"] = repr(e)

    ext = m.extents
    rec["max_extent"] = float(np.max(ext)) if len(ext) else None
    rec["raw_z_extent"] = float(ext[2]) if len(ext) > 2 else None

    if not rec["watertight"]:
        try:
            rec["boundary"] = boundary_defect(m)
        except Exception as e:
            rec["boundary"] = dict(error=repr(e))

    try:
        sdf, fp, height_m = building_to_sdf(m, R=r)
        rec["sdf_ok"] = True
        rec["height_m"] = height_m
        rec["fp_solidity"] = float(fp.mean())
        rec["occupancy_frac"] = float((sdf <= 0).mean())
    except Exception as e:
        rec["sdf_ok"] = False
        rec["sdf_error"] = repr(e)
    return rec


def profile_city(city: str, n_sample: int, seed: int, r: int) -> dict:
    zpath, other_zip_counts = canonical_zip(city)
    with zipfile.ZipFile(zpath) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".obj")]
        pop = len(names)
        k = min(n_sample, pop)
        sample = random.Random(seed).sample(names, k)
        recs = []
        for name in sample:
            data = zf.read(name)
            rec = profile_one(data, r)
            rec["name"] = name
            recs.append(rec)
    return dict(
        city=city,
        canonical_zip=os.path.relpath(zpath, MESH_ROOT),
        population=pop,
        other_zip_member_counts=other_zip_counts,  # non-empty => subfolder inconsistency, unresolved
        n_sampled=k,
        records=recs,
    )


def summarize(city_result: dict) -> dict:
    recs = city_result["records"]
    ok = [r for r in recs if r.get("load_ok")]
    n = len(recs)
    if not ok:
        return dict(n_sampled=n, n_load_ok=0)
    wt = np.array([r["watertight"] for r in ok])
    non_wt = [r for r in ok if not r["watertight"]]
    defect_counts = {}
    for r in non_wt:
        c = r.get("boundary", {}).get("defect_class", "unknown")
        defect_counts[c] = defect_counts.get(c, 0) + 1
    sdf_ok = [r for r in ok if r.get("sdf_ok")]
    out = dict(
        n_sampled=n,
        n_load_ok=len(ok),
        n_load_fail=n - len(ok),
        watertight_frac=float(wt.mean()),
        winding_consistent_frac=float(np.mean([r["winding_consistent"] for r in ok])),
        n_non_watertight=len(non_wt),
        defect_breakdown=defect_counts,
        n_sdf_ok=len(sdf_ok),
    )
    if sdf_ok:
        max_ext = np.array([r["max_extent"] for r in sdf_ok])
        height = np.array([r["height_m"] for r in sdf_ok])
        solidity = np.array([r["fp_solidity"] for r in sdf_ok])
        for label, arr in (("max_extent", max_ext), ("raw_height", height), ("fp_solidity", solidity)):
            out[f"{label}_p10"] = float(np.percentile(arr, 10))
            out[f"{label}_median"] = float(np.median(arr))
            out[f"{label}_p90"] = float(np.percentile(arr, 90))
        out["frac_extent_gt_90"] = float((max_ext > 90).mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_sample", type=int, default=400)
    ap.add_argument("--seed", type=int, default=158)
    ap.add_argument("--r", type=int, default=32, help="SDF grid resolution for the solidity probe")
    ap.add_argument("--cities", default="", help="comma-separated subset; default = all 19")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cities = args.cities.split(",") if args.cities else ALL_CITIES
    t0 = time.time()
    per_city = {}
    for city in cities:
        c0 = time.time()
        res = profile_city(city, args.n_sample, args.seed, args.r)
        per_city[city] = res
        s = summarize(res)
        print(f"[{city}] n={s.get('n_sampled')} watertight={s.get('watertight_frac', 0):.3f} "
              f"non_wt_defects={s.get('defect_breakdown')} "
              f"fp_solidity_med={s.get('fp_solidity_median', float('nan')):.4f} "
              f"({time.time()-c0:.0f}s)", flush=True)

    summaries = {c: summarize(r) for c, r in per_city.items()}

    print(f"\n=== BUILDINGWORLD MESH PROFILE (n_cities={len(cities)}, total {time.time()-t0:.0f}s) ===")
    hdr = f"{'city':16s} {'pop':>8} {'n':>5} {'watertight':>10} {'floor_open':>10} {'scattered':>9} " \
          f"{'ext_med':>8} {'ext>90':>7} {'solidity_med':>12}"
    print(hdr)
    for city in cities:
        s = summaries[city]
        r = per_city[city]
        db = s.get("defect_breakdown", {})
        print(f"{city:16s} {r['population']:>8} {s.get('n_sampled', 0):>5} "
              f"{s.get('watertight_frac', 0):>10.3f} {db.get('floor_open', 0):>10} "
              f"{db.get('scattered', 0):>9} {s.get('max_extent_median', float('nan')):>8.1f} "
              f"{s.get('frac_extent_gt_90', 0):>7.3f} {s.get('fp_solidity_median', float('nan')):>12.4f}")
        if r["other_zip_member_counts"]:
            print(f"  ! subfolder inconsistency: canonical={r['canonical_zip']} "
                  f"({r['population']} .obj) vs other zip(s)={r['other_zip_member_counts']}")

    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO),
                          capture_output=True, text=True).stdout.strip()
    suffix = f"_{args.tag}" if args.tag else ""
    art = REPO / f"execution/artifacts/buildingworld_mesh_profile{suffix}.json"
    art.parent.mkdir(parents=True, exist_ok=True)
    art.write_text(json.dumps(dict(
        meta=dict(git_rev=rev, created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  n_sample=args.n_sample, seed=args.seed, r=args.r, cities=cities),
        summaries=summaries,
        per_city=per_city,
    ), indent=2))
    print(f"\nartifact: {art}", flush=True)


if __name__ == "__main__":
    main()
