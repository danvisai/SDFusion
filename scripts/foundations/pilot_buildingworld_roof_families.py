"""#159 -- does BuildingWorld hold recoverable gable/hip/complex roof mass, or does it collapse
into the fitter's "complex, many-Layer" bucket like the five prior gable arms (#129/#132/#138/#139/
the combined assignment-type experiment) already did on NL/DE/JP?

A go/no-go signal, not a full ingest -- this script never writes to `real.h5`, `corpus_ledger.h5`,
or any other corpus artifact.

WHICH CITIES
------------
#159's own text pre-excludes two classes from BuildingWorld's 19 mesh cities, "for this pilot
only", rather than waiting on #165/#166's full ingestion-time policy:

  * **anisotropic-projection-error** cities (#157): raw X/Y is Web Mercator while Z is real
    metres, so every roof pitch is measured against a horizontally-stretched footprint -- this
    would bias the family classification itself, not just `height_m`. Toronto, Mississauga.
  * **"0/25-class watertight"** cities -- #159 cites the ORIGINAL 25-sample spot check by name,
    not #158's later n=400 profile (which moves some of these off exactly zero: Adelaide 6.0%,
    Greater Geelong 0.5%, Philadelphia 1.5%), and says explicitly not to reclassify on that "until
    those tickets' cities are separately handled". Adelaide, Calgary, Greater Geelong, Perth,
    Philadelphia, San Francisco, Wellington, Yarra.

Isotropic-unit-error cities (Boston, Cambridge, New York -- US survey feet, #157) are DELIBERATELY
kept in, uncorrected: a uniform scale error changes `height_m` but not a roof's rise/run ratio, so
it cannot bias which family a program fits into. That is why #159 only excludes the anisotropic
pair, not every CRS-flagged city.

The remaining nine: Berlin, Boston, Cambridge, Cape Town, Edmonton, Melbourne, Montreal, New York,
Tokyo.

METHOD
------
Per sampled BuildingWorld mesh: `trimesh.load` -> `ingest_3dbag.building_to_sdf` (unmodified, the
same function real ingestion would call) at R=64, matching the corpus's own grid -> `height_field`
(recover_massing_programs.py) turns that occupancy into the `(y0, y1, target)` triple the fitter
takes -> `fit_program_beam`, at its own defaults (`max_ops=4, allowance=CARVE_NEEDED, beam=6,
branch=6`, the full `Layer`/`CutRoof`/`Ramp` vocabulary) -- the literal function #159 names.

The comparison population (the "existing NL/DE/JP corpus") is `real.h5`'s own rows, keyed by
`source_id` (0/1/2 -> NL/DE/JP, `stratified_split.py`'s own mapping), fit the same way but from the
corpus's already-decoded occupancy grids -- no mesh, no `building_to_sdf` step.

`roof_family()` below turns the fitted program into one of `flat` / `gable` / `hip` / `complex`,
reusing three pieces of precedent already in this codebase rather than inventing a new rule (see
its own docstring). `dl_ops` / `dl_planar_fraction` are read directly off the SAME
`fit_program_beam` program (`len(program)`, and the fraction of its ops that are `Ramp`/`CutRoof`)
-- the identical fields `roof_description_length` (train_height_map_generator.py) reports, without
paying for that function's own second, independent greedy re-fit.

Run (smoke):
  env -u LD_PRELOAD ./venv/bin/python3 scripts/foundations/pilot_buildingworld_roof_families.py \\
      --n_per_city 20 --n_per_region 20 --cities Berlin,Tokyo
Run (full pilot, ~2,000/city + ~2,000/region):
  env -u LD_PRELOAD ./venv/bin/python3 scripts/foundations/pilot_buildingworld_roof_families.py
"""
from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
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

from scripts.foundations.profile_buildingworld_meshes import canonical_zip  # noqa: E402
from scripts.foundations.recover_massing_programs import (  # noqa: E402
    RES, fit_program_beam, height_field,
)
from scripts.ingest_3dbag import building_to_sdf  # noqa: E402
from utils.frozen_corpus import REAL_CORPUS_PATH, open_real_corpus  # noqa: E402

ANISOTROPIC_CITIES = frozenset({"Toronto", "Mississauga"})
ZERO_OF_25_WATERTIGHT_CITIES = frozenset({
    "Adelaide", "Calgary", "Greater Geelong", "Perth", "Philadelphia", "San Francisco",
    "Wellington", "Yarra",
})
ALL_19_CITIES = frozenset({
    "Adelaide", "Berlin", "Boston", "Calgary", "Cambridge", "Cape Town", "Edmonton",
    "Greater Geelong", "Melbourne", "Mississauga", "Montreal", "New York", "Perth",
    "Philadelphia", "San Francisco", "Tokyo", "Toronto", "Wellington", "Yarra",
})
PILOT_CITIES = sorted(ALL_19_CITIES - ANISOTROPIC_CITIES - ZERO_OF_25_WATERTIGHT_CITIES)

# stratified_split.py's own source_id -> region-name mapping; the comparison population.
SOURCE_NAMES = {0: "NL", 1: "DE", 2: "JP"}


# ------------------------------------------------------------------------------------------------
# roof-family classification
# ------------------------------------------------------------------------------------------------

def roof_family(ops: list, kinds: set) -> str:
    """flat / gable / hip / complex, from a `fit_program_beam` program's own op list and its
    `CutRoof` kinds -- three pieces of precedent already in this codebase, not a new rule:

      * zero ops (already flat, no carve needed) or exactly one `Layer` -> `flat`, matching
        `_ROOF_FAMILY_OF["Layer"] == "flat"` and `roof_description_length`'s own worked example
        ("flat roof: 1 op: Layer"). More than one `Layer` is NOT `flat` -- that is the "many-Layer"
        contour-terrace fallback `roof_description_length` uses a dome as its example of, and
        exactly the bucket this ticket is asking whether BuildingWorld collapses into.
      * a `CutRoof` kind set equal to `{"hip"}` -> `hip`; ANY other kind present (including a hip
        mixed with a gable) -> `gable`. This is `recover_massing_programs.main()`'s own
        `hip`/`gable` split on `roof_kinds` in its #128 edit-stack bridge report, applied here to
        every fit rather than only the sampled bridge subset.
      * two or more `Ramp` ops with no `CutRoof` at all -> `gable`, per #129's own framing ("a
        gable is two opposing ramps ... not in reach until the assignment head commits to a
        second") -- regardless of how many `Layer` ops (if any) sit underneath. A `Layer` here
        encodes WALL massing (a storey height, or a setback -- #4: "a setback *is* a Layer whose
        polygon is the inward offset of the footprint"), an axis this classifier does not score;
        two opposing `Ramp`s is a gable roof whether it sits on a plain box or a stepped one.

    Everything else -- chiefly a lone `Ramp` (a shed: one plane, no second one to pair with) -- is
    `complex`. A shed has no named bucket among the four the ticket asks for; counted here as
    `complex` and disclosed separately by the caller rather than silently folded in.
    """
    if not ops:
        return "flat"
    if kinds:
        return "hip" if kinds == {"hip"} else "gable"
    if ops.count("Ramp") >= 2:
        return "gable"
    if ops == ["Layer"]:
        return "flat"
    return "complex"


def fit_and_classify(fp: np.ndarray, y0: int, y1: int, target: np.ndarray) -> dict:
    """Run `fit_program_beam` at its own defaults and read the family/dl_ops/dl_planar_fraction
    diagnostics directly off the program it returns.

    `ops` (#176: the semicolon-joined op-type sequence, e.g. `"Layer;Ramp;Ramp"`) is included so a
    caller wanting the raw vocabulary -- not just this function's own flat/gable/hip/complex/shed
    summary of it -- does not need to re-run `fit_program_beam` a second time to get it.
    """
    program, _fitted = fit_program_beam(fp, y0, y1, target)
    ops = [o["op"] for o in program]
    kinds = {o["kind"] for o in program if o["op"] == "CutRoof"}
    planar = sum(1 for o in ops if o in ("Ramp", "CutRoof"))
    return dict(
        ok=True,
        family=roof_family(ops, kinds),
        dl_ops=len(ops),
        dl_planar_fraction=(planar / len(ops)) if ops else 0.0,
        is_shed=(ops == ["Ramp"]),
        ops=";".join(ops),
    )


# ------------------------------------------------------------------------------------------------
# worker tasks -- one per sampled building, dispatched through a fork pool
# ------------------------------------------------------------------------------------------------

_ZF_CACHE: dict = {}


def _zf_for(city: str, zpath: str):
    zf = _ZF_CACHE.get(city)
    if zf is None:
        zf = zipfile.ZipFile(zpath)
        _ZF_CACHE[city] = zf
    return zf


def _city_building_task(task: tuple) -> dict:
    city, zpath, name, r = task
    import trimesh
    try:
        data = _zf_for(city, zpath).read(name)
        m = trimesh.load(io.BytesIO(data), file_type="obj", process=False)
    except Exception as e:
        return dict(city=city, name=name, ok=False, stage="load", error=repr(e))
    if not hasattr(m, "faces") or len(m.faces) == 0 or len(m.vertices) == 0:
        return dict(city=city, name=name, ok=False, stage="load", error="no faces/vertices")
    try:
        sdf, fp, height_m = building_to_sdf(m, R=r)
    except Exception as e:
        return dict(city=city, name=name, ok=False, stage="sdf", error=repr(e))
    fp_bool = fp > 0
    if not fp_bool.any():
        return dict(city=city, name=name, ok=False, stage="empty_footprint")
    hf = height_field(sdf <= 0, fp_bool)
    if hf is None:
        return dict(city=city, name=name, ok=False, stage="empty_height_field")
    y0, y1, target = hf
    try:
        rec = fit_and_classify(fp_bool, y0, y1, target)
    except Exception as e:
        return dict(city=city, name=name, ok=False, stage="fit", error=repr(e))
    rec.update(city=city, name=name, height_m=height_m)
    return rec


_REAL_H5 = None


def _real_h5():
    global _REAL_H5
    if _REAL_H5 is None:
        _REAL_H5 = open_real_corpus(REAL_CORPUS_PATH)
    return _REAL_H5


def _region_building_task(task: tuple) -> dict:
    region, row = task
    g = _real_h5()
    gt = np.asarray(g["sdf"][row], np.float32) <= 0
    fp = np.asarray(g["footprint"][row]) > 0
    hf = height_field(gt, fp)
    if hf is None:
        return dict(region=region, row=row, ok=False, stage="empty_height_field")
    y0, y1, target = hf
    try:
        rec = fit_and_classify(fp, y0, y1, target)
    except Exception as e:
        return dict(region=region, row=row, ok=False, stage="fit", error=repr(e))
    rec.update(region=region, row=row)
    return rec


# ------------------------------------------------------------------------------------------------
# sampling and aggregation
# ------------------------------------------------------------------------------------------------

def sample_city_tasks(city: str, n: int, seed: int, r: int) -> list:
    zpath, _others = canonical_zip(city)
    with zipfile.ZipFile(zpath) as zf:
        names = [n_ for n_ in zf.namelist() if n_.lower().endswith(".obj")]
    sample = random.Random(seed).sample(names, min(n, len(names)))
    return [(city, zpath, name, r) for name in sample]


def sample_region_tasks(n: int, seed: int) -> dict:
    """-> {region_name: [(region, row), ...]}, sampled from real.h5's own `source_id` column."""
    with open_real_corpus() as g:
        source_id = np.asarray(g["source_id"])
    out = {}
    for code, name in SOURCE_NAMES.items():
        rows = np.nonzero(source_id == code)[0]
        picked = random.Random(seed).sample(list(rows), min(n, len(rows)))
        out[name] = [(name, int(row)) for row in picked]
    return out


def summarize(records: list) -> dict:
    n = len(records)
    ok = [r for r in records if r.get("ok")]
    fail_stages: dict = {}
    for r in records:
        if not r.get("ok"):
            stage = r.get("stage", "unknown")
            fail_stages[stage] = fail_stages.get(stage, 0) + 1
    out = dict(n_sampled=n, n_ok=len(ok), n_failed=n - len(ok), fail_stages=fail_stages)
    if ok:
        fam_counts: dict = {}
        for r in ok:
            fam_counts[r["family"]] = fam_counts.get(r["family"], 0) + 1
        out["family_counts"] = fam_counts
        out["family_fractions"] = {k: v / len(ok) for k, v in fam_counts.items()}
        out["dl_ops_median"] = float(np.median([r["dl_ops"] for r in ok]))
        out["dl_planar_fraction_median"] = float(np.median([r["dl_planar_fraction"] for r in ok]))
        out["n_shed"] = sum(1 for r in ok if r.get("is_shed"))
    return out


# ------------------------------------------------------------------------------------------------
# driver
# ------------------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_per_city", type=int, default=2000)
    ap.add_argument("--n_per_region", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=159)
    ap.add_argument("--r", type=int, default=RES)
    ap.add_argument("--cities", default="", help="comma-separated subset of PILOT_CITIES; "
                                                  "default = all nine")
    ap.add_argument("--workers", type=int, default=0, help="0 = all but two cores")
    ap.add_argument("--out", default="execution/artifacts/buildingworld_roof_family_pilot.json")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cities = args.cities.split(",") if args.cities else PILOT_CITIES
    unknown = set(cities) - set(PILOT_CITIES)
    if unknown:
        raise SystemExit(f"not in PILOT_CITIES (excluded by #159's own rule, or misspelled): "
                          f"{sorted(unknown)}")

    print(f"[#159] {len(cities)} BuildingWorld cities x up to {args.n_per_city}, "
          f"NL/DE/JP x up to {args.n_per_region}, seed={args.seed}, R={args.r}", flush=True)

    t0 = time.time()
    tasks = []
    for city in cities:
        tasks += sample_city_tasks(city, args.n_per_city, args.seed, args.r)
    region_tasks = sample_region_tasks(args.n_per_region, args.seed)
    for rows in region_tasks.values():
        tasks += rows
    print(f"[#159] {len(tasks)} buildings sampled ({time.time()-t0:.0f}s)", flush=True)

    n_workers = args.workers or min(len(tasks), max(mp.cpu_count() - 2, 1))
    by_city: dict = {c: [] for c in cities}
    by_region: dict = {r: [] for r in region_tasks}
    t0 = time.time()
    with mp.get_context("fork").Pool(n_workers) as pool:
        # `imap_unordered` takes one function per call, so the two task shapes (4-tuple mesh
        # tasks, 2-tuple corpus-row tasks) are split and run through their own map rather than
        # dispatched dynamically.
        city_tasks = [t for t in tasks if len(t) == 4]
        region_only_tasks = [t for t in tasks if len(t) == 2]
        results = []
        for k, rec in enumerate(pool.imap_unordered(_city_building_task, city_tasks, chunksize=4)):
            results.append(rec)
            if (k + 1) % 200 == 0:
                print(f"  [cities] {k+1}/{len(city_tasks)}  {time.time()-t0:.0f}s", flush=True)
        t1 = time.time()
        for k, rec in enumerate(pool.imap_unordered(_region_building_task, region_only_tasks,
                                                     chunksize=4)):
            results.append(rec)
            if (k + 1) % 200 == 0:
                print(f"  [regions] {k+1}/{len(region_only_tasks)}  {time.time()-t1:.0f}s",
                      flush=True)

    for rec in results:
        if "city" in rec:
            by_city[rec["city"]].append(rec)
        else:
            by_region[rec["region"]].append(rec)

    summaries_city = {c: summarize(recs) for c, recs in by_city.items()}
    summaries_region = {r: summarize(recs) for r, recs in by_region.items()}

    print(f"\n=== #159 ROOF-FAMILY PILOT (total {time.time()-t0:.0f}s) ===")
    hdr = f"{'population':16s} {'n':>5} {'ok':>5} {'flat':>6} {'gable':>6} {'hip':>6} " \
          f"{'complex':>8} {'dl_ops':>7} {'planar':>7}"
    print(hdr)
    for name, s in (*summaries_city.items(), *summaries_region.items()):
        ff = s.get("family_fractions", {})
        print(f"{name:16s} {s['n_sampled']:>5} {s['n_ok']:>5} "
              f"{ff.get('flat', 0):>6.3f} {ff.get('gable', 0):>6.3f} {ff.get('hip', 0):>6.3f} "
              f"{ff.get('complex', 0):>8.3f} {s.get('dl_ops_median', float('nan')):>7.2f} "
              f"{s.get('dl_planar_fraction_median', float('nan')):>7.3f}")

    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO),
                          capture_output=True, text=True).stdout.strip()
    suffix = f"_{args.tag}" if args.tag else ""
    out = REPO / args.out
    if suffix:
        out = out.with_name(out.stem + suffix + out.suffix)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        meta=dict(git_rev=rev, created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  seed=args.seed, r=args.r, n_per_city=args.n_per_city,
                  n_per_region=args.n_per_region, cities=cities,
                  excluded_anisotropic=sorted(ANISOTROPIC_CITIES),
                  excluded_zero_of_25_watertight=sorted(ZERO_OF_25_WATERTIGHT_CITIES)),
        summaries_by_city=summaries_city,
        summaries_by_region=summaries_region,
        by_city=by_city,
        by_region=by_region,
    ), indent=1))
    print(f"\n[artifact] {out}", flush=True)


if __name__ == "__main__":
    main()
