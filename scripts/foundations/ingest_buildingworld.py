"""#174 -- ingest BuildingWorld's untextured meshes into `real.h5`, mirroring
`ingest_citygml_lod2.py`'s structure and calling the existing, UNMODIFIED
`building_to_sdf` (`scripts/ingest_3dbag.py`) the same way 3D BAG/NRW/PLATEAU already do.

Decisions this script APPLIES (all already made by their own tickets; this ticket does not
re-open any of them):

  * [#165](../../docs/wayfinding/buildingworld-corpus/165-crs-units-policy-decision.md) -- per-city
    CRS/units correction, applied in `apply_geometric_correction` BEFORE the 90m extent filter and
    BEFORE `building_to_sdf` sees the mesh: isotropic US-survey-feet scaling (New York, Boston,
    Cambridge, and Philadelphia's `2010_ph_downtown` subfolder only), an anisotropic Web-Mercator ->
    UTM17N reprojection for Mississauga, Toronto dropped entirely, and Yarra's `richmond` subfolder
    Y-sign-flip-plus-rewind (the flip is a mirror transform; naively negating Y without reversing
    each face's winding would ingest an inside-out mesh -- #165's own code-review correction).
  * [#166](../../docs/wayfinding/buildingworld-corpus/166-watertightness-standard-decision.md) --
    relaxed watertightness gate (keep watertight/floor_open/non_boundary_defect, reject
    scattered/mixed), reusing `profile_buildingworld_meshes.boundary_defect`'s own classifier so the
    defect taxonomy never drifts between the profiling pass and the real ingest. `defect_class` is
    stamped per row (a schema addition beyond the ticket's literal column list, required by #166's
    own text: "so this bucket stays traceable").
  * [#167](../../docs/wayfinding/buildingworld-corpus/167-provenance-authority-field.md) -- `source_key`
    ('bw:<CitySlug>') is the authoritative provenance field, via `source_provenance.make_source_key`.
    `region_id_of` is deliberately NEVER called here: #171 (region-conditioning granularity) has not
    landed, so this script does not guess a real region id. `source_id` (the legacy int32 region
    column every real.h5 row must have) is written as `SOURCE_ID_UNASSIGNED` (-1) for every
    BuildingWorld row -- an explicit, disclosed, versioned-nowhere sentinel, not a guess. See
    `docs/wayfinding/buildingworld-corpus/174-ingest-buildingworld.md` for why -1 rather than a new
    small integer, and the matching `data_sources.py`/`stratified_split.py` entries this requires to
    keep `test_data_sources.py` passing.
  * [#168](../../docs/wayfinding/buildingworld-corpus/168-sampling-cap-corpus-balance-decision.md) --
    no per-city cap, no total cap, no reweighting. Every row that survives the gates above is kept.
  * [#160](../../docs/wayfinding/buildingworld-corpus/160-geometric-duplicate-gate.md) -- any
    Tokyo/Berlin candidate flagged as a match in `execution/artifacts/buildingworld_duplicate_gate.json`
    is excluded (today: zero matches, but the exclusion mechanism is live for a re-run against a
    different snapshot).
  * [#162](../../docs/wayfinding/buildingworld-corpus/162-frozen-corpus-identity.md) -- the frozen
    prefix (rows [0, `FROZEN_SPLIT_N_TOTAL`)) is verified via `utils.frozen_corpus` before this script
    reads `real.h5`, and again on the rebuilt file before it replaces `real.h5`.

Structure, mirroring `ingest_citygml_lod2.py`: this file has no live network dependency (BuildingWorld
meshes are already local, unlike NRW/PLATEAU) but real.h5's datasets are all fixed-size (no
`maxshape`), so "append" means a safe rebuild, not an in-place resize. Two stages, both driven by
`main()`:

  1. **Stage** (`stage_city`) -- per city, incrementally ingest every candidate mesh in its canonical
     zip into `data/real_massing_v1/buildingworld_staging/<CitySlug>.h5` (an `IncrementalCityWriter`,
     resumable at the per-mesh granularity the same way `precompute_vecset_latents.py`'s cache is --
     this is the ONLY way a multi-hundred-thousand-mesh city like Berlin can be ingested without
     holding every SDF volume in memory at once or losing all progress to a single crash).
  2. **Combine** (`combine_into_real`) -- rebuild `real.h5` once from the untouched original plus every
     requested staged city file, verify the frozen-prefix pin on the REBUILT file, then atomically
     replace. Never mutates `real.h5` in place.

Run (smoke -- never touches the real corpus, writes a throwaway `--out`):
  env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \\
    scripts/foundations/ingest_buildingworld.py --smoke --cities Berlin,Tokyo --limit 20

Run (production, one city at a time or all 18 -- writes staging/<slug>.h5, no real.h5 change yet):
  env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \\
    scripts/foundations/ingest_buildingworld.py --no_combine --cities Berlin

Run (combine everything staged so far into real.h5):
  env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \\
    scripts/foundations/ingest_buildingworld.py --no_stage --combine
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.foundations.profile_buildingworld_meshes import (  # noqa: E402
    boundary_defect, canonical_zip,
)
from scripts.foundations.source_provenance import make_source_key  # noqa: E402
from scripts.ingest_3dbag import building_to_sdf  # noqa: E402
from utils.frozen_corpus import assert_frozen_corpus, open_real_corpus  # noqa: E402

STAGING_DIR = REPO / "data/real_massing_v1/buildingworld_staging"
REAL_H5 = REPO / "data/real_massing_v1/real.h5"
DEDUP_ARTIFACT = REPO / "execution/artifacts/buildingworld_duplicate_gate.json"

# #165: Toronto is dropped outright (independently poor mesh quality, not worth reprojecting).
EXCLUDED_CITIES = frozenset({"Toronto"})
ALL_CITIES = frozenset({
    "Adelaide", "Berlin", "Boston", "Calgary", "Cambridge", "Cape Town", "Edmonton",
    "Greater Geelong", "Melbourne", "Mississauga", "Montreal", "New York", "Perth",
    "Philadelphia", "San Francisco", "Tokyo", "Toronto", "Wellington", "Yarra",
})
INGESTABLE_CITIES = sorted(ALL_CITIES - EXCLUDED_CITIES)

# #167's own motivating example ('BW_GreaterGeelong') is exactly this: strip spaces so the
# source_key 'place' component matches SOURCE_KEY_RE ([A-Za-z0-9_.-]+, no whitespace).
CITY_SLUG = {city: city.replace(" ", "") for city in ALL_CITIES}

# #165: exact US survey foot -> metre factor (isotropic; corrupts height_m/extent uniformly if
# skipped, but never distorts roof pitch, which is why #159's earlier pilot could leave these
# cities in uncorrected while still excluding the anisotropic pair).
FEET_TO_M = 1200.0 / 3937.0
FEET_CITIES = frozenset({"Boston", "Cambridge", "New York"})
# Philadelphia is a MIXED city: only this subfolder is US feet (EPSG:2272); '2015_scene' is
# already metric (EPSG:32129) and must be left alone.
PHILADELPHIA_FEET_SEGMENT = "2010_ph_downtown"

# #165: Mississauga's raw X/Y matches EPSG:3857 (Web Mercator) despite the city's own stated
# convention being UTM17N metres (NAD83) -- an anisotropic error, fixed with a real reprojection,
# not a scalar. Z is already real, undistorted metres and is left untouched.
MISSISSAUGA_SRC_CRS = "EPSG:3857"
MISSISSAUGA_DST_CRS = "EPSG:26917"  # NAD83 / UTM zone 17N

# #165 code-review correction: Yarra's richmond/ subfolder has its Y axis mirrored relative to
# every other Yarra suburb. Negating Y alone would ingest an inside-out mesh (a mirror transform
# inverts winding) -- each face's vertex order must also be reversed to restore outward normals.
YARRA_RICHMOND_SEGMENT = "richmond"

# #166: watertight or a defect class igl's fast-winding-number SDF is proven robust to.
KEEP_DEFECT_CLASSES = frozenset({"watertight", "floor_open", "non_boundary_defect"})

# #167: -1 is a versioned-nowhere sentinel, not a guess at #171's eventual region-id scheme.
# `source_key` (below) is the row's real, authoritative provenance; this legacy int32 column is
# left explicitly unresolved. See docs/wayfinding/buildingworld-corpus/174-ingest-buildingworld.md.
SOURCE_ID_UNASSIGNED = -1
STYLE_ID_REAL = 8  # unchanged convention: 8 = unknown/real, across every existing source

# ingest_3dbag.py's own defensive filters (#174: "Apply the SAME defensive filters").
DEFAULT_MIN_H = 2.5
DEFAULT_MAX_EXT = 90.0
DEFAULT_MIN_FP = 80

# #157: per-city z-origin sanity, cross-checked against the audit's own documented reference
# elevation (mode, (low_m, high_m)). 'asl' cities gate hard (tol_m below); 'relative_zero'
# (Melbourne: z_min is always exactly 0, a per-building height-above-base, never elevation) gates
# hard on being near zero; 'unreliable' (Adelaide, Perth: #157 found negative, internally
# inconsistent z across different building/component files) and 'undocumented' (San Francisco,
# Wellington: #157 stated no elevation figure for either) are logged, never gated -- gating on an
# undocumented or already-known-messy band would be false confidence, not a real check.
CITY_Z_REFERENCE = {
    "NewYork": ("asl", (-5.0, 50.0)),
    "Philadelphia": ("asl", (-5.0, 40.0)),
    "Boston": ("asl", (-5.0, 120.0)),
    "Cambridge": ("asl", (-5.0, 30.0)),
    "Calgary": ("asl", (950.0, 1200.0)),
    "Edmonton": ("asl", (550.0, 800.0)),
    "Montreal": ("asl", (-5.0, 260.0)),
    "Berlin": ("asl", (0.0, 80.0)),
    "Tokyo": ("asl", (-5.0, 60.0)),
    "CapeTown": ("asl", (-5.0, 350.0)),
    "Mississauga": ("asl", (60.0, 300.0)),
    "GreaterGeelong": ("asl", (-5.0, 40.0)),
    "Yarra": ("asl", (-5.0, 50.0)),
    "Melbourne": ("relative_zero", None),
    "Adelaide": ("unreliable", None),
    "Perth": ("unreliable", None),
    "SanFrancisco": ("undocumented", None),
    "Wellington": ("undocumented", None),
}
Z_ASL_TOL_M = 40.0
Z_RELATIVE_TOL_M = 2.0

DEDUP_PAIR_FOR_CITY = {"Tokyo": "tokyo_vs_plateau", "Berlin": "berlin_vs_nrw"}

STAGING_COLUMNS = ("sdf", "footprint", "height_m", "style_id", "source_id", "class_label",
                   "bag_id", "source_key", "defect_class")
STAGING_DTYPES = {"height_m": np.float32, "style_id": np.int32, "source_id": np.int32,
                  "class_label": "S16", "bag_id": "S64", "source_key": "S64",
                  "defect_class": "S32"}


# ---- pure per-city / per-row helpers (TDD seam: no I/O) -------------------------------------

def city_slug(city: str) -> str:
    return CITY_SLUG[city]


def source_key_for(city: str) -> str:
    return make_source_key("bw", city_slug(city))


def class_label_for(city: str) -> bytes:
    # #167's own motivating example: 'BW_GreaterGeelong' is 17 bytes, over the S16 storage width.
    # Explicit, disclosed truncation -- source_key (above) is the untruncated, authoritative field.
    return f"BW_{city_slug(city)}".encode("ascii")[:16]


def bag_id_for(city: str, member_name: str) -> bytes:
    return f"{city}#{member_name}".encode("utf-8")[:64]


def _path_segment_matches(member_name: str, segment: str) -> bool:
    return segment in member_name.split("/")


def apply_geometric_correction(mesh, city: str, member_name: str):
    """Mutate `mesh` in place per #165's per-city CRS/units decision; return it for chaining.

    Order matters for Philadelphia (per-subfolder) and Yarra (per-subfolder): the whole-city checks
    run first and are mutually exclusive with the per-subfolder ones (no city needs both).
    """
    if city in FEET_CITIES:
        mesh.vertices = mesh.vertices * FEET_TO_M
    elif city == "Philadelphia" and _path_segment_matches(member_name, PHILADELPHIA_FEET_SEGMENT):
        mesh.vertices = mesh.vertices * FEET_TO_M
    elif city == "Mississauga":
        import pyproj

        transformer = pyproj.Transformer.from_crs(MISSISSAUGA_SRC_CRS, MISSISSAUGA_DST_CRS,
                                                   always_xy=True)
        x, y = transformer.transform(mesh.vertices[:, 0], mesh.vertices[:, 1])
        v = mesh.vertices.copy()
        v[:, 0], v[:, 1] = x, y
        mesh.vertices = v

    if city == "Yarra" and _path_segment_matches(member_name, YARRA_RICHMOND_SEGMENT):
        v = mesh.vertices.copy()
        v[:, 1] = -v[:, 1]
        mesh.vertices = v
        mesh.faces = mesh.faces[:, ::-1]
    return mesh


def classify_defect(mesh) -> str:
    if bool(mesh.is_watertight):
        return "watertight"
    return boundary_defect(mesh)["defect_class"]


def z_reference_check(city: str, z_mins) -> dict:
    """Cross-check a sample of a city's (corrected) mesh z-origins against #157's documented
    reference elevation. Returns a report; raises ValueError for a hard ('asl'/'relative_zero')
    violation -- an actual scale/CRS bug, not the already-known messiness #157 disclosed for
    'unreliable'/'undocumented' cities."""
    slug = city_slug(city)
    mode, band = CITY_Z_REFERENCE[slug]
    z_mins = np.asarray(list(z_mins), dtype=float)
    if z_mins.size == 0:
        return dict(city=city, mode=mode, ok=True, note="no sample")
    median = float(np.median(z_mins))
    if mode == "asl":
        low, high = band
        ok = (low - Z_ASL_TOL_M) <= median <= (high + Z_ASL_TOL_M)
        if not ok:
            raise ValueError(
                f"#157/#174: {city}'s corrected z-origin (median {median:.1f} m over "
                f"{z_mins.size} sampled buildings) is outside the documented reference-elevation "
                f"band ({low}, {high}) m +/-{Z_ASL_TOL_M} m tolerance -- likely a wrong CRS/units "
                "correction, not real building variation. Refusing to ingest this city.")
        return dict(city=city, mode=mode, ok=True, median_z_min_m=median, band=list(band))
    if mode == "relative_zero":
        ok = abs(median) <= Z_RELATIVE_TOL_M
        if not ok:
            raise ValueError(
                f"#157/#174: {city} is expected to carry per-building height-above-base "
                f"(z_min always ~0), but its sampled median z-origin is {median:.2f} m -- this "
                "city's mesh export appears to have changed since #157's audit. Refusing to "
                "ingest until re-audited.")
        return dict(city=city, mode=mode, ok=True, median_z_min_m=median)
    # 'unreliable' / 'undocumented': disclosed by #157 as not gateable; log only.
    return dict(city=city, mode=mode, ok=True, median_z_min_m=median,
               note="#157 disclosed this city's z-origin as not reliably checkable; not gated")


def load_dedup_excluded_ids(artifact_path: Path) -> dict:
    """City -> set of BuildingWorld mesh ids (obj filename stem) #160's duplicate gate flagged as
    matching an existing PLATEAU/NRW row. Empty sets for both pairs at the recorded snapshot."""
    data = json.loads(Path(artifact_path).read_text())
    out = {}
    for city, pair_key in DEDUP_PAIR_FOR_CITY.items():
        pair = data.get("pairs", {}).get(pair_key, {})
        out[city] = {m["candidate_id"] for m in pair.get("matches", [])}
    return out


def process_member(city: str, member_name: str, obj_bytes: bytes, r: int, dedup_excluded: set,
                   min_h: float, max_ext: float, min_fp: int) -> dict:
    """One candidate mesh -> {"status": "kept", ...row fields} or {"status": "skip", "reason": ...}.

    Pure given its inputs (no zip/h5 I/O) -- the seam every filter/correction decision above is
    tested through.
    """
    import io

    import trimesh

    mesh_id = Path(member_name).stem
    if mesh_id in dedup_excluded:
        return dict(status="skip", reason="dedup_excluded")
    try:
        mesh = trimesh.load(io.BytesIO(obj_bytes), file_type="obj")
    except Exception as e:
        return dict(status="skip", reason="load_error", detail=repr(e))
    if not hasattr(mesh, "faces") or len(mesh.faces) == 0 or len(mesh.vertices) == 0:
        return dict(status="skip", reason="load_error", detail="no faces/vertices")

    mesh = apply_geometric_correction(mesh, city, member_name)
    raw_z_min = float(mesh.bounds[0, 2])
    max_extent = float(mesh.extents.max())
    extent_gt_90 = max_extent > DEFAULT_MAX_EXT

    try:
        defect_class = classify_defect(mesh)
    except Exception as e:
        return dict(status="skip", reason="defect_error", detail=repr(e))
    if defect_class not in KEEP_DEFECT_CLASSES:
        return dict(status="skip", reason=f"defect:{defect_class}", extent_gt_90=extent_gt_90,
                   raw_z_min=raw_z_min)

    if mesh.extents[2] < min_h or max_extent > max_ext:
        return dict(status="skip", reason="extent", extent_gt_90=extent_gt_90, raw_z_min=raw_z_min)

    try:
        sdf, fp, height_m = building_to_sdf(mesh, r)
    except Exception as e:
        return dict(status="skip", reason="sdf_error", detail=repr(e), extent_gt_90=extent_gt_90,
                   raw_z_min=raw_z_min)
    occ = float((sdf <= 0).mean())
    if int(fp.sum()) < min_fp or not (0.01 < occ < 0.7):
        return dict(status="skip", reason="footprint_or_occupancy", extent_gt_90=extent_gt_90,
                   raw_z_min=raw_z_min)

    return dict(status="kept", extent_gt_90=extent_gt_90, raw_z_min=raw_z_min,
               sdf=sdf, footprint=fp, height_m=np.float32(height_m),
               style_id=np.int32(STYLE_ID_REAL), source_id=np.int32(SOURCE_ID_UNASSIGNED),
               class_label=class_label_for(city), bag_id=bag_id_for(city, member_name),
               source_key=source_key_for(city).encode("ascii"),
               defect_class=defect_class.encode("ascii"))


# ---- incremental per-city staging writer (resumable; models precompute_vecset_latents.py) ----

class IncrementalCityWriter:
    """Resizable, resumable h5 writer for one city's kept rows.

    Holding every kept SDF volume for a large city (Berlin: ~500k candidates) in memory before one
    write is not viable (hundreds of GB); flushing periodically to a resizable dataset, with a
    `committed_rows` boundary advanced only after every column reaches disk, is #161's own pattern
    (`precompute_vecset_latents.py`) reused rather than reinvented.
    """
    SCHEMA_VERSION = 1

    def __init__(self, path: Path, r: int, resume: bool = True, flush_every: int = 200):
        self.path, self.r, self.flush_every = Path(path), r, flush_every
        self.buf: dict = {k: [] for k in STAGING_COLUMNS}
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
                raise SystemExit(f"#174: cannot resume {path}: schema {version}, need "
                                f"{self.SCHEMA_VERSION}")
            committed = int(self.f.attrs.get("committed_rows", 0))
            for k in STAGING_COLUMNS:
                if k in self.f and self.f[k].shape[0] > committed:
                    self.f[k].resize(committed, axis=0)
            self.f.flush()
            if "bag_id" in self.f:
                self.done = {bytes(b).rstrip(b"\0") for b in self.f["bag_id"][:committed]}

    def already_done(self, bag_id: bytes) -> bool:
        return bag_id in self.done

    def add(self, **row) -> None:
        for k in STAGING_COLUMNS:
            self.buf[k].append(row[k])
        if len(self.buf["bag_id"]) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        n = len(self.buf["bag_id"])
        if not n:
            return
        committed = int(self.f.attrs["committed_rows"])
        sdf_arr = np.stack(self.buf["sdf"]).astype(np.float32)
        fp_arr = np.stack(self.buf["footprint"]).astype(np.uint8)
        for k in STAGING_COLUMNS:
            if k in ("sdf", "footprint"):
                arr = sdf_arr if k == "sdf" else fp_arr
            else:
                arr = np.asarray(self.buf[k], STAGING_DTYPES[k])
            if k not in self.f:
                chunks = (1,) + arr.shape[1:] if arr.ndim > 1 else True
                self.f.create_dataset(k, data=arr, maxshape=(None,) + arr.shape[1:],
                                      chunks=chunks, compression="lzf" if k in
                                      ("sdf", "footprint") else None)
            else:
                d = self.f[k]
                d.resize(committed + n, axis=0)
                d[committed:] = arr
        self.f.flush()
        self.f.attrs.modify("committed_rows", committed + n)
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


# ---- per-city staging driver --------------------------------------------------------------

def stage_city(city: str, r: int, limit: int, min_h: float, max_ext: float, min_fp: int,
              dedup_excluded: dict, z_sanity_sample: int, out_dir: Path = STAGING_DIR) -> dict:
    zpath, other_zip_counts = canonical_zip(city)
    zpath = Path(zpath)
    slug = city_slug(city)
    out_path = out_dir / f"{slug}.h5"
    excluded = dedup_excluded.get(city, set())

    counters: dict = {}
    z_sample: list = []
    n_extent_gt_90 = 0
    n_seen = 0
    t0 = time.time()
    z_checked = False

    with zipfile.ZipFile(zpath) as zf, IncrementalCityWriter(out_path, r) as writer:
        names = sorted(n for n in zf.namelist() if n.lower().endswith(".obj"))
        if limit:
            names = names[:limit]
        for member_name in names:
            bag_id = bag_id_for(city, member_name)
            if writer.already_done(bag_id):
                continue
            n_seen += 1
            rec = process_member(city, member_name, zf.read(member_name), r, excluded,
                                min_h, max_ext, min_fp)
            if rec.get("extent_gt_90"):
                n_extent_gt_90 += 1
            if rec["status"] == "kept":
                if len(z_sample) < z_sanity_sample:
                    z_sample.append(rec["raw_z_min"])
                writer.add(**{k: rec[k] for k in STAGING_COLUMNS})
                counters["kept"] = counters.get("kept", 0) + 1
            else:
                counters[rec["reason"]] = counters.get(rec["reason"], 0) + 1
            if not z_checked and len(z_sample) >= min(z_sanity_sample, 20):
                z_reference_check(city, z_sample)
                z_checked = True
            if n_seen % 2000 == 0:
                print(f"  [{city}] seen={n_seen}/{len(names)} kept={counters.get('kept', 0)} "
                     f"({time.time() - t0:.0f}s)", flush=True)
        writer.flush()

    if not z_checked and z_sample:
        z_reference_check(city, z_sample)

    return dict(city=city, slug=slug, canonical_zip=str(zpath),
               other_zip_member_counts=other_zip_counts, population=len(names),
               n_seen=n_seen, n_extent_gt_90=n_extent_gt_90, counters=counters,
               out_path=str(out_path), elapsed_s=round(time.time() - t0, 1))


# ---- combine staged cities into real.h5 (safe rebuild + atomic replace) -------------------

def combine_into_real(real_path: Path, staged_paths: list, out_path: Path,
                      blk: int = 512) -> dict:
    """Rebuild `real.h5`-shaped output = the untouched original's rows + every staged city's rows.

    `real_path` is opened read-only and never mutated. The result is written to a temp path,
    the frozen-prefix pin re-verified on IT (not just the source), and only then atomically moved
    to `out_path` -- #162's own requirement, applied to the write side, not just the read side.
    """
    with open_real_corpus(real_path) as src:
        n_old = src["sdf"].shape[0]
        r_shape = src["sdf"].shape[1:]
        fp_shape = src["footprint"].shape[1:]

        staged = []
        for p in staged_paths:
            with h5py.File(p, "r") as sf:
                # A city that kept zero rows never creates its datasets (IncrementalCityWriter's
                # `flush` returns early on an empty buffer) -- an empty staged file, not an error.
                n = sf["bag_id"].shape[0] if "bag_id" in sf else 0
            if n:
                staged.append((p, n))
        n_new = sum(n for _, n in staged)
        n_total = n_old + n_new

        tmp = out_path.with_suffix(out_path.suffix + f".tmp{os.getpid()}")
        with h5py.File(tmp, "w") as dst:
            dst.create_dataset("sdf", (n_total, *r_shape), np.float32,
                              chunks=(1, *r_shape), compression="lzf")
            dst.create_dataset("footprint", (n_total, *fp_shape), np.uint8,
                              chunks=(1, *fp_shape), compression="lzf")
            dst.create_dataset("height_m", (n_total,), np.float32)
            dst.create_dataset("style_id", (n_total,), np.int32)
            dst.create_dataset("source_id", (n_total,), np.int32)
            dst.create_dataset("class_label", (n_total,), "S16")
            dst.create_dataset("bag_id", (n_total,), "S64")
            # New columns (#167/#166): absent from the historical 35,776 rows on purpose --
            # #167 explicitly scopes this ticket to NOT retrofitting them a value.
            dst.create_dataset("source_key", (n_total,), "S64")
            dst.create_dataset("defect_class", (n_total,), "S32")

            t0 = time.time()
            for i in range(0, n_old, blk):
                j = min(i + blk, n_old)
                dst["sdf"][i:j] = src["sdf"][i:j]
                dst["footprint"][i:j] = src["footprint"][i:j]
                dst["height_m"][i:j] = src["height_m"][i:j]
                dst["style_id"][i:j] = src["style_id"][i:j]
                dst["source_id"][i:j] = src["source_id"][i:j]
                dst["class_label"][i:j] = src["class_label"][i:j]
                dst["bag_id"][i:j] = src["bag_id"][i:j]
                if j % (blk * 20) == 0 or j == n_old:
                    print(f"  [combine] copied {j}/{n_old} existing rows "
                         f"({time.time() - t0:.0f}s)", flush=True)

            off = n_old
            per_city_written = {}
            for p, n in staged:
                with h5py.File(p, "r") as sf:
                    for i in range(0, n, blk):
                        j = min(i + blk, n)
                        for col in STAGING_COLUMNS:
                            dst[col][off + i:off + j] = sf[col][i:j]
                per_city_written[Path(p).stem] = n
                off += n

            dst.attrs["source"] = "buildingworld"
            dst.attrs["ingested_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            dst.attrs["buildingworld_cities"] = json.dumps(per_city_written, sort_keys=True)

    assert_frozen_corpus(h5py.File(tmp, "r"))
    tmp.replace(out_path)
    return dict(n_old=n_old, n_new=n_new, n_total=n_total, per_city=per_city_written,
               out_path=str(out_path))


# ---- CLI --------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cities", default="", help="comma-separated subset of INGESTABLE_CITIES; "
                                                  "default = all 18 (Toronto excluded per #165)")
    ap.add_argument("--limit", type=int, default=0, help="cap candidates PER CITY, 0 = no cap")
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--min_h", type=float, default=DEFAULT_MIN_H)
    ap.add_argument("--max_ext", type=float, default=DEFAULT_MAX_EXT)
    ap.add_argument("--min_fp", type=int, default=DEFAULT_MIN_FP)
    ap.add_argument("--z_sanity_sample", type=int, default=200)
    ap.add_argument("--dedup_artifact", default=str(DEDUP_ARTIFACT))
    ap.add_argument("--staging_dir", default=str(STAGING_DIR))
    ap.add_argument("--real", default=str(REAL_H5), help="corpus to combine ONTO (read-only)")
    ap.add_argument("--out", default="", help="combine output path; default = --real, or a "
                                              "_smoke-suffixed sibling under --smoke")
    ap.add_argument("--smoke", action="store_true",
                    help="small, safe defaults + writes combine output to a throwaway file")
    ap.add_argument("--no_stage", action="store_true", help="skip staging; combine only")
    ap.add_argument("--no_combine", action="store_true", help="stage only; do not touch real.h5")
    args = ap.parse_args()

    cities = args.cities.split(",") if args.cities else INGESTABLE_CITIES
    unknown = set(cities) - set(INGESTABLE_CITIES)
    if unknown:
        raise SystemExit(f"#174: not ingestable (excluded by #165, or misspelled): {sorted(unknown)}")

    staging_dir = Path(args.staging_dir)
    real_path = Path(args.real)
    out_path = Path(args.out) if args.out else (
        real_path.with_name(real_path.stem + "_smoke" + real_path.suffix) if args.smoke
        else real_path)

    dedup_excluded = {}
    needs_dedup = set(cities) & set(DEDUP_PAIR_FOR_CITY)
    if needs_dedup:
        if not Path(args.dedup_artifact).exists():
            raise SystemExit(f"#174: {args.dedup_artifact} not found -- required before ingesting "
                            f"{sorted(needs_dedup)} (#160's duplicate gate). Run "
                            "dedup_buildingworld_geometric.py first.")
        dedup_excluded = load_dedup_excluded_ids(Path(args.dedup_artifact))

    if not args.no_stage:
        for city in cities:
            print(f"[stage] {city}", flush=True)
            s = stage_city(city, args.res, args.limit, args.min_h, args.max_ext,
                          args.min_fp, dedup_excluded, args.z_sanity_sample, staging_dir)
            print(f"[stage:done] {s['city']}: kept={s['counters'].get('kept', 0)}/"
                 f"{s['n_seen']} seen, extent_gt_90={s['n_extent_gt_90']} "
                 f"reasons={s['counters']} ({s['elapsed_s']}s)", flush=True)

    if not args.no_combine:
        staged_paths = [staging_dir / f"{city_slug(c)}.h5" for c in cities]
        missing = [p for p in staged_paths if not p.exists()]
        if missing:
            raise SystemExit(f"#174: missing staged file(s), run without --no_stage first: "
                            f"{[str(p) for p in missing]}")
        print(f"[combine] rebuilding {out_path} from {real_path} + {len(staged_paths)} staged "
             "file(s) -- this copies the ENTIRE existing corpus (tens of GB of SDF volumes), "
             "not just the new rows; expect this to dominate the run's wall time", flush=True)
        result = combine_into_real(real_path, staged_paths, out_path)
        print(f"[combine] {result['n_old']} old + {result['n_new']} new = {result['n_total']} "
             f"rows -> {result['out_path']}", flush=True)


if __name__ == "__main__":
    main()
