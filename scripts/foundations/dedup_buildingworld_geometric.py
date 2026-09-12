"""#160: a GEOMETRIC duplicate gate between BuildingWorld candidates and the existing corpus.

BuildingWorld Tokyo mesh ids follow the pattern `bldg_<uuid>` -- PLATEAU's own gml:id convention --
and the corpus's existing JP rows (`source_id=2`) come from the same PLATEAU tokyo23ku tiles, so
some BuildingWorld Tokyo meshes may be geometric twins of buildings already in `real.h5`, possibly
including rows in the FROZEN held-out set the served arm's headline `extra=0.0603` is measured
against. Berlin's ids share Germany's official AdV LoD2 id family with the existing NRW rows, at
lower a priori risk. `bag_id` is a truncated fixed-width `S64` field (PLATEAU uuids are observed
cut off mid-string), so id matching cannot be trusted as the gate -- only as a cheap pre-filter.
This module computes the gate geometrically instead: footprint IoU plus |height_m| match, both
computed directly from geometry in a shared real-world coordinate frame, never from `real.h5`'s own
`footprint` column (RECENTRED, RESCALED into a per-building local frame that cannot be compared
across two independently-sourced datasets -- see the wayfinding doc for how this was established
before any matching code was written). `real.h5`'s `height_m` *is* read, but only as an independent
consistency check on the re-fetch itself (`height_consistency_mismatches`), never as a value fed
into the match decision.

Run: env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/dedup_buildingworld_geometric.py
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.foundations.corpus_ledger import LEDGER_PATH, read_ledger  # noqa: E402

REAL_H5 = REPO / "data/real_massing_v1/real.h5"
BW_ROOT = REPO / "data/buildingworld_mesh"
SOURCE_ID = {"nl": 0, "nrw": 1, "plateau": 2}
# PLATEAU tiles real.h5's 12,000 JP rows actually come from (verified against the live file --
# not the ingester's wider PLATEAU_TILES list, which includes tiles never actually ingested).
PLATEAU_TILES_IN_CORPUS = ["533937_2.zip", "533954_2.zip", "533957_2.zip"]


# ---- core geometry (TDD seam: pure, no I/O) -------------------------------------------------

def hull_polygon_and_centroid(xy: np.ndarray):
    """World-frame XY points -> (convex-hull Polygon, centroid). None if degenerate (<3 points,
    or collinear -- zero-area hull)."""
    from shapely.geometry import MultiPoint

    pts = np.asarray(xy, float)
    if len(pts) < 3:
        return None
    hull = MultiPoint(pts).convex_hull
    if hull.geom_type != "Polygon" or hull.area <= 0:
        return None
    return hull, np.asarray(hull.centroid.coords[0], float)


def polygon_iou(a, b) -> float:
    """Exact intersection-over-union of two shapely polygons. 0.0 for a zero-area union."""
    union = a.union(b).area
    return float(a.intersection(b).area / union) if union > 0 else 0.0


def is_duplicate(height_a: float, height_b: float, iou: float,
                 iou_threshold: float = 0.3, height_tol_m: float = 2.0) -> bool:
    """#160's match rule: footprint IoU AND |height_m| must both agree. Either alone is cheap to
    satisfy by coincidence between two independently-sourced buildings at the same rough location
    (an unrelated same-plot infill matches height; an unrelated taller/shorter neighbour matches
    footprint) -- only requiring both is the actual duplicate signature."""
    return iou >= iou_threshold and abs(height_a - height_b) <= height_tol_m


def spatial_bucket(xy: np.ndarray, cell_m: float) -> tuple[int, int]:
    """Grid-cell key for the coarse spatial pre-filter."""
    x, y = xy
    return int(np.floor(x / cell_m)), int(np.floor(y / cell_m))


def polygon_reach(polygon, centroid: np.ndarray) -> float:
    """Max distance from a polygon's own centroid to any of its hull vertices.

    Two polygons cannot possibly overlap (IoU > 0) if their centroids are farther apart than the
    SUM of their own reaches: every point in a polygon is within `reach` of its own centroid, so by
    the triangle inequality two centroids more than `reach_a + reach_b` apart cannot share a point.
    This is the one sound geometric bound for the bucketing pre-filter's cell size -- unlike a fixed
    cell size, it holds regardless of `iou_threshold` or how large a candidate building is.
    """
    pts = np.asarray(polygon.exterior.coords, dtype=float)
    if len(pts) == 0:
        return 0.0
    return float(np.max(np.linalg.norm(pts - np.asarray(centroid, dtype=float), axis=1)))


def bbox_overlaps(box_a, box_b, margin_m: float = 0.0) -> bool:
    """Whether two (xmin, ymin, xmax, ymax) world-frame boxes overlap after expanding both by
    margin_m -- the cheap city-level proof a candidate population can be run against an existing
    population's own extent before any per-building geometry is fetched or loaded."""
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ax0, ay0, ax1, ay1 = ax0 - margin_m, ay0 - margin_m, ax1 + margin_m, ay1 + margin_m
    return ax0 <= bx1 and bx0 <= ax1 and ay0 <= by1 and by0 <= ay1


def match_candidates(candidates: list, existing: list, cell_m: float = 15.0,
                     iou_threshold: float = 0.3, height_tol_m: float = 2.0) -> list:
    """Every (candidate, existing) pair that clears `is_duplicate`, restricted to pairs whose
    centroids share a spatial bucket or a neighbouring one -- true duplicates from two independent
    extraction pipelines will not centre on the exact same point, so the 3x3 neighbourhood (not
    just the candidate's own cell) is required, not an optimisation.

    `cell_m` is only a FLOOR on the bucket size actually used: a 3x3 neighbourhood only guarantees
    catching every pair whose centroids are within `cell_m` of each other (in each axis), so a
    fixed `cell_m` smaller than some pair's actual reach (see `polygon_reach`) would silently drop
    it. The effective cell size is widened to at least twice the largest polygon reach seen in
    `candidates`/`existing`, which by the triangle inequality guarantees every pair that could
    possibly overlap shares a bucket or a neighbouring one -- independent of building size or
    `iou_threshold`, not just true for the historical 15m default.

    Items: {id, polygon, height_m, centroid}. O(candidates x buildings-per-3x3-neighbourhood), not
    O(candidates x existing) -- the whole reason for the bucketing pre-filter.
    """
    reaches = [polygon_reach(it["polygon"], it["centroid"]) for it in (*candidates, *existing)]
    eff_cell_m = max(cell_m, 2.0 * max(reaches)) if reaches else cell_m

    buckets = defaultdict(list)
    for i, e in enumerate(existing):
        buckets[spatial_bucket(e["centroid"], eff_cell_m)].append(i)

    matches = []
    for c in candidates:
        cb = spatial_bucket(c["centroid"], eff_cell_m)
        seen = set()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for i in buckets.get((cb[0] + dx, cb[1] + dy), ()):
                    if i in seen:
                        continue
                    seen.add(i)
                    e = existing[i]
                    iou = polygon_iou(c["polygon"], e["polygon"])
                    if is_duplicate(c["height_m"], e["height_m"], iou, iou_threshold, height_tol_m):
                        matches.append(dict(candidate_id=c["id"], existing_id=e["id"], iou=iou,
                                            height_diff_m=abs(c["height_m"] - e["height_m"]),
                                            candidate_height_m=c["height_m"],
                                            existing_height_m=e["height_m"]))
    return matches


def height_consistency_mismatches(existing_rows: list, world: dict, tol_m: float = 0.05) -> list:
    """Every relocated building whose freshly-computed world-frame height disagrees with the
    height real.h5 already stored for that same row by more than tol_m.

    The two should be numerically identical: `height_m` is the one field `building_to_sdf` keeps
    in true, un-normalised metres (never recentred/rescaled, unlike `footprint`), computed from the
    SAME raw rings this module re-parses. A disagreement means the re-fetch matched the wrong
    building for that bag_id -- not just an id-construction coincidence -- so this is the check
    that would actually catch it, not merely a relocated-count tally.
    """
    mismatches = []
    for r in existing_rows:
        w = world.get(r["bag_id"])
        if w is None:
            continue
        diff = abs(w["height_m"] - r["height_m"])
        if diff > tol_m:
            mismatches.append(dict(bag_id=r["bag_id"], row=r["row"], stored_height_m=r["height_m"],
                                   recomputed_height_m=w["height_m"], diff_m=diff))
    return mismatches


def nearest_neighbor_distances(candidates: list, existing: list) -> np.ndarray:
    """Straight-line distance from every candidate centroid to its nearest existing centroid --
    the corroborating diagnostic for a zero-match result. `match_candidates` alone cannot
    distinguish "genuinely nothing nearby" from "something close that narrowly missed the IoU/
    height thresholds"; this can. inf for every candidate when `existing` is empty, never a crash.
    """
    if not candidates:
        return np.array([])
    if not existing:
        return np.full(len(candidates), np.inf)
    from scipy.spatial import cKDTree

    tree = cKDTree(np.asarray([e["centroid"] for e in existing]))
    d, _ = tree.query(np.asarray([c["centroid"] for c in candidates]), k=1)
    return np.atleast_1d(d)


# ---- real.h5 / corpus ledger index (I/O) --------------------------------------------------

def real_h5_index(h5_path: Path, source_id: int) -> list:
    """Every real.h5 row for one source: global row index, full bag_id, and its gml:id suffix."""
    with h5py.File(h5_path, "r") as f:
        sid = f["source_id"][:]
        bag_id = f["bag_id"][:]
        height_m = f["height_m"][:]
    out = []
    for i in np.flatnonzero(sid == source_id):
        bid = bag_id[i].decode()
        out.append(dict(row=int(i), bag_id=bid, height_m=float(height_m[i])))
    return out


def held_out_by_row(ledger_path: Path = LEDGER_PATH) -> dict:
    """real.h5 row index -> held_out flag, from the #161 ledger (split out of vecset_latents.h5, so
    this no longer opens the multi-gigabyte Dora-encoded latent store just for one uint8 column)."""
    ledger = read_ledger(ledger_path)
    return {int(r): int(h) for r, h in zip(ledger["row"], ledger["held_out"])}


# ---- PLATEAU: re-fetch the known tiles, world-frame (no per-building recentring) ------------

def plateau_world_buildings(tiles: list, wanted_bag_ids: set) -> dict:
    """Re-fetch PLATEAU tiles and return world-frame (EPSG:6677) footprint+height+centroid for
    every building whose reconstructed bag_id is one of `wanted_bag_ids` -- i.e. the ones actually
    ingested into real.h5. Reuses ingest_citygml_lod2.py's own GML parsing so the id constructed
    here is byte-identical to what real.h5 stores, but skips `_to_local_metres`'s per-building
    recentring: EPSG:6677 is the SAME plane rectangular CS BuildingWorld's own Tokyo meshes are
    already in (confirmed empirically before writing this function), so both sides land in one
    shared, directly-comparable frame without a manual reference-point choice.
    """
    import pyproj

    from scripts.foundations.ingest_citygml_lod2 import PLATEAU_BASE, _buildings_from_gml, _building_rings, _fetch

    # EPSG:6668 (JGD2011, horizontal-only) rather than 6697 (the compound 3D CRS
    # ingest_citygml_lod2.py's header sniff detects): only (lon, lat) are transformed here, height
    # comes straight from the posList's own metre column below, so the vertical-datum difference
    # between 6668 and 6697 never enters this computation. Same horizontal datum either way.
    to_6677 = pyproj.Transformer.from_crs("EPSG:6668", "EPSG:6677", always_xy=True)
    out = {}
    for tz in tiles:
        print(f"[plateau] fetch {tz}", flush=True)
        outer = zipfile.ZipFile(io.BytesIO(_fetch(PLATEAU_BASE + tz)))
        if "bldg.zip" not in outer.namelist():
            continue
        inner = zipfile.ZipFile(io.BytesIO(outer.read("bldg.zip")))
        for n in inner.namelist():
            if not n.endswith(".gml"):
                continue
            gml_key = f"{tz}:{n}"
            for gid, geographic, b in _buildings_from_gml(inner.read(n)):
                full_id = f"{gml_key}#{gid}"[:64]
                if full_id not in wanted_bag_ids:
                    continue
                rings = _building_rings(b)
                if len(rings) < 3:
                    continue
                allp = np.concatenate(rings, 0)
                if not geographic:
                    raise ValueError(f"#160: expected a geographic PLATEAU posList for {full_id}")
                x, y = to_6677.transform(allp[:, 1], allp[:, 0])   # always_xy: (lon, lat) -> (x, y)
                result = hull_polygon_and_centroid(np.stack([x, y], 1))
                if result is None:
                    continue
                poly, centroid = result
                height_m = float(allp[:, 2].max() - allp[:, 2].min())
                out[full_id] = dict(id=full_id, polygon=poly, height_m=height_m, centroid=centroid)
    return out


# ---- NRW: re-fetch the exact tiles real.h5's rows came from, world-frame -------------------

def nrw_tile_files_from_bag_ids(bag_ids: list) -> list:
    """Unique, sorted NRW tile filenames a set of bag_ids were ingested from. Needs no lookup
    table: unlike PLATEAU's opaque area-mesh codes, the NRW tile filename is embedded verbatim as
    the bag_id prefix, so this doubles as the exact fetch list for `nrw_world_buildings`."""
    tiles = set()
    for s in bag_ids:
        m = re.match(r"(LoD2_32_\d+_\d+_1_NW\.gml)", s)
        if not m:
            raise ValueError(f"#160: unrecognised NRW bag_id tile pattern: {s!r}")
        tiles.add(m.group(1))
    return sorted(tiles)


def nrw_bbox_from_bag_ids(bag_ids: list) -> tuple:
    """World bbox covering every NRW row, from its `LoD2_32_<E>_<N>_1_NW.gml` 1km UTM32 tile-grid
    id -- string parsing only, no fetch. Each id names a tile's SW corner, so the box is padded by
    1 km so the true extent (which reaches the tile's NE corner too) is never underclaimed."""
    Es, Ns = [], []
    for tf in nrw_tile_files_from_bag_ids(bag_ids):
        m = re.match(r"LoD2_32_(\d+)_(\d+)_1_NW\.gml", tf)
        Es.append(int(m.group(1)) * 1000.0)
        Ns.append(int(m.group(2)) * 1000.0)
    return (min(Es), min(Ns), max(Es) + 1000.0, max(Ns) + 1000.0)


def nrw_world_buildings(tile_files: list, wanted_bag_ids: set) -> dict:
    """Re-fetch NRW tile files and return world-frame footprint+height+centroid for every building
    whose reconstructed bag_id is one of `wanted_bag_ids`. NRW's posList is already projected
    (EPSG:25832, cols E/N/H in metres) -- unlike PLATEAU, used directly, no reprojection."""
    from scripts.foundations.ingest_citygml_lod2 import NRW_DIR, _buildings_from_gml, _building_rings, _fetch

    out = {}
    for tf in tile_files:
        print(f"[nrw] fetch {tf}", flush=True)
        for gid, geographic, b in _buildings_from_gml(_fetch(NRW_DIR + tf)):
            full_id = f"{tf}#{gid}"[:64]
            if full_id not in wanted_bag_ids:
                continue
            rings = _building_rings(b)
            if len(rings) < 3:
                continue
            allp = np.concatenate(rings, 0)
            if geographic:
                raise ValueError(f"#160: expected a projected NRW posList for {full_id}")
            result = hull_polygon_and_centroid(allp[:, :2])   # (E, N) metres, used directly
            if result is None:
                continue
            poly, centroid = result
            height_m = float(allp[:, 2].max() - allp[:, 2].min())
            out[full_id] = dict(id=full_id, polygon=poly, height_m=height_m, centroid=centroid)
    return out


# ---- BuildingWorld: already world-frame, load directly --------------------------------------

def buildingworld_world_buildings(zip_path: Path, ids=None, limit=None, seed=None):
    """Yield {id, polygon, height_m, centroid} for BuildingWorld meshes in a city zip -- already
    real-world-magnitude, non-recentred coordinates (confirmed empirically), used as-is.

    `ids`: only these member names (id = stem). `limit`+`seed`: a fixed-seed random sample instead
    (for the cheap Berlin bbox estimate) -- both None iterates every mesh in the zip.
    """
    import trimesh

    zp = zipfile.ZipFile(zip_path)
    names = [n for n in zp.namelist() if n.endswith(".obj")]
    if ids is not None:
        wanted = set(ids)
        names = [n for n in names if Path(n).stem.split("bldg_")[-1] in wanted
                or Path(n).stem in wanted]
    elif seed is not None:
        import random
        names = random.Random(seed).sample(names, min(limit or len(names), len(names)))
    elif limit is not None:
        names = names[:limit]
    for n in names:
        data = zp.read(n)
        m = trimesh.load(trimesh.util.wrap_as_stream(data), file_type="obj", process=False)
        v = np.asarray(m.vertices, float)
        result = hull_polygon_and_centroid(v[:, :2])
        if result is None:
            continue
        poly, centroid = result
        height_m = float(v[:, 2].max() - v[:, 2].min())
        yield dict(id=Path(n).stem, polygon=poly, height_m=height_m, centroid=centroid)


def buildingworld_bbox_sample(zip_path: Path, n: int = 500, seed: int = 160) -> tuple:
    """Empirical world bbox from a fixed-seed sample -- cheap enough to run without deciding in
    advance whether the full population is worth loading."""
    xs0, ys0, xs1, ys1 = [], [], [], []
    for b in buildingworld_world_buildings(zip_path, limit=n, seed=seed):
        bx0, by0, bx1, by1 = b["polygon"].bounds
        xs0.append(bx0); ys0.append(by0); xs1.append(bx1); ys1.append(by1)
    return (min(xs0), min(ys0), max(xs1), max(ys1))


# ---- per-pair orchestration -------------------------------------------------------------------

def _run_pair(pair_name: str, existing_rows: list, world: dict, candidates: list,
             cell_m: float, iou_threshold: float, height_tol_m: float,
             min_relocate_frac: float = 0.98, held_out: dict = None) -> dict:
    """Shared match/report step once both sides' world-frame geometry is in hand.

    A live re-fetch will not always cleanly reproduce every existing row: NRW's open-data portal
    is periodically republished. The real Berlin/NRW run found two distinct drift patterns, both
    confirmed by hand to be genuine upstream changes, not bugs in this module:

      * 150/12,000 rows (1.25%, scattered across 23 of 41 tiles) whose bag_id no longer appears in
        the current tile content at all (one tile came back a valid, well-formed, deliberately
        near-empty 1.5KB GML with zero Buildings) -- a SMALL, disclosed gap here is tolerated
        (`min_relocate_frac`) and reported; a larger one fails loudly rather than silently
        shrinking the population a 'zero matches' verdict is actually checked against.
      * 1,296/11,850 relocated rows (10.9%) whose freshly re-fetched height disagrees with
        real.h5's stored value -- verified for one case that this module's height computation
        exactly reproduces `building_to_sdf`'s own mesh-extent result on the SAME fresh fetch
        (3.828 m both ways), so the disagreement is the source building's geometry actually having
        changed since ingestion, not a computation bug. These rows are EXCLUDED from the matched
        `existing` population rather than hard-failing the run: their current geometry cannot be
        trusted to represent the row that is actually frozen in the corpus, so matching against it
        would be checking the wrong building either way -- exclusion is already the safe response,
        not something a human needs to approve before the run can proceed.
    """
    wanted = {r["bag_id"] for r in existing_rows}
    relocate_frac = len(world) / len(wanted) if wanted else 1.0
    if relocate_frac < min_relocate_frac:
        raise SystemExit(f"#160 [{pair_name}]: only relocated {len(world)}/{len(wanted)} "
                         f"({relocate_frac:.1%}) existing rows in world coordinates -- below the "
                         f"{min_relocate_frac:.0%} floor for disclosed upstream drift; a fetch or "
                         "parse likely failed. Refusing to report a 'safe' result off this.")
    unrelocated_bag_ids = sorted(wanted - set(world))

    # tol_m matches the gate's own height_tol_m: a disagreement the match rule itself would treat
    # as "the same building" is not evidence of source drift, only a stricter threshold here would
    # manufacture one (#160 code review: 0.05m excluded 1,296 NRW rows, 82.7% of which drifted
    # under 2m and would have matched correctly under the gate's own rule).
    drift = height_consistency_mismatches(existing_rows, world, tol_m=height_tol_m)
    drifted_ids = {d["bag_id"] for d in drift}
    if drifted_ids:
        print(f"[{pair_name}] excluding {len(drifted_ids)}/{len(world)} relocated rows from "
            "matching: freshly re-fetched height disagrees with real.h5's stored value "
            "(upstream source drift, not a fetch bug -- see _run_pair's docstring)", flush=True)

    existing = [w for bag_id, w in world.items() if bag_id not in drifted_ids]
    row_by_bag_id = {r["bag_id"]: r for r in existing_rows}
    held_out = held_out or {}

    matches = match_candidates(candidates, existing, cell_m, iou_threshold, height_tol_m)
    for m in matches:
        row = row_by_bag_id[m["existing_id"]]
        m["existing_row"] = row["row"]
        m["existing_held_out"] = held_out.get(row["row"])

    nn_dist = nearest_neighbor_distances(candidates, existing)
    nn_summary = (dict(min=float(nn_dist.min()),
                       percentiles={str(p): float(np.percentile(nn_dist, p)) for p in (1, 5, 25, 50)})
                 if len(nn_dist) else None)
    return dict(method="geometric", n_candidates=len(candidates),
               n_existing_relocated=len(world), n_existing_verified=len(existing),
               n_existing_total=len(existing_rows),
               unrelocated_bag_ids=unrelocated_bag_ids, height_drifted=drift,
               n_matches=len(matches), matches=matches,
               nearest_neighbor_distance_m=nn_summary)


def run_tokyo_vs_plateau(cell_m: float, iou_threshold: float, height_tol_m: float,
                         min_relocate_frac: float = 0.98) -> dict:
    existing_rows = real_h5_index(REAL_H5, SOURCE_ID["plateau"])
    world = plateau_world_buildings(PLATEAU_TILES_IN_CORPUS, {r["bag_id"] for r in existing_rows})
    print(f"[plateau] {len(world)}/{len(existing_rows)} real.h5 JP rows re-located in world "
        "coordinates", flush=True)

    zip_path = BW_ROOT / "Tokyo/mesh/mesh.zip"
    candidates = list(buildingworld_world_buildings(zip_path))
    print(f"[buildingworld] {len(candidates)} Tokyo meshes loaded", flush=True)

    return _run_pair("tokyo_vs_plateau", existing_rows, world, candidates,
                     cell_m, iou_threshold, height_tol_m, min_relocate_frac,
                     held_out=held_out_by_row())


def run_berlin_vs_nrw(cell_m: float, iou_threshold: float, height_tol_m: float,
                      bbox_margin_m: float, bbox_sample_n: int,
                      min_relocate_frac: float = 0.98) -> dict:
    existing_rows = real_h5_index(REAL_H5, SOURCE_ID["nrw"])
    wanted_bag_ids = {r["bag_id"] for r in existing_rows}

    # Cheap, informative pre-check -- logged, never a substitute for the full run below: the
    # ticket asks for every NRW row checked against every Berlin candidate, and that's what runs
    # regardless of what this says.
    zip_path = BW_ROOT / "Berlin/mesh/mesh.zip"
    nrw_box = nrw_bbox_from_bag_ids(list(wanted_bag_ids))
    berlin_sample_box = buildingworld_bbox_sample(zip_path, n=bbox_sample_n)
    print(f"[berlin_vs_nrw] pre-check: NRW bbox {nrw_box}, Berlin {bbox_sample_n}-sample bbox "
        f"{berlin_sample_box}, overlap={bbox_overlaps(nrw_box, berlin_sample_box, bbox_margin_m)}",
        flush=True)

    tile_files = nrw_tile_files_from_bag_ids(list(wanted_bag_ids))
    world = nrw_world_buildings(tile_files, wanted_bag_ids)
    print(f"[nrw] {len(world)}/{len(existing_rows)} real.h5 DE rows re-located in world "
        "coordinates", flush=True)

    candidates = list(buildingworld_world_buildings(zip_path))
    print(f"[buildingworld] {len(candidates)} Berlin meshes loaded", flush=True)

    result = _run_pair("berlin_vs_nrw", existing_rows, world, candidates,
                       cell_m, iou_threshold, height_tol_m, min_relocate_frac,
                       held_out=held_out_by_row())
    result["bbox_precheck"] = dict(nrw_bbox_m=list(nrw_box), berlin_sample_bbox_m=list(berlin_sample_box),
                                   sample_n=bbox_sample_n, bbox_margin_m=bbox_margin_m)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell-m", type=float, default=15.0)
    ap.add_argument("--iou-threshold", type=float, default=0.3)
    ap.add_argument("--height-tol-m", type=float, default=2.0)
    ap.add_argument("--bbox-margin-m", type=float, default=5_000.0)
    ap.add_argument("--berlin-sample-n", type=int, default=500)
    ap.add_argument("--min-relocate-frac", type=float, default=0.98,
                    help="Floor on the fraction of existing rows a live re-fetch must relocate in "
                         "world coordinates before a pair's result is trusted; below it fails "
                         "loudly rather than reporting a match count against a shrunken "
                         "population. See _run_pair's docstring for the disclosed real gap this "
                         "tolerates (NRW upstream source drift).")
    ap.add_argument("--out", type=Path,
                    default=REPO / "execution/artifacts/buildingworld_duplicate_gate.json")
    args = ap.parse_args()
    t0 = time.time()

    tokyo = run_tokyo_vs_plateau(args.cell_m, args.iou_threshold, args.height_tol_m,
                                 args.min_relocate_frac)
    print(f"[tokyo_vs_plateau] {tokyo['n_matches']} matches "
        f"({tokyo['n_candidates']} candidates x {tokyo['n_existing_verified']} verified existing, "
        f"{len(tokyo['unrelocated_bag_ids'])} unrelocated, {len(tokyo['height_drifted'])} drifted)",
        flush=True)

    berlin = run_berlin_vs_nrw(args.cell_m, args.iou_threshold, args.height_tol_m,
                               args.bbox_margin_m, args.berlin_sample_n, args.min_relocate_frac)
    print(f"[berlin_vs_nrw] {berlin['n_matches']} matches "
        f"({berlin['n_candidates']} candidates x {berlin['n_existing_verified']} verified existing, "
        f"{len(berlin['unrelocated_bag_ids'])} unrelocated, {len(berlin['height_drifted'])} drifted)",
        flush=True)

    out = dict(meta=dict(created=time.strftime("%Y-%m-%dT%H:%M:%S"), question="#160",
                        cell_m=args.cell_m, iou_threshold=args.iou_threshold,
                        height_tol_m=args.height_tol_m, bbox_margin_m=args.bbox_margin_m,
                        elapsed_s=round(time.time() - t0, 1)),
             pairs=dict(tokyo_vs_plateau=tokyo, berlin_vs_nrw=berlin))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[out] {args.out}", flush=True)


if __name__ == "__main__":
    main()
