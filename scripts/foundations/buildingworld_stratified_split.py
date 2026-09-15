"""#177 -- a roof-family-stratified, spatially-blocked held-out split for the new BuildingWorld
rows, EXTENDING (never mutating) the existing frozen 35,776-row held-out split #162 pins.

Two axes, per the issue text, not one:

  1. ROOF-FAMILY stratification, from #176's own `program_labels_buildingworld.h5` (`flat` / `gable`
     / `hip` / `complex`) -- a fitter-derived pseudo-label, not independent ground truth (a gable the
     fitter misreads as stacked Layers is filed under `complex` by construction; a model that
     reproduces the same misreading scores consistently on both the training label and this eval
     bucket). Treated here as "which roof shape the fitter can already find", used only to keep the
     held-out slice from skewing flat by accident -- not trusted as a ceiling on true difficulty.
  2. SPATIAL blocking within each city's own corrected coordinate system (#165), so dense contiguous
     coverage (Berlin ~457k kept rows, Calgary ~166k) cannot put near-identical neighbouring
     terrace/tract-housing buildings on both sides of the boundary. `real.h5`/`surfaces_
     buildingworld.h5` never stored a row's absolute position (#153 already named this same gap for
     NL); `buildingworld_spatial_keys.py` (#177) recovers it, once, from the raw mesh.

WHY A GRID, NOT #153'S OWN WHOLE-TILE KEY
------------------------------------------
#153's own "what this doesn't decide" section is explicit that #177 "must choose its own balance
policy and valid spatial keys; do not reuse the algorithm unchanged by assumption" -- NL/DE/JP had a
natural tiling (gemeente code, CityGML source filename) already baked into their `bag_id`.
BuildingWorld has no equivalent: its member-name conventions are inconsistent and not reliably
spatial across cities (sequential integers for Calgary, UUIDs for Tokyo, suburb-name subfolders only
for some). A fixed-size metre grid over each group's own corrected (x, y) is used instead --
`GRID_CELL_M` (default 150 m, comfortably larger than a terrace/row-house frontage) is the atomic,
never-split held-out unit, playing the same structural role #153's whole tile did: a row's tile
membership decides its split, never the row alone.

POOLING (#169)
---------------
Four named, pre-filter-small cities (Adelaide, Greater Geelong, Perth, Philadelphia) are POOLED into
one combined administrative bucket rather than each growing its own too-small slice -- #169's own
decision, applied here verbatim and scoped to exactly those four cities (no general small-city
threshold). Every other ingested city gets its own independent 2% grid-tiled, family-stratified
slice. Toronto was never ingested (#165) and needs no entry here.

EVERY ROW GETS AN EXPLICIT ASSIGNMENT
---------------------------------------
Per the issue text: an unassigned new row is an error, never a silent `held_out=0` default. `run()`
requires every `source_id == -1` row in `real.h5` to have both an `ok=1` centroid
(`buildingworld_spatial_keys.py`) and an `ok=1` program label (#176) before it will write anything;
missing coverage aborts loudly, naming the rows, rather than filling the gap with a default.

NEVER MUTATES THE EXISTING NL/DE/JP LEDGER
---------------------------------------------
`corpus_ledger.append_ledger` (#161) refuses to overwrite a row already present -- this script only
ever appends new BuildingWorld rows. `run()` additionally re-reads the ledger's pre-existing prefix
before and after the append and asserts it is byte-identical, on top of `append_ledger`'s own
refusal, as a direct regression guard on the issue's own "bit-identical" requirement.

Usage:
    buildingworld_stratified_split.py --dry_run     # plan and report, write nothing
    buildingworld_stratified_split.py               # plan, append to corpus_ledger.h5, write report
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.foundations import corpus_ledger  # noqa: E402
from scripts.foundations.buildingworld_spatial_keys import (  # noqa: E402
    OUT as CENTROIDS_H5, SLUG_TO_CITY,
)
from scripts.foundations.ingest_buildingworld import CITY_SLUG  # noqa: E402
from scripts.foundations.recover_program_labels_buildingworld import (  # noqa: E402
    BUILDINGWORLD_SOURCE_ID, OUT as PROGRAM_LABELS_H5,
)
from utils.frozen_corpus import assert_frozen_corpus, open_real_corpus  # noqa: E402

H5 = REPO / "data/real_massing_v1/real.h5"
REPORT_OUT = REPO / "execution/artifacts/buildingworld_stratified_split_177.json"

# #167/#174: the disclosed "BuildingWorld, region-conditioning granularity not yet decided (#171)"
# sentinel -- already registered in `stratified_split.SOURCE_NAMES` as "BW" and used as `real.h5`'s
# own `source_id` for these rows (`BUILDINGWORLD_SOURCE_ID`, imported above rather than
# re-declared, so this module and #176's own row selector cannot silently drift apart). Reused, not
# reinvented, as the ledger's `region` value too: #171 owns any finer BuildingWorld region scheme,
# and this ticket is not it.
LEDGER_REGION = -1

# #169, applied verbatim: exactly these four named cities, pooled -- not a general small-city rule.
POOLED_CITIES = frozenset({"Adelaide", "Greater Geelong", "Perth", "Philadelphia"})
POOLED_GROUP_NAME = "pooled_small_cities"

FAMILIES = ("flat", "gable", "hip", "complex")
SPLIT_FRAC = 0.02  # #153/#169's own shared convention; a starting point, not pinned as final.
GRID_CELL_M = 150.0


def group_of_city(city: str) -> str:
    """#169: the four named small cities collapse to one pooled bucket; every other ingested city
    is its own group."""
    return POOLED_GROUP_NAME if city in POOLED_CITIES else city


# --------------------------------------------------------------------------------------------
# pure planning logic (TDD seam: no I/O)
# --------------------------------------------------------------------------------------------

def tile_key(city: str, x_m: float, y_m: float, cell_m: float = GRID_CELL_M) -> str:
    """A fixed-size metre grid cell -- the atomic, never-split spatial held-out unit.

    `city`-namespaced (not just `(x, y)`): #169's pooled bucket grids four different cities'
    rows, in four different corrected local coordinate systems, together -- without the city
    prefix, Adelaide's and Perth's own (unrelated) cell `(5, 10)` would collide onto one key and
    silently merge two physically unrelated locations into one "tile."
    """
    return f"{city}:{int(np.floor(x_m / cell_m))}:{int(np.floor(y_m / cell_m))}"


def plan_group_split(row_ids: np.ndarray, city: np.ndarray, x_m: np.ndarray, y_m: np.ndarray,
                     family: np.ndarray, target_frac: float = SPLIT_FRAC,
                     cell_m: float = GRID_CELL_M, seed: int = 0) -> tuple:
    """Grid-tile + roof-family-stratified selection for ONE group (a standalone city, or #169's
    pooled bucket, in which case `city` varies row-to-row). Returns `(held_out: bool array
    aligned to row_ids, report: dict)`.

    Selection is a disclosed heuristic, not #153's exact single-axis subset-sum -- #153's own
    search targeted one absolute count; this ticket adds a second, family axis, and an exact joint
    optimum over two axes is a materially harder search #177's own issue text does not actually
    require ("consider... rather than trusting the bucket blindly" is advisory, not a request for
    optimality). Two rules govern which whole tiles get claimed, in this order of importance:

    1. TILES ARE OFFERED IN A SEEDED RANDOM ORDER, never sorted by size. An earlier version of
       this function tried "smallest tile first" (mirroring #153's `_region_tiles`, which orders
       NATURAL, already-coarse tiles -- 5 NL bboxes, 41 DE CityGML tiles -- to approach a target
       finely). Over this ticket's OWN fine-grained metre grid, "smallest first" instead means
       "sparsest, most isolated buildings first": measured against the production run, cities like
       San Francisco and Yarra ended up with a held-out slice at EXACTLY one building per tile,
       every time, regardless of the city's own average density. That is a scattered sample of
       buildings with no close neighbours -- structurally incapable of catching the dense-terrace-
       row leakage risk #177's own issue text names as the reason spatial blocking exists at all.
       A uniform random tile order instead draws a representative cross-section of the group's own
       density profile, so a dense cluster (where the leakage risk is real) has the same chance of
       being drawn -- and once drawn, is held out as one untouched block -- as a sparse one.
    2. FAMILIES CLAIM TILES BY LARGEST-DEFICIT-FIRST, not sequentially largest-target-first and
       not equal-turn round-robin either -- both were tried and both distort the result. Sequential
       (largest target first, breaking the whole selection once the OVERALL total target is
       reached) let an early, large-target family consume the entire budget before a small-target
       family (`hip`, typically) ever got a dedicated tile -- measured in production as e.g.
       Edmonton's `flat` share nearly halving between the city and its held-out slice. Equal-turn
       round-robin (one tile per ACTIVE family per round) fixed that but overcorrected: every
       family "spends" a tile's worth of the shared row budget at the same RATE per turn regardless
       of how much it actually needs, so a small-target family reaches its own (small) quota after
       only a few turns while a large-target family needs many more -- but by the time the large
       family has had enough turns, the overall total is already gone, having been partly spent on
       minority families' turns. Measured in production as the mirror-image distortion: Calgary's
       `gable` share (68% of the city) fell to 58% of its held-out slice. The fix used here instead
       claims, at every step, from whichever family currently has the LARGEST unmet need
       (`target - achieved so far`) among families with an eligible tile left -- giving each family
       roughly `target_f / mean_tile_size` turns, proportional to what it actually needs, rather
       than one turn per round regardless of size. A family drops out the moment its own target is
       met or its own eligible tiles run out (the same graceful degradation as before).

    Every tile's PLURALITY family (the family with the most rows in it; ties broken by `FAMILIES`'s
    own declaration order -- `numpy.argmax`'s documented first-index-wins rule, not left
    unspecified -- so `flat` wins a tie, deterministically) decides which family's turn can claim
    it; a family with no tiles of its own simply never gets a turn (graceful degradation, per
    #169's own "must degrade gracefully" requirement). A fill phase tops up to the overall target
    from whatever tiles remain, any family, once every family is either at its own target or has
    run out of eligible tiles. At least one tile is always selected for a non-empty group,
    mirroring #153's own `_closest_subset(min_items=1)` reasoning: representing a group with zero
    held-out rows defeats stratification's own point. A family whose own target rounds to 0 (a
    small-N artifact at a small `target_total`, not a tile-availability problem) gets no dedicated
    claims either -- the same small-sample limitation #153 already accepts for NL's coarse tiles,
    not something this function tries to engineer around.
    """
    n = len(row_ids)
    report: dict = dict(n=n, target_frac=target_frac, cell_m=cell_m)
    if n == 0:
        return np.zeros(0, dtype=bool), dict(report, n_tiles=0, held_out_n=0, held_out_frac=0.0,
                                             held_out_tiles=[], family_fractions={},
                                             held_out_family_fractions={},
                                             mean_tile_size_all=0.0, mean_tile_size_held_out=0.0)

    keys = np.array([tile_key(c, x, y, cell_m) for c, x, y in zip(city, x_m, y_m)])
    uniq_tiles, tile_of_row = np.unique(keys, return_inverse=True)
    n_tiles = len(uniq_tiles)
    tile_counts = np.bincount(tile_of_row, minlength=n_tiles)

    family_counts_all = {f: int((family == f).sum()) for f in FAMILIES}
    family_fractions = {f: family_counts_all[f] / n for f in FAMILIES}

    target_total = max(1, round(target_frac * n))
    target_family_counts = {f: round(target_total * family_fractions[f]) for f in FAMILIES}

    # per-tile per-family counts -> each tile's plurality family.
    per_family_tile_counts = {}
    for f in FAMILIES:
        per_family_tile_counts[f] = np.bincount(tile_of_row[family == f], minlength=n_tiles)
    stacked = np.stack([per_family_tile_counts[f] for f in FAMILIES])  # (4, n_tiles)
    plurality_idx = np.argmax(stacked, axis=0)
    tile_plurality_family = np.array([FAMILIES[i] for i in plurality_idx])

    rng = np.random.default_rng(seed)
    candidate_order = rng.permutation(n_tiles)  # unbiased draw -- see docstring point 1.
    per_family_order = {f: [ti for ti in candidate_order if tile_plurality_family[ti] == f]
                        for f in FAMILIES}
    next_idx = {f: 0 for f in FAMILIES}

    selected: set = set()
    running_total = 0
    running_family = {f: 0 for f in FAMILIES}

    def claim(ti: int) -> None:
        nonlocal running_total
        selected.add(ti)
        running_total += int(tile_counts[ti])
        for ff in FAMILIES:
            running_family[ff] += int(per_family_tile_counts[ff][ti])

    def _next_available(f: str):
        lst = per_family_order[f]
        while next_idx[f] < len(lst) and lst[next_idx[f]] in selected:
            next_idx[f] += 1
        return lst[next_idx[f]] if next_idx[f] < len(lst) else None

    # Greedy, deficit-proportional: at each step, whichever family has the LARGEST unmet need
    # (target - achieved so far) among families that still have an eligible tile claims its next
    # candidate tile -- see docstring point 2. An earlier version gave every active family exactly
    # one claim per round regardless of target size; that scaled each family's NUMBER OF TURNS
    # equally rather than to its own target, so a small-target family "spent" turns at the same
    # rate as a large one despite needing far fewer rows overall -- squeezing the dominant family
    # down (measured: Calgary's `gable` share fell from 68% of the city to 58% of its held-out
    # slice). Picking the largest current deficit each step instead gives a family roughly
    # `target_f / mean_tile_size` turns -- proportional to how much of the budget it actually
    # needs -- while still stopping a family the moment its own target is met or its own eligible
    # tiles run out (the same graceful degradation as before).
    while running_total < target_total:
        best_f, best_deficit, best_ti = None, 0, None
        for f in FAMILIES:
            deficit = target_family_counts[f] - running_family[f]
            if deficit <= 0:
                continue
            ti = _next_available(f)
            if ti is None:
                continue
            if deficit > best_deficit:
                best_f, best_deficit, best_ti = f, deficit, ti
        if best_ti is None:
            break
        claim(best_ti)
        next_idx[best_f] += 1

    # Fill phase: any remaining candidate, in the same random order, tops up to the total target.
    if running_total < target_total:
        for ti in candidate_order:
            if running_total >= target_total:
                break
            if ti in selected:
                continue
            claim(ti)

    held_out = np.isin(tile_of_row, np.fromiter(selected, dtype=int, count=len(selected)) if
                       selected else np.array([], dtype=int))
    held_out_n = int(held_out.sum())
    held_out_family_fractions = ({f: running_family[f] / held_out_n for f in FAMILIES}
                                 if held_out_n else {f: 0.0 for f in FAMILIES})

    report.update(
        n_tiles=n_tiles, target_total=target_total, held_out_n=held_out_n,
        held_out_frac=round(held_out_n / n, 4),
        held_out_tiles=sorted(uniq_tiles[i] for i in selected),
        family_fractions={f: round(v, 4) for f, v in family_fractions.items()},
        held_out_family_fractions={f: round(v, 4) for f, v in held_out_family_fractions.items()},
        # Diagnostic proving point 1 above actually holds: the held-out slice's own mean tile
        # size should track the group's overall mean tile size, not collapse toward 1.
        mean_tile_size_all=round(float(tile_counts.mean()), 3),
        mean_tile_size_held_out=(round(float(np.array([tile_counts[i] for i in selected]).mean()),
                                      3) if selected else 0.0),
    )
    return held_out, report


# --------------------------------------------------------------------------------------------
# I/O driver
# --------------------------------------------------------------------------------------------

def city_for_row(source_key: bytes) -> str:
    """`real.h5`'s `source_key` ('bw:<CitySlug>') -> the original city name (with spaces), the
    same name `ingest_buildingworld.CITY_SLUG`/`canonical_zip` use."""
    try:
        slug = source_key.decode().split(":", 1)[1]
        return SLUG_TO_CITY[slug]
    except (IndexError, KeyError) as e:
        raise ValueError(f"#177: unrecognized BuildingWorld source_key {source_key!r} -- expected "
                         f"'bw:<CitySlug>' for a slug in {sorted(SLUG_TO_CITY)}") from e


def load_inputs(h5_path: Path = H5, centroids_path: Path = CENTROIDS_H5,
                program_labels_path: Path = PROGRAM_LABELS_H5) -> dict:
    """Every BuildingWorld row's (city, x_m, y_m, family, height_m), and which rows are missing
    required inputs -- reported, never silently dropped or defaulted (see module docstring)."""
    with open_real_corpus(h5_path) as f:
        source_id = f["source_id"][:]
        source_key = f["source_key"][:]
        height_m = f["height_m"][:]
    rows = np.nonzero(source_id == BUILDINGWORLD_SOURCE_ID)[0].astype(np.int64)

    with h5py.File(centroids_path, "r") as f:
        n = int(f.attrs.get("committed_rows", f["row"].shape[0]))
        c_row, c_x, c_y, c_ok = f["row"][:n], f["x_m"][:n], f["y_m"][:n], f["ok"][:n]
    centroid = {int(r): (float(x), float(y)) for r, x, y, ok in zip(c_row, c_x, c_y, c_ok) if ok}

    with h5py.File(program_labels_path, "r") as f:
        n_p = int(f.attrs.get("committed_rows", f["row"].shape[0]))
        p_row, p_ok, p_family = f["row"][:n_p], f["ok"][:n_p], f["family"][:n_p]
    family = {int(r): fam.decode() for r, ok, fam in zip(p_row, p_ok, p_family) if ok}

    missing_centroid = [int(r) for r in rows if int(r) not in centroid]
    missing_family = [int(r) for r in rows if int(r) not in family]

    return dict(rows=rows, source_key=source_key, height_m=height_m, centroid=centroid,
               family=family, missing_centroid=missing_centroid, missing_family=missing_family)


def build_assignment(inputs: dict, target_frac: float = SPLIT_FRAC, cell_m: float = GRID_CELL_M,
                     seed: int = 0) -> tuple:
    """Every BuildingWorld row -> (row_ids, held_out, report). Raises if any row lacks a required
    input -- the issue's own "an unassigned row must be an error" requirement, enforced here rather
    than left to whichever downstream reader first notices a gap."""
    missing_c, missing_f = inputs["missing_centroid"], inputs["missing_family"]
    if missing_c or missing_f:
        raise ValueError(
            f"#177: {len(missing_c)} row(s) missing an ok=1 centroid, {len(missing_f)} row(s) "
            f"missing an ok=1 program label -- every BuildingWorld row needs both before a held-out "
            f"assignment can be made (first few missing centroid rows: {missing_c[:5]}; first few "
            f"missing family rows: {missing_f[:5]}). Re-run buildingworld_spatial_keys.py / "
            f"recover_program_labels_buildingworld.py to fill the gap; refusing to silently default "
            f"these rows to held_out=0.")

    rows = inputs["rows"]
    city = np.array([city_for_row(inputs["source_key"][r]) for r in rows])
    group = np.array([group_of_city(c) for c in city])
    x = np.array([inputs["centroid"][int(r)][0] for r in rows])
    y = np.array([inputs["centroid"][int(r)][1] for r in rows])
    fam = np.array([inputs["family"][int(r)] for r in rows])

    held_out = np.zeros(len(rows), dtype=bool)
    report: dict = {"groups": {}, "pooled_per_city": {}}
    for g in sorted(set(group.tolist())):
        mask = group == g
        # A per-group seed, not the same `seed` reused verbatim for every group: two groups that
        # happen to have an equal tile count would otherwise draw the IDENTICAL random permutation
        # (`rng.permutation(n_tiles)` depends only on the seed and n_tiles) -- unlikely across
        # cities of very different size, but not something to leave to chance when a deterministic
        # per-group derivation costs nothing. `hashlib` (not Python's own `hash()`, randomized
        # per-process for strings) keeps this reproducible across runs and machines.
        group_seed = int(hashlib.sha256(f"{seed}:{g}".encode()).hexdigest()[:8], 16)
        g_held, g_report = plan_group_split(rows[mask], city[mask], x[mask], y[mask], fam[mask],
                                            target_frac=target_frac, cell_m=cell_m,
                                            seed=group_seed)
        held_out[mask] = g_held
        report["groups"][g] = g_report
        if g == POOLED_GROUP_NAME:
            # #169's own "per-city breakdown reporting" requirement: diagnostic-only sub-scores
            # underneath the pooled bucket's combined numbers.
            for c in sorted(set(city[mask].tolist())):
                cmask = mask & (city == c)
                report["pooled_per_city"][c] = dict(
                    n=int(cmask.sum()), held_out_n=int(held_out[cmask].sum()),
                    held_out_frac=round(float(held_out[cmask].sum()) / max(int(cmask.sum()), 1), 4))

    assert held_out.shape[0] == len(rows)
    report["n_total"] = int(len(rows))
    report["held_out_n_total"] = int(held_out.sum())
    report["held_out_frac_total"] = round(float(held_out.sum()) / max(len(rows), 1), 4)
    return rows, held_out, report


def run(h5_path: Path = H5, centroids_path: Path = CENTROIDS_H5,
       program_labels_path: Path = PROGRAM_LABELS_H5, ledger_path: Path = corpus_ledger.LEDGER_PATH,
       target_frac: float = SPLIT_FRAC, cell_m: float = GRID_CELL_M, seed: int = 0,
       dry_run: bool = False, report_out: Path = REPORT_OUT) -> dict:
    t0 = time.time()
    with open_real_corpus(h5_path):
        pass  # #162: raises loudly before anything else if the frozen prefix has drifted.

    ledger_before = corpus_ledger.read_ledger(ledger_path) if Path(ledger_path).exists() else None

    inputs = load_inputs(h5_path, centroids_path, program_labels_path)
    rows, held_out, report = build_assignment(inputs, target_frac, cell_m, seed)
    report["elapsed_s"] = round(time.time() - t0, 1)

    if dry_run:
        print(json.dumps(report, indent=2, default=str))
        return report

    height_m = inputs["height_m"][rows]
    region = np.full(len(rows), LEDGER_REGION, dtype=np.int32)

    # `append_ledger` -> `write_ledger` replaces `ledger_path` wholesale (via its own atomic temp
    # file, so a CRASH mid-write cannot corrupt it) -- but a LOGIC bug in the concatenation itself
    # would still write successfully, just with wrong data. A pre-write backup, restored
    # automatically if the post-append prefix check below ever fires, closes that gap: this ticket
    # never leaves a caller with a silently-corrupted ledger and only an error message telling them
    # to go find a backup themselves.
    ledger_backup = None
    if Path(ledger_path).exists():
        ledger_backup = Path(ledger_path).with_suffix(Path(ledger_path).suffix + ".177-backup")
        shutil.copy2(ledger_path, ledger_backup)

    corpus_ledger.append_ledger(row=rows, region=region, held_out=held_out.astype(np.uint8),
                               height_m=height_m, path=ledger_path)

    if ledger_before is not None:
        ledger_after = corpus_ledger.read_ledger(ledger_path)
        n_before = len(ledger_before["row"])
        try:
            for col in corpus_ledger.COLUMNS:
                np.testing.assert_array_equal(
                    ledger_after[col][:n_before], ledger_before[col],
                    err_msg=f"#177: existing ledger column {col!r} changed by this append")
        except AssertionError:
            shutil.copy2(ledger_backup, ledger_path)
            raise RuntimeError(
                f"#177: the append corrupted the existing ledger prefix -- restored "
                f"{ledger_path} from the pre-append backup ({ledger_backup}), which is left in "
                f"place for inspection. No rows were durably appended.")

    if ledger_backup is not None:
        ledger_backup.unlink()

    Path(report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(report_out).write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    print(f"[#177] appended {len(rows)} rows to {ledger_path} ({report['held_out_n_total']} "
         f"held out, {report['held_out_frac_total']:.4f}) -> report at {report_out}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", default=str(H5))
    ap.add_argument("--centroids", default=str(CENTROIDS_H5))
    ap.add_argument("--program_labels", default=str(PROGRAM_LABELS_H5))
    ap.add_argument("--ledger", default=str(corpus_ledger.LEDGER_PATH))
    ap.add_argument("--target_frac", type=float, default=SPLIT_FRAC)
    ap.add_argument("--cell_m", type=float, default=GRID_CELL_M)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    run(Path(args.h5), Path(args.centroids), Path(args.program_labels), Path(args.ledger),
       args.target_frac, args.cell_m, args.seed, args.dry_run)
