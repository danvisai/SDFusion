"""End-to-end + geometry checks for town_generate_service (map #97, tickets #99/#100).

Two layers, because they cost wildly different amounts:

  --geometry-only   the pure footprint->grid math (no model, no GPU, instant)
  default           the above plus real A2/DoraCodec generation through TestClient (~10s/building)

The geometry layer exists because this service's whole risk surface is frame conventions: the
editor's polygon is world-meter (x, z), the corpus grid is [z, y, x], and #98 already shipped one
coordinate-mirroring bug in this demo's extrusion. `check_orientation` is the sharp version of that
test -- it does not merely assert "the mesh is roughly the right size", it asserts the generated
massing overlaps the drawn polygon STRICTLY BETTER than it overlaps that polygon mirrored, which is
the failure a size-only assertion sails straight past.

Run:
  ./venv/bin/python scripts/server/test_town_generate.py --geometry-only
  PYTHONPATH=. ./venv/bin/python scripts/server/test_town_generate.py
"""

from __future__ import annotations

import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.server.town_generate_service import (            # noqa: E402
    MARGIN, MAX_BUILDINGS, RES, _footprint_normalization, _height_voxel_range,
    _rasterize_footprint, app,
)

# A plain rect and an L -- the L is deliberately asymmetric in BOTH x and z, so any axis swap or
# mirror moves its notch into a different quadrant and the orientation check fails loudly.
RECT = [[-9.0, -6.0], [9.0, -6.0], [9.0, 6.0], [-9.0, 6.0]]
LSHAPE = [[0.0, 0.0], [20.0, 0.0], [20.0, 8.0], [12.0, 8.0], [12.0, 18.0], [0.0, 18.0]]
# Same L, translated well off the origin: catches a service that silently recenters output at 0,0.
LSHAPE_FAR = [[x + 55.0, z - 40.0] for x, z in LSHAPE]

CELL = 0.5          # metres per cell for the XZ overlap rasterisation below
# NOT a footprint-fidelity gate. CONTEXT.md is explicit that footprint-IoU as a lone number is to be
# avoided -- fidelity is the fringe/spill/uncovered split, and judging it lives in the eval harness,
# not here. This threshold answers a different and much coarser question: did the mesh come back in
# the right PLACE and ORIENTATION, or did a frame convention mirror/translate it? Measured ~0.84-0.94
# on rect/L at strength 0.5, with slack for a moving checkpoint and for the codec's stochastic
# surface sampling (DoraCodec seeds one RNG at construction, so repeat encodes differ).
MIN_PLACEMENT_OVERLAP = 0.75


# ------------------------------------------------------------------------------------------------
# layer 1: pure geometry, no model
# ------------------------------------------------------------------------------------------------
def check_geometry():
    # The service takes its margin from building_to_sdf's own default rather than restating it.
    # Pin that here: if the corpus function's default ever moves, this says so out loud instead of
    # letting generated buildings quietly land in a frame the model was not trained in.
    from scripts.ingest_3dbag import building_to_sdf
    corpus_margin = inspect.signature(building_to_sdf).parameters["margin"].default
    assert MARGIN == corpus_margin, f"service MARGIN {MARGIN} != corpus margin {corpus_margin}"
    print(f"[geom] margin pinned to building_to_sdf's own default ({MARGIN})  OK")

    # normalisation: the frame is keyed on the LARGER of footprint extent and height, per
    # ingest_3dbag.building_to_sdf, so a tall thin tower normalises on its height, not its plan.
    pts = np.asarray(RECT)
    s, cx, cz = _footprint_normalization(pts, 15.0)
    assert abs(cx) < 1e-9 and abs(cz) < 1e-9, f"centred rect should centre at origin, got {cx},{cz}"
    assert abs(s - 18.0 / 2 * 1.05) < 1e-9, f"18m span dominates 15m height, got s={s}"
    s_tall, _, _ = _footprint_normalization(pts, 40.0)
    assert abs(s_tall - 40.0 / 2 * 1.05) < 1e-9, f"40m height should dominate, got s={s_tall}"
    print(f"[geom] normalisation  s(rect,h=15)={s:.4f}  s(rect,h=40)={s_tall:.4f}  OK")

    # rasterisation: the mask is [D, W] = [z, x]. Feed the L and check the notch lands in the
    # quadrant the drawing put it in -- high x AND high z is the EMPTY corner of this L.
    lp = np.asarray(LSHAPE)
    s_l, cx_l, cz_l = _footprint_normalization(lp, 24.0)
    mask = _rasterize_footprint(lp, s_l, cx_l, cz_l)
    assert mask.shape == (RES, RES), f"mask shape {mask.shape}"
    assert mask.any(), "L footprint rasterised to nothing"
    half = RES // 2
    lo_z_lo_x = mask[:half, :half].sum()     # drawn-solid corner (x<10, z<9)
    hi_z_hi_x = mask[half:, half:].sum()     # drawn-empty corner (x>10, z>9) -- the L's notch
    assert hi_z_hi_x < 0.25 * lo_z_lo_x, (
        f"the L's notch is not where it was drawn: solid corner={lo_z_lo_x}, "
        f"notch corner={hi_z_hi_x} (axis swap or mirror in _rasterize_footprint?)")
    print(f"[geom] rasterisation  solid corner={lo_z_lo_x} notch corner={hi_z_hi_x}  OK")

    # vertical extent: symmetric about the grid centre (building_to_sdf centres the full bbox),
    # and monotone in height for a fixed plan.
    y0, y1 = _height_voxel_range(15.0, s)
    assert y0 + y1 == RES - 1, f"vertical slab should straddle the centre, got ({y0},{y1})"
    y0b, y1b = _height_voxel_range(8.0, s)
    assert (y1b - y0b) < (y1 - y0), "a shorter building should occupy fewer voxels"
    # degenerate heights must still yield a usable (non-empty) slab rather than an inverted one
    y0c, y1c = _height_voxel_range(0.01, s)
    assert y1c > y0c, f"near-zero height collapsed the slab to ({y0c},{y1c})"
    print(f"[geom] vertical range h=15 -> ({y0},{y1})  h=8 -> ({y0b},{y1b})  h=0.01 -> ({y0c},{y1c})  OK")


# ------------------------------------------------------------------------------------------------
# layer 2: model-backed HTTP behaviour
# ------------------------------------------------------------------------------------------------
def _xz_occupancy(points_xz: np.ndarray, lo: np.ndarray, shape: tuple) -> np.ndarray:
    """Bin world-metre (x, z) samples into a common CELL-metre grid anchored at `lo`."""
    idx = np.floor((points_xz - lo) / CELL).astype(int)
    ok = np.all((idx >= 0) & (idx < np.array(shape)[::-1]), axis=1)
    occ = np.zeros(shape, bool)
    occ[idx[ok, 1], idx[ok, 0]] = True
    return occ


def _polygon_occupancy(poly: np.ndarray, lo: np.ndarray, shape: tuple) -> np.ndarray:
    from skimage.draw import polygon2mask
    rc = np.stack([(poly[:, 1] - lo[1]) / CELL, (poly[:, 0] - lo[0]) / CELL], axis=1)
    return polygon2mask(shape, rc)


def _poly_poly_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two world-metre polygons, used to detect when a mirror IS the shape (see below)."""
    lo = np.minimum(a.min(axis=0), b.min(axis=0)) - 2.0
    hi = np.maximum(a.max(axis=0), b.max(axis=0)) + 2.0
    shape = (int(np.ceil((hi[1] - lo[1]) / CELL)), int(np.ceil((hi[0] - lo[0]) / CELL)))
    ma, mb = _polygon_occupancy(a, lo, shape), _polygon_occupancy(b, lo, shape)
    u = (ma | mb).sum()
    return float((ma & mb).sum() / u) if u else 0.0


def _placement_overlap(verts: list, poly: np.ndarray) -> float:
    """Overlap between the generated mesh's XZ shadow and the drawn polygon, on a shared grid.

    A closed massing mesh has a meshed roof and floor, so binning its vertices fills the plan area
    rather than tracing only its outline -- accurate enough to separate "right shape, right place"
    from "mirrored" or "translated", which is all this is asked to do. See MIN_PLACEMENT_OVERLAP:
    this is a placement check, not the footprint-fidelity measure, which is a three-way split.
    """
    v = np.asarray(verts, dtype=np.float64)
    vxz = v[:, [0, 2]]
    lo = np.minimum(vxz.min(axis=0), poly.min(axis=0)) - 2.0
    hi = np.maximum(vxz.max(axis=0), poly.max(axis=0)) + 2.0
    shape = (int(np.ceil((hi[1] - lo[1]) / CELL)), int(np.ceil((hi[0] - lo[0]) / CELL)))
    a = _xz_occupancy(vxz, lo, shape)
    # fill the mesh shadow's interior: marching-cubes vertices are dense at CELL=0.5m, but close
    # any single-cell pinholes so the IoU measures shape agreement, not sampling noise
    from scipy import ndimage
    a = ndimage.binary_closing(a, np.ones((3, 3)))
    a = ndimage.binary_fill_holes(a)
    b = _polygon_occupancy(poly, lo, shape)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def check_orientation(client, poly_list, height, label):
    """The mesh must match the drawn polygon better than its mirrors. This is the #98 bug class."""
    poly = np.asarray(poly_list, dtype=np.float64)
    r = client.post("/generate_building", json={"points": poly_list, "height": height, "seed": 0})
    assert r.status_code == 200, f"{label}: {r.status_code} {r.text[:300]}"
    body = r.json()
    v = np.asarray(body["vertices"], dtype=np.float64)

    iou = _placement_overlap(body["vertices"], poly)
    c = poly.mean(axis=0)
    mirror_z = poly.copy(); mirror_z[:, 1] = 2 * c[1] - mirror_z[:, 1]
    mirror_x = poly.copy(); mirror_x[:, 0] = 2 * c[0] - mirror_x[:, 0]
    swapped = poly[:, ::-1].copy()

    # A transform that maps the drawn shape onto ITSELF cannot witness a mirroring bug -- a
    # rectangle is its own z-mirror, so demanding a strictly better score against it is a test
    # that fails on correct code. Skip those, and say which were skipped, so the report never
    # reads as if this shape had ruled out a mirror it is physically incapable of ruling out.
    parts, skipped = [], []
    for name, other_poly in (("z-mirror", mirror_z), ("x-mirror", mirror_x), ("axis-swap", swapped)):
        if _poly_poly_iou(poly, other_poly) > 0.98:
            skipped.append(name)
            continue
        other = _placement_overlap(body["vertices"], other_poly)
        parts.append(f"{name}={other:.3f}")
        assert iou > other, (f"{label}: matches its {name} ({other:.3f}) at least as well as the "
                             f"drawing ({iou:.3f})")

    note = f"  [symmetric under {', '.join(skipped)}]" if skipped else ""
    print(f"[orient] {label:12s} IoU drawn={iou:.3f}  " + "  ".join(parts)
          + f"  vs_input={body['vs_input']:.4f}{note}")
    assert iou >= MIN_PLACEMENT_OVERLAP, (f"{label}: placement overlap {iou:.3f} < "
                                          f"{MIN_PLACEMENT_OVERLAP} -- the massing is not where it was drawn")

    # the building must stand ON the ground, spanning [0, height] -- not centred on it
    y0, y1 = v[:, 1].min(), v[:, 1].max()
    tol = 0.06 * height + 0.5
    assert abs(y0) < tol, f"{label}: base at y={y0:.2f}, expected ~0"
    assert abs(y1 - height) < tol, f"{label}: top at y={y1:.2f}, expected ~{height}"
    return body


def check_single(client):
    check_orientation(client, RECT, 15.0, "rect")
    check_orientation(client, LSHAPE, 24.0, "L-shape")
    check_orientation(client, LSHAPE_FAR, 24.0, "L off-origin")

    for bad, why in (({"points": [[0, 0], [1, 1]], "height": 10.0}, "2-point footprint"),
                     ({"points": RECT, "height": 0.0}, "zero height")):
        r = client.post("/generate_building", json=bad)
        assert r.status_code == 400, f"{why} should be rejected with 400, got {r.status_code}"
    print("[single] degenerate inputs rejected with 400  OK")


def _stream_town(client, payload):
    """POST /generate_town and collect its NDJSON records."""
    with client.stream("POST", "/generate_town", json=payload) as r:
        assert r.status_code == 200, f"/generate_town {r.status_code} {r.read()[:300]}"
        return [json.loads(line) for line in r.iter_lines() if line.strip()]


def check_town(client):
    """One call, N buildings, streamed -- each landing at its own drawn position."""
    far = [[x + 90.0, z + 30.0] for x, z in RECT]
    payload = {"buildings": [{"points": RECT, "height": 15.0},
                             {"points": far},                       # height omitted -> default
                             {"points": [[0, 0], [1, 1]]},          # degenerate -> per-item error
                             {"points": RECT, "height": 15.0}],     # a twin of building 0
               "default_height": 21.0, "seed": 0}
    t0 = time.time()
    recs = _stream_town(client, payload)
    dt = time.time() - t0

    kinds = [x["kind"] for x in recs]
    assert kinds[-1] == "done", f"stream must end with a done record, got {kinds}"
    done = recs[-1]
    built = {x["index"]: x for x in recs if x["kind"] == "building"}
    errs = {x["index"]: x for x in recs if x["kind"] == "error"}
    assert set(built) == {0, 1, 3}, f"expected buildings 0, 1 and 3, got {sorted(built)}"
    assert set(errs) == {2}, f"expected building 2 to fail, got {sorted(errs)}"
    assert done["ok"] == 3 and done["failed"] == 1, f"bad done record: {done}"
    print(f"[town] {len(built)} built, {len(errs)} failed in {dt:.1f}s "
          f"(server {done['total_seconds']:.1f}s)  OK")

    # each building lands where it was drawn, and the default height applied to the one that
    # omitted its own -- a batch that recentred or reordered its output fails both of these
    assert _placement_overlap(built[0]["vertices"], np.asarray(RECT)) >= MIN_PLACEMENT_OVERLAP, "town[0] misplaced"
    assert _placement_overlap(built[1]["vertices"], np.asarray(far)) >= MIN_PLACEMENT_OVERLAP, "town[1] misplaced"
    top = max(p[1] for p in built[1]["vertices"])
    assert abs(top - 21.0) < 2.0, f"default_height not applied: top at {top:.1f}, expected ~21"
    print(f"[town] positions and default height OK (b1 top={top:.1f}m)")

    # buildings 0 and 3 are the same footprint at the same height in the same request. The town's
    # seed is decorrelated per building (seed * 1000003 + index, as eval_massing_arms.py does), so
    # they must NOT come out identical -- one noise draw shared across a town reads as repetition
    # once there are more than a few buildings in view.
    a, b = np.asarray(built[0]["vertices"]), np.asarray(built[3]["vertices"])
    same = a.shape == b.shape and np.allclose(a, b)
    assert not same, "identical twins came out identical: the per-building seed is not decorrelated"
    print(f"[town] twin footprints differ under per-building seeds "
          f"({a.shape[0]} vs {b.shape[0]} verts)  OK")

    # generation is per-building, so its cost is linear -- the UI's progress state depends on
    # that staying true, and on the endpoint reporting the same per-building number it measures
    per = [built[i]["gen_seconds"] for i in sorted(built)]
    print(f"[town] per-building gen_seconds={[f'{p:.2f}' for p in per]}  "
          f"total={done['total_seconds']:.1f}s")

    r = client.post("/generate_town", json={"buildings": []})
    assert r.status_code == 400, f"empty town should 400, got {r.status_code}"
    r = client.post("/generate_town",
                    json={"buildings": [{"points": RECT, "height": 9.0}] * (MAX_BUILDINGS + 1)})
    assert r.status_code == 400, f"over-limit town should 400, got {r.status_code}"
    print(f"[town] empty and over-limit ({MAX_BUILDINGS}) towns rejected with 400  OK")


def main():
    geometry_only = "--geometry-only" in sys.argv
    check_geometry()
    if geometry_only:
        print("\n[skip] model-backed checks (--geometry-only)")
        return
    from fastapi.testclient import TestClient
    t0 = time.time()
    with TestClient(app) as client:
        print(f"[boot] models loaded in {time.time()-t0:.1f}s  health={client.get('/health').json()}")
        check_single(client)
        check_town(client)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
