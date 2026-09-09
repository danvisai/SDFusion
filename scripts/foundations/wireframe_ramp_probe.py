"""Wireframe-guided Ramp parameter extraction -- steps 1-3 of a 3-step precision check
(step 4, carving onto the extruded solid, and step 5, IoU validation, are NOT built yet).

Motivation: the height-map/SDF-fit route to `Ramp` parameters (#10's fitter) is imprecise
specifically in sloped-roof regions -- pitch/ridge/azimuth get smeared across a 64^3 voxel
grid. BuildingWorld ships a SEPARATE per-building wireframe representation
(`data/buildingworld_wireframe/<city>/wireframe/wireframe.zip`) that #156's own out-of-scope
note dismissed as "redundant with the mesh format" -- true for full-solid reconstruction, but
this file's own inspection of one building's wireframe found it is a pure edge list (zero
faces, only `l` elements): 97 vertical wall-corner edges, 133 horizontal eave/ground edges,
89 sloped edges -- i.e. an explicit, full-precision ridge/hip/rake skeleton the flat mesh
doesn't expose directly. That's the signal this script tries to recover.

Naming correspondence: mesh `obj/<prefix>_<i>.obj` and wireframe `wireframe/<prefix>_<j:05d>.obj`
share the same PREFIX group (a "lot", generally several adjacent buildings) and the SAME
coordinate frame, but `i` and `j` are NOT the same index -- checked and found false. Wireframe
generation drops ~22% of buildings (Adelaide: 5868 mesh files, only 4580 wireframes) and
renumbers what's left sequentially, so `j` drifts from `i` by however many earlier buildings in
that prefix were dropped. `wireframe/10_00010.obj`'s true partner is `obj/10_13.obj` (extent
match `[23.33, 9.85, 6.05]` exact, centroid distance 2.9 units), not `obj/10_10.obj` as same-
index pairing would assume. `match_wireframes_to_meshes` below re-derives the true pairing
geometrically (nearest centroid among same-prefix candidates) instead of trusting the index.

Pipeline (steps 1-3 only):
  1. classify_edges    -- wall / horizontal (eave, ridge, ground) / sloped (rake, hip)
                           via ANGLE thresholds (scale-invariant, not an absolute distance
                           epsilon -- the wrong thing to hardcode across cities/units).
  2. cluster_planes     -- group sloped edges into roof planes via greedy RANSAC plane
                           segmentation (no face data exists to do this topologically).
  3. plane_to_ramp      -- each recovered plane's pitch/azimuth, in the SAME Frame-N space
                           (`building_to_sdf`'s own center/scale/axis-swap) #10's fitter and
                           the SDF grid already use, so the numbers are directly comparable.

NOT built here: applying these parameters as an actual `Ramp` op onto the extruded
footprint (step 4), or measuring whether this improves 3D IoU/`extra` over the SDF-fit route
(step 5) -- both need a decision on scope first (see the conversation this came out of).

Run: env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
     scripts/foundations/wireframe_ramp_probe.py --city Adelaide --n 8
"""
from __future__ import annotations

import argparse
import io
import math
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[2]
MESH_DIR = REPO / "data" / "buildingworld_mesh"
WIRE_DIR = REPO / "data" / "buildingworld_wireframe"

# Angle thresholds (degrees) for edge classification -- scale-invariant, not a distance cutoff.
HORIZONTAL_MAX_DEG = 5.0   # edge direction within this angle of the xy-plane -> horizontal
VERTICAL_MAX_DEG = 5.0     # edge direction within this angle of the z-axis -> vertical
PLANE_DIST_TOL_FRAC = 0.01  # planarity tolerance, as a fraction of the building's own extent
# NOTE: a raw minimum-support-edge-count threshold was tried and dropped -- see
# cluster_planes()'s docstring. Acceptance is gated on _forms_closed_loop instead.


# ---- parsing -----------------------------------------------------------------------------

def load_obj_verts_edges(data: str) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Wireframe .obj -> (vertices (N,3), edges as 0-based index pairs). No 'f' lines expected."""
    verts: list[list[float]] = []
    edges: list[tuple[int, int]] = []
    for line in data.splitlines():
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("l "):
            a, b = line.split()[1:3]
            edges.append((int(a) - 1, int(b) - 1))
    return np.asarray(verts, dtype=np.float64), edges


def _obj_bounds(data: bytes | str) -> tuple[np.ndarray, np.ndarray]:
    """Fast (centroid, extent) from an .obj's 'v' lines alone -- works for both a full mesh
    and a wireframe file identically, and skips face/line parsing entirely since bounds are
    all the geometric matcher below needs."""
    if isinstance(data, bytes):
        data = data.decode()
    pts = [
        [float(x) for x in line.split()[1:4]] for line in data.splitlines() if line.startswith("v ")
    ]
    V = np.asarray(pts, dtype=np.float64)
    return V.mean(0), V.max(0) - V.min(0)


MATCH_MIN_MESH_EXTENT = 0.5  # meters-ish; a mesh smaller than this is a degenerate/corrupt entry
MATCH_MAX_CONFIDENT_DIST_FRAC = 1.5  # accept only if centroid_dist <= this * wireframe's own extent
MATCH_MAX_EXTENT_RATIO = 2.5  # accept only if mesh and wireframe extents are within this factor


def match_wireframes_to_meshes(
    zm: zipfile.ZipFile, zw: zipfile.ZipFile, mesh_names: list[str], wire_names: list[str]
) -> dict[str, str]:
    """True wireframe->mesh pairing within one prefix group, by GLOBALLY OPTIMAL nearest-
    centroid assignment (Hungarian algorithm, `scipy.optimize.linear_sum_assignment`) -- NOT
    filename index (see the module docstring: that assumption is false).

    A first version of this used GREEDY nearest-centroid assignment (each wireframe claims its
    best remaining mesh, in input order) and it was not good enough: on Adelaide's "1_" prefix
    group (25 mesh candidates, 14 wireframes, extents up to 215m -- a wide, high-variance
    group), greedy assignment produced centroid distances of 18-85 units and matched one
    wireframe (extent [72.8, 69.8, 27.0]) to a near-degenerate mesh file (extent
    [0.09, 0.0, 9.34] -- essentially a zero-area line, not a building) purely because a better
    candidate had already been claimed by an earlier wireframe in iteration order. Order-
    dependent greedy assignment doesn't reliably avoid that; an actual optimal assignment does.

    Degenerate mesh entries (extent below `MATCH_MIN_MESH_EXTENT`) are excluded from the
    candidate pool entirely before assignment. After the optimal assignment, a match is kept
    only if BOTH (a) its centroid distance is within `MATCH_MAX_CONFIDENT_DIST_FRAC` of the
    wireframe's own extent, AND (b) the mesh's and wireframe's extents are within
    `MATCH_MAX_EXTENT_RATIO` of each other on every axis -- (a) alone passed several bad
    matches (a mesh with extent [3.09, 0.94, 9.62] accepted for a wireframe with extent
    [111.25, 43.8, 22.3], an 11x size mismatch, because the centroid distance alone happened
    to be small relative to the wireframe's own large extent). A wireframe is a simplified
    SUBSET of its true mesh, so their extents should be genuinely comparable, not off by an
    order of magnitude. Otherwise the match is dropped rather than forced -- a bad-but-least-
    bad match is still wrong, and worse than admitting no confident pairing exists.
    """
    from scipy.optimize import linear_sum_assignment

    mesh_info = {n: _obj_bounds(zm.read(n)) for n in mesh_names}
    wire_info = {n: _obj_bounds(zw.read(n)) for n in wire_names}
    mesh_names = [n for n in mesh_names if float(mesh_info[n][1].max()) >= MATCH_MIN_MESH_EXTENT]
    if not mesh_names or not wire_names:
        return {}

    cost = np.zeros((len(wire_names), len(mesh_names)))
    for i, wname in enumerate(wire_names):
        wc, _we = wire_info[wname]
        for j, mname in enumerate(mesh_names):
            mc, _me = mesh_info[mname]
            cost[i, j] = np.linalg.norm(mc - wc)

    row_idx, col_idx = linear_sum_assignment(cost)
    pairing: dict[str, str] = {}
    for i, j in zip(row_idx, col_idx):
        wname, mname = wire_names[i], mesh_names[j]
        wc, we = wire_info[wname]
        _mc, me = mesh_info[mname]
        dist = float(cost[i, j])
        dist_ok = dist <= MATCH_MAX_CONFIDENT_DIST_FRAC * float(we.max())
        ratio = np.maximum(me, 1e-6) / np.maximum(we, 1e-6)
        extent_ok = bool(np.all((ratio >= 1 / MATCH_MAX_EXTENT_RATIO) & (ratio <= MATCH_MAX_EXTENT_RATIO)))
        if dist_ok and extent_ok:
            pairing[wname] = mname
    return pairing


# ---- step 1: classify edges --------------------------------------------------------------

def classify_edges(V: np.ndarray, edges: list[tuple[int, int]]) -> dict[str, list[tuple[int, int]]]:
    """Each edge -> 'vertical' / 'horizontal' / 'sloped', by the angle its direction makes
    with the z-axis. Angle-based (not an absolute-distance epsilon) so it doesn't silently
    break on a city in different raw units (#165's whole finding: units vary by source)."""
    horiz_max = math.radians(HORIZONTAL_MAX_DEG)
    vert_max = math.radians(VERTICAL_MAX_DEG)
    out: dict[str, list[tuple[int, int]]] = {"vertical": [], "horizontal": [], "sloped": []}
    for a, b in edges:
        d = V[b] - V[a]
        length = float(np.linalg.norm(d))
        if length < 1e-9:
            continue
        dz = abs(d[2]) / length
        dxy = math.hypot(d[0], d[1]) / length
        angle_from_horizontal = math.asin(min(1.0, dz))   # 0 = flat, pi/2 = vertical
        angle_from_vertical = math.asin(min(1.0, dxy))
        if angle_from_horizontal <= horiz_max:
            out["horizontal"].append((a, b))
        elif angle_from_vertical <= vert_max:
            out["vertical"].append((a, b))
        else:
            out["sloped"].append((a, b))
    return out


# ---- step 2: cluster sloped edges into roof planes ----------------------------------------

@dataclass
class RoofPlane:
    normal: np.ndarray          # unit normal, world/local (pre-Frame-N) coordinates
    d: float                    # plane offset: normal . x = d
    edges: list[tuple[int, int]] = field(default_factory=list)
    point_ids: set[int] = field(default_factory=set)


def _largest_connected_component(edge_list: list[tuple[int, int]]) -> list[int]:
    """Indices (into `edge_list`) of its largest connected component, via union-find on the
    vertex ids the edges touch. A real roof plane's boundary is one connected loop (rakes/hips
    meeting eaves/ridges); a purely coincidental "this edge happens to be coplanar too" match
    from an unrelated part of the building will usually NOT be connected to the rest of the
    support set at all -- catching that is a stronger check than raw support count, which
    can't tell a genuinely connected boundary from a handful of scattered coplanar coincidences."""
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in edge_list:
        union(a, b)

    groups: dict[int, list[int]] = {}
    for i, (a, _b) in enumerate(edge_list):
        groups.setdefault(find(a), []).append(i)
    return max(groups.values(), key=len)


MAX_CLOSING_HOPS = 3  # an eave/ridge can be several short segments, not always one edge


def _forms_closed_loop(
    support_edges: list[tuple[int, int]], horizontal_edges: list[tuple[int, int]]
) -> bool:
    """True if the support edges' LOOSE ENDS (degree-1 vertices within the support subgraph
    -- e.g. the two rake bases of a simple gable) are bridged by a short chain of HORIZONTAL
    edges (an eave, or a ridge closing two hip tops) within `MAX_CLOSING_HOPS` hops.

    First version of this checked only for a single DIRECT horizontal edge between the loose
    ends and rejected a real plane on Adelaide `10_10` as a result: its eave is TWO segments
    (a corner vertex in between), not one, so no direct edge existed even though the boundary
    genuinely closes. Bounded-hop BFS catches that. The bound matters: unbounded reachability
    through the horizontal-edge graph would trivially "close" via the building's ENTIRE
    perimeter regardless of which specific plane is being tested, which isn't evidence of
    anything -- a real eave directly under one roof plane is a short, local path, not a detour
    around the whole building.
    """
    if not support_edges:
        return False
    deg: dict[int, int] = {}
    for a, b in support_edges:
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    loose = [v for v, d in deg.items() if d == 1]
    if len(loose) < 2:
        return True  # sloped edges alone already close (rare, but consistent)

    adj: dict[int, list[int]] = {}
    for a, b in horizontal_edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    def reachable_within(start: int, target: int, max_hops: int) -> bool:
        frontier = {start}
        seen = {start}
        for _ in range(max_hops):
            nxt = set()
            for u in frontier:
                for v in adj.get(u, []):
                    if v == target:
                        return True
                    if v not in seen:
                        nxt.add(v)
                        seen.add(v)
            frontier = nxt
            if not frontier:
                break
        return False

    return any(
        reachable_within(loose[i], loose[j], MAX_CLOSING_HOPS)
        for i in range(len(loose))
        for j in range(i + 1, len(loose))
    )


def _fit_plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares plane through >=3 points via SVD. Returns (unit normal, offset)."""
    c = points.mean(0)
    _, _, vt = np.linalg.svd(points - c)
    n = vt[-1]
    n = n / (np.linalg.norm(n) + 1e-12)
    return n, float(n @ c)


def cluster_planes(
    V: np.ndarray,
    sloped_edges: list[tuple[int, int]],
    tol: float,
    horizontal_edges: list[tuple[int, int]] | None = None,
) -> list[RoofPlane]:
    """Greedy RANSAC plane segmentation. No face data exists to do this topologically (the
    wireframe carries no polygon boundaries), so this is inherently approximate: candidate
    planes come from pairs of sloped edges that share an endpoint (the common real-world case
    -- two rakes meeting at a ridge point, or a rake and a hip sharing an eave corner), scored
    by how many OTHER sloped-edge endpoints lie within `tol` of the fitted plane.

    Seeding is restricted to DEGREE-2 vertices -- points where exactly two sloped edges meet,
    an unambiguous single ridge/hip junction. Checked against a real building (Adelaide 10_0):
    its 93 sloped edges touch 127 vertices, 84 at degree 1 (a rake edge ending at an eave/wall
    corner -- not useful as a plane seed alone) and 34 at degree 2 (a real, unambiguous
    junction), but 9 at degree >=3, one as high as degree 7 -- far more edges than any real
    hip/valley apex has, almost certainly near-coincident points from unrelated facets rather
    than one true architectural corner. Seeding from those hubs generates C(k,2) spurious
    candidate pairs each (21 from the degree-7 point alone) that mix edges from different
    planes. Hub-touching edges can still join an already-found plane as support; they just
    never originate one.

    ACCEPTANCE is gated on `_forms_closed_loop`, not a raw support-edge count: a plain support
    threshold turned out to have no value that worked for both simple and complex buildings
    (support>=2 accepted noise on complex ones; support>=3+ rejected genuinely real, simple
    2-rake gable planes, whose boundary closes via a horizontal eave edge that a sloped-only
    count can never see). A connected-component filter alone was also insufficient: a bare
    2-edge seed is trivially "connected" to itself, so it only starts doing real work once
    there's a third candidate edge to test. Loop closure (see `_forms_closed_loop`) is the
    first version of this that correctly keeps `10_14`'s real 2-plane gable while rejecting
    most of `10_0`'s and `10_1`'s scattered coplanar noise -- verified empirically, not assumed.
    """
    horizontal_edges = horizontal_edges or []
    remaining = list(sloped_edges)
    planes: list[RoofPlane] = []

    while remaining:
        # index remaining edges by endpoint for fast "shares a vertex" lookup
        by_point: dict[int, list[int]] = {}
        for i, (a, b) in enumerate(remaining):
            by_point.setdefault(a, []).append(i)
            by_point.setdefault(b, []).append(i)

        best = None  # (support_count, normal, d, support_edge_idxs)
        seen_pairs = set()
        for pid, edge_idxs in by_point.items():
            if len(edge_idxs) != 2:  # exclude hub vertices (>=3) from SEEDING -- see docstring
                continue
            for i in range(len(edge_idxs)):
                for j in range(i + 1, len(edge_idxs)):
                    ei, ej = edge_idxs[i], edge_idxs[j]
                    key = (min(ei, ej), max(ei, ej))
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    a1, b1 = remaining[ei]
                    a2, b2 = remaining[ej]
                    pts = np.stack([V[a1], V[b1], V[a2], V[b2]])
                    pts = np.unique(pts, axis=0)
                    if len(pts) < 3:
                        continue
                    try:
                        n, d = _fit_plane(pts)
                    except np.linalg.LinAlgError:
                        continue
                    support = [
                        k for k, (a, b) in enumerate(remaining)
                        if abs(n @ V[a] - d) < tol and abs(n @ V[b] - d) < tol
                    ]
                    # Keep only the support that's actually CONNECTED to the seed pair (ei,
                    # ej) -- a scattered coincidentally-coplanar edge elsewhere on the
                    # building doesn't make this a real boundary, and would otherwise inflate
                    # `support`'s count without being part of the same physical plane.
                    support_edges_for_cc = [remaining[k] for k in support]
                    cc = _largest_connected_component(support_edges_for_cc)
                    cc_global_idx = {support[i] for i in cc}
                    if ei not in cc_global_idx or ej not in cc_global_idx:
                        continue  # the seed pair itself didn't survive -- not a real hit
                    support = sorted(cc_global_idx)
                    support_edges = [remaining[k] for k in support]
                    if not _forms_closed_loop(support_edges, horizontal_edges):
                        continue  # dangling chain, not a real boundary -- reject regardless of size
                    if best is None or len(support) > len(best[0]):
                        best = (support, n, d)
        if best is None:
            break  # no more real (loop-closing) planes findable -- leave the rest unclustered

        support, n, d = best
        support_edges = [remaining[k] for k in support]
        pts = np.unique(
            np.concatenate([[V[a], V[b]] for a, b in support_edges]), axis=0
        )
        n, d = _fit_plane(pts)  # refit on the full accepted support for a cleaner normal
        planes.append(RoofPlane(normal=n, d=d, edges=support_edges,
                                 point_ids=set(i for e in support_edges for i in e)))
        remaining = [e for k, e in enumerate(remaining) if k not in support]

    return planes


# ---- step 3: plane -> Ramp parameters, in Frame-N ------------------------------------------

def mesh_frame_n_transform(m: trimesh.Trimesh, margin: float = 1.05):
    """The exact center/scale `building_to_sdf` uses, so wireframe-derived planes land in the
    SAME normalized space as the SDF grid #10's fitter operates on."""
    ext = m.extents
    c = m.bounds.mean(0)
    s = float(ext.max()) / 2 * margin
    return c, s


def to_frame_n(points: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    """CityGML (x, y, z-up) -> Frame-N (x, y=up, z), matching `building_to_sdf`'s own swap."""
    p = (points - center) / scale
    return np.stack([p[:, 0], p[:, 2], p[:, 1]], axis=-1)


def plane_pitch_azimuth(normal_frame_n: np.ndarray) -> tuple[float, float]:
    """pitch: angle between the plane and horizontal (0=flat, 90=vertical/wall-like).
    azimuth: compass direction (degrees, 0=+D axis) the roof slopes DOWN toward."""
    n = normal_frame_n / (np.linalg.norm(normal_frame_n) + 1e-12)
    if n[1] < 0:
        n = -n  # normal should point "up-ish" out of the roof
    pitch = math.degrees(math.acos(min(1.0, max(-1.0, n[1]))))
    down_dw = np.array([n[0], n[2]])  # horizontal component of the normal = downslope direction
    if np.linalg.norm(down_dw) < 1e-9:
        azimuth = float("nan")  # flat plane, azimuth undefined
    else:
        azimuth = math.degrees(math.atan2(down_dw[1], down_dw[0])) % 360.0
    return pitch, azimuth


# ---- driver -------------------------------------------------------------------------------

def process_building(mesh_bytes: bytes, wire_data: str) -> dict:
    m = trimesh.load(io.BytesIO(mesh_bytes), file_type="obj", process=False)
    center, scale = mesh_frame_n_transform(m)

    V, edges = load_obj_verts_edges(wire_data)
    classes = classify_edges(V, edges)
    ext = m.extents
    tol = float(ext.max()) * PLANE_DIST_TOL_FRAC
    planes = cluster_planes(V, classes["sloped"], tol, classes["horizontal"])

    Vf = to_frame_n(V, center, scale)
    plane_reports = []
    for p in planes:
        # refit the plane directly in Frame-N space (a monotone transform, but cleanest to
        # avoid transforming a normal vector through a non-uniform coordinate swap by hand)
        pts_f = Vf[sorted(p.point_ids)]
        n_f, d_f = _fit_plane(pts_f)
        pitch, azimuth = plane_pitch_azimuth(n_f)
        plane_reports.append({
            "n_edges": len(p.edges),
            "n_points": len(p.point_ids),
            "pitch_deg": round(pitch, 1),
            "azimuth_deg": round(azimuth, 1) if not math.isnan(azimuth) else None,
        })
    n_unclustered = len(classes["sloped"]) - sum(p["n_edges"] for p in plane_reports)
    return {
        "n_edges_total": len(edges),
        "n_wall": len(classes["vertical"]),
        "n_horizontal": len(classes["horizontal"]),
        "n_sloped": len(classes["sloped"]),
        "n_planes": len(plane_reports),
        "planes": sorted(plane_reports, key=lambda r: -r["n_edges"]),
        "n_sloped_unclustered": n_unclustered,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="Adelaide")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--prefix", default=None, help="only buildings whose id starts with this")
    ap.add_argument("--out", default=None,
                     help="write the full per-building report here as json "
                          "(e.g. execution/artifacts/wireframe_ramp_probe_adelaide.json)")
    args = ap.parse_args()

    mesh_zip_path = MESH_DIR / args.city / "mesh" / "obj.zip"
    if not mesh_zip_path.exists():
        mesh_zip_path = MESH_DIR / args.city / "obj" / "mesh.zip"
    wire_zip_path = WIRE_DIR / args.city / "wireframe" / "wireframe.zip"

    zm = zipfile.ZipFile(mesh_zip_path)
    zw = zipfile.ZipFile(wire_zip_path)

    def prefix_of(name: str) -> str:
        return Path(name).stem.rsplit("_", 1)[0]

    mesh_by_prefix: dict[str, list[str]] = {}
    for n in zm.namelist():
        if not n.endswith(".obj"):
            continue
        if args.prefix and not Path(n).stem.startswith(args.prefix):
            continue
        mesh_by_prefix.setdefault(prefix_of(n), []).append(n)

    wire_by_prefix: dict[str, list[str]] = {}
    for n in zw.namelist():
        if not n.endswith(".obj"):
            continue
        wire_by_prefix.setdefault(prefix_of(n), []).append(n)

    checked = ok = 0
    results: dict[str, dict] = {}
    for prefix in sorted(wire_by_prefix):
        if ok >= args.n:
            break
        if prefix not in mesh_by_prefix:
            continue
        # Geometric matching, NOT filename-index matching -- see module docstring: same-index
        # pairing is provably wrong once any earlier building in the prefix group was dropped
        # during wireframe generation.
        pairing = match_wireframes_to_meshes(zm, zw, mesh_by_prefix[prefix], wire_by_prefix[prefix])
        for wname, mname in pairing.items():
            if ok >= args.n:
                break
            checked += 1
            stem = Path(mname).stem  # e.g. "10a_0"
            try:
                report = process_building(zm.read(mname), zw.read(wname).decode())
            except Exception as e:
                print(f"[skip] {stem}: {type(e).__name__}: {e}")
                continue
            ok += 1
            results[stem] = report
            clean = [p for p in report["planes"] if p["pitch_deg"] is not None and p["pitch_deg"] < 75]
            noisy = [p for p in report["planes"]
                     if not (p["pitch_deg"] is not None and p["pitch_deg"] < 75)]
            report["n_planes_plausible_ramp_lt75deg"] = len(clean)
            report["n_planes_near_vertical_ge75deg"] = len(noisy)
            print(f"\n=== {args.city}/{stem} (wireframe: {wname}) ===")
            print(f"  edges: total={report['n_edges_total']} wall={report['n_wall']} "
                  f"horizontal={report['n_horizontal']} sloped={report['n_sloped']}")
            print(f"  planes recovered: {report['n_planes']} "
                  f"({len(clean)} plausible-ramp <75deg, {len(noisy)} near-vertical/noise >=75deg) "
                  f"(unclustered sloped edges: {report['n_sloped_unclustered']})")
            for p in report["planes"]:
                print(f"    pitch={p['pitch_deg']:>5.1f} deg  azimuth={p['azimuth_deg']}  "
                      f"support: {p['n_edges']} edges / {p['n_points']} pts")

    print(f"\n[done] {args.city}: checked {checked} mesh entries, found wireframes for "
          f"{ok} (coverage-limited by --n / missing wireframes)")

    if args.out:
        import json

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": {
                "script": "scripts/foundations/wireframe_ramp_probe.py",
                "steps_covered": "1-3 only (edge classification, plane clustering, "
                                  "Ramp pitch/azimuth) -- NOT step 4 (carving onto the "
                                  "extruded solid) or step 5 (IoU validation)",
                "city": args.city, "n_requested": args.n, "prefix": args.prefix,
                "n_checked": checked, "n_found": ok,
                "pairing": "geometric (nearest-centroid within the same prefix group), not "
                           "filename-index -- see match_wireframes_to_meshes",
                "known_limitation": "plane clustering is more conservative now (loop-closure "
                                     "gated, see cluster_planes/_forms_closed_loop) but still "
                                     "leaves a near-vertical minority (gable-end walls or "
                                     "residual seeding noise) and a large unclustered-edge "
                                     "count on complex buildings -- see "
                                     "n_planes_near_vertical_ge75deg and "
                                     "n_sloped_unclustered per building.",
            },
            "buildings": results,
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"[wrote] {out_path}")


if __name__ == "__main__":
    main()
