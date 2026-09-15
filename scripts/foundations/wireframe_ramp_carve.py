"""Steps 4-5 of the wireframe-guided Ramp precision check (see wireframe_ramp_probe.py for
steps 1-3, which classify wireframe edges, cluster them into roof planes, and convert each plane
to pitch/azimuth). Exploratory -- not a ticket yet.

Method (v2 -- coherent search, not post-hoc substitution): for one BuildingWorld building (a
geometrically-matched mesh+wireframe pair -- see `match_wireframes_to_meshes`), run #10's own
beam-search fitter (`fit_program_beam`) twice on the identical inputs: once unbiased (the
BASELINE, exactly as it runs everywhere else on this map) and once with `wf_planes` set to the
wireframe-derived planes (`recover_massing_programs.py`'s `_wireframe_ramp_candidates` /
`_candidates_with_wireframe`). In the second run, a wireframe plane is just another `Ramp`
candidate the search can pick at any step -- it wins a region only if it removes more surplus
than every other candidate on offer there, under the exact same containment guard (never cuts
into GT) every other candidate obeys. Both programs are scored against GT with the standard
`missing`/`extra`/`vol_iou`/`vs_input` contract (`eval_massing_arms.py`).

**v1 (kept in history, not used here) force-substituted a wireframe plane into an
already-finished baseline program, region-for-region, after the fact.** That measured something
real (see the earlier commit) but wasn't coherent with the fitter: it couldn't let wireframe
guidance change WHICH region gets an op, how many ops get used, or compete honestly against the
LP's own candidate on merit -- it could only overwrite a plane equation post-hoc. This version
lets the wireframe candidate enter the SAME search everything else goes through, so if it wins,
it won on the fitter's own terms.

This still isolates a version of one question -- does wireframe guidance improve on the fitter's
own choices -- but now at the SEARCH level rather than the region level: a wireframe plane can
change region boundaries, op count, and op order too, not just override one plane equation. It
does NOT mean the wireframe planes' own region masks (from `wireframe_planes_in_voxel_space`)
are perfectly accurate -- they're still a convex-hull rasterization of the plane's support
vertices, an independent source of error from the plane equation itself.

Run: env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
     scripts/foundations/wireframe_ramp_carve.py --city Adelaide --n 8 \
     --out execution/artifacts/wireframe_ramp_carve_adelaide.json
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.ingest_3dbag import building_to_sdf                                # noqa: E402
from scripts.foundations.eval_massing_arms import RES, volume_split, vs_input   # noqa: E402
from scripts.foundations.recover_massing_programs import (                      # noqa: E402
    fit_program_beam,
    height_field,
    occupancy,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wireframe_ramp_probe import (   # noqa: E402
    MESH_DIR,
    WIRE_DIR,
    PLANE_DIST_TOL_FRAC,
    _fit_plane,
    classify_edges,
    cluster_planes,
    load_obj_verts_edges,
    match_wireframes_to_meshes,
    mesh_frame_n_transform,
    to_frame_n,
)

# ---- Frame-N plane -> recover_massing_programs.py's voxel-index (a, b, c) -------------------

def frame_n_plane_to_voxel_plane(
    normal: np.ndarray, d: float, y0: int, res: int = RES
) -> tuple[float, float, float]:
    """A Frame-N plane `normal . (x, y_up, z) = d` -> voxel-index `(a, b, c)` such that
    `a + b*xvox + c*zvox` reproduces the same plane's height, in voxels above `y0` -- the exact
    convention `_ramp_candidates`/`plane_surface` use (verified against their source: `zz, xx =
    np.mgrid[0:res, 0:res]`, `height = floor(a + b*xx + c*zz)`, where `xx` is the Frame-N X/"W"
    voxel index and `zz` is the Frame-N Z/"D" voxel index -- matching `building_to_sdf`'s own
    axis convention exactly, confirmed via its `P = stack([XX, YY, ZZ])` feed into `igl.
    signed_distance` against `Vn`'s own `(x, z, y)`-ordered columns).

    `res` voxels span Frame-N's `[-1, 1]`, so `coord = -1 + i * spacing` with
    `spacing = 2/(res-1)`. Solving the plane equation for the "up" (Frame-N Y) coordinate as a
    function of the two footprint coordinates, then converting that to a voxel height above `y0`,
    gives a plane linear in `(xvox, zvox)` -- exactly the form `_ramp_candidates` fits by LP.
    """
    n0, n1, n2 = (float(v) for v in normal)  # (X="W", Y="H"/up, Z="D")
    sp = 2.0 / (res - 1)
    # Y_frame_n(xvox, zvox) = (d - n0*(-1+xvox*sp) - n2*(-1+zvox*sp)) / n1
    #                       = (d + n0 + n2)/n1 - (n0*sp/n1)*xvox - (n2*sp/n1)*zvox
    # H_voxel = (Y_frame_n + 1) / sp
    a_h = ((d + n0 + n2) / n1 + 1.0) / sp
    b_h = -(n0 / n1)
    c_h = -(n2 / n1)
    a = a_h - y0 + 1.0
    return a, b_h, c_h


def plane_region_mask(points_xy: np.ndarray, fp: np.ndarray, res: int = RES) -> np.ndarray:
    """Convex hull of a plane's own boundary points (Frame-N X, Z, i.e. voxel-space x, z
    coordinates already converted), rasterized to a (res, res) boolean mask and intersected with
    the building's own footprint -- a wireframe plane's carve can never extend past the footprint
    the fitter itself is bounded by.
    """
    from matplotlib.path import Path as MplPath

    if len(points_xy) < 3:
        return np.zeros((res, res), bool)
    try:
        from scipy.spatial import ConvexHull

        hull = points_xy[ConvexHull(points_xy).vertices]
    except Exception:
        hull = points_xy
    zz, xx = np.mgrid[0:res, 0:res]
    pts = np.stack([xx.ravel(), zz.ravel()], axis=1).astype(float)
    mask = MplPath(hull).contains_points(pts).reshape(res, res)
    return mask & fp


def to_voxel_xz(frame_n_xz: np.ndarray, res: int = RES) -> np.ndarray:
    """Frame-N (X, Z) in [-1, 1] -> voxel-index (xvox, zvox), matching `frame_n_plane_to_voxel_plane`."""
    sp = 2.0 / (res - 1)
    return (frame_n_xz + 1.0) / sp


# ---- recover the wireframe planes for one building, in voxel space --------------------------

def wireframe_planes_in_voxel_space(mesh_bytes: bytes, wire_data: str, fp: np.ndarray, y0: int):
    """Steps 1-3 (reused from wireframe_ramp_probe.py) + the voxel conversion above. Returns a
    list of dicts: {plane: (a,b,c), region: bool mask, pitch_deg, azimuth_deg, n_support_edges}.
    """
    m = trimesh.load(io.BytesIO(mesh_bytes), file_type="obj", process=False)
    center, scale = mesh_frame_n_transform(m)

    V, edges = load_obj_verts_edges(wire_data)
    classes = classify_edges(V, edges)
    ext = m.extents
    tol = float(ext.max()) * PLANE_DIST_TOL_FRAC
    planes = cluster_planes(V, classes["sloped"], tol, classes["horizontal"])
    Vf = to_frame_n(V, center, scale)

    out = []
    for p in planes:
        pts_f = Vf[sorted(p.point_ids)]
        n_f, d_f = _fit_plane(pts_f)
        if n_f[1] < 0:
            n_f, d_f = -n_f, -d_f
        pitch_deg = float(np.degrees(np.arccos(np.clip(abs(n_f[1]), 0.0, 1.0))))
        if pitch_deg >= 75.0 or abs(n_f[1]) < 1e-6:
            continue  # near-vertical / degenerate -- not a servable ramp, matches probe's filter
        a, b, c = frame_n_plane_to_voxel_plane(n_f, d_f, y0)
        xz_vox = to_voxel_xz(pts_f[:, [0, 2]])
        region = plane_region_mask(xz_vox, fp)
        if not region.any():
            continue
        out.append({
            "plane": (a, b, c),
            "region": region,
            "pitch_deg": round(pitch_deg, 1),
            "n_support_edges": len(p.edges),
        })
    return out


# ---- one building end to end -------------------------------------------------------------

def run_one_building(mname: str, wname: str, zm: zipfile.ZipFile, zw: zipfile.ZipFile) -> dict | None:
    m = trimesh.load(io.BytesIO(zm.read(mname)), file_type="obj", process=False)
    sdf, fp_u8, height_m = building_to_sdf(m, R=RES)
    gt = sdf <= 0
    fp = fp_u8 > 0
    hf = height_field(gt, fp)
    if hf is None:
        return None
    y0, y1, target = hf
    bo_occ = occupancy(fp, y0, np.where(fp, np.int16(y1 - y0 + 1), 0).astype(np.int16))

    wf_planes = wireframe_planes_in_voxel_space(zm.read(mname), zw.read(wname).decode(), fp, y0)
    if not wf_planes:
        return None

    baseline_ops, baseline_h = fit_program_beam(fp, y0, y1, target)
    baseline_occ = occupancy(fp, y0, baseline_h)
    baseline_score = volume_split(baseline_occ, gt)
    baseline_score["vs_input"] = vs_input(baseline_occ, bo_occ)

    guided_ops, guided_h = fit_program_beam(fp, y0, y1, target, wf_planes=wf_planes)
    guided_occ = occupancy(fp, y0, guided_h)
    guided_score = volume_split(guided_occ, gt)
    guided_score["vs_input"] = vs_input(guided_occ, bo_occ)

    used = [o for o in guided_ops if o.get("source") == "wireframe"]
    if not used:
        return None  # offered, but never won the search on its own merits -- nothing to compare

    # Localized read: missing/extra WITHIN the region(s) a wireframe candidate actually WON, so
    # the effect isn't diluted by the rest of the building (which may differ between the two
    # programs anyway now -- this is a coherent re-search, not a single-op substitution).
    won_mask = np.zeros_like(fp)
    for op in used:
        if "_region" in op:
            won_mask |= op["_region"]

    def local_missing_extra(occ):
        region_gt = gt & np.broadcast_to(won_mask[:, None, :], gt.shape)
        region_occ = occ & np.broadcast_to(won_mask[:, None, :], gt.shape)
        gt_vox = int(region_gt.sum())
        missing = int((region_gt & ~region_occ).sum())
        extra = int((region_occ & ~region_gt).sum())
        return {
            "gt_vox": gt_vox,
            "missing_frac": round(missing / gt_vox, 4) if gt_vox else None,
            "extra_frac": round(extra / gt_vox, 4) if gt_vox else None,
        }

    return {
        "n_ramp_ops_baseline": sum(1 for o in baseline_ops if o["op"] == "Ramp"),
        "n_ramp_ops_guided": sum(1 for o in guided_ops if o["op"] == "Ramp"),
        "n_ops_baseline": len(baseline_ops),
        "n_ops_guided": len(guided_ops),
        "n_wireframe_ops_won": len(used),
        "wireframe_ops_won": [
            {"pitch_deg": o.get("wf_pitch_deg"), "area": o["area"], "plane": o["plane"]}
            for o in used
        ],
        "baseline": {k: round(v, 4) if isinstance(v, float) else v for k, v in baseline_score.items()},
        "wireframe_guided": {k: round(v, 4) if isinstance(v, float) else v for k, v in guided_score.items()},
        "local_baseline": local_missing_extra(baseline_occ),
        "local_wireframe_guided": local_missing_extra(guided_occ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="Adelaide")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default=None)
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
        if n.endswith(".obj"):
            mesh_by_prefix.setdefault(prefix_of(n), []).append(n)
    wire_by_prefix: dict[str, list[str]] = {}
    for n in zw.namelist():
        if n.endswith(".obj"):
            wire_by_prefix.setdefault(prefix_of(n), []).append(n)

    results = {}
    checked = 0
    for prefix in sorted(wire_by_prefix):
        if len(results) >= args.n:
            break
        if prefix not in mesh_by_prefix:
            continue
        pairing = match_wireframes_to_meshes(zm, zw, mesh_by_prefix[prefix], wire_by_prefix[prefix])
        for wname, mname in pairing.items():
            if len(results) >= args.n:
                break
            stem = Path(mname).stem
            checked += 1
            try:
                row = run_one_building(mname, wname, zm, zw)
            except Exception as e:
                print(f"[skip] {stem}: {type(e).__name__}: {e}")
                continue
            if row is None:
                continue
            results[f"{args.city}/{stem}"] = row
            b, w = row["baseline"], row["wireframe_guided"]
            lb, lw = row["local_baseline"], row["local_wireframe_guided"]
            print(f"=== {args.city}/{stem} ({row['n_wireframe_ops_won']} wireframe Ramp op(s) won "
                  f"the search; ops {row['n_ops_baseline']}->{row['n_ops_guided']}, "
                  f"Ramp ops {row['n_ramp_ops_baseline']}->{row['n_ramp_ops_guided']}) ===")
            print(f"  whole-building  missing {b['missing']:.4f}->{w['missing']:.4f}  "
                  f"extra {b['extra']:.4f}->{w['extra']:.4f}  vol_iou {b['vol_iou']:.4f}->{w['vol_iou']:.4f}")
            print(f"  wireframe-won region  missing {lb['missing_frac']}->{lw['missing_frac']}  "
                  f"extra {lb['extra_frac']}->{lw['extra_frac']}  ({lb['gt_vox']} GT voxels)")

    print(f"\n[done] {args.city}: checked {checked}, {len(results)} buildings had a wireframe "
          f"Ramp candidate actually win a region in the coherent search")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "meta": {
                "script": "scripts/foundations/wireframe_ramp_carve.py",
                "what_this_measures": "run #10's beam-search fitter twice, unbiased vs. with "
                                       "wireframe-derived Ramp planes offered as additional "
                                       "candidates (recover_massing_programs.py's "
                                       "_wireframe_ramp_candidates) -- a wireframe plane only "
                                       "wins a region if it beats every other candidate on raw "
                                       "gain under the same containment guard. Coherent search, "
                                       "not post-hoc substitution (see module docstring for why "
                                       "the v1 substitution approach was replaced). Steps 4-5 of "
                                       "the wireframe precision check (see wireframe_ramp_probe.py "
                                       "for steps 1-3)",
                "city": args.city, "n_checked": checked, "n_compared": len(results),
            },
            "buildings": results,
        }, indent=2))
        print(f"[wrote] {out_path}")


if __name__ == "__main__":
    main()
