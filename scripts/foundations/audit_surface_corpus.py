"""#74 -- what geometric signals can the recovered LoD2 surface corpus actually produce?

The supervision research (#72) concluded no *new* input signal is needed: normals and sharp-edge
samples are already fed, and our sampler matches Dora's recipe. This audits that conclusion against
the corpus on disk rather than trusting it, and prices anything missing.

Checks every mesh, not a sample -- the whole corpus is 18 MB, so there is no reason to extrapolate.

⚠️ **Orientation is checked, never presumed.** 400/400 surfaces once had inward-facing normals while an
alignment check passed at IoU 1.0000 (#62): signed-distance paths sign by winding number and simply do
not notice, but a vecset encoder consumes face normals directly and would be handed inside-out
surfaces. The check here is the sign of the enclosed volume, computed on the **raw stored** geometry --
`dora_frozen_gate.load_surfaces` flips negative-volume meshes at load time, so auditing post-load would
hide whether the corpus itself is sound.

Run:
    audit_surface_corpus.py                # whole corpus
    audit_surface_corpus.py --limit 500    # smoke
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SOURCES = ("bag3d", "nrw", "plateau")
SURF = REPO / "data/real_massing_v1"
H5 = SURF / "real.h5"
SHARP_DEG = 25.0          # the dihedral threshold `scene.surface_sampling` uses for the sharp stream


def audit_mesh(v: np.ndarray, f: np.ndarray) -> dict:
    """One mesh -> the facts a vecset encoder's supervision depends on."""
    import trimesh
    m = trimesh.Trimesh(np.asarray(v, np.float64), np.asarray(f), process=False)
    try:
        ang = m.face_adjacency_angles
        sharp = int((ang > np.deg2rad(SHARP_DEG)).sum())
    except Exception:
        sharp = 0
    try:
        vol = float(m.volume)
    except Exception:
        vol = 0.0
    tri = m.triangles
    area = float(m.area)
    return dict(
        n_verts=int(len(v)), n_faces=int(len(f)),
        watertight=bool(m.is_watertight),
        winding_consistent=bool(m.is_winding_consistent),
        volume=vol,
        outward=bool(vol > 0.0),          # negative volume == inward-facing normals
        area=area,
        sharp_edges=sharp,
        has_sharp=bool(sharp > 0),
        degenerate=int((np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
                                       axis=1) <= 1e-12).sum()) if len(tri) else 0,
    )


def pipeline_orientation(n: int = 8) -> dict:
    """Trace face orientation along the path a latent is actually built on.

    Raw storage being inward is survivable only if every consumer repairs it. This walks the four
    stages and reports the volume sign at each, so the claim "the encoder gets outward surfaces" is
    measured at the point of consumption rather than argued from the ingest's docstring.
    """
    import h5py
    import trimesh
    from models.shape_codec import Building
    from scene.surface_sampling import to_array_frame
    from scripts.foundations.dora_frozen_gate import load_surfaces

    def vol(v, f):
        return float(trimesh.Trimesh(np.asarray(v, np.float64), np.asarray(f),
                                     process=False).volume)

    with h5py.File(SURF / "surfaces_bag3d.h5", "r") as f:
        vo, fo, rr = f["vert_offset"][:], f["face_offset"][:], f["row"][:]
        V, F = f["verts"][:], f["faces"][:]
    raw = [vol(V[vo[i]:vo[i + 1]], F[fo[i]:fo[i + 1]]) for i in range(n)]
    surf = load_surfaces()
    loaded, arrayf, encoder = [], [], []
    for i in range(n):
        v, fa, _ = surf[int(rr[i])]
        loaded.append(vol(v, fa))
        av, af = to_array_frame(v, fa)
        arrayf.append(vol(av, af))
        encoder.append(float(Building(verts=av, faces=af).require_mesh().volume))
    return dict(
        n=n,
        raw_on_disk_all_negative=bool(all(x < 0 for x in raw)),
        after_load_surfaces_all_positive=bool(all(x > 0 for x in loaded)),
        after_to_array_frame_all_positive=bool(all(x > 0 for x in arrayf)),
        encoder_facing_all_positive=bool(all(x > 0 for x in encoder)),
        sample_volumes=dict(raw=raw, loaded=loaded, array_frame=arrayf, encoder=encoder),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = whole corpus")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    import h5py

    rows: dict[str, list] = {}
    all_rows: list[int] = []
    t0 = time.time()
    for src in SOURCES:
        p = SURF / f"surfaces_{src}.h5"
        if not p.exists():
            print(f"[warn] missing {p.name}")
            continue
        with h5py.File(p, "r") as f:
            vo, fo, rr = f["vert_offset"][:], f["face_offset"][:], f["row"][:]
            V, F = f["verts"][:], f["faces"][:]
        n = len(rr) if not args.limit else min(args.limit, len(rr))
        out = []
        for i in range(n):
            rec = audit_mesh(V[vo[i]:vo[i + 1]], F[fo[i]:fo[i + 1]])
            rec["row"] = int(rr[i])
            out.append(rec)
            all_rows.append(int(rr[i]))
            if (i + 1) % 2000 == 0:
                print(f"  [{src}] {i+1}/{n}  ({time.time()-t0:.0f}s)", flush=True)
        rows[src] = out
        print(f"[{src}] {len(out)} meshes audited ({time.time()-t0:.0f}s)", flush=True)

    def stats(rs: list) -> dict:
        if not rs:
            return {}
        g = lambda k: np.array([r[k] for r in rs])  # noqa: E731
        return dict(
            n=len(rs),
            watertight_frac=float(g("watertight").mean()),
            winding_consistent_frac=float(g("winding_consistent").mean()),
            outward_frac=float(g("outward").mean()),
            n_inward=int((~g("outward")).sum()),
            has_sharp_frac=float(g("has_sharp").mean()),
            sharp_edges_median=float(np.median(g("sharp_edges"))),
            faces_median=float(np.median(g("n_faces"))),
            faces_min=int(g("n_faces").min()), faces_max=int(g("n_faces").max()),
            verts_median=float(np.median(g("n_verts"))),
            degenerate_total=int(g("degenerate").sum()),
            n_with_degenerate=int((g("degenerate") > 0).sum()),
            area_median=float(np.median(g("area"))),
        )

    per_source = {s: stats(r) for s, r in rows.items()}
    overall = stats([r for rs in rows.values() for r in rs])

    # ---- the 153 that never came back ----------------------------------------------------------
    missing = {}
    if not args.limit:
        with h5py.File(H5, "r") as f:
            n_total = int(f["sdf"].shape[0])
            have = set(all_rows)
            absent = [i for i in range(n_total) if i not in have]
            src_of = f["source_id"][:]
            occ_fracs, fp_areas = [], []
            for i in absent:
                sdf = np.asarray(f["sdf"][i], np.float32)
                occ_fracs.append(float((sdf <= 0).mean()))
                fp_areas.append(float(np.asarray(f["footprint"][i]).mean()))
        missing = dict(
            n=len(absent), rows=absent[:200],
            by_source={str(int(s)): int((np.asarray([src_of[i] for i in absent]) == s).sum())
                       for s in sorted(set(int(x) for x in src_of))},
            occupancy_frac=dict(median=float(np.median(occ_fracs)) if occ_fracs else 0.0,
                                max=float(np.max(occ_fracs)) if occ_fracs else 0.0,
                                n_empty=int(sum(1 for o in occ_fracs if o <= 0.0))),
            footprint_frac=dict(median=float(np.median(fp_areas)) if fp_areas else 0.0,
                                n_empty=int(sum(1 for a in fp_areas if a <= 0.0))),
        )

    print(f"\n=== SURFACE CORPUS AUDIT (n={overall.get('n', 0)}) ===")
    hdr = f"{'source':10s} {'n':>7} {'watertight':>11} {'outward':>9} {'winding-ok':>11} " \
          f"{'has-sharp':>10} {'faces med':>10}"
    print(hdr)
    for s in list(per_source) + ["ALL"]:
        st = overall if s == "ALL" else per_source[s]
        if not st:
            continue
        print(f"{s:10s} {st['n']:>7} {st['watertight_frac']:>11.4f} {st['outward_frac']:>9.4f} "
              f"{st['winding_consistent_frac']:>11.4f} {st['has_sharp_frac']:>10.4f} "
              f"{st['faces_median']:>10.0f}")
    if overall:
        print(f"\ninward-facing (would be handed inside-out to a vecset encoder): "
              f"{overall['n_inward']} / {overall['n']}")
        print(f"meshes with degenerate faces: {overall['n_with_degenerate']} "
              f"({overall['degenerate_total']} faces total)")
        print(f"sharp edges per mesh at >{SHARP_DEG:.0f} deg: median "
              f"{overall['sharp_edges_median']:.0f}")
    if missing:
        print(f"\nunrecovered: {missing['n']} buildings, by source_id {missing['by_source']}")
        print(f"  their SDF occupancy: median {missing['occupancy_frac']['median']:.4f}, "
              f"max {missing['occupancy_frac']['max']:.4f}, "
              f"{missing['occupancy_frac']['n_empty']} completely empty")
        print(f"  their footprints: {missing['footprint_frac']['n_empty']} completely empty")

    orient = pipeline_orientation()
    print(f"\n--- orientation along the consumption path ---")
    print(f"  raw on disk all INWARD      : {orient['raw_on_disk_all_negative']}")
    print(f"  after load_surfaces outward : {orient['after_load_surfaces_all_positive']}")
    print(f"  after to_array_frame outward: {orient['after_to_array_frame_all_positive']}")
    print(f"  what the ENCODER receives   : "
          f"{'OUTWARD (ok)' if orient['encoder_facing_all_positive'] else 'INWARD (BROKEN)'}")

    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO),
                         capture_output=True, text=True).stdout.strip()
    suffix = f"_{args.tag}" if args.tag else ""
    art = REPO / f"execution/artifacts/surface_corpus_audit{suffix}.json"
    art.parent.mkdir(parents=True, exist_ok=True)
    art.write_text(json.dumps(dict(
        meta=dict(git_rev=rev, created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  sharp_deg=SHARP_DEG, limit=args.limit,
                  note="volume sign is measured on RAW stored geometry, before load_surfaces' flip"),
        overall=overall, per_source=per_source, unrecovered=missing,
        pipeline_orientation=orient), indent=2))
    print(f"\nartifact: {art}", flush=True)


if __name__ == "__main__":
    main()
