"""#63 — measure the crisp ceiling of a query-based (vecset) decoder against our dense-grid stack.

Answers map-#61's make-or-break question: does a query-based decoder visibly clear the 0.0047 wall
that #54/#59/#60 all hit, or does it land in the same band? Runs the vendored Hunyuan3D-2 as a
MEASURING INSTRUMENT (not a generator we adopt) and scores its output with OUR metric on OUR grid,
so the number is directly comparable to GT 0.0041 / round-trip 0.0044 / map-#24 0.00552.

Method, and why it is fair:
  `surface_roughness` is a band-weighted mean |Laplacian| on a VOLUMETRIC field, so the teacher's
  mesh must come back to the same 64^3 grid to be comparable. That is not stacking the deck against
  it: GT is itself a 64^3 field and scores 0.0041, and #56 showed the codec round-trips at 0.0044 --
  i.e. the 64^3 grid can REPRESENT crisp geometry. So if the teacher's surface is genuinely crisp it
  should score near the GT floor, exactly as GT does.

  The control arm is what makes the mesh->SDF path trustworthy: we re-voxelize GT's OWN extracted
  mesh through the identical path. If the control reproduces ~0.0041 the path is faithful and the
  teacher's number means something; if it inflates, the path is adding roughness of its own and the
  teacher's number must be discounted by that amount. Never report the teacher without the control.

Frame: real.h5 grids span [-1,1]^3 at spacing 2/63 (see datasets/bag3d_dataset.py height_n), values
clamped to +-0.2 at load time (trunc_thres). The teacher mesh is normalized onto the GT mesh's own
bounding box so both surfaces are sampled at the same spatial density.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.foundations.baseline_gate_eval import mesh_sdf_surface  # noqa: E402
from scripts.foundations.refiner_prototype import surface_roughness  # noqa: E402

H5 = REPO / "data/real_massing_v1/real.h5"
TRUNC = 0.2
RES = 64
SPACING = 2.0 / (RES - 1)          # grid spans [-1, 1]

# Reference numbers from prior efforts, for the ladder (n=24 held-out).
REF = {"gt_floor": 0.0041, "codec_roundtrip": 0.0044,
       "refiner_wall": 0.0047, "map24_sample": 0.00552}


def grid_points() -> np.ndarray:
    """The 64^3 sample points of the real.h5 frame, in world coords, index-order (x,y,z)."""
    ax = np.linspace(-1.0, 1.0, RES, dtype=np.float64)
    gx, gy, gz = np.meshgrid(ax, ax, ax, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)


def verts_to_world(verts: np.ndarray) -> np.ndarray:
    """marching_cubes returns verts in voxel-index space [0, 63]; map to the [-1,1] world frame."""
    return verts.astype(np.float64) * SPACING - 1.0


def mesh_to_sdf(verts_w: np.ndarray, faces: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Signed distance of `pts` to the mesh, via libigl's winding-number sign. Returns (64,64,64)."""
    import igl
    out = igl.signed_distance(pts, np.ascontiguousarray(verts_w),
                              np.ascontiguousarray(faces.astype(np.int32)))
    s = out[0] if isinstance(out, tuple) else out
    return np.asarray(s, dtype=np.float32).reshape(RES, RES, RES)


def roughness(field: np.ndarray) -> float:
    """Our metric, on a +-TRUNC clamped field -- the same convention the dataset loader applies."""
    t = torch.from_numpy(np.clip(np.asarray(field, np.float32), -TRUNC, TRUNC))
    return surface_roughness(t)


def normalize_onto(src_v: np.ndarray, ref_v: np.ndarray) -> np.ndarray:
    """Place `src_v` in `ref_v`'s bounding box (same centre, same max half-extent) so both surfaces
    are sampled at the same voxel density -- roughness is scale-dependent, so this must be done."""
    def box(v):
        lo, hi = v.min(0), v.max(0)
        return (lo + hi) / 2.0, float(np.abs(hi - lo).max()) / 2.0
    sc, ss = box(src_v)
    rc, rs = box(ref_v)
    return (src_v - sc) * (rs / max(ss, 1e-9)) + rc


def test_indices(n_total: int) -> np.ndarray:
    """Bag3dDataset's deterministic 96/2/2 split -- the same held-out slice the gate uses."""
    perm = np.random.default_rng(0).permutation(n_total)
    n_val = max(1, int(0.02 * n_total))
    return perm[n_val:2 * n_val]


def build_montage(out: Path) -> Path | None:
    """Honest side-by-side: GT surface vs teacher surface, same buildings, roughness annotated.
    Per #36 the visual is the primary arbiter and the scalar is the diagnostic, so both appear."""
    from PIL import Image, ImageDraw
    lad = json.loads((out / "ladder.json").read_text())
    rows = [r for r in lad["rows"] if "teacher" in r]
    if not rows:
        return None
    S, PAD, HDR = 320, 8, 26
    W, H = 2 * S + 3 * PAD, HDR + len(rows) * (S + HDR) + PAD
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    d.text((PAD, 7), f"GT (real LoD2)                          "
                     f"    Hunyuan3D-2 teacher  —  n={len(rows)} held-out", fill=(0, 0, 0))
    for i, r in enumerate(rows):
        y = HDR + i * (S + HDR)
        for j, (tag, lab) in enumerate((("prompt", "gt"), ("teacher", "teacher"))):
            p = out / f"{i:02d}_{tag}.png"
            if p.exists():
                canvas.paste(Image.open(p).convert("RGB").resize((S, S)),
                             (PAD + j * (S + PAD), y + HDR))
            d.text((PAD + j * (S + PAD) + 4, y + 6),
                   f"{lab}  roughness {r[lab]:.5f}", fill=(0, 0, 0))
    path = out / "ceiling_montage.png"
    canvas.save(path)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--montage_only", action="store_true",
                    help="rebuild the montage from an existing run")
    ap.add_argument("--n", type=int, default=8, help="held-out buildings to probe")
    ap.add_argument("--out_dir", default="outputs/vecset_ceiling")
    ap.add_argument("--control_only", action="store_true",
                    help="validate the mesh->SDF path on GT alone; no GPU, no teacher")
    ap.add_argument("--model", choices=["mini", "full"], default="full")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--octree_resolution", type=int, default=380)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.montage_only:
        print("montage:", build_montage(out))
        return
    pts = grid_points()

    import h5py
    with h5py.File(H5, "r") as f:
        idxs = test_indices(int(f["sdf"].shape[0]))[:args.n]
        gts = [np.asarray(f["sdf"][int(i)], dtype=np.float32) for i in idxs]
        bag_ids = [f["bag_id"][int(i)].decode() for i in idxs]

    rows = []
    for k, (gi, gt) in enumerate(zip(idxs, gts)):
        r = {"index": int(gi), "bag_id": bag_ids[k]}
        r["gt"] = roughness(gt)

        v, fc = mesh_sdf_surface(np.clip(gt, -TRUNC, TRUNC))
        if v is None:
            print(f"[{k}] #{gi} skipped (no zero crossing)")
            continue
        vw = verts_to_world(v)
        t0 = time.time()
        r["control"] = roughness(mesh_to_sdf(vw, fc, pts))
        r["control_sec"] = round(time.time() - t0, 1)
        np.save(out / f"{k:02d}_gt_mesh.npy", np.array([len(v), len(fc)]))
        np.savez_compressed(out / f"{k:02d}_gtmesh.npz", v=vw, f=fc)

        print(f"[{k}] #{gi} {bag_ids[k][:18]:18s} gt={r['gt']:.5f} "
              f"control={r['control']:.5f}  (delta {r['control']-r['gt']:+.5f}, "
              f"{r['control_sec']}s, V={len(v):,})")
        rows.append(r)

    if rows:
        g = float(np.mean([x["gt"] for x in rows]))
        c = float(np.mean([x["control"] for x in rows]))
        print(f"\n  MEAN gt={g:.5f}  control={c:.5f}  delta={c-g:+.5f}")
        print(f"  reference: GT floor {REF['gt_floor']}, wall {REF['refiner_wall']}, "
              f"map-#24 {REF['map24_sample']}")
        verdict = ("FAITHFUL — teacher numbers will be trustworthy"
                   if abs(c - g) < 0.0006 else
                   "PATH ADDS ROUGHNESS — discount the teacher number by this delta")
        print(f"  control verdict: {verdict}")
        json.dump({"rows": rows, "mean_gt": g, "mean_control": c,
                   "delta": c - g, "reference": REF, "verdict": verdict},
                  open(out / "control.json", "w"), indent=2)

    if args.control_only:
        print("\n[control_only] stopping before the teacher run.")
        return

    # --- teacher arm ------------------------------------------------------
    # Image-conditioned, so we prompt it with a render of the GT building itself: the fairest
    # available bridge, and the reason this measures ACHIEVABLE SURFACE QUALITY rather than
    # footprint-faithful generation. That limit is real and is reported with the result.
    os.environ.setdefault("HF_HOME", str(REPO / "external/hf_cache"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(REPO / "external/hf_cache/hub"))
    os.environ.setdefault("XDG_CACHE_HOME", str(REPO / "external/xdg_cache"))
    os.environ.setdefault("HY3DGEN_MODELS", str(REPO / "external/hy3dgen_models"))
    sys.path.insert(0, str(REPO / "external/Hunyuan3D-2"))

    import trimesh
    from scripts.hunyuan_building_mesh_smoke import load_pipeline, render_mesh_png

    print(f"\n[teacher] loading Hunyuan3D-2 ({args.model})", flush=True)
    pipe = load_pipeline(args.model)
    print("[teacher] ready", flush=True)

    for k, r in enumerate(rows):
        d = np.load(out / f"{k:02d}_gtmesh.npz")
        gv, gf = d["v"], d["f"]
        prompt = render_mesh_png(trimesh.Trimesh(gv, gf, process=False), image_size=512)
        prompt.save(out / f"{k:02d}_prompt.png")

        t0 = time.time()
        tm = pipe(image=prompt.convert("RGBA"), num_inference_steps=args.steps,
                  octree_resolution=args.octree_resolution, num_chunks=20000,
                  generator=torch.manual_seed(args.seed + k), output_type="trimesh")[0]
        secs = time.time() - t0

        tv = normalize_onto(np.asarray(tm.vertices, np.float64), gv)
        r["teacher"] = roughness(mesh_to_sdf(tv, np.asarray(tm.faces), pts))
        r["teacher_sec"] = round(secs, 1)
        r["teacher_verts"] = int(len(tm.vertices))
        tm.export(out / f"{k:02d}_teacher.glb")
        render_mesh_png(trimesh.Trimesh(tv, tm.faces, process=False),
                        image_size=512).save(out / f"{k:02d}_teacher.png")
        print(f"[{k}] gt={r['gt']:.5f}  teacher={r['teacher']:.5f}  "
              f"({secs:.0f}s, V={len(tm.vertices):,})", flush=True)

    done = [r for r in rows if "teacher" in r]
    if done:
        g = float(np.mean([x["gt"] for x in done]))
        t = float(np.mean([x["teacher"] for x in done]))
        clears = t < REF["refiner_wall"]
        print(f"\n=== LADDER (n={len(done)}, same buildings) ===")
        print(f"  GT (this sample)      {g:.5f}")
        print(f"  teacher               {t:.5f}")
        print(f"  --- references (n=24, prior runs) ---")
        print(f"  codec round-trip      {REF['codec_roundtrip']:.5f}")
        print(f"  refiner/corrector wall{REF['refiner_wall']:.5f}")
        print(f"  map-#24 deployed      {REF['map24_sample']:.5f}")
        print(f"\n  teacher clears the 0.0047 wall: {'YES' if clears else 'NO'}")
        json.dump({"rows": rows, "mean_gt": g, "mean_teacher": t,
                   "clears_wall": bool(clears), "reference": REF,
                   "limit": "image-conditioned teacher: measures achievable surface quality, "
                            "NOT footprint-faithful generation"},
                  open(out / "ladder.json", "w"), indent=2)
        print("  montage:", build_montage(out))


if __name__ == "__main__":
    main()
