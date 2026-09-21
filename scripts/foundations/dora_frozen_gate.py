"""Spec-#68 deliverable: the FROZEN round-trip gate, on the real LoD2 surfaces recovered by #62.

Encodes held-out buildings through the untrained (frozen) vecset codec and scores the decode with our
existing metric and harness. The number sizes the fine-tune that spec #67 gates on -- **no training
happens here**.

Why this supersedes the earlier n=1 smoke: that run had no surface corpus, so it fed the encoder a mesh
extracted from our own 64^3 field. It therefore encoded the grid roughness the project is trying to
escape and scored 0.00839, worse than the deployed 0.00552 -- a lower bound on a degraded input, not a
measurement of the codec. #62 recovered real surfaces (35,623/35,776, verified aligned), so the gate can
now be run honestly.

Three-way ladder per building, which is what separates the codec's contribution from its input's:
  * **GT**        roughness of the stored 64^3 field                     -- the floor
  * **input**     the recovered mesh re-voxelised, WITHOUT the codec     -- the CONTROL arm
  * **frozen**    encode -> decode -> query -> re-voxelised              -- the measurement

The control is essential: `frozen` above `input` means the codec added roughness; `input` above `GT`
means the recovered surface was already rough and the codec is not to blame for that part. Reporting
`frozen` alone would conflate the two -- the mistake the first smoke made.

Units: the codec returns a positive-inside TSDF normalised to ~[-1,1] over a narrow band, not metric
distance, so its raw Laplacian is meaningless against our numbers. Every arm is therefore meshed at
level 0.0 and re-voxelised through the same igl path, so all three are measured identically.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.frozen_corpus import FROZEN_SPLIT_N_TOTAL, open_real_corpus  # noqa: E402
from scripts.foundations.baseline_gate_eval import mesh_sdf_surface          # noqa: E402
from scripts.foundations.refiner_prototype import surface_roughness          # noqa: E402
from scripts.foundations.vecset_ceiling_probe import (                       # noqa: E402
    RES, TRUNC, REF, grid_points, verts_to_world, test_indices,
)
from scripts.foundations.dora_roundtrip_probe import (                       # noqa: E402
    _stub_absent_deps, load_dora, sample_surface, sample_sharp_edges, H5,
)

SURF = REPO / "data/real_massing_v1"
TRIPO_VAE = REPO / "external/triposg_vae"
# #175: "buildingworld" -> "BW" is a pipeline label, not a country code -- unlike bag3d/nrw/
# plateau, BuildingWorld's 18 cities span many countries, and #171 (its region-conditioning
# granularity) has not landed, so this does not claim a single region for it.
SOURCES = {"bag3d": "NL", "nrw": "DE", "plateau": "JP", "buildingworld": "BW"}


def _revoxel(v: np.ndarray, f: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Signed distance of a mesh on our grid, signed the way the corpus voxeliser signs (#62):
    fast-winding-number, because these meshes can be watertight yet negative-volume."""
    import igl
    fwn = igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER
    s = igl.signed_distance(pts, np.ascontiguousarray(v, np.float64),
                            np.ascontiguousarray(f, np.int32), fwn)[0]
    return np.asarray(s, np.float32).reshape(RES, RES, RES)


def _rough(field: np.ndarray) -> float:
    return surface_roughness(torch.from_numpy(np.clip(field, -TRUNC, TRUNC)))


def load_surfaces(sources=None, rows=None):
    """row -> (verts, faces, src) for every recovered building, across `sources` (default: every
    registered `SOURCES` key). `rows` narrows that to named corpus rows (default: all of them).

    Code-review finding on #175: registering `"buildingworld"` in `SOURCES` silently grew this
    function's default row set from 35,776 to 1,562,554 -- ~1.5M extra verts/faces materialised
    into RAM, plus a `trimesh.Trimesh(...).volume` build per row for the winding check, for EVERY
    caller that takes the default. `sources` lets a caller that isn't ready for BuildingWorld's
    rows (`precompute_vecset_latents.py`'s default path, this module's own `main()` -- see their
    own call sites) say so explicitly, rather than paying that cost or crashing on a #161 ledger
    that doesn't cover those rows yet (#177's job). The default stays "every registered source" so
    a caller that DOES want everything (or is written before this parameter existed) is unaffected.
    """
    import h5py
    wanted = None if rows is None else {int(r) for r in rows}
    out = {}
    for src in (sources if sources is not None else SOURCES):
        p = SURF / f"surfaces_{src}.h5"
        if not p.exists():
            print(f"[warn] missing {p.name}"); continue
        with h5py.File(p, "r") as f:
            vo, fo, src_rows = f["vert_offset"][:], f["face_offset"][:], f["row"][:]
            if wanted is None:
                keep = np.arange(len(src_rows))
                V, F = f["verts"][:], f["faces"][:]
                slice_of = lambda i: (V[vo[i]:vo[i + 1]], F[fo[i]:fo[i + 1]])  # noqa: E731
            else:
                keep = np.flatnonzero(np.isin(src_rows, np.fromiter(wanted, np.int64, len(wanted))))
                # Read only the kept rows. `verts`/`faces` are ~1.3 GB for BuildingWorld, so a
                # sampled call must not materialise them whole either.
                sliced = {int(i): (f["verts"][vo[i]:vo[i + 1]], f["faces"][fo[i]:fo[i + 1]])
                          for i in keep}
                slice_of = sliced.__getitem__
        for i in keep:
            i = int(i)
            r = src_rows[i]
            v, fa = slice_of(i)
            # Guarantee OUTWARD normals: a vecset encoder consumes them, and the Frame-N y/z swap is a
            # reflection that silently inverted every stored mesh. Cheap to assert here, so corpora
            # written before that was understood stay usable.
            #
            # ⚠️ This is the reason `rows` filters ABOVE rather than here: it is a mesh build per
            # row, and #188 keeps 60,000 of BuildingWorld's 1,526,778. Filtering after this point
            # would cost exactly as much as not filtering at all.
            import trimesh as _tm
            if _tm.Trimesh(np.asarray(v, np.float64), np.asarray(fa), process=False).volume < 0:
                fa = fa[:, ::-1]
            out[int(r)] = (v, fa, src)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24, help="held-out buildings (stratified by source)")
    ap.add_argument("--n_coarse", type=int, default=8192)
    ap.add_argument("--n_sharp", type=int, default=8192)
    ap.add_argument("--chunk", type=int, default=32768)
    ap.add_argument("--out_dir", default="outputs/dora_frozen_gate")
    ap.add_argument("--sweep", action="store_true",
                    help="ablate the sampler: does our coarse/sharp split explain the degradation?")
    ap.add_argument("--codec", choices=["dora", "triposg"], default="dora",
                    help="which frozen codec to gate; both satisfy encode/decode-at-query-points")
    ap.add_argument("--no_sharp", action="store_true",
                    help="feed a second UNIFORM stream instead of the sharp-edge stream")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    import h5py, trimesh

    # `held` only ever holds rows < FROZEN_SPLIT_N_TOTAL, so buildingworld (all rows appended after
    # that prefix, #175) could never contribute a pick below regardless -- scoping this call to the
    # three historical sources skips ~1.5M rows' worth of verts/faces and winding checks for a
    # source this gate structurally cannot select from (code-review finding on #175).
    surf = load_surfaces(sources=[s for s in SOURCES if s != "buildingworld"])
    with open_real_corpus(H5) as f:
        held = [int(i) for i in test_indices(FROZEN_SPLIT_N_TOTAL)]
    # stratify: take round-robin across sources so bag3d/nrw/plateau are all represented. `held`
    # only ever holds rows < FROZEN_SPLIT_N_TOTAL, so buildingworld (all rows appended after that
    # prefix, #175) never contributes a pick here -- registering it in SOURCES only makes
    # load_surfaces() findable for OTHER consumers, not this gate's own held-out sample.
    by_src = {s: [r for r in held if r in surf and surf[r][2] == s] for s in SOURCES}
    print("held-out with surfaces per source:", {SOURCES[s]: len(v) for s, v in by_src.items()})
    picks, i = [], 0
    while len(picks) < args.n and any(len(v) > i for v in by_src.values()):
        for s in SOURCES:
            if len(by_src[s]) > i and len(picks) < args.n:
                picks.append(by_src[s][i])
        i += 1

    if args.codec == "dora":
        model = load_dora(dev)

        def encode_query(coarse, sharp, q):
            _, kl, _ = model.encode(coarse, sharp, sample_posterior=False)
            lat = model.decode(kl)
            return torch.cat([model.query(q[:, j:j + args.chunk], lat).float()
                              for j in range(0, q.shape[1], args.chunk)], dim=1), kl
    else:
        # TripoSG's VAE is a separable, MIT-licensed diffusers module whose decode() already takes
        # the query points -- the same seam, single-stream rather than dual (no sharp branch).
        # TripoSG also imports torch_cluster.fps (to subsample encoder tokens) and it is absent
        # here, so register the same pure-torch FPS the Dora path uses before importing.
        _stub_absent_deps()
        sys.path.insert(0, str(REPO / "external/TripoSG"))
        from triposg.models.autoencoders.autoencoder_kl_triposg import TripoSGVAEModel
        model = TripoSGVAEModel.from_pretrained(str(TRIPO_VAE)).eval().to(dev)
        print(f"[triposg] loaded  params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

        def encode_query(coarse, sharp, q):
            x = torch.cat([coarse, sharp], dim=1)          # one stream; keep the same point budget
            z = model.encode(x).latent_dist.mode()
            return torch.cat([model.decode(z, q[:, j:j + args.chunk]).sample.float().squeeze(-1)
                              for j in range(0, q.shape[1], args.chunk)], dim=1), z

    pts = grid_points()
    rows = []

    with open_real_corpus(H5) as f:
        for k, r in enumerate(picks):
            v, fc, src = surf[r]
            gt = np.asarray(f["sdf"][r], np.float32)
            rec = {"row": r, "source": SOURCES[src]}
            rec["gt"] = _rough(gt)

            # CONTROL: the recovered surface itself, no codec involved
            rec["input"] = _rough(_revoxel(v, fc, pts))

            mesh = trimesh.Trimesh(np.asarray(v, np.float64), np.asarray(fc), process=False)
            sh = sample_surface if args.no_sharp else sample_sharp_edges
            coarse = torch.from_numpy(sample_surface(mesh, args.n_coarse, rng))[None].to(dev)
            sharp = torch.from_numpy(sh(mesh, args.n_sharp, rng))[None].to(dev)
            with torch.no_grad():
                q = torch.from_numpy(pts.astype(np.float32))[None].to(dev)
                vals, _lat = encode_query(coarse, sharp, q)
            # Sign convention differs BY CODEC and was determined empirically from occupancy
            # agreement with GT, not assumed: Dora returns positive-inside (needs negating),
            # TripoSG is already negative-inside like ours. Getting this wrong inverts the shape
            # and shows up as occupancy jumping to ~0.85.
            field = vals.view(RES, RES, RES).cpu().numpy()
            if args.codec == "dora":
                field = -field

            dv, df = mesh_sdf_surface(np.clip(field, -TRUNC, TRUNC))
            if dv is None:
                rec["frozen"] = float("nan")
                print(f"[{k}] row {r} ({rec['source']}) decode had no zero crossing")
            else:
                rec["frozen"] = _rough(_revoxel(verts_to_world(dv), df, pts))
            rec["occ_gt"] = float((gt <= 0).mean())
            rec["occ_frozen"] = float((field <= 0).mean())
            rows.append(rec)
            print(f"[{k:2d}] row {r:5d} {rec['source']}  gt={rec['gt']:.5f}  "
                  f"input={rec['input']:.5f}  frozen={rec['frozen']:.5f}  "
                  f"occ {rec['occ_gt']:.3f}->{rec['occ_frozen']:.3f}", flush=True)

    ok = [r for r in rows if np.isfinite(r["frozen"])]
    agg = {k: float(np.mean([r[k] for r in ok])) for k in ("gt", "input", "frozen")}
    print(f"\n=== FROZEN GATE [{args.codec}] (n={len(ok)}, real LoD2 surfaces, stratified) ===")
    print(f"  GT (stored field)          {agg['gt']:.5f}")
    print(f"  input surface  [CONTROL]   {agg['input']:.5f}   (codec not involved)")
    print(f"  FROZEN codec               {agg['frozen']:.5f}")
    print(f"  --- references ---")
    print(f"  refiner/corrector wall     {REF['refiner_wall']:.5f}")
    print(f"  map-#24 deployed           {REF['map24_sample']:.5f}")
    print(f"  earlier confounded smoke   0.00839  (grid-derived input mesh)")
    codec_delta = agg["frozen"] - agg["input"]
    print(f"\n  codec contribution = frozen - input = {codec_delta:+.5f}")
    print(f"  beats deployed: {'YES' if agg['frozen'] < REF['map24_sample'] else 'NO'}   "
          f"clears 0.0047 wall: {'YES' if agg['frozen'] < REF['refiner_wall'] else 'NO'}")
    json.dump({"rows": rows, "mean": agg, "codec_delta": codec_delta, "reference": REF},
              open(out / "gate.json", "w"), indent=2)
    print(f"  -> {out/'gate.json'}")


if __name__ == "__main__":
    main()
