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
SOURCES = {"bag3d": "NL", "nrw": "DE", "plateau": "JP"}


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


def load_surfaces():
    """row -> (verts, faces) for every recovered building, across all three sources."""
    import h5py
    out = {}
    for src in SOURCES:
        p = SURF / f"surfaces_{src}.h5"
        if not p.exists():
            print(f"[warn] missing {p.name}"); continue
        with h5py.File(p, "r") as f:
            vo, fo, rows = f["vert_offset"][:], f["face_offset"][:], f["row"][:]
            V, F = f["verts"][:], f["faces"][:]
        for i, r in enumerate(rows):
            out[int(r)] = (V[vo[i]:vo[i + 1]], F[fo[i]:fo[i + 1]], src)
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

    surf = load_surfaces()
    with h5py.File(H5, "r") as f:
        held = [int(i) for i in test_indices(int(f["sdf"].shape[0]))]
    # stratify: take round-robin across sources so all three regions are represented
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

    with h5py.File(H5, "r") as f:
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
