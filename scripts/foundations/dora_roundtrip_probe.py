"""Spec-#67 step 1+3 (n=1 smoke): load the pretrained Dora-VAE and round-trip one of OUR buildings.

Discharges "weights retrievable" past mere download -- it instantiates the autoencoder, encodes a real
LoD2 building from `real.h5`, and decodes it back through the QUERY interface
(`query(points, latents)`), which is the seam spec #67 is built on.

Two pieces of glue are ours, because Dora's own preprocessing needs Blender (`bpy`) which we don't have:
  * `sample_surface` -- uniform surface points + normals (their "coarse" stream)
  * `sample_sharp_edges` -- points along high-dihedral-angle edges, normal = mean of the two adjacent
    face normals (their "sharp" stream; matches sharp_sample.py's 0.5*n1 + 0.5*n2)
This is Seam 2 of the spec in embryo, deliberately kept pure and CPU-only.

Frames: real.h5 grids span [-1,1]^3 at spacing 2/63; Dora expects a shape normalised into [-1,1]. We
normalise the extracted mesh onto that box, query the SAME grid mapped through that normalisation, then
divide returned distances by the scale so the field returns to real.h5 units and roughness stays
comparable to GT 0.0041 / codec 0.0044 / wall 0.0047 / deployed 0.00552.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
DORA = REPO / "external/Dora/pytorch_lightning"
if str(DORA) not in sys.path:
    sys.path.insert(0, str(DORA))

from scripts.foundations.baseline_gate_eval import mesh_sdf_surface      # noqa: E402
from scripts.foundations.refiner_prototype import surface_roughness      # noqa: E402
from scripts.foundations.vecset_ceiling_probe import (                   # noqa: E402
    RES, TRUNC, REF, grid_points, verts_to_world, mesh_to_sdf, test_indices,
)

CKPT = REPO / "external/dora_vae_1_1.ckpt"
H5 = REPO / "data/real_massing_v1/real.h5"


def sample_surface(mesh, n: int, rng) -> np.ndarray:
    """Uniform surface points with outward normals -> [n, 6]. Dora's 'coarse' stream."""
    import trimesh
    pts, fid = trimesh.sample.sample_surface(mesh, n)
    nrm = mesh.face_normals[fid]
    return np.concatenate([pts, nrm], axis=1).astype(np.float32)


def sample_sharp_edges(mesh, n: int, rng, deg: float = 25.0) -> np.ndarray:
    """Points along sharp edges with the mean of the two adjacent face normals -> [n, 6].

    Dora's 'sharp' stream. Their pipeline selects these in Blender; face-adjacency dihedral angle is
    the same criterion without the dependency. Falls back to uniform sampling on a mesh with no sharp
    edges, so a smooth blob still encodes rather than crashing.
    """
    ang = mesh.face_adjacency_angles
    keep = ang > np.deg2rad(deg)
    if not keep.any():
        return sample_surface(mesh, n, rng)
    e = mesh.face_adjacency_edges[keep]                       # [E,2] vertex indices
    fn = mesh.face_normals[mesh.face_adjacency[keep]]          # [E,2,3] the two adjacent faces
    en = fn.mean(axis=1)
    en /= np.linalg.norm(en, axis=1, keepdims=True).clip(1e-9)
    a, b = mesh.vertices[e[:, 0]], mesh.vertices[e[:, 1]]
    w = np.linalg.norm(b - a, axis=1)                          # sample longer edges more often
    idx = rng.choice(len(e), size=n, p=w / w.sum())
    t = rng.random((n, 1))
    return np.concatenate([a[idx] * (1 - t) + b[idx] * t, en[idx]], axis=1).astype(np.float32)


def _stub_absent_deps() -> None:
    """Satisfy imports `craftsman` needs but this environment lacks, without installing a training
    stack we never run. Audited: across craftsman's systems/utils/data/models the only absent
    third-party imports are pytorch_lightning, lightning, apex, wandb, timm, diso and torch_cluster.

    All but the last are inert here -- lightning/apex/wandb are trainer, mixed-precision and
    experiment-logging machinery; timm's `Attention` is imported and never used; diso is their
    differentiable iso-surfacer, referenced only in a mesh-extraction path we replace with the repo's
    own level-0.0 meshing helper. `torch_cluster.fps` is the exception: it IS on the encoder's
    forward path, so it is implemented rather than stubbed (see below).
    """
    import importlib.machinery
    import types

    def _inert(name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (object,), {})

    def _mk(dotted: str):
        """Register a permissive stub module, binding it into its parent so `from a.b import C` works."""
        if dotted in sys.modules:
            return sys.modules[dotted]
        m = types.ModuleType(dotted)
        m.__getattr__ = _inert
        m.__path__ = []
        # diffusers probes optional deps with importlib.util.find_spec, which raises on a module
        # whose __spec__ is None -- so give stubs a real spec. Version lookup then fails cleanly
        # via PackageNotFoundError and the dep is simply reported absent, which is the truth.
        m.__spec__ = importlib.machinery.ModuleSpec(dotted, None)
        sys.modules[dotted] = m
        if "." in dotted:
            parent, child = dotted.rsplit(".", 1)
            setattr(_mk(parent), child, m)
        return m

    for dotted in ("pytorch_lightning.utilities.rank_zero", "pytorch_lightning.loggers",
                   "lightning", "apex", "wandb",
                   "timm.models.vision_transformer", "diso"):
        _mk(dotted)

    rz = sys.modules["pytorch_lightning.utilities.rank_zero"]
    rz.rank_zero_only = lambda fn: fn
    rz.rank_zero_debug = lambda *a, **k: None
    rz.rank_zero_info = lambda *a, **k: None

    # torch_cluster.fps -- REAL, not stubbable. With use_downsample=true the encoder chooses its
    # query points by farthest-point sampling. torch_cluster needs compiling against this torch
    # build, so implement FPS directly. Difference from upstream: torch_cluster defaults to a random
    # start, this starts at index 0 -- deterministic, and an equally valid farthest-point subset.
    if "torch_cluster" not in sys.modules:
        def _fps(pos: torch.Tensor, batch: torch.Tensor, ratio: float) -> torch.Tensor:
            out = []
            for b in torch.unique(batch, sorted=True):
                m = (batch == b).nonzero(as_tuple=True)[0]
                q = pos[m]
                n = max(1, int(np.ceil(ratio * q.shape[0])))
                sel = torch.zeros(n, dtype=torch.long, device=pos.device)
                d = torch.full((q.shape[0],), float("inf"), device=pos.device)
                far = torch.zeros((), dtype=torch.long, device=pos.device)
                for i in range(n):
                    sel[i] = far
                    d = torch.minimum(d, (q - q[far]).pow(2).sum(-1))
                    far = torch.argmax(d)
                out.append(m[sel])
            return torch.cat(out)
        tc = types.ModuleType("torch_cluster")
        tc.fps = _fps
        sys.modules["torch_cluster"] = tc


def load_dora(device: str):
    """Instantiate the autoencoder from the released config and load the checkpoint."""
    _stub_absent_deps()
    from craftsman.models.autoencoders.michelangelo_autoencoder import MichelangeloAutoencoder
    cfg = dict(pretrained_model_name_or_path=str(CKPT), embed_dim=64, point_feats=3, out_dim=1,
               embed_type="fourier", num_freqs=8, include_pi=False, heads=12, width=768,
               num_encoder_layers=8, num_decoder_layers=16, use_ln_post=True, init_scale=0.25,
               qkv_bias=False, use_flash=False, use_checkpoint=False, use_downsample=True)
    # torch>=2.6 defaults torch.load to weights_only=True and their loader doesn't pass it. The
    # checkpoint stores omegaconf hyperparameters containing `typing.Any`, which torch's allowlist
    # CANNOT express -- add_safe_globals rejects it because a typing._SpecialForm has no
    # __qualname__. So scope the exception to this single construction call.
    # Provenance: fetched over HTTPS from the official Seed3D/Dora-VAE-1.1 HF repo (public, ungated,
    # apache-2.0), verified as a torch zip archive before use.
    _orig_load = torch.load

    def _load_trusted(*a, **k):
        k.setdefault("weights_only", False)
        return _orig_load(*a, **k)

    torch.load = _load_trusted
    try:
            model = MichelangeloAutoencoder(cfg)
    finally:
        torch.load = _orig_load
    # `split` is normally injected by their Lightning system; standalone we pick the eval branch,
    # which uses fixed rather than randomly-sampled downsample ratios.
    model.split = "val"
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[dora] instantiated + loaded  params={n_par/1e6:.1f}M")
    return model.eval().to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1, help="held-out buildings")
    ap.add_argument("--n_coarse", type=int, default=8192)
    ap.add_argument("--n_sharp", type=int, default=8192)
    ap.add_argument("--chunk", type=int, default=32768)
    ap.add_argument("--out_dir", default="outputs/dora_roundtrip")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    import h5py, trimesh

    model = load_dora(dev)
    pts_world = grid_points()                                   # [-1,1]^3, index order

    with h5py.File(H5, "r") as f:
        idxs = test_indices(int(f["sdf"].shape[0]))[:args.n]
        gts = [np.asarray(f["sdf"][int(i)], np.float32) for i in idxs]

    for k, (gi, gt) in enumerate(zip(idxs, gts)):
        r_gt = surface_roughness(torch.from_numpy(np.clip(gt, -TRUNC, TRUNC)))
        v, fc = mesh_sdf_surface(np.clip(gt, -TRUNC, TRUNC))
        if v is None:
            print(f"[{k}] #{gi} no zero crossing, skipped"); continue
        mesh = trimesh.Trimesh(verts_to_world(v), fc, process=False)

        # normalise onto [-1,1] the way Dora expects, and remember the scale
        c = (mesh.vertices.min(0) + mesh.vertices.max(0)) / 2.0
        s = 0.95 / max(float(np.abs(mesh.vertices - c).max()), 1e-9)
        mesh.vertices = (mesh.vertices - c) * s

        coarse = torch.from_numpy(sample_surface(mesh, args.n_coarse, rng))[None].to(dev)
        sharp = torch.from_numpy(sample_sharp_edges(mesh, args.n_sharp, rng))[None].to(dev)
        print(f"[{k}] #{gi}  mesh V={len(v):,}  coarse={tuple(coarse.shape)} sharp={tuple(sharp.shape)}")

        with torch.no_grad():
            _, kl, _ = model.encode(coarse, sharp, sample_posterior=False)
            lat = model.decode(kl)
            q = torch.from_numpy(((pts_world - c) * s).astype(np.float32))[None].to(dev)
            vals = torch.cat([model.query(q[:, i:i + args.chunk], lat).float()
                              for i in range(0, q.shape[1], args.chunk)], dim=1)
        field = vals.view(RES, RES, RES).cpu().numpy() / s      # back to real.h5 units

        print(f"     latent={tuple(kl.shape)}  field range [{field.min():.3f}, {field.max():.3f}]  "
              f"frac<=0 {float((field<=0).mean()):.3f}  (GT frac<=0 {float((gt<=0).mean()):.3f})")
        for sign, tag in ((1.0, "as-is"), (-1.0, "flipped")):
            fl = np.clip(field * sign, -TRUNC, TRUNC)
            occ = float((fl <= 0).mean())
            rr = surface_roughness(torch.from_numpy(fl))
            print(f"     {tag:8s} occ={occ:.3f}  roughness={rr:.5f}")
        # Their field is a TSDF normalised to ~[-1,1] over a narrow band, NOT metric distance like
        # ours -- so its raw Laplacian is not comparable to GT. Convert it to our units the same way
        # #63 did for the teacher: mesh at 0 and re-voxelize through the igl signed-distance path,
        # whose control arm was already shown to add no roughness of its own.
        dv, df = mesh_sdf_surface(np.clip(field * -1.0, -TRUNC, TRUNC))
        if dv is None:
            print("     [!] decoded field has no usable zero crossing")
        else:
            dsdf = mesh_to_sdf(verts_to_world(dv), df, pts_world)
            r_dora = surface_roughness(torch.from_numpy(np.clip(dsdf, -TRUNC, TRUNC)))
            print(f"     RE-VOXELIZED  roughness={r_dora:.5f}   vs GT {r_gt:.5f}   "
                  f"(deployed {REF['map24_sample']}, wall {REF['refiner_wall']})")
        np.save(out / f"{k:02d}_dora_field.npy", field)
        print(f"     GT roughness={r_gt:.5f}   refs: floor {REF['gt_floor']}, "
              f"wall {REF['refiner_wall']}, deployed {REF['map24_sample']}")


if __name__ == "__main__":
    main()
