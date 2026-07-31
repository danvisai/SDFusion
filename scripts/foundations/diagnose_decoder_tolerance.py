"""#73 -- what is the melt made of? Measures how much latent error the Dora decoder actually tolerates.

The map's melt had three candidate explanations wanting opposite fixes: leftover **noise** (train
longer), **decoder intolerance** (training length is irrelevant), or **weak conditioning**. The
ticket's decisive experiment needed a trained denoiser, and both A2 checkpoints are void (#78).

It does not need one. *"What does a latent this close to the truth decode to?"* is a property of the
**decoder alone**. Two arms answer it, and the answer is in the gap between them:

  * **off-manifold** -- perturb the cached latent with isotropic Gaussian noise to a controlled cosine.
    This is the error model the forward diffusion produces (`SetSDEdit.noise_to` adds isotropic noise,
    reaching cos 0.707 at s=0.5).
  * **on-manifold** -- re-encode the *same mesh* with a fresh surface point sample. FPS then picks a
    different token **ordering**, so the element-wise cosine collapses to ~0.10 while the geometry is
    identical by construction. This is the control that gives the isotropic numbers their meaning.

Reference points, from the first A2 run's in-distribution diagnostic (⚠️ a void checkpoint, so read
them as "roughly what a 1.4-epoch denoiser reaches", not as gospel): the denoiser recovered to cos
0.995 / 0.989 / 0.980 / 0.935 at strength 0.1 / 0.2 / 0.3 / 0.5.

⚠️ Two traps this script is built around:
  * **Use independent noise per cosine.** Reusing one seed across the sweep walks a single random ray
    through latent space, and a 1-D slice is wildly non-monotonic -- an early version of this script
    reported the decode *improving* from cos 0.935 to cos 0.707 purely as an artifact of that.
    `--repeats` samples independent directions and reports the median with its spread.
  * **Ribbing is not melt** (#71). A Dora decode ribs at *every* cosine including 1.000, because its
    field is ~32x too steep for marching cubes to place the surface within a voxel. Ribbing is
    constant along a montage row, which is what makes these panels comparable.

Run:
    diagnose_decoder_tolerance.py --n 16 --repeats 3 --montage 5
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.foundations.eval_massing_arms import (  # noqa: E402
    H5, LATENTS, RES, build_montage, pick_ids, score_arm, summarise, volume_split,
)

COSINES = [0.999, 0.995, 0.980, 0.935, 0.900, 0.800, 0.707]
BASELINE = REPO / "execution/artifacts/massing_arms_eval_baseline.json"


def sigma_for_cosine(z: np.ndarray, c: float) -> float:
    """Noise scale sigma such that cos(z, z + sigma*eps) ~= c, for eps ~ N(0, I).

    With D = z.size and eps isotropic, <z, z+sigma eps> -> |z|^2 and |z+sigma eps|^2 -> |z|^2 +
    sigma^2 D, so cos -> |z| / sqrt(|z|^2 + sigma^2 D). Inverting gives the expression below. At
    D = 2048*64 = 131,072 the concentration is tight enough that the achieved cosine lands within
    ~1e-3 of the target -- measured and reported rather than assumed.
    """
    if c >= 1.0:
        return 0.0
    return float(np.linalg.norm(z) / np.sqrt(z.size) * np.sqrt(1.0 / (c * c) - 1.0))


def perturb(z: np.ndarray, c: float, seed: int) -> tuple[np.ndarray, float]:
    """-> (perturbed latent, achieved cosine), as a *z0 estimate* rather than a noised z_t.

    A denoiser outputs an estimate of the clean latent, so its error is a deviation from z0 at roughly
    unchanged norm -- additive noise without the forward process's sqrt(alpha_bar) shrinkage.

    `seed` must differ per cosine, or the sweep traverses one fixed random direction; see the module
    docstring.
    """
    if c >= 1.0:
        return z.copy(), 1.0
    rng = np.random.default_rng(seed)
    zp = z + sigma_for_cosine(z, c) * rng.standard_normal(z.shape).astype(np.float32)
    return zp.astype(np.float32), cosine(z, zp)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a.ravel(), b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16, help="buildings to score")
    ap.add_argument("--repeats", type=int, default=3, help="independent noise directions per cosine")
    ap.add_argument("--montage", type=int, default=5)
    ap.add_argument("--size", type=int, default=300)
    ap.add_argument("--latents", default=str(LATENTS))
    ap.add_argument("--ids_from", default=str(BASELINE) if BASELINE.exists() else None,
                    help="score the SAME buildings the #71 baseline did, so rows are comparable")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    import h5py
    import torch
    from models.shape_codec import Building, DoraCodec
    from scripts.foundations.dora_frozen_gate import load_surfaces
    from scripts.foundations.dora_roundtrip_probe import load_dora
    from scene.surface_sampling import to_array_frame

    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    cand, lat_of = pick_ids(Path(args.latents), args.ids_from)
    ids = cand[:args.n]
    print(f"[ids] {len(ids)} buildings"
          f"{' (pinned to the #71 baseline)' if args.ids_from else ''}", flush=True)

    # Normalisation measured from THIS cache, not lifted from a void checkpoint: those stored mu/sd
    # were computed over transposed latents, i.e. over different numbers.
    with h5py.File(args.latents, "r") as f:
        probe = np.asarray(f["latent"][:600], np.float32)
        mu, sd = float(probe.mean()), float(probe.std())
    print(f"[stats] mu={mu:.6f} sd={sd:.6f} (600 rows of this cache)", flush=True)

    codec = DoraCodec(load_dora(dev))
    surf = load_surfaces()

    arms = ["perfect", "reencoded"] + [f"iso_cos{c:.3f}" for c in COSINES]
    scores: dict = {a: {} for a in arms}
    fields: dict = {}
    cos_seen: dict = {a: [] for a in arms}
    vs_perfect: dict = {a: [] for a in arms}

    def decode(zn: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return codec.decode_grid(
                torch.from_numpy(zn * sd + mu)[None].to(dev), RES).cpu().numpy()[0, 0]

    with h5py.File(args.latents, "r") as lf, h5py.File(H5, "r") as gt:
        for k, bid in enumerate(ids):
            gocc = np.asarray(gt["sdf"][bid], np.float32) <= 0
            fp = np.asarray(lf["footprint"][lat_of[bid]])
            zn = (np.asarray(lf["latent"][lat_of[bid]], np.float32) - mu) / sd

            perfect = decode(zn)
            pocc = perfect <= 0
            scores["perfect"][bid] = score_arm(perfect, gocc, fp)
            cos_seen["perfect"].append(1.0)
            vs_perfect["perfect"].append(1.0)
            if len(fields) < args.montage:
                fields[bid] = {"perfect": perfect}

            # on-manifold control: same mesh, fresh point sample -> different token ORDER, same shape
            if bid in surf:
                v, fc, _ = surf[bid]
                av, af = to_array_frame(v, fc)
                codec.rng = np.random.default_rng(args.seed * 1000003 + bid)
                z2 = (codec.encode(Building(verts=av, faces=af)).float().cpu().numpy()[0] - mu) / sd
                fld = decode(z2)
                scores["reencoded"][bid] = score_arm(fld, gocc, fp)
                cos_seen["reencoded"].append(cosine(zn, z2))
                vs_perfect["reencoded"].append(volume_split(fld <= 0, pocc)["vol_iou"])
                if bid in fields:
                    fields[bid]["reencoded"] = fld

            for c in COSINES:
                arm = f"iso_cos{c:.3f}"
                rows, cs, vps = [], [], []
                for rep in range(args.repeats):
                    # Independent direction per (building, cosine, repeat) -- see module docstring.
                    # Built arithmetically, not with hash(): Python randomises string hashing per
                    # process, which would make the sweep unreproducible in exactly the way #71 just
                    # finished fixing elsewhere.
                    zp, cos = perturb(zn, c, seed=(args.seed * 1000003 + bid * 9973
                                                   + int(round(c * 1000)) * 13 + rep))
                    fld = decode(zp)
                    rows.append(score_arm(fld, gocc, fp)); cs.append(cos)
                    vps.append(volume_split(fld <= 0, pocc)["vol_iou"])
                    if rep == 0 and bid in fields:
                        fields[bid][arm] = fld
                scores[arm][bid] = {kk: float(np.median([r[kk] for r in rows])) for kk in rows[0]}
                cos_seen[arm].append(float(np.mean(cs)))
                vs_perfect[arm] += vps
            print(f"  [{k+1}/{len(ids)}] row {bid}", flush=True)

    summary = {a: summarise(scores[a].values()) for a in arms}
    for a in arms:
        if summary[a]:
            summary[a]["cos_to_true_latent"] = float(np.mean(cos_seen[a]))
            summary[a]["vol_iou_vs_perfect_decode"] = float(np.median(vs_perfect[a]))
            summary[a]["vs_perfect_p10"] = float(np.percentile(vs_perfect[a], 10))
            summary[a]["vs_perfect_p90"] = float(np.percentile(vs_perfect[a], 90))

    print(f"\n=== DECODER TOLERANCE (n={len(ids)}, {args.repeats} directions per cosine) ===")
    print(f"{'latent':16s} {'cos':>7} {'vs perfect':>11} {'p10..p90':>15} "
          f"{'fp-IoU':>8} {'missing':>9} {'extra':>8}")
    for a in arms:
        s = summary[a]
        if not s:
            continue
        print(f"{a:16s} {s['cos_to_true_latent']:>7.3f} {s['vol_iou_vs_perfect_decode']:>11.3f} "
              f"{s['vs_perfect_p10']:>7.3f}..{s['vs_perfect_p90']:<7.3f} "
              f"{s['fp_iou']:>8.3f} {s['missing']:>9.3f} {s['extra']:>8.3f}")
    print("\n🔑 'cos' is element-wise cosine to the cached latent. Compare `reencoded` (same geometry,"
          "\n   different token order) against the iso_* rows: cosine does not predict decode quality.")

    suffix = f"_{args.tag}" if args.tag else ""
    if args.montage and fields:
        p = build_montage(fields, arms, scores, summary,
                          REPO / f"docs/wayfinding/vecset-convergence/"
                                 f"decoder-tolerance-montage{suffix}.png", args.size)
        print(f"\nmontage: {p}", flush=True)

    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO),
                         capture_output=True, text=True).stdout.strip()
    art = REPO / f"execution/artifacts/decoder_tolerance{suffix}.json"
    art.parent.mkdir(parents=True, exist_ok=True)
    art.write_text(json.dumps(dict(
        meta=dict(git_rev=rev, created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  n=len(ids), repeats=args.repeats, latents=args.latents, mu=mu, sd=sd,
                  seed=args.seed, ids_from=args.ids_from,
                  note="cosines are element-wise, in the NORMALISED latent space the denoiser works in"),
        ids=ids, cosines=COSINES, summary=summary,
        per_building={a: {str(b): r for b, r in scores[a].items()} for a in arms},
    ), indent=2))
    print(f"artifact: {art}", flush=True)


if __name__ == "__main__":
    main()
