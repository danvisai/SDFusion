"""#71 -- THE evaluation harness map #69 is judged on. One script, one id set, every arm.

Supersedes `eval_vecset_projection.py` (A2 only, with map-#24 as a hardcoded constant measured on a
different sample) and `render_a2_comparison.py` (the montage, on its own ad-hoc slice). Those two
produced numbers that were routinely printed side by side and were **not comparable**: different n,
different held-out buildings, one of them not measured by any committed code at all. Everything scored
here is scored on the *same* fixed ids, in the same pass, into one artifact.

The arms, all optional except GT:

    gt              real LoD2 from real.h5                      -- the target, and the identity control
    blockout        signed-EDT extrusion of the footprint       -- "did this beat doing nothing?"
    codec_ceiling   the cached real latent, decoded             -- the best any generator over this
                                                                   codec could reach
    deployed_map24  the shipped dense-grid Stage3a prior        -- what we ship today
    a2_s<strength>  a vecset projection checkpoint              -- the candidate

What it reports, in the map's priority order:

  1. **The montage** (`--montage N`) is the primary output. Shaded renders, arms as columns, meshed at
     continuous-SDF level 0.0, **one fixed camera in the shared world frame** -- no per-mesh
     normalisation, so an arm that lost volume renders visibly smaller instead of being silently
     rescaled to look like GT.
  2. **fp-IoU** against the conditioning footprint. The hard criterion.
  3. **3D IoU split into `missing` and `extra`**, both as a fraction of GT volume, never a lone
     aggregate -- one number cannot tell "correctly carved the blockout's over-fill" from "ate the
     building", and those want opposite responses. The aggregate is still logged, as a diagnostic.

⚠️ `surface_roughness` is logged under a `guard_` prefix and appears in **no ranking**. It is
anti-correlated with the goal in this domain -- it ranks a melted blob above a crisp ribbed box
(`docs/wayfinding/crisp-massing-vecset/deployed-vs-dora.md`). Regression guard only, and comparable
for one arm across runs but **not between arms**: it is a raw |Laplacian|, so it scales with the
field's own slope, and the arms do not share one. `guard_field_slope` is logged beside it to keep that
visible -- a metric SDF on this grid is 0.032, Dora's decoded TSDF measures ~1.01.

Run:
    eval_massing_arms.py --n 48 --tag baseline
    eval_massing_arms.py --n 48 --a2 logs_building/vecset_v2/vecset_denoiser.pth --strength 0.5 \
                         --ids_from execution/artifacts/massing_arms_eval_baseline.json --tag v2
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.foundations.baseline_gate_eval import fp_iou, mesh_sdf_surface  # noqa: E402
from scripts.foundations.vecset_ceiling_probe import RES, TRUNC, verts_to_world  # noqa: E402

H5 = REPO / "data/real_massing_v1/real.h5"
LATENTS = REPO / "data/real_massing_v1/vecset_latents.h5"
MAP24 = REPO / "logs_building/2026-07-16-stage3a-lod2-fromscratch-region/ckpt/stage3a_steps-latest.pth"

# One camera for every panel of every arm of every building, so panels are comparable by eye.
CAM = dict(dist=2.5, elev=20.0, azim=35.0, scale=0.72)


def blockout_sdf(fp: np.ndarray, y0: int, y1: int):
    """Footprint mask + vertical extent -> a crisp extruded-prism SDF on the corpus grid.

    This is the analytic extrusion prior #53 measured as crisp (0.0035 roughness) and footprint-exact
    (IoU 0.96) -- exactly the "crude blockout" ADR 0003 says generation should project FROM. Signed
    EDT rather than marching a binary mask, so the surface is not pre-staircased.
    """
    from scipy import ndimage
    occ = np.zeros((RES, RES, RES), bool)                 # array axes are [z, y, x]
    occ[:, y0:y1 + 1, :] = fp.astype(bool)[:, None, :]
    if not occ.any():
        return None
    inside = ndimage.distance_transform_edt(occ)
    outside = ndimage.distance_transform_edt(~occ)
    return ((outside - inside) * (2.0 / (RES - 1))).astype(np.float32)


def _vertical_extent(gt_occ: np.ndarray):
    """(y0, y1) of the GT's occupied slab along H (axis 1), or None if the building is empty."""
    ys = np.nonzero(np.asarray(gt_occ).any(axis=(0, 2)))[0]
    return (int(ys.min()), int(ys.max())) if len(ys) else None


# --------------------------------------------------------------------------------------------------
# scoring -- pure numpy, no torch, so it is testable without a GPU
# --------------------------------------------------------------------------------------------------

def volume_split(arm_occ: np.ndarray, gt_occ: np.ndarray) -> dict:
    """3D agreement as **missing vs extra**, each a fraction of GT volume, plus the aggregate IoU.

    The aggregate alone is unreadable: a blockout that over-fills by 22% and a generator that ate 22%
    of the building can land on the same IoU while wanting opposite responses. Normalising both by GT
    volume (not by the union) is what makes "the blockout over-fills by +21.7% with 0% missing"
    reproducible and directly comparable across arms of different total size.
    """
    a, g = np.asarray(arm_occ, bool), np.asarray(gt_occ, bool)
    gv, av = int(g.sum()), int(a.sum())
    inter, union = int((a & g).sum()), int((a | g).sum())
    return dict(
        vol_iou=float(inter / union) if union else 0.0,
        missing=float((gv - inter) / gv) if gv else 0.0,   # GT volume the arm failed to fill
        extra=float((av - inter) / gv) if gv else 0.0,     # volume the arm added outside GT
        arm_vox=av, gt_vox=gv,
    )


def field_slope(field: np.ndarray, band_sigma: float = 0.05) -> float:
    """Mean |grad f| in a band around the field's own 0-level, weighted exactly as roughness is.

    Records the **scale** of an arm's field, and it is not decoration: `surface_roughness` is a raw
    |Laplacian|, so it scales with the field's slope. A metric SDF on this grid has slope 2/63 = 0.032;
    Dora's decoded TSDF measures ~1.01, i.e. **32x steeper**, and its roughness is inflated by roughly
    that factor for reasons that have nothing to do with how crisp the geometry is. Logging the slope
    beside the guard is what stops the guard being read across arms as if it were one scale.
    """
    f = np.asarray(field, np.float32)
    mag = np.sqrt(sum(g ** 2 for g in np.gradient(f)))
    w = np.exp(-np.abs(f) / band_sigma)
    return float((w * mag).sum() / max(w.sum(), 1e-8))


def score_arm(field: np.ndarray, gt_occ: np.ndarray, fp: np.ndarray) -> dict:
    """One arm's continuous SDF -> the row this harness ranks on, plus the non-ranking guards."""
    import torch
    from scripts.foundations.refiner_prototype import surface_roughness
    occ = np.asarray(field) <= 0
    row = dict(fp_iou=fp_iou(occ, fp))
    row.update(volume_split(occ, gt_occ))
    # guards only -- kept out of fp_iou/missing/extra and out of every ranking print
    row["guard_roughness"] = surface_roughness(
        torch.from_numpy(np.clip(np.asarray(field, np.float32), -TRUNC, TRUNC)))
    row["guard_field_slope"] = field_slope(field)
    return row


def summarise(rows) -> dict:
    """Per-arm medians. `guard_roughness` keeps its prefix so it cannot pass for a criterion."""
    rows = list(rows)
    if not rows:
        return {}
    med = lambda k: float(np.median([r[k] for r in rows]))  # noqa: E731
    out = dict(n=len(rows), fp_iou=med("fp_iou"), missing=med("missing"), extra=med("extra"),
               vol_iou=med("vol_iou"), guard_roughness=med("guard_roughness"))
    if "guard_field_slope" in rows[0]:
        out["guard_field_slope"] = med("guard_field_slope")
    return out


# --------------------------------------------------------------------------------------------------
# rendering -- shared world frame, one fixed camera
# --------------------------------------------------------------------------------------------------

def render_world(verts_w: np.ndarray, faces: np.ndarray, size: int, device):
    """Shaded render of a mesh **already in the [-1,1] world frame**, with a fixed camera.

    Deliberately NOT `scripts/hunyuan_building_mesh_smoke.render_mesh_png`: that recentres and rescales
    every mesh onto its own bounding box, which is right for comparing two unrelated shapes but wrong
    here -- it would scale an eroded or collapsed arm back up to GT's apparent size and hide exactly
    the failure criterion 3 exists to expose. Same lights and camera parameters, no normalisation.
    """
    import torch
    from PIL import Image
    from pytorch3d.renderer import (BlendParams, FoVOrthographicCameras, MeshRasterizer, MeshRenderer,
                                    PointLights, RasterizationSettings, SoftPhongShader, TexturesVertex,
                                    look_at_view_transform)
    from pytorch3d.structures import Meshes

    v = torch.as_tensor(np.asarray(verts_w, np.float32), device=device)[None]
    f = torch.as_tensor(np.asarray(faces, np.int64), device=device)[None]
    r, t = look_at_view_transform(dist=CAM["dist"], elev=CAM["elev"], azim=CAM["azim"], at=((0, 0, 0),))
    cams = FoVOrthographicCameras(device=device, R=r, T=t, scale_xyz=((CAM["scale"],) * 3,))
    lights = PointLights(device=device, location=((2.0, 2.0, 2.0),),
                         ambient_color=((0.45, 0.45, 0.45),), diffuse_color=((0.55, 0.55, 0.55),),
                         specular_color=((0.05, 0.05, 0.05),))
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cams, raster_settings=RasterizationSettings(
            image_size=size, blur_radius=0.0, faces_per_pixel=1, bin_size=0)),
        shader=SoftPhongShader(device=device, cameras=cams, lights=lights,
                               blend_params=BlendParams(background_color=(1.0, 1.0, 1.0))))
    tex = TexturesVertex(verts_features=torch.full_like(v, 0.72))
    img = renderer(Meshes(verts=v, faces=f, textures=tex))[0, ..., :3].clamp(0, 1).cpu().numpy()
    return Image.fromarray((img * 255).astype(np.uint8))


def sharp_normal_error(fields: dict, arm_order, device, views: int = 22, size: int = 256):
    """Dora's SNE, validated for this repo in #79. Lower is better. Returns {arm: mean SNE}.

    Normal maps from `views` directions -> Canny on the **GT** map to find salient regions -> dilate ->
    mean squared normal difference **inside those regions only**.

    🔑 **This is the first scalar in this project that ranks crisp above melted.** #36
    (`separation_ok: False`), #63 ("blind to this artifact class") and `deployed-vs-dora` all failed;
    `surface_roughness` ranks a melted blob ABOVE a crisp ribbed box. Measured in #79 on n=8:
    codec_ceiling (crisp) **0.084** vs deployed_map24 (melted) **0.636**, separated on 8/8 buildings
    with no overlap (crisp max 0.111 < melted min 0.517).

    Why masking rescues it: the ribs live on flat FACES, and the salient mask is a thin EDGE outline
    (~6% of pixels), so face ribbing barely enters the average. Whole-surface roughness drowns in it.

    ⚠️ **Still contaminated across arms, so it is reported, never ranked on.** On a row whose blockout
    occupancy is BYTE-IDENTICAL to GT, SNE is 0.241, not 0 (#79's C2) -- a faceted signed EDT also
    perturbs the edges the mask covers. The offset is not a constant that can be subtracted: the
    codec's own ribbing contaminates far less (0.084) than the EDT's. Safe within one arm across runs;
    across arms read it only for gaps far larger than that offset, as the crisp/melted 7.6x is.
    """
    import torch
    from scipy.ndimage import binary_dilation
    from skimage.feature import canny
    from pytorch3d.renderer import (FoVOrthographicCameras, MeshRasterizer, RasterizationSettings,
                                    look_at_view_transform)
    from pytorch3d.structures import Meshes

    ga = np.pi * (3.0 - np.sqrt(5.0))
    dirs = []
    for i in range(views):
        z = 1.0 - (2.0 * i + 1.0) / views
        r = np.sqrt(max(0.0, 1.0 - z * z))
        dirs.append((float(np.degrees(np.arcsin(np.clip(z, -1, 1)))),
                     float(np.degrees(np.arctan2(r * np.sin(ga * i), r * np.cos(ga * i))))))
    rs = RasterizationSettings(image_size=size, blur_radius=0.0, faces_per_pixel=1, bin_size=0)

    def maps(verts_w, faces):
        v = torch.as_tensor(np.asarray(verts_w, np.float32), device=device)[None]
        f = torch.as_tensor(np.asarray(faces, np.int64), device=device)[None]
        mesh = Meshes(verts=v, faces=f)
        fn = mesh.faces_normals_packed()
        out, hits = [], []
        for elev, azim in dirs:
            r, t = look_at_view_transform(dist=CAM["dist"], elev=elev, azim=azim, at=((0, 0, 0),))
            cams = FoVOrthographicCameras(device=device, R=r, T=t, scale_xyz=((CAM["scale"],) * 3,))
            pix = MeshRasterizer(cameras=cams, raster_settings=rs)(mesh).pix_to_face[0, ..., 0]
            hit = pix >= 0
            nrm = torch.zeros((size, size, 3), device=device)
            if hit.any():
                nrm[hit] = torch.nn.functional.normalize(fn[pix[hit]] @ r[0].to(device), dim=-1)
            out.append(nrm); hits.append(hit)
        return torch.stack(out), torch.stack(hits)

    acc = {a: [] for a in arm_order}
    for bid, per_arm in fields.items():
        gv, gf = mesh_sdf_surface(np.clip(per_arm["gt"], -TRUNC, TRUNC))
        if gv is None:
            continue
        gn, gh = maps(verts_to_world(gv), gf)
        m = []
        for i in range(gn.shape[0]):
            nm, hit = gn[i].cpu().numpy(), gh[i].cpu().numpy()
            e = np.zeros(hit.shape, bool)
            for c in range(3):
                e |= canny(nm[..., c], sigma=2.0)
            m.append(binary_dilation(e & hit, np.ones((3, 3), bool)))
        mask = torch.as_tensor(np.stack(m), device=device)
        if not mask.any():
            continue
        for arm in arm_order:
            fld = per_arm.get(arm)
            if fld is None:
                continue
            # the codec's TSDF is already truncated; clipping a metric SDF matches how it is meshed
            v, f = mesh_sdf_surface(fld if arm == "codec_ceiling" else np.clip(fld, -TRUNC, TRUNC))
            if v is None:
                continue
            an, _ = maps(verts_to_world(v), f)
            acc[arm].append(float((((an - gn) ** 2).sum(-1))[mask].mean()))
    return {a: (float(np.mean(v)) if v else float("nan")) for a, v in acc.items()}


def build_montage(fields: dict, arm_order, scores: dict, summary: dict, out: Path, size: int) -> Path:
    """Criterion 1. Rows are buildings, columns are arms, every panel meshed at continuous SDF 0.0.

    Labelled with fp / missing / extra -- roughness is deliberately absent, because putting it on the
    picture invites exactly the ranking it must never enter.

    ⚠️ The column header carries each arm's **field slope**, because the arms do not share a field
    representation and the difference is *visible*: `real.h5` stores a metric SDF (slope 0.031) and
    renders smooth-walled; the blockout's signed EDT is faceted, so it renders **ribbed even where its
    occupancy is byte-identical to GT**; Dora's decoded TSDF is ~32x too steep for marching cubes to
    localise the surface within a voxel, so it ribs too. Ribbing at equal occupancy is therefore a
    property of the arm's field, not of its geometry -- read the ribs against the slope, not as melt.
    """
    import torch
    from PIL import Image, ImageDraw
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ids = sorted(fields)
    PAD, HDR, LBL = 8, 44, 34
    W = len(arm_order) * size + (len(arm_order) + 1) * PAD
    H = HDR + len(ids) * (size + LBL) + PAD
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    d.text((PAD, 6), f"map #69 evaluation harness (#71)  --  n={len(ids)} of the fixed held-out set  "
                     f"--  labels: fp-IoU / missing / extra (fractions of GT volume)  --  "
                     f"one fixed camera, shared world frame", fill=(0, 0, 0))
    d.text((PAD, 22), "arms do NOT share a field representation: 'slope' is each arm's near-surface "
                      "|grad| (a metric SDF is 0.031). Ribbing at equal occupancy is a meshing "
                      "artifact of the field, not melt.", fill=(150, 60, 60))
    for i, bid in enumerate(ids):
        y = HDR + i * (size + LBL)
        for j, arm in enumerate(arm_order):
            x0 = PAD + j * (size + PAD)
            if i == 0:
                sl = summary.get(arm, {}).get("guard_field_slope")
                d.text((x0 + 3, y + 3), f"{arm}" + (f"   [slope {sl:.3f}]" if sl else ""),
                       fill=(0, 0, 0))
            fld = fields[bid].get(arm)
            if fld is None:
                d.text((x0 + 3, y + 18), "(arm not run)", fill=(150, 60, 60))
                continue
            mv, mf = mesh_sdf_surface(np.clip(fld, -TRUNC, TRUNC))
            if mv is None:
                d.text((x0 + 3, y + 18), "(no zero crossing)", fill=(150, 60, 60))
                continue
            canvas.paste(render_world(verts_to_world(mv), mf, size, dev), (x0, y + LBL))
            s = scores.get(arm, {}).get(bid)
            if s:
                d.text((x0 + 3, y + 18),
                       f"fp {s['fp_iou']:.3f}   miss {s['missing']:.3f}   extra {s['extra']:.3f}",
                       fill=(90, 90, 90))
        d.text((PAD + 3, y + LBL + size - 13), f"row {bid}", fill=(120, 120, 120))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out


# --------------------------------------------------------------------------------------------------

def pick_ids(latents: Path, ids_from):
    """The **one fixed held-out id set**, recorded in the artifact so every later run scores the same
    buildings. Ids are global rows of `real.h5` -- the only identity all arms share (the latent cache
    drops the 153 unrecoverable buildings, so its own row order is not it).

    Returns (candidate ids in ascending order, row -> latent-cache index). `--ids_from` replays a
    previous artifact's ids exactly, which is what makes a later checkpoint's numbers comparable to
    this one's rather than merely similar-looking.
    """
    import h5py
    with h5py.File(latents, "r") as f:
        held = np.nonzero(f["held_out"][:] == 1)[0]
        rows = f["row"][:][held]
        # Caches written before the region column existed still load; they just cannot be stratified.
        region = f["region"][:][held] if "region" in f else None
    lat_of = {int(r): int(h) for r, h in zip(rows, held)}
    if ids_from:
        ids = [int(i) for i in json.loads(Path(ids_from).read_text())["ids"]]
        absent = [i for i in ids if i not in lat_of]
        if absent:
            raise SystemExit(f"[ids] {len(absent)} pinned ids are absent from {latents}: {absent[:5]}")
        return ids, lat_of

    # ⚠️ Round-robin the regions. Ascending row order tracks SOURCE CORPUS, so the plain
    # `sorted(lat_of)` this used to return made the first 48 ids **100% BAG_real (Dutch)** -- zero
    # German, zero Japanese -- while the held-out set is 34.7/32.9/32.4. That is not a small sample of
    # the held-out set, it is a different population: region is the strongest variable here (mean
    # height 11.97/5.90/7.47 m, blockout `extra` median 0.223/0.162/0.000). It void-ed this map's
    # "gap to the blockout closes to 0.007" (really 0.071) and #80's 11.9% surplus reduction, and hid
    # that region predicts #84's collapse (solid rate 38.7/57.4/77.9%). Interleaving keeps any prefix
    # of the list region-balanced, so `--n 48` is now a sample rather than one country.
    if region is None:
        return [int(r) for r in sorted(lat_of)], lat_of
    region_of = {int(r): int(g) for r, g in zip(rows, region)}
    by_region: dict = {}
    for r in sorted(lat_of):
        by_region.setdefault(region_of[r], []).append(r)
    out = []
    for tup in zip_longest(*(by_region[k] for k in sorted(by_region))):
        out += [i for i in tup if i is not None]
    return out, lat_of


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48, help="held-out buildings to score (the map asks >= 40)")
    ap.add_argument("--ids_from", default=None, help="replay the id set of a previous artifact JSON")
    ap.add_argument("--latents", default=str(LATENTS))
    ap.add_argument("--a2", default=None, help="a vecset denoiser checkpoint to add as an arm")
    ap.add_argument("--strength", type=float, nargs="*", default=[0.5], help="A2 projection strengths")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--map24", default=str(MAP24), help="deployed dense-grid checkpoint; '' to skip")
    ap.add_argument("--ddim", type=int, default=100)
    ap.add_argument("--montage", type=int, default=8, help="buildings in the montage (0 disables)")
    ap.add_argument("--sne", type=int, default=22,
                    help="views for Sharp Normal Error on the montage subset (#79); 0 disables")
    ap.add_argument("--size", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="suffix for the artifact and montage filenames")
    args = ap.parse_args()

    import h5py
    import torch

    # Comparability is this harness's reason to exist, so the sampling noise floor is pinned down
    # rather than assumed. Seeding PER BUILDING (below, in phases B and C) is what does the work: it
    # took the deployed arm's run-to-run spread in median 3D IoU from 0.027 to 0.001.
    #
    # It is deliberately NOT bit-exact. The deployed arm still drifts by ~0.001 between processes,
    # and that residue is not cuDNN: `benchmark=False` alone does not remove it, and same-seed
    # inferences are already bit-identical *within* a process. `deterministic=True` plus TF32 off
    # would likely settle it but costs 13x (2 it/s vs 27), i.e. ~40 min for this arm alone. Against a
    # blockout-vs-deployed fp-IoU gap of 0.18, buying exactness at that price is the wrong trade --
    # so the residue is measured and reported (`noise_floor` in the artifact) instead of hidden.
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    suffix = f"_{args.tag}" if args.tag else ""
    art = REPO / f"execution/artifacts/massing_arms_eval{suffix}.json"
    montage_path = REPO / f"outputs/massing_arms_eval/montage{suffix}.png"

    cand, lat_of = pick_ids(Path(args.latents), args.ids_from)
    print(f"[ids] {len(cand)} held-out candidates in {args.latents}", flush=True)

    with h5py.File(args.latents, "r") as f:
        fp_of = {b: np.asarray(f["footprint"][lat_of[b]]) for b in cand}
        ht_of = {b: float(f["height_m"][lat_of[b]]) for b in cand}
        rg_of = {b: int(f["region"][lat_of[b]]) for b in cand}

    scores: dict = {}
    fields: dict = {}
    gt_occ: dict = {}
    ids: list = []

    # ---- phase A: geometry-only arms (no model). Also fixes the final id set. ----------------------
    t0 = time.time()
    with h5py.File(H5, "r") as gt:
        for bid in cand:
            if len(ids) >= args.n and not args.ids_from:
                break
            g = np.asarray(gt["sdf"][bid], np.float32)
            gocc = g <= 0
            ext = _vertical_extent(gocc)
            if ext is None:
                continue
            bo = blockout_sdf(fp_of[bid], *ext)
            if bo is None or mesh_sdf_surface(np.clip(bo, -TRUNC, TRUNC))[0] is None:
                continue
            ids.append(bid)
            gt_occ[bid] = gocc
            scores.setdefault("gt", {})[bid] = score_arm(g, gocc, fp_of[bid])
            scores.setdefault("blockout", {})[bid] = score_arm(bo, gocc, fp_of[bid])
            if len(fields) < args.montage:
                fields[bid] = {"gt": g.copy(), "blockout": bo.copy()}
    print(f"[phase A] gt + blockout on n={len(ids)}  ({time.time()-t0:.0f}s)", flush=True)
    if len(ids) < args.n:
        print(f"[warn] only {len(ids)} of {args.n} ids survived the degeneracy filter", flush=True)
    arm_order = ["gt", "blockout"]

    # ---- phase B: codec arms (Dora resident) -------------------------------------------------------
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from models.shape_codec import Building, DoraCodec
    from scripts.foundations.dora_roundtrip_probe import load_dora
    codec = DoraCodec(load_dora(dev))

    a2 = None
    if args.a2:
        from models.networks.vecset_denoiser import VecsetDenoiser
        from models.networks.vecset_projection import SetSDEdit
        ck = torch.load(args.a2, map_location="cpu", weights_only=False)
        ca = ck["args"]
        net = VecsetDenoiser(latent_channels=ck["latent_channels"], width=ca["width"],
                             depth=ca["depth"], heads=ca["heads"],
                             footprint_res=ck["footprint_res"]).to(dev)
        net.load_state_dict(ck["model"]); net.eval()
        a2 = dict(op=SetSDEdit(net, timesteps=ca["timesteps"]),
                  mu=ck["latent_mu"], sd=ck["latent_sd"], step=int(ck["step"]))
        print(f"[a2] {args.a2}  step {a2['step']}", flush=True)

    t0 = time.time()
    with h5py.File(args.latents, "r") as f:
        for k, bid in enumerate(ids):
            fp = fp_of[bid]
            # ceiling: the cached real latent decoded -- the best any generator over this codec reaches
            z_real = torch.from_numpy(np.asarray(f["latent"][lat_of[bid]], np.float32))[None].to(dev)
            with torch.no_grad():
                fld = codec.decode_grid(z_real, RES).cpu().numpy()[0, 0]
            scores.setdefault("codec_ceiling", {})[bid] = score_arm(fld, gt_occ[bid], fp)
            if bid in fields:
                fields[bid]["codec_ceiling"] = fld

            if a2 is not None:
                bo = blockout_sdf(fp, *_vertical_extent(gt_occ[bid]))
                bv, bf = mesh_sdf_surface(np.clip(bo, -TRUNC, TRUNC))
                z0 = ((codec.encode(Building(verts=verts_to_world(bv), faces=bf)).float()
                       - a2["mu"]) / a2["sd"])
                fpt = torch.from_numpy(fp.astype(np.float32))[None, None].to(dev)
                ht = torch.tensor([ht_of[bid]], device=dev)
                rg = torch.tensor([rg_of[bid]], device=dev)
                for s in args.strength:
                    zp = a2["op"].project(blockout=z0, footprint=fpt, height=ht, region=rg,
                                          strength=s, steps=args.steps, guidance=args.guidance,
                                          seed=args.seed * 1000003 + bid)
                    with torch.no_grad():
                        fld = codec.decode_grid(zp * a2["sd"] + a2["mu"], RES).cpu().numpy()[0, 0]
                    scores.setdefault(f"a2_s{s}", {})[bid] = score_arm(fld, gt_occ[bid], fp)
                    if bid in fields:
                        fields[bid][f"a2_s{s}"] = fld
            if (k + 1) % 10 == 0:
                print(f"  [phase B] {k+1}/{len(ids)}  ({time.time()-t0:.0f}s)", flush=True)
    arm_order.append("codec_ceiling")
    if a2 is not None:
        arm_order += [f"a2_s{s}" for s in args.strength]
    del codec, a2
    torch.cuda.empty_cache()
    print(f"[phase B] codec arms done ({time.time()-t0:.0f}s)", flush=True)

    # ---- phase C: the deployed dense-grid prior (15 GB checkpoint; loaded last, alone) --------------
    if args.map24:
        from datasets.bag3d_dataset import Bag3dDataset
        from models.stage3a_model import Stage3aModel
        from scripts.foundations.baseline_gate_eval import build_opt
        opt = build_opt(dev, ckpt=args.map24, use_region=True, use_extra_cond=False, use_ema=True)
        opt.bag3d_h5 = str(H5); opt.ddim_steps = args.ddim
        s3 = Stage3aModel(); s3.initialize(opt); s3.switch_eval()
        ds = Bag3dDataset(); ds.initialize(opt, phase="test")
        gof = {int(g): i for i, g in enumerate(ds.idxs)}
        t0 = time.time()
        for k, bid in enumerate(ids):
            if bid not in gof:
                print(f"  [phase C] row {bid} is not in the dataset test split -- skipped", flush=True)
                continue
            item = ds[gof[bid]]
            data = {kk: (v.unsqueeze(0).to(dev) if torch.is_tensor(v) else v)
                    for kk, v in item.items() if torch.is_tensor(v)}
            # Seed PER BUILDING, not once per run: `Stage3aModel.inference` draws its DDIM start from
            # the global RNG, so an unseeded harness moved the deployed arm's median 3D IoU by 0.027
            # between two otherwise identical runs -- larger than differences it is meant to resolve.
            # Keying on the id (not on loop position) also makes a building's sample independent of
            # how many preceded it, so scoring a subset reproduces the full run's per-building rows.
            torch.manual_seed(args.seed * 1000003 + bid)
            with torch.no_grad():
                fld = s3.inference(data, ddim_steps=opt.ddim_steps, uc_scale=1.0).cpu().numpy()[0, 0]
            scores.setdefault("deployed_map24", {})[bid] = score_arm(fld, gt_occ[bid], fp_of[bid])
            if bid in fields:
                fields[bid]["deployed_map24"] = fld
            if (k + 1) % 10 == 0:
                print(f"  [phase C] {k+1}/{len(ids)}  ({time.time()-t0:.0f}s)", flush=True)
        arm_order.append("deployed_map24")
        del s3
        torch.cuda.empty_cache()
        print(f"[phase C] deployed arm done ({time.time()-t0:.0f}s)", flush=True)

    # ---- report ------------------------------------------------------------------------------------
    summary = {arm: summarise(scores.get(arm, {}).values()) for arm in arm_order}
    print(f"\n=== MASSING ARMS (n={len(ids)} fixed held-out ids) ===")
    print(f"{'arm':18s} {'n':>4} {'fp-IoU':>8} {'missing':>9} {'extra':>8} {'3D IoU':>8}")
    for arm in arm_order:
        s = summary[arm]
        if s:
            print(f"{arm:18s} {s['n']:>4} {s['fp_iou']:>8.3f} {s['missing']:>9.3f} "
                  f"{s['extra']:>8.3f} {s['vol_iou']:>8.3f}")
    print("\n-- non-ranking regression guard (anti-correlated with the goal; never an arbiter) --")
    print("   ⚠️  comparable for ONE arm ACROSS RUNS only, never between arms: roughness is a raw")
    print("   |Laplacian|, so it scales with each arm's field slope (a metric SDF here is 0.032).")
    for arm in arm_order:
        if summary[arm]:
            print(f"  {arm:18s} surface_roughness {summary[arm]['guard_roughness']:.5f}   "
                  f"(field slope {summary[arm].get('guard_field_slope', float('nan')):.4f})")

    sne = {}
    if args.sne and fields:
        print(f"\ncomputing Sharp Normal Error ({len(fields)} buildings x {len(arm_order)} arms x "
              f"{args.sne} views)...", flush=True)
        sne = sharp_normal_error(fields, arm_order, dev, views=args.sne)
        print("\n-- Sharp Normal Error (#79): the ONE scalar here that ranks crisp above melted --")
        print("   Reported, never ranked on. ⚠️  Contaminated across arms: on a row whose blockout")
        print("   occupancy is BYTE-IDENTICAL to GT it still reads 0.241, not 0 -- a faceted field")
        print("   perturbs the edges the mask covers. Read only gaps far larger than that offset.")
        for arm in arm_order:
            if arm in sne and not np.isnan(sne[arm]):
                print(f"  {arm:18s} sharp_normal_error {sne[arm]:.4f}")

    if args.montage and fields:
        print(f"\nrendering montage ({len(fields)} buildings x {len(arm_order)} arms)...", flush=True)
        print("montage: "
              f"{build_montage(fields, arm_order, scores, summary, montage_path, args.size)}", flush=True)

    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO),
                         capture_output=True, text=True).stdout.strip()
    art.parent.mkdir(parents=True, exist_ok=True)
    art.write_text(json.dumps(dict(
        meta=dict(git_rev=rev, created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  n=len(ids), gt_h5=str(H5.relative_to(REPO)), latents=args.latents,
                  map24=args.map24 or None, a2=args.a2, strength=args.strength if args.a2 else [],
                  ddim=args.ddim, steps=args.steps, guidance=args.guidance, seed=args.seed,
                  ids_from=args.ids_from, montage=str(montage_path.relative_to(REPO))),
        ids=ids,
        summary=summary,
        # #79: reported, never ranked on. Computed on the montage subset, not all `ids` -- it needs
        # `views` rasterisations per arm per building. Cross-arm it carries a field-representation
        # offset (0.241 at byte-identical occupancy), so only large gaps are readable.
        sharp_normal_error=dict(
            values=sne, views=args.sne, n_buildings=len(fields),
            note="the one scalar in this project that ranks crisp above melted (#79: crisp 0.084 vs "
                 "melted 0.636, separated 8/8). surface_roughness ranks the same pair backwards. "
                 "Contaminated by field representation -- safe within one arm across runs."),
        # What a difference has to clear to be a difference. `gt`/`blockout`/`codec_ceiling` are
        # deterministic and reproduce bit-exactly; the sampled arms do not, so their medians carry
        # this much run-to-run slop at n=48 even fully seeded. Measured by running the harness twice.
        noise_floor=dict(
            deterministic_arms=["gt", "blockout", "codec_ceiling"],
            measured_on="deployed_map24, 3 seeded runs at n=48",
            median_range=dict(fp_iou=0.008, missing=0.016, extra=0.040, vol_iou=0.001),
            note="`extra` is by far the loosest -- it is unbounded above and dominated by the "
                 "over-fill, so a change in it under ~0.04 is noise even though vol_iou is "
                 "stable to 0.001. Unseeded, vol_iou alone spread 0.027."),
        per_building={arm: {str(b): r for b, r in scores.get(arm, {}).items()} for arm in arm_order},
    ), indent=2))
    print(f"artifact: {art}", flush=True)


if __name__ == "__main__":
    main()
