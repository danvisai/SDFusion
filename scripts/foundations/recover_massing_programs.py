"""Recover a semantic layer program for every real building, and measure how well it fits.

#10's question, answered on our own corpus rather than argued from the literature: can constrained
architectural volumes express real LoD2 massing, in how many operations, and with what residual?

WHY THIS IS A HEIGHT-MAP FITTER AND NOT A 3-D CSG FITTER
--------------------------------------------------------
Measured on the pre-registered 714 held-out buildings before any of this was written:

  * `missing` of the blockout against GT is **0.000000 on 714/714** -- the real building is always
    entirely inside its own extruded footprint, so nothing is ever *added* and the task is purely
    subtractive.
  * **100.0%** of the carve volume sits *above the topmost GT voxel in its column*. Through-voids
    (courtyard / passage / light well) account for **0 voxels**, and overhangs for 71 out of
    4,324,919.
  * Only **4 columns in 1,072,438** are not a solid run from the base.

So every building in this corpus is exactly `{(z,y,x) : y0 <= y <= top(z,x)}` -- a 64x64 **height
map**, not a general volume. That is the same object ArcPro's `CreateLayer` grammar produces
(vertically extruded polygonal layers), which is why a layer vocabulary is the right one here and
why `SubtractCourtyard` / `CutNotch` are dead operations on this data: they can never fire.

Fitting in height-map space rather than on the 64^3 grid is therefore not an approximation. It is
the exact representation, and it makes the containment invariant trivial to enforce: a fitted
height may never drop below the target height, so the program can never cut into GT and
`collapse_rate` is 0 by construction.

THE VOCABULARY
--------------
    Layer(height, polygon)        one connected region flattened to one height  (ArcPro CreateLayer)
    CutRoof(kind, eaves, rate)    height falls off with distance from the footprint edge;
                                  kind=hip erodes on all sides, gable_x / gable_z on one axis
    Ramp(region, slope)           the tightest PLANE above the target over one region -- the shed
                                  roof `CutRoof` cannot express, at arbitrary rotation

`ApplySetback` is not a separate operation: in a height field a setback *is* a Layer whose polygon
is the inward offset of the footprint, and the fitter finds it as one.

Output is a semantic program per building plus the recovery statistics. It trains nothing, touches
no GPU, and does not modify the active #92 experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.foundations.eval_massing_arms import (            # noqa: E402
    RES, S_STAR_VOXELS, COLLAPSE_MISSING, volume_split, footprint_split, fp_iou, vs_input,
)

H5 = REPO / "data/real_massing_v1/real.h5"
SHIP714 = REPO / "execution/artifacts/massing_arms_eval_ship714.json"

# A building only needs a carve at all if its blockout over-fills by more than this. Matches the
# `allowance` vocabulary in CONTEXT.md: a decision, recorded in one place so it cannot drift.
CARVE_NEEDED = 0.02


# ----------------------------------------------------------------------------------------------
# the height field
# ----------------------------------------------------------------------------------------------

def height_field(gt_occ: np.ndarray, fp: np.ndarray):
    """GT occupancy -> (y0, y1, target height map in voxels above y0).

    `top` is 0 off the footprint and in [1 .. y1-y0+1] on it. The blockout is the constant map
    `y1-y0+1`, so `blockout - target` is exactly the per-column carve depth.
    """
    ys = np.nonzero(gt_occ.any(axis=(0, 2)))[0]
    if not len(ys):
        return None
    y0, y1 = int(ys.min()), int(ys.max())
    sub = gt_occ[:, y0:y1 + 1, :]                              # [z, h, x]
    top = (sub.shape[1] - 1 - np.argmax(sub[:, ::-1, :], axis=1)) + 1
    return y0, y1, np.where(fp, top, 0).astype(np.int16)


def occupancy(fp: np.ndarray, y0: int, h: np.ndarray) -> np.ndarray:
    """Height map -> 64^3 occupancy, the inverse of `height_field`."""
    yy = np.arange(RES)[None, :, None]
    return fp[:, None, :] & (yy >= y0) & (yy < y0 + h[:, None, :].astype(np.int32))


def occ_to_field(occ: np.ndarray) -> np.ndarray:
    """Occupancy -> signed EDT on the corpus scale, identical to `blockout_sdf`'s convention.

    Scoring downstream reads `field <= 0`, so this keeps the recovered arm on exactly the same
    footing as `blockout` and `gt` in the shipped harness rather than inventing a second path.
    """
    inside = ndimage.distance_transform_edt(occ)
    outside = ndimage.distance_transform_edt(~occ)
    return ((outside - inside) * (2.0 / (RES - 1))).astype(np.float32)


def _dist_axis(fp: np.ndarray, axis: int) -> np.ndarray:
    """Cells to the nearest non-footprint cell along ONE axis (1 on the boundary cell).

    A gable roof insets along a single axis, so an isotropic distance transform cannot express it;
    a hip roof insets on all sides and needs the isotropic one. Both are required.
    """
    m = fp if axis == 1 else fp.T
    n = m.shape[1]
    fwd = np.zeros(m.shape, np.int16)
    acc = np.zeros(m.shape[0], np.int16)
    for i in range(n):
        acc = np.where(m[:, i], acc + 1, 0)
        fwd[:, i] = acc
    bwd = np.zeros(m.shape, np.int16)
    acc = np.zeros(m.shape[0], np.int16)
    for i in range(n - 1, -1, -1):
        acc = np.where(m[:, i], acc + 1, 0)
        bwd[:, i] = acc
    out = np.minimum(fwd, bwd)
    return out if axis == 1 else out.T


# ----------------------------------------------------------------------------------------------
# the fitter
# ----------------------------------------------------------------------------------------------

ROOF_RATES = (0.5, 1.0, 1.5, 2.0, 3.0)


def _roof_candidates(fp, dists, target, h):
    """Every (kind, eaves, rate) roof that stays at or above the target height everywhere."""
    for kind, d in dists.items():
        for rate in ROOF_RATES:
            slope = (d.astype(np.float32) - 1.0) * rate
            for eaves in range(1, int(h.max()) + 1):
                cand = np.minimum(h, np.floor(eaves + slope)).astype(np.int16)
                cand = np.where(fp, np.maximum(cand, 1), 0)
                if (cand[fp] < target[fp]).any():
                    continue                                   # would cut into GT -- rejected
                gain = int((h[fp] - cand[fp]).sum())
                if gain > 0:
                    yield gain, cand, dict(op="CutRoof", kind=kind, eaves=int(eaves),
                                           rate=float(rate))


def _layer_candidates(fp, target, h):
    """Every single connected polygon that can be flattened to one height.

    A Layer is ONE polygon by definition, so a candidate height whose eligible region breaks into
    several components is offered once per component rather than as a disconnected set.
    """
    for v in np.unique(target[fp]):
        region = fp & (target <= v) & (h > v)
        if not region.any():
            continue
        lab, n = ndimage.label(region)
        for c in range(1, n + 1):
            piece = lab == c
            gain = int((h[piece] - v).sum())
            if gain > 0:
                cand = np.where(piece, np.int16(v), h)
                yield gain, cand, dict(op="Layer", height=int(v), area=int(piece.sum()),
                                       components=1)


def _ramp_candidates(fp, target, h, max_regions: int = 3):
    """The tightest PLANE that stays at or above the target over a surplus region.

    Why a general plane and not an axis-aligned ramp: buildings sit at arbitrary rotations on the
    grid, so a shed roof's fall line almost never lines up with x or z. And why a plane at all --
    `CutRoof` measures distance to the *nearest* footprint edge, which is symmetric, so it can
    express a gable or a hip but never a **shed**. The worst-residual trace was entirely smooth
    roof ramps that `CutRoof` could not fire on, which the fitter then approximated with a
    staircase of flat `Layer`s.

    Finding the plane is a 3-variable linear program: minimise the surplus `sum(a + b*x + c*z -
    target)` subject to `a + b*x + c*z >= target` on every cell of the region, so the result is the
    optimal ramp rather than a sampled guess. Because `target` is integral and the plane dominates
    it, `floor(plane) >= target` still holds and the containment invariant survives the rounding.

    Offered per connected surplus region, so a gable is recoverable as two opposing ramps.
    """
    from scipy.optimize import linprog

    surplus = fp & (h > target)
    if not surplus.any():
        return
    lab, n = ndimage.label(surplus)
    order = sorted(range(1, n + 1), key=lambda c: -int((lab == c).sum()))[:max_regions]
    zz_g, xx_g = np.mgrid[0:RES, 0:RES]
    for c in order:
        piece = lab == c
        zz, xx = np.nonzero(piece)
        if len(xx) < 3:
            continue
        t = target[piece].astype(float)
        ones = np.ones(len(xx))
        # -(a + b*x + c*z) <= -target   <=>   plane >= target
        A = -np.stack([ones, xx.astype(float), zz.astype(float)], 1)
        obj = np.array([ones.sum(), xx.sum(), zz.sum()], float)
        try:
            r = linprog(obj, A_ub=A, b_ub=-t, bounds=[(None, None)] * 3, method="highs")
        except Exception:
            continue
        if not r.success:
            continue
        a, b, cz = r.x
        plane = np.floor(a + b * xx_g + cz * zz_g)
        cand = np.where(piece, np.minimum(h, plane).astype(np.int16), h)
        cand = np.where(fp, np.maximum(cand, 1), 0).astype(np.int16)
        if (cand[fp] < target[fp]).any():
            continue                                       # rounding guard; never cut GT
        gain = int((h[fp] - cand[fp]).sum())
        if gain > 0:
            yield gain, cand, dict(op="Ramp", area=int(piece.sum()),
                                   slope=[round(float(b), 4), round(float(cz), 4)])


def _all_candidates(fp, dists, target, h):
    """Every operation the vocabulary can offer against the current height map."""
    yield from _roof_candidates(fp, dists, target, h)
    yield from _layer_candidates(fp, target, h)
    yield from _ramp_candidates(fp, target, h)


def _dists_for(fp):
    return dict(hip=ndimage.distance_transform_edt(fp).astype(np.int16),
                gable_x=_dist_axis(fp, 1), gable_z=_dist_axis(fp, 0))


def fit_program(fp, y0, y1, target, max_ops=4, allowance=CARVE_NEEDED):
    """Greedy: repeatedly take the operation that removes the most surplus without cutting GT."""
    full = np.int16(y1 - y0 + 1)
    h = np.where(fp, full, 0).astype(np.int16)
    gt_vox = int(target[fp].sum())
    dists = _dists_for(fp)
    ops = []
    for _ in range(max_ops):
        surplus = int((h[fp] - target[fp]).sum())
        if gt_vox and surplus / gt_vox <= allowance:
            break
        best = max(_all_candidates(fp, dists, target, h), key=lambda t: t[0], default=None)
        if best is None or best[0] <= 0:
            break
        gain, h, meta = best
        meta["removed_voxels"] = int(gain)
        # the residual AFTER this operation, so one K=16 run yields the whole simplicity curve
        # instead of re-fitting the corpus once per K
        meta["residual_extra"] = (float((h[fp] - target[fp]).sum() / gt_vox) if gt_vox else 0.0)
        ops.append(meta)
    return ops, h


def fit_program_beam(fp, y0, y1, target, max_ops=4, allowance=CARVE_NEEDED,
                     beam=6, branch=6):
    """Beam search over programs, because greedy is provably myopic on gable roofs.

    The worst-residual trace after `Ramp` landed was entirely **symmetric double ramps**: a gable
    rises from both eaves to a ridge, so no single plane dominates it and it needs two opposing
    `Ramp`s. Greedy never gets there -- one large flat `Layer` always wins the immediate gain, and
    by the time the surplus has split into the two regions that would each take a ramp, the
    operation budget is spent. That is a search failure, not a missing operation: at K=16 greedy
    already reaches 3-D IoU 0.9981, so the vocabulary is sufficient and only the order is wrong.

    Beams are de-duplicated by the height map itself rather than by the operation list, since two
    different orders that reach the same massing are the same program for every purpose here.
    """
    import heapq

    full = np.int16(y1 - y0 + 1)
    h0 = np.where(fp, full, 0).astype(np.int16)
    gt_vox = int(target[fp].sum())
    dists = _dists_for(fp)
    surplus = lambda hh: int((hh[fp] - target[fp]).sum())

    beams = [(surplus(h0), h0, [])]
    for _ in range(max_ops):
        nxt, seen = [], set()
        for sur, h, ops in beams:
            if gt_vox and sur / gt_vox <= allowance:
                nxt.append((sur, h, ops))                  # already good enough: carry it forward
                continue
            top = heapq.nlargest(branch, _all_candidates(fp, dists, target, h),
                                 key=lambda t: t[0])
            for gain, hh, meta in top:
                if gain <= 0:
                    continue
                key = hh.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                m = dict(meta)
                m["removed_voxels"] = int(gain)
                s2 = surplus(hh)
                m["residual_extra"] = (float(s2 / gt_vox) if gt_vox else 0.0)
                nxt.append((s2, hh, ops + [m]))
        if not nxt:
            break
        nxt.sort(key=lambda t: (t[0], len(t[2])))
        beams = nxt[:beam]
    best = min(beams, key=lambda t: (t[0], len(t[2])))

    # A beam search is NOT guaranteed to dominate greedy: the greedy path can be pruned at an
    # intermediate step by siblings that look better then and end worse. Measured -- id 16764 went
    # 0.152 greedy -> 0.159 beam. Greedy is cheap, so run it too and keep whichever program is
    # actually better. This makes the beam a monotone improvement by construction.
    g_ops, g_h = fit_program(fp, y0, y1, target, max_ops, allowance)
    if surplus(g_h) < best[0]:
        return g_ops, g_h
    return best[2], best[1]


# ----------------------------------------------------------------------------------------------
# visual trace
# ----------------------------------------------------------------------------------------------

def build_montage(cases, out: Path, cell: int = 128) -> Path:
    """Height-map trace: footprint | GT | recovered | residual, one row per building.

    Rendered as height maps rather than shaded 3-D views on purpose. The corpus *is* a height
    field (see the module docstring), so this shows the actual object under fit rather than a
    projection of it, and it stays CPU-only while the A100 is busy with #92.
    """
    from PIL import Image, ImageDraw

    def tile(a, vmax, colour):
        """One 64x64 map -> an upscaled RGB tile. `colour` False renders a binary mask."""
        a = np.asarray(a, np.float32)
        if colour:
            n = np.clip(a / max(vmax, 1e-6), 0, 1)
            # dark blue (low) -> yellow (high); off-footprint stays near-black
            rgb = np.stack([np.clip(1.6 * n - 0.3, 0, 1),
                            np.clip(1.5 * n - 0.1, 0, 1),
                            np.clip(0.9 - 1.1 * n, 0, 1)], -1)
            rgb[a <= 0] = 0.06
        else:
            rgb = np.repeat((a > 0).astype(np.float32)[..., None], 3, -1) * 0.85 + 0.06
        img = Image.fromarray((rgb * 255).astype(np.uint8), "RGB")
        return img.resize((cell, cell), Image.NEAREST)

    cols = ["footprint", "GT height", "recovered", "residual (surplus)"]
    pad, head = 4, 22
    W = len(cols) * (cell + pad) + pad
    H = head + len(cases) * (cell + pad) + pad
    sheet = Image.new("RGB", (W, H), (16, 16, 18))
    d = ImageDraw.Draw(sheet)
    for j, c in enumerate(cols):
        d.text((pad + j * (cell + pad) + 4, 6), c, fill=(210, 210, 215))
    for i, cs in enumerate(cases):
        y = head + i * (cell + pad) + pad
        vmax = float(max(cs["target"].max(), cs["fitted"].max(), 1))
        for j, im in enumerate([tile(cs["fp"], 1, False), tile(cs["target"], vmax, True),
                                tile(cs["fitted"], vmax, True),
                                tile(cs["fitted"] - cs["target"], vmax, True)]):
            sheet.paste(im, (pad + j * (cell + pad), y))
        d.text((pad + 4, y + 4), f"id {cs['id']}", fill=(255, 255, 255))
        d.text((pad + 4, y + cell - 14),
               f"{cs['n_ops']} ops  extra {cs['extra']:.3f}", fill=(255, 240, 160))
        d.text((pad + 2 * (cell + pad) + 4, y + cell - 14),
               " > ".join(cs["ops"][:4]) or "empty", fill=(160, 230, 255))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids_from", default=str(SHIP714),
                    help="replay a pinned id set; default is the pre-registered 714")
    ap.add_argument("--n", type=int, default=0, help="0 = every id in the set")
    ap.add_argument("--max_ops", type=int, default=4)
    ap.add_argument("--allowance", type=float, default=CARVE_NEEDED)
    ap.add_argument("--out", default="execution/artifacts/program_recovery_714.json")
    ap.add_argument("--beam", type=int, default=1,
                    help="1 = greedy (the default all earlier numbers used); >1 runs a "
                         "beam search of this width over programs")
    ap.add_argument("--branch", type=int, default=6,
                    help="candidates expanded per beam per step")
    ap.add_argument("--montage", type=int, default=0,
                    help="rows per sheet; emits a worst-N and a representative-N trace")
    args = ap.parse_args()

    ids = [int(i) for i in json.load(open(args.ids_from))["ids"]]
    if args.n:
        ids = ids[:args.n]
    print(f"[ids] {len(ids)} buildings from {args.ids_from}", flush=True)

    rows, cases, t0 = {}, [], time.time()
    with h5py.File(H5, "r") as g:
        for k, b in enumerate(ids):
            gt = np.asarray(g["sdf"][b], np.float32) <= 0
            fp = np.asarray(g["footprint"][b]) > 0
            hf = height_field(gt, fp)
            if hf is None:
                continue
            y0, y1, target = hf
            bo_occ = occupancy(fp, y0, np.where(fp, np.int16(y1 - y0 + 1), 0).astype(np.int16))
            if args.beam > 1:
                ops, h = fit_program_beam(fp, y0, y1, target, args.max_ops, args.allowance,
                                          args.beam, args.branch)
            else:
                ops, h = fit_program(fp, y0, y1, target, args.max_ops, args.allowance)
            occ = occupancy(fp, y0, h)

            row = dict(fp_iou=fp_iou(occ, fp), n_ops=len(ops),
                       ops=[o["op"] for o in ops], program=ops)
            row.update(volume_split(occ, gt))
            row.update(footprint_split(occ, fp))
            row["vs_input"] = vs_input(occ, bo_occ)
            row["blockout_extra"] = volume_split(bo_occ, gt)["extra"]
            rows[str(b)] = row
            if args.montage:
                cases.append(dict(id=b, fp=fp.copy(), target=target.copy(), fitted=h.copy(),
                                  n_ops=len(ops), extra=row["extra"],
                                  ops=[o["op"] for o in ops]))
            if (k + 1) % 100 == 0:
                print(f"  {k+1}/{len(ids)}  {time.time()-t0:.0f}s", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(meta=dict(created=time.strftime("%Y-%m-%dT%H:%M:%S"), n=len(rows),
                             gt_h5=str(H5.relative_to(REPO)), ids_from=args.ids_from,
                             max_ops=args.max_ops, allowance=args.allowance, beam=args.beam,
                             vocabulary=["Layer", "CutRoof", "Ramp"]),
                   ids=[int(b) for b in rows], per_building=rows), open(out, "w"), indent=1)
    print(f"[artifact] {out}", flush=True)
    if args.montage:
        carve = [c for c in cases if rows[str(c["id"])]["blockout_extra"] >= CARVE_NEEDED]
        worst = sorted(carve, key=lambda c: -c["extra"])[:args.montage]
        rep = sorted(carve, key=lambda c: c["extra"])[len(carve) // 2:][:args.montage]
        for tag, sel in (("worst", worst), ("representative", rep)):
            if sel:
                p = build_montage(sel, REPO / f"outputs/program_recovery/{tag}.png")
                print(f"[montage] {p}", flush=True)
    report(rows, args)


def report(rows, args) -> None:
    """Split the two populations, never pooled -- #80's bimodal result is the precedent."""
    carve = {b: r for b, r in rows.items() if r["blockout_extra"] >= CARVE_NEEDED}
    flat = {b: r for b, r in rows.items() if r["blockout_extra"] < CARVE_NEEDED}

    def block(name, d):
        if not d:
            print(f"\n{name}: none")
            return
        med = lambda k: float(np.median([r[k] for r in d.values()]))
        coll = float(np.mean([r["missing"] >= COLLAPSE_MISSING for r in d.values()]))
        print(f"\n{name}  (n={len(d)})")
        print(f"  fp-IoU        {med('fp_iou'):.4f}      vs_input      {med('vs_input'):.4f}")
        print(f"  missing       {med('missing'):.6f}    collapse_rate {coll:.4f}")
        print(f"  3D IoU        {med('vol_iou'):.4f}      n_ops         {med('n_ops'):.1f}")
        print(f"  extra  before {med('blockout_extra'):.4f}  ->  after {med('extra'):.4f}")
        under = float(np.mean([r["extra"] <= args.allowance for r in d.values()]))
        print(f"  reach the allowance ({args.allowance}): {under*100:.1f}%")

    print("\n" + "=" * 78)
    block("CARVE-NEEDING buildings", carve)
    block("ALREADY-FLAT buildings", flat)
    if carve:
        names = [o for r in carve.values() for o in r["ops"]]
        vol = {}
        for r in carve.values():
            for o in r["program"]:
                vol[o["op"]] = vol.get(o["op"], 0) + o["removed_voxels"]
        tv = sum(vol.values()) or 1
        print("\noperation mix on carve-needing buildings")
        for k in sorted(vol, key=lambda k: -vol[k]):
            print(f"  {k:<10} used {names.count(k):>5}x   {vol[k]/tv*100:5.1f}% of removed volume")
    print("=" * 78)


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------------------------------
# isometric render
# ----------------------------------------------------------------------------------------------

def render_iso(h, fp, cell: int = 6, pad: int = 20, base=(196, 198, 203)):
    """Shaded isometric view of a height map, drawn on the CPU.

    The harness's `render_world` goes through pyrender/EGL and hangs on this node while the four
    #92 arms hold the GPU. It is also more machinery than this needs: the corpus is a height field,
    so the massing can be drawn exactly as one top face plus two side faces per column, with a
    painter's-algorithm ordering by (x + z). No marching cubes, no mesh, no GPU, and the result is
    the true geometry rather than an isosurface approximation of it.
    """
    from PIL import Image, ImageDraw

    H = np.asarray(h, np.int32)
    Z, X = H.shape
    cos30, sin30, hs = 0.866, 0.5, cell * 0.62
    sx = lambda x, z: (x - z) * cos30 * cell
    sy = lambda x, z, v: (x + z) * sin30 * cell - v * hs
    x0, x1 = sx(0, Z), sx(X, 0)
    y0, y1 = sy(0, 0, int(H.max())), sy(X, Z, 0)
    W, Ht = int(x1 - x0) + 2 * pad, int(y1 - y0) + 2 * pad
    ox, oy = -x0 + pad, -y0 + pad
    img = Image.new("RGB", (W, Ht), (255, 255, 255))
    d = ImageDraw.Draw(img)

    gz, gx = np.gradient(H.astype(np.float64))
    lam = 1.0 / np.sqrt(gx ** 2 + gz ** 2 + 1.0)              # Lambert against a vertical light
    # round to integers so neighbouring columns share exact vertices -- without this the
    # side faces are separated by hairline background gaps and the massing looks combed
    P = lambda x, z, v: (round(sx(x, z) + ox), round(sy(x, z, v) + oy))
    shade = lambda f: tuple(int(np.clip(c * f, 0, 255)) for c in base)

    order = sorted(((x + z, z, x) for z in range(Z) for x in range(X) if fp[z, x]))
    for _, z, x in order:
        v = int(H[z, x])
        if v <= 0:
            continue
        nx = int(H[z, x + 1]) if x + 1 < X and fp[z, x + 1] else 0
        nz = int(H[z + 1, x]) if z + 1 < Z and fp[z + 1, x] else 0
        d.polygon([P(x, z, v), P(x + 1, z, v), P(x + 1, z + 1, v), P(x, z + 1, v)],
                  fill=shade(0.62 + 0.55 * lam[z, x]))
        if v > nx:
            d.polygon([P(x + 1, z, v), P(x + 1, z + 1, v),
                       P(x + 1, z + 1, nx), P(x + 1, z, nx)], fill=shade(0.74))
        if v > nz:
            d.polygon([P(x, z + 1, v), P(x + 1, z + 1, v),
                       P(x + 1, z + 1, nz), P(x, z + 1, nz)], fill=shade(0.52))
    return img


def build_iso_sheet(cases, out: Path, cell: int = 6) -> Path:
    """Real building beside recovered building, one row each, as shaded 3-D massing."""
    from PIL import Image, ImageDraw

    tiles = [(c, render_iso(c["target"], c["fp"], cell), render_iso(c["fitted"], c["fp"], cell))
             for c in cases]
    tw = max(max(a.width, b.width) for _, a, b in tiles)
    th = max(max(a.height, b.height) for _, a, b in tiles)
    head, pad, lab = 26, 10, 30
    W = 2 * tw + 3 * pad
    sheet = Image.new("RGB", (W, head + len(tiles) * (th + lab)), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    d.text((pad, 8), "REAL BUILDING", fill=(0, 0, 0))
    d.text((pad + tw + pad, 8), "RECOVERED BY THE PROGRAM", fill=(0, 0, 0))
    for i, (c, a, b) in enumerate(tiles):
        y = head + i * (th + lab)
        sheet.paste(a, (pad + (tw - a.width) // 2, y + (th - a.height) // 2))
        sheet.paste(b, (2 * pad + tw + (tw - b.width) // 2, y + (th - b.height) // 2))
        d.text((pad, y + th + 6),
               f"id {c['id']}   {c['n_ops']} ops: {' > '.join(c['ops']) or 'empty'}"
               f"   surplus left {c['extra']:.3f}", fill=(40, 40, 40))
        d.line([(0, y + th + lab - 2), (W, y + th + lab - 2)], fill=(225, 225, 228))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out
