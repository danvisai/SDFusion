"""Does a footprint-conditioned height-map generator carve, or does it learn identity too?

#127's question. Every generator on this project's record -- #69 through #92, six arms across two
representations -- converged to returning its own input. This asks whether that was the *model* or
the *output space*, by moving the output space to the one #10 measured the corpus actually to be:
a 64x64 height map.

WHAT IS PREDICTED, AND WHY IT IS THE CARVE AND NOT THE HEIGHT
-------------------------------------------------------------
The label is the per-column **carve depth** `d = extent - top`, classified over 64 levels, not the
absolute top and not a regression. Three reasons, in the order they matter:

  * **Depth makes the arm purely subtractive.** `apply_depth` clamps to `[1, extent]`, so a
    prediction can never exceed the blockout it started from and `extra` can never come out worse
    than doing nothing. #10 measured `missing`=0 on 714/714 -- the real building is always inside
    its own extruded footprint -- so subtractive-only is the corpus's own structure, not a
    convenience.
  * **Classification, not MSE.** MSE returns the conditional mean, which on a bimodal roof
    distribution (flat top / pitched) is a roof nobody built -- the same regression-to-the-mean that
    produced the no-op. `--objective mse` exists to *test* that claim rather than assume it, and is
    scored as its own arm.
  * **The labels are exact integers already.** No quantisation, no codec, no latent. #10's
    reconstruction residual over the pinned 714 is 71 voxels in 4.3M.

WHAT THE OUTPUT SPACE GIVES FOR FREE, STATED PRECISELY
------------------------------------------------------
#127 claims a height map is "footprint-exact, collapse-impossible, and `missing` and `collapse_rate`
are 0 by clamping". Two of those are true and one is not, and the tests pin the difference:

  * footprint-exact  -- TRUE. `apply_depth` writes exactly the footprint mask, so fp-IoU is 1.0000
                        by construction for every prediction, good or bad.
  * a valid solid    -- TRUE. Every footprint column keeps at least one voxel, so no prediction can
                        punch a hole through the plan or return a hollow shell (#80's failure).
  * `missing` = 0    -- FALSE. Over-carving still eats GT. The collapse rate is measured on the
                        model's own output and published beside every number, exactly as #126
                        requires of the alternative-building arm that collapses on 16.7%.

THE BAR, PRE-REGISTERED BEFORE THE FIRST RUN
--------------------------------------------
Fixed here so a result cannot re-litigate it (map #87's discipline, and #10's record of stopping at
a dip and being wrong twice). Scored on the **411 carve-needing** buildings of the pinned 714 --
303 need no carve at all and a 42% no-op majority flatters every aggregate (#126 point 4).

  PASS   median `extra` strictly below the **1-NN retrieval** arm's, measured on the same rows in
         the same run. 1-NN is the bar, not the blockout (#127).
  GUARD  collapse rate no worse than 1-NN's, and `vs_input` < 0.98 -- an arm that did not move has
         not been measured as a generator at all (#75).
  KILL   median `extra` at or above the blockout's. That is identity, and it answers #127 "no".

The aggregate 3D IoU is reported to the right of the bar and is a diagnostic, never a gate: #126
demoted it because its median cannot rank a real building above the envelope.

Run (train + score every arm + montage):
    env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/train_height_map_generator.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.foundations.eval_massing_arms import (              # noqa: E402
    COLLAPSE_MISSING, RES, fp_iou, footprint_split, volume_split, vs_input,
)
from scripts.foundations.measure_scoring_optimum import (        # noqa: E402
    compare_to_envelope, transplant_height,
)
from scripts.foundations.recover_massing_programs import (       # noqa: E402
    CARVE_NEEDED, H5, SHIP714, height_field, occupancy, render_iso,
)

LATENTS = REPO / "data/real_massing_v1/vecset_latents.h5"
WORK = REPO / "outputs/height_map_generator"
CACHE = WORK / "height_fields.npz"

# One class per voxel of carve depth. The corpus's deepest carve is 53 voxels of a 60-voxel extent,
# so 64 covers the label range exactly and nothing is clipped away at training time.
DEPTH_CLASSES = RES

# Buildings held back from training to select the checkpoint. Drawn from the TRAINING rows -- the
# pinned 714 are never seen, not even for early stopping.
VAL_BUILDINGS = 1000

N_REGIONS = 3          # source corpora: 0 NL / 1 DE / 2 JP, the `region` column of the latent cache
COND_CHANNELS = 3 + N_REGIONS


# ==================================================================================================
# the label, and the invariants of the output space
# ==================================================================================================

def carve_depth(top: np.ndarray, fp: np.ndarray, extent: int) -> np.ndarray:
    """Height map -> per-column carve depth below the blockout. 0 off the footprint."""
    m = np.asarray(fp, bool)
    return np.where(m, int(extent) - np.asarray(top, np.int32), 0).astype(np.int16)


def apply_depth(fp: np.ndarray, extent: int, depth: np.ndarray) -> np.ndarray:
    """Carve depth -> height map, clamped so the result is a valid solid whatever was predicted.

    The clamp is the whole structural argument of #127 and it is deliberately total: it accepts any
    array at all, including negative and out-of-range depths, and still returns a height map that is
    footprint-exact and at least one voxel deep on every footprint column, never taller than the
    blockout. A prediction can therefore be *wrong*, but never *invalid*.
    """
    m = np.asarray(fp, bool)
    e = int(extent)
    h = np.clip(e - np.asarray(depth, np.int32), 1, max(e, 1))
    return np.where(m, h, 0).astype(np.int16)


def envelope_depth(fp: np.ndarray) -> np.ndarray:
    """The do-nothing prediction: carve nothing, which `apply_depth` renders as the blockout."""
    return np.zeros(np.shape(fp), np.int16)


# ==================================================================================================
# the conditioning -- footprint, conditioned height, region. Nothing else may enter.
# ==================================================================================================

def condition_channels(fp: np.ndarray, extent: int, height_m: float, region: int) -> np.ndarray:
    """[C, Z, X] network input built from #127's conditioning ONLY.

    The signature is the leakage guard: there is no argument through which the target height field
    could reach the model, and `test_two_buildings_with_the_same_conditioning_get_identical_input`
    pins it. Two real buildings with the same footprint, height and region are genuinely
    indistinguishable inputs -- #126 measured that they still differ by a median 3D IoU of 0.886,
    which is the irreducible ambiguity this arm is working inside.

    The distance transform is a deterministic function of the footprint, not new information. It is
    supplied because #10 found the roof operations are functions of distance-to-edge (a hip erodes
    on all sides, a gable on one), and a small convolutional net would otherwise spend capacity
    rediscovering it.
    """
    m = np.asarray(fp, bool)
    edt = ndimage.distance_transform_edt(m).astype(np.float32) / 8.0
    ch = [m.astype(np.float32),
          np.full(m.shape, float(extent) / RES, np.float32),
          np.full(m.shape, float(np.log1p(max(height_m, 0.0))) / 4.0, np.float32),
          np.clip(edt, 0.0, 4.0)]
    for r in range(N_REGIONS):
        ch.append(np.full(m.shape, 1.0 if int(region) == r else 0.0, np.float32))
    return np.stack(ch).astype(np.float32)


def decode_logits(logits: np.ndarray, fp: np.ndarray, extent: int) -> np.ndarray:
    """[K, Z, X] logits -> height map by **argmax**, never by expectation.

    Taking the mean of the predicted distribution would reintroduce at decode time exactly the
    regression-to-the-mean the classification objective exists to avoid: a column whose posterior is
    split between "flat at full height" and "cut to the eaves" has a mean at neither.
    """
    return apply_depth(fp, extent, np.argmax(np.asarray(logits), axis=0).astype(np.int16))


# ==================================================================================================
# the zero-training baselines #127 names
# ==================================================================================================

def mean_relative_depth(depths: np.ndarray, fps: np.ndarray, extents: np.ndarray) -> np.ndarray:
    """The corpus's mean roof, per grid cell, as a fraction of the building's own height.

    This is the *unconditional* conditional-mean -- the arm #127's design note warns an MSE
    objective converges to. Relative rather than absolute because the corpus normalises each
    building into the grid: averaging voxel depths across a 6-voxel and a 60-voxel building would
    measure the height distribution, not the roof.

    Cells no footprint covers get 0 rather than NaN, so the profile is defined everywhere.
    """
    f = np.asarray(fps, bool)
    rel = np.where(f, np.asarray(depths, np.float32) /
                   np.maximum(np.asarray(extents, np.float32), 1)[:, None, None], 0.0)
    cover = f.sum(0).astype(np.float32)
    return np.divide(rel.sum(0), cover, out=np.zeros(rel.shape[1:], np.float32), where=cover > 0)


def mean_roof_height(profile: np.ndarray, fp: np.ndarray, extent: int) -> np.ndarray:
    """The mean profile rendered on this footprint at this conditioned height."""
    return apply_depth(fp, extent, np.rint(np.asarray(profile, np.float32) * int(extent)))


def retrieve_nn(query_fps: np.ndarray, bank_fps: np.ndarray, chunk: int = 512) -> np.ndarray:
    """Index into `bank_fps` of the footprint-IoU-nearest bank row, for each query.

    Hyper-parameter free on purpose. The footprint is the shape half of the conditioning, and the
    height half is supplied exactly by `transplant_height`'s rescale, so a distance that mixed the
    two would need a weight -- and a *baseline* with a tuned weight is not a baseline. The bank is
    built from training rows only, so a held-out building can never retrieve itself.
    """
    q = np.asarray(query_fps, bool).reshape(len(query_fps), -1).astype(np.float32)
    b = np.asarray(bank_fps, bool).reshape(len(bank_fps), -1).astype(np.float32)
    qa, ba = q.sum(1), b.sum(1)
    out = np.zeros(len(q), np.int64)
    for s in range(0, len(q), chunk):
        inter = q[s:s + chunk] @ b.T
        union = qa[s:s + chunk, None] + ba[None, :] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        out[s:s + chunk] = np.argmax(iou, axis=1)
    return out


# ==================================================================================================
# the corpus as height fields, cached once
# ==================================================================================================

def build_cache(path: Path = CACHE, force: bool = False) -> dict:
    """Every corpus row as (footprint, base level, extent, target height map) + its conditioning.

    Keyed by the **latent cache**'s rows, because that file carries `held_out` -- the one split all
    of this project's arms have been scored against. Reading the 64^3 SDFs once and keeping only the
    height field turns 37 GB into 165 MB, which is the whole reason this task trains in minutes.
    """
    import h5py

    if path.exists() and not force:
        d = np.load(path)
        return {k: d[k] for k in d.files}
    with h5py.File(LATENTS, "r") as f:
        rows = f["row"][:].astype(np.int32)
        held = (f["held_out"][:] == 1).astype(np.uint8)
        region = f["region"][:].astype(np.int8)
        height_m = f["height_m"][:].astype(np.float32)
    n = len(rows)
    fps = np.zeros((n, RES, RES), np.uint8)
    targets = np.zeros((n, RES, RES), np.uint8)
    y0s = np.zeros(n, np.int16)
    extents = np.zeros(n, np.int16)
    ok = np.zeros(n, np.uint8)
    t0 = time.time()
    with h5py.File(H5, "r") as g:
        for k, b in enumerate(rows):
            gt = np.asarray(g["sdf"][int(b)], np.float32) <= 0
            fp = np.asarray(g["footprint"][int(b)]) > 0
            hf = height_field(gt, fp)
            if hf is None:
                continue
            y0, y1, target = hf
            fps[k] = fp
            targets[k] = np.clip(target, 0, 255)
            y0s[k], extents[k], ok[k] = y0, y1 - y0 + 1, 1
            if (k + 1) % 5000 == 0:
                print(f"  [cache] {k+1}/{n}  {time.time()-t0:.0f}s", flush=True)
    out = dict(row=rows, held=held, region=region, height_m=height_m,
               fp=fps, target=targets, y0=y0s, extent=extents, ok=ok)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    print(f"[cache] {path}  n={int(ok.sum())}/{n}  {time.time()-t0:.0f}s", flush=True)
    return out


# ==================================================================================================
# the model
# ==================================================================================================

def build_model(out_channels: int, width: int = 64):
    """A small U-Net over the 64x64 plan. ~4M parameters against A2's 49M and map-24's 947M.

    Depth is chosen so the bottleneck is 8x8 -- one cell there sees an eighth of the plan, which is
    the scale a setback or a ridge line lives at. Nothing here is novel and nothing needs to be:
    #127 is a question about the output space, so the network is the cheapest thing that can answer
    it, and a bigger one would confound the answer.
    """
    import torch
    import torch.nn as nn

    def block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU())

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            w = width
            self.e1, self.e2, self.e3 = block(COND_CHANNELS, w), block(w, 2 * w), block(2 * w, 4 * w)
            self.bot = block(4 * w, 4 * w)
            self.d3, self.d2, self.d1 = block(8 * w, 2 * w), block(4 * w, w), block(2 * w, w)
            self.head = nn.Conv2d(w, out_channels, 1)
            self.pool = nn.MaxPool2d(2)
            self.up = nn.Upsample(scale_factor=2, mode="nearest")

        def forward(self, x):
            s1 = self.e1(x)
            s2 = self.e2(self.pool(s1))
            s3 = self.e3(self.pool(s2))
            b = self.bot(self.pool(s3))
            x = self.d3(torch.cat([self.up(b), s3], 1))
            x = self.d2(torch.cat([self.up(x), s2], 1))
            x = self.d1(torch.cat([self.up(x), s1], 1))
            return self.head(x)

    return UNet()


def _d4(fp, target, k: int, flip: bool):
    """One of the 8 plan symmetries, applied to footprint and label together.

    Buildings sit at arbitrary grid rotations already (#10: an axis-aligned ramp could not fix the
    shed-roof residual), so the symmetry group is a property of the corpus rather than an assumption
    imposed on it. The conditioning is rebuilt from the rotated footprint, so nothing can desync.
    """
    fp, target = np.rot90(fp, k), np.rot90(target, k)
    if flip:
        fp, target = fp[:, ::-1], target[:, ::-1]
    return np.ascontiguousarray(fp), np.ascontiguousarray(target)


class HeightFieldSet:
    """Conditioning + label for one split, materialised on demand so augmentation stays honest."""

    def __init__(self, cache: dict, idx: np.ndarray, augment: bool, seed: int = 0):
        self.fp = cache["fp"][idx] > 0
        self.target = cache["target"][idx].astype(np.int16)
        self.extent = cache["extent"][idx].astype(np.int32)
        self.height_m = cache["height_m"][idx]
        self.region = cache["region"][idx].astype(np.int32)
        self.augment, self.rng = augment, np.random.default_rng(seed)

    def __len__(self):
        return len(self.fp)

    def batch(self, sel: np.ndarray):
        xs, ys = [], []
        for i in sel:
            fp, target = self.fp[i], self.target[i]
            if self.augment:
                fp, target = _d4(fp, target, int(self.rng.integers(4)), bool(self.rng.integers(2)))
            xs.append(condition_channels(fp, int(self.extent[i]), float(self.height_m[i]),
                                         int(self.region[i])))
            ys.append(carve_depth(target, fp, int(self.extent[i])))
        return (np.stack(xs), np.stack(ys).astype(np.int64),
                self.extent[sel].astype(np.float32))


def train(cache: dict, args) -> Path:
    """Train one arm and return the checkpoint chosen by validation loss.

    ⚠️ Selection is on a validation split drawn from the TRAINING rows. The pinned 714 are not read
    here at all. This project's record has two near-misses from reading a training curve as a trend
    (#80, twice), so the checkpoint is chosen by a held-in number and the whole curve is written to
    the artifact rather than summarised.
    """
    import torch
    import torch.nn.functional as F

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    pool = np.nonzero((cache["ok"] > 0) & (cache["held"] == 0))[0]
    perm = np.random.default_rng(args.seed).permutation(len(pool))
    val_idx, tr_idx = pool[perm[:VAL_BUILDINGS]], pool[perm[VAL_BUILDINGS:]]
    tr = HeightFieldSet(cache, tr_idx, augment=not args.no_aug, seed=args.seed)
    va = HeightFieldSet(cache, val_idx, augment=False)
    print(f"[train] {len(tr)} buildings, {len(va)} validation, objective={args.objective}, "
          f"device={dev}", flush=True)

    model = build_model(DEPTH_CLASSES if args.objective == "ce" else 1, args.width).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * max(len(tr) // args.batch, 1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    print(f"[train] {n_par/1e6:.2f}M parameters, {steps} steps", flush=True)

    def loss_of(x, y, ext):
        out = model(x)
        m = x[:, 0] > 0                                   # footprint columns only
        if args.objective == "ce":
            per = F.cross_entropy(out, y.clamp(0, DEPTH_CLASSES - 1), reduction="none")
        else:
            rel = y.float() / ext[:, None, None]
            per = (out[:, 0] - rel) ** 2 * (ext[:, None, None] ** 2)
        return (per * m).sum() / m.sum().clamp(min=1)

    def to_dev(b):
        x, y, e = b
        return (torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev),
                torch.from_numpy(e).to(dev))

    curve, best, best_path = [], float("inf"), WORK / f"{args.tag}.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed + 1)
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(len(tr))
        run = 0.0
        for s in range(0, len(order) - args.batch + 1, args.batch):
            x, y, e = to_dev(tr.batch(order[s:s + args.batch]))
            loss = loss_of(x, y, e)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += float(loss)
        run /= max(len(order) // args.batch, 1)
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([float(loss_of(*to_dev(va.batch(np.arange(s, min(s + 128, len(va)))))))
                                for s in range(0, len(va), 128)]))
        curve.append(dict(epoch=ep + 1, train=run, val=vl))
        mark = ""
        if vl < best:
            best, mark = vl, "  <- best"
            torch.save(dict(state=model.state_dict(), objective=args.objective, width=args.width,
                            epoch=ep + 1, val=vl, params=n_par), best_path)
        print(f"  epoch {ep+1:>3}/{args.epochs}  train {run:.4f}  val {vl:.4f}  "
              f"{time.time()-t0:.0f}s{mark}", flush=True)
    json.dump(curve, open(WORK / f"{args.tag}_curve.json", "w"), indent=1)
    print(f"[train] best val {best:.4f} -> {best_path}", flush=True)
    return best_path


def predict(ckpt: Path, held: dict, batch: int = 64, cpu: bool = False) -> np.ndarray:
    """Height maps for the pinned buildings from a trained checkpoint."""
    import torch

    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    dev = "cuda" if torch.cuda.is_available() and not cpu else "cpu"
    model = build_model(DEPTH_CLASSES if d["objective"] == "ce" else 1, d["width"]).to(dev)
    model.load_state_dict(d["state"])
    model.eval()
    out = np.zeros((len(held["fp"]), RES, RES), np.int16)
    with torch.no_grad():
        for s in range(0, len(out), batch):
            sel = range(s, min(s + batch, len(out)))
            x = np.stack([condition_channels(held["fp"][i], int(held["extent"][i]),
                                             float(held["height_m"][i]), int(held["region"][i]))
                          for i in sel])
            y = model(torch.from_numpy(x).to(dev)).cpu().numpy()
            for k, i in enumerate(sel):
                if d["objective"] == "ce":
                    out[i] = decode_logits(y[k], held["fp"][i], int(held["extent"][i]))
                else:
                    out[i] = apply_depth(held["fp"][i], int(held["extent"][i]),
                                         np.rint(y[k, 0] * float(held["extent"][i])))
    return out


# ==================================================================================================
# scoring
# ==================================================================================================

def score_arm(heights: np.ndarray, held: dict) -> list:
    """One row of metrics per pinned building, in the order #126 decided they must be read."""
    rows = []
    for i in range(len(heights)):
        fp, y0, extent = held["fp"][i], int(held["y0"][i]), int(held["extent"][i])
        gt = occupancy(fp, y0, held["target"][i])
        bo = occupancy(fp, y0, apply_depth(fp, extent, envelope_depth(fp)))
        occ = occupancy(fp, y0, heights[i])
        r = dict(id=int(held["row"][i]))
        r.update(volume_split(occ, gt))
        r.update(footprint_split(occ, fp))
        r["fp_iou"] = fp_iou(occ, fp)
        r["vs_input"] = vs_input(occ, bo)
        r["blockout_extra"] = volume_split(bo, gt)["extra"]
        rows.append(r)
    return rows


def summarise(rows: list) -> dict:
    med = lambda k: float(np.median([r[k] for r in rows])) if rows else float("nan")
    return dict(n=len(rows), missing=med("missing"), extra=med("extra"),
                vs_input=med("vs_input"),
                collapse_rate=float(np.mean([r["missing"] >= COLLAPSE_MISSING for r in rows]))
                if rows else float("nan"),
                fp_iou=med("fp_iou"), spill=med("spill"), vol_iou=med("vol_iou"))


def verdict(arms: dict, pop: str) -> dict:
    """The pre-registered bar, evaluated mechanically so the write-up cannot soften it."""
    out = {}
    bo, nn = arms["blockout"][pop], arms["nn_retrieval"][pop]
    for name, a in arms.items():
        if name in ("blockout", "nn_retrieval"):
            continue
        s = a[pop]
        out[name] = dict(
            beats_1nn_extra=bool(s["extra"] < nn["extra"]),
            collapse_no_worse_than_1nn=bool(s["collapse_rate"] <= nn["collapse_rate"]),
            moved=bool(s["vs_input"] < 0.98),
            killed_identity=bool(s["extra"] >= bo["extra"]),
        )
        out[name]["pass"] = bool(out[name]["beats_1nn_extra"] and
                                 out[name]["collapse_no_worse_than_1nn"] and out[name]["moved"])
    return out


def montage(cases, out: Path, cell: int = 5) -> Path:
    """Real building beside every arm, as shaded massing. The human's criterion, not a number.

    #10 recorded three separate occasions where reading a picture corrected a conclusion the scalar
    metric supported, so the arms are rendered side by side on the same buildings rather than
    summarised.
    """
    from PIL import Image, ImageDraw

    names = list(cases[0]["arms"])
    tiles = [[render_iso(c["target"], c["fp"], cell)] +
             [render_iso(c["arms"][n], c["fp"], cell) for n in names] for c in cases]
    tw = max(t.width for row in tiles for t in row)
    th = max(t.height for row in tiles for t in row)
    head, pad, lab, cols = 26, 8, 34, len(names) + 1
    sheet = Image.new("RGB", (cols * tw + (cols + 1) * pad,
                              head + len(tiles) * (th + lab)), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for j, title in enumerate(["REAL BUILDING"] + [n.upper() for n in names]):
        d.text((pad + j * (tw + pad), 8), title, fill=(0, 0, 0))
    for i, row in enumerate(tiles):
        y = head + i * (th + lab)
        for j, t in enumerate(row):
            sheet.paste(t, (pad + j * (tw + pad) + (tw - t.width) // 2, y + (th - t.height) // 2))
        c = cases[i]
        d.text((pad, y + th + 4), f"id {c['id']}   " + "   ".join(
            f"{n} extra {c['extra'][n]:.3f}" for n in names), fill=(40, 40, 40))
        d.line([(0, y + th + lab - 2), (sheet.width, y + th + lab - 2)], fill=(225, 225, 228))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def report(res: dict) -> None:
    print("\n" + "=" * 100)
    print("the aggregate is right of the bar: #126 demoted it, so it may not head the row")
    for pop, label in (("carve", "CARVE-NEEDING buildings -- the population the bar is set on"),
                       ("flat", "ALREADY-FLAT buildings -- reported, never pooled"),
                       ("all", "all pinned buildings")):
        print(f"\n== {label} (n={res['arms']['blockout'][pop]['n']}) ==")
        print(f"{'arm':16s} {'miss':>7} {'extra':>7} {'vs_inp':>7} {'collapse':>9} "
              f"{'>env:xtr':>9} {'fp_iou':>7} | {'(3D IoU)':>9}")
        for name, a in res["arms"].items():
            s = a[pop]
            w = a["beats_envelope_extra"][pop]["rate_ex_ties"]
            print(f"{name:16s} {s['missing']:>7.4f} {s['extra']:>7.4f} {s['vs_input']:>7.4f} "
                  f"{s['collapse_rate']:>9.4f} {w:>9.3f} {s['fp_iou']:>7.4f} | "
                  f"{s['vol_iou']:>9.4f}")
    print("\n== the pre-registered bar, on the carve-needing subset ==")
    for name, v in res["verdict"].items():
        print(f"  {name:16s} beats 1-NN `extra` {str(v['beats_1nn_extra']):>5}   "
              f"collapse ok {str(v['collapse_no_worse_than_1nn']):>5}   "
              f"moved {str(v['moved']):>5}   ->  {'PASS' if v['pass'] else 'NOT MET'}"
              + ("   [KILL: identity]" if v["killed_identity"] else ""))
    print("=" * 100)


# ==================================================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids_from", default=str(SHIP714))
    ap.add_argument("--objective", default="ce", choices=("ce", "mse"))
    ap.add_argument("--tag", default=None, help="run name; defaults to the objective")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_aug", action="store_true", help="disable the 8 plan symmetries")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--rebuild_cache", action="store_true")
    ap.add_argument("--ckpt", nargs="*", default=None,
                    help="score these checkpoints instead of training (name=path or path)")
    ap.add_argument("--montage", type=int, default=6, help="buildings per sheet; 0 disables")
    ap.add_argument("--out", default="execution/artifacts/height_map_generator_714.json")
    args = ap.parse_args()
    args.tag = args.tag or f"heightmap_{args.objective}"

    cache = build_cache(force=args.rebuild_cache)

    ckpts = {}
    if args.ckpt:
        for spec in args.ckpt:
            name, _, path = spec.rpartition("=")
            ckpts[name or Path(path).stem] = Path(path)
    else:
        ckpts[args.tag] = train(cache, args)

    # ---- the pinned population, in the pinned order -------------------------------------------
    ids = [int(i) for i in json.load(open(args.ids_from))["ids"]]
    row_to_idx = {int(r): i for i, r in enumerate(cache["row"])}
    sel = np.array([row_to_idx[i] for i in ids if i in row_to_idx and cache["ok"][row_to_idx[i]]])
    held = {k: cache[k][sel] for k in ("row", "fp", "target", "y0", "extent", "region", "height_m")}
    held["fp"] = held["fp"] > 0
    held["target"] = held["target"].astype(np.int16)
    print(f"[ids] {len(sel)} pinned buildings from {args.ids_from}", flush=True)

    # ---- the arms -------------------------------------------------------------------------------
    train_idx = np.nonzero((cache["ok"] > 0) & (cache["held"] == 0))[0]
    bank_fp = cache["fp"][train_idx] > 0
    bank_target = cache["target"][train_idx].astype(np.int16)
    bank_extent = cache["extent"][train_idx].astype(np.int32)
    print(f"[bank] {len(train_idx)} training buildings for retrieval and for the mean roof",
          flush=True)

    heights = {"blockout": np.stack([apply_depth(held["fp"][i], int(held["extent"][i]),
                                                 envelope_depth(held["fp"][i]))
                                     for i in range(len(sel))])}

    bank_depth = np.stack([carve_depth(bank_target[i], bank_fp[i], int(bank_extent[i]))
                           for i in range(len(train_idx))])
    profile = mean_relative_depth(bank_depth, bank_fp, bank_extent)
    heights["mean_roof"] = np.stack([mean_roof_height(profile, held["fp"][i],
                                                      int(held["extent"][i]))
                                     for i in range(len(sel))])

    t0 = time.time()
    nn = retrieve_nn(held["fp"], bank_fp)
    heights["nn_retrieval"] = np.stack([
        transplant_height(bank_target[j], bank_fp[j], int(bank_extent[j]),
                          held["fp"][i], int(held["extent"][i]))
        for i, j in enumerate(nn)])
    print(f"[1-NN] retrieved in {time.time()-t0:.0f}s  "
          f"(median footprint IoU to the retrieved row reported in the artifact)", flush=True)

    for name, path in ckpts.items():
        heights[name] = predict(path, held, cpu=args.cpu)

    # ---- score, split by population, never pooled -----------------------------------------------
    rows = {name: score_arm(h, held) for name, h in heights.items()}
    carve_mask = np.array([r["blockout_extra"] >= CARVE_NEEDED for r in rows["blockout"]])
    pops = dict(all=np.ones(len(carve_mask), bool), carve=carve_mask, flat=~carve_mask)
    env = {p: [dict(blockout=dict(extra=r["extra"], vol_iou=r["vol_iou"]))
               for r, m in zip(rows["blockout"], pops[p]) if m] for p in pops}

    arms = {}
    for name, rr in rows.items():
        a = {p: summarise([r for r, m in zip(rr, pops[p]) if m]) for p in pops}
        a["beats_envelope_extra"] = {}
        a["beats_envelope_iou"] = {}
        for p in pops:
            paired = [dict(arm=dict(extra=r["extra"], vol_iou=r["vol_iou"]), **e)
                      for r, m, e in zip(rr, pops[p], env[p]) if m]
            a["beats_envelope_extra"][p] = compare_to_envelope(paired, "arm", "extra", False)
            a["beats_envelope_iou"][p] = compare_to_envelope(paired, "arm", "vol_iou", True)
        arms[name] = a

    res = dict(
        meta=dict(created=time.strftime("%Y-%m-%dT%H:%M:%S"), question="#127",
                  ids_from=args.ids_from, gt_h5=str(H5.relative_to(REPO)),
                  n_pinned=len(sel), n_carve=int(carve_mask.sum()),
                  n_train=len(train_idx), depth_classes=DEPTH_CLASSES,
                  checkpoints={k: str(v) for k, v in ckpts.items()},
                  epochs=args.epochs, batch=args.batch, lr=args.lr, width=args.width,
                  seed=args.seed, augment=not args.no_aug),
        arms=arms, verdict=verdict(arms, "carve"),
        nn_footprint_iou=float(np.median([
            float((held["fp"][i] & bank_fp[j]).sum()) / max(float((held["fp"][i] | bank_fp[j]).sum()), 1)
            for i, j in enumerate(nn)])),
        per_building={name: rr for name, rr in rows.items()},
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    report(res)
    print(f"\n[artifact] {out}")

    if args.montage:
        model_names = [n for n in heights if n in ckpts]
        key = model_names[0] if model_names else "nn_retrieval"
        idx = [i for i in range(len(sel)) if carve_mask[i]]
        by = sorted(idx, key=lambda i: rows[key][i]["extra"])
        picks = dict(best=by[:args.montage],
                     representative=by[len(by) // 2:len(by) // 2 + args.montage],
                     worst=by[-args.montage:])
        for tag, sub in picks.items():
            cases = [dict(id=int(held["row"][i]), fp=held["fp"][i], target=held["target"][i],
                          arms={n: heights[n][i] for n in heights},
                          extra={n: rows[n][i]["extra"] for n in heights}) for i in sub]
            if cases:
                print(f"[montage] {montage(cases, WORK / f'{tag}.png')}")


if __name__ == "__main__":
    main()
