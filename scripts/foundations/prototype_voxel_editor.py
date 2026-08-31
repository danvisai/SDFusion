"""Throwaway feasibility prototype for an authentic-output, bounded voxel editor.

Question, and only question:

    Can a small occupancy-space editor improve real A2 outputs without collapsing the
    building or learning to return its input?

This deliberately starts with a deterministic classifier rather than another diffusion
model.  If a tiny 3-D UNet cannot learn useful KEEP / ADD / REMOVE actions from aligned,
authentic A2 outputs, adding a diffusion objective only makes the negative result more
expensive and harder to interpret.

The four commands form one falsifiable experiment::

    # CPU-only plumbing check; does not load A2/Dora or touch the training GPU.
    ./venv/bin/python scripts/foundations/prototype_voxel_editor.py smoke

    # GPU: create authentic (A2 decoded field, real occupancy) pairs.
    ./venv/bin/python scripts/foundations/prototype_voxel_editor.py cache \
      --out outputs/voxel_editor_prototype/pairs.h5 --n-train 384 --n-val 96

    # GPU recommended, CPU possible: fit the bounded editor.
    ./venv/bin/python scripts/foundations/prototype_voxel_editor.py train \
      --cache outputs/voxel_editor_prototype/pairs.h5

    # Fixed held-out report. Re-cache with --n-val 714 for the final gate.
    ./venv/bin/python scripts/foundations/prototype_voxel_editor.py evaluate \
      --cache outputs/voxel_editor_prototype/pairs.h5 \
      --checkpoint outputs/voxel_editor_prototype/editor.pth

The cache branches from ``DoraCodec.decode_grid`` directly.  It never meshes the field and
voxelises it again.  Meshing remains a visualisation/export concern, not a data conversion.

This file is intentionally self-contained and named ``prototype``.  It is not a production
model, service, dataset format, or new architectural commitment.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RES = 64
DEFAULT_A2 = REPO / "weights/massing-vecset/vecset_v4_surf.pth"
DEFAULT_LATENTS = REPO / "data/real_massing_v1/vecset_latents.h5"
DEFAULT_REAL = REPO / "data/real_massing_v1/real.h5"
DEFAULT_OUT = REPO / "outputs/voxel_editor_prototype"


# -------------------------------------------------------------------------------------------------
# Pure occupancy logic.  These functions are also what the synthetic smoke path exercises.
# Array axes follow this repository's convention: [z, y, x], with y vertical.
# -------------------------------------------------------------------------------------------------


def volume_metrics(candidate: np.ndarray, target: np.ndarray) -> dict[str, float]:
    candidate, target = np.asarray(candidate, bool), np.asarray(target, bool)
    inter = int((candidate & target).sum())
    union = int((candidate | target).sum())
    target_volume = max(int(target.sum()), 1)
    return {
        "vol_iou": float(inter / union) if union else 1.0,
        "missing": float((target & ~candidate).sum() / target_volume),
        "extra": float((candidate & ~target).sum() / target_volume),
    }


def occupancy_iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    union = int((a | b).sum())
    return float((a & b).sum() / union) if union else 1.0


def vertical_extent(occ: np.ndarray) -> tuple[int, int] | None:
    ys = np.nonzero(np.asarray(occ, bool).any(axis=(0, 2)))[0]
    return (int(ys.min()), int(ys.max())) if len(ys) else None


def envelope_occupancy(footprint: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Extrude the conditioning footprint across the target's specified vertical extent."""
    out = np.zeros_like(target, dtype=bool)
    ext = vertical_extent(target)
    if ext is not None:
        out[:, ext[0] : ext[1] + 1, :] = np.asarray(footprint, bool)[:, None, :]
    return out


def edit_mask(source: np.ndarray, footprint: np.ndarray, mode: str = "roof", band: int = 4,
              roof_fraction: float = 0.40) -> np.ndarray:
    """Source-derived cells the learned editor is allowed to change.

    ``roof`` is the first experiment: a near-surface band limited to the upper 40% of each
    occupied column, plus any source spill outside the footprint (which may only be removed).
    ``surface`` permits the same band around all walls and roofs and is a deliberately broader
    ablation.  Both masks are computable at inference; neither sees the target.
    """
    from scipy import ndimage

    source = np.asarray(source, bool)
    footprint = np.asarray(footprint, bool)
    if source.ndim != 3 or footprint.shape != (source.shape[0], source.shape[2]):
        raise ValueError(f"expected source [z,y,x] and footprint [z,x], got {source.shape}, "
                         f"{footprint.shape}")
    if mode not in {"roof", "surface"}:
        raise ValueError("mode must be 'roof' or 'surface'")
    if band < 1:
        raise ValueError("band must be >= 1")

    # A signed EDT is non-zero on both sides; abs(signed) <= band is a genuine local
    # surface neighbourhood.  OR-ing independent inside/outside EDT thresholds would select
    # the whole volume because either distance is zero at every voxel.
    signed = ndimage.distance_transform_edt(~source) - ndimage.distance_transform_edt(source)
    surface = np.abs(signed) <= float(band)
    fp3 = footprint[:, None, :]
    allowed = surface & fp3

    if mode == "roof":
        ny = source.shape[1]
        y = np.arange(ny)[None, :, None]
        occupied_column = source.any(axis=1)
        bottom = np.where(
            occupied_column,
            np.argmax(source, axis=1),
            ny,
        )
        top = np.where(
            occupied_column,
            ny - 1 - np.argmax(source[:, ::-1, :], axis=1),
            -1,
        )
        depth = np.maximum(top - bottom + 1, 1)
        roof_depth = np.maximum(band, np.ceil(depth * roof_fraction).astype(int))
        roof_floor = top - roof_depth + 1
        roof_zone = occupied_column[:, None, :] & (y >= roof_floor[:, None, :])
        roof_zone &= y <= (top + band)[:, None, :]
        allowed &= roof_zone

    # Footprint violations are never allowed to persist.  They are handled by a deterministic
    # clamp and scored as a separate baseline so the learned model cannot claim the gain.
    allowed |= source & ~fp3
    return allowed


def action_target(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Voxel labels: 0 KEEP, 1 ADD, 2 REMOVE."""
    source, target = np.asarray(source, bool), np.asarray(target, bool)
    action = np.zeros(source.shape, np.uint8)
    action[~source & target] = 1
    action[source & ~target] = 2
    return action


def apply_actions(source: np.ndarray, actions: np.ndarray, allowed: np.ndarray,
                  footprint: np.ndarray) -> np.ndarray:
    """Apply learned actions while making the locality/footprint constraints exact."""
    out = np.asarray(source, bool).copy()
    actions, allowed = np.asarray(actions), np.asarray(allowed, bool)
    out[allowed & (actions == 1)] = True
    out[allowed & (actions == 2)] = False
    out &= np.asarray(footprint, bool)[:, None, :]
    return out


def allowed_delta_coverage(source: np.ndarray, target: np.ndarray, allowed: np.ndarray,
                           footprint: np.ndarray) -> float:
    """Fraction of desired source->target changes representable by this mask + hard clamp."""
    delta = np.asarray(source, bool) ^ np.asarray(target, bool)
    if not delta.any():
        return 1.0
    representable = np.asarray(allowed, bool) | ~np.asarray(footprint, bool)[:, None, :]
    return float((delta & representable).sum() / delta.sum())


def occupancy_to_sdf(occ: np.ndarray) -> np.ndarray:
    """Crisp metric signed EDT for export/montage; negative is inside."""
    from scipy import ndimage

    occ = np.asarray(occ, bool)
    spacing = 2.0 / max(occ.shape[0] - 1, 1)
    return ((ndimage.distance_transform_edt(~occ) - ndimage.distance_transform_edt(occ))
            * spacing).astype(np.float32)


# -------------------------------------------------------------------------------------------------
# Tiny diagnostic model.  Imports are lazy so cache inspection/pure logic remain lightweight.
# -------------------------------------------------------------------------------------------------


def build_model(base: int = 8):
    import torch
    from torch import nn

    class Block(nn.Module):
        def __init__(self, cin: int, cout: int):
            super().__init__()
            groups = 4 if cout % 4 == 0 else 1
            self.net = nn.Sequential(
                nn.Conv3d(cin, cout, 3, padding=1), nn.GroupNorm(groups, cout), nn.SiLU(),
                nn.Conv3d(cout, cout, 3, padding=1), nn.GroupNorm(groups, cout), nn.SiLU(),
            )

        def forward(self, x):
            return self.net(x)

    class ActionUNet(nn.Module):
        """Eight input channels -> KEEP/ADD/REMOVE logits at the same resolution."""
        def __init__(self):
            super().__init__()
            c = base
            self.e1 = Block(8, c)
            self.e2 = Block(c, 2 * c)
            self.b = Block(2 * c, 4 * c)
            self.pool = nn.MaxPool3d(2)
            self.u2 = nn.ConvTranspose3d(4 * c, 2 * c, 2, 2)
            self.d2 = Block(4 * c, 2 * c)
            self.u1 = nn.ConvTranspose3d(2 * c, c, 2, 2)
            self.d1 = Block(2 * c, c)
            self.out = nn.Conv3d(c, 3, 1)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
            with torch.no_grad():
                self.out.bias[0] = 2.0  # begin as KEEP, then earn every edit

        def forward(self, x):
            e1 = self.e1(x)
            e2 = self.e2(self.pool(e1))
            b = self.b(self.pool(e2))
            d2 = self.d2(torch.cat((self.u2(b), e2), dim=1))
            d1 = self.d1(torch.cat((self.u1(d2), e1), dim=1))
            return self.out(d1)

    return ActionUNet()


def model_input(source_field: np.ndarray, footprint: np.ndarray, height_m: float,
                region: int) -> np.ndarray:
    """Build [occ, TSDF, footprint, y, height, region-one-hot] = 8 channels."""
    field = np.asarray(source_field, np.float32)
    source = field <= 0
    nz, ny, nx = source.shape
    fp3 = np.broadcast_to(np.asarray(footprint, np.float32)[:, None, :], source.shape)
    y = np.broadcast_to(np.linspace(-1, 1, ny, dtype=np.float32)[None, :, None], source.shape)
    h = np.full(source.shape, np.clip(float(height_m) / 30.0, 0.0, 2.0), np.float32)
    region_channels = [np.full(source.shape, float(region == r), np.float32) for r in range(3)]
    return np.stack([
        source.astype(np.float32) * 2.0 - 1.0,
        np.tanh(field / 0.10),
        fp3,
        y,
        h,
        *region_channels,
    ])


@dataclass
class Example:
    row: int
    region: int
    height_m: float
    footprint: np.ndarray
    source_field: np.ndarray
    target: np.ndarray
    envelope: np.ndarray

    @property
    def source(self) -> np.ndarray:
        return self.source_field <= 0


class VoxelCache:
    """Small, single-process reader; HDF5 handles are never shared across DataLoader workers."""
    def __init__(self, path: Path, split: int):
        import h5py

        self.path = Path(path)
        with h5py.File(self.path, "r") as h:
            self.indices = np.flatnonzero(h["split"][:] == split)

    def __len__(self):
        return len(self.indices)

    def get(self, i: int) -> Example:
        import h5py

        j = int(self.indices[i])
        with h5py.File(self.path, "r") as h:
            return Example(
                row=int(h["row"][j]),
                region=int(h["region"][j]),
                height_m=float(h["height_m"][j]),
                footprint=np.asarray(h["footprint"][j], bool),
                source_field=np.asarray(h["source_field"][j], np.float32),
                target=np.asarray(h["target_occ"][j], bool),
                envelope=np.asarray(h["envelope_occ"][j], bool),
            )


def tensor_example(ex: Example, mask_mode: str, band: int, roof_fraction: float,
                   augment: bool = False):
    import torch

    inp = model_input(ex.source_field, ex.footprint, ex.height_m, ex.region)
    mask = edit_mask(ex.source, ex.footprint, mask_mode, band, roof_fraction)
    labels = action_target(ex.source, ex.target)
    if augment:
        if np.random.random() < 0.5:
            inp, mask, labels = inp[..., ::-1].copy(), mask[..., ::-1].copy(), labels[..., ::-1].copy()
        if np.random.random() < 0.5:
            inp, mask, labels = inp[:, ::-1].copy(), mask[::-1].copy(), labels[::-1].copy()
    return (torch.from_numpy(inp), torch.from_numpy(labels.astype(np.int64)),
            torch.from_numpy(mask), ex)


# -------------------------------------------------------------------------------------------------
# Authentic A2 pair cache
# -------------------------------------------------------------------------------------------------


def _round_robin(by_region: dict[int, list[int]], limit: int) -> list[int]:
    out: list[int] = []
    keys = sorted(by_region)
    pos = {k: 0 for k in keys}
    while len(out) < limit:
        moved = False
        for key in keys:
            if pos[key] < len(by_region[key]) and len(out) < limit:
                out.append(by_region[key][pos[key]])
                pos[key] += 1
                moved = True
        if not moved:
            break
    return out


def select_rows(latents: Path, real: Path, n_train: int, n_val: int, identity_fraction: float,
                seed: int) -> list[tuple[int, int]]:
    """Return (row, split), limiting exact envelope=GT rows in training only.

    Validation remains an unbiased, region-round-robin held-out slice.  Training first fills an
    equal-per-region opportunity pool, then admits a small explicit identity quota.  This prevents
    PLATEAU's zero-edit pairs from teaching KEEP as the entire task.
    """
    import h5py

    rng = np.random.default_rng(seed)
    with h5py.File(latents, "r") as h:
        rows = h["row"][:].astype(int)
        held = h["held_out"][:].astype(bool)
        regions = h["region"][:].astype(int)
        footprints = h["footprint"][:]
    index_of = {int(row): i for i, row in enumerate(rows)}

    by_region_val: dict[int, list[int]] = {}
    for row, is_held, region in zip(rows, held, regions):
        if is_held:
            by_region_val.setdefault(int(region), []).append(int(row))
    for values in by_region_val.values():
        rng.shuffle(values)
    val = _round_robin(by_region_val, n_val)

    want_identity = min(int(round(n_train * identity_fraction)), n_train)
    want_opportunity = n_train - want_identity
    opportunity: dict[int, list[int]] = {0: [], 1: [], 2: []}
    identity: list[int] = []
    candidates = rows[~held].copy()
    rng.shuffle(candidates)
    with h5py.File(real, "r") as h:
        for row_value in candidates:
            row = int(row_value)
            li = index_of[row]
            region = int(regions[li])
            target = np.asarray(h["sdf"][row]) <= 0
            envelope = envelope_occupancy(footprints[li], target)
            if np.array_equal(envelope, target):
                if len(identity) < want_identity:
                    identity.append(row)
            elif sum(len(v) for v in opportunity.values()) < want_opportunity:
                opportunity[region].append(row)
            if (sum(len(v) for v in opportunity.values()) >= want_opportunity
                    and len(identity) >= want_identity):
                break
    train = _round_robin(opportunity, want_opportunity) + identity[:want_identity]
    rng.shuffle(train)
    if len(train) < n_train or len(val) < n_val:
        raise RuntimeError(f"could only select train={len(train)}/{n_train}, val={len(val)}/{n_val}")
    return [(row, 0) for row in train] + [(row, 1) for row in val]


def _create_cache(path: Path, attrs: dict) -> None:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h:
        h.attrs.update(attrs)
        scalar_specs = {
            "row": "i8", "split": "u1", "region": "i1", "height_m": "f4",
        }
        for name, dtype in scalar_specs.items():
            h.create_dataset(name, shape=(0,), maxshape=(None,), chunks=(128,), dtype=dtype)
        h.create_dataset("footprint", shape=(0, RES, RES), maxshape=(None, RES, RES),
                         chunks=(1, RES, RES), compression="gzip", dtype="u1")
        for name, dtype in (("source_field", "f2"), ("target_occ", "u1"),
                            ("envelope_occ", "u1")):
            h.create_dataset(name, shape=(0, RES, RES, RES), maxshape=(None, RES, RES, RES),
                             chunks=(1, RES, RES, RES), compression="gzip",
                             compression_opts=1, dtype=dtype)


def _append_cache(path: Path, values: dict) -> None:
    import h5py

    with h5py.File(path, "a") as h:
        n = len(h["row"])
        for name, value in values.items():
            ds = h[name]
            ds.resize(n + 1, axis=0)
            ds[n] = value
        h.flush()


def cache_command(args) -> None:
    import h5py
    import torch

    out = Path(args.out)
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists; pass --overwrite to replace this prototype cache")
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise SystemExit("CUDA/ROCm is not visible. Run cache when the A2 GPU is available, or "
                         "explicitly request --device cpu (very slow).")
    selected = select_rows(Path(args.latents), Path(args.real), args.n_train, args.n_val,
                           args.identity_fraction, args.seed)
    _create_cache(out, {
        "purpose": "throwaway authentic A2 -> real occupancy editor prototype",
        "a2_checkpoint": str(Path(args.a2).resolve()),
        "strength": args.strength,
        "steps": args.steps,
        "guidance": args.guidance,
        "seed": args.seed,
        "identity_fraction_train": args.identity_fraction,
    })

    from models.networks.vecset_denoiser import VecsetDenoiser
    from models.networks.vecset_projection import SetSDEdit
    from models.shape_codec import Building, DoraCodec
    from scripts.foundations.dora_roundtrip_probe import load_dora
    from scripts.foundations.eval_massing_arms import blockout_sdf
    from scripts.foundations.baseline_gate_eval import mesh_sdf_surface
    from scripts.foundations.vecset_ceiling_probe import TRUNC, verts_to_world

    device = args.device
    ck = torch.load(args.a2, map_location="cpu", weights_only=False)
    ca = ck["args"]
    net = VecsetDenoiser(latent_channels=ck["latent_channels"], width=ca["width"],
                         depth=ca["depth"], heads=ca["heads"],
                         footprint_res=ck["footprint_res"]).to(device)
    net.load_state_dict(ck["model"])
    net.eval()
    op = SetSDEdit(net, timesteps=ca["timesteps"])
    mu, sd = ck["latent_mu"].to(device), ck["latent_sd"].to(device)
    codec = DoraCodec(load_dora(device))

    with h5py.File(args.latents, "r") as lat, h5py.File(args.real, "r") as real:
        latent_row = {int(row): i for i, row in enumerate(lat["row"][:])}
        t0 = time.time()
        for k, (row, split) in enumerate(selected):
            li = latent_row[row]
            fp = np.asarray(lat["footprint"][li], bool)
            target = np.asarray(real["sdf"][row], np.float32) <= 0
            ext = vertical_extent(target)
            if ext is None:
                raise RuntimeError(f"selected row {row} has empty target")
            bo = blockout_sdf(fp, *ext)
            verts, faces = mesh_sdf_surface(np.clip(bo, -TRUNC, TRUNC))
            if verts is None:
                raise RuntimeError(f"selected row {row} has unmeshable envelope")
            codec.reseed(args.seed * 1000003 + row)
            z0 = (codec.encode(Building(verts=verts_to_world(verts), faces=faces)).float() - mu) / sd
            fpt = torch.from_numpy(fp.astype(np.float32))[None, None].to(device)
            height = torch.tensor([float(lat["height_m"][li])], device=device)
            region = torch.tensor([int(lat["region"][li])], device=device)
            zp = op.project(z0, fpt, height, region, strength=args.strength,
                            steps=args.steps, guidance=args.guidance,
                            seed=args.seed * 1000003 + row)
            with torch.no_grad():
                source_field = codec.decode_grid(zp * sd + mu, RES).cpu().numpy()[0, 0]
            _append_cache(out, {
                "row": row,
                "split": split,
                "region": int(lat["region"][li]),
                "height_m": float(lat["height_m"][li]),
                "footprint": fp.astype(np.uint8),
                "source_field": source_field.astype(np.float16),
                "target_occ": target.astype(np.uint8),
                "envelope_occ": (bo <= 0).astype(np.uint8),
            })
            if (k + 1) % 10 == 0 or k + 1 == len(selected):
                print(f"[cache] {k+1}/{len(selected)}  {time.time()-t0:.0f}s", flush=True)
    print(f"[cache] wrote authentic pairs to {out}")


# -------------------------------------------------------------------------------------------------
# Train and evaluate
# -------------------------------------------------------------------------------------------------


def _batch(examples, mask_mode: str, band: int, roof_fraction: float, augment: bool):
    import torch

    items = [tensor_example(ex, mask_mode, band, roof_fraction, augment) for ex in examples]
    return (torch.stack([x[0] for x in items]), torch.stack([x[1] for x in items]),
            torch.stack([x[2] for x in items]))


def train_command(args) -> None:
    import torch
    import torch.nn.functional as F

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    ds = VoxelCache(Path(args.cache), split=0)
    if not len(ds):
        raise SystemExit("cache has no training rows")
    model = build_model(args.base).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # KEEP is abundant. ADD and REMOVE must each carry enough weight to be learnable, but the
    # editor starts with a KEEP bias and every prediction is still hard-clamped by the mask.
    class_weight = torch.tensor([0.12, 1.0, 1.0], device=device)
    rng = np.random.default_rng(args.seed)
    losses = []
    t0 = time.time()
    model.train()
    for step in range(args.steps):
        indices = rng.integers(0, len(ds), size=args.batch)
        examples = [ds.get(int(i)) for i in indices]
        x, target, mask = _batch(examples, args.mask, args.band, args.roof_fraction, augment=True)
        x, target, mask = x.to(device), target.to(device), mask.to(device)
        logits = model(x)
        per_voxel = F.cross_entropy(logits, target, weight=class_weight, reduction="none")
        loss = (per_voxel * mask).sum() / mask.sum().clamp_min(1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % args.log_every == 0 or step == args.steps - 1:
            print(f"[train {step+1}/{args.steps}] loss={losses[-1]:.5f} "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)

    checkpoint = Path(args.out)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "base": args.base,
        "mask": args.mask,
        "band": args.band,
        "roof_fraction": args.roof_fraction,
        "cache": str(Path(args.cache).resolve()),
        "steps": args.steps,
        "seed": args.seed,
        "loss_first": losses[0],
        "loss_last_20": float(np.mean(losses[-20:])),
    }, checkpoint)
    print(f"[train] wrote {checkpoint}")
    evaluate_model(model, VoxelCache(Path(args.cache), split=1), device, args.mask, args.band,
                   args.roof_fraction, Path(args.report))


def evaluate_command(args) -> None:
    import torch

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(int(ck["base"]))
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    evaluate_model(model, VoxelCache(Path(args.cache), split=1), device,
                   str(ck["mask"]), int(ck["band"]), float(ck["roof_fraction"]),
                   Path(args.report))


def evaluate_model(model, ds: VoxelCache, device: str, mask_mode: str, band: int,
                   roof_fraction: float, report_path: Path) -> dict:
    import torch

    if not len(ds):
        raise SystemExit("cache has no held-out rows")
    rows = []
    model.eval()
    with torch.no_grad():
        for i in range(len(ds)):
            ex = ds.get(i)
            x, _, _, _ = tensor_example(ex, mask_mode, band, roof_fraction)
            actions = model(x[None].to(device)).argmax(1).cpu().numpy()[0]
            allowed = edit_mask(ex.source, ex.footprint, mask_mode, band, roof_fraction)
            sanitized = ex.source & ex.footprint[:, None, :]
            edited = apply_actions(ex.source, actions, allowed, ex.footprint)
            source_metrics = volume_metrics(ex.source, ex.target)
            clean_metrics = volume_metrics(sanitized, ex.target)
            editor_metrics = volume_metrics(edited, ex.target)
            envelope_metrics = volume_metrics(ex.envelope, ex.target)
            outside_changed = (edited != sanitized) & ~allowed
            row = {
                "row": ex.row,
                "region": ex.region,
                "source": source_metrics,
                "sanitized_source": clean_metrics,
                "editor": editor_metrics,
                "envelope": envelope_metrics,
                "vs_source": occupancy_iou(edited, ex.source),
                "vs_sanitized_source": occupancy_iou(edited, sanitized),
                "allowed_delta_coverage": allowed_delta_coverage(
                    ex.source, ex.target, allowed, ex.footprint),
                "outside_mask_violations": int(outside_changed.sum()),
                "identity_envelope": bool(np.array_equal(ex.envelope, ex.target)),
            }
            rows.append(row)
            if (i + 1) % 20 == 0:
                print(f"[eval] {i+1}/{len(ds)}", flush=True)

    def med(path):
        arm, metric = path
        return float(np.median([r[arm][metric] for r in rows]))

    opportunity = [r for r in rows if not r["identity_envelope"]]
    identity = [r for r in rows if r["identity_envelope"]]
    strict_win_source = float(np.mean([
        r["editor"]["vol_iou"] > r["source"]["vol_iou"] for r in rows]))
    strict_win_clean = float(np.mean([
        r["editor"]["vol_iou"] > r["sanitized_source"]["vol_iou"] for r in rows]))
    strict_win_opp = float(np.mean([
        r["editor"]["vol_iou"] > r["sanitized_source"]["vol_iou"]
        for r in opportunity])) if opportunity else 0.0
    strict_win_envelope = float(np.mean([
        r["editor"]["vol_iou"] > r["envelope"]["vol_iou"] for r in rows]))
    collapse_source = float(np.mean([r["source"]["missing"] >= 0.15 for r in rows]))
    collapse_clean = float(np.mean([
        r["sanitized_source"]["missing"] >= 0.15 for r in rows]))
    collapse_editor = float(np.mean([r["editor"]["missing"] >= 0.15 for r in rows]))
    source_med, editor_med = med(("source", "vol_iou")), med(("editor", "vol_iou"))
    clean_med = med(("sanitized_source", "vol_iou"))
    delta_source = float(np.median([
        r["editor"]["vol_iou"] - r["source"]["vol_iou"] for r in rows]))
    delta_clean = float(np.median([
        r["editor"]["vol_iou"] - r["sanitized_source"]["vol_iou"] for r in rows]))
    vs_source = float(np.median([r["vs_source"] for r in rows]))
    vs_clean = float(np.median([r["vs_sanitized_source"] for r in rows]))
    coverage_median = float(np.median([r["allowed_delta_coverage"] for r in rows]))
    identity_delta = (float(np.median([
        r["editor"]["vol_iou"] - r["sanitized_source"]["vol_iou"] for r in identity
    ])) if identity else None)
    preservation = int(sum(r["outside_mask_violations"] for r in rows)) == 0
    small_gate = {
        "held_out_n_at_least_96": len(rows) >= 96,
        "median_paired_delta_iou_over_sanitized_source_at_least_0.01": delta_clean >= 0.01,
        "opportunity_win_rate_at_least_0.55": strict_win_opp >= 0.55,
        "collapse_increase_over_sanitized_source_at_most_0.02": collapse_editor <= collapse_clean + 0.02,
        "outside_mask_exactly_preserved": preservation,
        "not_a_noop_vs_sanitized_source_below_0.99": vs_clean < 0.99,
        "identity_rows_not_degraded_over_0.005": identity_delta is None or identity_delta >= -0.005,
        "allowed_target_delta_coverage_at_least_0.70": coverage_median >= 0.70,
    }
    summary = {
        "n": len(rows),
        "n_opportunity": len(opportunity),
        "n_identity_envelope": len(identity),
        "source_vol_iou_median": source_med,
        "sanitized_source_vol_iou_median": clean_med,
        "editor_vol_iou_median": editor_med,
        "envelope_vol_iou_median": med(("envelope", "vol_iou")),
        "editor_paired_delta_vs_source_median": delta_source,
        "editor_paired_delta_vs_sanitized_source_median": delta_clean,
        "strict_win_rate_vs_source": strict_win_source,
        "strict_win_rate_vs_sanitized_source": strict_win_clean,
        "strict_win_rate_vs_sanitized_source_opportunity": strict_win_opp,
        "strict_beats_envelope_rate": strict_win_envelope,
        "source_collapse_rate": collapse_source,
        "sanitized_source_collapse_rate": collapse_clean,
        "editor_collapse_rate": collapse_editor,
        "vs_source_median": vs_source,
        "vs_sanitized_source_median": vs_clean,
        "identity_editor_delta_median": identity_delta,
        "allowed_delta_coverage_median": coverage_median,
        "small_gate": small_gate,
        "small_gate_pass": bool(all(small_gate.values())),
        "full_gate_note": "Re-run on all 714 held-out rows; require >5% strict beats-envelope, "
                          "no collapse regression, exact mask preservation, SNE and montage review.",
    }
    artifact = {"summary": summary, "rows": rows}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"[eval] wrote {report_path}")
    return artifact


# -------------------------------------------------------------------------------------------------
# CPU synthetic smoke: validates the data/action/mask/training plumbing, not the research claim.
# -------------------------------------------------------------------------------------------------


def synthetic_examples(n: int = 12, res: int = 16, seed: int = 0) -> list[Example]:
    rng = np.random.default_rng(seed)
    examples = []
    for i in range(n):
        fp = np.zeros((res, res), bool)
        margin = 2 + i % 2
        fp[margin : res - margin, margin : res - margin] = True
        source = np.zeros((res, res, res), bool)
        source[:, 2:13, :] = fp[:, None, :]
        target = source.copy()
        zz, _, xx = np.indices(target.shape)
        # Vary the ridge direction.  The target roof removes the upper corners of a flat box.
        horizontal = zz if i % 2 else xx
        centre = (res - 1) / 2
        roof_top = 12 - (np.abs(horizontal - centre) / 2.5).astype(int)
        yy = np.arange(res)[None, :, None]
        target &= yy <= roof_top
        if i % 3 == 0:
            target[4:7, 9:13, 4:7] = False  # small learned recess, still inside the mask
        source_field = occupancy_to_sdf(source)
        source_field += rng.normal(0, 0.002, source.shape).astype(np.float32)
        examples.append(Example(i, i % 3, 8.0 + i % 4, fp, source_field, target,
                                envelope_occupancy(fp, target)))
    return examples


def smoke_command(args) -> None:
    import torch
    import torch.nn.functional as F

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    examples = synthetic_examples(seed=args.seed)
    model = build_model(base=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    weights = torch.tensor([0.12, 1.0, 1.0])
    losses = []
    for step in range(args.steps):
        x, target, mask = _batch(examples, "roof", 3, 0.5, augment=True)
        logits = model(x)
        voxel_loss = F.cross_entropy(logits, target, weight=weights, reduction="none")
        loss = (voxel_loss * mask).sum() / mask.sum().clamp_min(1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    before, after, oracle_after, violations = [], [], [], 0
    model.eval()
    with torch.no_grad():
        for ex in examples:
            x, _, _, _ = tensor_example(ex, "roof", 3, 0.5)
            action = model(x[None]).argmax(1).numpy()[0]
            mask = edit_mask(ex.source, ex.footprint, "roof", 3, 0.5)
            edited = apply_actions(ex.source, action, mask, ex.footprint)
            oracle = apply_actions(ex.source, action_target(ex.source, ex.target), mask,
                                   ex.footprint)
            before.append(volume_metrics(ex.source, ex.target)["vol_iou"])
            after.append(volume_metrics(edited, ex.target)["vol_iou"])
            oracle_after.append(volume_metrics(oracle, ex.target)["vol_iou"])
            sanitized = ex.source & ex.footprint[:, None, :]
            violations += int(((edited != sanitized) & ~mask).sum())
    report = {
        "meaning": "plumbing check only; synthetic roofs are not evidence for the A2 hypothesis",
        "steps": args.steps,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "source_iou_median": float(np.median(before)),
        "edited_iou_median": float(np.median(after)),
        "oracle_action_iou_median": float(np.median(oracle_after)),
        "outside_mask_violations": violations,
    }
    print(json.dumps(report, indent=2))
    if (not np.isfinite(losses[-1]) or losses[-1] >= losses[0] or violations
            or np.median(oracle_after) <= np.median(before)):
        raise SystemExit("smoke failed")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    smoke = sub.add_parser("smoke", help="CPU synthetic plumbing check")
    smoke.add_argument("--steps", type=int, default=6)
    smoke.add_argument("--seed", type=int, default=0)
    smoke.set_defaults(func=smoke_command)

    cache = sub.add_parser("cache", help="GPU: cache authentic A2 output -> real target pairs")
    cache.add_argument("--out", default=str(DEFAULT_OUT / "pairs.h5"))
    cache.add_argument("--a2", default=str(DEFAULT_A2))
    cache.add_argument("--latents", default=str(DEFAULT_LATENTS))
    cache.add_argument("--real", default=str(DEFAULT_REAL))
    cache.add_argument("--n-train", type=int, default=384)
    cache.add_argument("--n-val", type=int, default=96)
    cache.add_argument("--identity-fraction", type=float, default=0.15)
    cache.add_argument("--strength", type=float, default=0.5)
    cache.add_argument("--steps", type=int, default=20)
    cache.add_argument("--guidance", type=float, default=1.0)
    cache.add_argument("--seed", type=int, default=0)
    cache.add_argument("--device", default="cuda")
    cache.add_argument("--overwrite", action="store_true")
    cache.set_defaults(func=cache_command)

    train = sub.add_parser("train", help="fit the deterministic action editor")
    train.add_argument("--cache", default=str(DEFAULT_OUT / "pairs.h5"))
    train.add_argument("--out", default=str(DEFAULT_OUT / "editor.pth"))
    train.add_argument("--report", default=str(DEFAULT_OUT / "validation.json"))
    train.add_argument("--steps", type=int, default=800)
    train.add_argument("--batch", type=int, default=2)
    train.add_argument("--base", type=int, default=8)
    train.add_argument("--lr", type=float, default=2e-4)
    train.add_argument("--mask", choices=("roof", "surface"), default="roof")
    train.add_argument("--band", type=int, default=4)
    train.add_argument("--roof-fraction", type=float, default=0.40)
    train.add_argument("--device", default=None)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--log-every", type=int, default=50)
    train.set_defaults(func=train_command)

    evaluate = sub.add_parser("evaluate", help="score a checkpoint on cached held-out pairs")
    evaluate.add_argument("--cache", default=str(DEFAULT_OUT / "pairs.h5"))
    evaluate.add_argument("--checkpoint", default=str(DEFAULT_OUT / "editor.pth"))
    evaluate.add_argument("--report", default=str(DEFAULT_OUT / "validation.json"))
    evaluate.add_argument("--device", default=None)
    evaluate.set_defaults(func=evaluate_command)
    return ap


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.func(arguments)
