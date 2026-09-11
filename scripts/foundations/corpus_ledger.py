"""#161 -- the row/region/held_out/height_m ledger, split out of `vecset_latents.h5`.

`train_height_map_generator.py`'s `build_cache` never reads a latent -- it only ever wanted four
small per-row facts (identity, source region, held-out flag, height in metres) that happened to be
stored beside the expensive 2048x64 fp16 Dora-VAE latents in `vecset_latents.h5`. That coupling meant
extending the corpus with BuildingWorld rows forced a choice between paying real GPU-hours to
Dora-encode a row nobody needed a latent for, or writing a latent-less row into the latent store and
silently corrupting every consumer that assumes every row there has one (the vecset-denoiser arms
whose eval artifacts are pinned under `execution/artifacts/`).

This module is the ledger on its own: a few small (row, region, held_out, height_m) columns in their
own file, cheap enough to rewrite wholesale on every change. `vecset_latents.h5` keeps its own copies
of these columns too (every existing reader of that file -- `train_vecset.py`, `eval_massing_arms.py`,
the `probe_*`/`prototype_voxel_editor.py` diagnostics -- keeps working unmodified, and none of them
can want a row that lacks a latent in the first place); this module is what lets a row's ledger entry
exist *before*, and independently of, whether it has ever been Dora-encoded.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

LEDGER_PATH = REPO / "data/real_massing_v1/corpus_ledger.h5"
SCHEMA_VERSION = 1
COLUMNS = ("row", "region", "held_out", "height_m")
DTYPES = {"row": np.int32, "region": np.int32, "held_out": np.uint8, "height_m": np.float32}


def _validate(row, region, held_out, height_m) -> dict[str, np.ndarray]:
    """Coerce the four ledger columns to their canonical dtypes and reject anything ambiguous."""
    arrays = {"row": np.asarray(row), "region": np.asarray(region),
             "held_out": np.asarray(held_out), "height_m": np.asarray(height_m)}
    n = len(arrays["row"])
    for name, arr in arrays.items():
        if arr.ndim != 1 or len(arr) != n:
            raise ValueError(f"#161: ledger column '{name}' must be 1-D with length {n} "
                             f"(row's length), got shape {arr.shape}")
    rows = arrays["row"].astype(np.int64, copy=False)
    if len(np.unique(rows)) != n:
        raise ValueError("#161: ledger row ids must be unique")
    held = arrays["held_out"].astype(np.int64, copy=False)
    bad_held = sorted(set(np.unique(held).tolist()) - {0, 1})
    if bad_held:
        raise ValueError(f"#161: held_out must be 0 or 1, found {bad_held}")
    heights = arrays["height_m"].astype(np.float64, copy=False)
    if not np.isfinite(heights).all():
        raise ValueError("#161: height_m contains non-finite values")
    return {name: arrays[name].astype(DTYPES[name], copy=False) for name in COLUMNS}


def write_ledger(row, region, held_out, height_m, *, path: Path = LEDGER_PATH,
                 source: str = "") -> None:
    """Write the ledger, replacing whatever was at `path`.

    The ledger is megabytes, not gigabytes -- unlike `vecset_latents.h5`'s incremental writer, there
    is no reason to append in place, so every write is a clean, wholesale rewrite (via a temp file,
    so a crash mid-write cannot leave a half-written ledger at `path`).
    """
    arrays = _validate(row, region, held_out, height_m)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with h5py.File(tmp, "w") as f:
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["source"] = source
        for name in COLUMNS:
            f.create_dataset(name, data=arrays[name])
    tmp.replace(path)


def read_ledger(path: Path = LEDGER_PATH) -> dict[str, np.ndarray]:
    """Read the ledger back as `{row, region, held_out, height_m}` numpy arrays."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"#161: no corpus ledger at {path}. Build one with "
            f"`precompute_vecset_latents.py --split_ledger` (extracts it from an existing "
            f"vecset_latents.h5) or `corpus_ledger.write_ledger(...)` directly.")
    with h5py.File(path, "r") as f:
        version = int(f.attrs.get("schema_version", 0))
        if version != SCHEMA_VERSION:
            raise ValueError(f"#161: {path} has ledger schema {version}, need {SCHEMA_VERSION}")
        missing = [c for c in COLUMNS if c not in f]
        if missing:
            raise ValueError(f"#161: {path} is missing ledger column(s) {missing}")
        raw = {name: f[name][:] for name in COLUMNS}
    return _validate(**raw)


def append_ledger(row, region, held_out, height_m, *, path: Path = LEDGER_PATH) -> None:
    """Add new rows to the ledger without touching `vecset_latents.h5` or the Dora codec.

    This is #161's actual unblock: a new BuildingWorld row's identity/region/held_out/height_m can be
    recorded the moment its split assignment is decided (#177), at zero GPU cost, with no risk to the
    separately versioned, far larger latent store. Refuses to silently overwrite a row already
    present -- a re-assignment must go through a fresh `write_ledger` call, not an append.
    """
    path = Path(path)
    existing = read_ledger(path) if path.exists() else {c: np.array([], DTYPES[c]) for c in COLUMNS}
    new = _validate(row, region, held_out, height_m)
    overlap = sorted(set(existing["row"].tolist()) & set(new["row"].tolist()))
    if overlap:
        shown = overlap[:5]
        raise ValueError(f"#161: row(s) {shown}{' ...' if len(overlap) > 5 else ''} already in the "
                         f"ledger at {path}; append_ledger will not silently overwrite them")
    merged = {name: np.concatenate([existing[name], new[name]]) for name in COLUMNS}
    write_ledger(**merged, path=path, source="append_ledger")


def index_by_row(ledger: dict[str, np.ndarray]) -> dict[int, int]:
    """Corpus row id -> its position in `ledger`'s arrays, for per-row metadata lookups."""
    return {int(r): i for i, r in enumerate(ledger["row"])}


def extract_from_vecset_latents(latents_path: Path) -> dict[str, np.ndarray]:
    """The one-time split: pull the ledger columns out of an existing `vecset_latents.h5`.

    Reads only the four small columns, never `latent`/`query_pos` -- the whole point of #161 is that
    this never has to touch, or reproduce, the expensive Dora-encoded arrays.
    """
    with h5py.File(latents_path, "r") as f:
        missing = [c for c in COLUMNS if c not in f]
        if missing:
            raise ValueError(f"#161: {latents_path} is missing column(s) {missing}")
        raw = {name: f[name][:] for name in COLUMNS}
    return _validate(**raw)
