"""Protect the historical real-corpus row identities used by the pinned split (#162).

The pin covers ordered (bag_id, height_m) pairs, not geometry or appended rows.
See docs/wayfinding/buildingworld-corpus/162-frozen-corpus-identity.md.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import h5py
import numpy as np

REAL_CORPUS_PATH = Path(__file__).resolve().parents[1] / "data/real_massing_v1/real.h5"
# #162: never derive this split's size or identity from a newly ingested corpus.
FROZEN_SPLIT_N_TOTAL = 35_776
FROZEN_CORPUS_SHA256 = "c823614b3e6019216919490e5265557aa177d5e889f9023453dff2005d521322"


def row_identity_sha256(bag_ids: np.ndarray, heights: np.ndarray) -> str:
    """SHA-256 of ordered, packed S64 / little-endian float32 records (68 bytes/row).

    Explicit widths, byte order, and no struct padding make the pin reproducible across hosts.
    Refuse lossy conversions, which could otherwise hide an identity change.
    """
    bag_ids, heights = np.asarray(bag_ids), np.asarray(heights)
    if bag_ids.ndim != 1 or heights.shape != bag_ids.shape:
        raise ValueError("#162: bag_id and height_m must be matching one-dimensional columns")
    if bag_ids.dtype != np.dtype("S64") or heights.dtype.kind != "f" or heights.dtype.itemsize != 4:
        raise ValueError("#162: identity requires bag_id S64 and height_m float32")
    if not np.isfinite(heights).all():
        raise ValueError("#162: height_m contains non-finite values")
    pairs = np.empty(len(bag_ids), dtype=[("bag_id", "S64"), ("height_m", "<f4")])
    pairs["bag_id"], pairs["height_m"] = bag_ids, heights
    return hashlib.sha256(pairs.tobytes()).hexdigest()


def assert_frozen_corpus(corpus: h5py.File) -> str:
    """Validate an open raw corpus before any row-indexed cache, training, or evaluation read.

    Append-only growth is allowed. Missing/short columns or a changed prefix fail closed.
    Uses explicit exceptions so checks remain enabled under Python's -O flag.
    """
    for name in ("sdf", "bag_id", "height_m"):
        if name not in corpus or corpus[name].ndim == 0:
            raise ValueError(f"#162: {corpus.filename}: missing row column {name}")
    n_total = corpus["sdf"].shape[0]
    if n_total < FROZEN_SPLIT_N_TOTAL:
        raise ValueError(f"#162: {corpus.filename}: {n_total} rows; frozen prefix requires "
                         f"at least {FROZEN_SPLIT_N_TOTAL}. Restore the original corpus.")
    for name in ("bag_id", "height_m", "footprint", "source_id"):
        if name in corpus and (corpus[name].ndim == 0 or corpus[name].shape[0] != n_total):
            raise ValueError(f"#162: {corpus.filename}: {name} row count differs from sdf")
    actual = row_identity_sha256(corpus["bag_id"][:FROZEN_SPLIT_N_TOTAL],
                                 corpus["height_m"][:FROZEN_SPLIT_N_TOTAL])
    if actual != FROZEN_CORPUS_SHA256:
        raise ValueError(f"#162: {corpus.filename}: frozen corpus identity mismatch: "
                         f"expected {FROZEN_CORPUS_SHA256}, got {actual}. "
                         "Rows were changed or reordered. Restore the original prefix; "
                         "do not regenerate the pin to accept a rebuild.")
    return actual


def open_real_corpus(path: str | Path = REAL_CORPUS_PATH) -> h5py.File:
    """Open a raw real corpus read-only, returning a handle only after checking its identity.

    Use as a context manager, or close the returned handle when its worker exits.
    Validation failure closes the file as well.
    """
    corpus = h5py.File(path, "r")
    try:
        assert_frozen_corpus(corpus)
    except BaseException:
        corpus.close()
        raise
    return corpus
