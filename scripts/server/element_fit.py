"""Phase R2 of GENERATIVE_MAKE_IT_ARCHITECTURE_BUILD_SPEC_2026-07-08: retrieval + fit.

Given the typed placement (planner typing from interpret_mass) and the placed box's shape
and height-on-building, rank the element library (real BuildingNet component geometry) and
sample an instance seeded — so 🎲 re-roll walks through different real architecture, the
same UX contract the procedural constructions had.

Scoring: aspect match (log-ratio of x/y and z/y aspect vs the box) + height-on-building
match (an element that lived at the roofline fits a roofline box) + class affinity
(RESIDENTIAL boxes prefer elements harvested from RESIDENTIAL* buildings). Softmax-sample
from the top-k so retrieval is diverse but sane.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scene import element_lib

# how many instances a type needs before retrieval-fit takes over from the procedural
# template (spec R5)
MIN_POOL = 8
MIN_SOLIDITY = 0.12
MAX_RELATIVE_SCALE_RATIO = 3.0
MAX_ASPECT_RATIO = 4.0


class RetrievalFeatures(NamedTuple):
    meta: list
    aspect: np.ndarray
    y_fraction: np.ndarray
    relative_extent: np.ndarray
    types: np.ndarray
    classes: np.ndarray
    solidity: np.ndarray


def _solidity(n_elements):
    """Load the derived occupancy cache, rebuilding it after any library change."""
    sol_p = element_lib.LIB / "solidity.npy"
    source_paths = (element_lib.LIB / "meta.json", element_lib.LIB / "elements_f16.npy")
    try:
        fresh = sol_p.stat().st_mtime_ns >= max(p.stat().st_mtime_ns for p in source_paths)
        sol = np.load(sol_p, allow_pickle=False)
        valid = (sol.shape == (n_elements,) and np.isfinite(sol).all()
                 and ((0.0 <= sol) & (sol <= 1.0)).all())
        if fresh and valid:
            return sol.astype(np.float32, copy=False)
    except (OSError, ValueError):
        pass

    crops = element_lib._crops_mm()
    if len(crops) != n_elements:
        raise ValueError(f"element library mismatch: {n_elements} metadata rows, {len(crops)} crops")
    sol = np.asarray([(np.asarray(crops[i]) <= 0).mean() for i in range(n_elements)],
                     np.float32)

    # The cache is an optimization, not a serving requirement. Replace atomically when the
    # library is writable; otherwise keep using the correctly computed in-memory values.
    tmp = sol_p.with_name(f".{sol_p.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as f:
            np.save(f, sol)
        tmp.replace(sol_p)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
    return sol


def _features():
    m = element_lib.meta()
    asp = np.asarray([e["aspect"] for e in m], np.float32)        # (N,2): x/y, z/y
    yfr = np.asarray([e["y_frac"] for e in m], np.float32)
    rel = np.asarray([e["ext_rel"] for e in m], np.float32)       # (N,3): ext / src bldg h
    typ = np.asarray([e["type"] for e in m])
    cls = np.asarray([e["cls"] for e in m])
    # Solidity is the occupied fraction of the element's own crop. Skeletal pieces
    # (crosses, railings, open shells) fill almost nothing and read broken when stretched.
    sol = _solidity(len(m))
    return RetrievalFeatures(m, asp, yfr, rel, typ, cls, sol)


_F = None


def pool_size(types):
    global _F
    if _F is None:
        _F = _features()
    return int((np.isin(_F.types, list(types)) & (_F.solidity >= MIN_SOLIDITY)).sum())


def retrieve(types, box_aspect_xz, y_frac, building_class="RESIDENTIAL",
             seed=None, k=12, box_rel_y=None):
    """-> (lib_id, meta_row) or (None, None) if the pool is too small.

    `types`: iterable of library type names to pool. `box_aspect_xz`: (ex/ey, ez/ey) of
    the placed box. `y_frac`: box-center height fraction on THIS building. `box_rel_y`:
    the box's FULL height / this building's height — matched against the element's own
    relative height on ITS source building, so a tiny pinnacle never gets stretched into
    a tower-sized placement (the 2026-07-09 'weird' failure)."""
    global _F
    if _F is None:
        _F = _features()
    m = _F.meta
    asp, yfr, rel = _F.aspect, _F.y_fraction, _F.relative_extent
    typ, cls, sol = _F.types, _F.classes, _F.solidity
    candidates = np.where(np.isin(typ, list(types)) & (sol >= MIN_SOLIDITY))[0]
    ta = np.log(np.clip(np.asarray(box_aspect_xz, np.float32), 1e-3, 1e3))
    la = np.log(np.clip(asp[candidates], 1e-3, 1e3))
    eligible = (np.abs(la - ta[None]) <= np.log(MAX_ASPECT_RATIO)).all(axis=1)
    if box_rel_y is not None:
        rel_gap = np.abs(np.log(np.clip(rel[candidates, 1], 1e-3, 10.0))
                         - np.log(max(float(box_rel_y), 1e-3)))
        eligible &= rel_gap <= np.log(MAX_RELATIVE_SCALE_RATIO)
    sel = candidates[eligible]
    la = la[eligible]
    if len(sel) < MIN_POOL:
        return None, None

    d = np.abs(la - ta[None]).sum(1)                              # aspect match
    d = d + 1.5 * np.abs(yfr[sel] - float(np.clip(y_frac, 0, 1.2)))  # height-on-building
    if box_rel_y is not None:
        d = d + 1.2 * np.abs(np.log(np.clip(rel[sel, 1], 1e-3, 10.0))
                             - np.log(max(float(box_rel_y), 1e-3)))  # relative SCALE match
    d = d - 0.4 * np.log(np.clip(sol[sel] / MIN_SOLIDITY, 1.0,
                                 1.0 / MIN_SOLIDITY))              # prefer solid pieces
    same_cls = np.char.startswith(cls[sel].astype(str), building_class.upper()[:6])
    d = d + np.where(same_cls, 0.0, 0.6)                          # class affinity
    order = np.argsort(d)[:max(k, 1)]
    rng = np.random.default_rng(seed)
    w = np.exp(-d[order] / max(d[order].std() + 1e-6, 0.15))
    i = order[rng.choice(len(order), p=w / w.sum())]
    lid = int(sel[i])
    row = dict(m[lid])
    row["solidity"] = float(sol[lid])
    return lid, row


def element_op(lib_id, center, half, rot_y=0.0, det="element", smooth=0.08):
    """EditOp dict for the fitted element (first-class op: undo/re-roll/round-trip/bake
    all reuse the existing machinery via scene/sdf_edit's 'element' kind)."""
    return dict(kind="element", lib_id=int(lib_id),
                center=[float(v) for v in center], size=[float(v) for v in half],
                rot_y=float(rot_y), mode="add", smooth=float(smooth), det=det, grp="gI")
