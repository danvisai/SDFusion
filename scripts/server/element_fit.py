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

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scene import element_lib

# how many instances a type needs before retrieval-fit takes over from the procedural
# template (spec R5)
MIN_POOL = 8


def _features():
    m = element_lib.meta()
    asp = np.asarray([e["aspect"] for e in m], np.float32)        # (N,2): x/y, z/y
    yfr = np.asarray([e["y_frac"] for e in m], np.float32)
    typ = np.asarray([e["type"] for e in m])
    cls = np.asarray([e["cls"] for e in m])
    return m, asp, yfr, typ, cls


_F = None


def pool_size(types):
    global _F
    if _F is None:
        _F = _features()
    _m, _asp, _yfr, typ, _cls = _F
    return int(np.isin(typ, list(types)).sum())


def retrieve(types, box_aspect_xz, y_frac, building_class="RESIDENTIAL",
             seed=None, k=12):
    """-> (lib_id, meta_row) or (None, None) if the pool is too small.

    `types`: iterable of library type names to pool. `box_aspect_xz`: (ex/ey, ez/ey) of
    the placed box. `y_frac`: box-center height fraction on THIS building (0=ground,
    1=roofline)."""
    global _F
    if _F is None:
        _F = _features()
    m, asp, yfr, typ, cls = _F
    sel = np.where(np.isin(typ, list(types)))[0]
    if len(sel) < MIN_POOL:
        return None, None
    ta = np.log(np.clip(np.asarray(box_aspect_xz, np.float32), 1e-3, 1e3))
    la = np.log(np.clip(asp[sel], 1e-3, 1e3))
    d = np.abs(la - ta[None]).sum(1)                              # aspect match
    d = d + 1.5 * np.abs(yfr[sel] - float(np.clip(y_frac, 0, 1.2)))  # height-on-building
    same_cls = np.char.startswith(cls[sel].astype(str), building_class.upper()[:6])
    d = d + np.where(same_cls, 0.0, 0.6)                          # class affinity
    order = np.argsort(d)[:max(k, 1)]
    rng = np.random.default_rng(seed)
    w = np.exp(-d[order] / max(d[order].std() + 1e-6, 0.15))
    i = order[rng.choice(len(order), p=w / w.sum())]
    lid = int(sel[i])
    return lid, m[lid]


def element_op(lib_id, center, half, rot_y=0.0, det="element", smooth=0.08):
    """EditOp dict for the fitted element (first-class op: undo/re-roll/round-trip/bake
    all reuse the existing machinery via scene/sdf_edit's 'element' kind)."""
    return dict(kind="element", lib_id=int(lib_id),
                center=[float(v) for v in center], size=[float(v) for v in half],
                rot_y=float(rot_y), mode="add", smooth=float(smooth), det=det, grp="gI")
