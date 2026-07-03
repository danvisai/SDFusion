"""Planner → sculptor bridge (detail-plan steps 3→5 + the deterministic half of step 4).

Takes the sculptor's massing volume (cube frame), samples a part layout from the trained
PartLayoutPlannerV2, SNAP-TO-SURFACE projects each box onto the massing (coherence rule #1 —
windows/doors flush with walls, chimneys/domes on the roof), and maps the parts to the
sculptor's typed detail EditOps (crisp analytic ops that survive snaps and bake at high res).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from models.networks.part_layout_planner import PartLayoutPlannerV2, TYPE_NAMES  # noqa: E402

PLANNER_CKPT = REPO / "outputs/part_layout_planner_v2/planner.pth"
REFINER_CKPT = REPO / "outputs/part_set_refiner/refiner.pth"
CLASSES = ["COMMERCIAL", "PUBLIC", "RELIGIOUS", "RESIDENTIAL"]
_planner = None
_refiner = None


def _get_refiner(device):
    global _refiner
    if _refiner is None:
        from models.networks.part_set_refiner import PartSetRefiner
        ck = torch.load(REFINER_CKPT, map_location=device)
        _refiner = PartSetRefiner(T=int(ck.get("T", 1000)), device=device)
        _refiner.net.load_state_dict(ck["net"])
        _refiner.net.eval()
    return _refiner


def _get_planner(device):
    global _planner
    if _planner is None:
        ck = torch.load(PLANNER_CKPT, map_location=device)
        _planner = PartLayoutPlannerV2(max_len=int(ck.get("max_len", 40))).to(device)
        _planner.load_state_dict(ck["model"])
        _planner.eval()
    return _planner


def _occ_frame(grid):
    """occ bbox center/scale of a cube-frame volume (margin-1.0 planner frame)."""
    occ = np.asarray(grid) <= 0
    R = occ.shape[0]
    g = np.linspace(-1, 1, R)
    wi, hi, di = np.where(occ.any((0, 1)))[0], np.where(occ.any((0, 2)))[0], np.where(occ.any((1, 2)))[0]
    if not len(wi):
        raise ValueError("empty massing")
    c = np.array([(g[wi.min()] + g[wi.max()]) / 2, (g[hi.min()] + g[hi.max()]) / 2,
                  (g[di.min()] + g[di.max()]) / 2], np.float32)
    s = max(g[wi.max()] - g[wi.min()], g[hi.max()] - g[hi.min()], g[di.max()] - g[di.min()]) / 2
    return occ, c, max(s, 1e-3)


def _snap_to_surface(occ, kind, c_pl, e_pl):
    """Project a planner-frame box onto the massing surface (planner frame ~= occ-bbox frame).

    walls (window/door/balcony/column): pull center to the nearest occupied-boundary in XZ.
    roof  (chimney/dome/tower-base):    set base y to the top occupied y at that column."""
    R = occ.shape[0]
    to_i = lambda v: int(np.clip((v + 1) * 0.5 * (R - 1), 0, R - 1))
    xi, yi, zi = to_i(c_pl[0]), to_i(c_pl[1]), to_i(c_pl[2])
    g = np.linspace(-1, 1, R)
    if kind in ("window", "door", "balcony", "column"):
        from scipy.ndimage import binary_erosion
        ys = np.where(occ.any(axis=(0, 2)))[0]            # occupied height range
        if not len(ys):
            return c_pl
        yi = int(np.clip(yi, ys.min(), ys.max()))         # clamp to where walls EXIST
        col = occ[:, max(yi, 1), :]                       # (D=z, W=x) slice at part height
        if not col.any():
            col = occ.any(axis=1)                         # fallback: footprint silhouette
        # target the BOUNDARY (wall), not any occupied voxel — a proposal landing INSIDE the
        # building is otherwise "nearest to itself" and never gets pulled out to the wall
        edge = col & ~binary_erosion(col)
        if edge.any():
            col = edge
        c_pl = np.array([c_pl[0], g[yi], c_pl[2]], np.float32)
        zz, xx = np.where(col)
        d2 = (g[xx] - c_pl[0]) ** 2 + (g[zz] - c_pl[2]) ** 2
        k = int(np.argmin(d2))
        wx, wz = g[xx[k]], g[zz[k]]
        v = np.array([c_pl[0] - wx, c_pl[2] - wz])
        n = v / (np.linalg.norm(v) + 1e-6)
        # subtract ops (window/door) must STRADDLE the wall (center ON the surface);
        # protruding adds (balcony) sit outward by ~their depth; columns hug the wall.
        push = {"window": 0.0, "door": 0.0, "balcony": 0.9, "column": 0.15}[kind]
        d = max(e_pl[0], e_pl[2])
        return np.array([wx + n[0] * d * push, c_pl[1], wz + n[1] * d * push], np.float32)
    if kind in ("chimney", "dome", "tower"):
        colY = occ[zi, :, xi]
        if colY.any():
            top = g[np.where(colY)[0].max()]
            return np.array([c_pl[0], top + (e_pl[1] if kind == "chimney" else 0.0), c_pl[2]], np.float32)
    return c_pl


def resnap_ops_to_surface(grid, ops):
    """Re-project existing detail EditOps onto a (possibly snapped/changed) massing surface.

    The snap changes the massing under the details — walls move, so a balcony placed on the
    OLD wall floats or sinks inside the NEW one, and subtract ops carve interior crevices.
    Wall elements (subtract ops / balconies / columns) re-snap to the nearest wall at their
    height; roof elements (chimney/dome/tower adds) re-seat on the new roof line."""
    occ = np.asarray(grid) <= 0
    snappable = ("window", "door", "balcony", "column", "chimney", "dome", "tower")
    # multi-op details (balcony = slab+parapet, door = hole+canopy) must move as ONE GROUP —
    # snapping members independently splits them onto different walls. Ops sharing a 'grp'
    # get a single offset computed from the group's first snappable member.
    groups = {}
    order = []
    for i, op in enumerate(ops):
        g = op.get("grp") or f"_solo{i}"
        groups.setdefault(g, []).append(op)
        order.append(g)
    snapped = {}
    for g, members in groups.items():
        lead = next((o for o in members if o.get("det") in snappable), None)
        if lead is None:
            snapped[g] = [dict(o) for o in members]   # never guess on untagged ops
            continue
        c = np.asarray(lead["center"], np.float32)
        e = np.asarray((list(lead["size"]) + [0.05, 0.05])[:3], np.float32)
        delta = _snap_to_surface(occ, lead["det"], c, np.abs(e)) - c
        outm = []
        for o in members:
            oo = dict(o)
            oo["center"] = [float(v + dv) for v, dv in zip(o["center"], delta)]
            outm.append(oo)
        snapped[g] = outm
    out, used = [], set()
    for g in order:
        if g not in used:
            out.extend(snapped[g]); used.add(g)
    return out


@torch.no_grad()
def recohere_ops(grid, ops, device="cuda", strength=0.2):
    """LEARNED re-coherence (detail-plan step 4 wired): encode the current detail ops as a part
    SET (groups = one part), jointly denoise with the set refiner conditioned on the massing,
    apply the refined validity (drop) + center deltas back to the ops, then surface-snap as a
    safety net. Returns (ops, n_dropped)."""
    from scipy.ndimage import distance_transform_edt
    from models.networks.part_set_refiner import SLOTS, PART_DIM, N_TYPES as NT
    occ, c, s = _occ_frame(grid)
    # massing input in the refiner's training frame
    R = occ.shape[0]
    g1 = torch.linspace(-1, 1, 64)
    Z, Y, X = torch.meshgrid(g1, g1, g1, indexing="ij")
    q = (torch.stack([X, Y, Z], -1) * float(s) + torch.from_numpy(c)).numpy()
    idx = np.clip(((q + 1) * 0.5 * (R - 1)).round().astype(int), 0, R - 1)
    occN = occ[idx[..., 2], idx[..., 1], idx[..., 0]]
    vox = 2.0 / 63
    sdfN = np.clip((distance_transform_edt(~occN) - distance_transform_edt(occN)) * vox,
                   -0.2, 0.2).astype(np.float32)
    # groups -> one part each (union bbox, planner frame)
    t2i = {n: i for i, n in enumerate(TYPE_NAMES)}
    groups, order = {}, []
    for i, op in enumerate(ops):
        gkey = op.get("grp") or f"_solo{i}"
        groups.setdefault(gkey, []).append(op)
        if gkey not in order:
            order.append(gkey)
    x0 = torch.zeros(1, SLOTS, PART_DIM)
    x0[..., :NT] = -1.0
    x0[..., -1] = -1.0
    slot_of = {}
    for si, gkey in enumerate(order[:SLOTS]):
        mem = groups[gkey]
        lead = next((o for o in mem if o.get("det") in t2i), None)
        if lead is None:
            continue                                      # untyped groups are left alone
        los = np.array([np.asarray(o["center"]) - np.asarray((list(o["size"]) + [0, 0])[:3])
                        for o in mem]).min(0)
        his = np.array([np.asarray(o["center"]) + np.asarray((list(o["size"]) + [0, 0])[:3])
                        for o in mem]).max(0)
        cc = ((los + his) / 2 - c) / s                    # cube -> planner frame
        ee = np.maximum((his - los) / 2 / s, 1e-3)
        x0[0, si, t2i[lead["det"]]] = 1.0
        x0[0, si, NT:NT + 3] = torch.from_numpy(cc.astype(np.float32))
        x0[0, si, NT + 3:NT + 6] = torch.from_numpy(ee.astype(np.float32))
        x0[0, si, -1] = 1.0
        slot_of[gkey] = si
    if not slot_of:
        return list(ops), 0
    model = _get_refiner(device)
    ref = model.refine(x0.to(device), torch.from_numpy(sdfN).view(1, 1, 64, 64, 64).to(device),
                       strength=strength, steps=12)[0].cpu()
    out, dropped = [], 0
    for gkey in order:
        mem = groups[gkey]
        si = slot_of.get(gkey)
        if si is None:
            out.extend(dict(o) for o in mem)
            continue
        if ref[si, -1] <= 0:
            dropped += len(mem)                           # the model deleted this part
            continue
        delta = (ref[si, NT:NT + 3].numpy() - x0[0, si, NT:NT + 3].numpy()) * s   # planner->cube
        for o in mem:
            oo = dict(o)
            oo["center"] = [float(v + dv) for v, dv in zip(o["center"], delta)]
            out.append(oo)
    return resnap_ops_to_surface(grid, out), dropped


# ===========================================================================
# CoherentPartRefiner integration (NEW, 2026-07-02) — the "neighbors re-harmonize" step for a
# NEWLY placed part. Separate model/cache from _get_refiner/recohere_ops (PartSetRefiner +
# refiner.pth + /recohere_details stay UNTOUCHED, per the original build spec's own principle:
# docs/COHERENT_ADD_PRIMITIVE_BUILD_SPEC_2026-06-15.md §7 step 3). Only applies to WALL-RHYTHM
# types (window/door/balcony/balcony_upper) — the only types CoherentPartRefiner was trained
# and coherence-scored on (CoherentPartRefiner.WALL_TYPES); massing/roof types (tower/dome/
# chimney/dormer/wing/bay/column/roof) are interpret_mass's procedural construction's job.
# ===========================================================================
COHERENT_REFINER_CKPT = REPO / "outputs/part_set_refiner/coherent_refiner.pth"
WALL_RHYTHM_TYPES = {"window", "door", "balcony", "balcony_upper"}
_coherent_refiner = None


def _get_coherent_refiner(device):
    global _coherent_refiner
    if _coherent_refiner is None:
        from models.networks.part_set_refiner import CoherentPartRefiner
        ck = torch.load(COHERENT_REFINER_CKPT, map_location=device)
        _coherent_refiner = CoherentPartRefiner(T=int(ck.get("T", 1000)), device=device)
        _coherent_refiner.net.load_state_dict(ck["net"])
        _coherent_refiner.net.eval()
    return _coherent_refiner


@torch.no_grad()
def integrate_new_part(grid, ops, new_op, building_class="RESIDENTIAL", device="cuda",
                       strength=0.5, neighbor_k=8, steps=16):
    """CoherentPartRefiner: given the EXISTING typed ops + a NEWLY typed op (or list of ops
    sharing one construction, e.g. interpret_mass's balcony = slab+door — already typed/
    constructed by interpret_mass), refine the new piece's pose so it aligns with same-type
    neighbors (row/spacing/wall-attach — 'neighbors re-harmonize'), then surface-snap as a
    safety net. A no-op passthrough (ops + new_op(s), used=False) when the LEAD op's type is
    outside WALL_RHYTHM_TYPES (the only types the model was trained/coherence-scored on).
    Defaults (strength=0.5, neighbor_k=8, steps=16) chosen by sweep 2026-07-02: best/most
    consistent wall-attachment across varied moldy-piece starting positions on the reference
    scenario (max wall-distance 0.017 vs 0.35-0.55+ scale (block half-extent) for worse settings).
    Returns (updated_ops, used: bool)."""
    new_ops = new_op if isinstance(new_op, list) else [new_op]
    # gate on the PRIMARY (first) op's type only -- interpret_mass's tower/bay/wing/dormer
    # constructions incidentally contain "window" sub-ops (e.g. a tower's window band) that
    # would wrongly match WALL_RHYTHM_TYPES if we searched the whole list; the primary piece
    # (index 0) is always the construction's real type.
    lead_new = new_ops[0] if new_ops and new_ops[0].get("det") in WALL_RHYTHM_TYPES else None
    if lead_new is None:
        return ops + new_ops, False
    from scipy.ndimage import distance_transform_edt
    from models.networks.part_set_refiner import SLOTS, PART_DIM, N_TYPES as NT
    occ, c, s = _occ_frame(grid)
    R = occ.shape[0]
    g1 = torch.linspace(-1, 1, 64)
    Z, Y, X = torch.meshgrid(g1, g1, g1, indexing="ij")
    q = (torch.stack([X, Y, Z], -1) * float(s) + torch.from_numpy(c)).numpy()
    idx = np.clip(((q + 1) * 0.5 * (R - 1)).round().astype(int), 0, R - 1)
    occN = occ[idx[..., 2], idx[..., 1], idx[..., 0]]
    vox = 2.0 / 63
    sdfN = np.clip((distance_transform_edt(~occN) - distance_transform_edt(occN)) * vox,
                   -0.2, 0.2).astype(np.float32)

    t2i = {n: i for i, n in enumerate(TYPE_NAMES)}
    groups, order = {}, []
    for i, op in enumerate(ops):
        gkey = op.get("grp") or f"_solo{i}"
        groups.setdefault(gkey, []).append(op)
        if gkey not in order:
            order.append(gkey)
    new_gkey = lead_new.get("grp") or "_new"
    groups[new_gkey] = new_ops
    order.append(new_gkey)

    x0 = torch.zeros(1, SLOTS, PART_DIM); x0[..., :NT] = -1.0; x0[..., -1] = -1.0
    mk = torch.zeros(1, SLOTS)
    slot_of = {}
    for si, gkey in enumerate(order[:SLOTS]):
        mem = groups[gkey]
        lead = next((o for o in mem if o.get("det") in t2i), None)
        if lead is None:
            continue
        los = np.array([np.asarray(o["center"]) - np.asarray((list(o["size"]) + [0, 0])[:3])
                        for o in mem]).min(0)
        his = np.array([np.asarray(o["center"]) + np.asarray((list(o["size"]) + [0, 0])[:3])
                        for o in mem]).max(0)
        cc = ((los + his) / 2 - c) / s
        ee = np.maximum((his - los) / 2 / s, 1e-3)
        x0[0, si, t2i[lead["det"]]] = 1.0
        x0[0, si, NT:NT + 3] = torch.from_numpy(cc.astype(np.float32))
        x0[0, si, NT + 3:NT + 6] = torch.from_numpy(ee.astype(np.float32))
        x0[0, si, -1] = 1.0
        slot_of[gkey] = si
        if gkey == new_gkey:
            mk[0, si] = 1.0
    if new_gkey not in slot_of:
        return ops + new_ops, False                    # ran out of slots (>SLOTS-1 existing parts)

    cu = building_class.upper()
    cls_id = CLASSES.index(cu) if cu in CLASSES else CLASSES.index("RESIDENTIAL")
    model = _get_coherent_refiner(device)
    ref = model.refine(x0.to(device), torch.from_numpy(sdfN).view(1, 1, 64, 64, 64).to(device),
                       mk.to(device), torch.tensor([cls_id], device=device),
                       strength=strength, steps=steps, neighbor_k=neighbor_k)[0].cpu()

    out = []
    for gkey in order:
        mem = groups[gkey]
        si = slot_of.get(gkey)
        if si is None:
            out.extend(dict(o) for o in mem)
            continue
        if ref[si, -1] <= 0 and gkey != new_gkey:
            continue                                  # coherence pass dropped a neighbor (rare)
        delta = (ref[si, NT:NT + 3].numpy() - x0[0, si, NT:NT + 3].numpy()) * s
        for o in mem:
            oo = dict(o)
            oo["center"] = [float(v + dv) for v, dv in zip(o["center"], delta)]
            out.append(oo)
    return resnap_ops_to_surface(grid, out), True


def regularize_ops(ops, occ, height_n):
    """Make a sampled layout READ as architecture (facades are regular; independent sampling
    gives confetti): windows snap to floor rows + uniform size + min spacing; ONE door at
    ground level; <=2 chimneys/towers; overlapping same-type ops deduped."""
    R = occ.shape[0]
    g = np.linspace(-1, 1, R)
    ys = np.where(occ.any(axis=(0, 2)))[0]
    y0, y1 = (g[ys.min()], g[ys.max()]) if len(ys) else (-1, 1)
    n_floors = max(int(round((y1 - y0) / 0.45)), 1)          # ~3m floors in cube units (~0.45)
    rows = [y0 + (y1 - y0) * (i + 0.55) / n_floors for i in range(n_floors)]
    windows = [o for o in ops if o.get("det") == "window"]
    doors = [o for o in ops if o.get("det") == "door"]
    rest = [o for o in ops if o.get("det") not in ("window", "door")]
    out = []
    # windows: row-snap y, uniform median size, dedupe by spacing
    if windows:
        wsz = np.median([o["size"][1] for o in windows])
        xsz = np.median([max(o["size"][0], o["size"][2]) for o in windows])
        placed = []
        for o in sorted(windows, key=lambda o: (min(rows, key=lambda r: abs(r - o["center"][1])),
                                                o["center"][0], o["center"][2])):
            c = list(o["center"])
            c[1] = min(rows, key=lambda r: abs(r - c[1]))
            if any(abs(c[1] - p[1]) < 0.05 and np.hypot(c[0] - p[0], c[2] - p[2]) < 2.6 * xsz
                   for p in placed):
                continue                                       # too close to a placed window
            placed.append(c)
            oo = dict(o); oo["center"] = c
            oo["size"] = [max(xsz, .02), max(wsz, .02), max(xsz, .02)]
            out.append(oo)
    # exactly one door, grounded — and GUARANTEED (a building without a door reads wrong)
    if doors:
        o = dict(doors[0])
        o["center"] = [o["center"][0], y0 + o["size"][1], o["center"][2]]
        out.append(o)
    else:
        gy = max(min(range(R), key=lambda i: abs(g[i] - (y0 + 0.07))), 1)
        row = occ[:, gy, :]
        if row.any():
            zz, xx = np.where(row)
            zfront = zz.max()                                   # front facade = max-z wall
            xs_f = xx[zz == zfront]
            cx_d = float(g[int(np.median(xs_f))])
            out.append(dict(kind="box", center=[cx_d, y0 + 0.09, float(g[zfront])],
                            size=[0.05, 0.09, 0.05], mode="subtract", smooth=0.0, det="door"))
    # caps for roof clutter
    seen = {"chimney": 0, "tower": 0, "dome": 0}
    for o in rest:
        k = o.get("det")
        if k in seen:
            if seen[k] >= (1 if k == "dome" else 2):
                continue
            seen[k] += 1
        out.append(o)
    return out


@torch.no_grad()
def propose_detail_ops(grid, building_class="RESIDENTIAL", device="cuda", temperature=0.7,
                       max_ops=14, seed=None):
    """Cube-frame massing volume -> planner layout -> snapped, typed detail EditOps (cube coords)."""
    from scipy.ndimage import distance_transform_edt
    if seed is not None:
        torch.manual_seed(int(seed))
    occ, c, s = _occ_frame(grid)
    R = occ.shape[0]
    # resample occupancy into the planner's margin-1.0 frame + EDT sdf input
    g1 = torch.linspace(-1, 1, 64)
    Z, Y, X = torch.meshgrid(g1, g1, g1, indexing="ij")
    q = (torch.stack([X, Y, Z], -1) * float(s) + torch.from_numpy(c)).numpy()
    idx = np.clip(((q + 1) * 0.5 * (R - 1)).round().astype(int), 0, R - 1)
    occN = occ[idx[..., 2], idx[..., 1], idx[..., 0]]
    vox = 2.0 / 63
    sdfN = np.clip((distance_transform_edt(~occN) - distance_transform_edt(occN)) * vox,
                   -0.2, 0.2).astype(np.float32)
    x = torch.from_numpy(sdfN).view(1, 1, 64, 64, 64).to(device)
    cls = torch.tensor([CLASSES.index(building_class) if building_class in CLASSES else 3],
                       device=device)
    planner = _get_planner(device)
    parts = planner.sample(x, cls, temperature=temperature)[0][:max_ops * 2]

    ops = []
    for t, b in parts:
        kind = TYPE_NAMES[t]
        c_pl, e_pl = np.asarray(b[:3], np.float32), np.abs(np.asarray(b[3:], np.float32))
        c_pl = _snap_to_surface(occN, kind, c_pl, e_pl)
        cc = (c_pl * s + c).tolist()                       # planner frame -> viewer cube frame
        ee = (np.clip(e_pl, 0.02, 0.6) * s).tolist()
        if kind == "window":
            ops.append(dict(kind="box", center=cc, size=[max(ee[0], .02), max(ee[1], .02), max(ee[2], .02)],
                            mode="subtract", smooth=0.0, det=kind))
        elif kind == "door":
            ops.append(dict(kind="box", center=[cc[0], cc[1], cc[2]],
                            size=[max(ee[0], .03), max(ee[1], .05), max(ee[2], .03)],
                            mode="subtract", smooth=0.0, det=kind))
        elif kind == "chimney":
            ops.append(dict(kind="box", center=cc, size=[max(ee[0], .03), max(ee[1], .06), max(ee[2], .03)],
                            mode="add", smooth=0.0, det=kind))
        elif kind == "dome":
            ops.append(dict(kind="sphere", center=cc, size=[float(np.clip(max(e_pl) * s, .06, .35))],
                            mode="add", smooth=0.25, det=kind))
        elif kind == "tower":
            ops.append(dict(kind="box", center=cc, size=[max(ee[0], .05), max(ee[1], .1), max(ee[2], .05)],
                            mode="add", smooth=0.05, det=kind))
        elif kind == "balcony":
            ops.append(dict(kind="box", center=cc, size=[max(ee[0], .05), .02, max(ee[2], .05)],
                            mode="add", smooth=0.0, det=kind))
        elif kind == "column":
            ops.append(dict(kind="cylinder", center=[cc[0], cc[1], cc[2]],
                            size=[max(min(ee[0], .04), .015), max(ee[1] * 2, .15)],
                            mode="add", smooth=0.0, det=kind))
        # roof / stairs / balcony_upper: skip in v1 (massing already has a roof)
        if len(ops) >= max_ops:
            break
    # regularize (rows/door/caps) THEN project onto the surface in the VIEWER frame with the
    # proven re-snapper (B5-validated <1 voxel) — covers planner-frame edge cases.
    ops = regularize_ops(ops, np.asarray(grid) <= 0, 1.0)
    return resnap_ops_to_surface(grid, ops)


# ---------------------------------------------------------------------------
# SMART ADD — interpret a placed mass as an architectural part
#   typing       = LEARNED when the planner's distribution speaks (part types sampled
#                  from the model trained on real BuildingNet layouts, scored against
#                  the placement), geometric rules as fallback/prior
#   construction = SAMPLED (seeded rng over proportions + real per-class statistics,
#                  e.g. religious towers carry domes at their measured 38% rate)
# ---------------------------------------------------------------------------

def _learned_type_scores(grid, op, building_class, device="cuda", temperature=0.9,
                         n_layouts=2):
    """Score part types for a placement from the TRAINED layout planner: sample layouts
    conditioned on this massing, weight each sampled part by proximity+size match to the
    placed op. Returns {type_name: score} (empty when the planner is silent here)."""
    try:
        c_op = np.asarray(op.get("center", [0, 0, 0]), np.float32)
        e_op = np.asarray(list(op.get("size", [0.1, 0.1, 0.1]))[:3], np.float32)
        scores = {}
        for k in range(n_layouts):
            parts = propose_detail_ops(grid, building_class=building_class, device=device,
                                       temperature=temperature, max_ops=14, seed=1000 + k)
            for p in parts:
                t = p.get("det")
                if not t:
                    continue
                pc = np.asarray(p["center"], np.float32)
                pe = np.asarray(list(p["size"])[:3], np.float32)
                d = float(np.linalg.norm(pc - c_op))
                if d > 0.45:
                    continue
                size_sim = float(np.exp(-abs(np.log((pe.max() + 1e-4) / (e_op.max() + 1e-4)))))
                scores[t] = scores.get(t, 0.0) + float(np.exp(-(d / 0.2) ** 2)) * size_sim
        return scores
    except Exception:
        return {}


def interpret_mass(grid, op, building_class="RESIDENTIAL", style="modern",
                   seed=None, device="cuda", temperature=0.9):
    """Make sense of a raw placed primitive: type it (learned planner distribution with
    geometric-rule fallback) and replace it with a SAMPLED typed construction (EditOp
    dicts, cube coords). Same box + different seed -> different plausible architecture.
    Returns {"kind", "ops", "source": "planner"|"rules", "p_types": {...}}."""
    rng = np.random.default_rng(seed)
    occ = np.asarray(grid) <= 0
    R = occ.shape[0]
    g = np.linspace(-1, 1, R)
    to_i = lambda v: int(np.clip((v + 1) * 0.5 * (R - 1), 0, R - 1))
    c = np.asarray(op.get("center", [0, 0, 0]), np.float32)
    kind_hint = op.get("kind", "box")
    # op['size'] layout differs by primitive kind (sphere=[r], cylinder=[r,h], cone=[angle,h],
    # box=[hx,hy,hz]) -- treating it as a raw 3-slice silently corrupts non-box kinds. A
    # sphere's size=[r] padded this way gave e=[r,0.1,0.1]: its "height" read as a hardcoded
    # 0.1 regardless of actual radius, which misrouted every reasonably-sized sphere into the
    # 'bay' rule (e[1]=0.1 fails the 'balcony' cutoff e[1]<0.09 by a hair) -> constructed as
    # kind='box'. Only sphere needed this fix; cylinder's h-as-full-height quirk is left alone
    # since it's what currently makes cylinders correctly read as slender/tower.
    raw_size = list(op.get("size", [0.1, 0.1, 0.1]))
    if kind_hint == "sphere":
        r = float(raw_size[0]) if raw_size else 0.1
        e = np.array([r, r, r], np.float32)
    elif kind_hint == "cone":
        # size=[angle_deg, height] -- e[0]/e[2] would otherwise read as the ANGLE (e.g. 28),
        # used downstream as a shaft radius (tower's r=max(e[0],e[2])) -- derive an actual
        # footprint radius from angle+height instead (base radius of a cone of that half-angle).
        ang, h = (raw_size + [30.0, 0.2])[:2]
        rad = max(float(h), 1e-3) * float(np.tan(np.radians(np.clip(ang, 1.0, 89.0))))
        e = np.array([rad, float(h) * 0.5, rad], np.float32)
    else:
        e = np.asarray((raw_size + [0.1, 0.1])[:3], np.float32)
    mode = op.get("mode", "add")

    ys = np.where(occ.any(axis=(0, 2)))[0]
    if not len(ys):
        return {"kind": "raw", "ops": [op]}
    y_ground, y_top = g[ys.min()], g[ys.max()]
    colY = occ[to_i(c[2]), :, to_i(c[0])]
    y_local_top = g[np.where(colY)[0].max()] if colY.any() else y_top

    slender = e[1] > 1.6 * max(e[0], e[2])
    top_of_op = c[1] + e[1]
    bot_of_op = c[1] - e[1]
    on_roof = abs(bot_of_op - y_local_top) < 0.12 or (colY.any() and bot_of_op > y_local_top - 0.05)
    near_ground = bot_of_op < y_ground + 0.18
    fp = occ.any(axis=1)                                   # (D=z, W=x) footprint
    inside_fp = bool(fp[to_i(c[2]), to_i(c[0])])

    def _win(cc, ee):
        return dict(kind="box", center=[float(v) for v in cc],
                    size=[float(v) for v in ee], mode="subtract", smooth=0.0,
                    det="window", grp="gI")

    U = lambda a, b: float(rng.uniform(a, b))              # sampled proportions

    # ---- 1. TYPE the placement: learned planner distribution > geometric rules ----
    if mode == "subtract":
        rule_kind = "door" if near_ground and e[1] > 0.07 else "window"
    elif kind_hint == "sphere":
        rule_kind = "dome"           # unambiguous by shape alone -- no size/position rule needed
    elif kind_hint == "cone":
        rule_kind = "tower"          # spire-shaped by construction; tower's own build adds the cone cap
    elif slender and top_of_op > y_local_top + 0.04:
        rule_kind = "tower"
    elif on_roof and inside_fp:
        rule_kind = "chimney" if max(e[0], e[2]) < 0.07 else "dormer"
    elif near_ground and not inside_fp:
        rule_kind = "wing"
    elif not near_ground and bot_of_op > y_ground + 0.1 and top_of_op < y_local_top + 0.05:
        rule_kind = "balcony" if e[1] < 0.09 else "bay"
    else:
        rule_kind = "raw"

    p_types = _learned_type_scores(grid, op, building_class, temperature=temperature)
    kind, source = rule_kind, "rules"
    if p_types:
        best, sc = max(p_types.items(), key=lambda kv: kv[1])
        # the planner only speaks at detail scale; trust it there, keep mass-scale rules
        if sc >= 0.35 and best in ("window", "door", "balcony", "column", "chimney",
                                   "dome", "tower"):
            same_mode = (mode == "subtract") == (best in ("window", "door"))
            if same_mode and best != rule_kind:
                kind, source = best, "planner"
            elif best == rule_kind:
                source = "planner+rules"

    out = {"kind": kind, "source": source,
           "p_types": {k: round(v, 3) for k, v in sorted(p_types.items(),
                                                         key=lambda kv: -kv[1])}}

    # ---- 2. CONSTRUCT it (sampled proportions; class statistics for options) -----
    if kind in ("window", "door"):
        cc = _snap_to_surface(occ, kind, c, e)
        out["ops"] = [dict(kind="box", center=[float(v) for v in cc],
                           size=[float(v) for v in e], mode="subtract",
                           smooth=0.0, det=kind, grp="gI")]
        return out

    if kind == "tower":
        r = float(max(e[0], e[2])) * U(0.85, 1.15)
        h_shaft = float(top_of_op - y_ground)
        ops = [dict(kind="cylinder",
                    center=[float(c[0]), float(y_ground + h_shaft / 2), float(c[2])],
                    size=[r, h_shaft], mode="add", smooth=0.02)]
        try:    # real per-class dome occurrence, measured from the BuildingNet part labels
            from scene.sdf_detail import CLASS_LANDMARK_PROB as _CLP
            dome_p = _CLP.get(building_class.upper(), {}).get("dome", 0.0)
        except Exception:
            dome_p = 0.0
        if rng.random() < dome_p:                          # real per-class dome rate
            ops.append(dict(kind="sphere", center=[float(c[0]), float(top_of_op), float(c[2])],
                            size=[r * U(1.1, 1.35)], mode="add", smooth=0.03))
        else:
            sp_h = r * U(1.8, 2.9)
            ops.append(dict(kind="cone", center=[float(c[0]), float(top_of_op), float(c[2])],
                            size=[float(np.degrees(np.arctan2(r * U(1.05, 1.3), sp_h))),
                                  float(sp_h)], mode="add", smooth=0.0))
        faces = [(r * 0.99, 0), (-r * 0.99, 0), (0, r * 0.99), (0, -r * 0.99)]
        if rng.random() < 0.5:
            faces = faces[:2] if rng.random() < 0.5 else faces[2:]
        wy, step = y_local_top - 0.08, U(0.13, 0.19)
        ww, wh = U(0.02, 0.032), U(0.03, 0.046)
        while wy > y_ground + 0.15:
            for dx, dz in faces:
                ops.append(_win([c[0] + dx, wy, c[2] + dz], [ww, wh, ww]))
            wy -= step
        out["ops"] = ops
        return out

    if kind == "chimney":
        cc = [float(c[0]), float(y_local_top + e[1] * 0.7), float(c[2])]
        out["ops"] = [dict(kind="box", center=cc,
                           size=[float(e[0] * U(0.8, 1.1)), float(e[1] * U(0.9, 1.3)),
                                 float(e[2] * U(0.8, 1.1))],
                           mode="add", smooth=0.0, det="chimney", grp="gI")]
        return out

    if kind == "dome":
        out["ops"] = [dict(kind="sphere",
                           center=[float(c[0]), float(y_local_top), float(c[2])],
                           size=[float(max(e[0], e[2]) * U(0.9, 1.2))],
                           mode="add", smooth=0.04, det="dome", grp="gI")]
        return out

    if kind == "column":
        cc = _snap_to_surface(occ, "column", c, e)
        out["ops"] = [dict(kind="cylinder", center=[float(v) for v in cc],
                           size=[float(np.clip(min(e[0], e[2]), 0.015, 0.04)),
                                 float(max(e[1] * 2, 0.15))],
                           mode="add", smooth=0.0, det="column", grp="gI")]
        return out

    if kind == "dormer":
        body_h = float(max(e[1] * U(1.0, 1.5), 0.08))
        base_y = float(y_local_top - 0.01)
        ops = [dict(kind="gable" if rng.random() < 0.75 else "hip",
                    center=[float(c[0]), base_y, float(c[2])],
                    size=[float(e[0] * 2), float(e[2] * 2), body_h,
                          float(body_h * U(0.45, 0.8))], mode="add", smooth=0.0)]
        ops.append(_win([c[0], base_y + body_h * 0.45, c[2] + e[2] * 0.99],
                        [min(U(0.035, 0.055), e[0] * 0.5), body_h * 0.3, 0.03]))
        out["ops"] = ops
        return out

    if kind == "balcony":
        cc = _snap_to_surface(occ, "balcony", c, e)
        ops = [dict(kind="box", center=[float(v) for v in cc],
                    size=[float(e[0]), U(0.018, 0.03), float(e[2])], mode="add",
                    smooth=0.0, det="balcony", grp="gI")]
        door = _win([cc[0], cc[1] + 0.06, cc[2]], [U(0.032, 0.045), U(0.05, 0.07), 0.04])
        door["det"] = "door"
        ops.append(door)
        out["ops"] = ops
        return out

    if kind == "bay":
        ops = [dict(kind="box", center=[float(v) for v in c],
                    size=[float(v) for v in e], mode="add", smooth=0.015)]
        n = np.array([c[0], c[2]], np.float32)
        n = n / (np.linalg.norm(n) + 1e-6)
        ww, wh = U(0.025, 0.038), U(0.035, 0.05)
        for wy in np.arange(c[1] - e[1] * 0.5, c[1] + e[1] * 0.9, U(0.12, 0.17)):
            ops.append(_win([c[0] + n[0] * max(e[0], e[2]) * 0.99, wy,
                             c[2] + n[1] * max(e[0], e[2]) * 0.99], [ww, wh, ww]))
        out["ops"] = ops
        return out

    if kind == "wing":
        body_h = float(max(e[1] * U(1.3, 1.9), 0.18))
        ops = [dict(kind="gable" if rng.random() < 0.7 else "hip",
                    center=[float(c[0]), float(y_ground), float(c[2])],
                    size=[float(e[0] * 2), float(e[2] * 2), body_h,
                          float(body_h * U(0.4, 0.7))], mode="add", smooth=0.0)]
        d = _win([c[0], y_ground + 0.07, c[2] + e[2] * 0.99], [0.035, U(0.05, 0.07), 0.03])
        d["det"] = "door"
        ops.append(d)
        n_w = max(1, int(e[0] / 0.12))
        for i in range(n_w):
            dx = (i - (n_w - 1) / 2) * e[0] * 1.2 / max(n_w, 1)
            if abs(dx) < 0.02 and n_w > 1:
                continue                                   # keep the door clear
            ops.append(_win([c[0] + dx, y_ground + body_h * 0.6, c[2] + e[2] * 0.99],
                            [0.03, U(0.03, 0.042), 0.03]))
        out["ops"] = ops
        return out

    out["ops"] = [op]
    return out


def adjust_ops_after_snap(grid, ops, device="cuda"):
    """Detail ops AFTER a massing change: re-seat geometrically onto the EXTERIOR surface, then
    row/size regularization + final surface snap. Untyped ops (incl. det:'roof') pass through.
    Returns (ops, n_dropped).

    NOTE (2026-06-15): the LEARNED set-refiner re-coherence (recohere_ops) was REMOVED from this
    path. It was trained on part instances that include INTERIOR parts, so it pulled exterior
    windows inward and carved nonsensical holes. The demo keeps elements on the EXTERIOR (geometric
    resnap-to-wall + row regularization only). recohere_ops still exists for /recohere_details."""
    det_ops = [o for o in ops if o.get("det")]
    other = [o for o in ops if not o.get("det")]
    if not det_ops:
        return list(ops), 0
    out = resnap_ops_to_surface(grid, det_ops)
    dropped = 0
    try:
        out = regularize_ops(out, np.asarray(grid) <= 0, 1.0)
        out = resnap_ops_to_surface(grid, out)
    except Exception as ex:
        print(f"[adjust_ops] regularize skipped ({ex})")
    return out + other, dropped
