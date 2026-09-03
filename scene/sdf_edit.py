"""Interactive SDF editing engine — the core of the sculpt-and-refine UX (Stage 4).

A building is one differentiable SDF. This wraps a *base* SDF (e.g. a B+.6-generated
recipe building) and a stack of user EDIT OPERATIONS — add/subtract a primitive from a
palette (box, sphere, cylinder, cone, gable/hip roof) with optional smooth blending,
position, scale, and Y-rotation. Everything composes via the existing CSG ops in
`scene/sdf_primitives.py`, so the result stays differentiable end-to-end (generation →
edit → AI refine all share one representation).

The host (Blender / web) supplies the GUI; this is the headless engine the host calls:
  - `EditableBuilding.add(op)` / `.undo()` / `.clear()`  — mutate the edit stack
  - `.to_mesh(bbox, res)`                                — extract a mesh (fast at low res
                                                            for drag preview, high res on commit)
  - `.evaluate(points)`                                  — raw SDF, for picking / AI refine

Design notes:
  - Additive to `sdf_recipes.py`, and additive to `sdf_primitives.py` except for one
    behaviour-preserving split: #128 needed the *distance to a footprint's walls* on its own, so
    `sdf_polygon_2d` was lifted out of `sdf_polygon_prism`, which now calls it.
  - Edit ops are plain dataclasses (JSON-serializable) so a host can store them as the
    building's editable state (mirrors docs/DEPLOYMENT_PLAN.md: host holds recipe_params +
    edit list; sliders/drag mutate locally; only "AI refine" hits the service).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from scene.sdf_primitives import (
    SDF, sdf_box, sdf_rounded_box, sdf_sphere, sdf_cylinder_y, sdf_cone_y,
    sdf_gable_roof, sdf_hip_roof, sdf_translate, sdf_rotate_y,
    sdf_polygon_prism, sdf_polygon_2d, sdf_plane_halfspace, sdf_intersect,
    sdf_union, sdf_subtract, sdf_smooth_union, sdf_smooth_subtract,
    sample_grid, grid_to_mesh,
)

# `PALETTE` and `PROGRAM_KINDS` are DERIVED from `ALGEBRA` below, so there is one list of kinds
# rather than three that can drift apart. They keep their names because a host imports them
# (`tools/blender_addon/.../bridge.py`).


# ================================================================================================
# #4 -- the semantic architectural edit algebra
#
# The palette above is a list of shapes the compiler accepts. This is the part that says what they
# MEAN: which are architecture and which are raw CSG, which may be added and which only cut, and --
# the distinction everything else turns on -- which are expressible as a **height field**.
#
# 🔑 That last one is load-bearing because of a measurement, not a preference. #10 measured this
# corpus at 64^3: `missing` = 0 on 714/714, 100% of carve volume above the topmost GT voxel in its
# column, **0 voxels** of through-void, 71 overhang voxels in 4,324,919. Every real building is
# exactly a 64x64 height map. So an operation that leaves the height field is not merely unusual
# here -- it has **no training signal at all**, and can never be learned from this corpus however
# good the generator is. `height_map` records that, and `learnable_here` is its consequence.
# ================================================================================================

CORE = "core"                 # 2.5-D; `layer`/`ramp` bidirectional (#140), `cut_roof` subtract-only;
                               # BOTH compilers run it; learnable from this corpus
VOLUMETRIC = "volumetric"     # only the SDF compiler runs it; zero training signal on this corpus


@dataclass(frozen=True)
class OpSpec:
    """What one operation kind is, in the algebra rather than in the compiler."""
    tier: str
    subtractive_only: bool
    height_map: bool                       # expressible as a per-column height?
    requires: Tuple[str, ...] = ()         # EditOp fields that may not be None
    plane_clauses: Optional[int] = None    # exact number of cap clauses, when the kind fixes it
    note: str = ""


_PRISM = ("polygon", "size")

ALGEBRA = {
    # -- the core: what #10 recovered and #6 generates -------------------------------------------
    # 🔑 #140: `layer` and `ramp` are bidirectional -- `mode="add"` raises a column exactly as
    # `mode="subtract"` already lowers one, validated and composed the same way. `cut_roof` stays
    # subtract-only; see its own note below.
    "layer": OpSpec(CORE, False, True, _PRISM,
                    note="one connected region flattened to one height, lowered (subtract) or "
                         "raised (add, #140). ArcPro's CreateLayer, and the operation a SETBACK "
                         "and a TERRACE both resolve to"),
    "ramp": OpSpec(CORE, False, True, _PRISM + ("planes",), plane_clauses=1,
                   note="the tightest PLANE above the target (subtract) or up to it (add, #140) "
                        "over one region -- the shed roof CutRoof cannot express, at arbitrary "
                        "rotation"),
    "cut_roof": OpSpec(CORE, True, True, _PRISM,
                       note="height falls off with distance from the region's edge: hip erodes on "
                            "all sides, gable on one axis. Stays subtract-only (#140) -- its "
                            "additive mirror already exists as the volumetric tier's gable/hip"),
    # -- volumetric: compilable, never learnable from THIS corpus --------------------------------
    "box": OpSpec(VOLUMETRIC, False, False, (), note="raw CSG; how a courtyard or light well is cut"),
    "rounded_box": OpSpec(VOLUMETRIC, False, False, (), note="raw CSG"),
    "sphere": OpSpec(VOLUMETRIC, False, False, (), note="raw CSG"),
    "cylinder": OpSpec(VOLUMETRIC, False, False, (), note="raw CSG; a round light well"),
    "cone": OpSpec(VOLUMETRIC, False, False, (), note="raw CSG"),
    "gable": OpSpec(VOLUMETRIC, False, False, (), note="a roof SOLID, added; the core cuts roofs instead"),
    "hip": OpSpec(VOLUMETRIC, False, False, (), note="a roof SOLID, added; the core cuts roofs instead"),
    "element": OpSpec(VOLUMETRIC, False, False, (), note="retrieved BuildingNet component geometry"),
}


# One source of truth for "what kinds exist", in the order the host shows them: volumetric first
# (the hand palette), then the three architectural operations #10 recovered.
PALETTE = tuple(k for k, v in ALGEBRA.items() if v.tier == VOLUMETRIC) + \
          tuple(k for k, v in ALGEBRA.items() if v.tier == CORE)
# The three layer-program kinds are all one shape -- a polygonal prism, optionally capped by
# half-spaces -- and differ only in the cap: `layer` has none (the whole prism goes), `ramp` has
# one plane, `cut_roof` has one per eave. See `layer_program_to_ops`.
PROGRAM_KINDS = tuple(k for k, v in ALGEBRA.items() if v.tier == CORE)


@dataclass(frozen=True)
class VocabEntry:
    """One name from #4's required vocabulary, and what it resolves to here."""
    kind: Optional[str]
    learnable_here: bool
    note: str


# ⚠️ #4 requires the algebra to represent nine named things. Four of them cannot fire on this
# corpus, two are spellings of `layer`, and one is not an edit at all. Recorded per name so that
# none of them is quietly dropped, and so a later corpus with real voids can flip `learnable_here`
# without anybody having to rediscover why it was False.
ARCHITECTURAL_VOCABULARY = {
    "setback": VocabEntry("layer", True,
                          "NOT a separate operation: in a height field a setback IS a Layer whose "
                          "polygon is the inward offset of the footprint, and #10's fitter finds "
                          "it as one"),
    "terrace": VocabEntry("layer", True, "a Layer, or a stack of them -- the same operation"),
    "roof cut": VocabEntry("cut_roof", True, "the core operation, hip or gable"),
    "roof volume": VocabEntry("ramp", True,
                              "a pitched roof is one or more Ramps; a gable is two opposing ones, "
                              "which is why the core cuts roofs rather than adding solids"),
    # ⚠️ TWO SENSES, and only one of them is "not an edit". Distinguished here because the first
    # version of this entry answered only the plan sense and read as if the system could not do
    # stepped massing at all -- which is false: `Layer` is exactly that operation, the recovered
    # programs average 3.06 of them per building, and the served arm produces 3.12 height plateaus
    # per building against GT's 3.34.
    "wing": VocabEntry("layer", True,
                       "a wing at a DIFFERENT HEIGHT is a Layer over that part of the plan -- the "
                       "single commonest operation in the corpus. A wing as plan GEOMETRY (the arm "
                       "of an L) is not an edit at all: it is part of the footprint, which is this "
                       "system's immutable input (#127 conditioning)"),
    "courtyard": VocabEntry("box", False,
                            "a through-void. 0 voxels in 4,324,919 of carve -- cuttable in the SDF, "
                            "never learnable from this corpus"),
    "passage": VocabEntry("box", False, "a through-void; see courtyard. 0 voxels measured"),
    "light well": VocabEntry("cylinder", False, "a through-void; see courtyard. 0 voxels measured"),
    "arcade": VocabEntry("box", False,
                         "an overhang, not a void, but still outside the height field: 71 overhang "
                         "voxels in 4,324,919, which is 0.0016%"),
}


def op_problems(op: "EditOp") -> List[str]:
    """Everything wrong with one operation, as sentences. Empty means well formed.

    #4 asks for *constrained geometry per type* and *invalid references*, and both were previously
    discovered by crashing somewhere inside a prism -- a stack trace rather than a diagnosis. This
    is the predicate a host can run before compiling, and the one `program_problems` reports by
    index so a bad operation can be pointed at rather than hunted.
    """
    out: List[str] = []
    spec = ALGEBRA.get(op.kind)
    if spec is None:
        return [f"unknown kind {op.kind!r}; expected one of {sorted(ALGEBRA)}"]
    if op.mode not in ("add", "subtract"):
        out.append(f"mode must be add|subtract, got {op.mode!r}")
    elif spec.subtractive_only and op.mode != "subtract":
        out.append(f"{op.kind!r} is subtract-only ({spec.note}); got mode {op.mode!r}")
    for fieldname in spec.requires:
        if getattr(op, fieldname, None) is None:
            out.append(f"{op.kind!r} requires {fieldname!r}")
    if op.polygon is not None:
        for i, ring in enumerate(op.polygon):
            if len(ring) < 3:
                out.append(f"ring {i} has {len(ring)} vertices; a polygon ring needs at least 3")
    if spec.plane_clauses is not None and op.planes is not None \
            and len(op.planes) != spec.plane_clauses:
        out.append(f"{op.kind!r} takes exactly {spec.plane_clauses} cap clause(s); "
                   f"got {len(op.planes)}")
    if op.kind == "cut_roof" and op.planes is None and op.roof is None:
        out.append("cut_roof needs either eave planes (gable) or a (rate, offset) cap (hip)")
    return out


def program_problems(ops: Sequence["EditOp"]) -> List[str]:
    """`op_problems` over a stack, each prefixed with the index of the operation it belongs to."""
    return [f"op {i}: {p}" for i, op in enumerate(ops) for p in op_problems(op)]


def commutes(ops: Sequence["EditOp"]) -> bool:
    """Does the order of these operations change the building? 🔑 #4's central question.

    **Measured before this was written**, on 250 recovered programs: 78% have two operations whose
    regions overlap, and permuting them changed the compiled building on **69.6%** -- so the
    serialised algebra WAS order-dependent, and nothing said so. The entire cause was that the
    height-map replay applied `Layer` as a SET, which can raise a column an earlier operation had
    lowered. Reading it as a MIN changed nothing on 0 of 250, and made permutation change nothing
    on 0 of 2,000 permutations.

    The condition is simply that every operation subtracts: `subtract(subtract(B, P), Q)` is
    `subtract(B, union(P, Q))`, and union is commutative and associative. One additive operation
    destroys it, because a union does not commute with a subtraction.

    ⚠️ This is what makes DELETION, EQUIVALENCE and a CANONICAL FORM well defined at all -- an
    ordered algebra has no normal form that is not just "the order you happened to write".

    ⚠️ #140 gives `layer`/`ramp` a real, learnable `mode="add"`, which reopens exactly this: a
    program mixing add and subtract is ordered, and this predicate already says so -- it checks
    every op's `mode`, not its `kind`, so no change was needed here to cover the new additive ops.
    """
    return all(op.mode == "subtract" for op in ops)


def is_height_map_representable(ops: Sequence["EditOp"]) -> bool:
    """Can the per-column compiler run this program, or does it need the full SDF?

    On this corpus that is the same question as "could this ever have been learned": a program that
    leaves the height field has zero training signal here (0 through-void voxels in 4.3M).
    """
    return all(ALGEBRA[op.kind].height_map for op in ops if op.kind in ALGEBRA) and all(
        op.kind in ALGEBRA for op in ops)


def canonical_form(ops: Sequence["EditOp"]) -> List[dict]:
    """The program's normal form: the same operations in a deterministic order.

    Two spellings of the same building compare equal on this, without compiling either. It is only
    sound because the core commutes -- so it REFUSES a program that does not, rather than sorting
    an ordered stack and silently changing what it denotes.

    The key is (kind, then the geometry) rather than anything positional, so it survives a round
    trip through JSON and does not depend on how the host happened to enumerate the ops.
    """
    if not commutes(ops):
        raise ValueError("canonical_form is only defined for a commuting (all-subtractive) "
                         "program; this one contains an additive operation")
    problems = program_problems(ops)
    if problems:
        raise ValueError("; ".join(problems))
    return sorted((op.to_dict() for op in ops), key=lambda d: json.dumps(d, sort_keys=True))


def equivalent(base_sdf: SDF, a: Sequence["EditOp"], b: Sequence["EditOp"],
               res: int = 64, device: str = "cpu") -> bool:
    """Do two programs denote the same building? Decided on the GEOMETRY, not the spelling.

    `canonical_form` is the cheap syntactic test and this is the semantic one: it catches the cases
    a sort cannot, such as an operation that removes nothing because another already removed it.
    """
    ea = EditableBuilding(base_sdf, list(a)).to_occupancy(res=res, device=device)
    eb = EditableBuilding(base_sdf, list(b)).to_occupancy(res=res, device=device)
    return bool(np.array_equal(ea, eb))

# Why every cut plane carries a `+1` on the y term, and why no epsilon is needed with it.
#
# The height-map compiler keeps voxel `y` where `y - y0 < floor(v)` for a real-valued surface `v`.
# For integer heights that is exactly `y - y0 + 1 <= v`, so the *continuous* plane `y - y0 + 1 = v`
# separates the two cases with no rounding left over -- the floor is not an approximation to model,
# it disappears. `y - y0 >= v` is the tempting reading and it is wrong: at `v = 10.5` it drops the
# voxel the compiler keeps.
#
# On the boundary itself -- `v` integral, so the plane passes exactly through a voxel centre -- the
# compiler keeps the voxel and `sdf_subtract` agrees in exact arithmetic, removing only where the
# cutting solid is strictly negative. In float32 it does not: the plane evaluates to a few times
# 1e-7 of either sign and the tie falls whichever way rounding went. `_TIE` pulls cut planes back
# by a margin well above that noise and well below anything geometric (~3e-4 of a voxel), so ties
# land on the kept side on purpose rather than by luck.
_TIE = 1e-5


@dataclass
class EditOp:
    """One user edit. `size` is primitive-specific (see _primitive)."""
    kind: str                          # one of PALETTE
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)   # world position
    size: Tuple[float, ...] = (1.0, 1.0, 1.0)
    mode: str = "add"                  # "add" (union) | "subtract"
    smooth: float = 0.0                # blend radius; 0 = hard CSG
    rot_y: float = 0.0                 # degrees about world Y
    round_r: float = 0.0               # corner rounding for box
    lib_id: int = -1                   # kind='element': index into data/element_library_v1
                                       # (real BuildingNet component geometry, Phase R3 of
                                       # GENERATIVE_MAKE_IT_ARCHITECTURE_BUILD_SPEC)
    # -- the layer-program kinds (#128). Both stay None for every earlier kind, so old edit state
    #    loads unchanged and `to_dict` output is still plain JSON.
    polygon: Optional[List[List[List[float]]]] = None
                                       # rings of world-XZ vertices, outer first then holes, all
                                       # CCW. For these kinds `size` is (y_low, y_high) in world
                                       # units -- the slab the region spans -- not a half-extent.
    planes: Optional[List[List[List[float]]]] = None
                                       # cap clauses, in disjunctive normal form: the op removes
                                       # where ANY clause holds, and a clause holds where EVERY
                                       # (nx, ny, nz, d) in it has n.p + d <= 0. A ramp is one
                                       # clause of one plane; a gable eave is a plane clipped to
                                       # the rows its wall spans. None removes the whole prism.
    roof: Optional[List[float]] = None
                                       # kind='cut_roof', hipped: (rate, offset). The op removes
                                       # where  y - rate * dist_inside_the_walls + offset > 0 -- a
                                       # cap following the region's outline, not any one wall.

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        # tolerate host-side annotation keys (e.g. 'det' type tags, 'layer') on the wire
        keys = EditOp.__dataclass_fields__.keys()
        return EditOp(**{k: v for k, v in d.items() if k in keys})


def _region_solid(op: EditOp) -> SDF:
    """The solid a `layer` / `ramp` / `cut_roof` op removes.

    One polygonal prism over the op's region, intersected with the union of its caps. A `layer`
    carries no cap, so the whole prism goes; a `ramp` carries one plane; a hipped `cut_roof` carries
    a cap over the region's outline; a gabled one carries a clause per eave. **Union** is the right
    combination for several caps because a roof height is the *minimum* of them -- a column is cut
    if it rises above any one.
    """
    rings = op.polygon or []
    if not rings:
        raise ValueError(f"kind '{op.kind}' needs a polygon: rings of world-XZ vertices")
    if len(op.size) < 2:
        raise ValueError(f"kind '{op.kind}' needs size=(y_low, y_high), got {op.size!r}")
    y_lo, y_hi = float(op.size[0]), float(op.size[1])
    if y_hi <= y_lo:
        raise ValueError(f"kind '{op.kind}' needs y_high > y_low, got {op.size!r}")

    def prism(ring):
        return sdf_translate(sdf_polygon_prism(np.asarray(ring, np.float32), y_hi - y_lo),
                             (0.0, y_lo, 0.0))

    solid = prism(rings[0])
    if len(rings) > 1:                                   # holes are subtracted, never unioned
        solid = sdf_subtract(solid, sdf_union(*[prism(r) for r in rings[1:]]))
    caps = [_clause_solid(clause) for clause in (op.planes or [])]
    if op.roof:
        caps.append(_roof_cap(rings, float(op.roof[0]), float(op.roof[1])))
    if caps:
        solid = sdf_intersect(solid, sdf_union(*caps) if len(caps) > 1 else caps[0])
    return solid


def _clause_solid(clause) -> SDF:
    """One cap clause: the intersection of its half-spaces."""
    if not clause:
        raise ValueError("a cap clause needs at least one plane")
    parts = [sdf_plane_halfspace(pl[:3], pl[3]) for pl in clause]
    return parts[0] if len(parts) == 1 else sdf_intersect(*parts)


def _region_outline(rings) -> SDF:
    """Signed XZ distance to a region's boundary: outer ring, minus its holes."""
    outer = sdf_polygon_2d(np.asarray(rings[0], np.float32))
    if len(rings) == 1:
        return outer
    holes = [sdf_polygon_2d(np.asarray(r, np.float32)) for r in rings[1:]]

    def f(p: torch.Tensor) -> torch.Tensor:
        d = outer(p)
        for h in holes:
            d = torch.maximum(d, -h(p))
        return d
    return f


def _roof_cap(rings, rate: float, offset: float) -> SDF:
    """A hipped cap over a region: removes where `y - rate * dist_inside_the_walls + offset > 0`.

    Written below against `_region_outline`, which is *signed and negative inside*, so the same
    statement reads `y + rate * outline + offset` there.

    Not a half-space, and that is the point. One inward-slanted plane per wall is the *straight
    skeleton*, which is right until the footprint turns a reflex corner -- there the nearest wall is
    a vertex, the roof surface is a cone around it, and the plane form keeps cutting on a line the
    building does not have. Reading the region's own outline distance gets the corner for free.
    """
    outline = _region_outline(rings)
    scale = float(np.hypot(1.0, rate))

    def f(p: torch.Tensor) -> torch.Tensor:
        return -(p[..., 1] + rate * outline(p) + offset) / scale + _TIE
    return f


def _primitive(op: EditOp) -> SDF:
    """Build the (origin-centered) primitive SDF for an op, then rotate+translate it."""
    k, s = op.kind, op.size
    if k == "box":
        prim = sdf_box(s[:3])
    elif k == "rounded_box":
        prim = sdf_rounded_box(s[:3], op.round_r)
    elif k == "sphere":
        prim = sdf_sphere(s[0])
    elif k == "cylinder":
        prim = sdf_cylinder_y(s[0], s[1])
    elif k == "cone":
        prim = sdf_cone_y(s[0], s[1])
    elif k == "gable":
        # size = (width, depth, body_height, roof_height); built with base at y=0.
        prim = sdf_gable_roof(s[0], s[1], s[2], s[3], center_xz=(0.0, 0.0))
    elif k == "hip":
        prim = sdf_hip_roof(s[0], s[1], s[2], s[3], center_xz=(0.0, 0.0))
    elif k in PROGRAM_KINDS:
        # region geometry is already in world coordinates, so it is not re-centred below
        return _region_solid(op)
    elif k == "element":
        # real library geometry stretched to fill the op's box; device follows the query
        # points at call time (the lib caches per-device tensors)
        from scene.element_lib import element_sdf
        _fns = {}

        def prim(p, _lid=int(op.lib_id), _half=tuple(float(v) for v in s[:3])):
            dev = str(p.device)
            if dev not in _fns:
                _fns[dev] = element_sdf(_lid, _half, device=p.device)
            return _fns[dev](p)
    else:
        raise ValueError(f"unknown primitive '{k}'; palette={PALETTE}")
    if abs(op.rot_y) > 1e-6:
        prim = sdf_rotate_y(prim, op.rot_y)
    return sdf_translate(prim, op.center)


class EditableBuilding:
    def __init__(self, base_sdf: SDF, ops: Optional[List[EditOp]] = None):
        self.base_sdf = base_sdf
        self.ops: List[EditOp] = list(ops) if ops else []

    # -- edit stack --------------------------------------------------------
    def add(self, op: EditOp) -> "EditableBuilding":
        self.ops.append(op)
        return self

    def undo(self) -> Optional[EditOp]:
        return self.ops.pop() if self.ops else None

    def remove(self, index: int) -> EditOp:
        """Delete operation `index`, not just the last one. #4's *deletion*.

        🔑 Only well defined because the core commutes: with every operation subtractive the stack
        denotes `subtract(base, union(all of them))`, so dropping one is exactly the program
        without it, whatever position it held. `undo()` can only unwind from the top, which cannot
        serve #3's edit locality -- a user re-rolling one decision while everything unrelated
        survives is this operation, not a rewind.

        ⚠️ A negative index is REFUSED rather than wrapped. Python would delete the last operation
        for `remove(-1)`, which is a silent wrong answer when the caller passed an id it failed to
        resolve -- #4's *invalid references*, and the kind of bug that shows up as a user losing an
        edit they did not touch.
        """
        if not isinstance(index, int) or isinstance(index, bool):
            raise IndexError(f"operation index must be an int, got {index!r}")
        if index < 0 or index >= len(self.ops):
            raise IndexError(f"no operation at index {index}; the stack has {len(self.ops)}")
        return self.ops.pop(index)

    def clear(self):
        self.ops.clear()

    # -- composed SDF ------------------------------------------------------
    def composed(self) -> SDF:
        s = self.base_sdf
        for op in self.ops:
            prim = _primitive(op)
            if op.mode == "add":
                s = sdf_smooth_union(s, prim, op.smooth) if op.smooth > 0 else sdf_union(s, prim)
            elif op.mode == "subtract":
                s = (sdf_smooth_subtract(s, prim, op.smooth) if op.smooth > 0
                     else sdf_subtract(s, prim))
            else:
                raise ValueError(f"mode must be add|subtract, got {op.mode}")
        return s

    def evaluate(self, points: torch.Tensor) -> torch.Tensor:
        return self.composed()(points)

    def to_mesh(self, bbox: Sequence[float], res: int = 64, device: str = "cpu"):
        grid = sample_grid(self.composed(), res, tuple(bbox), device=device)
        return grid_to_mesh(grid, tuple(bbox), iso=0.0)

    def to_occupancy(self, res: int = 64, device: str = "cpu",
                     chunk: int = 1 << 14) -> np.ndarray:
        """Occupancy on the corpus grid, [z, y, x] -- the array a voxel compiler produces.

        Sampled on the [-1, 1] frame `real.h5` uses, at `field <= 0`, so the result is directly
        comparable with `eval_massing_arms` and with the layer program's own compiler. `chunk` is
        smaller than `sample_grid`'s default because a polygon prism costs O(points x vertices).
        """
        grid = sample_grid(self.composed(), res, (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
                           device=device, chunk=chunk)
        return (grid <= 0.0).cpu().numpy()

    # -- serialization (host stores this as the building's editable state) --
    def edit_state(self):
        return [op.to_dict() for op in self.ops]

    @staticmethod
    def from_state(base_sdf: SDF, state):
        return EditableBuilding(base_sdf, [EditOp.from_dict(d) for d in state])


def recipe_base_sdf(style: str, params, polygon_xz, height: float, device: str = "cpu") -> SDF:
    """Wrap a DiffRecipe forward as a plain SDF callable, so a generated building can be
    the editable base."""
    from models.networks.diff_recipe import build_diff_recipe
    module = build_diff_recipe(style)[0].to(device)
    p = torch.as_tensor(np.asarray(params, np.float32), device=device)
    poly = torch.as_tensor(np.asarray(polygon_xz, np.float32), device=device)
    h = torch.as_tensor(float(height), device=device)

    def f(pts: torch.Tensor) -> torch.Tensor:
        return module(p, poly, h, pts)
    return f


# ==================================================================================================
# the layer-program bridge (#128)
#
# #10 recovered a semantic layer program for every real building, but those programs compile
# straight to a voxel grid through their own deterministic compiler. `CONTEXT.md` calls
# Editable / Reversible "the load-bearing claim of the project", and a program that cannot be
# re-rolled or undone through this stack is geometry evidence, not a recipe. These functions are
# the trip: recovered program -> `EditOp`s -> the same composed SDF everything else here speaks.
#
# ⚠️ Region polygons trace the voxel boundary exactly, at half-voxel offsets, with only collinear
#    vertices dropped. That is lossless, and it is *not* the polygon simplification a real DSL
#    token budget needs -- that question is unstarted and belongs to #4.
# ==================================================================================================



def _spacing(res: int) -> float:
    """World units per voxel on the `real.h5` frame, which spans [-1, 1] at every resolution."""
    return 2.0 / (res - 1)


def _to_world(index: float, res: int) -> float:
    return float(index) * _spacing(res) - 1.0


def _signed_area(ring: np.ndarray) -> float:
    x, z = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.sum(x * np.roll(z, -1) - np.roll(x, -1) * z))


def _drop_collinear(ring: np.ndarray) -> np.ndarray:
    """Remove vertices that lie on the segment between their neighbours. Exactly lossless."""
    keep = []
    n = len(ring)
    for i in range(n):
        a, b, c = ring[i - 1], ring[i], ring[(i + 1) % n]
        if abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) > 1e-9:
            keep.append(b)
    return np.asarray(keep, np.float64) if len(keep) >= 3 else ring


def _boundary_edges(m: np.ndarray):
    """Every cell-boundary segment between an inside and an outside cell, interior on the left.

    Directed so a loop traced by following them comes out counter-clockwise around solid and
    clockwise around a hole, which is what makes outer/hole classification a matter of sign rather
    than of a containment test.
    """
    Z, X = m.shape
    edges: dict = {}

    def add(a, b):
        edges.setdefault(a, []).append(b)

    for z, x in zip(*np.nonzero(m)):
        z, x = int(z), int(x)
        if z + 1 >= Z or not m[z + 1, x]:
            add((x + 0.5, z + 0.5), (x - 0.5, z + 0.5))
        if z - 1 < 0 or not m[z - 1, x]:
            add((x - 0.5, z - 0.5), (x + 0.5, z - 0.5))
        if x + 1 >= X or not m[z, x + 1]:
            add((x + 0.5, z - 0.5), (x + 0.5, z + 0.5))
        if x - 1 < 0 or not m[z, x - 1]:
            add((x - 0.5, z + 0.5), (x - 0.5, z - 0.5))
    return edges


def _chain_loops(edges: dict) -> List[np.ndarray]:
    """Walk the directed boundary segments into closed loops.

    Where four cells meet at a corner diagonally, a vertex carries two outgoing segments and the
    walk has a choice. It takes the sharpest right turn, which hugs the region it is already
    tracing instead of hopping across the pinch into the other diagonal.
    """
    loops = []
    while edges:
        start = next(iter(edges))
        loop, cur, incoming = [start], start, None
        while True:
            outs = edges[cur]
            if len(outs) == 1 or incoming is None:
                nxt = outs.pop(0)
            else:
                d_in = (cur[0] - incoming[0], cur[1] - incoming[1])
                def turn(o):
                    d = (o[0] - cur[0], o[1] - cur[1])
                    cross = d_in[0] * d[1] - d_in[1] * d[0]
                    return np.arctan2(cross, d_in[0] * d[0] + d_in[1] * d[1])
                nxt = min(outs, key=turn)
                outs.remove(nxt)
            if not edges[cur]:
                del edges[cur]
            incoming, cur = cur, nxt
            if cur == start:
                break
            loop.append(cur)
        loops.append(np.asarray(loop, np.float64))
    return loops


def mask_to_rings(mask: np.ndarray) -> List[np.ndarray]:
    """One connected voxel region -> its exact boundary rings, in (x, z) **voxel-index** units.

    Returns the outer ring first, then any holes, every ring counter-clockwise so each can be
    extruded by `sdf_polygon_prism` with an outward normal (holes are subtracted by position, not
    by winding). Vertices land on half-voxel offsets, so a voxel centre never lies on an edge and
    the point-in-polygon test can never tie.

    The boundary is traced along cell edges, so it is **rectilinear**: every segment is axis-aligned
    and a corner is a right angle. `skimage.measure.find_contours` is the obvious tool and is wrong
    for this -- marching squares chamfers each corner diagonally, which still covers the same cells
    but hands `CutRoof` four extra 45-degree eaves per corner that the footprint does not have.

    Raises if `mask` has more than one connected component: a `Layer` is one polygon by definition
    (`recover_massing_programs._layer_candidates` splits components), and silently merging them
    would produce geometry no operation in the vocabulary can express.
    """
    from scipy import ndimage

    m = np.asarray(mask, bool)
    if not m.any():
        return []
    _lab, n_comp = ndimage.label(m)
    if n_comp > 1:
        raise ValueError(f"mask has {n_comp} connected components; split them first "
                         "(one Layer is one polygon)")
    rings = []
    for loop in _chain_loops(_boundary_edges(m)):
        ring = _drop_collinear(loop)
        if len(ring) < 3:
            continue
        if _signed_area(ring) < 0:                           # a hole traces clockwise; flip it
            ring = ring[::-1].copy()
        rings.append(ring)
    rings.sort(key=lambda r: -abs(_signed_area(r)))          # the outer boundary is the largest
    return rings


def _rings_to_world(rings, res: int) -> List[List[List[float]]]:
    s = _spacing(res)
    return [[[float(x) * s - 1.0, float(z) * s - 1.0] for x, z in np.asarray(r, float)]
            for r in rings]


def _cut_plane(normal, offset) -> List[float]:
    """One stored cut plane: unit-length, so `d` reads as a world distance, and pulled back by
    `_TIE` so a voxel centre the plane passes exactly through is kept, as `floor` keeps it."""
    n = np.asarray(normal, float)
    norm = float(np.linalg.norm(n))
    return [*(n / norm), float(offset) / norm + _TIE]


def mask_components_rings(mask: np.ndarray) -> List[List[np.ndarray]]:
    """A mask of any shape -> one ring list per connected region, in voxel-index units.

    `mask_to_rings` deliberately refuses a mask with several components, because a `Layer` is one
    polygon. A *footprint* carries no such promise -- 2 of the pinned 714 are two separate pieces --
    so every caller handed a footprint goes through here instead.
    """
    from scipy import ndimage

    m = np.asarray(mask, bool)
    lab, n_comp = ndimage.label(m)
    return [mask_to_rings(lab == c) for c in range(1, n_comp + 1)]


def footprint_envelope_sdf(fp: np.ndarray, y0: int, y1: int, res: int = 64) -> SDF:
    """The footprint envelope, as the editable base a layer program carves from.

    The same solid `eval_massing_arms.blockout_sdf` produces, built analytically instead of by a
    distance transform, so `field <= 0` reproduces that occupancy voxel for voxel.
    """
    mask = np.asarray(fp, bool)
    if not mask.any():
        raise ValueError("empty footprint")
    half = _spacing(res) / 2.0
    y_lo, y_hi = _to_world(y0, res) - half, _to_world(y1, res) + half
    solids = [_region_solid(EditOp(kind="layer", size=(y_lo, y_hi),
                                   polygon=_rings_to_world(rings, res)))
              for rings in mask_components_rings(mask)]
    return solids[0] if len(solids) == 1 else sdf_union(*solids)


def _eave_planes(rings_idx, kind: str, eaves: int, rate: float, y0: int, res: int):
    """One cap clause per gable eave: its slanted plane, clipped to the wall it belongs to.

    The recovered rule is `height = eaves + (d - 1) * rate` with `d` the count of cells to the
    nearest cell outside the footprint, so a boundary cell has d = 1 and its centre sits half a
    voxel inside the wall: in continuous terms `height = eaves - rate/2 + rate * dist_to_wall`.
    A gable measures that distance **along one axis, staying in its own row**, and a bare half-space
    does neither -- the plane raised by the step of an L keeps cutting across the rows below it,
    and it also cuts on its outward side where the wall is not what a gable would meet. Each eave
    therefore becomes a clause of four half-spaces: the slanted plane, the two ends of the wall's
    own span, and the wall's interior side. Their union is then the nearest wall facing each cell,
    which is what a run length is.
    """
    s = _spacing(res)
    axis = {"gable_x": 0, "gable_z": 1}[kind]
    span = 1 - axis                                          # the wall's own extent runs across it
    clauses = []
    for r_i, ring in enumerate(rings_idx):
        ring = np.asarray(ring, float)
        for i in range(len(ring)):
            v, w = ring[i], ring[(i + 1) % len(ring)]
            d = w - v
            length = float(np.linalg.norm(d))
            if length < 1e-9:
                continue
            # interior lies left of travel on a CCW outer ring, right of it on a CCW hole
            n2 = np.array([-d[1], d[0]]) / length
            if r_i > 0:
                n2 = -n2
            if abs(n2[axis]) < 0.5:
                continue                                     # not an eave this gable sees
            v_w = v * s - 1.0
            # removed where  y_idx - y0 + 1 > eaves - rate/2 + rate * n2.(p_idx - v_idx)
            normal = (-rate * n2[0], 1.0, -rate * n2[1])
            off = 1.0 - s * (y0 + eaves - 0.5 * rate - 1.0) + rate * float(n2 @ v_w)
            clause = [_cut_plane([-c for c in normal], -off)]

            # the wall's interior side: n2 . (p - v) >= 0
            side = [0.0, 0.0, 0.0]
            side[0], side[2] = -n2[0], -n2[1]
            clause.append(_cut_plane(side, float(n2 @ v_w)))

            # the rows this wall spans, in world units; both ends sit on half-voxel offsets
            lo, hi = sorted((float(v[span]), float(w[span])))
            lo_w, hi_w = lo * s - 1.0, hi * s - 1.0
            low_n = [0.0, 0.0, 0.0]
            high_n = [0.0, 0.0, 0.0]
            low_n[2 * span] = -1.0                           # span 0 -> x, span 1 -> z
            high_n[2 * span] = 1.0
            clause.append(_cut_plane(low_n, lo_w))
            clause.append(_cut_plane(high_n, -hi_w))
            clauses.append(clause)
    return clauses


def layer_program_to_ops(program, fp: np.ndarray, y0: int, y1: int,
                         res: int = 64) -> List[EditOp]:
    """A recovered layer program -> the `EditOp` stack that carves the same building.

    `program` is the list `recover_massing_programs` writes: each entry carries its operation, and
    -- since the geometry is what makes it replayable -- its `region` rings in voxel-index units
    (`Ramp` also carries the `plane` its linear program solved for). Every op is subtractive: #10
    measured `missing` = 0.000000 on 714/714, so real massing is only ever cut out of its own
    envelope, never added to.
    """
    mask = np.asarray(fp, bool)
    half = _spacing(res) / 2.0
    top = _to_world(y1, res) + half
    ops: List[EditOp] = []
    for entry in program:
        kind = entry["op"]
        regions = [entry["region"]] if entry.get("region") is not None else None
        if regions is None:
            if kind != "CutRoof":
                raise ValueError(f"{kind} op carries no region; re-run the recovery to record it")
            # a roof is cut over the whole footprint, which may be in more than one piece
            regions = mask_components_rings(mask)
        for rings_idx in regions:
            ops.append(_op_for(entry, kind, rings_idx, _rings_to_world(rings_idx, res),
                               y0, half, top, res))
    return ops


def _op_for(entry, kind, rings_idx, rings, y0, half, top, res) -> EditOp:
    """One operation of a recovered program, over one connected region."""
    if kind == "Layer":
        cut = int(entry["height"])
        return EditOp(kind="layer", mode="subtract", polygon=rings,
                      size=(_to_world(y0 + cut, res) - half, top))
    if kind == "Ramp":
        if "plane" not in entry:
            raise ValueError("Ramp op carries no plane; re-run the recovery to record it")
        a, b, c = (float(v) for v in entry["plane"])
        # removed where  y_idx - y0 + 1 > a + b*x_idx + c*z_idx
        plane = _cut_plane([b, -1.0, c], -(1.0 - _spacing(res) * (y0 + a - 1.0) - b - c))
        return EditOp(kind="ramp", mode="subtract", polygon=rings, planes=[[plane]],
                      size=(_to_world(y0 + 1, res) - half, top))
    if kind == "CutRoof":
        roof_kind, eaves, rate = entry["kind"], int(entry["eaves"]), float(entry["rate"])
        slab = (_to_world(y0 + 1, res) - half, top)
        if roof_kind == "hip":
            # removed where  y_idx - y0 + 1 > eaves + rate * (dist_to_wall - 1/2)
            offset = 1.0 - _spacing(res) * (y0 + eaves - 0.5 * rate - 1.0)
            return EditOp(kind="cut_roof", mode="subtract", polygon=rings, size=slab,
                          roof=[rate, offset])
        return EditOp(kind="cut_roof", mode="subtract", polygon=rings, size=slab,
                      planes=_eave_planes(rings_idx, roof_kind, eaves, rate, y0, res))
    raise ValueError(f"unknown layer-program operation '{kind}'")
