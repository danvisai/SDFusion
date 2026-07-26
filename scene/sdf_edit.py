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
  - Additive only — does not modify `sdf_primitives.py` / `sdf_recipes.py`.
  - Edit ops are plain dataclasses (JSON-serializable) so a host can store them as the
    building's editable state (mirrors docs/DEPLOYMENT_PLAN.md: host holds recipe_params +
    edit list; sliders/drag mutate locally; only "AI refine" hits the service).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from scene.sdf_primitives import (
    SDF, sdf_box, sdf_rounded_box, sdf_sphere, sdf_cylinder_y, sdf_cone_y,
    sdf_gable_roof, sdf_hip_roof, sdf_translate, sdf_rotate_y,
    sdf_union, sdf_subtract, sdf_smooth_union, sdf_smooth_subtract,
    sample_grid, grid_to_mesh,
)

PALETTE = ("box", "rounded_box", "sphere", "cylinder", "cone", "gable", "hip", "element")


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

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        # tolerate host-side annotation keys (e.g. 'det' type tags, 'layer') on the wire
        keys = EditOp.__dataclass_fields__.keys()
        return EditOp(**{k: v for k, v in d.items() if k in keys})


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
