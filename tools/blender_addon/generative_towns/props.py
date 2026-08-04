"""Property groups for the GenerativeTowns add-on (Blender scene + the edit authoring UI)."""

from __future__ import annotations

import bpy  # type: ignore
from bpy.props import (EnumProperty, FloatProperty, IntProperty, BoolProperty,  # type: ignore
                       PointerProperty, FloatVectorProperty)

STYLES = ["modern", "colonial", "victorian", "industrial", "craftsman",
          "mediterranean", "contemporary", "public_civic"]
CLASSES = ["RESIDENTIAL", "COMMERCIAL", "PUBLIC", "RELIGIOUS"]
PALETTE = ["box", "rounded_box", "sphere", "cylinder", "cone", "gable", "hip"]

_styles_enum = [(s, s, "") for s in STYLES]
_classes_enum = [(c, c.title(), "") for c in CLASSES]
_palette_enum = [(p, p.replace("_", " ").title(), "") for p in PALETTE]
_mode_enum = [("add", "Add", "Union"), ("subtract", "Subtract", "Carve")]
_restyle_enum = [("__keep__", "Keep style (cleanup)", "")] + _styles_enum
_refine_mode_enum = [
    ("fast", "Fast — snap to style", "Amortized B+.6 head; cleans + re-styles, drops fine detail"),
    ("quality", "Quality — snap (optimize)", "Slower param-fit; ~= fast"),
    ("displacement", "Displacement — keep detail", "Base recipe + learned residual; preserves sculpted detail (~3s)"),
]


def _live(self, context):
    """Slider/enum update -> re-mesh the active building at preview res with this op."""
    obj = context.active_object
    if obj is None or "gt_params" not in obj or not context.scene.gt.live_preview:
        return
    try:
        from . import mesh_sync
        mesh_sync.rebuild(obj, context.scene.gt.preview_res, preview_op=self.to_op())
    except Exception as e:  # never crash the UI on a bad preview
        print("[gentowns] live preview failed:", e)


class GTEditProps(bpy.types.PropertyGroup):
    """The primitive currently being authored (the brush)."""
    kind: EnumProperty(name="Primitive", items=_palette_enum, default="box", update=_live)
    mode: EnumProperty(name="Mode", items=_mode_enum, default="add", update=_live)
    center: FloatVectorProperty(name="Center", size=3, default=(0, 5, 0), subtype="XYZ", update=_live)
    size: FloatVectorProperty(name="Size", size=3, default=(2, 2, 2), min=0.0, update=_live)
    roof_height: FloatProperty(name="Roof Height", default=3.0, min=0.0, update=_live)
    smooth: FloatProperty(name="Smooth", default=0.4, min=0.0, max=5.0, update=_live)
    rot_y: FloatProperty(name="Rotate Y", default=0.0, update=_live)
    round_r: FloatProperty(name="Round", default=0.0, min=0.0, update=_live)

    def to_op(self) -> dict:
        c = tuple(self.center)
        if self.kind in ("gable", "hip"):
            # size = (width, depth, body_height, roof_height)
            size = (self.size[0], self.size[2], self.size[1], self.roof_height)
        elif self.kind == "sphere":
            size = (self.size[0],)
        elif self.kind in ("cylinder", "cone"):
            size = (self.size[0], self.size[1])
        else:
            size = tuple(self.size)
        return {"kind": self.kind, "center": list(c), "size": list(size),
                "mode": self.mode, "smooth": self.smooth, "rot_y": self.rot_y,
                "round_r": self.round_r}


class GTSceneProps(bpy.types.PropertyGroup):
    """Generation + global settings."""
    style: EnumProperty(name="Style", items=_styles_enum, default="modern")
    building_class: EnumProperty(name="Class", items=_classes_enum, default="RESIDENTIAL")
    height: FloatProperty(name="Height (m)", default=10.0, min=2.0, max=120.0)
    width: FloatProperty(name="Footprint W (m)", default=16.0, min=2.0)
    depth: FloatProperty(name="Footprint D (m)", default=20.0, min=2.0)
    seed: IntProperty(name="Seed", default=0)
    guidance: FloatProperty(name="Diversity (guidance)", default=2.0, min=1.0, max=8.0)
    preview_res: IntProperty(name="Preview Res", default=40, min=16, max=96)
    commit_res: IntProperty(name="Commit Res", default=72, min=32, max=160)
    live_preview: BoolProperty(name="Live Preview", default=True)
    refine_target_style: EnumProperty(name="Refine into", items=_restyle_enum, default="__keep__")
    refine_mode: EnumProperty(name="Refine mode", items=_refine_mode_enum, default="fast")


CLASSES_TO_REGISTER = (GTEditProps, GTSceneProps)


def register():
    for c in CLASSES_TO_REGISTER:
        bpy.utils.register_class(c)
    bpy.types.Scene.gt = PointerProperty(type=GTSceneProps)
    bpy.types.Scene.gt_edit = PointerProperty(type=GTEditProps)


def unregister():
    del bpy.types.Scene.gt_edit
    del bpy.types.Scene.gt
    for c in reversed(CLASSES_TO_REGISTER):
        bpy.utils.unregister_class(c)
