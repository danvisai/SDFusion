"""N-panel UI (View3D sidebar > 'GenTowns' tab)."""

from __future__ import annotations

import bpy  # type: ignore

from . import bridge


class GT_PT_generate(bpy.types.Panel):
    bl_label = "Generate"
    bl_idname = "GT_PT_generate"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "GenTowns"

    def draw(self, context):
        gt = context.scene.gt
        col = self.layout.column(align=True)
        backend = "local (realtime)" if bridge.local_available() else "HTTP fallback"
        col.label(text=f"engine: {backend}", icon="PLUGIN")
        col.prop(gt, "style"); col.prop(gt, "building_class")
        row = col.row(align=True); row.prop(gt, "width"); row.prop(gt, "depth")
        col.prop(gt, "height"); col.prop(gt, "seed"); col.prop(gt, "guidance")
        col.separator()
        col.operator("gentowns.generate", icon="MESH_CUBE")


class GT_PT_sculpt(bpy.types.Panel):
    bl_label = "Sculpt"
    bl_idname = "GT_PT_sculpt"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "GenTowns"

    def draw(self, context):
        e = context.scene.gt_edit
        layout = self.layout
        obj = context.active_object
        if obj is None or "gt_params" not in obj:
            layout.label(text="Select a generated building", icon="INFO")
            return
        n = len(__import__("json").loads(obj.get("gt_edits", "[]")))
        layout.label(text=f"{obj.name}  —  {n} edits", icon="MOD_BUILD")

        box = layout.box(); box.label(text="Brush primitive")
        box.prop(e, "kind"); box.prop(e, "mode")
        box.prop(e, "center"); box.prop(e, "size")
        if e.kind in ("gable", "hip"):
            box.prop(e, "roof_height")
        if e.kind == "rounded_box":
            box.prop(e, "round_r")
        box.prop(e, "smooth"); box.prop(e, "rot_y")

        row = layout.row(align=True)
        row.operator("gentowns.add_primitive", icon="ADD")
        row.operator("gentowns.undo_edit", icon="LOOP_BACK")


class GT_PT_settings(bpy.types.Panel):
    bl_label = "Settings & Refine"
    bl_idname = "GT_PT_settings"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "GenTowns"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        gt = context.scene.gt
        col = self.layout.column(align=True)
        col.prop(gt, "live_preview")
        row = col.row(align=True); row.prop(gt, "preview_res"); row.prop(gt, "commit_res")
        col.operator("gentowns.commit", icon="CHECKMARK")
        col.separator()
        col.label(text="AI Refine / Re-style:")
        col.prop(gt, "refine_target_style"); col.prop(gt, "refine_mode")
        col.operator("gentowns.refine", icon="SHADERFX")


CLASSES_TO_REGISTER = (GT_PT_generate, GT_PT_sculpt, GT_PT_settings)


def register():
    for c in CLASSES_TO_REGISTER:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(CLASSES_TO_REGISTER):
        bpy.utils.unregister_class(c)
