"""Operators: Generate, Add Primitive, Undo Edit, Commit (high-res), AI Refine."""

from __future__ import annotations

import json

import bpy  # type: ignore

from . import bridge, mesh_sync


def _rect_footprint(w, d):
    hw, hd = w / 2.0, d / 2.0
    return [[-hw, -hd], [hw, -hd], [hw, hd], [-hw, hd]]


class GT_OT_generate(bpy.types.Operator):
    bl_idname = "gentowns.generate"
    bl_label = "Generate Building"
    bl_description = "Sample a building from the B+.6 generative head and create it"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        gt = context.scene.gt
        fp = _rect_footprint(gt.width, gt.depth)
        try:
            res = bridge.generate(fp, gt.style, gt.building_class, gt.height,
                                  seed=gt.seed, guidance=gt.guidance, res=gt.commit_res)
        except Exception as e:
            self.report({"ERROR"}, f"generate failed: {e}")
            return {"CANCELLED"}
        if res.get("backend") == "http":
            self.report({"ERROR"}, "Local engine unavailable; HTTP generate returns glb "
                                   "(import via File > Import > glTF). Edits need local engine.")
            return {"CANCELLED"}
        name = f"GT_{gt.style}_{gt.seed}"
        obj = mesh_sync.make_building_object(context, name, res["verts"], res["faces"])
        mesh_sync.set_state(obj, style=gt.style, building_class=gt.building_class,
                            height=gt.height, footprint=fp,
                            recipe_params=res["recipe_params"], edits=[])
        for o in context.selected_objects:
            o.select_set(False)
        obj.select_set(True); context.view_layer.objects.active = obj
        self.report({"INFO"}, f"generated {name} ({len(res['verts'])} verts)")
        return {"FINISHED"}


def _require_building(context):
    obj = context.active_object
    if obj is None or "gt_params" not in obj:
        return None
    return obj


class GT_OT_add_primitive(bpy.types.Operator):
    bl_idname = "gentowns.add_primitive"
    bl_label = "Add Primitive"
    bl_description = "Commit the current brush primitive to the active building"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        obj = _require_building(context)
        if obj is None:
            self.report({"ERROR"}, "select a generated building first")
            return {"CANCELLED"}
        s = mesh_sync.get_state(obj)
        edits = s["edits"] + [context.scene.gt_edit.to_op()]
        mesh_sync.set_state(obj, edits=edits)
        try:
            mesh_sync.rebuild(obj, context.scene.gt.commit_res)
        except Exception as e:
            self.report({"ERROR"}, f"edit failed: {e}")
            return {"CANCELLED"}
        self.report({"INFO"}, f"{len(edits)} edits")
        return {"FINISHED"}


class GT_OT_undo_edit(bpy.types.Operator):
    bl_idname = "gentowns.undo_edit"
    bl_label = "Undo Last Edit"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        obj = _require_building(context)
        if obj is None:
            return {"CANCELLED"}
        s = mesh_sync.get_state(obj)
        if not s["edits"]:
            self.report({"INFO"}, "no edits to undo")
            return {"CANCELLED"}
        mesh_sync.set_state(obj, edits=s["edits"][:-1])
        mesh_sync.rebuild(obj, context.scene.gt.commit_res)
        return {"FINISHED"}


class GT_OT_commit(bpy.types.Operator):
    bl_idname = "gentowns.commit"
    bl_label = "Commit (High-Res)"
    bl_description = "Re-mesh the active building at commit resolution"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        obj = _require_building(context)
        if obj is None:
            return {"CANCELLED"}
        mesh_sync.rebuild(obj, context.scene.gt.commit_res)
        return {"FINISHED"}


class GT_OT_refine(bpy.types.Operator):
    bl_idname = "gentowns.refine"
    bl_label = "AI Refine"
    bl_description = ("Project the sculpt onto a clean recipe building (cleanup, or re-style "
                      "into another style), keeping the massing")
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        obj = _require_building(context)
        if obj is None:
            self.report({"ERROR"}, "select a generated building first")
            return {"CANCELLED"}
        gt = context.scene.gt
        s = mesh_sync.get_state(obj)
        ts = None if gt.refine_target_style == "__keep__" else gt.refine_target_style
        base_state = {"style": s["style"], "recipe_params": s["recipe_params"],
                      "footprint": s["footprint"], "height": s["height"]}
        try:
            out = bridge.refine(base_state, s["edits"], target_style=ts, mode=gt.refine_mode,
                                building_class=s["building_class"], seed=gt.seed)
        except Exception as e:
            self.report({"ERROR"}, f"refine failed: {e}")
            return {"CANCELLED"}
        # Refine bakes the edits into a fresh clean recipe building -> reset the edit stack.
        mesh_sync._apply_mesh(obj, out["verts"], out["faces"])
        mesh_sync.set_state(obj, style=out["style"], height=out["height"],
                            footprint=out["footprint"], recipe_params=out["recipe_params"],
                            edits=[])
        self.report({"INFO"}, f"refined into {out['style']} (IoU→edit {out['iou_to_edit']:.2f})")
        return {"FINISHED"}


CLASSES_TO_REGISTER = (GT_OT_generate, GT_OT_add_primitive, GT_OT_undo_edit,
                       GT_OT_commit, GT_OT_refine)


def register():
    for c in CLASSES_TO_REGISTER:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(CLASSES_TO_REGISTER):
        bpy.utils.unregister_class(c)
