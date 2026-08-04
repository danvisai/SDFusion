"""Blender mesh <-> edit engine sync. Imports bpy + the bpy-free `bridge`.

A generated building is a Blender object carrying its editable state as custom properties:
  obj["gt_style"], obj["gt_class"], obj["gt_height"], obj["gt_footprint"] (JSON),
  obj["gt_params"] (JSON recipe params), obj["gt_edits"] (JSON list of edit-op dicts).
Re-meshing reads that state, calls bridge.apply_edits (~10 ms local), and swaps the mesh.
"""

from __future__ import annotations

import json

import bpy  # type: ignore

from . import bridge


def get_state(obj):
    return {
        "style": obj.get("gt_style", "modern"),
        "building_class": obj.get("gt_class", "RESIDENTIAL"),
        "height": float(obj.get("gt_height", 10.0)),
        "footprint": json.loads(obj.get("gt_footprint", "[]")),
        "recipe_params": json.loads(obj.get("gt_params", "[]")),
        "edits": json.loads(obj.get("gt_edits", "[]")),
    }


def set_state(obj, *, style=None, building_class=None, height=None, footprint=None,
              recipe_params=None, edits=None):
    if style is not None:        obj["gt_style"] = style
    if building_class is not None: obj["gt_class"] = building_class
    if height is not None:       obj["gt_height"] = float(height)
    if footprint is not None:    obj["gt_footprint"] = json.dumps(footprint)
    if recipe_params is not None: obj["gt_params"] = json.dumps(list(recipe_params))
    if edits is not None:        obj["gt_edits"] = json.dumps(edits)


def _apply_mesh(obj, verts, faces):
    mesh = obj.data
    mesh.clear_geometry()
    mesh.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    mesh.update()
    obj.data.update_tag()


def make_building_object(context, name, verts, faces):
    mesh = bpy.data.meshes.new(name + "_mesh")
    mesh.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    context.collection.objects.link(obj)
    return obj


def rebuild(obj, res, preview_op=None):
    """Re-mesh `obj` from its stored params + edit stack (+ an optional preview op)."""
    s = get_state(obj)
    if not s["recipe_params"]:
        return
    edits = list(s["edits"])
    if preview_op is not None:
        edits = edits + [preview_op]
    out = bridge.apply_edits(s["style"], s["recipe_params"], s["footprint"],
                             s["height"], edits, res=res)
    _apply_mesh(obj, out["verts"], out["faces"])
