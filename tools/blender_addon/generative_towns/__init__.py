"""GenerativeTowns — interactive sculpt-and-refine add-on for Blender.

Generate a building from symbolic input (footprint + class + height + style) with the B+.6
diffusion head, then sculpt it in realtime with a primitive palette (the building is one
differentiable SDF; edits re-mesh in ~10 ms locally). View3D sidebar > 'GenTowns'.

Requires our torch stack reachable from Blender's Python. Easiest: run Blender from a shell
where `GENTOWNS_REPO` and `GENTOWNS_VENV_SITE` point at the SDFusion repo + its venv
site-packages (see tools/blender_addon/README.md). Falls back to the Stage A HTTP service
for "Generate" only if local import fails.
"""

bl_info = {
    "name": "GenerativeTowns — SDF Sculpt",
    "author": "Danvi Simhadri",
    "version": (0, 1, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar > GenTowns",
    "description": "Generate buildings from OSM-style symbolic input and sculpt them with SDF primitives",
    "category": "Add Mesh",
}

import importlib

from . import bridge, mesh_sync, props, operators, panels

# Reload submodules on re-enable during dev.
for _m in (bridge, mesh_sync, props, operators, panels):
    importlib.reload(_m)

_MODULES = (props, operators, panels)


def register():
    for m in _MODULES:
        m.register()


def unregister():
    for m in reversed(_MODULES):
        m.unregister()


if __name__ == "__main__":
    register()
