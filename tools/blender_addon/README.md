# GenerativeTowns — Blender add-on (Stage B)

Interactive sculpt-and-refine UX for the Option B+ generative head. Generate a building
from symbolic input (footprint + class + height + style) with the B+.6 diffusion model,
then sculpt it in **realtime** with a primitive palette — the building is one
differentiable SDF, so edits re-mesh locally in ~10 ms.

## What it does

- **Generate** panel: pick style/class/footprint/height/seed/diversity → "Generate Building"
  (runs the B+.6 head, ~1–2 s warm-up on first call).
- **Sculpt** panel: a brush primitive (box / rounded-box / sphere / cylinder / cone / gable /
  hip), add or subtract, with position / size / smooth-blend / rotation. Live-preview on
  slider drag (~10 ms), "Add Primitive" to commit, "Undo".
- **Settings & Refine**: preview/commit resolution, "Commit (High-Res)", and an "AI Refine"
  stub (wired to the future `/refine_with_edit` endpoint).

Each building is a Blender object that stores its editable state as custom properties
(`gt_style`, `gt_params`, `gt_footprint`, `gt_edits`), so a .blend round-trips the edits.

## Architecture

```
Blender (bpy UI)  ──>  generative_towns.bridge  ──in-process──>  torch edit engine
   panels/operators        (no bpy)                              scene/sdf_edit.EditableBuilding
                                │                                models.networks B+.6 head
                                └──HTTP fallback (generate only)─> Stage A FastAPI service
```

The bridge runs our torch stack **inside Blender's Python** (the realtime local-edit loop).
That's why Blender, not Unreal, is the right prototype host: no service round-trip per edit.

## Install / run

Blender ships its own Python, which must be able to import torch + our repo. Two options:

**A. Point Blender's Python at the SDFusion venv (recommended on the dev box).**
Launch Blender from a shell with the env vars set, then enable the add-on:
```bash
export GENTOWNS_REPO=/scratch/gilbreth/dsimhadr/GenerativeTowns/SDFusion
export GENTOWNS_VENV_SITE=$GENTOWNS_REPO/sdfusion/lib/python3.9/site-packages
# (strip XALT on Gilbreth, as for any torch process)
env -u LD_PRELOAD -u LD_LIBRARY_PATH blender
```
The bridge injects `$GENTOWNS_REPO` and `$GENTOWNS_VENV_SITE` onto `sys.path`. This works
when Blender's bundled Python is binary-compatible with the venv wheels (matching minor
version, here 3.9–3.11; torch is fairly tolerant). The Generate panel shows
`engine: local (realtime)` when it succeeded.

**B. HTTP fallback (generate only).** If local import fails, start the Stage A service
(`... -m uvicorn scripts.server.inference_service:app --port 8077`) and set
`GENTOWNS_SERVICE_URL`. "Generate" then returns a glb (import via File > Import > glTF).
**Edits are not available over HTTP** (they must be local to be realtime).

### Enable the add-on
- Zip `generative_towns/` (or symlink it into Blender's `addons/`), then
  Edit > Preferences > Add-ons > Install… > pick the folder/zip > enable
  "GenerativeTowns — SDF Sculpt".
- The "GenTowns" tab appears in the View3D sidebar (press `N`).

## Status

Scaffold. Validated: the compute bridge (`tools/blender_addon/test_bridge.py`) generates +
edits at ~10 ms outside Blender. Not yet validated inside Blender (no Blender on the HPC
node). TODO: OSM-tile import operator (drive `scene/extract_osm.py` → many buildings),
drag-to-place via modal raycast, materials, and the trained `/refine_with_edit` model.
