"""Validate the add-on compute bridge OUTSIDE Blender (the bpy-free layer).

Run with the sdfusion venv python (which has all deps), simulating what Blender's Python
does once the venv site-packages are on sys.path.

  env -u LD_PRELOAD -u LD_LIBRARY_PATH PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
    ./sdfusion/bin/python tools/blender_addon/test_bridge.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "generative_towns"))
import bridge

print("local_available:", bridge.local_available())
print("palette:", bridge.palette())

fp = [[-8, -10], [8, -10], [8, 10], [-8, 10]]  # 16x20 m
t = time.time()
g = bridge.generate(fp, style="modern", building_class="COMMERCIAL", height=13.0, seed=2)
print(f"generate: backend={g['backend']} verts={len(g['verts'])} faces={len(g['faces'])} "
      f"params={len(g['recipe_params'])} | {1000*(time.time()-t):.0f}ms")

edits = [
    {"kind": "box", "center": [4, 16, 4], "size": [2.5, 3.5, 2.5], "mode": "add",
     "smooth": 0.6, "rot_y": 0.0, "round_r": 0.0},
    {"kind": "cylinder", "center": [-8, 8, 10], "size": [2.0, 18.0], "mode": "add",
     "smooth": 0.0, "rot_y": 0.0, "round_r": 0.0},
]
for n in range(1, len(edits) + 1):
    t = time.time()
    e = bridge.apply_edits("modern", g["recipe_params"], fp, 13.0, edits[:n], res=48)
    print(f"apply_edits[{n}]: verts={len(e['verts'])} faces={len(e['faces'])} "
          f"| {1000*(time.time()-t):.0f}ms")
print("BRIDGE OK")
