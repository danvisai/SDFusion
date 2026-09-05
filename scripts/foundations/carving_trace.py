"""#147/#7: the per-step visual carving trace.

Given a program and its base, produces one frame per operation showing the delta that operation
introduced -- the same occupancy-delta concept #144's `_contribution` already computes -- at 4
fixed views, no per-mesh rescaling. Standard artifact for every finalized program (#145/#146), not
a debug tool: this is what makes it #7's own "final visual carving trace" and the visual input
#148's human-eval rubric is judged against.

⚠️ Renders on the CPU via a fixed-isometric painter's algorithm -- the same technique
`recover_massing_programs.render_iso` established for exactly this reason: `eval_massing_arms.py`'s
`render_world` goes through pyrender/EGL and hangs on this node while a GPU-heavy arm holds the
device, and it is more machinery than a height-map-representable corpus needs anyway (no marching
cubes, no mesh). This module reimplements the technique rather than importing `render_iso`
directly, so it never touches the recovery pipeline's own settled, heavily-measured code.

🔑 4 fixed views ("front/back/left/right oblique") are the SAME fixed isometric projection applied
to the height map rotated 0/90/180/270 degrees -- which corner of the building faces the camera
changes, the camera itself never moves. Canvas size is fixed from `res` alone, never from a
building's own content bounds, so no program or step is silently rescaled relative to any other --
`eval_massing_arms.py`'s own "an arm that lost volume renders visibly smaller instead of being
silently rescaled" principle, generalized as #7's standing appearance-hiding-failure rule.

⚠️ Highlighting is per COLUMN, not per voxel: a column is "this operation's contribution" if ANY
voxel in it toggled (`occ ^ prev_occ`, #144's own definition), and the whole column's rendered
faces take the highlight color. This is coarser than a sub-column split at the exact toggled
voxels would be, and is a deliberate simplification -- it still correctly shows every column this
operation touched, distinguishable from the unchanged columns around it, which is what the ticket
asks for, without the added complexity of splitting a single face into two colors.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np

from scene.sdf_edit import EditableBuilding, EditOp, SDF

CELL = 6                              # pixels per grid unit, fixed across every frame/program
PAD = 20
BASE_COLOR = (196, 198, 203)          # unchanged cumulative state (matches render_iso's default)
HIGHLIGHT_COLOR = (219, 118, 44)      # this operation's own contribution


def carving_steps(base_sdf: SDF, ops: Sequence[EditOp], res: int = 64,
                  device: str = "cpu") -> List[dict]:
    """One entry per operation: its cumulative per-column height and which columns it touched.

    'Touched' is decided at the same granularity `render_carving_trace` draws at -- a column, not
    a single voxel -- but the underlying test is the exact voxel-level delta #144's `_contribution`
    uses (`occ ^ prev_occ`), so it stays correct even off the height-map (a same-height column that
    gained a mid-column void still counts as touched).

    Does not validate: runs against whatever program it is given, per #147's own acceptance
    criterion -- `finalize_problems`/`containment_problems` are #145/#146's job, upstream of this.
    """
    steps: List[dict] = []
    prev_occ = EditableBuilding(base_sdf, []).to_occupancy(res=res, device=device)
    for i in range(len(ops)):
        occ = EditableBuilding(base_sdf, list(ops[:i + 1])).to_occupancy(res=res, device=device)
        changed = (occ ^ prev_occ).any(axis=1)
        steps.append(dict(index=i, height=occ.sum(axis=1).astype(np.int32), changed=changed))
        prev_occ = occ
    return steps


def _render_frame(height: np.ndarray, fp: np.ndarray, changed: np.ndarray, res: int,
                  rotation: int, cell: int = CELL):
    """One fixed-isometric frame. Canvas is sized from `res` alone -- never from `height`/`fp`'s
    own content -- so every frame of every program, at every step, shares one pixel frame.
    `rotation` (a multiple of 90) picks which corner of the building faces the fixed camera.
    """
    from PIL import Image, ImageDraw

    k = (rotation // 90) % 4
    H = np.rot90(height, k)
    FP = np.rot90(fp, k)
    CH = np.rot90(changed, k)

    cos30, sin30, hs = 0.866, 0.5, cell * 0.62
    sx = lambda x, z: (x - z) * cos30 * cell
    sy = lambda x, z, v: (x + z) * sin30 * cell - v * hs
    x0, x1 = sx(0, res), sx(res, 0)
    y0, y1 = sy(0, 0, res), sy(res, res, 0)
    W, Ht = int(x1 - x0) + 2 * PAD, int(y1 - y0) + 2 * PAD
    ox, oy = -x0 + PAD, -y0 + PAD
    img = Image.new("RGB", (W, Ht), (255, 255, 255))
    d = ImageDraw.Draw(img)

    gz, gx = np.gradient(H.astype(np.float64))
    lam = 1.0 / np.sqrt(gx ** 2 + gz ** 2 + 1.0)              # Lambert against a vertical light
    # round to integers so neighbouring columns share exact vertices -- without this the
    # side faces are separated by hairline background gaps and the massing looks combed
    P = lambda x, z, v: (round(sx(x, z) + ox), round(sy(x, z, v) + oy))
    shade = lambda color, f: tuple(int(np.clip(c * f, 0, 255)) for c in color)

    Z, X = H.shape
    order = sorted(((x + z, z, x) for z in range(Z) for x in range(X) if FP[z, x]))
    for _, z, x in order:
        v = int(H[z, x])
        if v <= 0:
            continue
        color = HIGHLIGHT_COLOR if CH[z, x] else BASE_COLOR
        nx = int(H[z, x + 1]) if x + 1 < X and FP[z, x + 1] else 0
        nz = int(H[z + 1, x]) if z + 1 < Z and FP[z + 1, x] else 0
        d.polygon([P(x, z, v), P(x + 1, z, v), P(x + 1, z + 1, v), P(x, z + 1, v)],
                  fill=shade(color, 0.62 + 0.55 * lam[z, x]))
        if v > nx:
            d.polygon([P(x + 1, z, v), P(x + 1, z + 1, v),
                       P(x + 1, z + 1, nx), P(x + 1, z, nx)], fill=shade(color, 0.74))
        if v > nz:
            d.polygon([P(x, z + 1, v), P(x + 1, z + 1, v),
                       P(x + 1, z + 1, nz), P(x, z + 1, nz)], fill=shade(color, 0.52))
    return img


VIEWS = (0, 90, 180, 270)             # front / right / back / left oblique


def render_carving_trace(base_sdf: SDF, ops: Sequence[EditOp], fp: np.ndarray, res: int = 64,
                         device: str = "cpu", cell: int = CELL) -> List[dict]:
    """#147: the carving trace itself -- one entry per operation, each carrying the 4 fixed-view
    frames for that step, highlighting the columns that operation's own application touched.
    """
    return [
        dict(index=step["index"], op=ops[step["index"]],
             views=[_render_frame(step["height"], fp, step["changed"], res, r, cell)
                    for r in VIEWS])
        for step in carving_steps(base_sdf, ops, res=res, device=device)
    ]


def save_carving_trace(trace: List[dict], out_dir: Path) -> List[Path]:
    """Write every frame of `render_carving_trace`'s output to `out_dir`, named
    `step{index}_view{degrees}.png`. Returns the paths written, in the same order as `trace`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for entry in trace:
        for degrees, img in zip(VIEWS, entry["views"]):
            p = out_dir / f"step{entry['index']}_view{degrees}.png"
            img.save(p)
            paths.append(p)
    return paths
