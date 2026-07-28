"""Edit-pair generator (the key training ingredient for the coherent-add-primitive refiner).

Teaches "user adds a moldy/misplaced piece -> coherent element". From a CLEAN real part layout
(LoD3 CityGML or BuildingNet instances) synthesize `(x_corrupt, marker) -> x_clean` pairs:
  MOVE   displace a part off its row/wall          -> target = original pose   (snap back)
  RESIZE rescale a part off neighbor scale         -> target = original size   (match scale)
  DUP    duplicate a part (redundant)              -> target validity 0        (drop dupe)
  ADD    a coherent NEW part, corrupted into a     -> target = coherent pose   (moldy -> piece)
         crude off-position 'moldy' blob
`marker` (1 bit/slot) flags the user's edited piece — the conditioning that says "integrate THIS".
Encoding matches models/networks/part_set_refiner (SLOTS x [type-onehot N | box6 | validity], ±1).

  env -u LD_PRELOAD -u LD_LIBRARY_PATH PYTHONPATH=. \
    /tmp/sdfusion_venv/bin/python scripts/foundations/make_part_edit_pairs.py \
       [--source lod3|buildingnet] [--per 6]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from models.networks.part_layout_planner import RAW_TYPES, TYPE_NAMES, N_TYPES

SLOTS = 40
PART_DIM = N_TYPES + 6 + 1
RAW2IDX = {r: i for i, r in enumerate(RAW_TYPES)}
SOURCES = {"lod3": REPO / "data/lod3_tum/lod3_part_instances.npz",
           "buildingnet": REPO / "outputs/part_layouts_full/part_instances.npz"}
OUT = REPO / "outputs/part_edit_pairs"


def per_building(npz):
    d = np.load(npz, allow_pickle=True)
    r = d["rows"]
    out = {}
    for row in r:
        b = int(row[0]); t = int(row[1])
        if t not in RAW2IDX:
            continue
        out.setdefault(b, []).append((RAW2IDX[t], row[2:8].astype(np.float32)))
    return [v for v in out.values() if len(v) >= 4]


def encode(parts):
    """parts: list of (type_idx, box6) -> (SLOTS, PART_DIM) ±1 set."""
    x = np.full((SLOTS, PART_DIM), 0.0, np.float32)
    x[:, :N_TYPES] = -1.0; x[:, -1] = -1.0
    for i, (t, b) in enumerate(parts[:SLOTS]):
        x[i, :N_TYPES] = -1.0; x[i, t] = 1.0
        x[i, N_TYPES:N_TYPES + 6] = b
        x[i, -1] = 1.0
    return x


def make_pair(parts, rng):
    """Return (x_corrupt, x_clean, marker, op). Both encode the SAME slot order so a slot's
    target is well-defined."""
    parts = [(t, b.copy()) for t, b in parts][:SLOTS - 1]
    L = len(parts)
    op = rng.choice(["move", "resize", "dup", "add"], p=[0.4, 0.2, 0.2, 0.2])
    clean = [(t, b.copy()) for t, b in parts]
    corrupt = [(t, b.copy()) for t, b in parts]
    marker = np.zeros(SLOTS, np.float32)

    if op == "move":
        i = rng.integers(0, L)
        corrupt[i][1][:3] += rng.uniform(-0.4, 0.4, 3).astype(np.float32)
        marker[i] = 1.0
    elif op == "resize":
        i = rng.integers(0, L)
        corrupt[i][1][3:6] = np.clip(corrupt[i][1][3:6] * rng.uniform(0.35, 2.6), 1e-3, 1.0)
        marker[i] = 1.0
    elif op == "dup":
        i = rng.integers(0, L); src = clean[i]
        dup = (src[0], src[1].copy()); dup[1][:3] += rng.uniform(-0.08, 0.08, 3).astype(np.float32)
        clean.append((src[0], np.concatenate([src[1][:3], np.zeros(3, np.float32)])))  # validity 0 below
        corrupt.append(dup); marker[len(corrupt) - 1] = 1.0
        # mark the appended CLEAN slot invalid (drop the dupe): encode handles via validity
        clean_x = encode(clean); clean_x[len(clean) - 1, -1] = -1.0
        return encode(corrupt), clean_x, marker, op
    else:  # add: a coherent NEW window on an existing row, corrupted into a moldy blob
        wins = [b for t, b in parts if t == 0]
        ref = (wins[rng.integers(0, len(wins))] if wins else parts[rng.integers(0, L)][1]).copy()
        new = ref.copy()
        new[0] += rng.uniform(-0.5, 0.5)                    # new lateral position on the SAME row
        clean.append((0, new))                              # the coherent target piece
        moldy = new.copy(); moldy[:3] += rng.uniform(-0.45, 0.45, 3).astype(np.float32)
        moldy[3:6] = np.clip(moldy[3:6] * rng.uniform(0.6, 2.2), 1e-3, 1.0)
        corrupt.append((0, moldy)); marker[len(corrupt) - 1] = 1.0
    return encode(corrupt), encode(clean), marker, op


def draw(ax, x, marker, title):
    for s in range(SLOTS):
        if x[s, -1] <= 0:
            continue
        c, e = x[s, N_TYPES:N_TYPES + 3], np.abs(x[s, N_TYPES + 3:N_TYPES + 6]) + 1e-3
        lo, hi = c - e, c + e
        col = "tab:red" if marker[s] > 0 else "tab:gray"
        pts = [[lo[0],lo[1],lo[2]],[hi[0],lo[1],lo[2]],[lo[0],hi[1],lo[2]],[hi[0],hi[1],lo[2]],
               [lo[0],lo[1],hi[2]],[hi[0],lo[1],hi[2]],[lo[0],hi[1],hi[2]],[hi[0],hi[1],hi[2]]]
        for a_, b_ in [(0,1),(0,2),(1,3),(2,3),(4,5),(4,6),(5,7),(6,7),(0,4),(1,5),(2,6),(3,7)]:
            seg = np.array([pts[a_], pts[b_]])[:, [0, 2, 1]]
            ax.add_collection3d(Line3DCollection([seg], colors=col,
                                                 linewidths=1.4 if marker[s] > 0 else 0.5))
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1); ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=15, azim=-60); ax.set_title(title, fontsize=6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="lod3", choices=list(SOURCES))
    ap.add_argument("--per", type=int, default=6)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    blds = per_building(SOURCES[a.source])
    Xc, X0, M, OPS = [], [], [], []
    for parts in blds:
        for _ in range(a.per):
            xc, x0, mk, op = make_pair(parts, rng)
            Xc.append(xc); X0.append(x0); M.append(mk); OPS.append(op)
    Xc, X0, M = np.asarray(Xc), np.asarray(X0), np.asarray(M)
    import collections
    print(f"[{a.source}] {len(blds)} buildings -> {len(Xc)} edit-pairs  ops={dict(collections.Counter(OPS))}")
    np.savez(OUT / f"edit_pairs_{a.source}.npz", x_corrupt=Xc, x_clean=X0, marker=M,
             op=np.array(OPS), slots=SLOTS, part_dim=PART_DIM, n_types=N_TYPES)

    # montage: 4 example pairs (corrupt[red=marked] -> clean target)
    ex = rng.choice(len(Xc), 4, replace=False)
    fig = plt.figure(figsize=(2.6 * 4, 5.4))
    for j, i in enumerate(ex):
        draw(fig.add_subplot(2, 4, j + 1, projection="3d"), Xc[i], M[i], f"INPUT moldy ({OPS[i]})")
        draw(fig.add_subplot(2, 4, 4 + j + 1, projection="3d"), X0[i], M[i] * 0, "TARGET coherent")
    fig.suptitle(f"Part edit-pairs ({a.source}): red = the user's edited piece (marker) → "
                 f"coherent target", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / f"edit_pairs_{a.source}.png", dpi=115); plt.close(fig)
    print(f"[save] {OUT/('edit_pairs_'+a.source+'.npz')} + .png")


if __name__ == "__main__":
    main()
