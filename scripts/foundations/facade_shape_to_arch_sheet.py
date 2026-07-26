"""SHAPE -> ARCHITECTURE (+ grid-snap) — the user's actual idea, on a mesh.

Drop a primitive; its SHAPE decides what it BECOMES, then it snaps to the nearest exterior cell
and is GENERATED as that element (not just a box):
  flat horizontal slab  -> BALCONY (slab + railing + door)
  thin vertical plane   -> WINDOW  (recess)
  tall box @ ground     -> DOOR    (opening + canopy)
  chunky box            -> BAY     (protruding box + windows)
  tall slender          -> PILASTER (vertical strip)

  env -u LD_PRELOAD -u LD_LIBRARY_PATH PYTHONPATH=. \
    /tmp/sdfusion_venv/bin/python scripts/foundations/facade_shape_to_arch_sheet.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts" / "server"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from skimage.measure import marching_cubes

import facade_grammar as fg

OUT = REPO / "outputs/part_set_refiner"
GP = dict(floor_pitch=0.5, bay_spacing=0.5, win_h=0.15, win_w=0.13, margin=0.2, depth=0.09)
RES = 132
_g = np.linspace(-1, 1, RES, dtype=np.float32)
GX, GY, GZ = np.meshgrid(_g, _g, _g, indexing="ij")
_E = [(0,1),(0,2),(1,3),(2,3),(4,5),(4,6),(5,7),(6,7),(0,4),(1,5),(2,6),(3,7)]


def box(lo, hi):
    c, h = (np.asarray(lo)+np.asarray(hi))/2, (np.asarray(hi)-np.asarray(lo))/2 + 1e-4
    q = np.stack([np.abs(GX-c[0])-h[0], np.abs(GY-c[1])-h[1], np.abs(GZ-c[2])-h[2]], 0)
    return (np.sqrt((np.clip(q,0,None)**2).sum(0)) + np.minimum(q.max(0),0)).astype(np.float32)


def compose(block, recesses, blobs):
    v = box(*block)
    for c, h in recesses: v = np.maximum(v, -box(np.asarray(c)-h, np.asarray(c)+h))
    for c, h in blobs:    v = np.minimum(v, box(np.asarray(c)-h, np.asarray(c)+h))
    return v


def render(ax, vol, title):
    v, f, *_ = marching_cubes(vol, level=0.0)
    tris = v[f][:, :, [0, 2, 1]]
    fy = tris[:, :, 2].mean(1)
    ax.add_collection3d(Poly3DCollection(tris, facecolors=plt.cm.bone(0.32+0.5*(fy-fy.min())/(np.ptp(fy)+1e-9)),
                                         edgecolors="k", linewidths=0.02))
    for s in (ax.set_xlim, ax.set_ylim): s(0, RES)
    ax.set_zlim(0, RES); ax.set_box_aspect((1,1,1)); ax.view_init(elev=9, azim=-38)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.set_title(title, fontsize=9)


def mark(ax, center, half, color, lw=2.6):
    c = (np.asarray(center)+1)/2*(RES-1); hv = np.asarray(half)*(RES-1)/2
    lo, hi = c-hv*1.3, c+hv*1.3
    p = [[lo[0],lo[1],lo[2]],[hi[0],lo[1],lo[2]],[lo[0],hi[1],lo[2]],[hi[0],hi[1],lo[2]],
         [lo[0],lo[1],hi[2]],[hi[0],lo[1],hi[2]],[lo[0],hi[1],hi[2]],[hi[0],hi[1],hi[2]]]
    for a, b in _E:
        ax.add_collection3d(Line3DCollection([np.array([p[a],p[b]])[:,[0,2,1]]], colors=color, linewidths=lw))


def construct(etype, cell, bbox):
    c, n, nax, lax = cell["center"].copy(), cell["normal"], cell["normal_axis"], cell["lateral_axis"]
    s = int(np.sign(n[nax])); lo, hi = np.asarray(bbox[0]), np.asarray(bbox[1])
    rec, blob = [], []
    def H(d, h, w):
        a = np.zeros(3); a[nax] = d; a[1] = h; a[lax] = w; return a
    if etype == "window":
        rec.append((c.copy(), H(0.1, 0.11, 0.09)))
    elif etype == "door":
        cc = c.copy(); cc[1] = lo[1]+0.17; rec.append((cc, H(0.1, 0.17, 0.09)))
        cn = cc.copy(); cn[1] = lo[1]+0.37; cn[nax] += s*0.08; blob.append((cn, H(0.09, 0.03, 0.14)))
    elif etype == "balcony":
        sl = c.copy(); sl[nax] += s*0.12; blob.append((sl, H(0.13, 0.045, 0.18)))      # slab
        rl = c.copy(); rl[nax] += s*0.24; rl[1] += 0.08; blob.append((rl, H(0.03, 0.085, 0.18)))  # front rail
        for d in (-1, 1):
            sr = c.copy(); sr[nax] += s*0.13; sr[1] += 0.07; sr[lax] += d*0.17
            blob.append((sr, H(0.13, 0.075, 0.03)))                                    # side rails
        rec.append((c.copy(), H(0.1, 0.12, 0.06)))                                     # door behind
    elif etype == "bay":
        bc = c.copy(); bc[nax] += s*0.13; blob.append((bc, H(0.13, 0.2, 0.15)))        # protruding box
        for dy in (-0.1, 0.1):
            wc = c.copy(); wc[nax] += s*0.26; wc[1] += dy; rec.append((wc, H(0.05, 0.06, 0.06)))
    elif etype == "pilaster":
        pc = c.copy(); pc[1] = (lo[1]+hi[1])/2; pc[nax] += s*0.04
        blob.append((pc, H(0.06, (hi[1]-lo[1])/2*0.9, 0.06)))
    return rec, blob


def main():
    block = ([-0.55, -0.95, -0.4], [0.55, 0.7, 0.4])
    base_grid = [(np.asarray(o["center"]), np.asarray(o["size"])) for o in fg.full_facade_ops(block, GP)]
    base = compose(block, base_grid, [])
    # all dropped near the +x (right) wall so they're camera-facing; shape varies -> type varies
    X = 0.78
    inputs = [
        ("flat horizontal", np.array([X, 0.1, 0.0], np.float32),  np.array([0.1, 0.03, 0.16], np.float32), "add"),
        ("thin plane",      np.array([X, 0.35, 0.0], np.float32), np.array([0.02, 0.12, 0.1], np.float32), "add"),
        ("tall @ ground",   np.array([X, -0.8, 0.0], np.float32), np.array([0.06, 0.2, 0.07], np.float32), "add"),
        ("chunky box",      np.array([X, -0.2, 0.0], np.float32), np.array([0.13, 0.15, 0.13], np.float32), "add"),
        ("tall slender",    np.array([X, 0.45, 0.22], np.float32), np.array([0.04, 0.3, 0.04], np.float32), "add"),
    ]
    fig = plt.figure(figsize=(3.0 * 2, 2.7 * len(inputs)))
    for r_, (lab, pc, ps, mode) in enumerate(inputs):
        etype = fg.classify_shape(ps, pc, (*block[0], *block[1]), mode)  # classify_shape wants flat (x0,y0,z0,x1,y1,z1)
        cell, _ = fg.snap_cell(block, pc, GP)
        rec, blob = construct(etype, cell, block)
        axi = fig.add_subplot(len(inputs), 2, 2*r_+1, projection="3d")
        render(axi, compose(block, base_grid, [(pc, ps)]), f"INPUT: {lab}"); mark(axi, pc, ps, "tab:orange")
        axo = fig.add_subplot(len(inputs), 2, 2*r_+2, projection="3d")
        render(axo, compose(block, base_grid + rec, blob), f"→ {etype.upper()}")
        mark(axo, cell["center"], np.full(3, 0.16), "red")
    fig.suptitle("Drop a SHAPE → it becomes ARCHITECTURE on the facade grid "
                 "(orange = dropped shape, red = generated element)", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "facade_shape_to_arch_sheet.png", dpi=125); plt.close(fig)
    print(f"[save] {OUT/'facade_shape_to_arch_sheet.png'}")


if __name__ == "__main__":
    main()
