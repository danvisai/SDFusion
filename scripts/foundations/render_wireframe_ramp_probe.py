"""Visualize wireframe_ramp_probe.py's step 1-3 output: classified edges + recovered
roof planes, overlaid on the translucent solid mesh for context.

Color key:
  mesh       translucent gray/terrain-shaded (context only)
  wall edges thin black
  horizontal edges (eave/ridge/ground) thin blue
  sloped edges, clustered into a plane with pitch < 75deg (plausible ramp): warm color,
      one per plane, cycling a colormap
  sloped edges, clustered into a plane with pitch >= 75deg (near-vertical -- the known
      over-segmentation noise `wireframe_ramp_probe.py` already flags): dashed gray
  sloped edges NOT assigned to any plane (clustering leftover): dotted magenta

Run: env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
     scripts/foundations/render_wireframe_ramp_probe.py --city Adelaide --n 8 \
     --out outputs/wireframe_ramp_probe/adelaide_grid.png
"""
from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from matplotlib import cm  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wireframe_ramp_probe import (  # noqa: E402
    MESH_DIR,
    WIRE_DIR,
    PLANE_DIST_TOL_FRAC,
    classify_edges,
    cluster_planes,
    load_obj_verts_edges,
    match_wireframes_to_meshes,
)


def render_building(ax, mesh_bytes: bytes, wire_data: str, title: str) -> None:
    m = trimesh.load(io.BytesIO(mesh_bytes), file_type="obj", process=False)
    verts, faces = m.vertices, m.faces
    tris = verts[faces]
    z = tris[:, :, 2].mean(axis=1)
    zmin, zmax = z.min(), max(z.max(), z.min() + 1e-6)
    mesh_colors = cm.get_cmap("terrain")(0.15 + 0.7 * (z - zmin) / (zmax - zmin))
    mesh_colors[:, 3] = 0.25  # translucent -- context only, not the point of the figure
    ax.add_collection3d(Poly3DCollection(tris, facecolor=mesh_colors, edgecolor="none"))

    V, edges = load_obj_verts_edges(wire_data)
    classes = classify_edges(V, edges)
    ext = m.extents
    tol = float(ext.max()) * PLANE_DIST_TOL_FRAC
    planes = cluster_planes(V, classes["sloped"], tol, classes["horizontal"])

    def seglist(edge_list):
        return [(V[a], V[b]) for a, b in edge_list]

    ax.add_collection3d(Line3DCollection(seglist(classes["vertical"]), colors="k", linewidths=0.5))
    ax.add_collection3d(
        Line3DCollection(seglist(classes["horizontal"]), colors="tab:blue", linewidths=0.5)
    )

    clustered_edges = set()
    warm = cm.get_cmap("autumn")
    clean_planes = [p for p in planes if _pitch_ok(p)]
    for i, p in enumerate(planes):
        clustered_edges.update(p.edges)
        if _pitch_ok(p):
            color = warm(i / max(1, len(clean_planes) - 1)) if len(clean_planes) > 1 else warm(0.3)
            ax.add_collection3d(Line3DCollection(seglist(p.edges), colors=[color], linewidths=2.2))
        else:
            ax.add_collection3d(
                Line3DCollection(seglist(p.edges), colors="0.5", linewidths=1.2, linestyles="dashed")
            )

    leftover = [e for e in classes["sloped"] if e not in clustered_edges]
    if leftover:
        ax.add_collection3d(
            Line3DCollection(seglist(leftover), colors="magenta", linewidths=1.0, linestyles="dotted")
        )

    # Bug fix: axis limits must cover BOTH the mesh and the wireframe, not the wireframe
    # alone -- the wireframe's bbox is consistently narrower than the mesh's own (it's a
    # partial/simplified skeleton, confirmed on 10a_0: wire x-range 36 vs mesh x-range 56),
    # so using wireframe-only limits together with mesh-derived `ext` for box_aspect put the
    # two on inconsistent scales and silently pushed the translucent mesh out of view for
    # every building where that gap was large enough (6 of 8 in the first render).
    all_pts = np.concatenate([verts, V], axis=0)
    xs, ys, zs = all_pts[:, 0], all_pts[:, 1], all_pts[:, 2]
    ax.set_xlim(xs.min(), xs.max())
    ax.set_ylim(ys.min(), ys.max())
    ax.set_zlim(zs.min(), zs.max())
    combined_ext = all_pts.max(0) - all_pts.min(0)
    ax.set_box_aspect((combined_ext[0] or 1, combined_ext[1] or 1, max(combined_ext[2], 0.1) * 1.5))
    n_clean = sum(1 for p in planes if _pitch_ok(p))
    ax.set_title(f"{title}\n{n_clean} plausible ramp / {len(planes) - n_clean} near-vert / "
                 f"{len(leftover)} unclustered", fontsize=7)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])


def _pitch_ok(plane) -> bool:
    """Plausible-ramp filter, mirrored from wireframe_ramp_probe.py's report split
    (pitch < 75deg): recomputed here from the plane's own normal to avoid re-importing
    the Frame-N conversion just for a threshold check."""
    import math

    n = plane.normal / (np.linalg.norm(plane.normal) + 1e-12)
    nz = abs(n[2])  # raw (pre-Frame-N) coordinates: z is still "up" here
    pitch = math.degrees(math.acos(min(1.0, max(0.0, nz))))
    return pitch < 75.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="Adelaide")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--out", default="outputs/wireframe_ramp_probe/grid.png")
    ap.add_argument("--cols", type=int, default=4)
    args = ap.parse_args()

    mesh_zip_path = MESH_DIR / args.city / "mesh" / "obj.zip"
    if not mesh_zip_path.exists():
        mesh_zip_path = MESH_DIR / args.city / "obj" / "mesh.zip"
    wire_zip_path = WIRE_DIR / args.city / "wireframe" / "wireframe.zip"
    zm = zipfile.ZipFile(mesh_zip_path)
    zw = zipfile.ZipFile(wire_zip_path)

    def prefix_of(name: str) -> str:
        return Path(name).stem.rsplit("_", 1)[0]

    mesh_by_prefix: dict[str, list[str]] = {}
    for n in zm.namelist():
        if not n.endswith(".obj"):
            continue
        if args.prefix and not Path(n).stem.startswith(args.prefix):
            continue
        mesh_by_prefix.setdefault(prefix_of(n), []).append(n)

    wire_by_prefix: dict[str, list[str]] = {}
    for n in zw.namelist():
        if not n.endswith(".obj"):
            continue
        wire_by_prefix.setdefault(prefix_of(n), []).append(n)

    # Geometric matching (nearest centroid within the same prefix group), NOT filename-index
    # pairing -- see wireframe_ramp_probe.py's module docstring for why index pairing is wrong.
    picked = []
    for prefix in sorted(wire_by_prefix):
        if len(picked) >= args.n:
            break
        if prefix not in mesh_by_prefix:
            continue
        pairing = match_wireframes_to_meshes(zm, zw, mesh_by_prefix[prefix], wire_by_prefix[prefix])
        for wname, mname in pairing.items():
            if len(picked) >= args.n:
                break
            picked.append((mname, wname, Path(mname).stem))

    cols = args.cols
    rows = (len(picked) + cols - 1) // cols
    fig = plt.figure(figsize=(4 * cols, 4 * rows))
    for i, (mname, wname, stem) in enumerate(picked):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        try:
            render_building(ax, zm.read(mname), zw.read(wname).decode(), f"{args.city}/{stem}")
        except Exception as e:
            ax.set_title(f"{args.city}/{stem}\nFAILED: {type(e).__name__}: {e}", fontsize=7)
        print(f"[rendered] {stem} ({i + 1}/{len(picked)})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[wrote] {out_path}")


if __name__ == "__main__":
    main()
