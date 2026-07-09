"""Phase R1 of GENERATIVE_MAKE_IT_ARCHITECTURE_BUILD_SPEC_2026-07-08: extract the element
library from BuildingNet's component-labeled meshes.

For each of the 1,849 labeled buildings: parse the raw OBJ with a minimal fan-triangulating
parser (face order must match model_data/obj/faceindex_componentID, which was produced from
the raw face sequence — trimesh would resplit/reorder), group faces per component, merge
same-label components whose bboxes touch (a tower = shaft+cap+finial), and for each adopted
ADD-type instance emit a normalized 48^3 SDF crop + metadata.

Windows/doors are deliberately excluded: carves stay procedural in Phase R; the library's
value is the ADD vocabulary interpret_mass can't build today (real towers/domes/chimneys/
rooftop structures/balconies/stairs/columns).

Out: data/element_library_v1/{elements_f16.npy (N,48,48,48 float16 SDF), meta.json}
     outputs/element_library_v1/montage_<type>.png  (QA)
Run:  ./sdfusion/bin/python scripts/foundations/build_element_library.py [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
BN = REPO / "data/BuildingNet_dataset_v0_1"
OBJ = BN / "OBJ_MODELS"
CLBL = BN / "model_data/obj/component_labels"
FIDX = BN / "model_data/obj/faceindex_componentID"
OUT_DATA = REPO / "data/element_library_v1"
OUT_QA = REPO / "outputs/element_library_v1"

# adopted ADD vocabulary (label ids per outputs/part_labels_full/label_names.json)
TYPES = {7: "tower", 22: "dome", 15: "chimney", 4: "roof_structure", 14: "balcony",
         16: "balcony_upper", 12: "column", 17: "stairs"}
RES = 48
MIN_FACES = 60
MAX_PER_TYPE_PER_BLDG = 4
MAX_PER_TYPE = 3000


def parse_obj(path):
    """Minimal OBJ parser preserving face order, fan-triangulated in place (the
    convention faceindex_componentID was generated against)."""
    vs, fs = [], []
    with open(path, errors="ignore") as f:
        for ln in f:
            if ln.startswith("v "):
                p = ln.split()
                vs.append((float(p[1]), float(p[2]), float(p[3])))
            elif ln.startswith("f "):
                idx = [int(t.split("/")[0]) for t in ln.split()[1:]]
                idx = [i - 1 if i > 0 else len(vs) + i for i in idx]
                for k in range(1, len(idx) - 1):        # fan triangulation, in order
                    fs.append((idx[0], idx[k], idx[k + 1]))
    return np.asarray(vs, np.float32), np.asarray(fs, np.int64)


def face_components(fidx_path, n_faces):
    """faceindex json {start: {end: comp}} -> per-face component id array."""
    comp = np.full(n_faces, -1, np.int64)
    for a, sub in json.load(open(fidx_path)).items():
        for b, cid in sub.items():
            lo, hi = int(a), min(int(b), n_faces - 1)
            if lo <= hi:
                comp[lo:hi + 1] = int(cid)
    return comp


def merge_instances(boxes, touch):
    """union-find over component bboxes (N,6): merge if expanded boxes intersect."""
    n = len(boxes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            a, b = boxes[i], boxes[j]
            if (a[0] - touch <= b[3] and b[0] - touch <= a[3]
                    and a[1] - touch <= b[4] and b[1] - touch <= a[4]
                    and a[2] - touch <= b[5] and b[2] - touch <= a[5]):
                pa, pb = find(i), find(j)
                if pa != pb:
                    parent[pa] = pb
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


def sdf_crop(verts, faces, res=RES, n_samp=180_000):
    """Sub-mesh -> normalized [-1,1] 48^3 SDF (surface splat + border flood fill sign;
    watertightness NOT assumed — thin parts become ~1-voxel shells, which is fine for
    trilinear compositing)."""
    from scipy.ndimage import binary_dilation, distance_transform_edt, label
    lo, hi = verts.min(0), verts.max(0)
    c, s = (lo + hi) / 2, max((hi - lo).max() / 2, 1e-6)
    v = (verts - c) / s                                        # [-1,1], aspect preserved
    tri = v[faces]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    p = area / max(area.sum(), 1e-12)
    pick = np.random.default_rng(0).choice(len(faces), n_samp, p=p)
    r1, r2 = np.random.default_rng(1).random((2, n_samp, 1), dtype=np.float32)
    sq = np.sqrt(r1)
    pts = (1 - sq) * tri[pick, 0] + sq * (1 - r2) * tri[pick, 1] + sq * r2 * tri[pick, 2]
    ijk = np.clip(((pts + 1) * 0.5 * (res - 1)).round().astype(int), 0, res - 1)
    occ = np.zeros((res, res, res), bool)
    occ[ijk[:, 2], ijk[:, 1], ijk[:, 0]] = True                # (z,y,x) layout
    occ = binary_dilation(occ)
    # outside = empty region connected to the border; enclosed empty space counts as inside
    empty_lab, _ = label(~occ)
    border = np.unique(np.concatenate([
        empty_lab[0].ravel(), empty_lab[-1].ravel(), empty_lab[:, 0].ravel(),
        empty_lab[:, -1].ravel(), empty_lab[:, :, 0].ravel(), empty_lab[:, :, -1].ravel()]))
    outside = np.isin(empty_lab, border[border > 0])
    inside = ~outside
    vox = 2.0 / (res - 1)
    sdf = (distance_transform_edt(outside) - distance_transform_edt(inside)) * vox
    return sdf.astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N buildings")
    a = ap.parse_args()
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_QA.mkdir(parents=True, exist_ok=True)

    names = sorted(p.stem.replace("_label", "") for p in CLBL.glob("*_label.json"))
    if a.limit:
        names = names[:a.limit]
    crops, meta = [], []
    per_type = defaultdict(int)
    for bi, name in enumerate(names):
        obj_p = OBJ / f"{name}.obj"
        fidx_p = FIDX / f"{name}.json"
        if not obj_p.exists() or not fidx_p.exists():
            continue
        try:
            verts, faces = parse_obj(obj_p)
            if len(faces) < 100:
                continue
            comp_of_face = face_components(fidx_p, len(faces))
            comp_label = {int(k): int(v) for k, v in
                          json.load(open(CLBL / f"{name}_label.json")).items()}
            blo, bhi = verts.min(0), verts.max(0)
            bh = max(bhi[1] - blo[1], 1e-6)
            bmax = max((bhi - blo).max(), 1e-6)
            cls = name.split("_")[0]                          # e.g. COMMERCIALcastle
            # faces per component, only adopted labels
            fset = defaultdict(list)
            for fi, ci in enumerate(comp_of_face):
                if ci >= 0 and comp_label.get(int(ci), -1) in TYPES:
                    fset[int(ci)].append(fi)
            by_label = defaultdict(list)
            for ci, fl in fset.items():
                by_label[comp_label[ci]].append((ci, np.asarray(fl)))
            for lab, comps in by_label.items():
                boxes = []
                for _ci, fl in comps:
                    vv = verts[faces[fl].reshape(-1)]
                    boxes.append(np.concatenate([vv.min(0), vv.max(0)]))
                inst_groups = merge_instances(boxes, touch=0.02 * bmax)
                n_kept = 0
                seen_sig = set()
                for g in inst_groups:
                    fl = np.concatenate([comps[i][1] for i in g])
                    if len(fl) < MIN_FACES:
                        continue
                    vv = verts[faces[fl].reshape(-1)]
                    lo, hi = vv.min(0), vv.max(0)
                    ext = hi - lo
                    if ext[1] < 0.015 * bh and max(ext) < 0.05 * bmax:
                        continue                              # speck
                    tname = TYPES[lab]
                    if tname == "roof_structure":
                        # only structures ABOVE the roofline, smaller than the roof itself
                        if lo[1] < blo[1] + 0.55 * bh or max(ext[0], ext[2]) > 0.6 * bmax:
                            continue
                    # dedupe repeated identical components (100 copies of one balcony)
                    sig = (tname, tuple((ext / bmax * 40).round().astype(int)),
                           int((lo[1] - blo[1]) / bh * 20))
                    if sig in seen_sig or n_kept >= MAX_PER_TYPE_PER_BLDG \
                            or per_type[tname] >= MAX_PER_TYPE:
                        continue
                    seen_sig.add(sig)
                    sub_f = faces[fl]
                    uniq, inv = np.unique(sub_f.reshape(-1), return_inverse=True)
                    crops.append(sdf_crop(verts[uniq], inv.reshape(-1, 3)))
                    meta.append(dict(
                        type=tname, building=name, cls=cls,
                        ext_rel=[round(float(e / bh), 4) for e in ext],
                        y_frac=round(float(((lo[1] + hi[1]) / 2 - blo[1]) / bh), 4),
                        aspect=[round(float(ext[0] / max(ext[1], 1e-6)), 3),
                                round(float(ext[2] / max(ext[1], 1e-6)), 3)],
                        n_faces=int(len(fl))))
                    per_type[tname] += 1
                    n_kept += 1
        except Exception as ex:
            print(f"[skip] {name}: {type(ex).__name__}: {str(ex)[:80]}")
        if (bi + 1) % 100 == 0:
            print(f"[{bi + 1}/{len(names)}] elements so far: {dict(per_type)}", flush=True)

    if not crops:
        print("no elements extracted!")
        sys.exit(1)
    np.save(OUT_DATA / "elements_f16.npy", np.stack(crops))
    json.dump(meta, open(OUT_DATA / "meta.json", "w"))
    print(f"[done] {len(crops)} elements -> {OUT_DATA}  by type: {dict(per_type)}")

    # QA montages: marching-cubes renders of random crops per type
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage import measure
    rng = np.random.default_rng(3)
    types = sorted({m["type"] for m in meta})
    for t in types:
        idx = [i for i, m in enumerate(meta) if m["type"] == t]
        pickn = rng.choice(idx, min(24, len(idx)), replace=False)
        fig, axs = plt.subplots(4, 6, figsize=(15, 10), subplot_kw={"projection": "3d"})
        for ax, i in zip(axs.ravel(), pickn):
            g = crops[i].astype(np.float32)
            ax.set_axis_off()
            if (g <= 0).sum() > 8:
                try:
                    v, f, *_ = measure.marching_cubes(g, 0.0)
                    ax.plot_trisurf(v[:, 2], v[:, 0], f, v[:, 1], color="#c9b790",
                                    edgecolor="none", shade=True)
                    lim = [0, RES]
                    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_zlim(lim)
                except Exception:
                    pass
            ax.set_title(f"{meta[i]['building'][:18]}\ny={meta[i]['y_frac']:.2f}", fontsize=5)
            ax.view_init(elev=14, azim=-60)
        fig.suptitle(f"element library v1 — {t} ({len(idx)} instances)")
        fig.tight_layout()
        fig.savefig(OUT_QA / f"montage_{t}.png", dpi=95)
        plt.close(fig)
    print(f"[QA] montages -> {OUT_QA}")


if __name__ == "__main__":
    main()
