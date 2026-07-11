"""Ticket 06: test the fixed detail-scale coincidence.

`s*` is fixed a priori at 5 voxels @96^3 (ADR 0004) BEFORE this measurement runs. This script
does not choose `s*` -- it tests whether the fixed semantic-detail vocabulary falls below it
while the fixed massing vocabulary falls above, and reports the result honestly (pass, partial
coincidence, or failure) without moving the boundary to fit the outcome (ADR 0002 D3).

Vocabulary (fixed a priori, drawn from what the project has ALREADY committed to elsewhere --
not invented for this ticket, so it cannot have been tuned to this result):
  massing = wall(1), roof(4)
      CONTEXT.md "Massing": "base mass, wings, overall roof form".
  detail  = window(2), door(6), tower(7), column(12), balcony(14), chimney(15),
            balcony_upper(16), stairs(17), dome(22)
      CONTEXT.md "Detail": "windows, doors, balconies, cornices, ornament, facade
      articulation", extended with the ADD-element vocabulary `build_element_library.py`
      (ticket 04) already adopted and shipped (tower/dome/chimney/balcony/balcony_upper/
      column/stairs) -- CONTEXT.md's "Composition" entry describes retrieval from exactly
      this element set as a detail operation. Window/door are added even though ticket 04's
      library excludes them: its own docstring says so only because they are realized by
      procedural carving, not retrieval -- CONTEXT.md still calls them detail.
  excluded: undetermined(0, a noise bucket, not a semantic category), ground(9)/floor(23)
      (site/terrain, not building massing or detail -- outside CONTEXT.md's building-focused
      definitions), the "roof_structure" reuse of label 4 that ticket 04 carves out via a
      component-position heuristic (above the roofline, smaller than the main roof) -- that
      is an engineering heuristic for what to RETRIEVE, not a distinct semantic label, so
      giving label 4 to massing here and leaving it there is the honest single assignment.
      Every other label id is "uncertain" in outputs/part_labels_full/label_names.json and
      was never adopted anywhere else in the project -- including a guess here, to help or
      hurt the result, would violate "fixed a priori".

Scale measurement: per building, per adopted label, faces are grouped into geometric
INSTANCES via the same bbox-touch union-find `build_element_library.py` uses (so a label with
several disjoint occurrences -- a dozen windows -- is measured as a dozen small instances, not
one facade-spanning box). Each instance's characteristic scale is the MEDIAN of its 3 bbox side
lengths (robust to a near-zero carve-depth axis on one side and a long run on another -- either
would make min/max misleading), normalized by the building's OWN max AABB extent. BuildingNet
meshes carry no absolute real-world units (every model's raw OBJ is already normalized to
max-extent 1.0 -- verified empirically, not assumed), so `s*` is compared on the same
resolution-tied, per-building-relative basis ADR 0004 derives it from: `s*/bmax = voxels/(res-1)`
(5/95 @96^3), not an assumed meters-per-building conversion.

Out: execution/artifacts/scale_spectrum.json (per-label distributions + provenance),
     outputs/scale_spectrum/scale_spectrum.png (the coincidence figure).
Run:  ./sdfusion/bin/python scripts/eval/measure_scale_spectrum.py \
        --split data/splits_v1/train_100.json [--limit N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "foundations"))
sys.path.insert(0, str(REPO / "scripts" / "eval"))
import build_element_library as bel  # noqa: E402
from render_facades import WORKING_RES  # noqa: E402  ADR 0004 locked shared resolution

OBJ = bel.OBJ
CLBL = bel.CLBL
FIDX = bel.FIDX

# Fixed a priori -- see module docstring for provenance of every entry.
MASSING_LABELS = {1: "wall", 4: "roof"}
DETAIL_LABELS = {2: "window", 6: "door", 7: "tower", 12: "column", 14: "balcony",
                  15: "chimney", 16: "balcony_upper", 17: "stairs", 22: "dome"}
VOCAB = {**MASSING_LABELS, **DETAIL_LABELS}
S_STAR_VOXELS = 5  # ADR 0004: 5 voxels @96^3


def s_star_normalized(res: int = WORKING_RES, voxels: int = S_STAR_VOXELS) -> float:
    """`s*` as a fraction of a building's own max AABB extent -- resolution-tied (ADR 0004's
    `voxel = 2*scale/(res-1)`, scale = half-max-extent), not an assumed meters-per-building
    conversion BuildingNet meshes don't carry."""
    return voxels / (res - 1)


def instance_char_scale(ext: np.ndarray) -> float:
    """Characteristic scale of one instance: the median of its 3 bbox side lengths."""
    return float(np.median(np.sort(np.abs(np.asarray(ext, dtype=np.float64)))))


def classify(scale_normalized: float, threshold: float) -> str:
    return "above_s*" if scale_normalized >= threshold else "below_s*"


def extract_instances_for_building(name, obj_dir=OBJ, clbl_dir=CLBL, fidx_dir=FIDX):
    """Yield (label_id, char_scale_normalized) for every adopted-vocabulary instance in one
    building. Reuses ticket 04's exact parse/instance-merge machinery so results are directly
    comparable to the element library it built."""
    obj_p = Path(obj_dir) / f"{name}.obj"
    fidx_p = Path(fidx_dir) / f"{name}.json"
    clbl_p = Path(clbl_dir) / f"{name}_label.json"
    if not (obj_p.exists() and fidx_p.exists() and clbl_p.exists()):
        return
    verts, faces = bel.parse_obj(obj_p)
    if len(faces) < 1:
        return
    comp_of_face = bel.face_components(fidx_p, len(faces))
    comp_label = {int(k): int(v) for k, v in json.load(open(clbl_p)).items()}
    blo, bhi = verts.min(0), verts.max(0)
    bmax = max((bhi - blo).max(), 1e-6)

    fset = defaultdict(list)
    for fi, ci in enumerate(comp_of_face):
        if ci >= 0 and comp_label.get(int(ci), -1) in VOCAB:
            fset[int(ci)].append(fi)
    by_label = defaultdict(list)
    for ci, fl in fset.items():
        by_label[comp_label[ci]].append((ci, np.asarray(fl)))

    for lab, comps in by_label.items():
        boxes = []
        for _ci, fl in comps:
            vv = verts[faces[fl].reshape(-1)]
            boxes.append(np.concatenate([vv.min(0), vv.max(0)]))
        for g in bel.merge_instances(boxes, touch=0.02 * bmax):
            fl = np.concatenate([comps[i][1] for i in g])
            vv = verts[faces[fl].reshape(-1)]
            lo, hi = vv.min(0), vv.max(0)
            yield lab, instance_char_scale(hi - lo) / bmax


def aggregate(scales) -> dict:
    a = np.asarray(scales, dtype=np.float64)
    q = np.quantile(a, [0.25, 0.5, 0.75])
    return dict(n=len(a), median=float(q[1]), q25=float(q[0]), q75=float(q[2]),
                mean=float(a.mean()), min=float(a.min()), max=float(a.max()))


def _git_provenance():
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:  # noqa: BLE001
        return dict(git_rev=None, dirty_digest=None)
    try:
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    except Exception:  # noqa: BLE001
        status = ""
    digest = hashlib.sha1(status.encode()).hexdigest()[:12] if status.strip() else None
    return dict(git_rev=rev, dirty_digest=digest)


def _plot(per_label_scales, thr, fig_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = list(MASSING_LABELS) + list(DETAIL_LABELS)
    names = [VOCAB[lab] for lab in order]
    data = [per_label_scales.get(lab, []) for lab in order]
    colors = ["#3b6fb3" if lab in MASSING_LABELS else "#c9752f" for lab in order]

    fig, ax = plt.subplots(figsize=(12, 6))
    positions = np.arange(1, len(order) + 1)
    plot_data = [d if d else [np.nan] for d in data]
    bp = ax.boxplot(plot_data, positions=positions, showfliers=False, patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.axhline(thr, color="red", linestyle="--", linewidth=1.5,
               label=f"s* = {thr:.4f} (5 vox @{WORKING_RES}^3)")
    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{n}\n(n={len(d)})" for n, d in zip(names, data)], rotation=30, ha="right")
    ax.set_ylabel("characteristic scale / building max-extent (log)")
    ax.set_title("Massing/detail scale coincidence vs the fixed a priori s* (ticket 06)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default=str(REPO / "data/splits_v1/train_100.json"),
                    help="population of building ids (default: the non-test train_100 split, "
                         "so this measurement never touches the sealed test set)")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N buildings")
    ap.add_argument("--out", default=str(REPO / "execution/artifacts/scale_spectrum.json"))
    ap.add_argument("--fig-out", default=str(REPO / "outputs/scale_spectrum/scale_spectrum.png"))
    ap.add_argument("--no-fig", action="store_true")
    a = ap.parse_args()

    ids = sorted(bel.load_id_list(a.split))
    if a.limit:
        ids = ids[: a.limit]

    per_label_scales = defaultdict(list)
    per_label_buildings = defaultdict(set)
    n_ok = 0
    for bi, name in enumerate(ids):
        try:
            for lab, sc in extract_instances_for_building(name):
                per_label_scales[lab].append(sc)
                per_label_buildings[lab].add(name)
            n_ok += 1
        except Exception as ex:  # noqa: BLE001
            print(f"[skip] {name}: {type(ex).__name__}: {str(ex)[:80]}")
        if (bi + 1) % 200 == 0:
            print(f"[{bi + 1}/{len(ids)}]", flush=True)

    thr = s_star_normalized()
    results = {}
    n_pass = 0
    n_tested = 0
    for lab, name in sorted(VOCAB.items()):
        role = "massing" if lab in MASSING_LABELS else "detail"
        expect = "above_s*" if role == "massing" else "below_s*"
        scales = per_label_scales.get(lab, [])
        if not scales:
            results[name] = dict(label_id=lab, role=role, expected=expect, n=0, verdict="no_data")
            print(f"  {name:16s} [no_data]")
            continue
        agg = aggregate(scales)
        observed = classify(agg["median"], thr)
        verdict = "pass" if observed == expect else "fail"
        n_tested += 1
        n_pass += verdict == "pass"
        results[name] = dict(label_id=lab, role=role, expected=expect, observed=observed,
                              verdict=verdict, n_buildings=len(per_label_buildings[lab]), **agg)
        print(f"  {name:16s} n={agg['n']:5d} median={agg['median']:.4f} (s*={thr:.4f}) "
              f"expected={expect:9s} got={observed:9s} [{verdict}]")

    if n_tested == 0:
        overall = "no_data"
    elif n_pass == n_tested:
        overall = "pass"
    elif n_pass == 0:
        overall = "fail"
    else:
        overall = "partial_coincidence"

    manifest = dict(
        s_star_voxels=S_STAR_VOXELS, working_res=WORKING_RES, s_star_normalized=thr,
        split=a.split, n_buildings_population=len(ids), n_buildings_parsed=n_ok,
        vocabulary=dict(massing=MASSING_LABELS, detail=DETAIL_LABELS),
        per_label=results, n_pass=n_pass, n_tested=n_tested, overall=overall,
        **_git_provenance(),
    )
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(manifest, open(out_path, "w"), indent=2)
    print(f"\n[done] {n_pass}/{n_tested} categories on their expected side of s* -> {overall}")
    print(f"[save] {out_path}")

    if not a.no_fig:
        _plot(per_label_scales, thr, Path(a.fig_out))
        print(f"[save] {a.fig_out}")


if __name__ == "__main__":
    main()
