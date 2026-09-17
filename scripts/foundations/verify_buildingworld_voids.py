"""Reproducible BuildingWorld void/passage verification for the whole-volume voxel effort.

The research report at ``data/sdfusion-voxel-research/report.md`` found a coarse signal: many
BuildingWorld occupancy columns contain empty voxels below their topmost solid voxel.  That test
cannot distinguish an exterior-reachable architectural opening from a sealed defect cavity.  This
script closes that gap with the project's already-tested ``hollow_shell_voxels`` contract.

The production corpus and raw BuildingWorld archives are intentionally inputs, never outputs.  The
only writes are a JSON result and two review montages under ``--out-dir``.

Run from the repository root (CPU only; no model/checkpoint/GPU access)::

    env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
      scripts/foundations/verify_buildingworld_voids.py \
      --real-h5 data/real_massing_v1/real.h5 \
      --mesh-root data/buildingworld_mesh \
      --out-dir docs/wayfinding/whole-volume-voxel-transform/119-buildingworld-void-verification

The committed result used ``--seed 119 --tokyo-n 1500 --melbourne-n 1000``.  Row sampling is
uniform without replacement from sorted corpus row IDs via NumPy ``default_rng``.  Samples are
drawn independently per cohort so adding another cohort cannot change either frozen row list.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.foundations.prototype_voxel_editor import hollow_shell_voxels


RES = 64
DEFAULT_SEED = 119
DEFAULT_TOKYO_N = 1500
DEFAULT_MELBOURNE_N = 1000


@dataclass(frozen=True)
class VolumeResult:
    has_gap: bool
    solid_voxels: int
    footprint_columns: int
    gap_columns: int
    gap_voxels: int
    reachable_gap_voxels: int
    sealed_gap_voxels: int
    hollow_shell_voxels: int

    @property
    def classification(self) -> str:
        if not self.has_gap:
            return "no_gap"
        if self.reachable_gap_voxels and not self.sealed_gap_voxels:
            return "reachable_only"
        if self.reachable_gap_voxels and self.sealed_gap_voxels:
            return "mixed_reachable_and_sealed"
        return "sealed_only"


def below_roofline_gap_mask(occ: np.ndarray) -> np.ndarray:
    """Empty voxels from the building's inferred base to each column's topmost solid voxel.

    ``real.h5`` stores ``(D=z, H=up, W=x)``.  Its per-building centered frame does not put the
    ground at H=0, so the base is inferred as the first H plane containing any solid voxel, exactly
    as the report's coarse measurement describes.
    """
    occ = np.asarray(occ, dtype=bool)
    if occ.ndim != 3:
        raise ValueError(f"expected a 3-D occupancy volume, got {occ.shape}")
    occupied_y = occ.any(axis=(0, 2))
    if not occupied_y.any():
        return np.zeros_like(occ)
    y0 = int(np.flatnonzero(occupied_y)[0])
    column_has_solid = occ.any(axis=1)
    top = np.full(column_has_solid.shape, -1, dtype=np.int16)
    top[column_has_solid] = (
        occ.shape[1] - 1 - np.argmax(occ[:, ::-1, :], axis=1)[column_has_solid]
    )
    y = np.arange(occ.shape[1], dtype=np.int16)[None, :, None]
    below_roofline = column_has_solid[:, None, :] & (y >= y0) & (y <= top[:, None, :])
    return below_roofline & ~occ


def analyze_volume(occ: np.ndarray) -> tuple[VolumeResult, np.ndarray, np.ndarray]:
    """Return scalar evidence plus the coarse-gap and sealed-empty masks.

    ``hollow_shell_voxels`` is deliberately called unmodified.  In the committed implementation
    its empty-space flood fill is 6-connected; the related solid-component helper is 26-connected.
    """
    occ = np.asarray(occ, dtype=bool)
    gap = below_roofline_gap_mask(occ)
    hollow = hollow_shell_voxels(occ)
    reachable_gap = gap & ~hollow
    sealed_gap = gap & hollow
    result = VolumeResult(
        has_gap=bool(gap.any()),
        solid_voxels=int(occ.sum()),
        footprint_columns=int(occ.any(axis=1).sum()),
        gap_columns=int(gap.any(axis=1).sum()),
        gap_voxels=int(gap.sum()),
        reachable_gap_voxels=int(reachable_gap.sum()),
        sealed_gap_voxels=int(sealed_gap.sum()),
        hollow_shell_voxels=int(hollow.sum()),
    )
    return result, gap, hollow


def _decode(value) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def cohort_rows(h5: h5py.File, source_key: str, defect_class: str | None = None) -> np.ndarray:
    keys = h5["source_key"][:]
    mask = keys == source_key.encode("ascii")
    if defect_class is not None:
        mask &= h5["defect_class"][:] == defect_class.encode("ascii")
    return np.flatnonzero(mask)


def sample_rows(rows: np.ndarray, n: int, seed: int) -> np.ndarray:
    rows = np.sort(np.asarray(rows, dtype=np.int64))
    if n > len(rows):
        raise ValueError(f"requested {n} rows from a population of {len(rows)}")
    # The seed is cohort-local by caller, so future cohorts cannot perturb these samples.
    return np.sort(np.random.default_rng(seed).choice(rows, size=n, replace=False))


def analyze_cohort(h5: h5py.File, name: str, rows: np.ndarray) -> tuple[dict, list[dict]]:
    records: list[dict] = []
    occupancy_digest = hashlib.sha256()
    for ordinal, row in enumerate(rows, 1):
        sdf = h5["sdf"][int(row)]
        occ = sdf <= 0
        occupancy_digest.update(np.asarray(int(row), dtype="<i8").tobytes())
        occupancy_digest.update(np.packbits(occ, bitorder="little").tobytes())
        result, _, _ = analyze_volume(occ)
        if result.has_gap:
            record = asdict(result)
            record.update(
                row=int(row),
                bag_id=_decode(h5["bag_id"][int(row)]),
                source_key=_decode(h5["source_key"][int(row)]),
                defect_class=_decode(h5["defect_class"][int(row)]),
                classification=result.classification,
            )
            records.append(record)
        if ordinal % 100 == 0:
            print(f"[{name}] {ordinal}/{len(rows)}", flush=True)

    n_gap = len(records)
    reachable = [r for r in records if r["reachable_gap_voxels"] > 0]
    sealed_only = [r for r in records if r["classification"] == "sealed_only"]
    mixed = [r for r in records if r["classification"] == "mixed_reachable_and_sealed"]
    summary = {
        "population_rows": int(len(cohort_rows(
            h5,
            _decode(h5["source_key"][int(rows[0])]),
            (_decode(h5["defect_class"][int(rows[0])]) if name == "melbourne_non_boundary_defect"
             else None),
        ))),
        "sample_n": int(len(rows)),
        "sample_rows": [int(x) for x in rows],
        "sample_occupancy_sha256": occupancy_digest.hexdigest(),
        "gap_flagged_rows": n_gap,
        "gap_flagged_fraction": n_gap / len(rows),
        "reachable_gap_rows": len(reachable),
        "reachable_gap_fraction_of_sample": len(reachable) / len(rows),
        "reachable_fraction_of_gap_flagged": len(reachable) / max(n_gap, 1),
        "sealed_only_rows": len(sealed_only),
        "mixed_rows": len(mixed),
        "gap_voxels": int(sum(r["gap_voxels"] for r in records)),
        "reachable_gap_voxels": int(sum(r["reachable_gap_voxels"] for r in records)),
        "sealed_gap_voxels": int(sum(r["sealed_gap_voxels"] for r in records)),
        "reachable_gap_voxel_fraction": (
            sum(r["reachable_gap_voxels"] for r in records)
            / max(sum(r["gap_voxels"] for r in records), 1)
        ),
    }
    if records:
        for field in ("gap_voxels", "reachable_gap_voxels", "sealed_gap_voxels"):
            values = np.asarray([r[field] for r in records])
            summary[f"{field}_affected_median"] = float(np.median(values))
            summary[f"{field}_affected_p90"] = float(np.percentile(values, 90))
            summary[f"{field}_affected_max"] = int(values.max())
    return summary, records


def choose_examples(records: list[dict], n: int) -> list[dict]:
    """Choose deterministic size-spread examples, avoiding one extreme dominating the montage."""
    candidates = sorted(
        (r for r in records if r["reachable_gap_voxels"] > 3),
        key=lambda r: (r["reachable_gap_voxels"], r["row"]),
    )
    if len(candidates) <= n:
        return candidates
    positions = np.linspace(0.20, 0.95, n)
    return [candidates[min(round(p * (len(candidates) - 1)), len(candidates) - 1)]
            for p in positions]


def _add_surface(ax, volume: np.ndarray, color: str, alpha: float) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage import measure

    if not volume.any() or volume.all():
        return
    verts, faces, _, _ = measure.marching_cubes(volume.astype(np.float32), level=0.5)
    # Stored arrays are (D=z, H=up, W=x); matplotlib expects its third coordinate to be vertical.
    verts = verts[:, [2, 0, 1]]
    mesh = Poly3DCollection(verts[faces], alpha=alpha, linewidth=0.0)
    mesh.set_facecolor(color)
    mesh.set_edgecolor("none")
    ax.add_collection3d(mesh)


def _style_3d(ax, title: str) -> None:
    ax.set(xlim=(0, RES - 1), ylim=(0, RES - 1), zlim=(0, RES - 1), title=title)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=24, azim=-54)
    ax.set_axis_off()


def render_contact_sheet(h5: h5py.File, examples: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = 4
    rows = int(np.ceil(len(examples) / cols))
    fig = plt.figure(figsize=(15, 3.9 * rows), facecolor="#f5f1e8")
    for i, record in enumerate(examples):
        sdf = h5["sdf"][record["row"]]
        occ = sdf <= 0
        result, gap, hollow = analyze_volume(occ)
        reachable = gap & ~hollow
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        _add_surface(ax, occ, "#66788a", 0.34)
        _add_surface(ax, reachable, "#e4572e", 0.88)
        city = record["source_key"].split(":", 1)[-1]
        _style_3d(ax, f"{city} · row {record['row']}\n"
                      f"reachable {result.reachable_gap_voxels:,} · sealed {result.sealed_gap_voxels:,}")
    fig.suptitle("BuildingWorld gap verification: solid massing (slate) + exterior-reachable gap (orange)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _canonical_zip(mesh_root: Path, city: str) -> Path:
    paths = list((mesh_root / city).rglob("*.zip"))
    by_name = {p.name: p for p in paths}
    path = by_name.get("mesh.zip") or by_name.get("obj.zip")
    if path is None:
        raise FileNotFoundError(f"no mesh.zip/obj.zip under {mesh_root / city}")
    return path


def _raw_mesh(mesh_root: Path, bag_id: str):
    import trimesh

    city, member = bag_id.split("#", 1)
    with zipfile.ZipFile(_canonical_zip(mesh_root, city)) as zf:
        loaded = trimesh.load(io.BytesIO(zf.read(member)), file_type="obj", process=True)
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.to_geometry()
    return np.asarray(loaded.vertices), np.asarray(loaded.faces)


def _normalized(vertices: np.ndarray) -> np.ndarray:
    v = np.asarray(vertices, dtype=np.float64)
    lo, hi = v.min(axis=0), v.max(axis=0)
    scale = max(float((hi - lo).max()), 1e-9)
    return (v - (lo + hi) / 2.0) / scale + 0.5


def _plot_triangles(ax, vertices: np.ndarray, faces: np.ndarray, color: str) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    v = _normalized(vertices)
    collection = Poly3DCollection(v[faces], alpha=0.88, linewidth=0.02)
    collection.set_facecolor(color)
    collection.set_edgecolor("#26323b")
    ax.add_collection3d(collection)
    ax.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1))
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=24, azim=-54)
    ax.set_axis_off()


def render_reconstruction_check(h5: h5py.File, mesh_root: Path, examples: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage import measure

    fig = plt.figure(figsize=(12, 3.5 * len(examples)), facecolor="#f5f1e8")
    for i, record in enumerate(examples):
        raw_v, raw_f = _raw_mesh(mesh_root, record["bag_id"])
        sdf = h5["sdf"][record["row"]]
        mc_v, mc_f, _, _ = measure.marching_cubes(sdf, level=0.0)

        left = fig.add_subplot(len(examples), 2, i * 2 + 1, projection="3d")
        _plot_triangles(left, raw_v, raw_f, "#7d8e9e")
        left.set_title(f"raw mesh · row {record['row']}")
        right = fig.add_subplot(len(examples), 2, i * 2 + 2, projection="3d")
        _plot_triangles(right, mc_v[:, [2, 0, 1]], mc_f, "#d76f45")
        right.set_title("stored SDF → marching cubes @ 0")
    fig.suptitle("BuildingWorld raw-mesh / ingested-SDF reconstruction check",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_verification(args: argparse.Namespace) -> dict:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.real_h5, "r") as h5:
        real_h5_rows = int(len(h5["sdf"]))
        cohorts = {
            "tokyo": sample_rows(cohort_rows(h5, "bw:Tokyo"), args.tokyo_n, args.seed),
            "melbourne_non_boundary_defect": sample_rows(
                cohort_rows(h5, "bw:Melbourne", "non_boundary_defect"),
                args.melbourne_n,
                args.seed,
            ),
        }
        summaries = {}
        all_records = {}
        for name, rows in cohorts.items():
            summaries[name], all_records[name] = analyze_cohort(h5, name, rows)

        tokyo_examples = choose_examples(all_records["tokyo"], 6)
        melbourne_examples = choose_examples(all_records["melbourne_non_boundary_defect"], 6)
        contact_examples = tokyo_examples + melbourne_examples
        render_contact_sheet(h5, contact_examples, out_dir / "contact_sheet.png")
        reconstruction_examples = tokyo_examples[1::2][:3] + melbourne_examples[1::2][:3]
        render_reconstruction_check(
            h5, Path(args.mesh_root), reconstruction_examples, out_dir / "recon_compare.png"
        )

    result = {
        "schema": "buildingworld-void-verification.v1",
        "method": {
            "seed": args.seed,
            "occupancy": "sdf <= 0",
            "axis_order": "(D=z, H=up, W=x)",
            "gap_test": "empty from inferred global base through per-column topmost solid voxel",
            "reachability": (
                "scripts.foundations.prototype_voxel_editor.hollow_shell_voxels; committed "
                "implementation uses 6-connected empty-space flood fill from all boundary faces"
            ),
            "real_h5": str(args.real_h5),
            "mesh_root": str(args.mesh_root),
            "real_h5_rows": real_h5_rows,
        },
        "cohorts": summaries,
        "gap_flagged_records": all_records,
        "visual_examples": {
            "contact_sheet_rows": [r["row"] for r in contact_examples],
            "reconstruction_rows": [r["row"] for r in reconstruction_examples],
        },
    }
    (out_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    printable = {
        name: {key: value for key, value in summary.items() if key != "sample_rows"}
        for name, summary in summaries.items()
    }
    print(json.dumps({"cohorts": printable, "visual_examples": result["visual_examples"]}, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-h5", type=Path, required=True)
    parser.add_argument("--mesh-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--tokyo-n", type=int, default=DEFAULT_TOKYO_N)
    parser.add_argument("--melbourne-n", type=int, default=DEFAULT_MELBOURNE_N)
    run_verification(parser.parse_args())


if __name__ == "__main__":
    main()
