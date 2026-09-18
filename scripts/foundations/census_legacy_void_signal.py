"""Census the exterior-reachable void signal over the A2-eligible legacy population (#119).

[The BuildingWorld void/passage verification](../../docs/wayfinding/whole-volume-voxel-transform/119-buildingworld-void-passage-verification.md)
sampled BuildingWorld and found its below-roofline gaps are predominantly exterior-reachable
architecture.  It left open whether the *legacy* population could serve the same purpose.  This
script settles that by census rather than sample: it measures every row the frozen A2 source can
actually generate for.

That population is exactly `vecset_latents.h5` -- 35,623 rows carrying region 0/1/2.  The frozen
`vecset_v4_surf` @240k conditions through `region = nn.Embedding(3, 512)`, so rows outside it
(every BuildingWorld row, region 3-8) raise `IndexError` rather than producing a source.

CPU only, read-only: no checkpoint is loaded, no GPU is touched, no corpus file is written.

Run from the repository root::

    env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
      scripts/foundations/census_legacy_void_signal.py \
      --out docs/wayfinding/whole-volume-voxel-transform/119-legacy-void-census.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.foundations.prototype_voxel_editor import S_STAR_VOXELS, hollow_shell_voxels  # noqa: E402
from utils.frozen_corpus import open_real_corpus  # noqa: E402

DEFAULT_LATENTS = REPO / "data/real_massing_v1/vecset_latents.h5"
DEFAULT_REAL = REPO / "data/real_massing_v1/real.h5"
REGION_NAMES = {0: "bag3d_nl", 1: "nrw_de", 2: "plateau_jp"}

# A reachable void counts as structural when it could survive ADR 0004's fixed detail scale
# s* = 3 voxels @64^3 in every direction.  Smaller openings are real but below the scale this
# effort claims to model, so counting them would overstate the available signal.
STRUCTURAL_VOXELS = S_STAR_VOXELS ** 3


def below_roofline_gap_mask(occ: np.ndarray) -> np.ndarray:
    """Empty voxels from the building's inferred base to each column's topmost solid voxel.

    Verbatim copy of `verify_buildingworld_voids.below_roofline_gap_mask`, duplicated rather than
    imported so this census does not depend on that script's CLI/mesh-archive imports.
    `test_census_legacy_void_signal.py` asserts the two stay identical whenever both are present.

    `real.h5` stores `(D=z, H=up, W=x)`.  Its per-building centered frame does not put the ground
    at H=0, so the base is inferred as the first H plane containing any solid voxel.
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


def census_row(occ: np.ndarray) -> tuple[int, int, int, int]:
    """`(solid, gap, reachable_gap, sealed_gap)` voxel counts for one occupancy volume."""
    solid = int(occ.sum())
    if not solid:
        return 0, 0, 0, 0
    gap = below_roofline_gap_mask(occ)
    n_gap = int(gap.sum())
    if not n_gap:
        return solid, 0, 0, 0
    hollow = hollow_shell_voxels(occ)
    reach = int((gap & ~hollow).sum())
    return solid, n_gap, reach, n_gap - reach


def run(latents: Path, real: Path, progress_every: int = 2000) -> dict:
    with h5py.File(latents, "r") as lat:
        rows = np.asarray(lat["row"][:], np.int64)
        region = np.asarray(lat["region"][:], np.int64)
        held_out = np.asarray(lat["held_out"][:], np.uint8)
    order = np.argsort(rows)
    rows, region, held_out = rows[order], region[order], held_out[order]

    recs: list[dict] = []
    t0 = time.time()
    with open_real_corpus(real) as corpus:
        sdf = corpus["sdf"]
        for i, row in enumerate(rows):
            solid, gap, reach, sealed = census_row(np.asarray(sdf[int(row)], np.float32) <= 0)
            recs.append({"row": int(row), "region": int(region[i]),
                         "held_out": int(held_out[i]), "solid": solid,
                         "gap": gap, "reach": reach, "sealed": sealed})
            if progress_every and (i + 1) % progress_every == 0:
                el = time.time() - t0
                print(f"  {i + 1}/{len(rows)}  {el:.0f}s  "
                      f"eta {el / (i + 1) * (len(rows) - i - 1):.0f}s", flush=True)

    stream = b"".join(f"{r['row']}:{r['solid']}:{r['gap']}:{r['reach']}:{r['sealed']}\n".encode()
                      for r in recs)

    def bucket(subset: list[dict]) -> dict:
        structural = [r for r in subset if r["reach"] >= STRUCTURAL_VOXELS]
        return {
            "n": len(subset),
            "empty_rows": sum(1 for r in subset if not r["solid"]),
            "gap_rows": sum(1 for r in subset if r["gap"]),
            "reachable_rows": sum(1 for r in subset if r["reach"]),
            "structural_rows": len(structural),
            "structural_held_out_rows": sum(1 for r in structural if r["held_out"]),
            "sealed_only_rows": sum(1 for r in subset if r["gap"] and not r["reach"]),
            "reachable_voxels": sum(r["reach"] for r in subset),
            "sealed_voxels": sum(r["sealed"] for r in subset),
        }

    summary = {REGION_NAMES[g]: bucket([r for r in recs if r["region"] == g]) for g in (0, 1, 2)}
    summary["total"] = bucket(recs)
    return {
        "population": "vecset_latents.h5 rows: the region 0/1/2 rows the frozen A2 can condition on",
        "latents_path": str(latents),
        "real_corpus_path": str(real),
        "structural_reach_voxel_threshold": STRUCTURAL_VOXELS,
        "rows_censused": len(recs),
        # Digest over EVERY censused row, including the silent majority omitted from
        # `gap_rows_detail` below, so it still pins the whole population's content.
        "row_stream_sha256": hashlib.sha256(stream).hexdigest(),
        "summary": summary,
        "structural_rows": sorted(r["row"] for r in recs if r["reach"] >= STRUCTURAL_VOXELS),
        # Only rows carrying a below-roofline gap. A gap-free row's per-row record says nothing
        # the summary does not, and keeping all 35,623 of them made this artifact 5 MB.
        "gap_rows_detail": [r for r in recs if r["gap"]],
        "elapsed_seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--latents", type=Path, default=DEFAULT_LATENTS)
    ap.add_argument("--real", type=Path, default=DEFAULT_REAL)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    result = run(args.latents, args.real)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
