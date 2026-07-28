# The surface corpus — #62 re-ingest, executed

**Date:** 2026-07-28 · **Result: 35,623 / 35,776 recovered (99.6%), 18 MB, all three sources verified
aligned.** The hard prerequisite for [spec #68](https://github.com/danvisai/SDFusion/issues/68)'s frozen
gate — and for the whole A2 direction — is discharged.

## What was recovered

| source | recovered | rate | alignment (IoU vs stored field) |
|---|---|---|---|
| **NL — 3D BAG** | 11,773 / 11,776 | **99.97 %** | **1.0000** (L1 0.00000) |
| **DE — NRW OpenGeodata** | 11,850 / 12,000 | 98.8 % | **0.9904** (L1 0.00084) |
| **JP — PLATEAU** | 12,000 / 12,000 | **100 %** | **1.0000** (L1 0.00000) |
| **total** | **35,623 / 35,776** | **99.6 %** | — |

**18 MB for the entire corpus** — LoD2 massing is a handful of vertices per building (NL 534k verts,
DE 182k, JP 146k across ~35.6k buildings). The thing that blocked the vecset direction turned out to
cost less storage than a single checkpoint shard.

**Integrity:** 35,623 buildings map to 35,623 *distinct* rows, so no row is claimed twice.

## How it was done

Both existing ingests already construct a `trimesh` per building and discard it at the voxelisation
step. The re-ingest walks the same sources with the same parsers and keeps the mesh, storing it in
**Frame-N** — the exact normalisation the voxeliser applies (centre on bbox, divide by
`max_extent/2 × 1.05`, reorder CityGML z-up to y-up).

Rows are joined **by identifier, never by order**, so upstream need not reproduce the original ordering
or filter outcomes for the pairing to hold.

Costs: JP 3 nested archives / 45 MB / 161 s. DE the 41 referenced tiles / 234 MB / 75 s. NL 11,776
individual API lookups / 1,466 s.

## Four faults the alignment check caught

None of these would have raised an error. Each would have silently corrupted the pairing or the
encoding, and all were caught by re-voxelising recovered meshes and comparing occupancy against the
stored fields.

1. **Axis order.** The voxeliser builds its grid as `meshgrid(ZZ, YY, XX)`, so the **stored array is
   indexed [z, y, x]** while the natural query ordering is [x, y, z]. Mixing a world-frame mesh with a
   stored field requires transposing one. (The earlier ceiling probe was unaffected — its mesh and its
   queries were both in the array's own index frame.)
2. **Sign type.** The recovered meshes are watertight but **negative-volume**, and the default
   signed-distance sign test then reports **zero occupancy** — the first check returned IoU exactly
   0.0000 across the board. Signing must use the fast-winding-number method the original voxeliser used.
3. **Winding — the one that matters downstream.** A vecset encoder consumes **face normals**. Inverted
   winding would have handed it inside-out surfaces, degrading every encoding without ever erroring.
   Repaired at save time.
4. **Tile selection.** The upstream NRW iterator *strides across the whole state* to sample tiles, which
   would have downloaded thousands of files and might still have missed ours. Recovery resolves the exact
   41 tiles our rows reference.

## Why the NL arm took a second attempt

The 3DBAG API costs ~2 s per building, so 11,776 sequential lookups projected to **399 minutes**; the
first run was killed by its own timeout before writing anything. The calls are I/O-bound, so the network
was parallelised across a small thread pool with mesh construction kept single-threaded, and requests
chunked so fetched JSON does not accumulate.

Observed effective concurrency was ~5× rather than the pool's 16, which looks like server-side rate
limiting. The worker count was deliberately **left modest rather than raised** — 3DBAG is a free public
service, and hammering it harder to save twenty minutes is not a good trade.

## What this unblocks

The frozen round-trip gate can now run on **real LoD2 surfaces**. That matters concretely: the first
round-trip attempt had to feed the encoder a mesh extracted from our own 64³ field, so it faithfully
encoded the grid roughness the project is trying to escape and scored **0.00839** — worse than the
deployed 0.00552. That number was a lower bound on a degraded input, not a measurement of the codec.

The gate can now be run honestly.

## Reproduce

```
scripts/foundations/ingest_surfaces.py --source plateau
scripts/foundations/ingest_surfaces.py --source nrw
scripts/foundations/ingest_surfaces.py --source bag3d
scripts/foundations/ingest_surfaces.py --source <s> --verify --verify_n 10
```

Artifacts: `data/real_massing_v1/surfaces_{bag3d,nrw,plateau}.h5` — packed `verts`/`faces` with
offsets, plus `row` (index into `real.h5`) and `bag_id` for provenance. Local only; `data/` is
gitignored.
