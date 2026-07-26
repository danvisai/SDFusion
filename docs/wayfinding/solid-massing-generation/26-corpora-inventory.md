# Solid-Massing Corpora Inventory

Resolves [Audit the solid-massing corpora and their footprint alignment](https://github.com/danvisai/SDFusion/issues/26).

## Headline

**The solid-massing corpus already exists, is footprint-paired, and spans three cultures — no onboarding required.** `data/real_massing_v1/real.h5` holds **35,776 real LoD2 buildings**, genuinely solid (median occupancy 13–30%, vs BuildingNet's 0.56% thin shells), already in the exact Stage3a footprint-conditioned format (64³ SDF + 64×64 footprint + `height_m` + `source_id`/`style_id`). LoD2 is watertight-solid by construction, so this data needs **no** solidification.

## Inventory

| corpus | file | N | class | occ median | footprint paired | status | alignment cost |
|---|---|---|---|---|---|---|---|
| NL 3D BAG | `bag3d.h5` / `real.h5` src 0 | 11,776 | BAG_real | **13.7%** | 64×64 + `height_m` | ingested | ~0 |
| German LoD2 (NRW) | `real.h5` src 1 | 12,000 | DE_real | ~solid | 64×64 + `height_m` | ingested | ~0 |
| JP PLATEAU | `plateau.h5` / `real.h5` src 2 | 12,000 | JP_real | **28.9% / 30.2%** | 64×64 + `height_m` | ingested | ~0 |
| **combined** | **`real.h5`** | **35,776** | 3 cultures | 13–30% | yes | **ingested (Stage3a corpus)** | **~0** |
| BuildingNet | `resolution_64/` | 1,849 | 4 types | 0.56% (thin) | 64×64 | ingested; needs #28 solidify + #32 mask | solidify+mask |
| more NL/DE LoD2 | not persisted | millions avail | — | ~solid | derivable | pipeline proven (NRW), not ingested | run ingest (moderate) |
| lod3_tum | `data/lod3_tum` | small | DE LoD3 | — | — | detail/elements, **not massing** | out of massing scope |

## Findings

1. **Solid-massing data is plentiful and already footprint-conditioned** — 35,776 buildings, all with the same 64×64 footprint + `height_m` contract, so the footprint-alignment cost across the three LoD2 sources is **≈ 0** (already unified at ingest).
2. **LoD2 is solid by construction** (occ 13–30%) and does **not** need the #28 solidify; only BuildingNet (0.56%) does.
3. **Three cultures** (NL / DE / JP via `source_id`) give real cross-cultural style diversity — matching the "different styles" preference from the gate discussion (#27). The `reference-building-datasets` note recommends conditioning `source_id` in the Stage3a style embedding so cultures are conditioned, not blurred.

## Confirms and extends #25's reframe (diversity, not necessity)

Solid massing needs **no** BuildingNet solidification at all — the LoD2 corpora *are* the solid-massing source. BuildingNet (+ #28 solidify + #32 mask) is an **optional shape-variety add-on**, not a necessity.

## Key input to the retrain recipe (#30) — flagged, not decided here

- The massing generator can train on **35,776 already-solid, footprint-paired LoD2 buildings across 3 cultures**. This is very likely the direct fix for "breaking apart" — the fragments were observed on *BuildingNet thin-shell* targets, not on this solid LoD2 data.
- Open for #30: **is BuildingNet needed in the massing mix at all?** If yes (shape variety), apply #28/#29/#32 to its portion; if no, those are moot for massing. Plus: NL/DE/JP weighting, and whether to condition `source_id`.

## Implication for prior decisions

#28 (solidify), #29 (fallback), #32 (asset mask) apply **only to the BuildingNet portion** and are **conditional on #30 including BuildingNet in the massing mix**. They are not invalidated — just scoped to BuildingNet.
