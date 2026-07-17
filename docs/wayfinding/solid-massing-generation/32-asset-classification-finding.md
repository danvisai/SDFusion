# Finding: the near-empty tail is mostly thin buildings — mask non-building labels, don't filter the tail

Resolves ticket [Determine whether the near-empty-footprint tail is non-building assets, and decide massing-side handling](https://github.com/danvisai/SDFusion/issues/32).

## Method

BuildingNet ships **per-face semantic labels** (`model_data/obj/face_labels/<id>.json`, all 1,849 meshes) over a taxonomy that includes explicit **non-building** classes: `9 = ground` ("the big flat sheet around the base", 44% presence / 17.3 pts%), `19/23 = road/floor`, `5 = plant/tree`, `13 = fence`. Compared non-building face share for the near-empty tail (occ < 0.3%) vs healthy buildings (occ > 1%) over 120 sampled held-out meshes. Face-count weighting *under*-counts ground (few large quads), so any ground share it reports is a conservative floor.

## Results

| group | n | ground %face (median) | non-building %face (median) | share >50% non-building |
|---|---|---|---|---|
| near-empty tail (occ<0.3%) | 44 | 0.0 | 0.0 | 0% |
| healthy (occ>1%) | 38 | 0.0 | 0.1 | 5% |

Worst contaminated tail examples: `church_mesh0887` (43% props), `house_mesh1281` (31% ground), `house_mesh6798` (24% props), several at 15–19%.

## Verdict

**The tail is NOT primarily non-building assets — the median tail mesh is a genuinely thin building with ~0% non-building geometry.** So the solidify-in-place plan (#25/#28/#29) stays valid for the majority; do **not** filter the whole tail.

**But a minority (~10–15% of the tail, ~5% of healthy) carries real non-building geometry** — ground sheets, plants, fences. This matters because the stored `footprint` field #28/#29 extrude is derived from the *whole* mesh (`occ.any(axis=1)`), so it **includes the ground sheet** — extruding it would build a block out of the ground exactly as you flagged, and face-count under-states how much of the *footprint* a flat ground sheet occupies.

## Decision — mask, don't filter

1. **Mask to building-part labels** when forming both the solid target and the footprint: exclude `9 ground`, `19/23 road/floor`, `5 plant`, `13 fence`. Per-face labels exist for all 1,849 meshes, so this is a cheap data-prep pass (re-voxelize building-labeled faces, or project the building-only footprint).
2. **Filter** only meshes left with negligible building mass after masking (a small set) — those aren't usable building samples.
3. Do **not** blanket-filter the near-empty tail; most of it is legitimate thin-building data.

## Implications for other tickets

- **#28 / #29 (closed):** use a **building-only footprint** (mask non-building faces before projecting), not the raw stored footprint, so the extrusion fallback never solidifies a ground sheet.
- **[Acceptance gate #27](https://github.com/danvisai/SDFusion/issues/27):** score building-only solidity and exclude ground from footprint-IoU.
- **[Corpora audit #26](https://github.com/danvisai/SDFusion/issues/26):** apply the same building-only masking to any new corpus.
- **[Retrain recipe #30](https://github.com/danvisai/SDFusion/issues/30):** the label-masking + negligible-building-mass filter is a data-prep step in the recipe.
- Out of scope (unchanged): routing image-derived non-building assets and composing a coherent scene.
