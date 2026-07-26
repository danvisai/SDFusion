# Freeze Leakage-Safe BuildingNet Splits

Type: task
Status: resolved
Blocked by:

## Question

Create deterministic, class-stratified, nested BuildingNet test/train_25/train_50/train_100 id lists;
verify source coverage, nesting, disjointness, and the sealed-test invariant; and record the seed,
counts, class balance, and reproducible command so every later arm consumes the same detail data.

## Comments

## Answer

Implemented `scripts/foundations/make_splits.py` (pure `make_splits(items, seed, test_frac)` seam +
BuildingNet enumeration matching the element-library universe: `component_labels/*_label.json`,
top-level class = leading uppercase run). TDD against `scripts/foundations/test_make_splits.py`
(10 contract tests, all green): deterministic reproduction, seed-sensitivity, no duplicates, nesting
(`train_25 ⊂ train_50 ⊂ train_100`), sealed-test disjointness, full coverage, ~15% class-stratified
holdout, per-fraction class presence, and fraction sizes.

Frozen split (seed 0, test_frac 0.15) written to `data/splits_v1/` (gitignored, reproducible via the
tracked script + `manifest.json`): **test 277, train_25 392, train_50 785, train_100 1572** over 1849
buildings; coverage 277+1572=1849; all three invariants (`sealed_test_disjoint`, `nested`,
`full_coverage`) true; per-class test holdout ~15% across all 4 classes (RESIDENTIAL 184/1225,
RELIGIOUS 67/445, COMMERCIAL 18/123, PUBLIC 8/56). Regenerating == saved confirmed.

Unblocks ticket 04 (leakage-safe library builder consumes `train_X` / excludes `test`).
