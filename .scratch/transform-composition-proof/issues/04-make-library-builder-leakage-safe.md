# Make the Element-Library Builder Leakage-Safe

Type: task
Status: resolved
Blocked by: 03

## Question

Extend `scripts/foundations/build_element_library.py` with explicit include/exclude id inputs and
configurable output locations, then prove from emitted metadata that excluded test buildings never
contribute elements and that repeated builds are deterministic enough for fraction comparisons.

## Comments

## Answer

Extended `scripts/foundations/build_element_library.py`: `--include-ids` / `--exclude-ids` (exclude
ALWAYS wins over include, via the pure `select_building_ids` seam), `--out` / `--qa-out` / `--no-qa`.
Every build emits `manifest.json` with `contributing_buildings` provenance plus a
`leakage_excluded_contributors` audit that is **asserted empty** at build end. Default behavior (the
live `data/element_library_v1` build) is unchanged.

TDD: `scripts/foundations/test_build_element_library.py`, 7/7 green — include restricts, exclude
removes, **exclude wins over include**, unknown ids ignored, sorted/order-independent, `load_id_list`.

Integration proof (small real `--no-qa` builds, from emitted metadata):
- **Leakage-safe:** lib1 (include `train_25`, exclude `test`, limit 60) → 110 elements;
  `leakage_excluded_contributors == []`; 277 test ids present in the universe; contributing ∩ test = ∅;
  contributing ⊆ train_25.
- **Deterministic:** identical re-build → `elements_f16.npy` bytes AND `meta.json` identical.
- **Removal proof (not a no-op):** lib3 (built FROM `test`, limit 40) → 79 elements from test
  buildings; the 29 contributing test buildings are ABSENT from lib1.

Unblocks ticket 08 (per-fraction libraries from `train_X` excluding `test`) and ticket 05 (FID harness).
