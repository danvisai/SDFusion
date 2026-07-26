# Build the Full-Data Experiment Element Library

Type: task
Status: resolved
Blocked by: 02, 04

## Question

Build the train_100-only retrieval library with the validated Phase R representation, verify zero
held-out provenance, quantify per-type pool size and solidity/scale distributions, and establish the
frozen library configuration used by the 100% decomposition arm.

## Comments

## Answer

**Safety catch before building anything:** `scene/element_lib.py`'s `LIB = REPO /
"data/element_library_v1"` is the LIVE path the deployed sculptor/server reads elements from
(`scripts/server/element_fit.py`). That directory on disk predates ticket 03/04 entirely (built
`aa376a4`/`d781bb4`, before the sealed split existed) and has no `manifest.json` at all — so it is
almost certainly not leakage-safe against `data/splits_v1/test.json`. This experiment build
therefore writes to a SEPARATE path, `data/element_library_train100_v1/`, never touching the
production directory (verified: `elements_f16.npy`/`meta.json` mtimes unchanged after this run).
Whether/when to refresh the live path is a product decision out of this ticket's scope.

**Built:** extended `scripts/foundations/build_element_library.py` (ticket 04's leakage-safe
builder — no new script) with `crop_solidity`, `scale_rel`, and `distribution_stats` (TDD, 9 new
contract tests, 16/16 total in the file). `crop_solidity` is verified byte-identical (max abs diff
0.0 over a 200-element cross-check) to `scripts/server/element_fit.py`'s `_solidity` fallback
formula — one definition, measured at build time here instead of lazily cached at serve time.

**Frozen library configuration** (recorded per-build in `manifest.json.frozen_config`, unchanged
from ticket 04): `RES=48`, `MIN_FACES=60`, `MAX_PER_TYPE_PER_BLDG=4`, `MAX_PER_TYPE=3000`, the
8-type ADD vocabulary (tower/dome/chimney/roof_structure/balcony/balcony_upper/column/stairs).
`MIN_SOLIDITY=0.12` (matching `element_fit.py`'s retrieval threshold) is recorded and used only to
report `pool_size_above_min_solidity` per type — never used to filter the library itself, so the
manifest reflects exactly what retrieval will see at serve time.

**Full `train_100` build** (`--include-ids data/splits_v1/train_100.json --exclude-ids
data/splits_v1/test.json --out data/element_library_train100_v1`):

| check | result |
|---|---|
| buildings selected / universe | 1572 / 1849 (277 excluded = the sealed test set) |
| elements extracted | 2744 (contributing buildings: 1133 / 1572) |
| leakage (`leakage_excluded_contributors`) | `[]` — 0 |
| by type | dome 429, balcony 352, roof_structure 305, column 472, balcony_upper 171, stairs 223, tower 534, chimney 258 |

2744/3204 ≈ 85.6% of the production (all-1849-building) element count, closely tracking
1572/1849 ≈ 85.0% of buildings selected — consistent with excluding a stratified 15% slice, not a
sign of a broken filter.

**Solidity/scale distributions** (`solidity_by_type`/`scale_by_type` in the manifest,
`outputs/element_library_train100_v1/distributions.png`): every type is heavily right-skewed
toward low solidity (means 0.02–0.11; BuildingNet ADD components are mostly thin/skeletal
shells, matching ticket 02's finding that motivated the solidity filter in the first place).
`pool_size_above_min_solidity`: tower 122, dome 104, chimney 48, roof_structure 12, column 8,
balcony 5, **balcony_upper 0, stairs 0**. The zero pools are not a bug in this build — the same
zero (0/251 stairs, 1/198 balcony_upper) holds in the full 3204-element production library,
cross-checked directly. This is a genuine, previously-unmeasured finding: at `MIN_SOLIDITY=0.12`,
retrieval can currently never actually serve a `stairs` element and only very rarely a
`balcony_upper`, regardless of which BuildingNet fraction backs the library — worth flagging for
ticket 12/13's failure-localization if the decomposition arm underperforms on those categories,
but out of scope to fix here (the ticket asks to quantify, not to retune the threshold).

**Out:** `data/element_library_train100_v1/{elements_f16.npy, meta.json, solidity.npy,
manifest.json}` (gitignored), `outputs/element_library_train100_v1/{montage_<type>.png,
distributions.png}`.

Unblocks ticket 12 (generate the full-data decomposition arm).
