# Validate the Element Retrieval Baseline

Type: task
Status: resolved
Blocked by:

## Question

Do Claude's existing Phase R quality changes eliminate skeletal, badly scaled, and voxel-crushed
retrievals without regressing API and sculpt flows? Inspect the four modified server modules, run
focused retrieval checks plus the applicable gates, and record whether this implementation is a
stable input to the decomposition experiment or which narrow defect still blocks it.

## Comments

- Claimed for implementation after the user approved the tracer-bullet breakdown via `$implement`.

## Answer

The element retrieval baseline is stable for the experiment-facing CPU paths.

- Retrieval now rejects skeletal crops, aspect mismatches above 4x, and source-relative height
  mismatches above 3x; eligible denser geometry is preferred monotonically. The production library
  retains 136 eligible towers, 118 domes, and 72 combined chimney/roof structures.
- Solidity-cache reuse validates shape, range, and freshness against the library, rebuilding
  atomically when the library changes instead of crashing or using stale values.
- Element provenance now exposes library id, source type/building, source-relative height, and
  solidity through the interpretation API.
- Real elements and the ordered edit suffix beginning with the first element are realized
  analytically at output resolution. Add/subtract semantics, mixed-operation order, undo/round-trip,
  town neural render, textured town export, bake, relief, and single-building preview share this
  contract without a 64^3 element pre-bake.

Verification:

- Focused retrieval/realization suite: 8/8 passing.
- Production-library public handlers: retrieved tower `lib_id=995`, source-relative height `0.294`,
  solidity `0.1343`, and 1,005 newly visible output-resolution voxels.
- Production scale-bound sweep: 16 seeds at each of six requested relative heights; every selected
  tower remained within the declared 3x bound.
- All changed Python modules compile; `git diff --check` is clean.
- Independent Standards and Spec reviews report no remaining actionable findings.

The full branch and sculpt-flow suites were not runnable in this sandbox: no CUDA driver is exposed
(`nvidia-smi` cannot communicate with the driver), while those suites require a live CUDA service and
include Stage 3a/SDXL flows. The focused public handler checks cover the changed interpretation and
detail-volume behavior; the full GPU gates remain a residual environment-dependent verification.
