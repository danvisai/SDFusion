# Build the Neutral Facade and FID Harness

Type: task
Status: resolved
Blocked by: 01, 03

## Question

Implement identical neutral-shader facade rendering and FID measurement for real, monolith, and
decomposition SDFs at the locked resolution. Establish camera determinism, representation parity,
feature-extractor provenance, and a real-vs-real split sanity baseline before using the harness for
either claim.

## Comments

## Answer

**Built:** `scripts/eval/fid.py` (Fréchet distance + pinned `InceptionExtractor` + group-aware
bootstrap CI), `scripts/eval/render_facades.py` (deterministic orbit cameras + neutral normal-shader
render via the existing SDF sphere-tracer + `resample_sdf_grid`), `scripts/eval/sanity_real_vs_real.py`
(the real-vs-real sanity CLI), `scripts/eval/test_fid.py` (22 contract tests, pure numpy/scipy, no
GPU). Rendering is arm-agnostic by construction — `render_sdf_neutral` takes any cube-frame SDF grid,
so the same function will render the monolith (ticket 07) and decomposition (ticket 08+) arms once
those tickets produce data; no source-type branching exists to violate representation parity.

**Standards + Spec review (two-axis, run against the initial implementation) — findings and fixes:**
- *Standards:* dead code — `render_building_set` (unused, duplicated by the sanity script's grouped
  helpers) → removed. Naming — `math_hypot` → `vec_norm`. `linalg.sqrtm(..., disp=True)` fallback
  path (deprecated arg, prints on inaccuracy) → `disp=False`. Correlated multi-view bootstrap treated
  as independent samples → `bootstrap_fid_ci` gained `groups_a`/`groups_b` for GROUP-level resampling
  (resamples whole buildings, never a partial group; tested).
- *Spec:* **resolution-parity bug** — real BuildingNet SDFs load at their native 64³, but ADR 0004
  locks the shared working resolution at 96³ for every arm; comparing a 64³-quantized real field
  against 96³-realized generated arms would confound genuine detail differences with sampling-density
  differences. Fixed: `resample_sdf_grid` trilinearly upsamples to `WORKING_RES=96`
  (`align_corners=True`, matching `sphere_trace`'s own `grid_sample` convention) —
  `load_buildingnet_sdf` now resamples by default. Also fixed: extractor provenance was the floating
  `Inception_V3_Weights.DEFAULT` alias (could silently repoint) → now also records the resolved
  concrete checkpoint URL (`weights_url`, captured lazily on first load to keep imports cheap).
  Failure accounting and full run provenance (camera params, git rev, package versions) were missing
  from the sanity output → added.

**Two more bugs caught DURING verification itself (the exact purpose of "establish ... before using
the harness"):**
1. **Resolution conflation.** The sanity script's own `--res` flag fed BOTH the SDF voxel grid
   resolution (the ADR-0004-locked, research-critical variable) AND the rendered image PIXEL
   resolution (a rendering-quality knob, independent of voxel density — sphere-tracing samples the
   SDF continuously). Caught mid-render before it produced a silently-degraded number. Fixed: split
   into `--sdf-res` (default 96) and `--img-res` (default 256).
2. **BLAS thread-oversubscription stall.** The render appeared hung (12+ min, no output, ~3300% CPU,
   121 threads on a 40-core node) with `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS`
   unset — numpy/scipy/torch's BLAS backends each spawned a thread pool sized to `nproc`
   independently. Confirmed via `/proc` (state R, no blocking I/O, thread count) and a timed tiny run
   (0.2–2.3 s/building once capped vs. no completion after 12 min uncapped). Fixed: capped to 4 via
   `os.environ.setdefault` at the top of `sanity_real_vs_real.py` (before numpy/torch import; respects
   an explicit caller override) — a 48-building run then completes in ~2 minutes.

**A genuine statistical finding, not a bug — FID small-sample/high-dimension bias.** At the sanity
scale (48 buildings, 144 images/half), the point estimate (132.96) fell **outside its own 95%
bootstrap CI** ([152.6, 183.5]). Reproduced on synthetic same-distribution data (true FID = 0) at the
identical shape (24/60/120 groups × 6 correlated views, 2048-d features): estimate stayed in the
thousands and did **not** shrink toward 0 proportionally up to 720 samples, because every tested N
stayed below the 2048-d feature dimensionality (rank-deficient covariance estimate) — consistent with
Heusel et al. 2017's ≥10,000-sample recommendation for 2048-d Inception FID. **Not a code bug**
(confirmed on synthetic data, independent of the render/extraction pipeline) and **not fixed by
resampling** — it is fixed only by more samples. Added a permanent guard: `fid.undersampled(a, b)` +
a `warnings.warn` in `fid_from_features` (fires once per `bootstrap_fid_ci` call, not per bootstrap
iteration) whenever either set has fewer rows than the feature dimensionality; the sanity output now
records `undersampled_for_reliable_fid` explicitly. **Recommendation carried to tickets 07/08:** the
headline C1/C2 FID comparisons must use substantially more than 2048 images per arm (more views per
building and/or more of the 277-building test set) before the FID number is trustworthy — this
sanity run's `132.96` is a pipeline-correctness check, not a metric floor to compare against.

**Final verified sanity run** (`execution/artifacts/fid_sanity.json`): 48/48 buildings rendered, 0
failures, `sdf_working_res=96`, `image_res=256`, `render_deterministic=true`,
`bootstrap="group-level (per building)"`, `undersampled_for_reliable_fid=true` (expected at this
scale, see above), pinned provenance (weights URL, git rev, package versions) recorded.

Unblocks the C1 transform-vs-noise experiment and C2's headline FID comparison (tickets 07/08),
carrying the minimum-sample-size finding forward.
