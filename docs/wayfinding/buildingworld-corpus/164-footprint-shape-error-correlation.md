# #164 — Does the served model's error correlate with candidate footprint-shape statistics?

*Effort: fold BuildingWorld into the massing corpus (arm six on the gable bar), map
[#156](https://github.com/danvisai/SDFusion/issues/156). `wayfinder:task` — a measurement, not a
channel-set change; `train_height_map_generator.py`'s `CONDITIONING_CHANNELS` is untouched.*

> The tentative design proposes adding footprint-shape-derived conditioning channels (solidity,
> aspect ratio, perimeter^2/area, vertex count) alongside the existing distance-to-edge channel,
> using the same non-leakage justification already used for edt. But "derivable from the footprint"
> is not by itself a reason to add a channel — the raw footprint mask is also derivable from the
> footprint, and the network already has it. `make_model` is a UNet with an 8x8 bottleneck and no
> global-pooling path, so plan-global integrals like footprint area may genuinely be uncomputable
> within its receptive field, or the net may already be picking them up fine. Test this cheaply,
> with no training run: on the pinned 714, using the CURRENTLY SERVED model, measure whether
> per-building error correlates with each candidate statistic. Report the correlation (and scatter)
> for each.

## Method

`scripts/foundations/measure_footprint_shape_correlation.py`. For each of the pinned 714:

- **Error metrics** are read as-is from `execution/artifacts/height_map_generator_class_714.json`'s
  `heightmap_ce_median` arm — confirmed the served model (`docs/PROJECT_STATE.md`: "#127's base
  CE+median model ... is what's served today"; the artifact's `meta.checkpoints` path and the live
  `weights/massing-heightmap/heightmap_ce.pt` are the same file, matching sha256). No training or
  inference was run. Scoped narrowly on purpose: `town_generate_service.py`'s actual town-wide
  default arm is A2 (vecset diffusion, a different architecture); `heightmap_ce_median` is the
  served arm among the height-map-UNet arms specifically (`HEIGHTMAP_ARMS["heightmap_median"] =
  ("ce", "posterior median")`), which is the model this ticket's question about `make_model`'s
  UNet bottleneck is actually about.
  Following this project's own convention (`eval_massing_arms.py`: 3D IoU is always reported split
  into `missing`/`extra`, never as a lone number, because they are opposite failure modes), all
  three of `extra`, `missing`, and `vol_iou` are correlated against, not one cherry-picked scalar.
- **Shape statistics** are computed fresh from each building's binary footprint mask (the same
  `outputs/height_map_generator/height_fields.npz` cache training reads, joined by row id):
  - `solidity` = mask pixel area / convex-hull area, `scipy.spatial.ConvexHull` over foreground
    pixel coordinates — the same formula `train_vecset.py`'s `Bag3dDataset` already precomputes for
    the vecset denoiser's conditioning.
  - `aspect_ratio` = bounding-box `max(h,w)/min(h,w)` of the mask's nonzero extent.
  - `perimeter_sq_over_area` = (Douglas-Peucker-simplified polygon perimeter)^2 / mask pixel area.
  - `vertex_count` = vertex count of that same simplified polygon (`skimage.measure.find_contours`
    + `footprint_image._simplify_corners`, the corner-preserving simplification the town editor's
    image-upload path already uses — not the fixed 16-point uniform resampling `refine.py` uses for
    rendering, which would make vertex count a constant rather than a measurement).
- **Correlation**: Pearson and Spearman (`scipy.stats`), each statistic against each error metric,
  over every building whose mask yields computable shape stats (all 714 did — no exclusions).

Run: `env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/measure_footprint_shape_correlation.py`.
Full per-building records, correlations, and scatter data:
[`execution/artifacts/footprint_shape_error_correlation.json`](../../../execution/artifacts/footprint_shape_error_correlation.json).
Scatter plots: `execution/artifacts/footprint_shape_error_correlation_plots/{solidity,aspect_ratio,perimeter_sq_over_area,vertex_count}.png`.

## Results

714/714 pinned buildings matched, 0 skipped.

| Statistic | vs `extra` (Pearson / Spearman) | vs `missing` (Pearson / Spearman) | vs `vol_iou` (Pearson / Spearman) |
|---|---|---|---|
| solidity | -0.093 (p=.013) / -0.129 (p=5.5e-4) | -0.218 (p=3.8e-9) / -0.109 (p=.0036) | +0.193 (p=1.9e-7) / +0.146 (p=8.6e-5) |
| aspect_ratio | +0.103 (p=.0061) / -0.019 (p=.62) | +0.072 (p=.055) / -0.024 (p=.52) | -0.113 (p=.0025) / -0.010 (p=.79) |
| **perimeter_sq_over_area** | **+0.168 (p=6e-6) / +0.246 (p=2.8e-11)** | **+0.275 (p=7.3e-14) / +0.228 (p=6.7e-10)** | **-0.301 (p=2e-16) / -0.303 (p=1.3e-16)** |
| vertex_count | +0.084 (p=.025) / +0.031 (p=.40) | +0.054 (p=.15) / -0.015 (p=.70) | -0.087 (p=.021) / -0.018 (p=.63) |

Population medians (p10–p90): solidity 1.00 (0.91–1.00), aspect_ratio 1.25 (1.03–1.76),
perimeter_sq_over_area 18.1 (15.9–25.8), vertex_count 6 (4–11).

## Key findings

**`perimeter_sq_over_area` is the one candidate with a real, consistent signal**, on all three
error metrics and by both correlation measures (|r| 0.17–0.30, every p ≤ 6e-6 at n=714). The scatter
(`perimeter_sq_over_area.png`) shows why: it's an upper-bound-shaped relationship, not a linear
one — low-complexity footprints (perimeter²/area near the square's minimum of 16) span the model's
*entire* error range from near-perfect to poor, but no building with a jagged/complex footprint
(perimeter²/area > 30) ever reaches `vol_iou` much above 0.9. Complexity looks like it caps
achievable accuracy without guaranteeing bad accuracy on its own — a pattern a linear correlation
coefficient understates but a scatter shows directly, which is exactly why the ticket asked for
both.

**`solidity` shows a real but weaker signal** (|r| 0.09–0.22, all significant), same direction as
expected (less convex → worse `missing`/`vol_iou`) and, being an inverse-flavored measure of the
same jaggedness `perimeter_sq_over_area` captures, is likely partly redundant with it rather than
an independent second lever.

**`aspect_ratio` and `vertex_count` show no signal worth acting on.** Pearson correlations are tiny
(|r| ≤ 0.11) and, tellingly, Spearman — robust to the extreme-value pull a handful of very elongated
or high-vertex outliers can put on Pearson at n=714 — drops to statistically indistinguishable from
zero for both (p ≥ 0.40 on every pairing except one at p=.021). Whatever these two statistics
measure, it is not something the served model's error tracks.

**This does not by itself answer whether new channels would help**, only whether a signal for them
to learn is present in this population at all. A correlation here is necessary but not sufficient:
the open question the design ticket must still address is whether `make_model`'s 8x8-bottleneck UNet
can already form these plan-global integrals from the mask it's given, in which case adding
`perimeter_sq_over_area`/`solidity` explicitly might still be a wash. This ticket does not run that
ablation — it answers only "is there a statistic worth trying," and for two of the four candidates
the answer is no.

## Scope and disclosed limitations

- Correlational, not causal, and observational at that: these are the same 714 buildings the served
  checkpoint was scored on, not an intervention. A channel-ablation training run (out of this
  ticket's no-training-run scope) is the only way to see whether the UNet is already extracting
  these integrals.
- `perimeter_sq_over_area` and `vertex_count` both depend on the Douglas-Peucker tolerance
  (`--tol-px`, default 1.0 px on a 64x64 mask); not swept here. The qualitative ranking across the
  four candidates is unlikely to flip with a different tolerance, but the exact coefficients would
  shift.
- `solidity`'s convex-hull-over-pixel-centers formula (inherited unchanged from `train_vecset.py`'s
  precedent) systematically reads slightly higher than a true polygon-area solidity for concave
  shapes near a mask's discretisation limit — a known property of the reused formula, not new to
  this measurement, and it uses the exact same code path already trusted for vecset conditioning.
- 714 buildings is the pinned regression population, not the full training/eval corpus; no claim is
  made about whether this correlation structure holds elsewhere.

## Verification

```bash
env -u LD_PRELOAD ./sdfusion/bin/python \
  scripts/foundations/test_measure_footprint_shape_correlation.py
```

11 tests: the four shape statistics against masks of known geometry (square, elongated rectangle,
re-entrant L, degenerate/empty), the correlation arithmetic (perfect correlation, perfect
anti-correlation, the n<3 NaN guard), and the join logic (a missing-row id and a degenerate mask are
both counted as skipped, never silently dropped).
