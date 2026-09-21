# #180 — Test multi-footprint coordination against real product footprint sets

*Effort: solid-first semantic architectural carving. Opened 2026-09-06, built 2026-09-09. Part of
[#1](https://github.com/danvisai/SDFusion/issues/1). Implements the H3 evaluation protocol
[#8](8-falsifiable-proof.md) decided; [#9](9-multi-footprint-coordination.md) built the
coordination mechanism and explicitly deferred its evaluation here.*

> Using footprint sets as they actually occur in the product (drawn or imported together via
> `town_generate_service.py`'s town demo), not a synthetic assembly of held-out corpus rows, run
> #9's block-coordination bias across each of the four live axes (height rhythm, roof family,
> setbacks, orientation) independently, one axis at a time, and combined. Score exactly one thing:
> does every footprint in the block still pass #7's finalize-time validity gate after the
> coordinated re-fit?

Code `scripts/foundations/block_coordination_real_footprints.py`, tests
`scripts/foundations/test_block_coordination_real_footprints.py` (23 cases). Artifact
`execution/artifacts/block_coordination_real_footprints.json`. Traces
`outputs/block_coordination_traces/<scene>/<footprint>/step*_view*.png` (#147, reused unmodified).
Run on 8 real block scenes, 55 footprints total, 6 conditions each (an explicit unbiased baseline +
4 axes + combined) = **330** footprint-condition pairs.

---

## Where the footprints come from

Not corpus rows relabeled as a block: every footprint here is extracted from the two bundled
**real-place** samples the town editor's own image-import already ships —
`scripts/server/web/samples/munich_oldtown.png` (Munich Altstadt, 29 buildings — the same preset
`town_generate_service.py`'s own `MAX_BUILDINGS` comment cites) and `lafayette.png` (32 buildings) —
run through `footprint_image.extract_footprints`/`to_meters`, the exact function the product's
image-import calls. The third bundled sample, `synthetic_blocks.png`, is excluded on purpose: it is
literally named for being one, and #180's own bar is "not a synthetic assembly."

Each image's own extracted layout is split into 2x2 spatial quadrants (median centroid split, real
adjacency — wherever an extracted building's own centroid falls, never a hand-picked grouping),
kept only if 3-8 footprints land in it. All 4 quadrants of both images cleared that bar: **8 block
scenes**, 55 footprints, 5-8 per scene. Every footprint is rasterized to `(fp, y0, y1)` through
`town_generate_service.py`'s own, unmodified `_footprint_normalization`/`_rasterize_footprint`/
`_height_voxel_range` — the identical conversion the product already applies to a drawn/imported
polygon, at its own `default_height=12.0`. One real extracted contour rasterized to two disconnected
components at RES=64 (a thin neck some real building outlines actually have) and was excluded rather
than crashing three calls deeper in `mask_to_rings`'s own one-polygon rule.

## The placeholder massing, and a real methodological correction

A raster import carries no existing massing decision, so every footprint needs *some* starting
program before "coordination" (a re-fit) means anything. #181/H1a owns the generation-quality
question, so this ticket's own bar (does coordination preserve validity) only needs a deterministic
placeholder — not a claim about what a real generator would produce.

⚠️ **The first version of this placeholder was a single, spatially uniform flat `Layer`, and it
produced a nearly vacuous test.** Against a perfectly flat target, `_current_target`'s own re-fit
has exactly one optimal shape regardless of which axis is biased: a `Ramp` "tightest plane over a
flat target" *is* that flat plane, `setback` has only one candidate shape to measure, and
`roof_family`/`azimuth` have nothing non-flat to act on. Measured directly: the first full run
produced the **identical single-op program on all 55/55 real footprints under every one of the 5
conditions** — the gate held 100%, but because nothing distinguishable from an unbiased re-fit ever
ran, not because coordination was tested under real pressure.

**Fix:** the placeholder is now TWO flat `Layer`s over a real split of the footprint (along
whichever of rows/columns its own bounding box spans more, at its own median — real geometry, not
an arbitrary axis choice), cut to two genuinely different depths. A half that rasterizes to more
than one connected piece (a concave real footprint split by a straight line) becomes one `Layer`
entry per piece, at that half's shared height (`_layer_candidates`'s own convention). Re-run: every
one of the 55 footprints' *combined*-condition program now compiles to **2 ops**, not 1 — confirmed
via the rendered traces (440 PNGs: 55 footprints x 4 views x 2 ops, against the first run's 220 at
4 views x 1 op) — a genuinely non-trivial re-fit is now happening on every footprint, not a
structurally-guaranteed no-op.

## Result — the gate holds, 330/330

| condition | footprints | pass |
|---|---|---|
| unbiased (explicit baseline) | 55 | 55 |
| height_rhythm | 55 | 55 |
| roof_family (`"ramp"`) | 55 | 55 |
| setback | 55 | 55 |
| azimuth | 55 | 55 |
| combined | 55 | 55 |
| **total** | **330** | **330 (100%), 95% CI [1.000, 1.000]** |

n = 8 block scenes (above this package's own undersampled threshold), 55 distinct footprints, 330
footprint-condition pairs, **zero regressions**. "Regression" is measured directly, not inferred: an
explicit `"unbiased"` condition (`commit_block_program` re-fit through an empty `FitBias`) is the
literal "uncoordinated fit" #180's own acceptance criterion names, and every later axis/combined
failure is checked against that SAME footprint's own unbiased result before being flagged a
regression (`aggregate`'s `is_regression` field) — none needed to be, since nothing failed at any
condition, on any footprint, in any of the eight sampled block scenes from two source layouts.

The reported bootstrap resamples 330 footprint-condition flags, not independent scene clusters.
Repeated conditions share footprints and scenes. All flags are 1, so [1,1] is a degenerate
empirical bootstrap interval, NOT certainty that unseen blocks cannot fail. Scope the result to
the observed 8 scenes / 55 footprints and this placeholder-massing protocol.

**This is #7's FULL bar, not half of it under the same name** — per this ticket's own reconciliation
note, `commit_block_program` calls `finalize_problems` only (`program_problems`/`commutes`/
`is_height_map_representable`); `containment_problems` is a separate function it does not call.
Both are run here, against each committed occupancy, and reported as one `gate_pass`. Confirmed by
a dedicated test (`test_containment_is_checked_in_addition_to_finalize_problems`) that forces
`commit_block_program`'s own report clean while `containment_problems` is made to fail, and checks
`run_block_scene` still reports the failure — the wiring is exercised, not just present in the
source.

## Why 100% is not surprising, and is still worth having measured

#9's own `FitBias`/`_select` design is explicitly **soft by construction**: a bias "can never make
the fitter prefer a candidate that removes dramatically less surplus" (`_select`'s own docstring),
and every coordinated footprint is re-fit independently through the identical, already-safe #10
fitter that produces `finalize_problems`-clean, contained-by-construction output regardless of bias
(see [#179](179-guided-edit-completion-proxy.md)'s own confirmation that this fitter family is
containment-safe by construction). A validity regression under coordination would have meant the
bias mechanism itself broke that guarantee — worth measuring directly on real, structurally diverse
footprint geometry rather than assumed from the mechanism's design, and it did not.

## Scope and disclosed limitations

- No coordination-consistency statistic is computed, by #180's own explicit decision — "did the
  shared axis actually get more consistent" is qualitative, left to a human reviewing the rendered
  carving trace.
- Traces are rendered for the **combined** condition only (one trace per footprint, not five) — a
  disclosed scoping choice; #180's own bar is that a trace exists per block scene for review, not
  that every condition gets its own.
- `height_rhythm`/`setback`/`azimuth` use one representative value each (drawn from the scene's own
  placeholder heights where that matters, per #9's own "a level the footprint can actually reach"
  finding) — #180 tests whether coordination stays valid under a real bias, not which bias value is
  best, so a sweep was out of scope.
- The placeholder massing is a deterministic two-level cut, not a generated program — #181/H1a owns
  whether autonomous generation itself is good; this ticket's own bar is coordination validity, and
  a placeholder with genuine (if simple) internal shape is enough to make that a real test, per the
  correction above.
- 8 scenes / 55 footprints is a real but modest sample, bounded by how many buildings the two
  bundled real-place images contain after quadrant-clustering; not flagged as undersampled against
  this package's own threshold, but a larger real-place corpus would sharpen it further.

## Verdict, per #8's own per-axis shape

- **H3: SUPPORTED.** Every one of 330 footprint-condition pairs, across 8 structurally different
  real block scenes, an explicit unbiased baseline, all 4 coordination axes, and the combined
  program, held #7's full gate (finalize + containment). No regression found.

## What follows

- [#2](https://github.com/danvisai/SDFusion/issues/2) (the integration boundary) was already
  unblocked by #8 directly; this result is additional evidence for it.
- H3's scorecard row (SUPPORTED) is ready for #8's own cross-hypothesis rollup once H1a (#181,
  unblocked now that #153 is closed) lands. H1b ([#179](179-guided-edit-completion-proxy.md)) is
  already closed: validity passed, locality-on-refit was falsified as implemented.
