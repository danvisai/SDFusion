# #178 - Rebuild a BuildingWorld-specific 1-NN baseline and pre-register its bar

*Effort: fold BuildingWorld into the massing corpus (arm six on the gable bar), map
[#156](https://github.com/danvisai/SDFusion/issues/156). `ready-for-agent` label. Blocked by
[#177](https://github.com/danvisai/SDFusion/issues/177) (held-out split), #170 (falsification
rule), #168 (sampling-cap policy) - all three closed. Blocks
[#183](https://github.com/danvisai/SDFusion/issues/183) (the retrain ladder), which is explicitly
"blocked by #178 landing."*

> Rebuild the 1-NN retrieval baseline against the new train bank (now including BuildingWorld) and
> compute its score on the new BuildingWorld held-out set. Pre-register this as a separate bar from
> the pinned-714 bar, following #170's falsification rule, with per-city and per-roof-family
> breakdowns. Keep the pinned-714 bar completely unchanged as a fixed regression control.

## What this ticket builds

`scripts/foundations/buildingworld_baseline.py`. Train bank: every `held_out=0` ledger row
(1,530,908 rows) in ascending corpus-row order, no source cap or reweighting, per #168. Query
population: the full held-out slice of #170's nine named cities (Berlin, Boston, Cambridge, Cape
Town, Edmonton, Melbourne, Montreal, New York, Tokyo) - 20,720 buildings. For each query, exact
footprint-IoU 1-NN retrieval against the train bank (`exact_retrieval`: a packed-bitmap exact-match
shortcut before falling back to `train_height_map_generator.retrieve_nn`'s chunked IoU argmax),
then `score_arm` on the height-field GT, greedy form fitter at `max_ops=16`. Output:
`execution/artifacts/buildingworld_baseline_178.json` - the pinned-714 control (copied verbatim,
never re-scored), the new BuildingWorld bar (`overall` / `per_city` / `per_family` / `gable_hip`),
and #170's `registration` (the quoted PASS/GUARD/KILL floors).

Run in production: 1,530,908-row bank, 20,720 queries, 8 CPU workers, ~25 minutes
(`meta.elapsed_seconds = 1515.8`).

## `retrieve_nn` had to scale from 35,776 rows to 1.56M

`train_height_map_generator.retrieve_nn` originally materialized the full query-chunk × bank IoU
matrix in one `float32` array - fine at the historical 35,776-row bank, not at 1.53M. Extended
(uncommitted before this ticket, landed alongside it) with `bank_chunk` (scans the bank in blocks,
bit-identical result including tie-break: first/lowest-index bank row wins) and `device` (moves the
intersection matmul - the dominant cost at this bank size - onto a torch device; the CPU-side
`float32` cast of the full bank is skipped when a device is given). Every chunk's IoU still comes
back to CPU before the numpy tie-break `argmax`, so the device path is bit-identical to CPU-only,
pinned by `test_the_gpu_device_path_matches_cpu_exactly_including_ties` (skipped without CUDA).

## The infeasible-floor problem, and the decision that resolves it

#170's rule quotes every floor mechanically from 1-NN's own score on that population - the same
method the original `PROGRAM_BAR`'s `max_extra`/`kill_planar` were quoted from. On four of the nine
per-city floors, that mechanical quoting produces a floor no arm can ever clear:

| city | n | 1-NN `extra` | 1-NN `dl_planar_fraction` | quoted floor | why it's unreachable |
|---|---|---|---|---|---|
| Boston | 2,294 | 0.0 | 0.0 | `max_extra=0.0`, `kill_planar=0.0` | `extra` is nonnegative - `< 0.0` can never be true; any nonzero `extra` fails PASS |
| Melbourne | 156 | 0.0 | 0.0 | same shape | same |
| New York | 413 | 0.0 | 0.0 | same shape | same |
| Cambridge | 5 | 0.0 | 1.0 | `kill_planar=1.0` | `dl_planar_fraction` cannot exceed 1.0 - `> 1.0` can never be true; `clear_kill` always fails |

`preregister()` flags these as `infeasible_floors` and explicitly refuses to relax them - per
#170's own text, "never relax a floor automatically." That leaves a real decision: an arm that is
otherwise excellent still mechanically fails the ladder's "no single city may fail" requirement on
these four, for reasons that have nothing to do with model quality (Cambridge's `n=5` and the other
three's `extra=0.0`/`planar_fraction=0.0` are 1-NN artifacts of small or degenerate held-out
slices, not floors any generative model is expected to clear).

**Project owner's call (2026-09-15):** ship these four as a **GUARD-only ceiling, no PASS
defined** - reported, not blocking. This is implemented as a sign-off step separate from
`preregister()`, not a change to the quoted numbers:

- `apply_signoff(registration, guard_only_floors, note)` - a pure function that marks a
  project-owner-chosen subset of `infeasible_floors` as `guard_only_floors` and flips
  `status` from `pending_owner_signoff` to `signed_off`. It refuses a floor outside
  `infeasible_floors` and refuses a second sign-off on an already-signed registration. The quoted
  `floors` bar is copied through unchanged - sign-off decides which clauses of an already-honest
  quote gate the ladder, never the quote itself.
- `buildingworld_verdict` grades a `guard_only` floor on its two GUARD-shaped bounds only
  (`collapse_rate <= max_collapse`, `vs_input < max_vs_input`); the PASS-shaped clauses
  (`form_ops`, `form_planar`, `beats_extra`, `clear_kill`) are still computed and reported per-city
  for visibility, but excluded from that floor's `pass_` and therefore from the ladder-wide
  `numeric_pass` aggregate. Every other floor keeps all six clauses gating, unchanged
  (`pass_available=True`).
- `signoff_file(path, guard_only_floors, note)` applies this to an already-written artifact's
  `registration` object in place, without repeating the ~25-minute retrieval-and-scoring run.

Applied to the production artifact:

```
python scripts/foundations/buildingworld_baseline.py --signoff \
  --guard-only-floors city:Boston city:Cambridge city:Melbourne city:New York \
  --signoff-note "..."
```

`execution/artifacts/buildingworld_baseline_178.json`'s `registration.status` is now
`"signed_off"`; `registration.guard_only_floors` names the four cities above. The other seven
floors (`overall`, `gable_hip`, and cities Berlin/Cape Town/Edmonton/Montreal/Tokyo) remain fully
gating, unmodified.

## Pre-registered numbers now binding #183

From `execution/artifacts/buildingworld_baseline_178.json` (`registration.floors`):

| floor | max_ops | min_planar | max_extra | kill_planar | max_collapse | gating |
|---|---|---|---|---|---|---|
| overall (9-city pooled) | 2.0 | 0.0 | 0.0262 | 0.0 | 0.1345 | full |
| gable_hip (pitched subset) | 2.0 | 0.2614 | 0.0828 | 0.2614 | 0.0719 | full |
| city:Berlin | 2.0 | 0.2 | 0.0382 | 0.2 | 0.1276 | full |
| city:Cape Town | 2.0 | 0.0 | 0.0779 | 0.0 | 0.1213 | full |
| city:Edmonton | 2.0 | 0.125 | 0.0189 | 0.125 | 0.1067 | full |
| city:Montreal | 1.0 | 0.0 | 0.0322 | 0.0 | 0.1125 | full |
| city:Tokyo | 1.0 | 0.0 | 0.0483 | 0.0 | 0.1445 | full |
| city:Boston | 1.0 | 0.0 | 0.0 | 0.0 | 0.2681 | **GUARD-only** |
| city:Cambridge (n=5) | 1.0 | 1.0 | 0.0 | 1.0 | 0.0 | **GUARD-only** |
| city:Melbourne | 1.0 | 0.0 | 0.0 | 0.0 | 0.2308 | **GUARD-only** |
| city:New York | 0.0 | 0.0 | 0.0 | 0.0 | 0.0920 | **GUARD-only** |

Pinned-714 control, copied verbatim (never re-scored): `nn_retrieval.carve.extra = 0.1031`,
`served_ce_median` (`heightmap_ce.pt`, the frozen deployed checkpoint) reported alongside it - the
regression-guard comparison #183's arm 1 sets its noise band against.

## What this ticket does not do

- **Does not touch `real.h5`, the ledger, or any pseudo-label file** - read-only against #174–#177's
  outputs, exactly as its own module docstring states.
- **Does not decide #183's noise-band tolerance** - that is arm 1's own job, run first and
  pre-registered before arm 2's score is looked at, per #172(b).
- **Does not widen the bar to Mississauga or #169's pooled cities** - #170 scoped the bar to
  exactly the nine cities #159 measured; those five remain explicitly out of scope here, unchanged.
- **Does not revisit the infeasibility itself** - if a future re-split gives Cambridge (or the other
  three) a larger, less-degenerate held-out slice, that is grounds for a follow-up ticket to
  re-quote the floor from scratch, not to retroactively relax this one.

## Status: implemented, unit-tested, run to completion, and signed off (2026-09-15)

`execution/artifacts/buildingworld_baseline_178.json` exists with `registration.status ==
"signed_off"`. 22 tests pass in `scripts/foundations/test_buildingworld_baseline.py`, including
five covering `apply_signoff` and the `guard_only` verdict path. #183 is unblocked.
