# #179 — Build the guided-edit completion proxy and re-verify locality on it

*Effort: solid-first semantic architectural carving. Opened 2026-09-06, built 2026-09-09. Part of
[#1](https://github.com/danvisai/SDFusion/issues/1). Implements the H1b guided-edit-proxy protocol
[#8](8-falsifiable-proof.md) decided.*

> A headless test harness for H1's guided-edit half — no gesture/UI layer exists or is being built
> here, per #3's own "not-yet-specified" decision. For a sample of held-out buildings, take #10's
> already-recovered program and synthesize a "rough gesture": a box-shaped add or subtract volume,
> footprint-column-local, at several sizes and positions on top of the current solid. Re-run #10's
> constrained beam-search fitter on the gesture-modified target to produce a completed program.
> Score exactly two things per completion: does it pass #7's finalize-time gate, and does #144's
> edit-locality invariant hold on this specific re-fit path.

Code `scripts/foundations/guided_edit_completion_proxy.py`, tests
`scripts/foundations/test_guided_edit_completion_proxy.py` (19 cases, synthetic/corpus-free).
Artifact `execution/artifacts/guided_edit_completion_proxy.json`. Run on 40 carve-needing
buildings sampled (seed 179) from #10's own `program_recovery_714.json`, 3 box sizes × 2
positions × 2 modes (`add`/`subtract`) ≈ 12 gestures/building → **461** distinct gesture/building
pairs, `beam=12 branch=10` (#10's own bar-passing settings), op budget = the pre-gesture
program's own op count + 4.

---

## Method, restated precisely

For each sampled building: take #10's already-recovered program (up to K=4 ops), replay it to
get `h_prog` — "the current solid" — and its own per-op *contribution* masks (the columns each op
actually lowered, replaying exactly as `replay_program` does, not each op's *declared* region —
load-bearing for `CutRoof`, which carries no `region` field at all and would otherwise never count
as "outside" anything). A synthetic gesture is a box, sized as a fraction of the footprint's own
bounding box (0.12/0.22/0.32), placed at up to 2 random positions, that raises (`add`, capped at
the envelope) or lowers (`subtract`, floored at 1) `h_prog` by 3 voxels inside the box only —
footprint-column-local by construction, never the whole footprint. The gesture-modified target is
handed straight back into `fit_program_beam` (#10's own fitter, unmodified) to produce a completed
program from scratch.

**Locality is checked both directions**, not just "does every completed op have a match" — #8's own
wording is one-directional, but its closing clause ("any op outside that region that changed is a
reported failure, **not silently ignored**") would miss an original op the re-fit's own op-budget
silently drops (nothing left in the completed program to iterate over and catch it). So: build the
set of *outside-gesture* op signatures (kind + rounded scalar geometry + exact contribution-mask
bytes) for the pre-gesture program and for the completion, and any signature on one side but not
the other is a reported locality failure.

---

## 🔑🔑 The gate always holds; locality holds on **31.7%**

| check | pass rate | 95% CI | n |
|---|---|---|---|
| #7 gate (`finalize_problems` **and** `containment_problems`) | **1.000** | [1.000, 1.000] | 461 |
| #144 locality, re-checked on this re-fit path | **0.317** | [0.275, 0.358] | 461 |

n=461 is well above this package's own undersampled threshold (30), and the CI is tight — this is
not a small-sample artifact. **H1b's validity axis holds; its locality axis does not.**

The gate runs BOTH of #7's checks, not `finalize_problems` alone — CONTEXT.md's own standing note
is explicit that "the footprint-adherence claim requires both checks on compiled geometry," since
`containment_problems` is a separate function #145/#146 never merged into `finalize_problems`. The
result is close to a foregone conclusion given #10/#146's containment-by-construction design (a
fitted height can never leave the footprint's envelope, and always claims the ground-level
perimeter since every column clamps to at least height 1) — reported anyway because #8's own bar
asks for it, and because it rules out a confound: the locality result below is not explained by
broken completions.

## Root cause: search-order **reallocation**, not op-budget starvation

Only **1 of 461** rows exhausted its op budget (`max_ops_margin=4` is not the bottleneck). Of the
315 locality failures, **295** involve an op that *appears* outside the gesture region with no
byte-identical match in the pre-gesture program (a *new*, unexplained change), against **127** with
a pre-gesture op *missing* from the completion (107 rows show both). The dominant failure mode is
the re-fit **restructuring** an unrelated part of the building, not merely running out of budget to
redo it.

Mechanistically: `_layer_candidates`/`_ramp_candidates` rank candidates by raw gain across the
**whole footprint** at every beam step, not per-region. Introducing the gesture's own local surplus
adds a new competing candidate into that global ranking, which can change *which* candidate wins a
beam slot at an early step — and once the search diverges there, it can settle on a different (but
equally valid, equally low-residual) decomposition of an area whose **target height never changed**.
A concrete, representative case (id 3305, `fp_area` 1535): the pre-gesture program's second `Layer`
covers 1386 of 1535 footprint cells (90%) at one height — with almost any small gesture box
overlapping that dominant region by construction, the re-fit sometimes **splits** it into two or
more smaller `Layer`s, one of which lands entirely outside the gesture and has no match in the
original program.

⚠️ **This means a full re-fit is not, on this evidence, a locality-preserving guided-edit
completion strategy** at these settings — 2 times in 3, some part of the building the user did not
touch changes anyway. Pass rate does not trend cleanly with the pre-gesture program's own op count
(1 op: 0.308, 2 ops: 0.456, 3 ops: 0.208, 4 ops: 0.283 — see the artifact for the full breakdown),
consistent with a structural property of global candidate competition rather than a simple
"more ops = more starvation" effect.

## Verdict, per #8's own per-axis shape

- **H1b / validity: SUPPORTED.** Every completion the harness produced was well-formed and
  contained (100%, tight CI, n=461).
- **H1b / locality-on-refit: FALSIFIED as implemented.** The re-fit path does not preserve edit
  locality reliably (31.7%, tight CI well below any reasonable "holds" bar, n=461). #144's own
  structural proof (locality under `remove_by_id`, direct algebra edits) does **not** transfer to
  this different code path, and this ticket's whole reason for existing was to check that rather
  than assume it — confirmed negative.

## Scope and disclosed limitations

- Uses #10's own pinned 714/`program_recovery_714.json` (K=4, beam=12/branch=10), not #153's newer
  region/tile-stratified split — #8 sequences the stratified split onto H1a's five-arm scorecard
  (#181) specifically; nothing in #179's own spec blocks it on #153, and reusing #10's own
  already-recovered programs is exactly "take a held-out building's program already recovered by
  #10" as written.
- Gesture parameterization (box fractions, 2 positions, `dy=3`, both modes) is this ticket's own
  methodology choice, left unspecified by #8 on purpose (see #8's "What this ticket explicitly does
  not decide") — disclosed here, not hidden: `--box_fracs`/`--n_positions`/`--dy` on the CLI.
- Sampled from the carve-needing (`n_ops > 0`) population only, matching #10's own precedent of
  reporting the already-flat 42% majority separately: an empty pre-gesture program makes the
  locality check vacuous, not a meaningful test of this question.
- No gesture-accuracy/IoU score, by #8's own explicit decision (no real "intended" shape a rough
  box implies; see the module docstring and `NOVELTY_SURVEY.md` risk 5).
- `max_ops_margin=4` is generous by measurement (1/461 rows capped) but was not swept — a much
  larger margin was not tried and might trade locality for op count differently; not measured here.

## What follows

- The dominant failure mode (global candidate competition restructuring unrelated regions) points
  toward a **locally-constrained re-fit** — one that holds the pre-gesture program's own ops fixed
  outside some margin around the gesture and only searches inside it — as the natural next design if
  a real guided-edit feature is ever built. Not scoped or attempted here (#179 is a proxy
  measurement, not a fix), and no ticket exists for it yet.
- [#2](https://github.com/danvisai/SDFusion/issues/2) (the integration boundary) was already
  unblocked by #8 directly; this result is additional evidence for it, not a new blocker.
- H1b's scorecard row (validity: SUPPORTED, locality-on-refit: FALSIFIED as implemented) is ready
  for #8's own cross-hypothesis rollup once H1a (#181, blocked by #153) and H3 (#180) land.
