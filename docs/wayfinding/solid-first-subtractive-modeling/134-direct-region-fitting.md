# #134 — Fit few-vertex regions directly instead of trimming exact rings

*Effort: solid-first semantic architectural carving. Opened 2026-09-01 from
[#131](131-vertex-budget.md), which left open whether a fitter that searched few-vertex regions
directly would land somewhere better than trimming. Run and written 2026-09-03. CPU only, no
training, no GPU.*

> A fitter that searched 8-vertex regions *directly* would place its vertices differently and
> could land anywhere between these rows and the exact one.

**KILLED.** At the pre-registered 24-vertex cap, the direct fitter's `spiked` is **0.9173** —
*worse* than the trimmed row's **0.8954**, not better, and the pre-registered bar asked only that
it beat that number. It does not, at 24 vertices or at any other budget tested. 🔑 The cause is
diagnosed, not a mystery: the growth strategy plateaus **far** short of its own budget on nearly
every region in the corpus, because its one admission test — is this vertex a convex "ear" against
the polygon's *current, coarse* shape — is the wrong local criterion for a concave boundary. Per
#134's own pre-registration, the base-`Layer` floor is the remaining route, and it was run here as
the required confound control: it cuts `spiked` from 0.895 to **0.482**, but at a real, measured
fidelity cost the naive floor choice does not pay for free.


## What was built

A `direct` rule beside `contained`/`lossless`/`free` in `simplify_region`
(`scripts/foundations/recover_massing_programs.py`), plus a `program_floor`/`replay_program(...,
floor=)` confound control:

- **`_seed_triangle`** — the coarsest possible start: one "ear" (three consecutive ring vertices
  whose triangle is entirely inside the exact region). Ear-clipping theory guarantees at least one
  exists for any simple polygon, so this is a property, not a heuristic that can fail.
- **`_fit_outer_ring_direct`** — grows the outer ring from that seed by repeatedly inserting the
  ring vertex that recovers the most currently-uncovered area, subject to the *identical*
  cell-level containment test `contained` already uses for deletion, applied here to insertion.
  Each step's kept vertices are a strict superset of the previous step's — RADmesh's "carry the
  fit forward across re-discretization," by construction rather than extra bookkeeping.
- **Holes** are reduced to their own lossless floor (free, no fidelity cost) and left ungrown; a
  documented scope decision, not an oversight (below).
- **`program_floor`** — the lowest height any `Layer` op in a recovered program specifies, a
  per-building, data-derived stand-in for #131's own untested "base `Layer` under the cascade."
  `replay_program` grew an optional `floor` parameter: an uncovered column now starts there
  instead of at the full envelope extent, with `floor=None` (the default) exactly reproducing
  every caller's existing behaviour.

Both are wired into the existing measurement path (`_budget_case`, `measure_vertex_budget`,
`report_vertex_budget`) so the rows land beside #131's own table, directly comparable.


## The measurement

n = 411 carve-needing buildings (#126 subset), the same pinned set and the same `replay_program`
scoring #131 used. `direct{v}` runs the new fitter at each of #131's own budgets; `floor{v}`
re-scores the *existing* `inner` (`contained`) trim with the floor applied, isolating the floor's
own effect from the search's, per #134's "confound to control first." Artifact:
`execution/artifacts/program_recovery_714_vertex_budget_134.json`; command:
`scripts/foundations/recover_massing_programs.py --vertex_budget`, then `report_vertex_budget` on
the result (both wired through the existing CLI path). 183 s on 62 cores.

| budget | arm    | verts | tokens | missing | extra  | vs_input | collapse | ops | planar | spike | spiked | contained | met  |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exact  | -      | 284   | 578    | 0.0000  | 0.0030 | 0.8221   | 0.0000   | 2.0 | 0.50   | 1     | 0.397  | 1.0000    | 1.00 |
| needed | -      | 163   | 342    | 0.0000  | 0.0030 | 0.8221   | 0.0000   | 2.0 | 0.50   | 1     | 0.397  | 1.0000    | -    |
| 4      | inner  | 16    | 44     | 0.0000  | 0.0856 | 0.8856   | 0.0000   | 2.0 | 0.50   | 18    | 0.920  | 1.0000    | 0.84 |
| 4      | floor  | 16    | 44     | 0.0908  | 0.0079 | 0.7500   | 0.3966   | 1.0 | 0.00   | 4     | 0.504  | 1.0000    | 0.84 |
| 4      | direct | 16    | 44     | 0.0000  | 0.2068 | 0.9870   | 0.0000   | 0.0 | 0.00   | 18    | 0.920  | 1.0000    | 0.84 |
| **24** | **inner** | 72 | 161    | 0.0000  | 0.0182 | 0.8463   | 0.0000   | 2.0 | 0.50   | 15    | **0.8954** | 1.0000 | 0.94 |
| **24** | **floor** | 72 | 161    | 0.0908  | 0.0030 | 0.7389   | 0.3966   | 1.0 | 0.00   | 3     | **0.4818** | 1.0000 | 0.94 |
| **24** | **direct** | 32 | 76    | 0.0000  | 0.0932 | 0.9027   | 0.0000   | 2.0 | 0.50   | 18    | **0.9173** | 1.0000 | 0.90 |
| 94     | inner  | 214   | 442    | 0.0000  | 0.0046 | 0.8253   | 0.0000   | 2.0 | 0.50   | 7     | 0.635  | 1.0000    | 0.99 |
| 94     | floor  | 214   | 442    | 0.0908  | 0.0002 | 0.7331   | 0.3966   | 1.0 | 0.00   | 1     | 0.285  | 1.0000    | 0.99 |
| 94     | direct | 32    | 76     | 0.0000  | 0.0901 | 0.9002   | 0.0000   | 2.0 | 0.50   | 18    | 0.917  | 1.0000    | 0.98 |

Full table (all seven budgets, `free` arm included) is in the artifact and reproducible via
`report_vertex_budget`. The 24-vertex row is the pre-registered comparison point.


## 🔑🔑 Why it plateaus: the ear test is the wrong local criterion for a concave boundary

`direct`'s own `verts` column is the tell: **32 at budget 24, and still 32 at budget 94** — the
fitter is not spending the budget it is given, at any of the three largest tested budgets. Checked
directly on a single 60-vertex region (building 11807): grown to completion at budget=94, it stops
at **12 vertices**, leaving **160 cells uncovered** — `contained` holds throughout (0 cells ever
wrongly added) but `missing`-from-the-simplified-region-vs-lossless is large. This is not one
unlucky region: over the whole corpus, **99.5% of buildings** have `direct94`'s vertex count
*strictly below* what their own lossless floor needs (median 32 against a lossless median of 163,
against an exact median of 284) — the fitter is stopping at roughly a fifth of the vertices it
would need, regardless of how much budget it is handed.

The mechanism is structural. Each insertion is admitted only when the candidate vertex forms a
convex ("ear") triangle *against the polygon's current, coarse chord* — but a genuinely concave
notch in the true boundary looks reflex relative to a wide chord spanning it, precisely because the
chord is already cutting straight across the notch from a coarse vantage point. Individual vertices
inside that notch can never look convex until *other* vertices near them are already in place, and
this greedy, one-ear-at-a-time search never finds an order that reaches them. RADmesh's own
mechanism has no analogous failure mode because it optimizes a *continuous* objective with no
binary per-vertex admission test to get stuck behind; the translation to a discrete,
containment-gated insertion search introduces exactly the kind of local optimum this run exposes.


## The confound, controlled: the floor helps, and is not free

#131 diagnosed the spike as structural, not a fitter defect: a column no operation covers reverts
to the *full envelope height* because `replay_program`'s cascade starts there. Applying
`program_floor` to the *existing* `inner` trim alone — no change to the fitter at all — moves
`spiked` from **0.8954 to 0.4818** at 24 vertices, and to **0.285** at 94 (against inner's 0.635).
🔑 This confirms #131's own diagnosis directly: most of the spike **is** the missing floor, not
the trimming strategy.

⚠️ **It is not free.** `missing` moves from 0.0000 to **0.0908** and `collapse` from 0.0000 to
**0.3966**, flat across every budget. The floor used here — the lowest height any `Layer` op in
the *program itself* specifies — is a real, data-derived value, not an arbitrary constant, but it
is still frequently *below* what some abandoned columns genuinely needed, and flattening them to it
cuts into real geometry the exact ring had. The floor trades one failure mode (a spike standing
proud of GT) for another (a cut running below it) rather than eliminating the trade-off outright.
A better-chosen floor — per-column, or per-operation, rather than one scalar per building — is the
natural next question this leaves open, not a re-run of this one.


## GUARD held

`contained` = **1.0000** and `missing` = `collapse` = **0.0000** for `direct` at every budget from
4 to 94 — the new fitter never violates #10's containment guarantee, regardless of how far short of
budget it stops. `scene/test_sdf_edit.py::TestVertexBudget` — the pinned plain-shed check among
them — passes unchanged, 7/7. `scripts/foundations/test_recover_massing_programs.py` (16 new tests:
containment on concave/holed/staircase shapes, the plain-shed and exact-diagonal checks restated
for `direct`, the coarse-to-fine "carries the fit forward" property, and the floor's own
regression/clamp behaviour) passes, 16/16.


## What this settles

1. **Direct search, as specified here, is not the lever.** The pre-registered KILL fired: at 24
   vertices `direct`'s `spiked` (0.9173) does not beat `inner`'s (0.8954). #131's own fallback
   applies: the base-`Layer` floor is the remaining route, not a further search over this design.
2. **The floor is real and worth pursuing, but the naive version costs real fidelity.** A single
   scalar floor per building is not the final answer #131 left open; it is evidence the mechanism
   works, priced honestly rather than left untested.
3. **The failure mode is diagnosed, not just measured.** An ear-insertion test against the
   *current* coarse chord is provably the wrong local criterion once a boundary has real
   concavity — which nearly every recovered region does (#131: 15.7% carry holes; concave outer
   boundaries are the common case, not the exception). A future direct-fitting attempt needs a
   growth rule that does not require convexity against an intermediate, still-coarse shape.


## What this does not settle

- ⚠️ Whether a **differently-ordered** growth search — one that does not require each insertion to
  look convex against the current coarse chord, e.g. one that can accept a temporarily
  concave-looking step because a later insertion will resolve it — reaches further. That is a
  different algorithm, not a parameter of this one.
- Whether a **per-column or per-region floor**, rather than one scalar per building, closes the
  missing/collapse cost the floor control measured here without giving back the spike win.
- Whether growing **holes** too, with a correctly signed incremental update, changes the picture
  materially — #131 already found holes are usually near their own floor regardless, so this is
  expected to be a second-order question, untested here.
- The measurement is at 64³, matching #131 and #126; nothing here re-examines that choice.

See [131-vertex-budget.md](131-vertex-budget.md), [10-program-recovery.md](10-program-recovery.md),
[126-massing-scoring.md](126-massing-scoring.md).
