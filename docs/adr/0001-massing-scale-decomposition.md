# Research thesis: massing-scale decomposition (generate the mass, compose/retrieve the detail)

**Status:** accepted (2026-07-10, grilling session)

## Context
The original goal was to generate a building from a footprint. End-to-end footprint→mesh
generation failed at *detail* scale — three detailizers (REPA, adaLN, L1-GAN) and the Layer-A
context-snap model all produced blurry geometry that could not carve windows. The project then
pivoted to: learned models make *decisions*, but geometry is *realized* procedurally or by
retrieving real BuildingNet element geometry. This ADR records what that pivot **means as a
research claim**, so the pivot reads as a finding rather than a retreat.

## Decision
The research claim is **massing-scale decomposition**: footprint-conditioned generation is
*well-posed for coarse massing but ill-posed for fine detail at achievable data scale*, so the
correct factorization is **generate the mass, compose/retrieve the detail**.

- **Boundary (defined in advance, not post-hoc):** fix a detail scale `s*` (~0.5 m) AND a
  semantic detail set (windows, doors, balconies, cornices, ornament, facade articulation); the
  *finding* is that the semantic-detail set and the sub-`s*` high-frequency band **coincide**.
- **Durability defense:** *data-efficiency + real-fidelity*. In the regime academic and most
  industrial work lives in (~1e3–1e4 real shapes), decomposition beats monolithic generation on
  detail fidelity because (a) the massing target is low-dimensional and learnable from little
  data, and (b) detail is sourced from **real** library geometry, whose fidelity a generator
  cannot match at equal data budget. Concedes that web-scale data may close the gap and argues
  that regime is irrelevant to the real targets (heritage / regional / one-off corpora).

## Considered options (rejected)
- **System/artifact-only** (breadth of the editable pipeline as the contribution) — not
  falsifiable on its own; demoted to the *demo wrapper*.
- **Minimal-sufficient-constraint-set** (footprint alone is underdetermined; {footprint, class,
  style, height} is minimal) — true and possibly a supporting ablation, but not the spine.
- **Pure data-bound negative result** — honest but a harder publish; folded in as the *mechanism*
  behind "ill-posed at achievable data scale."
- **Defense = "fundamentally ill-posed"** — stronger but attackable ("generative models are built
  for one-to-many"); not chosen.
- **Defense = "own the small-data regime"** — most honest, lowest ceiling; the data-efficiency
  framing subsumes it without conceding as much.

## Consequences
- **Headline experiment:** (1) a **data-scaling curve** — monolithic generation vs decomposition
  at 25/50/100% of real data; (2) **detail-fidelity** — retrieved-real detail vs generated detail
  at equal data budget.
- The three detailizer negatives + the Layer-A negative are **evidence for the claim**, not
  wasted work.
- Editability, recipe-closure, snap, sketch-relief, weathering, ornaments become the **demo
  wrapper** — they make the artifact impressive and editable but are explicitly *not* the proof.
- A fair, strong **monolithic baseline at equal data** must be chosen (see the follow-up
  decision) or the comparison is attackable as a weak-baseline strawman.
