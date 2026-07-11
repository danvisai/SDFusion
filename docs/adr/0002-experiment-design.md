# Experiment design for the massing-scale-decomposition proof (D1–D4)

**Status:** accepted (2026-07-10, plan-coherence grilling session)

Refines the experiment in `0001-massing-scale-decomposition.md` after re-reading the three
`execution/` plans and finding they contradicted each other. Four decisions:

## D1 — What "the decomposition" arm is
Headline decomposition = **Stage 3a SDF massing + retrieved detail** — a matched SDF-vs-SDF
comparison against the monolith that isolates the *factorization* only, and the honest evidence
for "generation is well-posed for massing" (Stage 3a emits a real mass, not params + a template).
The recipe-param → *procedural* massing path is kept as a **secondary robustness ablation** (shows
the win survives changing the massing source), not the headline.

## D2 — Equal-data means the retrieval library is fractioned too
At data fraction *X*, **both arms see exactly `train_X` of BuildingNet detail**: the monolith trains
on `train_X` real pairs; the decomposition's element library is **rebuilt from `train_X` ids**.
Stage 3a massing stays at full data (not the contested variable). Keeping the library full while
starving the monolith would make "equal data" false and void the headline. The curve therefore
measures the real result: *retrieval degrades more gracefully than training as detail data shrinks.*

## D3 — `s*` is fixed a priori; the coincidence is a test
`s*` is fixed **before** looking at results, as `k` voxels at the working resolution (≈0.5 m). The
scale-spectrum measurement then **tests** whether semantic-detail categories fall below `s*` and
massing above — it does **not** choose `s*`. This keeps "fixed in advance" true and makes the
coincidence a genuine, falsifiable finding rather than a line fit to the outcome.

## D4 — Drop the web-scale giant
Hunyuan3D-2 is **image-conditioned** and cannot take a footprint, so it can't be a task-matched
competitor. Rather than benchmark a different (easier) question, the *"just get more data"* objection
is answered by **extrapolating the fractioned monolith's data-scaling slope**. Tighter paper, one
fewer integration; loses a concrete web-scale datapoint (accepted).

## Consequences
- The monolith trains on **real** `make_real_detail_pairs.py` pairs (not synthetic `detail_pairs_v1`).
- `build_element_library.py` gains `--include-ids/--exclude-ids`; per-fraction, leakage-safe libraries.
- All three `execution/` plans were rewritten (rev2) to these decisions; `CONTEXT.md` terms updated.
