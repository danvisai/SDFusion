# Test the Fixed Detail-Scale Coincidence

Type: research
Status: resolved
Blocked by: 01, 03

## Question

At the fixed-a-priori `s*`, measure whether BuildingNet semantic detail categories occupy the sub-`s*`
band while massing occupies larger scales. Produce the reproducible measurement artifact and report
the result honestly as pass, partial coincidence, or failure without moving the boundary.

## Comments

## Answer

**Built:** `scripts/eval/measure_scale_spectrum.py` (TDD, 12 contract tests in
`test_measure_scale_spectrum.py`, including a synthetic-geometry integration test — a big cube
labeled wall + a small disjoint cube labeled window, written to a real OBJ/component-label/
faceindex fixture and parsed through the actual extraction pipeline, not mocked).

**Vocabulary (fixed a priori, not invented for this ticket):** massing = wall(1), roof(4) —
CONTEXT.md's "base mass ... overall roof form". Detail = window(2), door(6), tower(7),
column(12), balcony(14), chimney(15), balcony_upper(16), stairs(17), dome(22) — CONTEXT.md's
"windows, doors, balconies, cornices..." extended with the ADD-element vocabulary
`build_element_library.py` (ticket 04) already shipped and tested. Excluded: `undetermined`
(noise bucket), `ground`/`floor` (site, not building), the roof-vs-roof_structure reuse of
label 4 (a component-position heuristic for what to retrieve, not a distinct semantic label),
and every remaining id, which is "uncertain" in `label_names.json` and was never adopted
anywhere else in the project.

**Scale metric:** per building, per label, faces are grouped into geometric instances via the
same bbox-touch union-find `build_element_library.py` uses (so several disjoint occurrences of
one label are measured as several small instances, not one facade-spanning box). Characteristic
scale = median of an instance's 3 bbox side lengths (robust to a thin carve/shell axis on one
side and a long run on another — either would make min/max misleading), normalized by the
building's own max AABB extent. BuildingNet meshes carry no absolute real-world units (verified,
not assumed: every raw OBJ's own max extent is already 1.0), so `s*` is compared on the same
resolution-tied, per-building-relative basis ADR 0004 derives it from —
`s*/bmax = voxels/(res-1)` = 5/95 ≈ 0.0526 @96³ — with no invented meters-per-building
conversion.

**Run:** all 1,572 `data/splits_v1/train_100.json` buildings (never the sealed test set), 0
parse failures. `execution/artifacts/scale_spectrum.json` (committed); the box-plot figure at
`outputs/scale_spectrum/scale_spectrum.png` (gitignored like other QA montages, regeneratable).
An earlier 50-building dry run gave noisy per-category verdicts at low n (e.g. chimney n=3
flipped fail→pass once the full 1,572-building population raised it to n=480) — the full
population, not a `--limit` sample, is the reported result.

**Result: 6/11 — `partial_coincidence`.** Massing passes cleanly: wall (n=2382, median 0.309)
and roof (n=3230, median 0.224) both land well above `s*`=0.0526. Detail splits into two
groups:
- **Matches the prediction (below `s*`):** window (n=22288, median 0.035), door (n=3401,
  0.042), column (n=3271, 0.013), chimney (n=480, 0.040).
- **Fails the prediction (above `s*`):** tower (n=1042, median 0.063), balcony (n=695, 0.352),
  balcony_upper (n=656, 0.220), stairs (n=456, 0.156), dome (n=687, 0.079).

**Honest interpretation, boundary not moved:** the coincidence holds cleanly for *thin facade
articulation* (windows, doors, columns, chimneys) but fails for the *discrete large ADD
elements* — precisely ticket 04's already-adopted retrieval vocabulary (tower, balcony,
balcony_upper, stairs, dome). These are large in absolute scale (a tower or a balcony/veranda
strip can span a large fraction of the building) despite being compositionally "detail" — ill-
posed to *generate* at achievable data scale, hence composed/retrieved. This suggests `s*` (a
voxel-Nyquist representability threshold on the massing generator) and "generatable vs.
composable" are not the same axis for every category: the ADD elements are composed not because
they are too *small* for the massing grid to resolve, but because they are too *varied/rare* to
learn a generative distribution over — a related but distinct reason. This does not weaken C2
(the composition claim never depended on elements being sub-`s*`-scale), but it means the paper
should frame the massing/detail *scale* split and the generate/compose *tractability* split as
correlated, not identical, and should not cite this coincidence as clean support for the ADD-
element categories specifically. No downstream ticket is blocked by this result; it feeds the
paper's detail-vocabulary framing and evidence package (ticket 18) directly.
