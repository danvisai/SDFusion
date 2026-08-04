# Recoherence improvement plan — rhythm/alignment-aware part-set refiner

**Date:** 2026-06-14 · **Status:** PLAN ONLY (user: "plan only, don't build yet").
**Implements:** `PROGRESS_AND_RESEARCH_2026-06-14.md` §3 thread #1 ("rhythm/alignment-aware
re-cohere") — the doc's explicitly-recommended next step, a stepping stone to #2 (neuro-symbolic
facade-program).
**Companion docs:** `TRACK2_part_mixing_design.md`, `ARCHITECTURAL_DETAIL_RESEARCH_2026-06-10.md`.
**Companion memories:** `project_architecture_generation_research`, `project_part_coherence_research`,
`project_branch_test_suite`, `project_localized_snap`, `project_part_vocabulary_full`.

---

## 0. Diagnosis (grounded in the current code + data)

**Refiner** (`models/networks/part_set_refiner.py`): a conditional set-DDPM over **40 slots ×
`[type-onehot 10 | box6 | validity 1]`** (`PART_DIM=17`), cross-attending the 64³ massing via
`MassingEncoderSpatial`. `box6 = center3 + size3`, **axis-aligned — no orientation**. Already has:
corruption-robust *deletion* supervision (`loss(x_corrupt=...)`), channel weights (validity ×3,
box ×1.5), set-SDEdit `refine(strength, steps)`.

**Data** (`outputs/part_layouts_full/part_instances.npz`): **28,421 instances / 1,849 buildings**,
mean 15.4 parts/bldg. Rows = `[building, type, cx,cy,cz, sx,sy,sz, npts]`, centers in `[-1,1]`,
sizes in `[0,1]`. Types (`RAW_TYPES`/`TYPE_NAMES` in `part_layout_planner.py`): window(13858),
roof(3843), door(3368), column(2647), tower(1165), balcony(861), dome(810), balcony_upper(756),
chimney(592), stairs(521). **No orientation, no floor label, no adjacency — all derivable, none
present.**

**⚠ SPLICE was already integrated — this plan STRENGTHENS it, doesn't add it.** The current
`PartSetRefiner` *is* the SPLICE-style integration (its docstring; TRACK2 "modeled on SPLICE";
trained `refiner.pth` 2026-06-11). It KEPT SPLICE's joint set-denoise + validity(add/remove) +
edit-as-set-SDEdit, but DROPPED the three components our data can't feed: per-part **pose
orientation** (we're axis-aligned), per-part **shape code** (no shape latents — parts render
procedurally), and the **attention-guiding anti-float loss** (needs per-part point/occupancy
assignment). The montage confirms the result: junk gets *moved not killed* and refined sets stay
cluttered (no rows). → Lever B's **wall-attachment + relational losses are the data-compatible
substitute for SPLICE's dropped attention-guiding loss**; lever A is its editing-locality. Do NOT
re-scope this as "integrate SPLICE."

**The exact gap** (verifiable in code):
1. The ε-loss (`PartSetRefiner.loss`) has **no reward for alignment / rhythm / symmetry /
   attachment** → the refiner can only nudge boxes + drop dupes.
2. The *only* thing producing rows today is `layout_detail.regularize_ops` — a **hand-rule
   post-process** (windows/doors only, runs *after* the refiner so the two can fight, can't
   generalize to balconies/columns).
3. The edit is a **whole-set** noise (`refine(strength)`), not X-Part **localized
   neighbor-regen** → a one-window edit re-noises the whole untouched facade.

**Measurement caveat:** `regularize_ops` currently *masks* refiner weakness. When evaluating any
learned gain, score the refiner output **before** `regularize_ops`, else the hand-rules hide the
delta.

### 0.1 What our data is — and what it is NOT (align approaches to the data, not vice-versa)
Each instance = **type + axis-aligned bbox (center + size)**. There is **no floor metadata, no
orientation, no shape code, no adjacency** — BuildingNet gives per-point labels, nothing more. The
published methods we cite assume richer inputs (OmniPart/SPLICE: oriented parts + shape codes;
facade grammars: explicit floors/bays). **Those assumptions must be adapted DOWN to our data, not
imported.** Anything we want beyond type+bbox (a "band", a "wall side") is a **derived, noisy**
quantity we compute — and we only rely on it as far as it measurably exists.

**"Height-band" (formerly mis-named "floor index") — measured, not assumed.** Clustering window
heights (`cy`) per building over the 28k instances (buildings with ≥6 windows, n=1000):
~2.8 bands/bldg (median 3); within-band `cy` std 0.011 vs window half-height ~0.04 (so bands are
genuinely co-planar); **78% of windows within 0.02 of their band centroid (22% off — NOT a rigid
grid).** Conclusion: a horizontal-band structure is real and derivable, but it is emergent and
imperfect. It is **not** an architectural floor (a band can be a sill row, clerestory, etc.).
→ The core losses use it **relationally only** (no label, no assumed grid); any explicit band/side
channel (lever C) stays optional and caveated.

---

## 1. Levers (research → concrete change). All reuse the 28k instances, 1 GPU.

### A. Localized neighbor-regeneration (X-Part / `project_localized_snap`) — inference-only
**Why:** untouched facade should stay bit-stable when one part is edited.
**Change:** `PartSetRefiner.refine(x_init, sdf, strength, steps, edit_slots=None, neighbor_k=4,
floor_strength=0.04)`:
- Build per-slot noise scale `m ∈ [floor_strength, strength]^SLOTS`: edited slots + their
  `neighbor_k` nearest (by box-center L2 among valid slots) get `strength`; all others get
  `floor_strength` (≈ frozen).
- In `q_sample` use `t0_slot = (m * (T-1)).long()` per slot; in the DDIM loop step each slot from
  its own `t0_slot` down (or simpler: single global schedule but blend `x ← (1-α_slot)·x_init +
  α_slot·x_step`, `α_slot = m/strength`). Keep the existing whole-set path when `edit_slots=None`
  (back-compat — the gates pin current behavior).
- Wire in `layout_detail.recohere_ops`: pass the edited group's slot as `edit_slots`. In
  `adjust_ops_after_snap`, on a wall-moving snap the moved-wall parts are the edit set.
**Cost:** minutes, no retrain. Composes with B/C.

### B. Auxiliary structure losses (OmniPart/SPLICE; the doc's "learned floor-grid + spacing +
wall-attach in-model") — retrain, keeps representation
**Where:** add to `PartSetRefiner.loss`, computed on the implied **x0 prediction**
`x0_pred = (x_t - sqrt_1mab[t]·eps) / sqrt_ab[t]`, masked to slots valid in the **clean** x0.
Decode `cy = x0_pred[..., NT+1]`, `size = x0_pred[..., NT+3:NT+6]`, `type = argmax(clean x0)`.
All terms differentiable; sum with small weights (start 0.05–0.2) ramped in after ~500 iters so
they don't fight denoising. Use a **relational** form (pull predicted *relationships* toward the
GT relationships precomputed in §2) — generalizes better than per-coord L2 and is the architecture
signal.

1. **Band-rhythm** `L_row`: for same-type part pairs `(i,j)` already at similar GT height
   (`|cy_gt_i − cy_gt_j| < τ_row`, τ_row≈0.04 — the measured within-band std is 0.011, so this
   threshold cleanly captures real bands), penalize `(cy_pred_i − cy_pred_j)²`. Rewards co-planar
   parts staying co-planar. **Uses only type + cy — no floor label, no assumed grid, no band
   estimation.** This is the term that does the real "rhythm" work and is fully data-grounded.
2. **Uniform-size** `L_size`: same-type-same-row parts → pull `size_pred` toward the GT
   per-(building,type,row) median size (variance-reduction).
3. **Spacing** `L_space`: within a GT row, sort parts by lateral coord (the dominant horizontal
   axis), penalize variance of consecutive gaps (CV of spacing). Sorting passes gradients to
   values.
4. **Mirror-symmetry** `L_sym` — ⚠ CONDITIONAL, data says go easy: measured window mirror-chamfer/
   spread median 0.70; only **10% of buildings are clearly symmetric (<0.3), 27% loosely (<0.5)**.
   So a blanket symmetry loss would FIGHT the 73% of asymmetric facades. Apply it **per-building,
   gated**: compute each building's symmetry score in §2; only add `L_sym` for the ~10-27% that are
   already symmetric (reflect centers across the footprint-centroid plane, soft-chamfer). Low λ.
   Do NOT make it a global prior.
5. **Wall-attachment** `L_attach`: trilinearly sample the conditioning `sdf` at each valid
   `center_pred`; wall types (window/door/balcony/column) penalize `sdf²` (sit on surface); roof
   types (chimney/dome/tower) penalize `relu(sdf)` (sit on/above the top). Makes attachment learned
   in-model (today it's the post-hoc `resnap_ops_to_surface`).

`loss_total = L_eps (weighted, current) + λ_row·L_row + λ_size·L_size + λ_space·L_space +
λ_sym·L_sym + λ_attach·L_attach`.
**Cost:** retrain `train_part_set_refiner.py` (~hours, same 28k data).

### C. Representation augmentation (StructureNet/SPLICE) — retrain + re-encode. OPTIONAL, caveated.
Add per-slot **derived (noisy)** channels — only justified if B's relational losses plateau. Both
are computed, NOT labeled; carry their imperfection (22% off-band) as soft inputs, never hard
constraints:
- **side_id** — 4-way one-hot nearest-wall **or** `[sin θ, cos θ]` bearing from the footprint
  centroid (NOT orientation — we have none; this is *position-derived* "which side"). Lets the
  model say "this wall".
- **band_id** — normalized height-band index from §2 clustering (the measured ~3 bands/bldg). A
  *soft* hint, not a floor. Lets the model reason "same band" natively instead of re-deriving it
  from cy each step.
`PART_DIM: 17 → ~22`. Re-encode dataset, retrain, then `regularize_ops` becomes a *fallback* not
the engine. Stepping stone to the facade-grammar model (thread #2): side×band is the (approximate)
cell grid a split-grammar predicts into — but our grid is inferred + noisy, so the grammar model
would need to tolerate that (a reason #2 is weeks, not days).

### D. (Later) learned attention-guiding / adjacency (SPLICE) — anti-float + relationship logic.
Out of scope for this pass; note it for the facade-program follow-on.

---

## 2. Data prep (CPU, once) — `scripts/foundations/derive_part_structure.py` (new)
From `part_instances.npz`, per building compute and cache to
`outputs/part_layouts_full/part_structure.npz`:
- **Footprint centroid + principal axis** (PCA on all centers' xz) → mirror plane for `L_sym`,
  lateral axis for `L_space`.
- **Height-bands** (derived, noisy — measured ~3/bldg, 78% co-planar): 1D gap-cluster the cy of
  row-forming parts (window/door/balcony) per building (sorted-gap split at τ_row). Store
  per-instance `band_id` + per-(building) band centroids. **For B these are only used to FORM the
  same-band pairs of `L_row`; the loss itself stays relational (no grid asserted).** For C they
  become the soft `band_id` channel.
- **side_id**: nearest of the 4 footprint walls (or bearing angle) per instance — position-derived,
  not orientation.
- **Same-band/size partners**: per-(building,type,band) groupings for `L_row`/`L_size`/`L_space`.
This single file feeds both B (relational pair targets) and C (optional soft channels). Emit a
per-building "band cleanliness" stat so we can DROP buildings whose bands are too noisy to supervise
on (don't train rhythm on confetti).

---

## 3. Acceptance metrics + gates (the product gate — `project_branch_test_suite`)
**Hard gates (must stay green, run before+after — user mandate):**
`scripts/server/test_branches.py` 12/12 · `scripts/server/test_sculpt_flows.py` 17/17 ·
`scripts/server/eval_visual.py` photo sheet sane.

**New refiner eval** (extend `train_part_set_refiner.py` montage; score on held-out val,
**pre-`regularize_ops`**, A/B vs current `refiner.pth`):
- keep existing: parts-kept ratio, junk-killed %, junk-moved.
- NEW structure metrics (lower = better unless noted): **row-σ** (median within-row cy std),
  **spacing-CV** (inter-window gap coefficient of variation), **symmetry-chamfer** (set vs mirror),
  **attach** (mean `|sdf(center)|` for wall parts).
**Success = structure metrics improve materially with no regression in parts-kept / junk-killed /
val-ε.** Plus a new visual: perturbed confetti facade → re-cohered into rows montage
(`outputs/part_set_refiner/recohere_rhythm_montage.png`).

---

## 4. Sequencing + file map
1. **A** (inference): edit `part_set_refiner.py::refine` (+ `recohere_ops`/`adjust_ops_after_snap`
   in `layout_detail.py`). Montage + gates. *No retrain.*
2. **§2 data**: new `scripts/foundations/derive_part_structure.py` → `part_structure.npz`.
3. **B** (retrain): aux losses in `part_set_refiner.py::loss`; `train_part_set_refiner.py` loads
   `part_structure.npz`, adds metrics to the eval. Retrain → A/B eval → gates. Save new
   `outputs/part_set_refiner/refiner.pth` (back up the old one first).
4. **C** (optional, after B proves out): bump `PART_DIM`, re-encode (`encode_sets` + new channels),
   retrain; demote `regularize_ops` to fallback.
5. Keep `regularize_ops` as a safety net throughout; everything behind back-compat defaults.

## 5. Risks / notes
- Aux losses can fight denoising → ramp-in + small λ; watch val-ε doesn't blow up.
- Window-dominated data → symmetry/spacing learned mostly from windows (fine; that's the visible
  rhythm). Columns/balconies benefit from `L_attach` most.
- Always measure learned gain **before** `regularize_ops` (it masks the delta).
- A is the cheapest real win and de-risks B; ship + montage it first even though this pass is
  plan-only.

## 6. Sources
OmniPart 2507.06165 · SPLICE 2512.04514 · X-Part 2509.08643 · StructureNet 1908.00575 ·
FacAID 2406.01829 · Pro-DG 2504.01571 (facade-grammar follow-on).
