# #125 — Absolute-occupancy corrector plumbing, cache manifest fixed on the real corpus

*2026-09-11. CPU only: no A2/Dora load, no GPU touched, no training run.*

Implements the plumbing #125 specifies, replacing `scripts/foundations/prototype_voxel_editor.py`'s
prior roof/surface-masked, KEEP/ADD/REMOVE-action prototype (the throwaway prototype #125's Further
Notes describes) with the absolute-occupancy, whole-volume corrector the spec requires. Contract
tests in `scripts/foundations/test_prototype_voxel_editor.py` (61 tests, CPU synthetic geometry).

A two-axis review (Standards / Spec, against #125's text) of the first pass found and fixed: a
top-level import that silently forced torch+scipy onto every import of this module (broke the
file's own lazy-import convention); `assert_manifest_matches_corpus` and `verify_cache_integrity`
existing but never called from `cache`/`train`; `build_replay_envelope` never written to disk; and
`evaluate-full` not checking that the referenced screening report actually passed its gate, or
that it was a screen of the SAME checkpoint being scored. All four are now wired in.

**This is plumbing evidence only, in the sense #125's own Testing Decisions define for the prior
synthetic smoke test: it validates representation, isolation, and gate mechanics. It is not a
preregistered result.** No training run has happened. No screening gate has been evaluated. Nothing
here should be read as evidence toward or against #120's question.

## What changed from the prior prototype

- Removed `edit_mask` (the `roof`/`surface` band) entirely. Every voxel is eligible for learned
  change; the only deterministic projection is `sanitize_footprint` (footprint intersection),
  applied identically to the learned candidate and every baseline arm.
- Removed the KEEP/ADD/REMOVE action head. The corrector's sole output is one absolute occupancy
  logit per voxel (`build_corrector`). `derive_action`/`apply_action_to_source` remain as a pure,
  tested reduction identity used only to reweight the training loss toward rare edits — never a
  second output, never the stored state.
- Added the explicit validity contract #125 asks for (pending #117's formal decision):
  `ground_connected_ok` (every solid component touches y=0 — no floating fragments),
  `hollow_shell_voxels` (empty space unreachable from any boundary face — courtyards/passages
  that stay reachable from outside remain admissible), `min_thickness_survival` (erosion-survival
  proxy for ADR 0004's s\*=3 voxels @64³, with `border_value=1` so a building touching the grid's
  own edge — every building's y=0 ground layer — isn't penalised as "thin" for that alone).
- Added two surface-realization arms over identical corrected occupancy: `recover_surface_control`
  (signed EDT) and `recover_surface_narrowband` (blends toward the A2 source field within a band,
  provably sign-exact by construction — a convex combination of two same-signed values can't cross
  zero).
- Added the manifest/cache apparatus: fixed salted-hash row selection (`_salted_score`, order- and
  pool-independent), per-row and per-role-file content/row-list digests, and role-tagged cache
  files (`train`/`screen`/`full`) so `open_training_cache` structurally cannot open a screen or
  full-role file (`RoleIsolationError`) — isolation checked on disk, not just assumed.
- Added the exact preregistered gate math from #125's Testing Decisions (`screening_gate`,
  `full_gate`), and E/A correction-difficulty strata fixed a priori (`error_stratum`, bounds
  `(0.05, 0.20)`) before any corrector has produced a result.

## The manifest, fixed against the real corpus

`manifest` is CPU-only (reads `vecset_latents.h5` + `real.h5` metadata; never loads A2/Dora/GPU) and
was run against the actual production corpus, not just synthetic tests:

| role | n | region split |
|---|---|---|
| train | 384 | 182 BAG / 182 NRW / 20 PLATEAU (= 163 opportunity + 19 identity per BAG/NRW, 20 identity PLATEAU — matches spec exactly) |
| screen | 96 | 32 / 32 / 32 |
| full | 714 | 248 / 235 / 231 (the corpus's own held-out split; not a fixed target) |

Screen ⊂ full, train disjoint from full, all confirmed programmatically. Rerunning `manifest`
against the same corpus reproduces byte-identical row lists and digests (checked directly, not
assumed). Committed at `outputs/voxel_editor_prototype/manifest.json`.

## The re-scoping question is still open, and this session did not adjudicate it

#125 has one unanswered comment (2026-08-27, from the issue author) arguing the corpus's real
target is a 64×64 height field, not a volume (`missing`=0 on 714/714 buildings), so a dense-64³
formulation is ~64x over-parameterised relative to its own label — see
`docs/wayfinding/solid-first-subtractive-modeling/10-program-recovery.md` and `127-height-map-generator.md`. `127` separately records the owner's 2026-08-29 ruling that
**#113 stays open and this finding must not be used to fold or retire the voxel route without a
fresh decision.**

This session built #125 **as literally specified** (dense absolute binary occupancy, every voxel
editable) on explicit instruction, not as a resolution of that question on the merits. The
re-scoping proposal remains open and unaddressed.

## Not done this session (needs GPU, isolated from any active A2 run)

- `cache --role {train,screen,full}`: materialise the authentic A2-decode pairs for each role file
  against the frozen checkpoint (sha256 `643aed08...`). Untested beyond unit coverage of its pure
  helpers (`row_content_digest`, `sanitize_footprint`, the reseed-key formula) — the full
  Dora/A2-decode path itself has not been run.
- `train`: fit the corrector on the 384-row train cache.
- `evaluate`: the one-time 96-row screen against #125's exact preregistered gate.
- `evaluate-full`: only if the screen passes — the 714-row confirmation plus the sealed 618-row
  complement, against the exact preregistered full-gate.
- Human visual review (fixed-frame plan/facade/isometric/section montages, Sharp Normal Error with
  nonzero views) — `sharp_normal_error_for_arms` is wired but has not been run against real geometry.

## Files

`scripts/foundations/prototype_voxel_editor.py` (rewritten), `scripts/foundations/test_prototype_voxel_editor.py` (new, 61 tests), two `test_frozen_corpus.py` tests updated for the
new single-role `VoxelCache` API, `outputs/voxel_editor_prototype/manifest.json` (committed).
