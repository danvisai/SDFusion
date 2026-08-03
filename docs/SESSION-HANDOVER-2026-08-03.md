# Session handover — 2026-08-01 → 08-03

Map [#69](https://github.com/danvisai/SDFusion/issues/69). 16 commits, `a03f79d` → HEAD.
Five tickets closed (#71, #73, #74, #75, #76), three raised (#79, #81→closed, #82), one delivered
in-flight (#80). Two long training runs, ~24 GPU-hours.

## The headline

**The generator is the best this map has produced and still loses to extruding the footprint** —
0.838 vs 0.845 on 3D IoU, after the gap narrowed from 0.036 to **0.007**.

Two levers were measured against each other:

| lever | cost | gain in 3D IoU |
|---|---|---|
| training length (13.8 → 41 epochs) | 11 GPU-h | **+0.008** |
| decoded-surface loss (60k steps) | 10 GPU-h | **+0.029** |

Training length is **exhausted**. The surface loss is the biggest lever found, and it was found by
diagnosis rather than search.

## The chain of findings

1. **[#71](https://github.com/danvisai/SDFusion/issues/71) — one harness.** Every prior number was
   incomparable: the deployed figure was a hardcoded constant with no artifact, printed beside numbers
   from a different sample and script. `scripts/foundations/eval_massing_arms.py` scores all arms on 48
   pinned ids in one pass. Splitting 3D IoU into **missing vs extra** immediately revealed the deployed
   model fails by **over-filling (+45.5%)**, not eroding — invisible in the aggregate.
2. **[#73](https://github.com/danvisai/SDFusion/issues/73) — the melt is decoder intolerance.** Measured
   model-free: a latent at **cos 0.083** decodes perfectly; one at **cos 0.995** is destroyed. The 0.083
   arm is the *same mesh* re-encoded, so FPS reorders the tokens. **Latent cosine predicts nothing**,
   which voids the "working denoiser" evidence the map inherited.
3. **[#74](https://github.com/danvisai/SDFusion/issues/74) — data is not the constraint.** All 35,623
   meshes audited. ⚠️ The corpus on disk is **inward-wound** (35,602/35,623); the encoder gets outward
   only via `load_surfaces`, so never read the h5 directly. 🔑 Meshes are **coarse — median 20 faces**,
   which bounds what sharpness supervision can teach.
4. **[#76](https://github.com/danvisai/SDFusion/issues/76) — the objective cannot rank its candidates.**
   Spearman of latent distance vs decoded IoU: **−0.50 within one error family, +0.12 pooled** (wrong
   sign). A loss is a ranking device; pooled is the situation training is in.
5. **[#75](https://github.com/danvisai/SDFusion/issues/75) — length exhausted at 41 epochs.**
6. **[#80](https://github.com/danvisai/SDFusion/issues/80) — the surface loss works.** Every column
   improved; human confirmed visual improvement (criterion 1).

## ⚠️ Traps this session added

- **Never extrapolate the training curve.** It went 0.719 → 0.657 → 0.532 → **0.840** by epoch. Three
  monotonic points did not predict the fourth. A stop was recommended at the dip and would have recorded
  a false negative.
- **Always report `vs input` beside any quality number.** The generator scores near the blockout by
  *declining to act*. At s=0.45 it returned its input at 99.9% and inherited its score. A previous eval
  made this mistake; `probe_vecset_checkpoint.py` now warns above 0.95.
- **n=10 probes are not quotable.** Adjacent-checkpoint swing (0.59–0.78) is as large as any apparent
  trend. Only the 48-id harness settles anything.
- **Quote the outcome, not the peak.** #80's pre-registered criterion was met at 9.5k steps (`extra`
  0.178) and lost by 60k (0.191). Recorded as **not-met**.
- **`verify_frame` was crying wolf** — it gated on the *minimum* of 4 samples, and ~4% of real buildings
  legitimately fall below tolerance. Now gates on the median (`3ad2f65`).
- **Height is a USER INPUT** ([#81](https://github.com/danvisai/SDFusion/issues/81)). The blockout's
  `missing 0.000` is therefore not cheating, the evaluation was fair, and no re-score is needed — only
  the task description was wrong. Consequence: the blockout's *only* flaw is over-fill, so **reducing
  `extra` without adding `missing` is the entire game**.

## State of the code

- `eval_massing_arms.py` — THE harness. `--ids_from` replays a pinned id set.
- `probe_vecset_checkpoint.py` — cheap tracker with the no-op detector. Not authoritative.
- `train_vecset.py` — `--resume`, optimizer state, `--archive_every`, `--surf_weight`,
  `--surf_t_center`.
- `DoraCodec(differentiable=True).freeze()` — the gradient path, off by default.

## Next

**Rerun the surface loss with `--surf_t_center 0.55`.** The run that produced the +0.029 was supervising
at **t/T 0.401** while inference runs at **0.5–0.6** — it taught the model to reproduce its input rather
than carve it (`vs input` 0.993). Fixed in `8a3fca5`, untested. Same checkpoint, same length, one
variable: a clean controlled comparison.

Then: **#79** (SNE — the only proposed instrument that might separate crisp from melted, and still
absent from the harness), **#77** (decoder fine-tune sizing), **#82** (footprint-only height inference,
optional rework).

⚠️ **Open criterion-1 question never put to the human:** *"would you take the model's output over the
extruded footprint?"* Only the before/after pairing has been shown. That question decides whether this
generator ships, and no scalar overrides it.
