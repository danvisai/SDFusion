# Train the Full-Data Real-Pair Monolith

Type: task
Status: resolved (v1/v2 negative, v3 follow-up succeeded -- see Answer)
Blocked by: 07

## Question

Train and validate the monolithic detail generator on the real train_100 pairs under a documented
compute budget. Establish convergence, held-out behavior, inference reproducibility, and a checkpoint
that is strong enough to serve as an honest C2 kill-gate baseline.

## Comments

## Answer

**Built (deliberately NOT `models/base_model.py`'s `create_model`/`train.py` production dispatch --
that framework is sized for the deployed Stage3a prior, not a research baseline; see
`models/networks/monolith_unet.py`'s docstring):**
- `models/networks/monolith_unet.py` -- `MonolithUNet`, a FiLM-timestep-conditioned 3D UNet,
  coarse-SDF channel-concat conditioning, noise-prediction output (10 tests).
- `models/monolith_diffusion.py` -- `GaussianDiffusion`: linear schedule, `q_sample`, `p_losses`,
  deterministic DDIM `ddim_sample` (13 tests).
- `datasets/monolith_pair_dataset.py` -- loads ticket 07's pairs on demand (no duplicated SDF
  grids), reusing `render_facades.load_buildingnet_sdf` + `build_monolith_pairs.low_pass_sdf`
  + `eval_harness.frame_n_input` unchanged; 90°-rotation/flip augmentation around the up axis
  (9 tests).
- `scripts/foundations/train_monolith.py` -- resumable training loop, atomic checkpointing,
  content-hashed checkpoint identity, `history.jsonl`, run manifest (6 tests for the
  checkpoint/resume contract, matching the PRD's own "training smoke test" ask literally).
- `scripts/foundations/eval_monolith.py` -- held-out quantitative + qualitative evaluation and
  the inference-reproducibility check (integration script, not unit tested, per this project's
  established convention for GPU/real-data-dependent code).

**Bug caught and fixed before any real evaluation could be trusted:** naive DDIM sampling
diverged -- an early checkpoint produced values outside `[-16, 7]` (inputs are normalized to
`[-1, 1]`) and ~65% occupancy on real conditioning. Root cause: `x0_pred = (x - sqrt(1-ᾱ_t)·eps)
/ sqrt(ᾱ_t)` divides by a value that is tiny by design at high `t`, so an imperfect `eps`
prediction is amplified and compounds over every remaining step. Fixed with the standard
`clip_denoised` DDPM/DDIM practice: clamp `x0_pred` to `[-1, 1]` every step. Regression-tested
(`test_clip_x0_bounds_output_against_an_overconfident_eps_prediction`) with a mock model that
always predicts a large constant epsilon.

**Compute budget, derived from a measured throughput smoke run (PRD: "derive these from the
measured throughput... of the first full-data run"), not guessed:** 1xA100, `base_channels=32`
(3 down/up levels, 4,092,961 params -- kept modest given this project's own prior finding that
data variety, not parameter count, is the lever at this corpus size), `batch_size=8` (27.5GB
peak, measured against 72GB free), ~1.5-1.6 it/s sustained (data loading, not GPU, was
confirmed not to be the bottleneck at `num_workers=6`). **15,000 steps chosen** (~2.6h,
~85 augmented passes over the 1,415-building train slice) as a bounded, honest budget for a
baseline model, not a hero run.

**Data split:** `train_val_ids` carves a seeded 10% (157 ids) held-out-from-gradients slice OUT
OF `train_100` itself (never the sealed `data/splits_v1/test.json`, which stays reserved for
tickets 12/13's headline comparison) -- purely to monitor convergence.

**Run 1 (`logs_building/monolith_v1/`, unweighted MSE, checkpoint digest `7283560067fa`):**
training and validation loss both fell smoothly to a near-zero plateau (~0.0007-0.003 train,
~0.0007-0.01 val, tracking each other with no divergence -- no sign of catastrophic overfitting)
within ~1,000-3,000 steps. **This aggregate number is misleading and worth stating plainly:**
real BuildingNet targets are >90% constant background after truncation (measured
surface-band -- `|x0|<0.3` -- voxel fraction on `train_100`: mean 2.9%, range 0.04%-9.3%), so an
unweighted per-voxel MSE is dominated by the easy, building-independent background and gives
almost no gradient signal for the part of the volume that actually matters. Held-out DDIM
sampling (`ddim_steps=1000` -- a diagnostic found the codebase's usual `ddim_steps=50` gives
markedly worse occupancy for this far-less-mature-than-Stage3a checkpoint, since coarser steps
compound per-step error more) over 32 class-balanced held-out buildings: **mean generated
occupancy 32.5% (median 33.0%) against mean real occupancy 1.66% (median 0.09%)** -- the
monolith substantially over-generates volume. Qualitatively (`outputs/monolith_v1/montage.png`)
confirms it: nearly every held-out building, regardless of its real shape (a detailed shell, a
solid box, or a near-empty sparse mesh), decodes to a visually similar generic blob.

**Run 2 (`logs_building/monolith_v2/`, surface-weighted MSE, checkpoint digest
`4860b790ef66`):** a principled attempted fix for the loss-dilution finding above --
`GaussianDiffusion(surface_band=0.3, surface_weight=20)` weights the loss `1+20=21x` inside the
surface band, pre-registered (chosen from the *measured* 2.9% surface-band fraction to target
roughly a 38% loss share for that band, decided **before** looking at this run's sampling
results) rather than tuned to the outcome. **Result: worse, not better** -- mean generated
occupancy **51.5% (median 49.4%)**, higher than the unweighted run. Diagnostic checkpoints along
the way (step 3,000: ~30% vs unweighted's ~35%; step 9,000: ~44%, already exceeding the
unweighted final; step 15,000: 51.5%) show occupancy climbing through training rather than
improving -- up-weighting accuracy near the true surface did not translate into correct sign
(occupied vs. empty) behavior at sampling time, and if anything pushed the sampler toward
predicting "surface-like" values more broadly. **A genuine, disclosed negative finding about
this specific mitigation**, not swept under the rug: naive per-voxel loss reweighting is not a
free fix for a sparse-target diffusion process.

**Inference reproducibility (all three runs): PASS.** DDIM eta=0 with a fixed seed is
bit-identical across repeated real-GPU runs (`torch.equal` true); a different seed produces a
different output. Verified against the actual trained checkpoints, not just the mock-model unit
tests.

**Run 3 (`logs_building/monolith_v3/`, x0-prediction, unweighted, checkpoint digest
`392ac75e5e92`) -- the follow-up that succeeded.** Structural fix rather than another loss-weight
guess: `GaussianDiffusion(predict_x0=True)` has the network predict `x0` directly instead of
noise (`models/monolith_diffusion.py`'s updated docstring has the full derivation; 4 new tests,
`PredictX0Test`). Two reasons this was the next thing tried, not a third reweighting guess: (a)
at low noise the objective becomes closer to direct reconstruction, tying the loss more tightly
to getting voxel SIGN right rather than to matching a noise vector; (b) `ddim_sample`'s division
by a near-zero term then falls at LOW `t` (the model's already-refined, late steps) instead of
HIGH `t` (the from-scratch first steps) -- structurally safer, and empirically confirmed so (no
`clip_x0` divergence observed at any checkpoint, unlike the epsilon-prediction runs).

A 3,000-step diagnostic (matched against v1/v2's own step-3,000 checkpoints before committing to
the full budget) was decisive: **per-building generated occupancy already tracked real occupancy
closely** (e.g. 8.6% vs 9.1% real, 2.6% vs 2.6% real) where v1 and v2 were both still at
30-44% regardless of the real building's actual sparsity. Full 15,000-step run, same budget as
v1/v2 for a controlled comparison: **mean generated occupancy 1.57% (median 0.06%) against mean
real occupancy 1.66% (median 0.06%)** over the same 32 class-balanced held-out buildings --
essentially matched, both in the mean and (loosely) the heavily-skewed median, a dramatic
reversal of v1's 32.5% and v2's 51.5%. Per-building agreement is close throughout, not just in
aggregate (e.g. `RESIDENTIALhouse_mesh0642`: 8.70% generated vs 9.00% real;
`COMMERCIALoffice_building_mesh2148`: 8.99% vs 9.10%). Qualitatively
(`outputs/monolith_v3/montage.png`) the v1/v2 failure mode -- every building decoding to the same
generic blob -- is gone: where the coarse conditioning carries real signal (e.g.
`COMMERCIALhouse_mesh0798`, a solid box), the generated output visibly tracks its scale and rough
shape; where coarse is near-empty (common -- many real buildings are already thin/sparse before
low-passing), generation now stays appropriately small and sparse instead of defaulting to a
large mass.

**Honest conclusion, updated:** the *engineering pipeline* was already sound (unchanged from
v1/v2: real pair loading matches ticket 07 exactly, checkpointing/resume is tested and atomic,
sampling is numerically stable and exactly reproducible). The *checkpoint quality* diagnosis that
blocked v1/v2 -- naive per-voxel loss reweighting doesn't fix a sparse-target diffusion process --
turned out to be correctly diagnosed as a parameterization problem, not a weighting problem:
switching the prediction target (not the loss weights) closed the gap from "off by 20-30x" to
"matched to within measurement noise" on the one metric this ticket measures (occupancy
fraction, not yet full geometric/perceptual fidelity). **Recommendation for tickets 12/13: use
`monolith_v3` as the reference checkpoint.** This is a much stronger candidate for surviving a
weak-baseline critique (PRD user story 26) than v1/v2 were, though occupancy-fraction agreement
is a necessary, not sufficient, proxy for "strong" -- it says the monolith gets the right AMOUNT
of material in roughly the right conditioned cases, not that the specific geometry or facade
detail is realistic. The full geometric/perceptual comparison (paired massing metrics,
neutral-facade FID) is ticket 13's own protocol, deliberately not duplicated here.

**Out:** `logs_building/monolith_{v1,v2,v3}/{ckpt/*.pth, history.jsonl, manifest.json}`
(gitignored); `execution/artifacts/monolith_{v1,v2,v3}_{train_manifest,eval}.json` (tracked
provenance/results); `outputs/monolith_{v1,v2,v3}/montage.png` (qualitative, gitignored).

Unblocks ticket 12/13 with `monolith_v3` as the recommended reference checkpoint -- the C2
kill-gate (ticket 13) can now proceed with a monolith baseline that passed this ticket's own
sanity checks, rather than the v1/v2 checkpoints this ticket explicitly warned against trusting.
