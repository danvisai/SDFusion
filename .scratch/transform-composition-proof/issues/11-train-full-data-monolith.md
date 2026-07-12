# Train the Full-Data Real-Pair Monolith

Type: task
Status: resolved (negative/partial result -- see Answer)
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

**Inference reproducibility (both runs): PASS.** DDIM eta=0 with a fixed seed is bit-identical
across repeated real-GPU runs (`torch.equal` true); a different seed produces a different
output. Verified against the actual trained checkpoints, not just the mock-model unit tests.

**Honest conclusion -- this is the ticket's central finding:** the *engineering pipeline* is
sound and fully verified (real pair loading matches ticket 07 exactly, checkpointing/resume is
tested and atomic, training/validation loss behave sanely, sampling is numerically stable and
exactly reproducible). The *checkpoint quality* is not yet where PRD user story 26 ("a strong
full-data monolith checkpoint, so the C2 comparison survives a weak-baseline critique") needs it
to be: both the 15k-step unweighted and surface-weighted monoliths substantially over-generate
occupied volume relative to real held-out buildings. **Recommendation for tickets 12/13: use
`monolith_v1` (unweighted; the better of the two, though still weak) as the reference
checkpoint if a preliminary comparison is wanted, but do NOT treat a decomposition-vs-monolith
result against either checkpoint as a trustworthy C2 kill-gate verdict without first improving
monolith quality** -- a decomposition arm "beating" a monolith this weak would not be evidence
for C2, only evidence of an unfair fight. Concrete next steps for whoever picks this back up:
substantially more training steps now that the pipeline/budget is known-fast (~1.6 it/s), more
model capacity, an x0-prediction (rather than epsilon-prediction) parameterization -- known in
the diffusion literature to behave better for sparse/imbalanced targets -- or a loss weighting
scheme less naive than a flat per-voxel multiplier (e.g. focal-style weighting on the predicted
occupancy sign, not just proximity to the surface value).

**Out:** `logs_building/monolith_{v1,v2}/{ckpt/*.pth, history.jsonl, manifest.json}`
(gitignored); `execution/artifacts/monolith_{v1,v2}_{train_manifest,eval}.json` (tracked
provenance/results); `outputs/monolith_{v1,v2}/montage.png` (qualitative, gitignored).

Does not unblock ticket 12 in the sense PRD intended ("a strong... checkpoint") -- ticket 12 can
proceed using `monolith_v1` as a working (not yet strong) reference if the project owner wants
to see a preliminary decomposition-vs-monolith shape, but the C2 kill-gate (ticket 13) should
wait on a monolith-quality follow-up before its verdict is trusted.
