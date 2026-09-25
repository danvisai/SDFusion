# GPU and CUDA performance: how the A100 was used, what bottlenecked, and what CPU would have cost

This document collects every GPU-related measurement already recorded in this repository into one
place. Each number cites the file it came from. Nothing here is a new measurement.

Hardware for all "A100" figures: one NVIDIA A100 80 GB, driver 590.48.01, CUDA 12.6, torch 2.8.0+cu126
(`REPRODUCING.md`, `requirements-frozen.txt`). Peak VRAM for any job stayed under about 25 GB.

## 1. Honest scope

The repo shows **hardware selection and pipelining**, not kernel-level tuning.

- No `torch.autocast`, `GradScaler` or `torch.compile` appears in `scripts/` or `models/`.
- Mixed precision appears only in Dora's own `16-mixed` fine-tune recipe (`docs/wayfinding/vecset-convergence/decoder-finetune-sizing.md`).
- fp16 is otherwise used for storage (caches, SDFs) and for the appearance pipeline (`scripts/appearance/gbuffer_neural_render.py`).
- Models that were trained are small (3 to 49 M params), so they were already fast enough on one A100.

## 2. Where the A100 did the work

### 2.1 Frozen Dora-VAE encoding (the heaviest GPU job)

| item | measurement | source |
|---|---|---|
| one real-surface vecset latent | 207 ms (2.9 ms CPU sampling, 200 ms Dora forward) | `whole-volume-voxel-transform/188-a2-retrain.md` |
| one blockout latent, serial | 415 ms (141 ms `real.h5` read, 51 ms extrude and mesh, 222 ms encode) | same |
| re-encode of 35,623 buildings | 7,019 s (about 1.95 h), 9,338 MB fp16 | `vecset-convergence/frame-fix-result.md` |
| all 1,526,778 BuildingWorld rows, real pass only | about 88 GPU-hours, about 412 GB | `188-a2-retrain.md` |

Consequence: the full-corpus encode was too expensive, so #188 uses a designed, reproducible
capped-proportional cohort (`scripts/foundations/vecset_cohort.py`) and not the whole corpus.

### 2.2 Training

| run | size | cost | source |
|---|---|---|---|
| vecset denoiser | 49 M params | about 305 ms/step, 60k steps about 5 h; 240k-step runs about 10 to 11 GPU-h each | `latent-token-order/map-87.md`, `transfer/huggingface/AGENT-HANDOFF.md` |
| height-map generators | 3.4 to 3.58 M params | about 16 to 35 min per arm | `scripts/foundations/train_height_map_generator.py`, `solid-first-subtractive-modeling/127`, `132`, `138` |
| monolith diffusion UNet | 4.1 M params | 15k steps, about 2.6 h | `.scratch/transform-composition-proof/issues/11-train-full-data-monolith.md` |
| Dora decoder-only fine-tune (sizing) | 120.5 M trainable | 332 ms/step, 19.7 GB, 32.9 h for 10 epochs | `vecset-convergence/decoder-finetune-sizing.md` |
| Dora full-VAE fine-tune (sizing) | 191.6 M trainable | 381 ms/step, 21.2 GB, 37.7 h for 10 epochs | same |

Batch size and VRAM headroom were never the constraint on the 80 GB card.

### 2.3 Inference and serving

Warm timings on one A100 (`scripts/server/town_generate_service.py`):

| arm | seconds |
|---|---|
| A2 (20 denoise steps through a 191 M codec) | about 7.0 |
| retrieval | about 0.5 |
| height-map arms (3.4 M convnet plus marching cubes) | about 0.3 |
| envelope | about 0.2 |

Neural appearance renders take 10 to 15 s using fp16 and `enable_model_cpu_offload()`
(`scripts/server/neural_appearance.py`). This is the one place VRAM was traded for speed.

## 3. Bottlenecks actually hit

### 3.1 GPU idle during CPU-bound preprocessing (#188)

The blockout pass spends 141 ms reading a 64^3 volume from the 1.55 TB `real.h5` and 51 ms on
extrusion plus marching cubes, against 222 ms of encode. That is **192 ms per row with the A100 idle**,
about 3.2 GPU-hours over a 60,000-row cohort.

Fix: `prefetch()` in `scripts/foundations/precompute_vecset_latents.py`, a producer thread that runs one
step ahead of the GPU consumer (tests in `scripts/foundations/test_precompute_vecset_latents.py`).

Design constraints worth knowing:

- **Exactly one thread, and it owns every HDF5 handle.** h5py is not thread-safe.
- **No multi-process DataLoader.** Each worker would fork its own copy of the codec's per-row RNG and
  break the contract that row `k`'s latent is a function of `k` alone (`encode_row` reseeds from the row).
- **The real-surface pass is not overlapped**, because it is already GPU-bound (2.9 ms vs 200 ms).
- **Errors travel through the queue.** One bad row costs one `[skip]` line, not the run.

### 3.2 Determinism versus throughput

Fully deterministic evaluation (`deterministic=True` plus TF32 off) cost **13x** (2 it/s vs 27), about
40 minutes for one arm. The run instead sets `cudnn.benchmark = False`, seeds everything, and reports the
residue (about 0.001 between processes, in median 3D IoU) as a `noise_floor` in the artifact. The effect
being measured was 0.18, so exactness at that price was the wrong trade
(`scripts/foundations/eval_massing_arms.py`, around lines 612 to 620).

### 3.3 Storage and I/O

- Regenerating decoder targets on the fly costs 0.27 s per building, cheaper than reading 16 MB from disk.
  A target cache would have been 570 GB. Recommendation recorded: no target cache
  (`decoder-finetune-sizing.md`).
- The `real.h5` read was the dominant CPU cost in the blockout pass (3.1).

### 3.4 Memory hygiene

`torch.cuda.empty_cache()` and `reset_peak_memory_stats()` are used between arms in
`scripts/foundations/eval_massing_arms.py` and `probe_surface_loss.py` so sequential evaluations do not
inherit each other's fragmentation.

## 4. What slower hardware cost (measured)

After the A100 instance was lost, the caches were rebuilt on an **AMD Radeon 8060S iGPU**, not a CPU-only
machine. A pure CPU run would be worse, so treat these as a lower bound on the CPU penalty
(`docs/wayfinding/latent-token-order/RECOVERY.md`).

| pass | A100 | slower box | ratio |
|---|---|---|---|
| encode real | 2.04 h | about 15 h | about 7x slower (GPU-bound) |
| encode envelope | 3.04 h | about 16 h | about 5x slower |
| alignment (KD-tree matching) | 5.68 h | about 1.1 h | about 5x **faster** (CPU-bound) |
| total | 11.3 h | about 32 h | |

Per building: about 0.2 s on the A100 against about 1.5 s on the iGPU.

Takeaways:

- GPU-bound stages (the Dora forward pass) suffer badly on weaker accelerators.
- CPU-bound stages (KD-tree matching) can be faster on a machine with a better CPU. **Profile each stage
  before deciding where it belongs.**
- Anything that avoids re-encoding is worth more on slow hardware, which is why the cohort sample exists.

### Estimates, not measurements

These were **not** measured in this repo and should be presented as reasoning:

- A CPU-only 20-step A2 denoise through the 191 M codec would take tens of seconds to minutes, against
  about 7 s on the A100.
- A CPU-only decoder fine-tune (32.9 h for 10 epochs on an A100) is not practical.

## 5. Opportunities not yet taken

Proposals only, none implemented:

1. **Batch the Dora encode.** The encode runs one row at a time (207 ms each).
2. **bf16 autocast on the Dora forward pass.** The A100 supports bf16 natively.
3. **`torch.compile` on the 49 M denoiser** for training throughput.
4. **A second prefetch stage** if the batched encode makes the GPU faster than the read.

Each needs its own measurement, and any change to the encode must preserve the per-row reproducibility
contract in 3.1.

## 6. Quick reference for discussion

- Bottleneck found by timing each stage per row, not by guessing.
- Idle GPU time was the largest recoverable waste (192 ms per row).
- One producer thread was chosen for correctness (h5py, RNG), not for simplicity.
- Determinism was traded against a 13x slowdown, and the residual noise is measured and reported.
- The cohort sample exists because the full corpus encode is 88 GPU-hours.
- The recovery run gave a real GPU-versus-slower-hardware comparison, including a stage that ran faster
  on the other machine.
