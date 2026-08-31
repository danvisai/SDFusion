# Agent handoff — continuing this work on the AMD workstation

Companion to [`README.md`](README.md) (the public model card, which holds the results, method,
literature, and traps). **Read that first.** This file is the operational half: environment, layout,
and what to actually do next.

Written 2026-08-04, at the migration off the Gilbreth cluster.

---

## 1. The machine changed, and it is AMD

| | |
|---|---|
| CPU | AMD Ryzen AI Max+ 395 (Strix Halo), 16C/32T |
| GPU | **Radeon 8060S iGPU — gfx1151, RDNA 3.5** |
| memory | 128 GB unified (61 GB system + 64 GB VRAM carve-out), 8 GB swap |
| disk | 1.9 TB NVMe, ~1.4 TB free |
| OS | Ubuntu 26.04 |

Everything before this ran on an **A100 80GB with CUDA 12.6**. The port was assessed by reading
imports, not by assumption:

**✅ What is fine**
- `scripts/train_vecset.py` imports only numpy / torch / h5py + project modules. **Training does not
  need pytorch3d.**
- **No custom CUDA kernels anywhere in project code** — no `CUDAExtension`, no `cpp_extension.load`.
- No flash-attn, xformers, deepspeed, apex, or bitsandbytes. The usual ROCm blockers are all absent.
- 🔑 **The ~40 hard-coded `.cuda()` / `device='cuda'` call sites are NOT a problem.** ROCm PyTorch keeps
  the `torch.cuda` namespace and maps it onto HIP. **Do not "fix" them** — rewriting them to
  `device_type` strings is churn that buys nothing and risks breaking the CUDA path.

> ### 🛑 SUPERSEDED (2026-08-13) — read [`ROCM-EXTENSION-BUILDS.md`](ROCM-EXTENSION-BUILDS.md) instead
>
> The block immediately below is **wrong** and is kept only as a record of the migration-day
> assessment. There is no pytorch3d blocker, and CUDA extensions in general are not blocked here.
>
> The cause was never a missing port: this venv's ROCm is the **wheel-based TheRock SDK**, so
> `torch.utils.cpp_extension` finds no `ROCM_HOME` and reports `IS_HIP_EXTENSION = False`. Every
> CUDA-extension build then fails or silently degrades to CPU-only. **Set `ROCM_HOME` and they build.**
>
> Already fixed and installed in the venv: **pytorch3d with GPU support** (512px raster 4464 ms → 160 ms)
> and **`diso` DiffMC/DiffDMC** (512³ in 29 ms, gradients OK), which unblocks Sharp-Edge Sampling.
> **FlexiCubes** needs no build at all — it is pure PyTorch, not Kaolin-bound.

**⚠️ The one real blocker — pytorch3d** ← *superseded, see above*
- No ROCm wheels exist. Needed by `utils/util_3d.py` and `scripts/foundations/eval_massing_arms.py`
  (**the harness**) for `MeshRasterizer` / `MeshRenderer`.
- Build it **CPU-only** (`FORCE_CUDA=0`). At n=48 buildings the rasterisation cost is tolerable.
- If the CPU build also fights you, the rendering is separable from the metrics — `fp_iou`, `missing`,
  `extra`, `vol_iou` are voxel operations and do not need pytorch3d. Only the montages do.

  *(Second correction: the harness no longer uses pytorch3d at all — `eval_massing_arms.render_world`
  renders via pyrender/EGL, and `scripts/train_vecset.py` never imported it. pytorch3d is now only on
  the superseded dense-grid path, `utils/util_3d.py` → `stage3a_model.py` / `vqvae_model.py`.)*

**⚠️ Do NOT `pip install -r requirements-frozen.txt` verbatim.** It pins 14 `nvidia-*-cu12` packages,
`torch==2.8.0+cu126`, `torchvision==0.23.0+cu126`, and `triton==3.4.0`. Install ROCm torch wheels first,
then the non-torch remainder. Note ROCm's official Ubuntu support may lag 26.04; gfx1151 support is
recent (ROCm 6.4+).

**⚠️ Expect materially slower training.** The denoiser is only 49M params and fits the 64 GB carve-out
with enormous headroom, but an integrated GPU is not an A100. Budget accordingly before planning a
240k-step run — the runs behind this work were ~10–11 GPU-hours *each* on A100.

---

## 2. What to bring over

**Do not copy the cluster tree.** It is 1.4 TB: `data/` 493 GB, `logs_building/` 406 GB, `legacy/`
369 GB. Roughly 700 GB of that is superseded dense-grid snapshots — **delete, don't copy.**

| item | how |
|---|---|
| code, docs, results, montages | `git clone`, branch `massing-solid-gate-retrain` |
| the corpus | **regenerate** — 25 MB in git rebuilds 35 GB SDF + 17.4 GB latents (`REPRODUCING.md` §4) |
| model weights | this folder — **945 MB** for the five vecset checkpoints |
| `stage3a_lod2_deployed.pth` | 7.2 GB, **optional** — superseded baseline, comparison arm only |
| Dora-VAE (`dora_vae_1_1.ckpt`) | 2.1 GB, re-download from Hugging Face |

Regenerated `data/` lands near ~55 GB, not 493 — `REPRODUCING.md` §4d notes `nrw.h5` and `plateau.h5`
are intermediate staging nothing current uses.

Verify after transfer: `sha256sum -c SHA256SUMS`.

---

## 3. Where things are

| what | path |
|---|---|
| **THE harness** — 48 pinned ids, all arms, one pass | `scripts/foundations/eval_massing_arms.py` |
| cheap tracker (has a no-op detector; **not authoritative**) | `scripts/foundations/probe_vecset_checkpoint.py` |
| training | `scripts/train_vecset.py` (`--resume`, `--surf_weight`, `--surf_t_center`, `--archive_every`) |
| the model | `models/networks/vecset_denoiser.py` |
| the gradient path | `DoraCodec(differentiable=True).freeze()` — off by default |
| corpus loading | `load_surfaces` — ⚠️ **never read the h5 directly**, it is inward-wound |
| frame conversion | `scene.surface_sampling.to_array_frame`, guarded by `verify_frame` |
| results artifacts | `execution/artifacts/massing_arms_eval_*.json` |
| reasoning trail | `docs/wayfinding/vecset-convergence/` |

`--ids_from` replays a pinned id set so runs stay comparable. **Use it.** Every number in the model
card is on the same 48 ids; a number from a different sample is not comparable to any of them.

---

## 4. What to do next

**The open question: what separates the 29 solid buildings from the 19 hollow ones?**

This is the first *specific* investigation this project has had — every prior step was a blind lever
pull. If the split correlates with something legible, it is a targeted fix rather than a sweep.

- ❌ **Building size is ruled out** *(tested 2026-08-04)*. Median GT volume 50,515 (solid) vs 43,554
  (hollow), Mann-Whitney **p=0.246**, point-biserial r=0.186 (p=0.204). Not significant at n=48.
- ⬜ **Untested:** footprint complexity (vertex count, concavity, aspect ratio), source corpus
  (3DBAG / NRW / PLATEAU), height, roof-form class.

Secondary, cheaper: **lower `--surf_weight` at the same band** — the collapses may simply be the term
overshooting on harder cases.

**Then, in priority order:**

1. **Criterion 2 — footprint fidelity 0.962 → 1.000.** This is the *hard, non-negotiable* gate and it
   was under-weighted for a whole cycle in favour of 3D IoU, which the specification marks
   diagnostic-only. Weight it first.
2. **Add a collapse rate to the harness.** It reports medians, which is correct for unimodal noise and
   actively misleading here. The bimodality was caught by eye, from a render — not by the metrics.
3. **#79 — SNE (sharp-normal-error).** The only proposed instrument that might separate crisp from
   melted, and still absent from the harness. ⚠️ `guard_roughness` is **not cross-arm comparable** —
   do not use it to compare models.
4. **#82 — footprint-only height inference.** Until this lands, the claim is *"footprint + height →
   mass"*, not *"footprint alone"*. **Keep that exact wherever the work is written up.**

---

## 5. Read this before trusting any number

The full list is in the model card's **Measurement traps** section. The three that will actually cost
you time:

1. **Never extrapolate the training curve.** It went 0.719 → 0.657 → 0.532 → **0.840**. Two separate
   runs were nearly killed during multi-checkpoint collapses that recovered. A 30,000-step window of
   garbage output is not evidence of a dead run *in this model*.
2. **Always report `vs input` beside any quality number.** The generator scores well by *declining to
   act* — at s=0.45 it returned its input at 99.9% and inherited its score. A model that does nothing
   looks excellent on every aggregate metric here.
3. **Render before you conclude.** The aggregate went 0.195 → 0.200 ("better") while a building went
   from a box to a shredded cage. Both bimodality and the cage failure were caught by eye, never by a
   scalar. This project is judged visually and the metrics have repeatedly failed to see what mattered.

⚠️ **One historical correction to be aware of:** a "beat 0.840 3D IoU" bar appears throughout older
documents. It is **retired**. The criteria are: (1) human visual judgement — primary, (2) footprint
match — hard gate, (3) 3D IoU — diagnostic only. A retired criterion was re-imposed and dominated the
framing of two tickets before this was caught.
