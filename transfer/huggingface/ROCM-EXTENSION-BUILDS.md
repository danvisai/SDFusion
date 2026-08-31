# Building CUDA-extension packages on the AMD box

Companion to [`AGENT-HANDOFF.md`](AGENT-HANDOFF.md). That file's §1 pytorch3d assessment is
**superseded by this one** — see "What this corrects" at the bottom.

Written 2026-08-13. Machine: Strix Halo / Radeon 8060S / gfx1151, torch 2.12 nightly + ROCm 7.13.

---

## The root cause: `ROCM_HOME` is never set

This machine's ROCm is the **wheel-based TheRock SDK** (`_rocm_sdk_core` / `_rocm_sdk_devel` inside
the venv), not a system `/opt/rocm`. `torch.utils.cpp_extension` looks for `ROCM_HOME` / `ROCM_PATH`
in the environment, or `/opt/rocm` on disk. It finds neither:

```
CUDA_HOME = None
ROCM_HOME = None
IS_HIP_EXTENSION = False
```

Every CUDA-extension package therefore either **fails to build** (`fatal error: cuda_runtime.h`) or
**silently degrades to a CPU-only build**. That is the whole story behind "no ROCm wheels exist" —
one unset variable, not a missing port. No source patching was needed for anything below.

## The fix

```bash
R="$VIRTUAL_ENV/lib/python3.14/site-packages/_rocm_sdk_devel"
export ROCM_HOME="$R" ROCM_PATH="$R"
export PYTORCH_ROCM_ARCH=gfx1151
export CPATH="$R/include"        # ⚠️ hipcub / rocprim / rocThrust live in _rocm_sdk_devel,
                                 #    NOT in _rocm_sdk_core. Without this, hipcub is not found.
export PATH="$R/bin:$PATH"       # hipcc
```

With those set `IS_HIP_EXTENSION` becomes `True`; torch auto-hipifies `.cu` sources and swaps
`nvcc` → `hipcc`.

⚠️ **Always pass `--no-deps`.** A plain `pip install diso` pulls numpy 2.5.2 and trimesh 5.0.0,
which breaks `opencv-python` (needs numpy <2.3) and `numba` (needs numpy <2.5) in this venv.

---

## Installed and verified

### diso (DiffMC / DiffDMC) — Dora's Sharp-Edge-Sampling dependency

```bash
pip install --no-deps --no-build-isolation diso
```

Measured on gfx1151:

| grid | DiffMC | DiffDMC |
|---|---|---|
| 64³  | 0.3 ms | 1.2 ms |
| 512³ | 29 ms  | 34.5 ms |

Two independent correctness checks, because "it imports" is not evidence:

- Unit-sphere vertices land at radius **0.4999 ± 0.0001** (expect 0.5).
- At 512³ it emits **407,136 verts / 814,268 faces — identical topology to
  `skimage.measure.marching_cubes`** on the same field, and ~28× faster (29 ms vs 0.8 s).
- Gradients flow to the input SDF for both classes.

This unblocks the **Sharp-Edge Sampling** path in
[`docs/wayfinding/vecset-convergence/supervision-signals.md`](../../docs/wayfinding/vecset-convergence/supervision-signals.md)
(−18% Chamfer / −13.6% SNE in Dora's own ablation), which is the reason to care.

⚠️ **diso is CC BY-NC 4.0 (non-commercial).** Fine for research; a licensing problem if anything ships.

### pytorch3d — rebuilt with GPU support

Not on PyPI; build from GitHub (its `setup.py` already has explicit ROCm support — `is_rocm =
torch.version.hip is not None`, hipcub instead of NVIDIA CUB):

```bash
git clone --depth 1 https://github.com/facebookresearch/pytorch3d.git
MAX_JOBS=28 FORCE_CUDA=1 pip install --no-deps --no-build-isolation ./pytorch3d
```

| 30k-face mesh @512px | time |
|---|---|
| the old CPU-only build | 4463.6 ms |
| **ROCm GPU** | **159.9 ms** (28×) |

**This repaired a path that was dead, it did not risk a working one.** Before the rebuild,
`utils/util_3d.render_sdf` — imported by `models/stage3a_model.py`, `models/vqvae_model.py` and the
`sdfusion_*_model.py` family — raised `RuntimeError: Not compiled with GPU support` on this machine.
After, it returns a `(1, 4, 128, 128)` render. Nothing on the *current* vecset path was affected:
`scripts/train_vecset.py` and `eval_massing_arms.py` never import pytorch3d at all.

---

## Needs no build

- **FlexiCubes** — the nv-tlabs reference implementation is **pure PyTorch** (`import torch` + a
  lookup-table module). Runs unmodified: 64³ in 356 ms, gradients flow to both the SDF and the
  per-cube weights. Only *Kaolin's packaging* is NVIDIA-only, never FlexiCubes itself. Relevant
  because `docs/research/crisp-massing-literature.md` lists FlexiCubes as a forward option and
  attributes it to Kaolin.
- **torch_cluster.fps** — already stubbed with a pure-torch FPS in `scripts/foundations/dora_frozen_gate.py`.
- **flash-attn / xformers** — unnecessary. ROCm AOTriton covers it via
  `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` (30× measured on the vecset attention shape).

## Genuinely unavailable

- **nvdiffrast** — no ROCm support upstream (zero mentions of hip/rocm/amd in its `setup.py`), and
  **zero usages anywhere in this repo's Python**; the only mention is a Dora README note about an
  experiment that failed. If a differentiable rasteriser is ever wanted, pytorch3d-GPU is the substitute.
- **open3d** — no cp314 wheel. Used in `external/Dora/sharp_edge_sampling/sharp_sample.py` for exactly
  one thing: writing a point cloud to PLY (`save_vertices_as_ply_open3d`).
  `trimesh.PointCloud(pts).export(path)` replaces it.

---

## Keep pyrender for the montages

pytorch3d-GPU does **not** supersede the pyrender/EGL path in `eval_massing_arms.render_world`.
On the same 512px view pyrender takes **2.1 ms** against pytorch3d-GPU's 159.9 ms — it is a fixed-function
rasteriser doing far less work. pytorch3d-GPU earns its place only if a *differentiable* render is
wanted, which per Dora's own README is the experiment that failed.

## What this corrects

`AGENT-HANDOFF.md` §1 says pytorch3d is "the one real blocker", that "no ROCm wheels exist", and that
it should be built **CPU-only** with `FORCE_CUDA=0`. That was a reasonable read at migration time but
is wrong: the GPU build works once `ROCM_HOME` is set. It also lists `diso` nowhere, and
`crisp-massing-literature.md` implies FlexiCubes needs Kaolin. All three are corrected above.

## Caveats

- Built against nightly torch 2.12 / ROCm 7.13. A stack bump may require rebuilding both extensions.
- pytorch3d was installed from git `main` (`9381c40`), which self-reports 0.7.9 — the same version
  string as the previous CPU-only install, but not the identical tree. If a legacy `sdfusion_*` script
  ever fails on a pytorch3d API, that is the first thing to suspect.
