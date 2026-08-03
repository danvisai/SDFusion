# Reproducing this on a new machine

Everything needed to clone, rebuild and continue. Written 2026-08-03, at map
[#69](https://github.com/danvisai/SDFusion/issues/69).

**The short version:** the repo plus ~25 MB of committed data regenerates the full 67 GB corpus. Only
the model weights (~1 GB) and one third-party checkpoint (2.1 GB) come from elsewhere.

| what | size | where it comes from |
|---|---|---|
| code, docs, results, montages | 20 MB | this repo |
| `data/real_massing_v1/corpus_identity.h5` | **6.4 MB** | this repo |
| `data/real_massing_v1/surfaces_*.h5` | **18 MB** | this repo |
| `external/dora_vae_1_1.ckpt` | 2.1 GB | Hugging Face (third-party) |
| `external/Dora` source | ~50 MB | `git clone`, pinned commit below |
| model weights | ~1 GB | **not yet published — see §6** |
| `real.h5` SDF field | 34.9 GB | **regenerated**, §4 |
| vecset latent caches | 17.4 GB | **regenerated**, §4 |

⚠️ **Do not try to copy the 67 GB of data or the 400 GB of checkpoints.** Most of it is derived, and
~400 GB is intermediate snapshots of a superseded architecture.

---

## 1. Clone

```bash
git clone https://github.com/danvisai/SDFusion.git
cd SDFusion
git checkout massing-solid-gate-retrain      # the active branch; `main` is far behind
```

## 2. Environment

⚠️ Until now this repo had **no dependency manifest**. `requirements-frozen.txt` is a literal freeze of
the environment that produced every result here — 248 packages, exact versions.

```bash
python3.9 -m venv sdfusion                   # 3.9.23 is what was used
./sdfusion/bin/python -m pip install -r requirements-frozen.txt
```

Load-bearing versions, in case the freeze needs relaxing:

| package | version | note |
|---|---|---|
| `torch` | 2.8.0+cu126 | CUDA 12.6 build |
| `pytorch3d` | 0.7.8 (git `75ebeee`) | montage rendering; wheels are scarce, may need a source build |
| `numpy` | 2.0.2 | |
| `trimesh` | 4.12.2 | mesh handling, winding repair |
| `libigl` | 2.6.2 | signed distance via winding number |
| `h5py` | 3.14.0 | every corpus file |
| `scikit-image` | 0.24.0 | marching cubes |

Verified on driver 590.48.01, A100 80 GB. Nothing here needs more than ~25 GB of VRAM.

⚠️ Commands in this repo are run as `env -u LD_PRELOAD ./sdfusion/bin/python …` — the `LD_PRELOAD` unset
is a cluster quirk. Harmless to drop elsewhere.

## 3. Third-party model

The codec is **Dora-VAE 1.1** (Apache-2.0), used frozen. Two pieces:

```bash
mkdir -p external && cd external
git clone https://github.com/Seed3D/Dora.git       # pinned at a166e21
# then fetch dora_vae_1_1.ckpt (2.1 GB) from the Dora release / HF and place at
#   external/dora_vae_1_1.ckpt
```

⚠️ **Hunyuan3D-2 also appears under `external/`. Its outputs are evidence-only** — licence §5.b forbids
using them as training data. Nothing in the pipeline depends on it.

## 4. Rebuild the data

The corpus is defined by two committed files. Everything else is derived.

### 4a. What is committed

- **`corpus_identity.h5`** (6.4 MB) — `bag_id`, `footprint`, `height_m`, `source_id`, `class_label`,
  `style_id` for all 35,776 buildings. This pins *which* buildings, *in what order*, which is what makes
  held-out splits and pinned id sets reproducible.
- **`surfaces_{bag3d,nrw,plateau}.h5`** (18 MB) — the recovered LoD2 meshes, 35,623 of 35,776.

⚠️ **`surfaces_*.h5` is the one genuinely irreplaceable artifact.** It was recovered from CityGML
sources that can and do change upstream. If it is lost, the exact corpus cannot be rebuilt.

⚠️ **The meshes are stored INWARD-wound** (35,602 of 35,623 — measured in #74). Every consumer repairs
this at load time, so always go through `dora_frozen_gate.load_surfaces`, **never** `h5py` directly. The
signed-distance path will not notice inside-out surfaces; a vecset encoder will.

### 4b. Regenerate `real.h5`'s SDF field (34.9 GB)

`real.h5` is 99.6% SDF, and the SDF is a voxelisation of the meshes:

```bash
env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/ingest_surfaces.py --verify --source plateau
```

`--verify` re-voxelises recovered meshes and compares against a stored field, which is the check that
this path is faithful. ⚠️ Regeneration is **equivalent, not bit-identical** — verification compares
occupancy IoU, not float equality. And a rebuilt corpus has **35,623 rows, not 35,776**: the 153
unrecovered buildings have no mesh. #74 established they are ordinary buildings (median occupancy 0.18,
none empty), 150 of them German, so nothing is lost analytically.

### 4c. Regenerate the latent caches (17.4 GB, ~2 h each)

```bash
# real-surface latents
env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/precompute_vecset_latents.py
# the aligned blockout partners
env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/precompute_vecset_latents.py --blockout \
    --out data/real_massing_v1/vecset_blockout_latents.h5
```

🔑 **A write-time guard runs automatically** (`verify_frame`) and refuses to write a cache whose latents
do not decode onto their own footprints. This exists because a frame bug silently voided two full
training runs (#70/#78): the corpus was encoded in Frame-N while everything else spoke the array frame,
so training learned a **transposed** building. Expect `median ≈ 0.997`; a frame error reads ≈ 0.17.

### 4d. What you do *not* need

`nrw.h5` (10.9 GB) and `plateau.h5` (8.6 GB) are intermediate staging from the original ingest.
`*_smoke.h5` are small test slices. None are used by anything current.

## 5. Verify the rebuild

Run the harness on the pinned id set. It should reproduce the committed baseline exactly for the
deterministic arms:

```bash
env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/eval_massing_arms.py \
    --n 48 --ids_from execution/artifacts/massing_arms_eval_baseline.json --tag rebuild
```

Expect (from `execution/artifacts/massing_arms_eval_baseline.json`):

| arm | fp-IoU | missing | extra | 3D IoU |
|---|---|---|---|---|
| gt | 1.000 | 0.000 | 0.000 | 1.000 |
| blockout | 1.000 | 0.000 | 0.183 | 0.845 |
| codec_ceiling | 0.997 | 0.000 | 0.001 | 0.999 |

`gt`, `blockout` and `codec_ceiling` are deterministic and should match. ⚠️ `deployed_map24` is a
sampled arm and carries a measured noise floor (fp ±0.008, extra ±0.040, 3D IoU ±0.001).

## 6. Model weights — not yet published

⚠️ **The weights are not in this repo and not yet hosted anywhere.** They currently exist only on the
cluster at `/scratch/gilbreth/dsimhadr/GenerativeTowns/SDFusion/logs_building/`. **Copy them before
decommissioning that machine**, or the trained models are gone.

Worth keeping (189 MB each once optimizer state is stripped; 566 MB with it):

| checkpoint | what it is |
|---|---|
| `vecset_v4_surf/vecset_denoiser.pth` | best model — surface-loss fine-tune |
| `vecset_v3_pair_long/vecset_denoiser_step180000.pth` | 41-epoch control, the surface-loss starting point |
| `vecset_v5_surfband/…` | the band-fix run (in flight at time of writing) |
| `2026-07-16-stage3a-lod2-fromscratch-region/ckpt/stage3a_steps-latest.pth` | deployed map-#24 baseline — **14.2 GB** |

Recommended host is **Hugging Face** (free, unlimited for public model repos). ⚠️ If published publicly,
the corpus derives from **3DBAG** (NL), **NRW open data** (DE) and **PLATEAU** (JP), all of which carry
attribution terms — write a model card citing them.

The remaining ~400 GB under `logs_building/` is intermediate snapshots of the superseded dense-grid
architecture. **Delete rather than copy.**

## 7. Where the work stands

Read in this order:

1. `docs/SESSION-HANDOVER-2026-08-03.md` — current state, criteria, and the traps
2. `docs/SESSION-HANDOVER-2026-07-29.md` — the previous session
3. `docs/wayfinding/vecset-convergence/` — one write-up per closed ticket

Live map: [#69](https://github.com/danvisai/SDFusion/issues/69). Frontier: **#77**, **#79**, **#82**.

**Immediate next step:** the band-fix run (`--surf_t_center 0.55`) was launched and may not have
finished. Check `logs_building/vecset_v5_surfband/` and `logs_building/_launch_logs/v5_surfband.log`. It
is a controlled comparison against `vecset_v4_surf` — same checkpoint, same 60k steps, one variable.

⚠️ **Read the traps in the handover before trusting any number.** The load-bearing ones: never
extrapolate the training curve (it went 0.719 → 0.657 → 0.532 → 0.840 by epoch), always report
`vs input` beside a quality number (the generator scores well by *declining to act*), and n=10 probes are
not quotable — only the 48-id harness settles anything.
