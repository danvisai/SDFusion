# Reproducing this on a new machine

Environment and artifact reference, originally written 2026-08-03 for map #69.
Reconciled 2026-09-09. See [PROJECT_STATE.md](docs/PROJECT_STATE.md) for the current work.
This is **not yet a verified clean-clone, end-to-end corpus rebuild guide**; §4b identifies the gap.

The repo contains compact identity and recovered-surface artifacts. Large SDF/latent stores are
derived, but rebuilding them must preserve original row IDs and the historical split. Model
weights and the third-party Dora checkpoint are separate downloads.

| what | size | where it comes from |
|---|---|---|
| code, docs, results, montages | 20 MB | this repo |
| `data/real_massing_v1/corpus_identity.h5` | **6.4 MB** | this repo |
| `data/real_massing_v1/surfaces_*.h5` | **18 MB** | this repo |
| `external/dora_vae_1_1.ckpt` | 2.1 GB | Hugging Face (third-party) |
| `external/Dora` source | ~50 MB | `git clone`, pinned commit below |
| model weights | varies by bundle | publication locations documented in §6 |
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

### 4b. SDF regeneration gap

`ingest_surfaces.py --verify --source plateau` **requires an existing real.h5**.
It re-voxelizes a sample of recovered surfaces and compares occupancy; it does not write or
reconstruct `real.h5`. Its normal mode recovers surface meshes from source data, also using
the existing corpus identity in `real.h5`.

The 35,623 recovered meshes can support a future rebuild, but 153 of the original 35,776
rows have no recovered surface. Compacting away those rows changes row identities and the
deterministic split; do not claim that a compacted file reproduces the pinned evaluation.
A complete rebuild needs a row-preserving writer, explicit missing-row handling and a
verification against the identity/split artifacts. That work is not supplied by this command.

On a machine that already has the original corpus, the verification command remains useful:

```bash
env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/ingest_surfaces.py --verify --source plateau
```

### 4c. Regenerate the latent caches (17.4 GB, ~2 h each)

```bash
# real-surface latents
env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/precompute_vecset_latents.py
# blockout partner encodings (token alignment is a separate pass)
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

With the original row identities, required caches and checkpoints available, run the harness on
the pinned id set. Compare against the recorded baseline; regenerated fields and surface sampling
must not be promised bit-identical:

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

`gt` and `blockout` are fixed by the supplied rows. `codec_ceiling` also depends on the
surface/encoding path; compare its measured tolerance rather than assuming bit equality. ⚠️ `deployed_map24` is a
sampled arm and carries a measured noise floor (fp ±0.008, extra ±0.040, 3D IoU ±0.001).

## 6. Model weights — published

✅ **The weights are hosted at <https://huggingface.co/danvisimhadri/SDFUSION>** (public), under
`massing-vecset/`. This closed the risk this section used to record: they existed only on the cluster.

```bash
hf download danvisimhadri/SDFUSION --include 'massing-vecset/*' --local-dir weights/
cd weights/massing-vecset && sha256sum -c SHA256SUMS
```

**Historical vecset checkpoints**, reported on the 48-id harness (later 714-row results
and the closed #87 investigation supersede these as current research status):

| checkpoint | what it is |
|---|---|
| `vecset_v5_surfband_step240000.pth` | **the band-fix model** — final, 29/48 solid |
| `vecset_v5_surfband_step230000.pth` | best 3D IoU (0.825), post-recovery |
| `vecset_v5_surfband_step220000.pth` | the collapse checkpoint, kept as evidence |
| `vecset_v4_surf.pth` | surface-loss model, pre-band-fix |
| `vecset_v3_pair_long_step180000.pth` | 41-epoch control, no surface loss |

Nine more are published for provenance — the deployed stage3a baseline (7.2 GB), the early vecset runs,
two VQVAEs and three monolith arms. ⚠️ `vecset_v1`/`v2` are **void, not weak**: they trained on
transposed latents and learned a compensating axis swap.

⚠️ **Optimizer state is stripped**, so the published checkpoints are inference/fine-tune ready but
**not resume-ready**. To resume a run, use the originals under `logs_building/`.
`scripts/foundations/stage_weights_for_transfer.py` regenerates the published set.

The corpus derives from **3DBAG** (NL), **NRW open data** (DE) and **PLATEAU** (JP).
Their source-specific licensing/provenance must be recorded by #152; do not assume one shared
attribution policy. See the dated [data audit](docs/wayfinding/solid-first-subtractive-modeling/5-data-audit.md).

**What is still cluster-only.** ~700 GB across `logs_building/` and `legacy/` — intermediate step
checkpoints and snapshots of the superseded dense-grid architecture, including six documented-negative
runs (`x0sharp-*`, the smoke test, the xcultural fine-tunes). The findings are written up in `docs/`;
the weights are not published and are **not worth copying**.

⚠️ **No corpus data is published anywhere.** The 25 MB in §4a is in *this git repo* and nothing larger
exists off-cluster — the 493 GB `data/` tree is regenerated, never transferred.

## 7. Where the work stands

Read [PROJECT_STATE.md](docs/PROJECT_STATE.md), [INTEGRATION_STATE.md](docs/INTEGRATION_STATE.md),
then the relevant `docs/wayfinding/` directory and live issue.

Map #69 and token-alignment map #87 are closed. The v5 band-fix run is complete and its collapse
findings are historical; do not resume it based on an old handover. Current work concerns semantic
height-map/program generation (#1), BuildingWorld (#156), and the separate demo maps (#97/#106).

The historical pinned-714 set remains a regression control. #153's new proof split and
#177's BuildingWorld split have distinct purposes and must be versioned separately.
