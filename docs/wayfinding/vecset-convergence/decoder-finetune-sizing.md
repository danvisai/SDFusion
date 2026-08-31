# Sizing a Dora-decoder fine-tune — and why it should not happen yet

Ticket: [Size a Dora-decoder fine-tune using the sharp/coarse split supervision](https://github.com/danvisai/SDFusion/issues/77)
· Map: [#69](https://github.com/danvisai/SDFusion/issues/69) · 2026-08-07 · A100 80GB

**Recommendation: STAY FROZEN.** Not because a fine-tune is expensive — it is affordable (~33 h on one
A100) and the targets are cheap and provably correct. Because **the reason for doing it does not
survive checking**, and the cost lands across the one thread that is currently working.

---

## 0. The premise this ticket was opened on is void

The ticket's motivation, quoted from its own body:

> the frozen gate found Dora's round-trip (**0.00818**) *worse on our data* than our own VQVAE's
> (**0.00360**). We adopted a **reconstruction downgrade**. A fine-tune is how that gets repaid.

That comparison is **exactly the one [the harness ticket](https://github.com/danvisai/SDFusion/issues/71)
later invalidated.** `surface_roughness` is a raw |Laplacian|, so it scales with the field's own slope,
and the two arms do not share one: a metric SDF on this grid measures |∇| = 0.031, Dora's decoded TSDF
measures **1.31 — 32× steeper**. Dora's roughness is inflated by about that factor *for reasons
unrelated to crispness*. The number is not cross-arm comparable, and #71 removed it from every ranking.

On the metrics that **are** comparable, #71 measured the frozen Dora codec round-tripping our buildings
as the `codec_ceiling` arm:

| arm | fp-IoU | missing | extra | 3D IoU |
|---|---|---|---|---|
| gt | 1.000 | 0.000 | 0.000 | 1.000 |
| **codec_ceiling** (frozen Dora round-trip) | **0.997** | **0.000** | **0.001** | **0.999** |
| blockout | 1.000 | 0.000 | 0.183 | 0.845 |
| deployed_map24 | 0.817 | 0.041 | 0.455 | 0.635 |

**There is no reconstruction downgrade to repay.** The frozen codec reconstructs our massing at 3D IoU
0.999 and loses 0.3% of footprint. This ticket's stated purpose was to buy back a loss that a later
ticket showed was a measurement artifact.

## 1. The live gap is 92% generator-side

Criterion 2 (footprint fidelity, the map's hard gate) currently stands at **A2 0.962 against a 1.000
gate**. Split it:

| segment | size | owned by |
|---|---|---|
| 0.962 → 0.997 | **0.035 (92%)** | the **generator** |
| 0.997 → 1.000 | 0.003 (8%) | the codec |

A perfect decoder fine-tune — one that lifted `codec_ceiling` to a literal 1.000 — would close **8%** of
the gap the map says is the live one.

> ⚠️ **A finding for the map, not for this ticket to decide.** Criterion 2 as literally written —
> fp-IoU **1.000** — is not reachable through the current evaluation *by any model*, because the codec
> ceiling itself measures 0.997, and #71 attributed part of that to marching-cubes discretisation at
> RES=64 (`codec_ceiling` ribs precisely because its field is too steep for the crossing to be located
> within a voxel). The gate may want restating as "at the codec ceiling" rather than a literal 1.000.
> That is a Destination question.

## 2. Can we generate the targets? Yes — and they are exact

Dora's query distribution, reproduced from `sharp_edge_sampling/sharp_sample.py:194-222`:

| stream | construction | count (Dora) |
|---|---|---|
| `sharp_near_surface` | sharp-edge points + N(0, σ), σ ∈ {0.001, 0.005, 0.007, 0.01} | 4 × 100,000 |
| `rand_points[:400000]` | uniform surface points + N(0, σ), σ ∈ {0.001, 0.005} | 2 × 200,000 |
| `rand_points[400000:]` | uniform in [−1.05, 1.05]³ | 200,000 |

Consumed per training step (`Dora-VAE-train.yaml`): `n_supervision = [21384, 10000, 10000]` = **41,384
query points**, MSE on TSDF, sharp weighted **2×** coarse, `lambda_kl` 0.001, AdamW lr 1e-5, batch 1.

### Signing: use `FAST_WINDING_NUMBER`, and only that

The ticket flagged that CityGML meshes are negative-volume and that signing needs fast-winding-number.
Measured, on an analytic 12-face box (our corpus's median complexity) against its closed-form SDF, and
on an open box standing in for the 10.7% non-watertight NRW meshes:

| `igl` sign type | outward | **inward-wound** | open (non-watertight) |
|---|---|---|---|
| `DEFAULT` | 1.0000 | 0.9381 | 0.9990 |
| `PSEUDONORMAL` | 1.0000 | **0.0000** | 0.9705 |
| `WINDING_NUMBER` | 1.0000 | 0.9381 | 0.9990 |
| **`FAST_WINDING_NUMBER`** | **1.0000** | **1.0000** | **0.9990** |

On the outward mesh the max |error| against the analytic SDF is **2.2e-16** — the signing is exact.
`FAST_WINDING_NUMBER` is the **only** setting robust to both hazards
[the corpus audit](https://github.com/danvisai/SDFusion/issues/74) recorded. `WINDING_NUMBER`'s 0.9381
is not "mostly right": it is exactly `1 − inside_fraction`, i.e. **it loses the entire interior** and
calls an inverted solid empty.

> ⚠️ **A wrong reason recorded in the code.** `scene/surface_sampling.py`'s module docstring says
> *"Signed-distance paths never notice, because fast-winding-number signing is orientation-agnostic."*
> The claim is true only of the **FAST** variant. The paths it is written about —
> `vecset_ceiling_probe.py:68` (`mesh_to_sdf`, whose own docstring says "via libigl's winding-number
> sign"), `dora_frozen_gate.py:59`, `scene/test_surface_sampling.py:106` — all pass the **DEFAULT**
> sign type, which measures identically to `WINDING_NUMBER` and is **not** orientation-agnostic.
> **Not a live bug** — every caller flips to outward first, via `load_surfaces` or `ensure_outward` —
> so this is the same posture as #74's winding finding: correct output, wrong stated reason, and the
> reason is what licenses the next mistake. The ingest path already does it right
> (`ingest_3dbag.py:104`, `ingest_surfaces.py:208` pass `FWN` and document it as orientation-robust).

### The targets carry real information the 64³ field cannot

Sign agreement between igl's targets and the stored 64³ SDF, binned by distance to the surface
(12 buildings, 3.0M query points):

| distance to surface | agreement | n points |
|---|---|---|
| [0.0, 0.5) voxels | 0.8743 | 2,306,897 |
| [0.5, 1.0) | 0.9999 | 129,701 |
| [1.0, 2.0) | 1.0000 | 40,135 |
| [2.0, 4.0) | 1.0000 | 72,294 |
| ≥ 4.0 | 1.0000 | 450,973 |

Beyond half a voxel the two agree **exactly**. The disagreement is entirely inside the half-voxel shell
where the stored grid physically cannot resolve a sign — a voxel is 0.03175 world units and Dora's
sharp σ range is **0.03–0.32 voxels**, i.e. wholly sub-voxel.

**This is the honest case *for* a fine-tune** and it should be recorded as such: the targets are not
merely derivable, they encode near-surface structure that our 64³ representation cannot express. That
is a genuine asset — it is just not one that the current gap needs.

## 3. Compute and storage

Target generation, measured (12 buildings, quarter density, single core), extrapolated ×4 to Dora's
full density:

| | quarter density (measured) | full Dora density (×4) |
|---|---|---|
| time / building | 0.067 s | 0.27 s |
| **all 35,623, 1 core** | **0.66 h** | **2.7 h** |
| **storage** | **142 GB** | **570 GB** |

**Compute is a non-issue; storage is the only real cost — and it is avoidable.** Training consumes
41,384 points per step out of ~1M cached. At 0.27 s/building, regenerating on the fly is cheaper than
reading 16 MB off disk. **Recommend on-the-fly generation, no target cache at all.**

Fine-tune step cost, Dora's own recipe on one A100 (batch 1, 41,384 queries, 16-mixed, token count
pinned via `split="val"` — `split="train"` randomises the downsample ratio and made the first attempt
at this comparison meaningless, 4096 tokens against 2048):

| regime | trainable | step | peak mem | 1 epoch | 10 epochs |
|---|---|---|---|---|---|
| **decoder-only** | 120.5 M (62.9%) | 332 ms | 19.7 GB | **3.29 h** | **32.9 h** |
| full VAE (Dora's recipe) | 191.6 M (100%) | 381 ms | 21.2 GB | 3.77 h | 37.7 h |

Parameter split of the 191.6M: `transformer` (the 16 decoder-side layers) **113.37 M / 59.2%**,
`encoder` 70.94 M / 37.0%, `decoder` (query head) 7.13 M / 3.7%, `pre_kl`+`post_kl` 0.15 M.

⚠️ **"Decoder-only" is not a cheap tweak — it is 63% of the model.** It saves 13% of step time over
training everything. The intuition that a decoder fine-tune is a light touch does not hold for this
architecture.

## 4. What breaks downstream

| invalidated | size | recovery |
|---|---|---|
| `vecset_latents.h5` | 9.36 GB | re-encode, ~1.95 h (#78's measured figure) |
| `vecset_blockout_latents.h5` | 9.36 GB | re-encode |
| `vecset_v2…v5` checkpoints | 13 GB | **re-train** — #75's 41 epochs and all of #80 |
| `vecset_v6_surfband_solidity` | live | the **running** #84 experiment |

And a mechanism-level conflict, not just a bookkeeping one:
[#80](https://github.com/danvisai/SDFusion/issues/80)'s decoded-surface loss — the biggest lever this
map has found — trains the denoiser against **this decoder's manifold**.
[#73](https://github.com/danvisai/SDFusion/issues/73) measured that the decoder tolerates a wholly
different-looking on-manifold latent (cos 0.083 → IoU 0.999) but not 0.5% of isotropic error (cos 0.995
→ IoU 0.053). **Fine-tuning the decoder moves the manifold the generator was taught to stay on.** Any
fine-tune must be sequenced strictly *across* the live thread, never beside it.

## 5. The one literature support does not transfer

#72 flagged this for full-text verification before anything leaned on it:

> staged VAE training with a **decoder-only second-stage fine-tune** is reported to improve fine detail
> and mesh smoothness at far less cost than training at higher resolution
> **[paper, snippet-level — not verified in full text]**. If we ever do fine-tune Dora's decoder on our
> corpus, **that is the shape to copy**.

**Verified.** The source is [3DGen: Triplane Latent Diffusion for Textured Mesh
Generation](https://arxiv.org/pdf/2303.05371) (Table 6). The claim is real and #72 reported it
accurately: *"Second stage decoder only finetuning improves fine details and mesh smoothness while
avoiding the computational cost of directly training with a higher tetrahedral resolution."*

**But the mechanism is specific to a decoder our decoder is not.** 3DGen's VAE decodes to an explicit
**DMTet tetrahedral grid**, trained with differentiable *render* losses rather than SDF regression. Its
staged fine-tune adapts a decoder to a **higher tetrahedral output resolution than it was trained at** —
which is exactly why it is cheaper than training at that resolution throughout, and the improvement is
reported in **shading FID**.

Dora's decoder is **query-based implicit**. It has no output resolution to raise. The knob that
3DGen's second stage exists to turn does not exist here, and the loss is not the same kind of loss.
**"That is the shape to copy" is not supported by its own source.**

## 6. Verdict

**Stay frozen.** Four independent reasons, in order of weight:

1. **The premise is void** — the "reconstruction downgrade" is a cross-arm roughness artifact (§0); the
   frozen codec round-trips at 3D IoU **0.999**.
2. **92% of the live gap is generator-side** (§1); a perfect decoder buys 8%.
3. **The literature support does not transfer** (§5).
4. **The cost lands across the working thread** (§4) — 18.7 GB of caches, 13 GB of checkpoints, all of
   #75 and #80, and the running #84 experiment.

Reasons 1–3 would hold even if the fine-tune were free. It is not the cost that decides this.

### Pre-registered condition for re-opening

Re-open **only** if criterion 2 closes to within **0.005 of the codec ceiling** — A2 fp-IoU ≥ **0.992**
— *and* the residual is shown to be decoder-side. At that point the codec's 0.003 becomes the binding
constraint and §3's costing applies as measured. Not before.

### What to keep from this ticket regardless

- **`FAST_WINDING_NUMBER`, always** (§2) — and the docstring in `scene/surface_sampling.py` states a
  wrong reason that would license a real bug the first time someone signs a raw h5 mesh.
- **On-the-fly target generation beats a cache** (§3) — 0.27 s/building against 570 GB.
- **The targets encode sub-voxel structure our 64³ field cannot** (§2). Filed as an asset, not a
  motivation.

---

### Reproducing

Probes are throwaway and live in this session's scratchpad, not the repo — they answer a sizing
question once and have no downstream caller:
`probe_targets.py` (query generation, timing, storage), `probe_agree.py` (binned sign agreement +
analytic-box control), `probe_finetune_step.py` (parameter split, step cost).
