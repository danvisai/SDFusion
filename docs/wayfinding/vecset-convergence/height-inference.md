# Inferring height from the footprint: feasible, and it costs more than it looks

Ticket: [Infer height from the footprint, so the pipeline can run footprint-only](https://github.com/danvisai/SDFusion/issues/82)
· Map: [#69](https://github.com/danvisai/SDFusion/issues/69) · 2026-08-07
· Reproduce: `scripts/foundations/probe_height_inference.py`

**Verdict: measured, and recommend NOT adopting now.** Height is inferable only to **R² ≈ 0.55**, and
inferring it costs the blockout baseline **−0.074 mean 3D IoU** with a **37.5% rate of under-building
by >10% of GT volume** — against 0.0% when height is specified. Footprint-only is a genuinely **weaker
task definition**, not a free alternative. [#81](https://github.com/danvisai/SDFusion/issues/81)'s
decision to treat height as a user input is reinforced, not overturned.

## 1. The predictability ceiling

Held-out n=714, trained on 34,909.

| predictor | test R² | MAE | median abs err |
|---|---|---|---|
| B0 global mean | −0.001 | 3.39 m | 2.51 m |
| **B1 per-region mean** | **0.364** | 2.53 m | 1.87 m |
| B2 linear on log(area) | 0.173 | 3.05 m | 2.42 m |
| **B3 GBM, hand features + region** | **0.537** | 2.16 m | 1.70 m |
| CNN on the raw 64×64 footprint + region | 0.510 | 2.18 m | 1.70 m |

🔑 **Two thirds of what is predictable is just "which country".** The per-region mean alone reaches
0.364; every footprint feature together adds only ~0.17 on top. Region means: 11.97 m (BAG) / 5.90 m
(NRW) / 7.47 m (PLATEAU).

🔑 **A CNN does not beat hand features** (0.510 vs 0.537), and it overfits after ~10 epochs. Shape
detail beyond area / perimeter / compactness / elongation carries no additional height signal, so the
~0.55 ceiling is a property of the data, not a modelling failure. B3's relative error: **median 20.3%,
p90 80.0%.**

## 2. ⚠️ Metric height is not what the harness consumes

Before re-scoring anything: `blockout_sdf` takes a **voxel extent** `(y0, y1)`, not metres. Measured on
n=400 — `corr(voxel extent, height_m) = 0.428`, and voxels-per-metre spans **3.826–6.796** (IQR).
**Buildings are per-instance normalised into the cube**, so a metric height prediction does not map
onto the grid the harness scores on.

The well-posed target is therefore the **voxel span**, which lives in the same normalised frame as the
footprint mask. It predicts about equally well — and no better:

| target | R² | MAE |
|---|---|---|
| metric height (m) | 0.537 | 2.16 m |
| **voxel span** | **0.548** | 7.10 vox (median 6.05, on a mean span of 38.9) |
| `y0` ground level | 0.548 | 3.55 vox |

⚠️ **`y0` is not constant** (mean 12.54, sd 7.06), so footprint-only must predict *two* numbers, and
both are only half-determined.

## 3. The re-score

n=48 pinned ids. **As the harness prints it:**

| arm | fp-IoU | missing | extra | 3D IoU |
|---|---|---|---|---|
| blockout, specified height | 1.000 | 0.000 | 0.183 | 0.845 |
| blockout, inferred height | 1.000 | 0.000 | **0.167** | 0.780 |

**That row is misleading, and in two separate ways.**

⚠️ **First, `summarise` takes per-column medians independently**, so the four numbers in a row are not
the same building. `missing 0.000` does not mean nothing is missing — it means *more than half* of
buildings miss nothing. The map already recorded this trap ("medians lie on bimodal outcomes; the
harness must report a collapse rate") and it fires again here.

⚠️ **Second, `extra` IMPROVES (0.183 → 0.167) — for the wrong reason.** The map calls `extra` "the
decisive column". Under-predicting height removes surplus volume by removing volume *generally*. **A
scorecard led by `extra` would rank the weaker baseline as better.**

The distribution is what actually changed:

| arm | IoU mean | IoU med | IoU p10 | miss mean | **>10% missing** |
|---|---|---|---|---|---|
| blockout, specified | 0.842 | 0.845 | 0.728 | 0.000 | **0.0%** |
| blockout, inferred | 0.768 | 0.780 | 0.612 | 0.075 | **37.5%** |

Per-building: mean **−0.074** 3D IoU, median −0.066, worst **−0.297**, and **69% of buildings are
hurt**.

## 4. Recommendation

**Do not adopt footprint-only as the default task.** The ticket itself framed this as *"rework, not a
defect: nothing currently depends on it"*, and the measurement says the rework buys a task that is
harder for the honest reason that the information is not in the footprint.

**If bulk city generation is later wanted**, this is feasible — R² 0.55 gives plausible skylines, which
is the stated use — but it must be scored as **its own task with its own baseline**, never mixed with
specified-height numbers.

⚠️ **The published A2 figure (fp 0.962 / miss 0.002 / extra 0.191 / IoU 0.838) is a specified-height
number and is not comparable to any inferred-height row.**

## What was not run, and why

**A2 itself was not re-scored under inferred height.** It would require a persisted predictor wired
into `eval_massing_arms.py` as an `--infer_height` mode — which is precisely the "rework nothing
depends on" the ticket flags, and the decision does not turn on it: A2 projects *from* the blockout, so
it inherits the −0.074 and the 37.5% under-build rate before it starts. If footprint-only is ever
adopted, that integration is the first step, not this ticket's.
