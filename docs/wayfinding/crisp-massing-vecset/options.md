# Query-based implicit decoding: three adoption options, with cost and risk

**Written:** 2026-07-27 · **Effort:** crisp-massing-vecset · **Status:** decision input, nothing chosen yet

Follows [map #52](https://github.com/danvisai/SDFusion/issues/52) (which found the way) and
[map #58](https://github.com/danvisai/SDFusion/issues/58) (which exhausted the cheap half of it).
This document exists because #58 closed without chartering the surviving direction.

---

## Where this comes from

Every cheap lever is spent, and the negatives are consistent across four independent attempts:

| what was tried | where | result |
|---|---|---|
| codec / grid upgrade | #56 | **not the bottleneck** — `decode(encode(GT))` = **0.0044** vs GT **0.0041** |
| analytic composite over an extrusion prior | #56 | SDF-combine on the 64³ grid **corrupts** crispness |
| post-decode SDF refiner | #54 | plateau **0.0047**; sharpness loss un-minimizable at any weight |
| latent-space corrector | #59 | plateau **0.0047**; latent L1 fell 4× with **no** decoded gain |
| x0-sharp smoothness finetune | #60 | **no crisp regime** — diverges at w=0.1, flat at w=0.05 (0.00547 vs 0.00552) |

The shared shape of these results: **a diffusion model forced to hit an exact value in every cell of a
fixed dense grid does not produce crisp surfaces, and no amount of correcting its output recovers
them.** A query-based implicit (vecset) decoder is the only direction that removes that mechanism
rather than compensating for it — spatial resolution becomes a decode-time query, so crispness is a
property of the decoder and its *sampling*, which is exactly where Dora and Hunyuan3D-2 attack it.

Baseline to beat: **0.00552** (map-#24 deployed). Wall that stopped everything: **0.0047**.
Target floor: **0.0041**.

---

## Two facts established while writing this, both of which move the numbers

### 1. We no longer hold the source surface geometry — verified

`data/real_massing_v1/real.h5` is the massing corpus: **35,776 real LoD2 buildings**
(NL 11,776 / DE 12,000 / JP 12,000), stored as `sdf` + `footprint` + `height_m` + `source_id`, at
**64³ / 64²** (per `docs/wayfinding/crisp-massing-model/residual-retrain-design.md`).
`data/bag3d_v1/` holds only `bag3d.h5` and `bag_labels.npz`. A search for `*.obj`, `*.gml`, `*.ply`,
`*.city.json` under the LoD2 data returned **nothing**.

**Why this matters more than it looks.** A vecset encoder does not consume a grid — it point-samples
the *surface*, and Dora's entire contribution is that **where you sample determines whether sharp
edges survive**. Training a query decoder on samples derived from our 64³ SDF would inherit the 64³
low-pass: we would rebuild the architecture and keep the ceiling. So surface geometry is a **hard
prerequisite**, not a nice-to-have. 3D BAG (NL) and PLATEAU (JP) are public and re-downloadable, and
`scripts/ingest_3dbag.py` shows the ingest path was walked once — but the *surface-sampling* path is new.

### 2. The "~1849 shapes" data-bottleneck figure does not apply to massing

`docs/research/crisp-massing-literature.md` carries ~1849 real shapes into its feasibility argument.
That figure is from the **BuildingNet / detail-element** era. For **massing**, #26 established
**35,776** solid LoD2 buildings. That is a **19× correction** and it materially improves the odds for
a from-scratch vecset AE. It is still 3–30× below what CLAY / Hunyuan3D-2 / TRELLIS train on
(100K–1M+), so the gap is real — just far smaller than the literature review implies.

### 3. Hunyuan3D-2 is already exercised in this repo

Not just vendored — **run**. `scripts/osm_hunyuan_pipeline_smoke.py` (810 lines) is an end-to-end
OSM → retrieval → Hunyuan → simplify → place → contact-sheet pipeline; `hunyuan_building_mesh_smoke.py`
and `compose_hunyuan_scene_smoke.py` support it. Weights are local: **9.2G** (`dit-v2-0`) and **7.2G**
(`2mini`).

**The catch:** Hunyuan3D-2 is **image-conditioned**, not footprint-conditioned. The existing pipeline
bridges that by *retrieving a BuildingNet exemplar and rendering it as the image prompt* — a proxy,
not footprint control. Any option that uses it as a generator has to solve footprint conditioning.

---

## Option A — Build the vecset recipe into our own stack

**What.** A new point-sampled encoder with Dora-style sharp-edge-aware sampling → a latent **token
set** → a cross-attention **query decoder** returning SDF at arbitrary (x, y, z). Replace the 3D UNet
with a set-transformer / DiT, footprint injected as conditioning tokens (CLAY is direct precedent that
geometric-primitive control is a solved pattern). Marching cubes or FlexiCubes at export.

**Honors the recorded constraint** on map #52: *improve OUR model; do not adopt an off-the-shelf
frontier model, even as a baseline.*

**Cost.**
- **Blocked on mesh recovery** (fact 1). Re-ingest 3D BAG + PLATEAU + the DE source, then build a new
  surface-sampling stage. Unknown, plausibly 1–2 weeks before any model training starts.
- A **new autoencoder** *and* a **new diffusion backbone**, both from scratch, plus new conditioning
  plumbing, training loop, and eval bridge. For scale: the map-#24 retrain was 120k iters / **~54 h**
  for the *diffusion alone* on an already-trained codec. This is two campaigns, not one.
- Realistic: **multi-week**, two training campaigns, highest engineering surface of the three.

**Risk.**
- **The bet is unproven at our scale.** #56 already showed our *codec* is not the bottleneck — so
  Option A's payoff rests entirely on a **token-set diffusion being better-behaved than a dense-grid
  diffusion**. That is a plausible reading of the literature, but we have no direct evidence for it on
  35k building shapes. If it fails, it fails after weeks.
- **Data scale.** 35,776 is far better than the stale 1849, still below the field's exemplars. The part
  most likely to transfer at our size is Dora's **sampling** strategy (needs no extra data); the
  capacity/scale wins likely do not.
- **Highest cost, highest uncertainty — and the only option that keeps the constraint intact.**

## Option B — Adapt the vendored Hunyuan3D-2 to footprint conditioning

**What.** Reuse the on-disk Hunyuan3D-ShapeVAE — a vecset AE already trained with sharp-edge
importance sampling, already decoding crisp queried SDFs — and either fine-tune its DiT on our
footprint signal or train a footprint adapter (ControlNet-style) over it. Bridge its output back into
our editable-SDF downstream.

**Breaks the recorded map-#52 constraint.** Aligns with the standing preference to repurpose a trained
model over hand-rolled machinery.

**Cost.**
- **No AE campaign and no download** — the expensive, data-hungry half is already paid for by Tencent.
- Real work is the **footprint conditioning adapter** + fine-tune on `real.h5`, plus the downstream
  bridge. Partial scaffolding already exists in the smoke pipelines.
- Still wants surface supervision for the fine-tune, though **less acutely** than A: the pretrained VAE
  already knows what crisp is, so we are mostly teaching *control*, not *geometry*.
- Realistic: **weeks, one campaign, substantially smaller than A.**

**Risk.**
- **Editability — the constraint that has governed every prior decision.** Our downstream (sculpt/carve,
  SDEdit transform, the C1/C2 thesis) assumes an editable SDF we control. Hunyuan's decoder *does*
  return a queried SDF, so this is plausible — but our SDEdit and latent-editing machinery is built for
  a 16³×3 **grid** latent, not a token set. **This needs an explicit check before committing.**
- **Thesis framing.** The research claims are about *our* model. Building the generator on a frontier
  model may weaken the contribution. This is a supervisor question, not only an engineering one — and
  it is the actual reason the constraint was written in the first place.
- **Licensing.** Tencent's Hunyuan3D-2 ships under a custom license with use restrictions. Must be read
  before this touches a demo or a publication.
- **Lowest cost to crisp output; highest cost to the project's framing.**

## Option C — Use Hunyuan3D-2 as a crisp teacher / oracle, then decide

**What.** Do **not** adopt it as the generator. Run it offline as a *measuring instrument* to answer,
cheaply, the questions A and B are currently being decided blind:
1. What does crisp actually look like on our data — what is the real upper bound, in our own
   `surface_roughness` metric and on honest montages?
2. Does a query decoder have headroom on *our* corpus at all, or is 35k shapes the binding constraint?
3. *(optional, deferred)* **Distill it** — generate a large synthetic corpus of crisp building SDFs and
   train Option A's AE on that, sidestepping A's data-scale risk entirely.

**Cost.**
- **Smallest by far.** Inference only, weights local, and the harness substantially exists
  (`osm_hunyuan_pipeline_smoke.py`). Realistic: **days**, not weeks.
- The distillation arm is a real cost — but it is **deferred**, and only paid if the oracle says there
  is headroom worth chasing.

**Risk.**
- **It is not a fix.** C alone ships nothing crisper; it buys information. If the answer is "yes, there
  is headroom," A or B still has to be paid.
- The image-conditioning gap (fact 3) means the oracle measures *achievable surface quality*, not
  *footprint-faithful generation* — a genuine limit on what the measurement proves.
- Distillation would inherit the licensing question for the synthetic corpus.
- **Lowest risk, and it de-risks both other options — but it defers the decision rather than making it.**

---

## Recommendation

**Run C first, explicitly framed as feeding A.**

The reasoning is not that C is easy. It is that **every prior effort on this problem guessed at the
headroom and was wrong** — #54, #56, #59, and #60 each spent real work discovering a ceiling that a
measurement would have revealed first. C costs days and directly measures the two quantities that
determine whether A is worth multiple weeks: the achievable crispness ceiling on our data, and whether
corpus size is the binding constraint.

It also does not spend the constraint. Running a model as a **measuring instrument** is materially
different from shipping it as the generator, and that distinction is defensible to a supervisor in a
way that Option B is not automatically. If the oracle shows a large gap, A becomes well-evidenced
rather than speculative — and the distillation arm becomes available to neutralise A's single biggest
risk. If it shows a small gap, we have saved weeks and the honest answer is that 64³ was never really
the enemy.

**Sequence:** recover the meshes (prerequisite for everything) → run the oracle → *then* settle the
A/B/C posture with numbers in hand, including the constraint, thesis, and licensing questions that
only a human can answer.

**What is genuinely undecided and needs the human:** whether the no-frontier-model constraint still
holds, and under which of the three readings — never / only-as-an-instrument / freely.
