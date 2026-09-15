# #182 — Scoping geometry-derived style embeddings with text-assisted refinement

*Fact-finding report, not a decision. [#182](https://github.com/danvisai/SDFusion/issues/182) itself
says scoping "has not been done at all" and exists "to record the idea, not to answer it." This
report does that scoping pass: it inventories what already exists in this codebase that bears on the
idea, names the constraint any implementation must reconcile with, and lays out the open design-space
questions as candidate decision tickets. No model code changed. Needs a `/grilling` pass against the
project owner before anything below is implemented.*

## What #182 actually asks

Originated from a deferred idea during [#171](https://github.com/danvisai/SDFusion/issues/171)'s
grilling interview. Two distinct mechanisms, explicitly not one:

1. **Content-derived style signal** — extracted from a reference building's or footprint's own
   geometry (roof form, vertex/edge characteristics), not looked up from a fixed table.
2. **Text-assisted refinement** — a person further adjusts that derived signal with a text prompt.

#182's own body is explicit that this is materially different from #171's `region_id_of` /
style-bucket channel (a one-hot naming which pipeline/place-group a training row came from) and
explicitly out of map #156's scope. No parent map exists yet.

## Existing precedent in this codebase

Nothing here answers #182 outright, but four things already sit in the repo that bear directly on
it — none of them wired into the current production generator.

### 1. Stage3a already has a discrete style/era/floors/region conditioning stack

`models/stage3a_model.py` (the dense-grid SDF-diffusion predecessor to A2) has, gated behind flags:

- `class_emb = nn.Embedding(53, 32)`, `style_emb = nn.Embedding(9, 16)` (8 manually-picked style
  recipes + 1 "unknown") — `style_id` is a hand-picked bucket, not geometry-derived.
- `era_emb` / `floors_emb` (`use_extra_cond`) and `region_emb` (`num_regions`, gated identically to
  `use_extra_cond`) — a region/culture token predating #171's own decision on the same axis.
- A `neutral_style` argument to `sdedit()` (`self.style_id = torch.full_like(..., neutral_style)`)
  for **style-isolating guidance** — i.e. classifier-free-guidance-style steering already exists for
  a style embedding on this model.

### 2. The exact ablation #182 implies was already run once, for manual buckets, then deleted as dead

`git show 1ba0add~1:scripts/foundations/eval_hybrid_style.py` (removed by `1ba0add`, "103 modules
that nothing imports and no live doc names" — dead code, not a finding retraction) ran: *same
footprint, vary `style_id` — do the generated buildings differ by style?*, from-noise and via SDEdit,
with style-isolating guidance. That is #182's evaluation question, already asked and tooled, just for
a hand-picked bucket rather than a derived one, and on the abandoned Stage3a path.

### 3. Text/image conditioning mechanisms exist upstream, unused by anything current

- `models/sdfusion_txt2shape_model.py` + `models/networks/bert_networks/network.py`
  (`BERTTextEncoder`) — a from-scratch-trained BERT-style transformer text encoder, cross-attended
  into an SDF latent-diffusion UNet (`self.df(x_noisy, t, **cond)`). This is the base SDFusion
  project's original text-to-shape path, inherited scaffolding, never trained on this project's
  massing/building vocabulary.
- `models/networks/clip_networks/network.py` (`CLIPImageEncoder`) — a CLIP **image** encoder used by
  `models/sdfusion_img2shape_model.py`. This project's own `models/` code has no CLIP *text* tower;
  the only text encoder this project trains or serves is the from-scratch BERT one above. A CLIP text
  tower does exist, vendored: `external/TripoSG`, `external/Hunyuan3D-2`, and `external/DiffSplat`
  each import `CLIPTextModel`/`CLIPTextModelWithProjection` for their own pipelines. None of that is
  wired to this project's massing generator, but it's a concrete, already-vendored reference
  implementation if a CLIP-style text tower is the direction chosen (open question 3 below).

### 4. The closest existing "geometry → style-ish label" precedent is rule-based, not learned

`roof_family()` in `scripts/foundations/pilot_buildingworld_roof_families.py` turns a *fitted edit
program* (via `fit_program_beam`) into one of `flat`/`gable`/`hip`/`complex` — a discrete,
rule-based classification off recovered plane geometry, reusing three pieces of existing precedent
per its own docstring. It is not a learned embedding, and it is expensive: it requires running the
beam-search program fitter, not a cheap function of raw mesh geometry.

Cheaper precedent for hand-crafted geometric descriptors exists too, but only for 2D footprints:
`#173`'s opt-in perimeter²/area and solidity channels, and `probe_height_inference.py`'s `feats()`
(area, perimeter, compactness, bbox, elongation from a 64×64 footprint mask). Nothing in the repo
today extracts a 3D roof/vertex/edge descriptor as a cheap, differentiable, or even just fast
function of raw geometry the way #182 imagines.

`docs/research/SDFUSION_RELATED_WORK.md` already surveys related work broadly and separately notes
Stage3a "conditions on footprint, class/style, height, and region-related data paths" and flags
CityCraft's "add district/road/neighborhood features to style and retrieval decisions" as an
adjacent idea — worth reading alongside this report rather than duplicating here.

## The constraint this must reconcile with

The current production/paper-claim generator is **A2** (`scripts/train_vecset.py`,
`models/networks/vecset_denoiser.py`), not Stage3a. Its own header is explicit and was written as a
design constraint, not an oversight:

> **Conditioning is purely geometric — footprint, height, region. No text, no images. That is
> load-bearing for the contribution claim, which is footprint-ONLY generation.**

And per `docs/adr/0003-two-claim-thesis.md` (C1): generation *is* projection — SDEdit from a
footprint blockout, never a from-noise sample at this data scale (`vecset_projection.py`'s
`from_noise` is "DIAGNOSTIC ONLY... never the claim"). A signal computed from a reference building's
*own geometry* sits comfortably inside that framing — it's a measurement off real geometry, the same
category of thing footprint/height/region already are. **Free-text conditioning does not** — it's a
different, non-geometric input axis the "footprint-ONLY" claim was written specifically to exclude.

Practically: this is not a drop-in extra input to A2. It is one of:
- reopening/qualifying the footprint-ONLY contribution claim (a real research-claim change, not an
  implementation detail), or
- shipping as a clearly separate, ablatable arm/checkpoint that never touches the claim — the same
  discipline #171 applied to `region` (kept ablatable, retrain ladder includes a drop-region-entirely
  control) rather than silently baking a new signal into the served model.

The geometry-derived half could plausibly extend A2's existing `region` conditioning slot (same
cross-attention mechanism, same classifier-free-guidance `drop_cond` path already wired in
`vecset_denoiser.py`/`vecset_projection.py`). The text-assisted-refinement half has no compatible slot
on A2 at all today — it would need either a new arm or a return to Stage3a-style dense-grid
conditioning, which the project already moved away from for surface-quality reasons unrelated to
conditioning.

## Cautionary precedent, from the ticket that spawned this one

#171's own decision comment recounts finding an apparent "regional" difference (JP rows reading
100% flat-roofed) that traced to an unrelated *ingestion gap*, not culture — which is why "cross-
cultural conditioning" was retired in favor of the mechanically honest "style bucket" framing. A
**learned or rule-based geometry-derived signal is at least as exposed to this failure mode as a
manual bucket, arguably more so**: nobody eyeballs a learned embedding's axes the way a human names
8 style buckets, so a capture-pipeline artifact, a watertightness defect, or a CRS/units quirk could
get encoded as "style" and nobody would notice without deliberately checking for it. Any scoping
decision needs an explicit plan for telling real architectural variation apart from pipeline
artifact — not just a held-out FID number — before training anything.

## Open design-space questions

Sharper versions of #182's own list, now grounded in the above:

1. **Geometry feature extraction.** Rule-based (extend `roof_family`'s discrete classification) vs.
   hand-crafted scalar descriptors (extend #173's pattern to roof pitch / vertex count / edge-length
   statistics) vs. a learned point-cloud/mesh encoder (e.g. small PointNet/DGCNN-style network
   trained from scratch, since nothing like that exists in this repo today). Cost, interpretability,
   and artifact-risk differ sharply across the three.
2. **Where it attaches.** New cross-attention conditioning path into A2's denoiser (reopens the
   footprint-ONLY claim) vs. a guidance-time-only technique — a latent offset/style vector applied at
   sampling the way `guidance`/`drop_cond` already steer `region`, never touching training — vs. a
   wholly separate arm that extends Stage3a's already-built style/era/region machinery instead of
   touching A2 at all.
3. **Text-conditioning mechanism**, if pursued at all: reuse the legacy from-scratch `BERTTextEncoder`
   (implemented, never trained on this domain's vocabulary) vs. add a pretrained CLIP text tower
   (mirrors the existing CLIP *image* encoder, but off-the-shelf CLIP text-image alignment was
   trained on photographic captions, not massing/roof vocabulary, so transfer is unproven) vs. an
   LLM-driven adapter mapping free text to a small discrete/continuous style adjustment — cheaper,
   more controllable, and closer in spirit to #171's own finding that "region" is best understood as
   a manually-picked preset rather than an auto-derived lookup.
4. **Evaluation / falsification.** What would demonstrate this earns its complexity over the
   already-decided, working style-bucket mechanism? Needs a pre-registered bar and a drop-signal
   ablation, the same discipline #171's retrain ladder applies to `region` — not just a demo that
   outputs "look different."

## Candidate child tickets — NOT YET FILED, proposed for review

| # (proposed) | Title | Type | Blocked by |
|---|---|---|---|
| A | Decide whether a geometry-derived style signal is compatible with A2's footprint-ONLY contribution claim, or must ship as a separate ablatable arm (ADR 0003) | human decision | none — gates B–D |
| B | Survey/prototype geometry-feature-extraction candidates for a content-derived style signal (rule-based / hand-crafted descriptors / learned encoder) and report artifact-risk for each | ready-for-agent research spike | A |
| C | Decide the text-conditioning mechanism (reused BERT scaffold / CLIP text tower / LLM-driven style adapter), or defer text entirely to a later phase | human decision | A |
| D | Pre-register a falsification criterion and drop-signal ablation for a geometry-derived style arm before any training | human decision | A, B |

These are drafts for review, not filed on GitHub. #182 has no parent map yet per its own text
("likely deserves its own map once scoped") — filing A–D either under a new map issue or by
promoting #182 itself into that role is a call for the project owner, not this report.

## Recommendation

Run a `/grilling` session against this report to decide ticket A first — it gates everything else,
the same way #167 gated #171. Nothing below A should be scheduled before it resolves.

## Status: not decided

Leaving open for review per this repo's convention for consequential research/scoping tickets —
@danvisai please have a look, starting with candidate ticket A.
