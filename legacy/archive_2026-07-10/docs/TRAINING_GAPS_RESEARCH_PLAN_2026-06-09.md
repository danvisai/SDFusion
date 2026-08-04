# Training-Gaps Research Plan — filling the gaps in the generative model

**Date:** 2026-06-09 · **Author:** Claude (for Danvi) · **Scope:** the generative *model* training
(the Stage3a SDEdit massing prior ① + its VQVAE; secondarily the recipe-param head and the ② composer).
Companion to `memory/project_training_audit.md` + `memory/project_gap_fixes_research.md` (this refreshes
and expands them with mid-2026 papers and a measurement plan).

> Why this doc: the prior is **plateaued and partly broken in ways that "train longer" cannot fix**
> (style/class conditioning is ignored; output is soft/lumpy; tall massing is absent; we can't even *see*
> the model improving because there's no held-out eval). Below: (0) how to **track** model change, then
> each gap → the paper(s) that fix it → **what it concretely improves in our model** → cost.

---

## 0. First: make model change *measurable* (answers "gaps in how the model is changing")

Right now `eval_metrics` is a stub and only train loss is logged, and train loss is the *denoise-from-GT+noise*
objective — which the audit shows is **decoupled from generation quality** (low loss, speckle samples). So we
literally cannot see whether a change helped. **Before any paper-driven change, build a fixed eval harness:**

- **Fixed eval set** (~32 held-out buildings + ~8 canonical sculpt edits), frozen, versioned.
- **Metrics per checkpoint:** footprint-IoU (the correct structural metric), **SDEdit faithfulness↔realism curve**
  (IoU-to-edit vs. `strength`), occupancy/volume, vertex count (a soft-surface proxy), per-style **divergence**
  (does varying `style` change the output — the conditioning test), and a **held-out val loss**.
- **Per-checkpoint montage PNG** → `outputs/eval/<run>/<ckpt>.png`, plus a `metrics.csv` appended every eval.
- This is **engineering, not research** (~1 day), but it is the prerequisite for everything below — it turns
  "is the model changing?" into a curve. Closes audit gap #4. Reuse `scripts/foundations/eval_hybrid_style.py`
  + `scripts/sdedit_bag3d_test.py` patterns.

---

## 1. Gap → paper → what it improves (master table)

| # | Gap (status) | Lead paper(s) | What it improves in OUR model | Cost |
|---|---|---|---|---|
| 1 | **Conditioning DEAD** (style/class ignored; footprint `c_concat` dominates, cross-attn ignored). OPEN, biggest | **DiT / adaLN-Zero** (2212.09748); caveat **Hidden Semantic Bottleneck** (2602.21596); labels via **BAG function attr** + **OpenShape** (2305.10764) / **cluster-cond diffusion** (2403.00570) / **RCG** (2312.03701) | adaLN-Zero injects style/class/era/height at *every* block (FiLM-style) so they actually steer massing — the original "style-conditional" goal. Labels give the signal 3D BAG lacks. "Cond > uncond" ⇒ also lifts quality/softness | Med |
| 2 | **Tall/tower massing absent** (3.3% >20m). OPEN | **CBDM** (2305.00562) + **PD-CBDM** (2026); data (row 6) | A height-bin pseudo-class + balancing regularizer rescues tall/tower from mode-collapse → sculpted towers stop being wavy columns | Med |
| 3 | **Guidance** (CFG branch was untrained). Mostly DONE (p_uncond=0.1 in 20k) | **Autoguidance** (2406.02507) + **In-situ AG** (2510.17136) + **interval guidance** (Kynkäänniemi 2024) | No-retrain quality: guide the strong prior with a weak ckpt (we have 5k/10k); interval-guidance lifts quality *and* diversity. Already partly wired | Cheap |
| 4 | **No val / can't track** | — (the §0 harness) | We can finally *see* the model change | Cheap |
| 5 | **EMA / training dynamics**. EMA DONE in 20k | **EDM2** (Karras 2024, NVlabs/edm2) | Magnitude-preserving training + **post-hoc EMA sweep** → less soft/noisy samples, pick best EMA *after* training | Cheap–Med |
| 6 | **VQVAE mismatch**. DONE (clean finetune) → next: co-tune | **REPA-E** (2504.10483) / **REPA** (2410.06940) | Align latent features to a pretrained encoder ⇒ **17–45× faster convergence + sharper**; REPA-E jointly tunes VAE+diffusion (perfect now that the VQVAE was just finetuned). **Highest-leverage training upgrade** | Med |
| 7 | **Corpus narrow** (Dutch-only). OPEN | **BuildingWorld** (2511.06337, AAAI'26) + **GlobalBuildingAtlas** (2506.04106) | ~5M LoD2 buildings, 44 cities, 5 continents → global massing/style diversity + tall coverage + real city/function labels. The data foundation (also feeds rows 1,2) | Med–Big |
| 8 | **Soft/lumpy massing + diversity ceiling**. OPEN | **TRELLIS** structured latents (2412.01506); **CLAY** vecset (2406.13897); **3DShape2VecSet** (2301.11445); **Direct3D-S2 / Sparc3D / XCube** (2312.03806); **Hunyuan3D-2.5** (2506.16504); **LoG3D** (2511.10040) | Replace the 64³ VQVAE with a higher-fidelity / higher-res latent → crisp faces, breaks the recipe/diversity ceiling, scales detail. The real shape-latent prior | Big |
| 9 | **Part-coherence** (heuristic union; floating/duplicate; semantics-blind snap). OPEN (Track 2) | **OmniPart** (2507.06165); **PartGen** (2412.18608); **PartDiffuser** (2511.18801); **SPLICE** (2512.04514); SALAD/SPAGHETTI/StructureNet | Part-aware gen with **structural cohesion**: add a part → it attaches/dedupes/re-coheres; semantic edits (real door/window). The principled ② upgrade | Big |
| 10 | **OSM/urban conditioning** (style assignment is a heuristic) | **Context-informed urban-morphology diffusion** (2409.17049, Feb'26); **PrITTI** (2506.19117); ControlCity | Cross-city style transfer + context/zero-shot conditioning → smarter per-tile style/height instead of class+area heuristics | Med |

---

## 2. Deep dives — the highest-leverage moves

### A. REPA / REPA-E — the biggest *training* win (gap 6, and helps 1/8)
Our prior is under-trained on ~11k buildings and plateaued. **REPA** (Representation Alignment) adds a loss that
aligns the diffusion network's internal features to a frozen pretrained encoder (e.g. DINOv2) and reports
**17–45× faster convergence** and better FID. **REPA-E** unlocks *end-to-end* tuning of the VAE+diffusion together
and is a drop-in that improves convergence/quality across latent-diffusion architectures. **Why it fits us now:**
we *just* finetuned the VQVAE (clean SDFs) — REPA-E would co-adapt the latent and the prior so softness is fixed at
the source rather than papered over with `margin=1.5`+Taubin. Practical recipe: "REPA Works Until It Doesn't"
(2505.16792) — early-stopped, holistic alignment. **Improves:** convergence speed, sharpness, and (because better
features) the conditioning signal-to-noise. This is the single change most likely to make "the model visibly improve."

### B. Make conditioning real (gap 1, the original-goal blocker)
The 20k hybrid eval proved style is **ignored**: same footprint, all 8 styles → identical blob, even at guidance=5,
because the footprint enters on the strong **spatial `c_concat`** path while style/class/era enter on a weak
**cross-attn** path the UNet learned to ignore. Fixes:
- **adaLN-Zero** (DiT, 2212.09748) injects conditioning as per-block scale/shift at *every* layer — empirically
  stronger and cheaper than cross-attn; this is the "FiLM-at-every-block" the audit prescribed. **Improves:** style/
  class/era/height actually steer the output.
- **Caveat — Hidden Semantic Bottleneck** (2602.21596): global adaLN can collapse different conditions to >99%-similar
  embeddings. So pair adaLN with their decorrelation fix / keep a residual cross-attn for fine semantics.
- **Labels** (3D BAG has none): use the **BAG `gebruiksdoel`/function attribute** (cheap real labels at ingest),
  **OpenShape/MV-CLIP** multi-view pseudo-labels, or **cluster-conditioned diffusion** (2403.00570)/**RCG** (2312.03701)
  with self-sup features → k-means cluster-id as a pseudo-class. **Improves:** gives the conditioning something to learn.
- **Important caveat from our own data:** style differences are mostly *detail*, not 64³ *massing* — so even perfect
  conditioning will mostly move massing (height, footprint shape, roof slope), and **named-style/ornament belongs in ②**.
  Condition the prior on **function/height/era** (massing-relevant), not fine style.

### C. Data scale — BuildingWorld (gaps 2 & 7)
3D BAG is clean but Dutch-only and short (median 12 m). **BuildingWorld** (2511.06337) is ~5M LoD2 buildings across
44 cities / 5 continents with diverse morphologies, styles, and **tall** stock, plus structured labels.
**GlobalBuildingAtlas** (2506.04106) adds global polygons+heights. **Improves:** corpus diversity (gap 7), tall/tower
coverage (gap 2), and real city/function/height labels that feed the conditioning fix (B). Ingest a balanced subset via
the existing `scripts/ingest_3dbag.py` pattern (same igl winding-number SDF). **This is the data lever the prior needs.**

### D. Representation upgrade — past the 64³ ceiling (gaps 8 & 9)
The soft/lumpy output and the diversity ceiling are partly inherent to a 64³ VQVAE. The 2025–26 SOTA moved to
**structured/sparse latents** (TRELLIS 2412.01506, Direct3D-S2, Sparc3D, XCube) and **vec-set latents**
(CLAY 2406.13897, 3DShape2VecSet 2301.11445). **Improves:** crisp massing, higher effective resolution, a richer prior
that breaks the recipe ceiling and supports native completion (ideal for SDEdit-style editing). Biggest lift; stage it
after A–C prove the cheaper wins. **LoG3D** (2511.10040) and **Hunyuan3D-2.5** (2506.16504) are reference points for
ultra-high-res / high-fidelity if we go this route.

### E. Part-coherence — the ② upgrade (gap 10, = Track 2)
The snap is a *massing projector*; it can't add a coherent door/window (a carved door fills back in — confirmed in
`outputs/sdedit_bag3d/sdedit_semantic_edits.png`). **OmniPart** (2507.06165) — part-aware generation with **semantic
decoupling + structural cohesion** — is the closest match to "add a part and it integrates without floating/duplicating."
**PartGen** (2412.18608), **PartDiffuser** (2511.18801), **SPLICE** (2512.04514) round out the family.
**Improves:** floating/duplicate parts, replace-not-accumulate edits, real semantic elements. Aligns with
`docs/TRACK2_part_mixing_design.md`.

---

## 3. Sequenced plan (cheap → big), each with how we'll *measure* it

1. **Eval harness (§0)** — *prerequisite*, ~1 day. Deliver the metrics CSV + montage so every later step has a curve.
2. **Guidance, inference-only** — autoguidance (5k/10k guides 20k) + interval guidance. ~hours. *Measure:* faithfulness/
   realism curve + occupancy vs. the current default. (Largely wired; just sweep + lock defaults.)
3. **EDM2 post-hoc EMA sweep** — pick the best EMA length of the 20k run. ~hours. *Measure:* surface softness (vertex count) + IoU.
4. **REPA on the prior** (Theme A) — add the alignment loss; if it lands, **REPA-E** co-tune with the clean VQVAE. ~days.
   *Measure:* convergence speed + sharpness + val-loss curve. **Do this early — biggest training ROI.**
5. **Conditioning fix** (Theme B) — adaLN-Zero injection + BAG-function labels (+ cluster pseudo-labels); keep p_uncond.
   ~days. *Measure:* per-style/per-function **divergence** > 0 (output actually changes), without quality loss.
6. **Long-tail / tall** (gap 2) — ingest tall stock + CBDM/PD-CBDM fine-tune with a height-bin class. ~days.
   *Measure:* tall-massing fidelity (sculpt a tower → crisp, not wavy).
7. **Data scale** — ingest a balanced **BuildingWorld** subset; re-train. ~weeks. *Measure:* diversity + tall + style coverage.
8. **Representation upgrade** (Theme D) — TRELLIS/vec-set latent to replace the 64³ VQVAE. ~weeks. *Measure:* crispness + diversity-ceiling break.
9. **Part-coherence / Track 2** (Theme E) — OmniPart-style part model; runs in parallel (it's the ② layer, orthogonal to ①).

**Recommended immediate focus:** §0 harness → step 4 (REPA) → step 5 (conditioning). Those three convert "the model
isn't visibly changing" into a measurable, faster-converging, actually-controllable prior — without yet paying for the
big data/representation rebuilds.

---

## 4. Sources
- adaLN-Zero / DiT — https://arxiv.org/abs/2212.09748 · conditioning bottleneck — https://arxiv.org/abs/2602.21596
- REPA-E — https://arxiv.org/abs/2504.10483 · REPA early-stop — https://arxiv.org/abs/2505.16792
- Autoguidance — https://arxiv.org/abs/2406.02507 · in-situ AG — https://arxiv.org/abs/2510.17136 · EDM2 — https://github.com/NVlabs/edm2
- CBDM — https://arxiv.org/abs/2305.00562 · PD-CBDM — https://doi.org/10.3390/math14101576
- OpenShape — https://arxiv.org/abs/2305.10764 · cluster-conditioned — https://arxiv.org/abs/2403.00570 · RCG — https://arxiv.org/abs/2312.03701
- TRELLIS — https://arxiv.org/abs/2412.01506 · 3DShape2VecSet — https://arxiv.org/abs/2301.11445 · Hunyuan3D-2.5 — https://arxiv.org/abs/2506.16504 · LoG3D — https://arxiv.org/abs/2511.10040
- OmniPart — https://arxiv.org/abs/2507.06165 · PartGen — https://arxiv.org/abs/2412.18608 · PartDiffuser — https://arxiv.org/abs/2511.18801 · SPLICE — https://arxiv.org/abs/2512.04514
- BuildingWorld — https://arxiv.org/abs/2511.06337 · GlobalBuildingAtlas — https://arxiv.org/abs/2506.04106 · urban-morphology diffusion — https://arxiv.org/abs/2409.17049 · PrITTI — https://arxiv.org/abs/2506.19117
