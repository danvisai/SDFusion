<!-- Mirrored from the tracker, 2026-08-14. -->

# Map #87 — Token order in the pair training target, and whether fixing it gives the transform a usable band

*State: open · opened 2026-08-09 · mirrored 2026-08-14*


> Local mirror of the map. The tracker remains the source of truth; this copy exists
> because the effort's committed assets were lost with the machine that made them.


## Destination

A decided answer on whether correcting the **token-order corruption in the pair training target**
gives the A2 transform a **usable editing band** — a strength at which it makes a net-positive edit to
the footprint envelope instead of either returning it or shredding it.

**Execution carries into this map.** The cache rebuild and the retrain happen here, not after it.

### Why this map exists

Map [#69](https://github.com/danvisai/SDFusion/issues/69) closed with one untried candidate in its fog:
a *re-ordering-invariant parameterisation*. The premise was checked before chartering
(`docs/research/latent-token-order.md`, `1d68b4c`) and it **holds**:

- Dora picks its 2048 tokens by farthest-point sampling over a **random surface draw**, so token *i* is
  the *i*-th selected surface point and a fresh draw rotates the whole ordering.
- On the actual training pair, the element-wise token distance is **1.093** against a random-pairing
  floor of **1.089** — **100.3% of the way to random** — while the *matched* distance is **0.037**.
- `train_vecset.py:300` sets the ε target to `sqrt(a)/sqrt(1-a) * (zb - z) + eps` on pair steps, which
  are **80% of training**. That difference is element-wise, so the bridge the model learns is a
  difference between unrelated locations.
- The decoder is **permutation-invariant** (max field difference 6.8e-06, occupancy IoU 1.000000), so
  token order is pure gauge and the fix costs nothing in reconstruction.

⚠️ Plain (non-pair) steps are **not** affected — their target is self-consistent within the sample. Only
the cross-latent bridge is corrupted.

### Pre-registered success criteria, in priority order

Fixed **before** the run, because this model's curve is non-monotonic and #69 twice re-litigated a
result after seeing it.

1. **Visual (primary, human-judged).** A shaded montage against the footprint envelope, `vecset_v4_surf@240k`,
   and GT on the same held-out ids. A human says whether the massing reads as a real building. **No
   scalar overrides this.**
2. **A usable band exists (the claim of this map).** On the **full 714** held-out set, at the strength
   that maximises median 3D IoU, **all three** must hold:
   - `vs_input` median **< 0.98** — the transform actually acts (today: **0.985**);
   - median 3D IoU **≥ 0.876** — no quality paid for it (today: **0.876**);
   - beats the envelope on **> 5%** of buildings (today: **0.7%**, 5 of 714).
   Failing any one is recorded as **not met**. A transient value at one checkpoint is **not** a pass.
3. **Footprint fidelity — spill ≤ 5% and uncovered ≤ 5% at s\*** (ADR 0004, 3 voxels @64³), judged by
   the human on the **worst-first plan view**. Today **77.0%** of 714 pass. Must not regress.
4. **3D IoU split into missing vs extra — diagnostic only, never pass/fail.**

## Notes

**Domain:** footprint-conditioned building massing; vecset latent diffusion; the C1 transform.

**Settled upstream, not up for re-decision here:**
- Posture **A2** — pretrained Dora codec + our own footprint-conditioned denoiser.
- Generation **is** projection (ADR 0003) — envelope → partial noise → denoise. Never from-noise.
- The codec **stays frozen** ([#77](https://github.com/danvisai/SDFusion/issues/77)); a fine-tune would
  *move* the manifold this map is trying to land on.
- **Height is a user input** ([#81](https://github.com/danvisai/SDFusion/issues/81)); the task is
  (footprint + height) → massing.

**⚠️ Standing traps, all of them paid for on #69:**
- 🔑🔑 **NEVER extrapolate this training curve.** #75 went 0.719 → 0.657 → **0.532** → **0.840** by
  epoch; the band-fix run went **0.200 → 0.825** between adjacent checkpoints. A stop was recommended
  at a dip **twice** and was wrong **twice**.
- 🔑 **Report `vs_input` beside every quality number.** A near-no-op inherits the envelope's perfect
  footprint and is scored for it.
- 🔑 **Score on the full 714, never a prefix.** `pick_ids` row order tracks source corpus; the old
  pinned 48 were **100% Dutch** and voided three headline figures.
- ⚠️ **Medians lie on bimodal outcomes** — always publish a collapse rate beside the median.
- ⚠️ **`extra` improves for the wrong reason** when the model under-builds; a scorecard led by `extra`
  ranks the weaker arm better. This hazard has fired **three** times.
- ⚠️ `surface_roughness` is anti-correlated with the goal. **SNE** ([#79](https://github.com/danvisai/SDFusion/issues/79))
  is the crisp-vs-melt instrument, reported never ranked on.
- ⚠️ Array frame is **[z, y, x]**; the Frame-N reorder is a **reflection** that inverts winding —
  `ensure_outward` is the guard.
- ⚠️ A passing verification can still be wrong (#62 aligned at IoU 1.0000 with normals inverted).

**Compute:** A100 80GB, ~305 ms/step, so a 60k-step run is ~5 h. Not a constraint, and this map is
directed to **buy the informative experiment rather than the cheap one**: #92 runs a **2x2** (~20 h)
instead of a single arm, because one run against a differently-built cache would be unattributable.
Confounds get cleared **before** a long run, not explained after it.

🔑 **The design that makes this affordable:** alignment is a *permutation of the same tokens*, so
#91 derives the aligned cache from the unaligned one with no second encode. The two #92 arms are then
literally the same numbers in a different order — a single-variable comparison, out of one encode pass.

**Skills:** `/grilling`, `/domain-modeling`, `/diagnosing-bugs`, `/prototype`.
**No sub-agents** — research is done in-session.

**Assets:** `docs/wayfinding/latent-token-order/`. Do not write into other efforts' folders.

**Prior art to zoom, not re-derive:** `docs/research/latent-token-order.md` (the premise check),
map [#69](https://github.com/danvisai/SDFusion/issues/69) (closed — 15 tickets, and its Decisions-so-far
keeps voided numbers beside their replacements: **anything from the n=48 era is not quotable**).

**Control arm for everything here:** `logs_building/vecset_v4_surf/vecset_denoiser_step240000.pth` —
fp-IoU 0.958 / missing 0.002 / extra 0.092 / 3D IoU 0.876 / `vs_input` 0.985 on n=714
(`execution/artifacts/massing_arms_eval_ship714.json`).

## Decisions so far

<!-- one line per closed ticket -->

- [Does the decoded-surface loss escape the ordering corruption?](https://github.com/danvisai/SDFusion/issues/89) — **YES, and the map's premise survived the control that could have voided it.** Swept across the schedule, the epsilon loss varies **1.53% / 0.97% / 0.31%** at t/T 0.40 / 0.55 / 0.70 while the surface term varies **0.0003% / 0.0005% / 0.0001%** — **1,900–5,300× less order-sensitive**, and order-insensitive at *every* t, which matters because the shipped `v4_surf` **predates `--surf_t_center`** and its own surface term ran at ≈0.40. Decoder permutation-invariance is now pinned by a test rather than a docstring. 🔑🔑 **The control:** reordering `zb` changes the bridge by **1.007** and the cosine goes 0.039→0.032, so today's order is worth no more than random — which reads as *"nothing to align"* and is the **wrong contrast**. Against the **matched** order: as-encoded **+0.0527** · permuted **+0.0366** · **matched +0.3235** (~6×, and 16.8× the identity-vs-random gap). Both halves are needed. ⚠️ **Two corrections to this map's own framing**: the bridge is *not* mostly noise — **51.1%** survives averaging over orderings and the loss moves only ~1%, so the corruption degrades what can be *learned* rather than making the loss meaningless; and *"the only uncorrupted signal"* is left **explicitly unreconciled** with that. ⚠️ #84's tension filed as [#94](https://github.com/danvisai/SDFusion/issues/94). ▶️ **Not a recommendation of a method** — #90 still owes a Morton-sort latent cosine, bijectivity and stability. ▶️ **Re-weight #92 down**: a ~1% swing is not a catastrophe being repaired. Asset `89-surface-loss-immunity.md` (`4dcdccd`).

> **2026-09-01 recovery check:** the reconstructed actual-training-path probe reproduces the decision.
> Epsilon spread is **1.407% / 1.039% / 0.286%** and decoded-surface spread is
> **0.0000618% / 0.0001749% / 0.0000345%**, making the surface term **5,941–22,751×** less
> order-sensitive at every measured t. Artifact `execution/artifacts/surface_loss_order_probe.json`.

- [Capture the codec's token query positions](https://github.com/danvisai/SDFusion/issues/88) — **done, and the question changed on contact.** 🔑 For the **existing** caches the positions are not merely expensive to recover, they are **gone**: `DoraCodec` holds one generator that every `encode` advances (so a draw depends on every building before it, and the loop's `[skip]` desynchronises the rest), **and** `sample_uniform` accepted an `rng` and **ignored it** — the coarse stream, the bulk of the encoder's input, came from numpy's **global** generator, so every cached latent is a function of global interpreter state at write time. Captured during encoding instead, via `DoraCodec.encode_with_positions` (patches the FPS name *inside the encoder's module*, since it holds a direct reference), with `ShapeCodec.reseed` and per-**row** seeding making one building reproducible on its own. **Decision the ticket asked for: store POSITIONS, not a permutation** — matching is a function of *two* position sets and cannot be baked into a single-latent permutation before #90 chooses; fp16 (resolves ~1e-3, **30×** finer than the voxel pitch) halves 875 MB → **437 MB**. ⚠️ **The first guard was useless and #89 is why**: decoding a latent at its own positions cannot see a within-building permutation, because the decoder is permutation-invariant — the exact property I had just proven. `verify_positions` now **re-encodes** through the same `encode_row` the write loop uses: stored-vs-re-encoded **0.0002** against a shuffled **1.0584** and a cross-building **1.1148**. ✅ Acceptance met from disk with nothing encoded (99.9% of the way to random vs the doc's 100.3%). 🔑 **Forced a correction to #89**: its matched cosine was **+0.3235**, really **+0.7288** — the published figure **understated** the recoverable alignment by >2×, strengthening #90/#91. ⚠️ Production caches deliberately **not** rewritten (the shipped checkpoint and #92's control depend on them); corpus-scale capture belongs to #91. Asset `88-query-positions.md` (`e20f9af`).

- [Does an arbitrary token order at inference break a model trained on aligned pairs?](https://github.com/danvisai/SDFusion/issues/95) — **no: order at inference is NOISE, not bias. #91/#92 proceed.** ⚠️ **This ticket's own premise was wrong**, and running it literally is what showed that: it predicted decoded fields "must be identical" under a token permutation and would otherwise have triggered "the whole plan needs rethinking". Measured on real weights at 2048 tokens — permute tokens **and noise** → max field diff **7.36e-04** (equivariant), permute tokens **only** → **2.03** (the full ±1 range). The two are different claims; at inference the noise is independent of token order, so a permuted input is a different *sample*, not a broken symmetry. Quality across 5 orderings of the same envelope: range median **0.0208** / mean 0.0851 / **max 0.5386**, and **17 of 24 buildings move <0.05 while three move >0.20**. 🔑 The deciding number is the aggregate: **SE on a median over 714 = 0.00132**, at the harness's own 0.001 noise floor and ~20× under the +0.029 being chased — **and in #92 both arms see the same envelope in the same order, so it cancels in a paired difference** (now a requirement on #92). ✅ **All of #90's methods survive** — matching is not disqualified by having no partner at inference. ⚠️ **But the sort's tie-breaker evaporates**: a canonical Morton order at inference scores **−0.0150** vs as-encoded, slightly *worse* and inside the noise, so #90 chooses on alignment quality and cost alone. 🔑 **Unexpected: the collapse is order-triggerable** — row 23903 scores 0.969/0.944/0.974/**0.436**/0.973 across five orderings, and the unstable buildings carry the lowest `vs_input` (0.57–0.89 vs 0.986 median), i.e. the ones the model actually acts on, at #73's tolerance boundary. Gives #84/#94 a property their size/solidity/region analyses lacked. ⚠️ **Two process failures recorded**: the probe reintroduced the Dutch-only trap by reimplementing `pick_ids` (10/10 region 0; it changed the numbers and flipped the Morton sign), and both review sub-agents died on a session limit so this ticket had a **self-review** only. Asset `95-inference-order.md` (`3c4bac5`).

- [Choose the token alignment: canonical sort, or explicit matching](https://github.com/danvisai/SDFusion/issues/90) — **greedy matching at k=256.** n=102, 34 per region. Cosine: as-encoded **0.0405** · morton **0.2112** · **greedy 0.5387** · hungarian 0.5106 · nn (upper bound, NOT a permutation) 0.7079. Greedy wins cosine, cost (**4.93 h** vs 9.66 h) and unmatched mass, and ties stability. 🔑🔑 **Hungarian wins total position error and LOSES latent agreement** — total distance was never the objective, only the proxy, and minimising the *sum* degrades a token with an excellent partner to rescue one with none. The far/near split proves it: greedy genuinely matches **36.7%** of tokens against Hungarian's **21.2%**, 73% more real correspondences. **Optimising the proxy harder made the real objective worse.** ⚠️ **The bijection costs a quarter of the headline**: #89's 0.73 was `cdist().argmin`, many-to-one and unusable for reordering; a real permutation reaches **0.5387**, still 13× today's 0.0405, and **#92 must expect the smaller number**. ⚠️ **~63% of tokens are unmatched** (beyond a voxel, mean 0.2783 vs 0.0186 for the matched) — the envelope and the real building are genuinely different surfaces, which is why the ceiling is 0.71 not 1.0. ❌ **Morton rejected**: 0.2112, least stable, and #95 already killed its inference-time advantage. ⚠️ **I nearly chose wrong twice the same way** — measured greedy unrestricted (0.5596, beats Hungarian), then added a k-NN restriction for speed and re-measured (0.4881, below Hungarian), reporting each as a finding while **the algorithm changed between them**; the k sweep (16→0.4783, 64→0.5141, 256→0.5329, 2048→0.5392) settles it and the default sits at the knee, pinned by a test. Asset `90-alignment-choice.md` (`26e19e8`).

- [Rebuild the blockout cache in the aligned order, with a write-time guard](https://github.com/danvisai/SDFusion/issues/91) — **built, 11.3 h.** Three caches, 35,623 rows each, 714 held out: real (`_v2`), envelope **as encoded** (#92's control), and the same latents **permuted** (#92's treatment). 🔑 The last two are *literally the same numbers in a different order*, so **#92's arms differ in token order and nothing else** — the confound in this map's original charter is gone. **Acceptance, read from disk with nothing encoded:** unaligned **99.7%** of the way to a random pairing, aligned **13.7%** (elementwise 1.1138 → **0.1904**, matched floor 0.0423). Corpus token cosine **0.5172** against #90's as-encoded 0.0405 (⚠️ different samples, so the 12.8× is indicative not paired). Guards: frame 0.9970/1.0000, positions 0.0001/0.0002 against a same-building **shuffle** at 1.0557/1.0892; on the aligned cache **35,623/35,623 token-wise permutations**, **35,623/35,623 reordered**, positions followed at **0.1762**. ⚠️🔑 **The guard could not have proven what I claimed** — review found the identity check read a loop variable *after* the loop (so it tested one row), the permutation check compared 131,072 *scalars* rather than 2,048 *token vectors*, and this ticket's explicit ask to extend `verify_positions` to the aligned cache **was skipped**. All fixed and re-run against the already-built cache via `--verify_only` (minutes, not another 5.68 h); the cache is sound, but that distinction is on the record. Timings for the next sizing: real 2.04 h · envelope **3.04 h** (an EDT + marching cubes per building; #78's precedent under-budgets it) · alignment 5.68 h. ▶️ **Also delivered: `watch_checkpoints.py`** — GT | envelope | model on fixed region-stratified buildings for every checkpoint, `vs_input` in every caption, running outside training. Assets `91-cache-rebuild.md`, `91-rebuild.log` (`9b9bdda`).

> **2026-08-15 correction to #91:** the lost cloud caches above were rebuilt against the current
> corpus: **36,818 rows / 715 held out** in each of the three caches. Exhaustive readback proves
> **36,818/36,818** token-wise permutations and reordered rows, with positions following exactly
> (**0.0** error); query-position distance is **1.135835 -> 0.183678** against a **0.037845** floor,
> or **13.2818%** of the random span. See the final section of `91-cache-rebuild.md` and
> `execution/artifacts/align_cache_v2.json`.

- [Retrain with the aligned pair target](https://github.com/danvisai/SDFusion/issues/92) — **NEGATIVE: alignment does not restore a usable band.** All four arms reached 240k and all six 10k checkpoints were scored on the same 714 IDs. B's best strength is 0.45: 3D IoU **0.8753**, `vs_input` **0.9938**, collapse **10.64%**, beats envelope **5.74%** — it passes only the last clause of the registered AND bar and visually returns the extrusion. Strength 0.30 acts but costs quality (0.8075); 0.70–0.85 collapse into rubble (91.88–100%). At the matched 240k/0.5 endpoint B scores 0.7616 against A's 0.8573 and collapses 46.36% against 8.96%. #89's side-prediction holds: the surface term's endpoint marginal changes from **+0.0241 IoU / -9.0 collapse points** encoded to **-0.1093 / +27.3 points** aligned. The candidate never meets the bar, even transiently, so the gated from-scratch aligned run is not triggered. Assets `92-aligned-retrain.md`, `issue92_2x2_summary.json`.

- [Re-measure the strength band on the retrained model](https://github.com/danvisai/SDFusion/issues/93) — **a band exists, but for under 10% of the corpus.** Reading #92's own full-714 arm-B sweep per-building instead of in aggregate (`scripts/foundations/analyze_issue93_strength_band.py`): a strength exists where a building is acted on, survives, and beats its own envelope for **69 of 714 (9.66%)**. 🔑 **38 of those 69 have exactly one working strength, and 19 of those 38 work only at 0.60** — a setting with a 27.17% collapse rate, higher than every other sampled strength short of 0.30 (32.63%) and the 0.70/0.85 wipeout — so the population a higher strength helps is not the population 0.45–0.50 helps; no single dial setting serves both. ⚠️ **This qualifies #92's own headline**: of the 67/714 (9.38%) rows #92 counted as "beats envelope" at strength 0.5, **47 are no-ops** (`vs_input ≥ 0.98`, IoU nudged up while the building barely moved), leaving only **20/714 (2.8%)** that genuinely acted and improved — the map's own `vs_input`-beside-every-number trap, firing on #92's own table. Not a reopening of #92: the candidate still does not meet the registered bar. Asset `execution/artifacts/issue93_strength_band_armB.json`.

- [Why does weighting up the surface term collapse the model?](https://github.com/danvisai/SDFusion/issues/94) — **not disagreement; a magnitude imbalance that compounds under alignment.** A gradient-conflict probe (`scripts/foundations/probe_surface_gradient_conflict.py`) computes d(eps_loss)/d(pred) vs d(surf_loss)/d(pred) directly — the exact point `train_vecset.py` sums the two losses. ❌ **Candidate 3 (the terms disagree) is refuted**: mean cosine similarity between the two gradients stays under 0.005 in magnitude at every measured t, in both regimes, at two checkpoints (individual rows reach 0.0145, still two orders of magnitude short of real opposition) — orthogonal, not opposed. 🔑 The real effect is **magnitude**: at the weight A/B actually train with (`surf_weight 1.0`) the surface gradient is **49–128x** the epsilon gradient's norm at the arms' shared starting checkpoint, **in both regimes almost equally** — so alone it doesn't explain why B collapses harder than A. ⚠️ **But it grows specifically under alignment as training proceeds**: on each arm's own diverged step-220000 checkpoint (A on its own encoded data, B on its own aligned data), B's norm ratio is **1.4–2.3x** A's at every t, tracking that checkpoint's own collapse gap (10.22% vs 18.21%, widening to 8.96% vs 46.36% by 240k). Candidates 1 (decoder-path intolerant) and 2 (displacement) predict the same observable and this measurement can't separate them; what it adds is that alignment doesn't introduce a new failure mode, it compounds an existing one. ⚠️ Not a second controlled trial — the 220k read uses each arm's own 40k-step-diverged model, the confound the 180k shared-checkpoint measurement was built to avoid. Not urgent per the ticket: B is already a rejected branch (#92); this closes the diagnosis without reopening whether it ships. Assets `94-surface-weight-collapse.md`, `execution/artifacts/surface_gradient_conflict_probe.json`, `..._armA_step220000.json`, `..._armB_step220000.json`.

## Not yet specified

- **If alignment does not restore a usable band, what replaces the element-wise loss?** Candidates not
  yet sharp enough to ticket: a set-based objective (Chamfer or optimal transport directly on tokens),
  or predicting in a parameterisation where the tokens carry their positions explicitly so distance
  means geometric distance. Only becomes specifiable once the aligned retrain has a result.
- **Does the aligned target survive a bijection?** #90 measures it. But if forcing a permutation costs
  most of the 0.048 → 0.73 gain, the approach needs re-thinking rather than re-parameterising, and what
  replaces it is not specifiable until #90 reports.
- **Whether the corrupted bridge explains the *strength cliff* specifically**, as opposed to the
  no-op behaviour generally. #73 attributed the cliff to decoder intolerance measured model-free; both
  can be true, and their relative share is untested.

## Out of scope

- **Fine-tuning or replacing the codec** — settled against in #77, and it would move the manifold.
- **Architectural detail** — windows, facades, ornament. Massing only.
- **Demo integration** — map [#50](https://github.com/danvisai/SDFusion/issues/50) /
  [#86](https://github.com/danvisai/SDFusion/issues/86). A better checkpoint drops into the shipped
  endpoints without changing them.
- **Retraining or repairing the dense-grid map-#24 stack.** Five documented negatives.
- **Inferred height** — [#82](https://github.com/danvisai/SDFusion/issues/82) measured it; it is a
  different task definition and never comparable to a specified-height number.


---

## Comment — danvisai, 2026-08-14

**Control-arm band data from the town demo**, recorded on [Re-measure the strength band on the retrained model](https://github.com/danvisai/SDFusion/issues/93#issuecomment-5289480706).

Wiring A2 into the footprint-town demo (map #97) drove `vecset_v4_surf@240k` over a population it was never evaluated on — hand-drawn and OSM-extracted footprints. Short version, relevant to this map's premise that the transform may lack a usable band:

- On a plain convex blockout the band is a **cliff, not a dial**: strength 0.5 → `vs_input` 1.0000 with spread 0.0000 (a literal no-op), 0.7 → 0.0108 with the vertex count collapsing 15.2k → 2.8k.
- **Past the cliff it is not reproducible**: the L-shape swings 0.52–0.75 across identical requests (spread 0.23, ~30x the spread at 0.5).
- The two shapes fail in **opposite** directions — the rectangle collapses toward empty, the L gets noisier (15.2k → 49k verts) — so one scalar at one strength hides which failure is happening.

Caveat that bounds all of it: drawn footprints have no ground truth, so this measures departure from the blockout, not quality. It is control-arm data on an out-of-corpus population, not a measurement of the retrained model.
