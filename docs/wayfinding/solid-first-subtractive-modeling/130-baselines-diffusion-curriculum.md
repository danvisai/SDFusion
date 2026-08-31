# #130 — The named baselines, set diffusion, and curriculum

*Effort: solid-first semantic architectural carving. Opened 2026-08-30 from
[#6](6-program-generator.md), which answered five of its eight comparisons — four by measurement —
and left three. Written 2026-08-31. **No training run**, per the ticket.*

> #6 asked for a comparison across eight axes and addressed five. Three were not addressed at all:
> the named baselines (ArcPro, Building-Gym, ShapeAssembly/CSG, CoMa, CityGenAgent), graph/set
> **diffusion**, and **curriculum**.

#6 substituted measurement for reading and said so on its own page. This page does the reading —
and where reading and measuring disagreed, it measured. Two of the three items came back with a
free number attached, and those numbers come from a committed code path (`complexity_strata` and
`label_complexity`, artifacts `execution/artifacts/height_map_generator_strata_714.json` and
`..._714_diagnostics.json`) rather than from this prose. On this project a number that cannot be
re-derived from the repo is an anecdote, and that applies to a literature page too.

⚠️ **This is not a related-work section.** A baseline earns a row here only by saying something
about a decision this effort has already made or measured. Five did. Three of them say something
that changes what a specification should claim.


## 1. The five named baselines

Every representation below was read from the primary paper on 2026-08-31, not from the survey; the
two places where they disagree are in §4.

| | representation | supervision | output space | the claim of ours it touches |
|---|---|---|---|---|
| **[ArcPro](https://arxiv.org/abs/2503.02745)** (CVPR 2025) | a **tree** of `CreateLayer(parent, h, contour)` under one `SetGround`, BFS-serialised to tokens | **100% synthetic.** Forward procedural generation; root contours drawn from 872,487 cleaned Bing Maps footprints, children sampled and validated | stacked flat prisms → mesh, via a learning-free interpreter | ✅ #10's `Layer`; ✅ #6's total compiler; ⚠️ adds a supervision option #6's table has no row for; 🔑 names `Ramp` as its own missing statement |
| **[Building-Gym](https://arxiv.org/abs/2309.02583)** (2023) | ordered actions over a 10×10×10 voxel grid, 2 channels (size, room type), 7 **interior** room types | **synthetic**, from a heuristic agent on randomly generated site conditions | a voxel design state; a ψ layer forbids deleting an existing room | ✅ #6's set head — same structural fact, opposite conclusion; and a *trained* preference model, which is a second answer to #126's scoring problem |
| **[ShapeAssembly](https://arxiv.org/abs/2009.08026)** / **[CSGNet](https://arxiv.org/abs/1712.08290)** / **[PLAD](https://arxiv.org/abs/2011.13045)** | cuboid proxies + attachments (SA); recursive ∪ ∩ − over primitives (CSGNet) | PartNet extraction (SA); policy gradient (CSGNet) or pseudo-labels / approximate distributions (PLAD) | an executable program → shape, via a differentiable or plain interpreter | ✅ PLAD *is* the citation #6's supervision row argues against; ⚠️ SA's hierarchical sequence VAE is the closest published thing to "generate over the program" |
| **[CoMa](https://arxiv.org/abs/2601.08464)** (Jan 2026) | per building, a list of horizontal extrusions — polygon + `bottom_elevation` + `top_elevation` in metres — emitted as **JSON text** | fine-tune Qwen3-VL 2B/4B/8B on CoMa-20K (City of Melbourne open data); also zero-shot Qwen3-VL-235B | tokenised JSON that must parse, then extrude | ✅ #6's totality, with a number; ✅ #126's scorecard, hard; 🔑 its output space fires our **KILL** by construction; ✅ independent support for §3 |
| **[CityGenAgent](https://arxiv.org/abs/2602.05362)** (Feb 2026) | Block Program (`id`, `type`, `polygon` in metres, `floor_count`, `facade` **string**) + Building Program (facade/window/door/**roof** as text descriptors) | SFT for schema validity, then PPO with GPT-4o and VLM-as-judge rewards | footprint × floor count → base mesh, then **asset retrieval** by semantic matching | 🔑 its whole per-building massing is our `blockout` arm; it does not contest this question, it concedes it |

### ArcPro — the closest thing to our compiler, and it names our seam

`CreateLayer(parent = L_j, h = h_i, c = c_i)` is a contour extruded to a height, with a parent
pointer that fixes both the hierarchy and the coordinate frame. That is **exactly** `Layer` from
[#10](10-program-recovery.md), and #10 already recorded the convergence as empirical rather than
borrowed — it fell out of measuring that the corpus is a height field.

Four things, three of which it settles.

**✅ It supports #6's "rejection / repair: not needed", by contrast.** ArcPro predicts a token
sequence that *can be syntactically invalid*, so it needs a finite-state machine masking illegal
next-tokens at inference. `compile_program` needs no equivalent: any assignment, any type and any
plane at all compile to a footprint-exact height map with a voxel under every footprint column. Two
different routes to the same destination, and ours is free because the invalidity does not exist
rather than because it is masked away.

**⚠️ It adds a supervision option #6's table does not have.** #6's row reads "program induction with
pseudo-labels / RL / relaxation → not needed". ArcPro chose **none of those**: it synthesises
programs *forward* from a hand-authored prior and learns the inverse, never seeing a real building's
program. On our data that route is dominated — our labels are exact, free (0.2 s/building, the whole
35,623-row corpus in 56 s), and on **real geometry** — and the survey's own novelty risk #5
("synthetic-generator ceiling") is the reason to want it that way. But the row as written implies a
choice between exact and approximate supervision, and there is a third option that the strongest
architecture-program paper took. A specification should say so.

**🔑 It names the seam.** ArcPro's own Future Work asks for "new statements for more geometric
features, such as curved surfaces or **sloped roofs**". `Ramp` is precisely the statement it says it
does not have — and #10 added it only after reading a worst-residual montage and finding that all
eight failures were smooth roof ramps.

And the price of not having it is already on our record. #6's `flatten_ramps` control takes the
**perfect** program and flattens only its `Ramp`s, which leaves exactly ArcPro's (and CoMa's)
representation with perfect parameters:

| on the 411 carve-needing | `extra` | `missing` | `vs_input` | collapse | **ops** | **planar** |
|---|---|---|---|---|---|---|
| the compiled label as fitted | 0.0035 | 0.0000 | 0.8226 | 0.0000 | 2.0 | 0.50 |
| **the same program, every `Ramp` flattened — a stacked-flat-layer output space at its ceiling** | 0.0528 | 0.0000 | 0.8847 | 0.0024 | **1.0** | **0.00** |

*(`vs_input` and collapse are on this table because #126's rule is about every table, and here they
carry their own point: flattening does not make the arm stop acting — `vs_input` 0.8226 → 0.8847 is
a representation that still carves, just never at an angle.)*

🔑🔑 **A stacked-flat-layer representation, given parameters that are exactly right, fires #6's
pre-registered KILL clause** (`planar ≤ 0.20`) — while passing its `extra` clause at 0.0528 < 0.0603.
That is the trap #6 wrote both halves of the form bar to catch, and the two closest published
massing representations sit squarely in it.

**✅ And its Limitations section is the strongest external evidence on §2.** ArcPro reports that
"structure recovery from sparse point clouds may have multiple valid solutions", that "our method
currently infers only a single solution via **top-1 sampling**", and that "using **top-3 sampling
reduces output quality rather than improving diversity**". That is [#129](129-classified-plane-parameters.md)'s
`argmax`-beats-`circmean` result arriving independently in a different representation. It is not an
argument against joint diffusion — top-k over an autoregressive decoder is not that — but it is a
second measurement saying the cheap version loses.

### Building-Gym — the same structural fact, and the opposite conclusion

Its design state is a 10×10×10 voxel grid with a size and a **room type** per voxel over seven
interior types (elevators, stairs, mechanical rooms, restrooms, corridors, offices, …), under FAR
and target-program-ratio constraints. That is **interior programming, not exterior massing**, so it
is not a baseline anyone can run against a roof. What it does contribute is one mechanism.

🔑 **Building-Gym enforces monotonicity to keep an autoregressive rollout consistent; #6 exploits
monotonicity to delete the rollout.** Its mapping layer ψ exists so that "no existing room from the
previous state gets deleted" — the sequence has to stay coherent because the model reads its own
history. Our vocabulary has the mirror-image property, measured rather than imposed: **every
operation only ever lowers the height map** ([#10](10-program-recovery.md)), so the last operation
to touch a column *is* that column's height, and recording the owner per column replays the whole
cascade in one pass. Same monotonicity, and where Building-Gym pays for a constraint, #6 collects a
lossless set head (`program_to_slots`, pinned by
`test_the_slots_replay_the_fitted_height_map_exactly`).

⚠️ **And it is the second of these five whose supervision is synthetic, for the reason it states
plainly**: "large collections of architectural volumetric designs are prohibitively expensive". With
ArcPro and CityGenAgent that makes **three of five**. #6's "the labels are free" reads like a
convenience on its own page; against the baselines it is the least common thing about this setup and
belongs in a specification as a contribution rather than an aside.

**One thing it has that we do not.** It learns a **preference model** by density estimation over
sequence representations, ~90% accurate against random sequences. [#126](126-massing-scoring.md)
spent a whole ticket on the same problem — how do you score a design when the target does not
determine it — and landed on `extra`/`missing` + `vs_input` + collapse rate. A learned preference
model is a different answer to that question. ⚠️ It is not the only one of the five to engage it —
CoMa's "Contextual Relevance" prompts a general VLM as a judge — but Building-Gym's is *trained on
the design distribution itself* rather than borrowed, which is the version that could transfer.

### ShapeAssembly and CSG induction — PLAD is the paper #6's supervision row argues against

CSGNet parses a shape top-down into recursive union/intersection/subtraction over primitives, and
trains by policy gradient where no ground-truth programs exist. PLAD exists for the same reason,
stated in its first line: "paired (shape, program) data is not readily available for many domains,
making exact supervised learning infeasible", so it compromises either the labels (pseudo-labels) or
the shape distribution (approximate distribution) — and beats policy-gradient RL on both accuracy
and convergence.

**✅ #6's supervision row is right, and this is its citation.** Everything PLAD buys is machinery for
not having exact labels. #10's fitter is deterministic, sees GT, reaches a median `extra` of 0.0035,
and costs 0.2 s per building. None of it is bought.

⚠️ **But ShapeAssembly is the closest published precedent for §2 and should not be filed away with
the rest.** Its generative half is a hierarchical **sequence VAE over programs** — a latent-variable
generative model whose samples are programs, which is the shape of the question set diffusion asks,
one architecture-generation before diffusion. It is also the only one of the five with a
*differentiable interpreter*, which is what you need when your labels are approximate. Ours are not.

⚠️ **And one settled row of #6's has moved, which this comparison is the right place to record.**
"Canonicalise by area; a matching loss is not worth it" was measured at **2.7%** of the plane error
on #6's arm. That arm used 1.19 slots. On [#132](132-overcarve-and-assignment.md)'s arm, which uses
**2.03**, the same measurement reads **15.0%** (`canonicalisation.cost_share`: 0.0269 → 0.1194 →
0.1503 across #6 → #129 → #132). The conclusion probably still holds — 0.0657 of absolute plane
error is not obviously worth a Hungarian matching loss — but the evidence behind it has moved by
5.6× and it was measured when permutation barely mattered. **Program non-identifiability is the
survey's novelty risk #4 and this is the number that tracks it.** Re-check it on any arm that uses
more slots.

### CoMa — the closest published massing generator, and it has no roofs

The survey called this "the closest footprint-and-massing prior". It is a **dataset plus a VLM
benchmark**: CoMa-20K, 20,000 sites from City of Melbourne open data, benchmarked by fine-tuning
Qwen3-VL at 2B/4B/8B to emit tokenised JSON geometry, against zero-shot Qwen3-VL-235B.

**🔑 Its output space is `Layer`-only.** A massing is "a list of horizontal extrusions", each a
polygon with a `bottom_elevation` and a `top_elevation`. Every top face is horizontal. Step-backs and
varied heights, and **no roof form of any kind** — the underlying City of Melbourne table is
footprint polygons "each with an extrusion height". So CoMa-20K contains no example of the object
this ticket has now spent three arms failing to draw, and its representation could not carry one if
it did. Given perfect parameters it lands on the `flatten_ramps` row above: `extra` 0.0528, ops 1.0,
**planar 0.00** — our KILL.

**✅ It supports #6's totality claim with a number.** CoMa's reported metrics:

| | Pattern Match ↑ | **JSON Validity ↑** | ID IoU ↑ | Floor Err ↓ | Area Err ↓ | Site IoU ↑ | Ctx. Relevance ↑ |
|---|---|---|---|---|---|---|---|
| CoMa-8B *(fine-tuned)* | 0.94 | **0.79** | 0.75 | 0.42 | 1.90 | 0.05 | 0.24 |
| Qwen3-VL-235B *(zero-shot)* | **1.00** | **0.99** | 0.99 | 0.12 | 0.79 | 0.10 | 0.25 |

**21% of the fine-tuned 8B model's outputs are not valid JSON**, and CoMa's qualitative analysis
reports self-intersecting polygons and irregular shapes among those that are. Two of its seven
metrics are spent on whether the model produced a parseable object at all. `compile_program` cannot
fail to compile — spill 0 and uncovered 0 **by construction** — so those two metrics have no
counterpart here, and #6's "a prediction can be wrong; it cannot be invalid" now has a published
number for what the alternative costs.

**✅ And it supports [#126](126-massing-scoring.md) harder than anything in the survey.** ⚠️ **None
of CoMa's seven metrics compares generated geometry to the ground-truth massing.** Two are
formatting; ID IoU is a set of identifiers; Floor Error and Area Error are scalar attributes; Site
IoU is the site contour against the union of the *bottom* polygons, i.e. site coverage; Contextual
Relevance is a binary VLM judge. There is no `extra`, no `missing`, no volumetric IoU against the
target building. So the benchmark structurally cannot distinguish a good roof from a bad one — which
is exactly the hole #126 exists to close, and it is worth noting that a January-2026 massing paper
shipped without closing it.

**✅ Third, and it is §3's independent corroboration.** CoMa's own explanation for why its fine-tuned
models produce artifacts: *"the CoMa-20K dataset is dominated by simpler massings, creating an
imbalance that challenges the models' ability to generalize effectively for complex geometric
generation."* That is the curriculum hypothesis, reached on a different corpus, a different
architecture and a different task. §3 takes it seriously for that reason and then measures it here.

⚠️ **Finally: its fine-tuned models lose to an off-the-shelf zero-shot VLM on all seven metrics.**
That is not a verdict on small bespoke models in general — our trunk is a 3.4M-parameter U-Net over a
height field with an exact label, not an 8B model emitting text that must parse — but any
specification citing CoMa as a baseline has to cite this too.

### CityGenAgent — it does not contest this question, it concedes it

The survey calls CityGenAgent "the strongest whole-system novelty risk". On **this** question it is
not a risk at all, and the reason is one field.

    Block Program:  id, type, polygon (metres), floor_count, facade (a natural-language string)
    Building Program:  facade / window / door / ROOF, each a comma-separated text descriptor

🔑 **A CityGenAgent building's entire massing is a footprint polygon and a floor count.** That is one
extruded prism — our `blockout` arm, the do-nothing baseline every table on this effort is scored
against, which on the 411 carve-needing buildings scores `extra` **0.2308** at 3-D IoU 0.8125 and
carves **0.000** of columns. Everything above `floor_count` is a *text* descriptor ("flat slab,
parapet edge, concealed drainage, concrete, clean silhouette") handed to semantic asset retrieval.
The roof is not geometry CityGenAgent predicts; it is a string it looks up.

So CityGenAgent is a real risk to claims about **editable city programs**, **block coordination**,
and **language-driven manipulation** — the survey is right about that, and those are
[#4](4-edit-algebra.md)'s territory and a later ticket's. It is not a risk to any claim about roof
form, and a specification that lists it as a massing baseline has mis-scoped it.

**✅ It is also the third synthetic-supervision paper**, and it says why it needed RL: "Simple SFT on
our limited synthetic dataset reliably teaches BlockGen to produce well-formed block programs, but
does not yield robust spatial reasoning or generalization". #6 declined RL on the grounds that exact
labels are free. CityGenAgent is what the same decision looks like when they are not.

⚠️ One operational note: CityGenAgent expands its asset database with Hunyuan3D text-to-3D. This
project has a standing rule from #61/#66 ([`../crisp-massing-vecset/options.md`](../crisp-massing-vecset/options.md))
that Hunyuan outputs are evidence only and never training data. Nothing here changes that; it is recorded so a future reader
does not import the pipeline wholesale.


## 2. Graph / set diffusion — the position

**What it would buy: the ability to hold a distribution where we currently hold a point estimate.**
And that is not a speculative upgrade, because every failure on this effort's record is the same
disease in a new place:

| ticket | the posterior | the point estimate it collapsed to |
|---|---|---|
| #127 | GT lies inside a column's 80% band on 95% of columns, and that band is **13 voxels wide** | the pointwise median — a mound that is none of the roofs in it |
| #6 | the signed slope of every `Ramp` in the corpus is **exactly symmetric** (mean +0.0009, 50.0% / 49.6%, median &#124;slope&#124; 0.646) | flat — the objective's own Bayes act |
| #129 | the azimuth is **antipodally bimodal by construction** | fixed by re-parametrising so the symmetry lands in one categorical, then `argmax`-ing that one |
| #132 | the assignment head is **diffuse, not wrong** — measured on #129's weights at confidence 0.431, normalised entropy 0.799 | fixed by reweighting the classes in the loss |

🔑 **#129's fix is a hand-built, one-variable sampler.** It works, and it is the only thing on this
map that has ever put a real pitch in a trained arm (realised rise inside a `Ramp`-typed slot
0.00 → **22 voxels** on #129's arm of record, 20 on its endpoint — this ticket's own naming hazard,
so the checkpoint is named). A set
diffusion is the general form of that move, over the joint `(assignment, types, planes)`, without
needing someone to find the right re-parametrisation one variable at a time. Four tickets have now
hit the same disease in four places, and each was fixed by hand.

**⚠️ And "the labels are free" does not answer this.** #6's page discusses supervision and output
distribution in the same breath, and they are separable. A free deterministic label gives you `(x,
y)` pairs; it does not make `p(y | x)` unimodal. The fitter is deterministic **given GT** — but the
conditioning is footprint + height, and [#126](126-massing-scoring.md) measured that this does not
determine the roof: two matched real buildings differ by a **median 3-D IoU of 0.886**. Exact
supervision settles the supervision question and is silent on this one.

### 🔑🔑 What would have to be true, and the first thing is not about diffusion at all

**(a) The bar has to be able to score a sampler, and right now it structurally cannot.** `extra`,
`missing` and 3-D IoU are all computed against **one** GT building, so a model that samples a
*plausible* roof which is not the real one is scored as a failure. We already know what that costs,
from two independent directions:

| a correct one-sample draw, scored at the point metric | `extra` | `vs_input` | collapse |
|---|---|---|---|
| a **real matched building**, offered footprint-exact (#126, carve-needing pairs, n=72) | **0.0974** | 0.8446 | 0.1667 |
| 1-NN retrieval — a nonparametric draw from roofs on near-identical footprints (fp IoU 0.952) | **0.1031** | 0.8743 | 0.1582 |
| **the bar's `extra` clause** | **< 0.0603** | — | — |

Both act (`vs_input` well under the 0.98 guard), so neither is buying its `extra` by declining to
carve — the 0.10 is what a *committed, plausible, wrong* roof costs.

**A correct one-sample generative model of this task scores about 0.10 and fails the bar.** Every arm
on this record that commits to a shape lands at 0.08–0.15 (#6 0.1236, #129 0.1507, #132 0.0832); the
only things below 0.0603 either hedge (#127's served CE+median arm, 0.0603 at **6.0 ops**) or see GT.
Running a diffusion arm against `PROGRAM_BAR` as written is pre-committing it to a KILL for a reason
that has nothing to do with diffusion.

→ **The precondition is a distributional read on the bar**: best-of-n against 0.0974 as the number a
correct sampler has to beat, reported beside the median-of-n so an arm cannot buy the best by being
wild — and the collapse rate on every row, as always.

**(b) The head a sampler would fix has to be the binding one, and it is not.** #132's finding is that
the **type** head is now binding: `Ramp` share of used slots 0.522 → 0.390, 61% of used slots typed
`Layer`, so more regions bought fewer planes each. #132 named the next two moves — a prior adjustment
on the type head, and a **per-slot over-carve guard**, which nothing has yet priced — and both are two
orders of magnitude cheaper than a diffusion arm. Running diffusion first prices the wrong head.

**(c) Naive sampling has now been measured to lose twice, in two representations.** #129's azimuth
`circmean` has lower `extra` than `argmax` and **collapses 50.9% of buildings**; ArcPro's top-3
sampling "reduces output quality rather than improving diversity". Neither is evidence against a
*joint* set diffusion — both are per-variable reads of a factorised posterior — but both say the win
has to come from the joint.

⚠️ **Which is why there is no cheap proxy, and I looked for one.** The obvious free experiment —
sample from the existing per-column assignment posterior instead of taking its argmax — is a
**strawman**, not a price. Sampling 4,096 columns independently would speckle the plan, which is
#127's mound arriving by a fourth route; and #6's whole insight was that a program is a *joint
commitment* across a run of columns rather than 4,096 independent summaries. A measurement that can
only produce a bad number is not worth running, and reporting it as "we priced diffusion" would be
worse than not running it.

### Position

**Not now; second in line, not dead.** Precisely: after the type-head fix and the per-slot
over-carve guard #132 named, and only once `PROGRAM_BAR` carries a best-of-n clause — because
without that clause the arm is scored by a metric that structurally prefers its opposite. If a run is
proposed before then, ⚠️ it needs its own pre-registered bar in #6's form with **both** halves
together, machine-checked in `verdict()`, per this ticket's own instruction.


## 3. Curriculum — the position, and it is measured

#132 left a live thread: `report_program_diagnostics` has printed, since that ticket, that a
minor-slot recall near zero means "a loss **or a curriculum**". #132 turned the loss dial
(`τ·log(prior)` on the assignment logits) and never priced the other one. CoMa reached for exactly
this hypothesis on its own data. And #6's arm collapsed to 1.19 slots where its label uses 3.06 —
which is the shape of failure a curriculum is usually reached for.

So this was asked the way #132 established: **the free question first**, before a schedule is
proposed. `complexity_strata` is new, runs inside `--diagnose_program` off saved weights and the
label cache, and buckets both populations by **the label's** slot count — never by anything the arm
produced.

### What an ordering over complexity would actually feed the head

Training rows only (n = 34,909); the assignment class prior is recomputed **inside** each bucket. No
model appears in this table at all — it is a property of the corpus and of #6's canonicalisation by
owned area.

| bucket | n | share | slot0 | slot1 | slot2 | slot3 | uncarved | slot0 : slot3 |
|---|---|---|---|---|---|---|---|---|
| 0 slots — needs no program | 15,562 | 0.446 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | — |
| 1 slot | 2,568 | 0.074 | 0.8182 | 0.0000 | 0.0000 | 0.0000 | 0.1818 | ∞ |
| 2 slots | 3,556 | 0.102 | 0.5349 | 0.3280 | 0.0000 | 0.0000 | 0.1371 | ∞ |
| 3 slots | 3,051 | 0.087 | 0.4709 | 0.2656 | 0.0964 | 0.0000 | 0.1670 | ∞ |
| 4 slots | 10,172 | 0.291 | 0.3958 | 0.2447 | 0.1712 | 0.0840 | 0.1043 | **4.7×** |
| **≤2 — an easy-first schedule's first phase** | 21,686 | 0.621 | 0.1701 | 0.0520 | **0.0000** | **0.0000** | 0.7779 | ∞ |
| **≥3 — its last phase** | 13,223 | 0.379 | 0.4133 | 0.2496 | 0.1537 | 0.0644 | 0.1189 | **6.4×** |
| ≥1 — carve-needing, the population the bar is set on | 19,347 | 0.554 | 0.4956 | 0.2314 | 0.0997 | 0.0418 | 0.1316 | **11.9×** |
| ALL — what #132's prior was computed on | 34,909 | 1.000 | 0.2495 | 0.1165 | 0.0502 | 0.0210 | 0.5627 | **11.9×** |

🔑 **An easy-first curriculum is backwards on this corpus, and definitionally so.** A building whose
label uses two slots cannot supervise slots 2 or 3 *at all* — they have exactly zero support in the
easy bucket, and 62% of training rows are in it. The first phase of a conventional schedule would
train the assignment head on a population where the classes it is already failing on do not occur,
which deepens the very imbalance #132 had to correct in the loss. The hard bucket is the **less**
imbalanced of the two (6.4× against ∞, and 4.7× on 4-slot buildings alone against 11.9× corpus-wide).
Pinned in `test_a_low_slot_bucket_gives_the_high_slots_ZERO_support`, because it is a property of the
canonicalisation rather than an observation about a run.

### Is the failure graded by complexity? Yes — steeply, on every axis

#132's arm (`heightmap_program_adj`), same 411 carve-needing rows, same buckets, scored the way #126
requires — `vs_input` and the collapse rate beside every median.

| bucket | n | `extra` | `missing` | `vs_input` | collapse | ops | **planar** | slots (arm) | slots (label) |
|---|---|---|---|---|---|---|---|---|---|
| 1 slot | 64 | 0.0257 | 0.0473 | 0.8904 | 0.0938 | 1.0 | **1.00** | 1.56 | 1.00 |
| 2 slots | 66 | 0.0646 | 0.0762 | 0.8451 | 0.1364 | 1.0 | **1.00** | 1.97 | 2.00 |
| 3 slots | 62 | 0.0861 | 0.0431 | 0.8993 | 0.2581 | 1.0 | 0.33 | 2.15 | 3.00 |
| 4 slots | 219 | **0.1360** | 0.0830 | 0.8055 | **0.3425** | 1.0 | **0.00** | **2.16** | **4.00** |
| ≤2 | 130 | 0.0459 | 0.0647 | 0.8675 | 0.1154 | 1.0 | 1.00 | 1.77 | 1.51 |
| ≥3 | 281 | 0.1199 | 0.0679 | 0.8200 | 0.3238 | 1.0 | 0.00 | 2.16 | 3.78 |
| **ALL — the pre-registered population** | **411** | **0.0832** | **0.0659** | **0.8470** | **0.2579** | **1.0** | **0.12** | **2.03** | **3.06** |

⚠️⚠️ **These are post-hoc subgroups and none of them passes anything.** `PROGRAM_BAR` is
pre-registered on the 411 and stays there. #6's write-up already carries one
narrowing-after-the-fact and flags it as the post-hoc move it was; "but it passes on the easy half"
would be the same error committed with a population instead of a clause. The warning is in the
function's docstring and printed above the table, so it cannot be lost in a rewrite. These rows say
*where* the residual sits ([#131](131-vertex-budget.md)'s lesson), not that any part of the arm
succeeded.

Read that way, the grading is unambiguous. Surplus rises 0.0257 → 0.1360 with complexity, the
collapse rate rises 0.094 → **0.343**, and `planar` falls **1.00 → 0.00**. The whole failure lives in
the ≥3-slot buildings.

### 🔑🔑 But the arm's slot count barely responds — and that is the answer

The last two columns are the load-bearing ones. **The label goes 1.00 → 2.00 → 3.00 → 4.00. The arm
goes 1.56 → 1.97 → 2.15 → 2.16.** It over-fragments the simplest buildings and saturates at ~2.16 by
the three-slot bucket, then does not move at all for the four-slot bucket.

🔑 **And it is not one arm's quirk. #6's arm — a different plane head, from before the classified
parameters and before the assignment adjustment — is flatter still:**

| label slots | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| the label | 1.00 | 2.00 | 3.00 | 4.00 |
| **#6 (`regress`)** | 0.98 | 1.02 | 1.19 | **1.31** |
| **#132 (`class` + logit-adjusted assignment)** | 1.56 | 1.97 | 2.15 | **2.16** |

#132's fix moved the whole curve up, which is the K = 1 ceiling breaking — and it did **not** make
the curve respond to the building. Two arms, two plane heads, two assignment losses, and both emit a
near-constant program size.
*(`--diagnose_program outputs/height_map_generator/heightmap_program.pt --out
execution/artifacts/height_map_generator_program_strata_714.json`, which writes
`..._strata_714_diagnostics.json`; that arm's ALL row reproduces #6's record exactly at `extra`
0.1236 / `vs_input` 0.8952 / collapse 0.0073 / 1.19 slots, so the table is the same arm #6 scored.)*

**The arm has learned a constant, not a function of the building.** That separates the two
hypotheses cleanly:

* If it were capacity-limited on hard buildings, its slot count would rise with complexity and fall
  short. It does not rise.
* If it were starved of hard examples, more of them would be the fix. **It is not starved.** 4-slot
  buildings are 10,172 of the 19,347 carve-needing training rows — **52.6%, the outright majority** —
  and the arm still answers 2.16 on them.

🔑 **A curriculum reweights *exposure*, and exposure is not the scarce thing.** So the answer to #130
item 3 is **no**, and it is a measured no rather than a dismissal.

**⚠️ The scarce thing is exposure to the minor SLOTS, and no reordering of buildings can reach it.**
Slot 3 is 0.0210 of footprint columns corpus-wide; inside the 4-slot bucket — the most favourable
population that exists — it is still only 0.0840 against slot 0's 0.3958. That imbalance is created
**inside each building** by #6's canonicalisation by owned area, so it survives any ordering over
buildings. The best a complexity curriculum could do to the assignment prior is 11.9× → 4.7×, and
only by discarding 47% of the carve-needing corpus. **#132's logit adjustment takes it to 1× by
construction and already did**, buying minor-slot recall 0.0000 → 0.2801 under a plain argmax.

**The imbalance is within the example, not between examples. A curriculum can only move the
between-examples dial.**

### What the strata table says to do instead

Two things, and neither is a schedule:

1. 🔑 **The collapse is a complex-building phenomenon that #132's own fix created.** Against
   `class129_at_q025` — the same weights with only the pitch read changed, i.e. before the assignment
   adjustment — collapse by bucket runs 0.0625 / 0.0606 / 0.0484 / **0.0320**: it *falls* with
   complexity, because a single shed plane over a complicated building leaves surplus (`extra` 0.1733)
   without eating anything. #132's adjustment takes the 4-slot bucket **0.0320 → 0.3425, a 10.7×
   increase**, while the 1-slot bucket only moves 0.0625 → 0.0938. So #132's "the two fixes pull
   opposite ways" is not a population-wide trade — it is **entirely a complex-building phenomenon**,
   and the per-slot over-carve guard #132 named now has a stated place to bite.
   *(`complexity_strata` stratifies the one checkpoint under diagnosis, so this cross-arm row is
   derived from the scorecard's committed `per_building` rows joined against `label_slots`, both in
   `height_map_generator_strata_714.json` and both written by the same `label_complexity`.)*
2. 🔑 **There IS one example-selection lever this table does not refute, and it is not an
   ordering — it is the 44.6% of rows that need no program at all.** `train()` draws its pool from
   every row (`ok & not held`), so **15,562 of 34,909 training buildings supervise "predict
   nothing"**, and `uncarved` reads **0.5627** corpus-wide against **0.1316** on the carve-needing
   rows the bar is actually set on. Dropping or downweighting them — in the sampler, or in
   `assignment_prior` — is the one member of the curriculum family that is untested here, and it is
   a one-line change rather than a schedule.
   ⚠️ **But it is a caution, not a recommendation, and the same row says why.** The slot 0 : slot 3
   ratio is **11.9× on both** the `≥1` and the `ALL` rows: excluding the empty-program buildings
   changes only how hard `uncarved` is penalised *relative to every slot* and leaves the imbalance
   *among* the slots exactly where it was. So all it can buy is more pressure to carve — on an arm
   that already eats a quarter of its buildings (collapse 0.2579, and 0.3425 on the 4-slot bucket).
   ⚠️ **And it is a proposed run, so #130's own rule binds it**: before anyone trains it, it needs a
   pre-registered bar in #6's form — *both* halves of form together (`dl_ops` **and**
   `dl_planar_fraction`), the `extra` clause, and the collapse guard, all machine-checked in
   `verdict()` rather than argued in prose. Nothing here lowers that requirement; if anything the
   collapse direction raises it.
   *(I had written this down as a clean win before reading the `≥1` row, which is why that row is in
   the artifact.)*

**⚠️ Two caveats I will not paper over.** This refutes an ordering over *complexity*; it says nothing
about the lever in point 2 above, which is example selection of a different kind. And it is measured
on the **assignment** head — the **type** head is the binding constraint now and this table does not
stratify it. If someone proposes a curriculum for the type head, it needs its own free measurement
first: the same question, asked of the type head's prior.


## 4. Corrections `NOVELTY_SURVEY.md` needs

Two of the survey's characterisations did not survive reading the primary papers. Both are recorded
here and applied to the survey in the same commit, with the survey's cutoff line updated rather than
silently overwritten.

| the survey says | the paper says |
|---|---|
| CoMa "conditions on site contours and context and **emits separate polygonal extrusion sequences**", scored **P** on "learned executable program" and **Y** on "deterministic realization" | It is a **dataset (CoMa-20K) plus a VLM benchmark**. The extrusion list is the *dataset schema*, not a learned executable program, and there is no interpreter — the model emits JSON that is extruded. Its fine-tuned models **lose to zero-shot Qwen3-VL-235B on all seven metrics**, and 21% of the best fine-tuned outputs do not parse. → two cells move, **P → N** and **Y → P**; "coordinated multiple buildings" stays **Y**, because the schema really does carry several identified buildings per site. |
| CityGenAgent is "the strongest whole-system novelty risk", scored **P** on architectural typed +/− | Correct for **editing and block coordination**; wrong for massing. A building's massing is `polygon` + `floor_count` — one prism, i.e. our `blockout` arm at `extra` 0.2308 — and the roof is a **text descriptor** consumed by asset retrieval. |

And one addition rather than a correction: the survey's data strategy says "follow ArcPro's
successful use of cleaned real footprints to ground procedural synthesis". ArcPro's programs are
**entirely synthetic**; the footprints ground the synthesiser, not the supervision. On this corpus
that route is dominated by #10's exact fitter on real geometry, and the survey's own novelty risk #5
says why we should prefer it.


## What this settles, and what it does not

**Settles:**

* 🔑🔑 **The two closest published massing representations cannot express a pitched roof, and we can
  price exactly what that costs**: a stacked-flat-layer output space with *perfect* parameters scores
  `extra` 0.0528 at ops 1.0 / **planar 0.00** on the 411 — passing #6's `extra` clause and firing its
  KILL. Neither ArcPro (which names sloped roofs as future work) nor CoMa (whose corpus has none)
  contests the question this effort is actually stuck on.
* 🔑 **CityGenAgent's per-building massing is our do-nothing baseline.** `polygon` + `floor_count`,
  with the roof as a retrieved asset. It is a novelty risk to #4's editing claims, not to this one.
* ✅ **#6's set head, canonicalisation and totality decisions all survive contact with the
  baselines**, and two of them are *supported* by what the baselines had to build instead: ArcPro's
  FSM syntax mask, and Building-Gym's ψ layer forbidding deletion.
* 🔑 **Three of the five baselines train on synthetic programs**, each stating that real ones are
  unavailable or prohibitively expensive. "The labels are exact, free, and on real geometry" is the
  least common property of this setup and belongs in a specification as a contribution.
* 🔑🔑 **A curriculum over building complexity is refuted, by measurement rather than by argument.**
  The failure *is* steeply graded by complexity — `extra` 0.0257 → 0.1360, collapse 0.094 → 0.343,
  `planar` 1.00 → 0.00 — but the arm's slot usage is a near-constant 1.56 → 2.16 against a label that
  runs 1.00 → 4.00, and 4-slot buildings are already **52.6%** of the carve-needing training rows.
  Exposure is not scarce. The binding imbalance is **inside** each building (slot 3 at 0.0840 even in
  the most favourable bucket), which no ordering over buildings can reach and which #132's
  logit-adjusted loss already flattens. 🔑 **And the constant is not one arm's quirk** — #6's
  `regress` arm is flatter still, 0.98 → 1.31 across the same buckets. ⚠️ Scoped to an ordering over *complexity*, and to the
  *assignment* head — see the two caveats and the one untested example-selection lever in §3.
* ✅ **Set diffusion is the right general answer to the disease this map has hit four times**, and it
  is second in line rather than dead — with a precondition that is not about diffusion at all.

**Does not settle:**

* ⚠️ **The bar cannot currently score a sampler.** A correct one-sample generative model scores
  `extra` ≈ 0.10 (a real matched building offered footprint-exact: 0.0974; 1-NN: 0.1031) against a
  clause set at 0.0603. Until `PROGRAM_BAR` carries a best-of-n read, any generative arm is
  pre-committed to a KILL for the wrong reason. **This is the cheapest unblocking change on the
  effort and nothing has been done about it.**
* ⚠️ **"A matching loss is not worth it" was measured on a K = 1 arm.** The canonicalisation cost has
  gone 2.7% → 15.0% of the plane error as the arm learned to use two regions. The conclusion probably
  holds; the evidence behind it no longer says what it said.
* ⚠️ **Nothing here prices the per-slot over-carve guard**, which #132 named and which the strata
  table now localises to the ≥3-slot buildings (collapse 0.0320 → 0.3425 under #132's own fix).
* ⚠️ **The type head is still untouched and still binding**, and the curriculum question has been
  answered only for the *assignment* head. An ordering aimed at the type head is not refuted here; it
  is unmeasured.
* ⚠️ No baseline here is **runnable**, so this comparison is representational and a specification
  should say so rather than promise numbers it cannot produce. Building-Gym's task is interior room
  programming on a 10³ grid; ShapeAssembly and CSGNet are not architectural. For the other three the
  survey's "no official code release located" still looks right but ⚠️ **my recheck was narrow and I
  will not overstate it**: a GitHub repository search on 2026-08-31 returned nothing for ArcPro or
  CoMa-20K and only a project-page repo for CityGenAgent, and ArcPro's project page
  (`vcc.tech/research/2025/ArcPro`) is JavaScript-rendered so its static HTML settles nothing.
  Recorded as *not found by that search*, not as *not released*.
