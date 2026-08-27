# #10 — Program recovery on real buildings

*2026-08-26. CPU only, no GPU touched, no training. The four #92 arms kept the A100 throughout.*

Asset for [#10](https://github.com/danvisai/SDFusion/issues/10), answering the first item on
[#1](https://github.com/danvisai/SDFusion/issues/1)'s *"Not yet specified"* list: whether
constrained architectural volumes can fit real LoD2 massing without a free-form residual branch.

Code `scripts/foundations/recover_massing_programs.py`; artifacts
`execution/artifacts/program_recovery_714.json` (K=4) and `..._k16.json` (K=16); traces
`outputs/program_recovery/{worst,representative}.png`. Scored on the pre-registered 714 held-out
buildings from `massing_arms_eval_ship714.json`.

---

## 🔑🔑 The corpus is a height field, not a volume

Measured before any fitter was written, on all 714 held-out buildings:

| measurement | result |
|---|---|
| `missing` of the blockout against GT | **0.000000 on 714/714** |
| carve volume *above the topmost GT voxel in its column* | **100.0%** (4,324,848 / 4,324,919) |
| carve volume in through-voids (courtyard / passage / light well) | **0 voxels** |
| carve volume in overhangs (GT above the carve) | **71 voxels** |
| columns that are not a solid run from the base | **4 of 1,072,438** |
| GT columns outside the footprint mask | **0** |

So every building in this corpus is exactly `{(z,y,x) : y0 <= y <= top(z,x)}` — a **64×64 height
map**. Consequences, in order of how much they change:

1. **The massing task is 2.5-D.** It is footprint + height → a 64×64 height map. The vecset latent
   diffusion has been solving an image-to-image problem in a 3-D set-of-tokens representation that
   destroys the structure it needed.
2. **`SubtractCourtyard`, `CutNotch`, `CutEntrancePassage`, `CreateArcade` are dead operations
   here.** They cannot fire: there are no through-voids at all at LoD2/64³. They stay in the
   algebra design ([#4](https://github.com/danvisai/SDFusion/issues/4)) but are untestable on this
   data.
3. **Additive operations are dead too**, which `missing`=0 already implied.
4. This is precisely the object ArcPro's `CreateLayer` grammar produces — vertically extruded
   polygonal layers. The convergence is empirical, not borrowed.

⚠️ This is a statement about **this corpus at this resolution**, not about architecture. A courtyard
narrower than `s*` cannot be represented at 64³ regardless of whether the real building has one.

## The vocabulary that follows

    Layer(height, polygon)       one connected region flattened to one height   (ArcPro CreateLayer)
    CutRoof(kind, eaves, rate)   height falls off with distance from the nearest edge;
                                 kind = hip | gable_x | gable_z
    Ramp(region, slope)          the tightest PLANE above the target over one region

Fitted by **beam search** (width 12, branch 10), not greedily -- see below.

`ApplySetback` is **not** a separate operation: in a height field a setback *is* a Layer whose
polygon is the inward offset of the footprint, and the greedy fitter finds it as one.

Median connected components per height level is **1.00** — each level really is a single polygon.

## Results — like-for-like on the 411 carve-needing buildings

The 303 buildings whose blockout already equals GT (`extra` < 0.02) are reported separately and
never pooled; #80's bimodal result is the precedent, and a 42% no-op majority flatters every
aggregate.

| arm | 3D IoU | missing | extra | vs_input | collapse | beats envelope |
|---|---|---|---|---|---|---|
| gt | 1.0000 | 0.0000 | 0.0000 | — | 0.0000 | — |
| blockout | 0.8125 | 0.0000 | 0.2308 | — | 0.0000 | — |
| codec_ceiling | 0.9977 | 0.0007 | 0.0015 | — | 0.0049 | — |
| deployed_map24 | 0.5192 | 0.0504 | 0.7152 | — | 0.1630 | **0.5%** |
| a2_s0.5 (shipped) | 0.7736 | 0.0027 | 0.2357 | 0.9852 | 0.1241 | **1.2%** |
| **program, K=4 (beam 12)** | **0.9826** | **0.0000** | **0.0177** | 0.8340 | **0.0000** | **100.0%** |
| **program, K=16 (greedy)** | **0.9981** | **0.0000** | **0.0019** | 0.8192 | **0.0000** | **100.0%** |

🔑 On the buildings that actually need a carve, the shipped A2 model is **worse than its own
input** — 3D IoU 0.7736 against the blockout's 0.8125, `extra` 0.2357 against 0.2308 — and
collapses on 12.4% of them. The recovered program is the first arm in this project's history to
beat the envelope on more than a rounding error, and it does so on **411/411**.

`collapse_rate` is 0.0000 **by construction**, not by luck: a fitted height may never drop below
the target height, so the program cannot cut into GT. `missing` is 0.000000 for the same reason.
The 303 already-flat buildings recover with the **empty program**, exactly, at 3D IoU 1.0000.

## Program simplicity

| K | median residual `extra` | % under the 0.02 allowance | % recovered exactly | (Layer+CutRoof only) |
|---|---|---|---|---|
| 0 | 0.2308 | 0.0% | 0.0% | 0.2308 |
| 1 | 0.1031 | 17.3% | 4.1% | 0.1121 |
| 2 | 0.0591 | 25.5% | 8.0% | 0.0696 |
| **4** | **0.0280** | 41.4% | 15.6% | 0.0367 |
| **5** | **0.0197** | 50.4% | 18.2% | 0.0282 |
| 8 | 0.0083 | 69.8% | 21.2% | 0.0151 |
| 16 | 0.0019 | 92.9% | 32.4% | 0.0051 |

Ops needed to reach the allowance: p25 **2**, p50 **5**, p75 **9**; **92.9%** get there within 16.

Operation mix at K=4 with beam 12: `Layer` **75.4%** of removed volume, `Ramp` **23.4%**,
`CutRoof` **1.2%**. (The K-curve above is the **greedy** fit; the beam was run at K=4, where the
bar sits.)

🔑 At K=16 the program reaches 3-D IoU **0.9981**, *above* the codec ceiling of 0.9977 — it
compiles straight to the voxel grid and never passes through the Dora codec, so that ceiling does
not bind it.

## The pre-registered bar — met, but only on the third attempt

The bar was *median residual `extra` <= 0.02 at K <= 4*, with a kill criterion at *fewer than 50%
of carve-needing buildings reaching the allowance within 4 ops*. Both are now met — **0.0177** and
**51.8%** — but the first fit missed and it took two changes to get there. The whole sequence, so
the result is not read as a first-try pass:

| fit | median `extra` at K=4 | verdict |
|---|---|---|
| greedy, `Layer` + `CutRoof` | 0.0367 | not met |
| + `Ramp` operation | 0.0280 | not met |
| + beam search 6/6 | 0.0214 | not met |
| **+ beam search 12/10** | **0.0177** | **met** |

Neither change touched the bar, the metric, or the population. Both were found by reading the
worst-residual montage.

### What the first miss actually was

The residual was **roof-slope terracing**, the staircase a sloped roof makes on a 64³ grid, which
`Layer` steps through one tread at a time.

🔑 **Reading that trace found a missing operation.** All eight worst residuals were smooth roof
ramps, and every program was `Layer > Layer > Layer > Layer` — `CutRoof` never fired on the shapes
it exists for. The cause: `CutRoof` measures distance to the *nearest* footprint edge, which is
symmetric, so it can express a gable or a hip but **never a shed**. Buildings also sit at arbitrary
grid rotations, so an axis-aligned ramp would not have fixed it. `Ramp` — the tightest plane above
the target over a region, found by a 3-variable LP — took median ops-to-allowance from **7 to 5**
and reached-within-16 from **85.2% to 92.9%**.

🔑 **Reading the trace again found a search failure, not a second missing operation.** What was
left were **symmetric double ramps**: a gable rises from both eaves to a ridge, so no single plane
dominates it and it needs two opposing `Ramp`s. Greedy never gets there — one large flat `Layer`
always wins the immediate gain, and by the time the surplus has split into two regions the budget
is spent. Beam search fixes exactly that, and the operation mix confirms the mechanism: `Ramp` goes
from 12.3% of removed volume under greedy to **23.4%** under beam 12.

⚠️ **A beam search does not automatically dominate greedy.** The greedy path can be pruned at an
intermediate step by siblings that look better then and finish worse — measured on id 16764, 0.152
greedy against 0.159 beam. `fit_program_beam` therefore runs greedy as well and returns whichever
program is actually better, which makes the beam a monotone improvement by construction.

Also tested and **rejected**: forcing a roof operation first. Greedy reaches the allowance in a
median 5 ops on a 150-building sample against roof-first's 6 (84.7% vs 82.7% reached).

## ⚠️ The visual check qualifies the bar

`extra` is a volume fraction, and it hid a shape failure that a shaded 3-D view makes obvious.
`outputs/program_recovery/iso_real_vs_recovered.png` and `iso_op_budget.png` render the massing
isometrically (CPU only — pyrender/EGL hangs on this node while the #92 arms hold the GPU, and a
height field needs no marching cubes anyway: one top face and two side faces per column, painter's
order by `x + z`).

On the tail, **K=4 does not read as a building**. Ids 21991 and 21393 are smooth sloped roofs in
reality; at K=4 the program returns a **ziggurat of three or four giant steps**. Raising the budget
fixes it outright:

| | id 21991 | id 21393 |
|---|---|---|
| K=4 | 0.185 — a staircase | 0.138 — a staircase |
| K=12 | 0.026 | 0.049 |
| **K=24** | **0.003 — visually identical to real** | **0.001 — visually identical to real** |

🔑 So the pre-registered bar is met on the **median** (`extra` 0.0177 at K=4) while the **tail is
visually wrong at that budget**. Both statements are true and neither should be quoted alone. This
project's stated priority is *visual first, footprint match second*, so for label generation the
budget should be set by the visual criterion — **K≈24**, not K=4. K=4 remains the right number to
report against the bar as written, because that is what the bar asked.

⚠️ This is the third time in this ticket that reading a picture corrected a conclusion the scalar
metric supported.

## Limits, stated

- **This measures expressiveness, never learnability.** It shows a program *exists* for each
  building; it says nothing about whether a network can predict one from a footprint. The whole
  #69–#92 history is the gap between those two.
- ⚠️ **The 42%-empty-program majority is the copy incentive in new clothes.** A model that always
  predicts the empty program already scores 3-D IoU 1.0000 on 303/714. Any generator built on this
  must be scored on the carve-needing subset, as this document does.
- Op counts remain an **upper bound**: beam 12 is still not an exhaustive search, and widening it
  kept paying (0.0367 → 0.0280 → 0.0214 → 0.0177). The true minimum program is shorter than
  reported, so program length should be read as "at most this", never as the DSL's real token cost.
- Beam 12/10 costs **~4 min** for 714 buildings against ~50 s greedy. Fine for label generation,
  irrelevant to inference, but it is not free.
- `Layer` polygons are recovered as voxel regions, not simplified polygons. Turning a region into a
  polygon with a vertex budget is unstarted, and is where a real DSL token count would be decided.
- Not novel as a method: `NOVELTY_SURVEY.md` §3 already records that synthetic architectural
  programs on real footprints, predicted by a Transformer and deterministically compiled, is
  established prior work. Nothing here changes that.

## What follows

- The height-field finding is **route-neutral and bigger than this ticket**. It applies equally to
  the voxel route ([#113](https://github.com/danvisai/SDFusion/issues/113)), which is currently
  specified as *binary segmentation of the start box* — on this corpus that is over-parameterised
  by a factor of 64, since the label is a height map, not a volume.
- [#4](https://github.com/danvisai/SDFusion/issues/4) can drop the void operations from the
  testable vocabulary on this corpus and say why.
- [#6](https://github.com/danvisai/SDFusion/issues/6) now has a concrete supervision target with
  exact labels, where identical geometry gives an identical program — which is what
  `docs/research/why-pair-training-does-not-carve.md` said the supervision had to move to.
