# Footprint to clean building: diagnosis and adoption path

**Research date:** 2026-08-22  
**Status:** review finding; proposed input to future Wayfinder and architecture decisions  
**Question:** Why does the current footprint-to-massing path fail to turn a supplied 2D footprint
into a clean, recognizably architectural building, and should the missing mechanism be a
whole-volume voxel transform?

**Scope:** diagnosis and experiment sequencing. This note does not change `CONTEXT.md`, an ADR, the
symbolic recipe, or production code. It records what the repository evidence already establishes,
what remains untested, and the conditions under which a future result may be incorporated.

## Executive finding

Whole-volume voxels are a technically appropriate **state representation** for testing massing
correction at `64^3`. They are not, by themselves, a learning mechanism and should not be understood
as persistent cubes that a model physically moves.

The present failure is upstream of meshing:

1. the footprint envelope is already a very strong target approximation, so ordinary paired
   training is heavily rewarded for copying it;
2. the current vecset pair target is expressed between independently ordered token sets, so most of
   the apparent latent difference is unrelated to the geometric carve;
3. projection strength behaves as a cliff between no-op and collapse, not as a stable
   faithfulness/realism control; and
4. the query-based codec can faithfully represent the required geometry, so replacing the decoder
   does not address the demonstrated failure.

The missing experiment is therefore not “adopt a larger diffusion model.” It is a spatially aligned,
change-aware correction task on authentic A2 outputs, with exact footprint and physical-validity
constraints. The cheapest falsifier is a small deterministic whole-volume occupancy corrector. A
stochastic discrete-diffusion arm is justified only if deterministic correction first establishes
that useful signal exists.

For durable production state, a semantic architectural edit program with deterministic SDF/CSG
realization remains the stronger interpretation and editability contract. A successful voxel result
would be geometry evidence that must pass a later recipe-compatibility decision; it would not silently
replace the symbolic recipe.

## Desired behavior

The user-visible contract is:

```text
2D footprint + declared height/context
                  |
                  v
clean, solid, building-like massing
                  |
                  +-- adheres to the supplied footprint
                  +-- has architectural roof/setback/wing/void structure
                  +-- remains grounded, connected, and usable
                  +-- supports reversible future editing
```

At the repository's fixed massing/detail scale, the target includes roofs, setbacks, wings,
courtyards, passages, and other structural-scale volumes. Windows, doors, facade articulation,
materials, and appearance remain downstream detail.

## Reproducible failure signal

The fixed held-out artifact is
[`execution/artifacts/massing_arms_eval_ship714.json`](../../execution/artifacts/massing_arms_eval_ship714.json).
The registered transform gate requires all three of:

```text
median vs_input < 0.98
median 3D IoU >= 0.876
strict beats-envelope rate > 0.05
```

Replaying that gate on the shipped `n=714` artifact gives:

| measure | observed | required | verdict |
|---|---:|---:|---|
| median `vs_input` | 0.9846 | < 0.98 | fail: too close to the envelope |
| median 3D IoU | 0.8756 | >= 0.876 | fail, narrowly |
| strict beats-envelope rate | 0.0070 | > 0.05 | fail: 5 of 714 |

The replay command is:

```bash
jq -e '(
  .per_building["a2_s0.5"] as $a |
  .per_building.blockout as $b |
  {
    vs_input: .summary["a2_s0.5"].vs_input,
    vol_iou: .summary["a2_s0.5"].vol_iou,
    beats_envelope_rate: (
      ([$a | to_entries[] |
        select(.value.vol_iou > $b[.key].vol_iou)] | length) / .meta.n
    )
  }
) as $m |
if ($m.vs_input < 0.98 and
    $m.vol_iou >= 0.876 and
    $m.beats_envelope_rate > 0.05)
then $m
else error("RED transform gate: " + ($m | tostring))
end' execution/artifacts/massing_arms_eval_ship714.json
```

Its diagnostic value is not merely that A2 misses one threshold. It says the transform almost never
improves on the deterministic object it started from.

## What the held-out geometry says

On the same 714 buildings:

| arm or relation | result |
|---|---:|
| footprint-envelope median 3D IoU to target | **0.9334** |
| maximum target volume missing from the envelope | **0.0000** |
| A2 median 3D IoU to target | **0.8756** |
| A2 median overlap with its envelope input | **0.9846** |
| A2 median extra volume | **0.0922** |
| A2 median missing volume | **0.0024** |
| query-codec round-trip median 3D IoU | **0.9986** |

The target is entirely contained by its starting envelope on every held-out row. In this fixed
benchmark, the principal geometric lesson is therefore subtractive:

```text
filled footprint envelope -> classify material as KEEP or REMOVE
```

The complete A2-to-real correction task may still contain additions because A2 itself can under-build
or erode material. An authentic whole-volume editor must therefore permit both `EMPTY -> SOLID` and
`SOLID -> EMPTY`, but it should not let the rare addition class obscure the established envelope
carving signal.

## Ranked diagnosis

### 1. Copy incentive dominates — supported

The envelope already achieves 0.9334 median 3D IoU. A source-conditioned model can obtain most of the
paired objective by preserving it, and the current A2 output does exactly that at `vs_input = 0.9846`.
Changing the diffusion loss does not remove this incentive.

A future task must make edits identifiable and valuable. Available mechanisms include:

- direct absolute occupancy supervision with change-aware weighting;
- an auxiliary `KEEP / ADD / REMOVE` loss reduced immediately to absolute occupancy;
- correction-opportunity sampling separated from identity controls; and
- masked/block corruption if a later diffusion arm is intended to rebuild genuinely missing state.

Identity examples remain necessary to measure preservation, but they must not dominate a correction
objective.

### 2. The vecset pair direction is spatially corrupted — supported, final treatment untested

A vecset latent is an unordered set of 2,048 tokens. The blockout and real-building caches were
encoded independently, so token `k` in one set generally has no geometric relationship to token `k`
in the other.

The measured unaligned pair distance is 1.3837, versus 1.3965 for a randomly chosen different
building. Explicit token alignment improves the relationship substantially:

| measurement | unaligned | aligned |
|---|---:|---:|
| rank correlation between latent gap and geometric carve | 0.436 | **0.710** |
| gap separation, carved versus zero-carve rows | 0.018 | **0.114** |
| same-index token cosine | 0.041 | **0.464** |

However, geometrically identical rows still have an aligned latent distance of 0.904. The correction
signal is improved but remains a modulation on a large representation floor.

The aligned treatment Arm B in the preregistered 2x2 has not yet run. This prevents a claim that the
vecset route is exhausted. Run Arm B before using a new representation as evidence that alignment
could not rescue the existing model.

See [Why pair training does not carve](why-pair-training-does-not-carve.md) and
[#92 — aligned retrain](../wayfinding/latent-token-order/92-aligned-retrain.md).

### 3. Projection strength is a cliff — supported

On a plain rectangular footprint, the shipped model returns the envelope at strength 0.5 and nearly
empties it at 0.7. The L-plan can instead become increasingly noisy and fragmented. The same scalar
therefore hides opposite failure modes, and no stable middle band has been demonstrated.

This is evidence against treating strength tuning as the missing fix. Re-measure strength only after
the target representation and supervision change.

See [#93 — strength band](../wayfinding/latent-token-order/93-strength-band.md).

### 4. The prior may lack enough architectural signal — open

Footprint plus height underdetermines roof family and some structural choices. The current copy
failure prevents a clean test of whether the corpus nevertheless contains enough conditional signal
for plausible architectural correction.

The discriminating experiment is masked or explicit whole-volume reconstruction on authentic real
building targets. If a model cannot rebuild a removed roof/setback region from the remaining real
body and footprint, the limitation is closer to data, conditioning, or model capacity. If it can do
that but cannot correct authentic A2 output, the deployment-source formulation remains the problem.

### 5. Decoder or meshing is the bottleneck — falsified for volumetric fidelity

The query codec round-trips held-out ground truth at about 0.9986 median 3D IoU, and the earlier
ceiling probe showed that query-based decoding can reach the ground-truth crispness range. This does
not guarantee that every terminal surface will look good, but it rules out the codec as the cause of
the present no-op/carving failure.

Surface recovery remains a separate gate because binary occupancy discards sub-voxel surface
location. Any continuous recovery candidate must preserve the corrected occupancy signs exactly and
be compared against deterministic signed-distance recovery.

See [the crisp ceiling measurement](../wayfinding/crisp-massing-vecset/ceiling-probe-result.md).

## What “whole-volume voxel rearrangement” should mean

Do not assign identity or a trajectory to individual voxels. A dense occupancy grid is a fixed set
of spatial addresses:

```text
V[z,y,x] = 0  EMPTY
V[z,y,x] = 1  SOLID
```

An edit changes cell states:

```text
source 1 -> output 0    remove material
source 0 -> output 1    add material
```

This gives the property the vecset target lacks:

```text
identical geometry -> exactly zero difference
```

Every cell may be eligible for learned change without implying that every cell must change. The
primary model endpoint should be absolute binary occupancy, not a persistent action lattice. An
auxiliary action head is useful for class balance and diagnosis but must reduce deterministically to
the same absolute result.

At `64^3`, dense state storage is small. The limitations are instead:

- grid resolution and stair-stepped surface extraction;
- severe imbalance between `KEEP` and changed cells;
- unrestricted models producing holes, fragments, or erosion;
- ambiguity among several plausible buildings for the same footprint; and
- lack of semantic operation identity and independent editability.

These limitations require losses, conditioning, hard validity checks, surface recovery, and an
explicit recipe decision. They do not make whole-volume occupancy intrinsically unworkable.

## Existing prototype evidence and its boundary

The current CPU smoke command is:

```bash
./venv/bin/python scripts/foundations/prototype_voxel_editor.py smoke
```

On seed 0 and the default six steps it reports:

| check | result |
|---|---:|
| first synthetic loss | 0.2437 |
| last synthetic loss | 0.2266 |
| source median IoU | 0.9242 |
| learned edited median IoU | 0.9242 |
| oracle-action median IoU | **1.0000** |
| changes outside the permitted mask | **0** |

This establishes only that the action representation, application logic, and exact preservation
mechanics are expressible. It does **not** establish learned improvement: the six-step model did not
move the median. More importantly, the existing prototype uses a roof/surface mask on synthetic
examples, whereas the current specification requires complete whole-volume eligibility and authentic
A2 inputs.

Treat this smoke as plumbing evidence only.

## Adoption assessment

| candidate idea | useful contribution | why it is not the direct fix |
|---|---|---|
| DIF-Net / ShapeFlow deformation | correspondence and meaningful transport of existing geometry | smooth deformation is a poor primary mechanism for cuts, new voids, or attached components; neither enforces the footprint contract |
| CSGNet-style inverse parsing | recovers an inspectable Boolean program from final geometry | this repository should predict and retain the recipe before realization, not discard it and guess it back afterward |
| DVD-style discrete voxel diffusion | native categorical occupancy, block corruption, clamped known state | the released checkpoint models sparse general-object surface scaffolds, not filled building solids; diffusion is premature before deterministic correction passes |
| ArchComplete-style voxel hierarchy | building-specific completion and coarse-to-fine refinement | its house domain and native generation task do not establish post-hoc correction of authentic A2 outputs |
| deterministic procedural roofs/CSG | exact footprint adherence, crisp surfaces, reversible named operations | requires a learned or authored decision layer to choose among plausible roof, setback, courtyard, and wing programs |
| small deterministic whole-volume corrector | cheapest test of whether authentic A2-to-real spatial correction is learnable | produces anonymous occupancy and therefore still needs a surface and recipe-compatibility gate |

The literature mechanisms and their learning objectives are explained in
[How learned 3D shape editing actually works](learned-3d-shape-editing-primer.md). The masking and
diffusion variants are compared in
[How a diffusion model learns to fill voxels](voxel-diffusion-fill-mechanisms.md).

## Recommended experiment order

### Step 1 — finish the current single-variable latent test

Run the aligned Arm B from the existing preregistration. It is the cheapest experiment capable of
showing whether corrected correspondence rescues the current vecset transform. Report the full
714-building gate, collapse, `vs_input`, strict envelope wins, and visual montage.

Do not infer Arm B's result from cache statistics alone.

### Step 2 — run deterministic authentic whole-volume correction

Use the implementation-ready contract in
[Build the authentic whole-volume A2 voxel-correction feasibility experiment](https://github.com/danvisai/SDFusion/issues/125):

```text
inputs
  exact A2 occupancy
  normalized read-only A2 source field
  footprint
  metric height and normalized vertical extent
  region / existing massing conditions

model endpoint
  absolute EMPTY/SOLID probability for every 64^3 cell

constraints and reporting
  identical footprint sanitization for baseline and learned arms
  grounding / connectivity / thickness / cavity validity
  missing and extra reported separately
  vs-input and strict paired wins
  identity controls and correction-opportunity strata
  fixed-frame plan / facade / isometric / section evidence
```

Train first on the fixed 384 authentic pairs and screen once on the outcome-blind 96. Only a frozen
small-gate pass may open the fixed 714-building confirmation.

Add a source-dependence ablation: replace authentic A2 evidence with the deterministic envelope while
holding footprint/height/context constant. If performance does not fall, the editor is ignoring A2
and is effectively a second footprint-to-building model rather than an A2 corrector.

### Step 3 — decide whether stochastic diffusion adds value

Only after the deterministic model passes the full gate, ask whether several plausible roof or
structural outcomes are useful. A discrete-diffusion arm should operate on the same absolute binary
state, use an explicitly declared corruption process, and retain identical hard validity and
evaluation contracts.

Diffusion should earn its complexity through useful diversity or better conditional quality, not
merely reproduce deterministic correction with more sampling cost.

### Step 4 — resolve recipe compatibility

A passing voxel editor supports one opaque, reproducible whole-volume transform. It does not provide
independent identity for a courtyard, roof, wing, or setback.

The future decision must choose explicitly among:

1. retain the whole-volume transform as an opaque rerollable recipe operation;
2. distill/propose its result into a semantic architectural edit program;
3. keep it as evidence or a ranking prior but not production state; or
4. reject the route if recipe locality and reversibility cannot be preserved.

Until that decision, the accepted production direction remains:

```text
footprint / user gesture
          |
          v
learned semantic architectural decisions
          |
          v
deterministic extrusion + SDF/CSG realization
          |
          v
derived occupancy, SDF, and terminal mesh
```

## Success and kill criteria

The small whole-volume screen should require at least:

- median paired IoU gain of `+0.01` over sanitized A2;
- at least 55% strict wins on authentic correction-opportunity rows;
- collapse increase no greater than two percentage points;
- median overlap with sanitized A2 below `0.99`, proving that the model acts;
- identity-row median loss no worse than `-0.005`; and
- no hidden or unreported validity projection.

The fixed full-population confirmation must additionally require:

- more than 5% strict beats-envelope;
- no collapse, identity, footprint, or physical-validity regression;
- improvement on the sealed 618-row complement, reported separately from the screen;
- exact occupancy agreement across surface-recovery arms; and
- human-visible architectural improvement in fixed-frame montages.

Kill or redirect the voxel route if:

- it improves A2 only by returning the envelope;
- it ignores authentic A2 in the source-dependence ablation;
- gains come from deterministic spill cleanup applied only to the learned arm;
- median gains hide broad paired losses or identity degradation;
- occupancy improves but terminal surfaces remain visibly faceted, ribbed, hollow, or disconnected;
- a deterministic model cannot learn useful correction and diffusion is proposed without a new
  falsifiable reason; or
- recipe compatibility cannot preserve the project's required editability.

## Conditions for future incorporation

This finding may be incorporated into `CONTEXT.md`, an ADR, or production plans only after review of:

- the completed aligned Arm B result;
- the authentic 384/96 deterministic correction artifact;
- the frozen 714-building confirmation, if the small gate passes;
- source-dependence and envelope-only ablations;
- occupancy-preserving surface recovery evidence;
- fixed-frame visual review; and
- an explicit recipe-compatibility decision.

Until then, the defensible statement is:

> Dense absolute `64^3` occupancy removes the current spatial-correspondence ambiguity and makes
> whole-volume addition/removal well-posed. The repository has proven the need and the plumbing, but
> not yet that a learned authentic whole-volume correction produces better buildings.

