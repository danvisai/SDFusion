# Transform + Composition Research Proof

Status: ready-for-agent

## Problem Statement

GenerativeTowns has an extensive editable-building system and several promising experimental
observations, but it does not yet have one leakage-safe, reproducible body of evidence for its central
research thesis: generality comes from transforming rough inputs and composing real architectural
elements, rather than sampling complete buildings from noise or scaling a monolithic detail generator.

The present evidence is fragmented across live demo behavior, prior negative experiments, mutable data
libraries, hand-inspected outputs, and plans. The existing element library includes future held-out
buildings, the old detailizer learned synthetic composer outputs rather than real building detail, and
the neutral facade/FID evaluation surface does not yet exist. Without correcting those issues, a paper
reviewer cannot distinguish a genuine factorization result from leakage, an unfair baseline, renderer
bias, or post-hoc metric selection.

The project owner needs an execution path that preserves GenerativeTowns' domain model, tests both
claims honestly, fails early when a claim is unsupported, and produces paper-ready artifacts without
turning peripheral demo features into research claims.

## Solution

Build a reproducible experiment pipeline around two falsifiable claims:

- **C1, transform rather than sample:** compare the live Stage 3a prior sampled honestly from noise
  against the same prior used as an SDEdit transform from a footprint-extrude blockout. Demonstrate
  that the transform improves real-building massing fidelity, and measure how the same operator trades
  edit faithfulness against realism across a sculpt-strength sweep.
- **C2, compose rather than synthesize:** compare Stage 3a massing plus retrieved real architectural
  elements against a strong monolithic SDF generator trained on real coarse-to-detailed BuildingNet
  pairs. Give both arms exactly the same BuildingNet detail-data fractions and evaluate them at
  25%, 50%, and 100% data.

Freeze a held-out BuildingNet test split before training or library construction. Build nested training
fractions, leakage-safe retrieval libraries, and real monolith pairs from those same ids. Evaluate every
arm through one deterministic neutral facade-rendering surface. Use paired Chamfer/IoU for massing,
because the footprint and height determine the target, and distributional facade FID plus a blinded
two-AFC study for detail, because detail is underdetermined.

Use 96^3 as the initial shared working resolution, matching the existing detail realization and
monolith architecture. The real low-pass SDF is the primary coarse input for the monolith because it
keeps source and target aligned; footprint extrusion is an explicitly labeled robustness variant, not
a replacement primary input. Fix the integer voxel definition of `s*` from the metric normalization
needed to approximate 0.5 m before inspecting semantic-category outcomes. If a preflight montage shows
that facade elements are not represented at 96^3, move all arms together to 128^3 and record that
preflight decision before computing headline results.

Run the 100% C2 comparison as a kill-gate before spending compute on the lower data fractions. If the
decomposition does not win on detail fidelity while retaining comparable massing fidelity, stop the
scaling curve and diagnose the failure rather than retrofitting the hypothesis. Preserve negative and
partial results in the final evidence package.

## User Stories

1. As the project owner, I want one canonical experiment specification, so that agents do not execute conflicting versions of the research plan.
2. As the project owner, I want the research claims separated from demo-wrapper features, so that the paper argues only what the experiments establish.
3. As a researcher, I want a frozen held-out BuildingNet test set, so that evaluation buildings never influence training or retrieval.
4. As a researcher, I want class-stratified splits, so that architectural-class imbalance does not dominate comparisons.
5. As a researcher, I want nested 25%, 50%, and 100% training sets, so that the data-scaling curve changes only the amount of detail data.
6. As a researcher, I want deterministic split generation, so that every experiment can reconstruct the exact data assignment.
7. As a researcher, I want retrieval libraries built only from the matching training fraction, so that equal-data means the same thing for composition and synthesis.
8. As a researcher, I want element provenance recorded in every library, so that leakage can be audited after generation.
9. As a researcher, I want real coarse-to-detailed monolith pairs, so that the baseline learns real BuildingNet detail rather than imitating the procedural composer.
10. As a researcher, I want pair alignment and SDF conventions verified visually and numerically, so that training failures are not caused by corrupted supervision.
11. As a researcher, I want one shared working resolution for all arms, so that apparent detail differences do not come from unequal sampling density.
12. As a researcher, I want `s*` fixed before examining semantic-detail results, so that the massing/detail coincidence remains falsifiable.
13. As a researcher, I want a fixed semantic detail vocabulary, so that categories are not added or removed to improve the result.
14. As a researcher, I want neutral-shader facade renders, so that appearance generation does not confound geometry fidelity.
15. As a researcher, I want deterministic cameras and lighting, so that repeated runs and different arms are directly comparable.
16. As a researcher, I want a real-versus-real FID sanity baseline, so that I understand the metric floor and finite-sample variation.
17. As a researcher, I want massing evaluated with paired Chamfer and IoU, so that footprint-conditioned geometry is judged against its determined target.
18. As a researcher, I want detail evaluated distributionally, so that plausible alternative windows and facade elements are not penalized for differing from one arbitrary target.
19. As a researcher, I want uncertainty reported for FID and paired metrics, so that small differences are not presented as definitive wins.
20. As a researcher, I want the best honest Stage 3a from-noise sample as the C1 baseline, so that the transform comparison is not a strawman.
21. As a researcher, I want the SDEdit blockout contract recorded, so that C1 can be reproduced without knowledge of the interactive demo.
22. As a researcher, I want a sculpt-strength sweep, so that generation and editing are shown as operating points of the same transform.
23. As a researcher, I want edit faithfulness and realism plotted together, so that the transform tradeoff is visible rather than selected qualitatively.
24. As a researcher, I want the prior residual-correction result incorporated with provenance, so that the claim that transforms recover massing rather than detail has independent support.
25. As a researcher, I want the active element-retrieval quality fixes validated before C2, so that skeletal or voxel-crushed elements do not invalidate the decomposition arm.
26. As a researcher, I want a strong full-data monolith checkpoint, so that the C2 comparison survives a weak-baseline critique.
27. As a researcher, I want the 100% comparison run first, so that a failed core claim stops expensive lower-fraction experiments.
28. As a researcher, I want failure causes separated into massing, retrieval, renderer, and monolith effects, so that the kill-gate produces a useful decision.
29. As a researcher, I want the 25% and 50% experiments to reuse the proven full-data contracts, so that the scaling curve introduces no new confounders.
30. As a researcher, I want the monolith scaling slope reported without claiming web-scale certainty, so that the data-scale rebuttal stays within measured evidence.
31. As a researcher, I want a recipe-massing robustness arm, so that any composition result is not tied exclusively to Stage 3a massing.
32. As a researcher, I want LoD3 element enrichment isolated as an ablation, so that extra real element data does not contaminate the equal-BuildingNet-data headline.
33. As a study participant, I want blinded and randomized facade comparisons, so that model identity and presentation order do not bias my choices.
34. As a researcher, I want the two-AFC protocol and analysis fixed before collecting responses, so that human preference is not analyzed post hoc.
35. As a reviewer, I want every reported result linked to its configuration, data provenance, checkpoint, renders, and metric outputs, so that I can audit the evidence chain.
36. As a reviewer, I want representative failures shown alongside successes, so that the claimed operating range is clear.
37. As a reviewer, I want negative `s*`, C1, or C2 outcomes retained, so that falsifiable claims are treated honestly.
38. As a maintainer, I want smoke runs on a small split, so that pipeline regressions can be caught without launching full GPU experiments.
39. As a maintainer, I want resumable and deterministic long-running jobs, so that cluster interruption does not corrupt or silently change results.
40. As a maintainer, I want generated datasets and experiment outputs outside version control with tracked manifests and summaries, so that the repository remains usable while experiments remain reproducible.
41. As the paper author, I want figures and tables generated from recorded results, so that manuscript numbers cannot drift from experiment artifacts.
42. As the paper author, I want related work and limitations written against the measured scope, so that the paper does not claim universal impossibility of generative detail.
43. As the project owner, I want a final paper-ready evidence package, so that the work can move from an impressive demo to a defensible research contribution.

## Implementation Decisions

- The accepted domain model is binding: learned models make decisions, deterministic procedures realize
  symbolic recipes, SDEdit is the core transform, massing is above `s*`, and detail is below `s*` or in
  the fixed semantic detail set.
- The implementation is organized around one versioned experiment-run artifact as the primary seam.
  Each run records configuration, git revision and dirty-state digest, split ids, dataset/library
  provenance, checkpoint identity, random seeds, renderer settings, per-sample status, metrics, and
  links to qualitative outputs.
- Data preparation produces a sealed class-stratified test split and nested training fractions from
  the remaining BuildingNet ids. Split generation is deterministic and validates uniqueness,
  coverage, nesting, class balance, and test disjointness.
- The element-library builder accepts explicit include and exclude id sets plus an output destination.
  Every emitted element retains source-building provenance. Experiment libraries are immutable inputs
  once a run begins.
- The current retrieval representation is retained, including element-relative scale and solidity
  filtering, but it must pass focused retrieval QA and existing service gates before becoming the C2
  baseline. Element ops remain first-class reversible recipe operations and are sampled analytically at
  the output resolution.
- The monolith trains on real BuildingNet targets. Its primary coarse input is a low-pass transform of
  the same building's real SDF; synthetic composer pairs are retained only as a documented negative.
- The shared initial working resolution is 96^3. A single pre-result representation gate may promote
  every arm to 128^3 if 96^3 cannot express the fixed facade-detail categories. Mixed-resolution
  headline comparisons are prohibited.
- `s*` is expressed as an integer number of working-resolution voxels derived from approximately 0.5 m
  under the dataset's metric normalization. It is recorded before semantic spectrum measurements and
  is never tuned to their result.
- The neutral facade renderer uses identical geometry conversion, cameras, projection, framing,
  lighting, background, and normal/material treatment for real and generated samples. SDXL texture
  bake and photoreal rendering are excluded from research metrics.
- Detail FID uses one pinned feature extractor and preprocessing contract. Real-versus-real split FID,
  bootstrap intervals, sample counts, and generation failures accompany every headline value.
- C1 compares honest unconditional or lowest-information Stage 3a sampling with SDEdit initialized from
  the held-out building's footprint-extrude blockout. It does not provide the from-noise arm with
  information unavailable under that arm's declared contract.
- The sculpt sweep uses the live transform implementation and fixed crude-edit cases. Strength is the
  controlled variable; faithfulness and realism are both reported at every operating point.
- C2 uses Stage 3a massing plus retrieval as the headline decomposition. Recipe-parameter diffusion plus
  procedural massing is a robustness ablation, not the headline.
- At fraction X, the monolith's real pairs and the decomposition's element library both derive only
  from train_X. Stage 3a massing remains full-data because detail data efficiency is the contested
  variable.
- The 100% C2 cell is a preregistered kill-gate. Lower-fraction training, the scaling claim, and optional
  enrichments do not proceed until the full-data result and failure audit justify them.
- LoD3 enrichment remains a separately labeled real-element-data ablation. Its data never enters the
  BuildingNet-only headline libraries or the monolith.
- Human preference uses blinded, randomized two-AFC neutral-render pairs with a declared sampling and
  analysis protocol. It supports detail FID and does not replace the automated metric.
- Long-running data builds, training, generation, and rendering are resumable and idempotent. Partial
  results are distinguished from complete runs in the manifest and are never silently included in
  aggregate metrics.
- Large generated datasets, checkpoints, and renders remain outside version control. Small manifests,
  summaries, protocol documents, tables, and figure-generation metadata are tracked.
- The paper artifact includes C1, C2, the fixed-scale test, negative prior work, limitations, exact
  experiment contracts, and explicit separation of core claims from the editable demo wrapper.

## Testing Decisions

- Tests assert externally observable contracts: provenance, determinism, disjointness, artifact
  completeness, metric behavior, and service output. They do not lock internal helper structure or
  exact floating-point tensors unless those values are part of a published artifact contract.
- The highest and primary seam is the experiment-run artifact. A small smoke run must be loadable by
  the same evaluator used for full experiments and must prove that inputs, outputs, failures, metrics,
  and provenance form one complete auditable chain.
- Split tests verify deterministic reproduction, no duplicate ids, nested training fractions, sealed
  test disjointness, expected coverage, and acceptable class-stratification deviation.
- Library-builder tests construct a tiny known library and verify include/exclude behavior, source
  provenance, deterministic metadata, configurable output isolation, and zero held-out contributors.
- Real-pair tests verify that targets come from original real SDFs, coarse and detailed volumes are
  aligned, axes/signs are correct, resolutions match the locked contract, and no synthetic composer
  target enters the dataset.
- Renderer tests use simple known geometry to verify deterministic camera framing, surface visibility,
  normal orientation, identical treatment across source representations, and stable output dimensions.
- Metric tests verify identical-set behavior, deterministic preprocessing, finite outputs, bootstrap
  reproducibility, correct failure accounting, and the real-versus-real sanity baseline.
- Retrieval QA verifies scale and solidity filtering, deterministic seeded selection, provenance
  propagation, and visible non-empty analytic composition at the target resolution.
- The existing branch/API and sculpt-flow suites are prior art and remain regression gates for the
  active Phase R changes. Their purpose is service and reversible-edit behavior, not research-metric
  validation.
- Training smoke tests run only enough data and iterations to validate loading, checkpointing, resume,
  validation output, and manifest registration. They do not assert that a model converges in CI.
- Full experiment acceptance is based on protocol completeness and correctly computed results, not on
  forcing the predicted hypothesis to pass. A scientifically valid negative result satisfies the
  pipeline contract.
- Every headline table and figure is regenerated from tracked summaries or run manifests, with a check
  that manuscript values match source metrics.

## Out of Scope

- General town-page or sculptor polish unrelated to measuring C1 or C2.
- Rebuilding the stale demo bundle, weathering live preview, ornament-library expansion, relief-stack
  persistence, and deferred disk cleanup.
- Treating recipe closure, weathering, ornaments, sketch relief, texture bake, or photoreal rendering as
  evidence for the two research claims.
- Training Phase G crop-inpainting as a product feature. A narrow experiment may be separately specified
  only if the C2 failure audit identifies missing element coverage as the decisive confounder.
- Comparing against Hunyuan3D-2 or another image-conditioned system that cannot consume the task's
  footprint input.
- Claiming that generative detail is universally impossible or extrapolating measured data trends to
  unobserved web-scale regimes as fact.
- Changing `s*`, semantic categories, evaluation cameras, or primary metrics after viewing headline
  results.
- Deleting or rewriting Claude's pre-existing uncommitted Phase R changes as part of specification work.

## Further Notes

- This specification publishes the execution-carrying wayfinder effort titled **Transform +
  Composition Research Proof**. Its child tickets define the current dependency graph and enforce the
  100% C2 kill-gate before the scaling curve.
- Accepted architectural decisions take precedence over older handoffs when they conflict. The July 10
  research, experiment-design, and two-claim ADRs are the current decision record.
- The repository currently contains uncommitted Phase R quality changes in four server modules. They
  are prior work, must be reviewed in place, and should not be conflated with changes made while
  implementing this specification.
- Existing large data and checkpoint locations are operational dependencies but are not portable source
  artifacts. Reproducibility therefore depends on checksums, provenance manifests, and documented build
  commands rather than committing those binaries.
- A failed hypothesis is an acceptable destination when the protocol is sound. The final evidence
  package must say what failed, where, and why the measured evidence does not support the planned claim.
