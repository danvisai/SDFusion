# Session handover — 2026-07-27 → 2026-07-29

Written so a cold session can pick this up. 26 commits (`cfc4f6d..HEAD`). Living per-effort status stays
in the `docs/wayfinding/` maps; this is the cross-cutting summary and the trap list.

---

## 1. Status right now

**Nothing is running.** One decision is open and is the user's to make.

> **OPEN DECISION: stop the vecset thread, or continue it?**
> Recommended (by the assistant, **not yet accepted**): stop. See §4 for the evidence and §8 for the
> case against stopping.

| effort | state |
|---|---|
| **Map #61** — crisp massing via a query-based decoder | destination reached; all 4 children closed |
| #62 recover surfaces · #63 ceiling probe · #64 data scale · #65 posture | **closed** |
| **#68** spec — seams, corpus, frozen gate | **closed**, delivered |
| **#66 / #67** specs | **open** — #67 holds the A2 plan; its later stages are what the open decision is about |
| **#50** clean structural generation | open; now carries the blockout finding (§4) |
| **#47 / #49** supervisor demo | **parked**, deliberately, pending a better model |
| **#51** prototype | closed — *do not integrate map-#24* |

---

## 2. The headline finding

**The representation was never the bottleneck.**

| stack | representation | params | 3D IoU |
|---|---|---|---|
| map-#24 (deployed) | 64³ dense grid | 947M | **0.601** |
| A2 (this session) | 2048-token vecset | 49.4M | **0.611** |
| **blockout** | signed-EDT footprint extrusion | **none** | **0.840** |

Two maximally different architectures converge on ~0.60, and a **no-model baseline beats both**. The
whole #52 → #58 → #61 chain was premised on the dense grid being the limiting factor; the vecset
rebuild tested that premise directly and it did not hold.

**Corollary already actionable:** an extruded footprint is a better massing than the model the demo
ships (`fp-IoU 1.000 vs 0.863`). Posted to [#50](https://github.com/danvisai/SDFusion/issues/50).

The blockout's only error is **over-fill**: 0.00% of GT missing, **+21.7% extra** (n=60), because it
extrudes a single height across a plan that has several. That is a much narrower problem than
"generate a building".

---

## 3. Timeline

| date | what | commits |
|---|---|---|
| 07-27 | Rejected arXiv 2301.11656 (eikonal FVM) as inapplicable | `f139b79` |
| 07-27 | Chartered **map #61** + specs #66/#67; three adoption options costed | `730b378` |
| 07-27 | **#63 ceiling probe POSITIVE** — query decoder 0.00328 vs deployed 0.00552 | `57a05a9` |
| 07-27 | **#64** — from-scratch AE evidenced-risky; option A split A1/A2; ~1849→35,776 corrected | `7bccf92` |
| 07-27 | **#62 GO** — all 35,776 recoverable, ~280 MB | `eca8a54` |
| 07-28 | #51 closed (*do not integrate map-#24*); #49 cull committed; `_shrink_polygon` fixed | `f3e9a41` `fe9cf7b` `c8e4494` |
| 07-28 | Dora-VAE loads + round-trips one building | `3cecdfc` |
| 07-28 | **#62 executed** — 35,623/35,776 (99.6%) surfaces recovered, verified | `74e9c5f` `dce8274` `104fca0` |
| 07-28 | **Frozen gate NEGATIVE** → then **CORRECTED** (inside-out normals) | `4bca17b` `0f401e3` `2122517` `9b0a7f1` |
| 07-28 | **deployed-vs-Dora** — roughness shown anti-correlated with the goal | `3b95229` |
| 07-28 | **Seams**: codec contract + surface sampler, 23 tests | `2f19e60` |
| 07-28 | Denoiser, then **ADR-0003 alignment** → set-SDEdit projection | `2293418` `6c08d58` |
| 07-28 | Latent precompute + training loop + retention eval | `c4501f5` `9a0fa0f` |
| 07-29 | **A2 run 1 NEGATIVE**; diagnosed as distribution shift, not undertraining | `566f61a` `5a69274` |
| 07-29 | **Aligned-pair training** → **run 2, sixth negative** | `cced7c1` `312064e` |

---

## 4. Measured numbers

### Surface quality (roughness — ⚠️ see trap 1, this metric misleads)

| arm | roughness |
|---|---|
| GT floor | 0.0041 |
| VQVAE round-trip | 0.0044 |
| refiner/corrector wall (#54, #59) | 0.0047 |
| **map-#24 deployed** | **0.00552** |
| Dora frozen (corrected) | 0.00796 |
| TripoSG frozen (corrected) | 0.00847 |
| vecset *generation* ceiling (#63) | 0.00328 |

### Shape quality — **use these instead**

| arm | fp-IoU | 3D IoU |
|---|---|---|
| **blockout (extrusion, no model)** | **1.000** | **0.840** |
| VQVAE round-trip | 0.991 | 0.995 |
| Dora round-trip | 0.996 | 0.999 |
| map-#24 deployed | 0.863 | 0.601 |
| A2 run 1 (best, s=0.1) | 1.000 | 0.840 *(no-op — blockout passing through)* |
| **A2 run 2, aligned pairs (s=0.5)** | 0.854 | **0.611** |

### Figures (committed)

| shows | path |
|---|---|
| A2 run 1 vs blockout vs deployed | `docs/wayfinding/crisp-massing-vecset/a2-comparison.png` |
| A2 run 2 (aligned pairs) | `docs/wayfinding/crisp-massing-vecset/a2-pair-comparison.png` |
| what the blockout is missing | `docs/wayfinding/crisp-massing-vecset/blockout-gap.png` |
| roughness misleads (melted vs ribbed) | `docs/wayfinding/crisp-massing-vecset/deployed-vs-dora.png` |
| frozen-gate corrugation | `docs/wayfinding/crisp-massing-vecset/frozen-gate-montage.png` |
| #63 crisp-ceiling probe | `docs/wayfinding/crisp-massing-vecset/ceiling-montage.png` |
| spike-forest root cause | `docs/wayfinding/clean-structural-generation/shrink-polygon-fix.png` |
| #51 procedural vs map-#24 | `docs/wayfinding/clean-structural-generation/compare_closeups.png` |

---

## 5. ⚠️ Traps — do not re-hit these

1. **`surface_roughness` is anti-correlated with the goal.** It ranks a *melted blob* (0.00571) above a
   *crisp ribbed box* (0.00818). It punishes high-frequency ripple hard and low-frequency melting
   barely — and melting is what destroys architecture. **Judge on fp-IoU, 3D IoU and montages.**
   Evidence: `deployed-vs-dora.md`.
2. **Winding is silently inverted by the frame transform.** The Frame-N y/z swap is a *reflection*, so
   repairing winding before it is undone by it. 400/400 surfaces had inward normals and a whole round of
   measurements was wrong. Signed-distance paths never notice (fast-winding-number signing is
   orientation-agnostic) — a vecset encoder does, because it eats face normals. Guarded in
   `scene/surface_sampling.ensure_outward`.
3. **Array axis order.** The stored SDF is indexed **[z, y, x]** (the voxeliser grids `meshgrid(ZZ,YY,XX)`)
   while natural query order is **[x, y, z]**. Mixing a world-frame mesh with a stored field needs a
   transpose.
4. **Signed distance must use fast-winding-number.** CityGML-derived meshes are watertight but
   *negative-volume*; the default sign test reports **zero occupancy**.
5. **Hunyuan3D-2 outputs must never enter training data.** Its licence §5.b forbids using Output to
   improve any other model. Its artifacts in `outputs/` are evidence only. Distillation is closed.
6. **A passing verification can still be wrong.** #62's alignment check passed at **IoU 1.0000** while
   normals were inverted — it validates *position*, not *orientation*.
7. **Robustness across downstream variations proves nothing against a common-mode fault.** Two codecs ×
   five sampler configs all agreed — and all shared one bad input.

---

## 6. What survives and is reusable

- **Surface corpus** — 35,623/35,776 LoD2 surfaces in `data/real_massing_v1/surfaces_{bag3d,nrw,plateau}.h5`
  (18 MB), joined to `real.h5` **by id**, alignment verified (IoU 1.0000 / 0.9904 / 1.0000).
  Rebuild: `scripts/foundations/ingest_surfaces.py`.
- **Latent caches** — `vecset_latents.h5` and `vecset_blockout_latents.h5`, 9.3 GB each, 35,623 × 2048 × 64
  fp16, held-out split carried inside.
- **Seams, 23 tests** — `models/shape_codec.py` (encode / query-at-points / decode_grid; VQVAE *and* Dora
  both satisfy one shared `ContractSuite`) and `scene/surface_sampling.py`.
- **A2 model code, 27 tests** — `models/networks/vecset_denoiser.py`, `vecset_projection.py` (set-SDEdit),
  `scripts/train_vecset.py`, `scripts/foundations/eval_vecset_projection.py`.
- **Checkpoints** — `logs_building/vecset_v1` (plain), `logs_building/vecset_pair_v1` (aligned pairs).
- **Measurement discipline** — every new measurement path carries a **control arm**; the blockout is
  always scored as an arm so "did this beat doing nothing?" stays visible.

Full test run: `scene/test_surface_sampling.py` (8) · `models/test_shape_codec.py` (15) ·
`models/networks/test_vecset_denoiser.py` (9) · `models/networks/test_vecset_projection.py` (9) — all green.

---

## 7. Decisions recorded on the tracker

- **#65** — posture **A2**; the no-frontier-model constraint reinterpreted as protecting the *research
  contribution*, so inheriting a **component** is allowed. Base = **Dora-VAE** (Apache-2.0);
  **Hunyuan3D-2 excluded on licence**; TripoSG (MIT) the fallback.
- **#67** — **ADR-0003 conflict surfaced and resolved**: the generator is a **projection**
  (blockout → partial noise → denoise), never a from-noise sampler. `from_noise()` exists as a
  diagnostic only. No rebuild was needed — from-noise vs projection is an *inference* choice.
- **#68** — closed with an explicit **go** decision on the frozen gate.

---

## 8. The honest case against stopping

Stated so the open decision is fair:

- Both runs were **~1.4 epochs** (6,000 steps, batch 8) and the loss was **still descending**.
- Aligned-pair training **did** fix what it targeted — run 1 shredded blockouts, run 2 produces coherent
  masses. The distribution-shift diagnosis was correct.
- A2 has **19× fewer parameters** than map-#24 and matches it.
- The convergence argument (§2) rests on two short runs, not on converged models.

Against that: six negatives, each fix revealing the next problem rather than closing the gap — the shape
of a wrong premise rather than of nearly-there. And the demo has a viable path (§2 corollary) that needs
none of this.

---

## 9. If continuing, the next steps in order

1. **Per-channel latent normalisation** (currently one global mean/std) and confirm the codec round-trip
   survives normalise→denormalise. If this is wrong, no length of run fixes it.
2. **Longer run** — only after (1).
3. **Attack the over-fill instead** — the blockout's sole error is single-height extrusion across a
   multi-height plan. Predicting height *structure* over the footprint is a far narrower problem, and
   probably does not need a 3D generative model. ⚠️ Check against map #52's ruling that a 2.5D
   height-field "breaks the editable-SDF carving downstream" before going this way.
