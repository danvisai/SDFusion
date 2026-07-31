# #78 — the frame fix, the re-encode, and the guard

**Date:** 2026-07-29 · ticket
[Fix the latent-cache frame bug and re-encode the real latents](https://github.com/danvisai/SDFusion/issues/78)
· map [#69](https://github.com/danvisai/SDFusion/issues/69) · root cause in
[`latent-normalisation-result.md`](latent-normalisation-result.md) §1

---

## 1. The two frames, named once

The bug was not a typo — it was an **unnamed distinction**. Two mesh frames existed, both legitimate,
with nothing in the codebase asserting which one a given mesh was in:

| frame | what lives in it |
|---|---|
| **Frame-N** | what `ingest_surfaces.to_frame_n` writes; what `building_to_sdf` normalises to. **The recovered LoD2 surface corpus.** |
| **the array frame** | the stored 64³ SDF is indexed `[z, y, x]`, so a mesh extracted from it (marching cubes + `verts_to_world`) carries its coordinate components in that same `(z, y, x)` order. `grid_points` queries in this order, so `decode_grid` output is directly comparable to a stored SDF. **The generator, the codec and the eval harness all speak this frame.** |

They differ by an **x↔z swap**, which is also a **reflection** (determinant −1) and therefore reverses
face winding.

Evidence the distinction was already half-known but never named: `ingest_surfaces.py:212` applies
`.transpose(2, 1, 0)` to the stored SDF in its *own* verification, with a comment noting the mesh and
query were "both in the array's own index frame". The knowledge was in a comment, not in a function.

**The fix is one named conversion**, `scene.surface_sampling.to_array_frame`, placed in the module that
already documents the Frame-N reflection hazard. It reverses the coordinate columns *and* flips winding.
`precompute_vecset_latents.py` now routes the corpus through it; the blockout path already produced
array-frame meshes and is unchanged.

Two supporting corrections:

- **`Building`'s docstring said "One building, in Frame-N."** That is the contract the buggy call site was
  written against, and it was wrong — every codec in this repo consumes array-frame meshes. Corrected, with
  the hazard named.
- **`to_array_frame` is an involution** (applying it twice is the identity), so there is no "which
  direction" ambiguity to get wrong at a call site.

## 2. The guard

`verify_frame` in `precompute_vecset_latents.py`, run **before** the cache is written: decode a few
sampled latents and assert each reproduces **its own footprint** at fp-IoU ≥ 0.85. A frame error moves the
mass off the footprint and collapses the score; a correct cache scores ~1.0 because the codec round-trips
at ~0.999. It validates against the footprint stored *beside each latent in the same file*, so it cannot
drift out of sync with what it checks.

**Demonstrated to work in both directions:**

| cache | min fp-IoU | median | verdict |
|---|---|---|---|
| old real cache (buggy) | **0.1735** | 0.4240 | **REFUSED** ✅ |
| blockout cache (always correct) | 1.0000 | 1.0000 | passed |
| re-encoded real cache (smoke, n=24) | 0.9970 | 0.9976 | passed |
| **re-encoded real cache (full 35,623, write-time guard)** | **0.9966** | 0.9974 | **passed** |

This bug class is now **two-for-two, and both instances passed the verification that existed**: #62
aligned surfaces at IoU 1.0000 while every normal was inverted (it validated *position*, not
*orientation*), and #70 found every real latent transposed with nothing checking frame at all. The guard
costs seconds against a ~2 h encode.

⚠️ One honest limit: this guard catches **frame** errors, not **orientation** errors. Inverted winding
still decodes onto the right footprint. `ensure_outward` in the samplers is what covers that, and the two
guards are complements, not substitutes.

## 3. Tests

`scene/test_surface_sampling.py` — **13 pass**, 5 of them new (`TestArrayFrameConversion`): x/z exchange,
winding flip because the swap is a reflection, volume stays positive for a solid, the conversion is an
involution, and outputs are C-contiguous so downstream encoders accept them.

## 4. What was and was not corrupted

Checked rather than assumed, since the map's own trap list warns that robustness across downstream
variations proves nothing against a common-mode fault.

**Corrupted — everything that consumed the latent cache:**

- `logs_building/vecset_v1` and `logs_building/vecset_pair_v1` — both **void, not negative**. Marked in
  place (`logs_building/VOID-vecset-v1-and-pair-v1.md`); 189 MB each, kept rather than deleted.
- The A2 eval numbers **0.347** and **0.611**, and therefore the map-level conclusion *"the representation
  was never the bottleneck"*, which rested entirely on 0.611 sitting beside map-#24's 0.601.
- The committed figures `a2-comparison.png` and `a2-pair-comparison.png` depict void models.

**Not corrupted:**

- **The blockout baseline, fp-IoU 1.000 / 3D IoU 0.840.** Confirmed by inspection:
  `eval_vecset_projection.py:104-106` builds that arm from `blockout_sdf(fp, …)` and the stored SDF, with
  no latent involved. The cache supplied only `row`, `footprint`, `region`, `height_m` — and the footprints
  were independently verified against GT at IoU 1.0000.
- **The blockout latent cache itself** — always array-frame, passes the guard at 1.0000. Deliberately
  **not** regenerated: re-running it under a changed convention would only risk swapping which cache is
  wrong.
- **The frozen gate, `deployed_vs_dora`, and `dora_surface_metrics`.** These score `roughness` and
  occupancy *fraction*, both invariant under a transpose. Dora 0.00796 / TripoSG 0.00847 / deployed
  0.00571 all stand, as does the roughness-is-anti-correlated finding.
- Everything in [#72](https://github.com/danvisai/SDFusion/issues/72).

## 5. The re-encode — done

**35,623 latents, 9,338 MB fp16, 714 held out, 7,019 s (~1.95 h) on one A100.** Written first to a distinct
path so a crash could not leave a half-written cache that looks legitimate.

### Independent verification, n=12 held-out, paired, medians

| | identity | transposed |
|---|---|---|
| **fixed** cache decode vs GT | **0.9985** | 0.4330 |
| old cache decode vs GT | 0.4332 | *(0.998 — the original finding)* |

A clean inversion of the original diagnostic. Footprint scores: **fixed 0.9978** (worst of 12: **0.9920**)
against **old 0.4364**.

**And the pair is now genuinely aligned:** decoding the cached blockout and the cached real latent for the
same building gives **IoU 0.8623** between them — essentially the blockout's own 0.840 against GT, which is
what an aligned pair should look like. Before the fix these two were in different frames.

### Promotion

Renamed in place rather than repointing consumers, since one definition of the path is the same principle
that fixed the bug:

| path | contents |
|---|---|
| `vecset_latents.h5` | **the fixed cache** — verified across its full range at min fp-IoU **0.9955** / median 0.9977 (n=12 spanning all 35,623 rows), tail block finite |
| `vecset_blockout_latents.h5` | unchanged, always correct |
| `vecset_blockout_smoke.h5` | checked — **clean** (1.0000), kept |

So `train_vecset.py`, `eval_vecset_projection.py` and `render_a2_comparison.py` need no edits — their
defaults now resolve to the fixed cache.

**Both void caches deleted (2026-07-30), 8,978 MB freed**, after the canonical cache was verified end to
end:

- `vecset_latents_TRANSPOSED_void.h5` (9.3 GB) — the old buggy cache.
- `vecset_latents_smoke.h5` (52 MB) — **also transposed** (min fp-IoU 0.1728). Worth naming separately: it
  was the live hazard, because it is a plausible `--latents` target for a quick smoke test and would have
  handed back garbage silently. The bug's blast radius included a file nobody had thought to check.

The guard's negative-control demonstration survives as the numbers recorded in §2, which is what it was
for.

## 6. What this does not do

It does **not** re-open the question of whether A2 works. It removes a reason the answer was
uninterpretable. The convergence run
([Train the aligned-pair generator to convergence](https://github.com/danvisai/SDFusion/issues/75)) is
what measures that, and its "extend the aligned-pair arm" premise now needs re-deciding, because the reason
pairs looked better than plain was that pairs learned the transpose *explicitly*.
