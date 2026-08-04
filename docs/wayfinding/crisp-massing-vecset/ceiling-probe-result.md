# #63 — the crisp ceiling, measured

**Date:** 2026-07-27 · **Verdict: POSITIVE — large headroom confirmed.** First positive result in this
line after five straight negatives (#54, #56, #59, #60, and the #34 refiners).

A query-based (vecset) decoder produces building surfaces **at the GT crispness floor** — clearing by a
wide margin the 0.0047 wall that every dense-grid lever hit.

## Ladder (n=6 held-out, same buildings, our metric, our grid)

| arm | roughness | note |
|---|---|---|
| **GT (this sample)** | **0.00344** | the floor for *these* buildings |
| **Hunyuan3D-2 teacher** | **0.00328** | **at/below the GT floor** |
| mesh→SDF control | 0.00332 | vs GT 0.00344 — path adds nothing (see below) |
| *codec round-trip* | *0.0044* | reference, n=24, #56 |
| *refiner / corrector wall* | *0.0047* | reference — **#54 and #59 both stop here** |
| *map-#24 deployed* | *0.00552* | reference, what we ship today |

Per-building: teacher **0.00386 / 0.00218 / 0.00354 / 0.00295 / 0.00290 / 0.00425** against GT
**0.00389 / 0.00290 / 0.00358 / 0.00278 / 0.00300 / 0.00446**.

**The gap that matters:** our deployed model sits at **0.00552** and nothing we tried moved it below
**0.0047**. The teacher sits at **0.00328**. The thing we could not reach by correcting a dense grid is
reached *natively* by a query decoder.

## Why the number is trustworthy

`surface_roughness` is a band-weighted mean |Laplacian| on a **volumetric** field, so the teacher's mesh
has to come back to a 64³ grid to be comparable. That is not stacking the deck: GT is itself a 64³ field,
and #56 showed the codec round-trips at 0.0044 — **the grid can represent crisp geometry**. A genuinely
crisp surface should therefore score near the floor, exactly as GT does.

The **control arm** is what licenses the comparison. Re-voxelizing GT's *own* marching-cubes mesh through
the identical mesh→SDF path scores **0.00332 vs GT 0.00344** — delta **−0.00012**. The path adds no
roughness of its own, so the teacher's number is the teacher, not the harness.

Frame: real.h5 grids span [-1,1]³ at spacing 2/63, clamped ±0.2 at load (`datasets/bag3d_dataset.py`).
Teacher meshes are normalized onto each GT mesh's own bounding box, so both surfaces are sampled at the
same spatial density — roughness is scale-dependent and this must be done.

## What the montage shows that the scalar does not

`ceiling_montage.png`. Per #36 the visual is the primary arbiter, and here it both confirms and qualifies
the number.

**Confirms (rows 1, 2, 5, 6):** genuinely **flat faces and sharp edges** — including a crisp roof crease
in row 5 — which is precisely what the dense-grid diffusion cannot produce at any setting we tried.

**Qualifies (rows 3, 4):** a **fine-scale striation / ribbing artifact** on flat faces. Row 4 is the
important one: the teacher is **visibly worse than GT** — heavy horizontal rippling across the top and
side — yet scores **0.00295 against GT's 0.00278**, essentially identical. **Our roughness metric is
blind to this artifact class.** That is a direct corroboration of #36's finding that no scalar reliably
separates crisp from rough, and it means the headline number should be read as *necessary but not
sufficient* evidence. A vecset rebuild must be judged visually too, not signed off on roughness alone.

## Limits — state these with the result

1. **This measures achievable surface quality, not footprint-faithful generation.** Hunyuan3D-2 is
   **image-conditioned**; it was prompted with a render of the GT building itself. Row 2 reproduces a
   noticeably different proportion from its GT. Nothing here says a vecset model can be *controlled* by a
   footprint — that is exactly the open work, and CLAY is the precedent, not this probe.
2. **n=6**, and this sample's GT floor is **0.00344**, below the 0.0041 reference from the n=24 runs.
   Roughness varies per building; compare teacher to GT *within* this sample, and treat 0.0047 / 0.00552
   as reference lines rather than same-sample values.
3. **The teacher scores slightly below GT** on 4 of 6. Being *under* the floor hints at mild
   over-smoothing on some shapes as much as superior crispness — another reason the visual governs.
4. **Not an adoption.** The model was run as a measuring instrument. Whether it is ever adopted is #65.

## What this does to the options

Option A's single biggest risk was that the whole bet — *a token-set decoder produces crisper surfaces
than a dense grid* — was **unproven at our scale**. It is now proven on our data, with our metric, on our
grid. The direction is no longer speculative.

The risk that remains, and is now the pivotal one, is **#64**: whether **35,776** buildings is enough to
train such a decoder *ourselves*. This probe used a model trained on orders of magnitude more data — it
establishes that the representation has headroom, not that we can reach it from our corpus. If #64 comes
back negative, the distillation arm (generate a crisp synthetic corpus from a teacher) becomes the live
route, and it inherits the licensing question in #65.

## Reproduce

```
sdfusion/bin/python scripts/foundations/vecset_ceiling_probe.py --n 6 --model full \
    --out_dir outputs/vecset_ceiling
sdfusion/bin/python scripts/foundations/vecset_ceiling_probe.py --control_only --n 4   # path validation
```

Artifacts: `outputs/vecset_ceiling/` — `ladder.json`, `control.json`, `ceiling_montage.png`,
per-building `*_prompt.png` / `*_teacher.png` / `*_teacher.glb`. ~26–38 s per building on one A100 80GB.
