# The evaluation harness, and its first baseline

Ticket: [Build the one evaluation harness this map is judged on](https://github.com/danvisai/SDFusion/issues/71)
· Map: [#69](https://github.com/danvisai/SDFusion/issues/69)

| what | where |
|---|---|
| harness | `scripts/foundations/eval_massing_arms.py` |
| contract tests | `scripts/foundations/test_eval_massing_arms.py` (12, CPU-only) |
| artifact | `execution/artifacts/massing_arms_eval_baseline.json` |
| montage | `outputs/massing_arms_eval/montage_baseline.png` |

Run it, and pin the ids so a later checkpoint is comparable rather than merely similar-looking:

```
eval_massing_arms.py --n 48 --tag baseline
eval_massing_arms.py --a2 <ckpt> --strength 0.5 --tag v2 \
                     --ids_from execution/artifacts/massing_arms_eval_baseline.json
```

It replaces `eval_vecset_projection.py` (A2 only, map-#24 as a hardcoded constant from a different
sample) and `render_a2_comparison.py` (the montage, on its own ad-hoc slice). Their numbers were
routinely printed side by side and were never comparable.

## The baseline — n = 48 fixed held-out ids

| arm | fp-IoU | missing | extra | 3D IoU |
|---|---|---|---|---|
| gt | 1.000 | 0.000 | 0.000 | 1.000 |
| blockout | **1.000** | 0.000 | 0.183 | 0.845 |
| codec_ceiling | 0.997 | 0.000 | 0.001 | 0.999 |
| deployed_map24 | 0.817 | 0.041 | **0.455** | 0.635 |

`missing` and `extra` are fractions of GT volume. Ids are global rows of `real.h5`, listed in the
artifact.

### What the split shows that the aggregate hid

**The deployed model fails by over-filling, not by eroding.** It leaves only 4.1% of GT unfilled while
adding **+45.5%** — two and a half times the blockout's over-fill — and it loses footprint fidelity on
the way (1.000 → 0.817). On all three criteria it is *worse than doing nothing*. A lone 3D IoU of 0.635
vs the blockout's 0.845 could not say that, and "carved the over-fill" and "ate the building" would
have been indistinguishable.

The blockout's over-fill is now reproducible from committed code: **0% missing, +18.3% extra** (the
prose figure was +21.7% on a different sample).

The hardcoded constant it replaces was `fp-IoU 0.863 / 3D IoU 0.601`, "measured n=15". Measured
properly it is **0.817 / 0.635** — right about the aggregate, optimistic about the footprint.

## Three confounds found while building it

**1. `surface_roughness` is not comparable across arms.** It is a raw |Laplacian|, so it scales with
the field's own slope, and the arms do not share one: a metric SDF on this grid has near-surface
|∇| = 0.031, and Dora's decoded TSDF measures **1.31 — 32× steeper**. Its roughness is inflated by
about that factor for reasons unrelated to crispness. The harness logs `guard_field_slope` beside
`guard_roughness` so this stays visible. Roughness remains a within-arm, across-runs regression guard
and enters no ranking.

**2. Ribbing in the montage is a meshing artifact of the field, not melt.** On row 2 the blockout's
occupancy is **byte-identical to GT** (`(bo<=0) == (g<=0)` everywhere, per-slice areas equal), yet it
renders visibly ribbed with 8.9× GT's roughness at a comparable slope (0.038 vs 0.031). The signed EDT
is faceted, so marching cubes at level 0 ribs it. `codec_ceiling` ribs for the opposite reason — its
field is too steep for the crossing to be located within a voxel. ⚠️ **This matters beyond the
picture:** `precompute_vecset_latents.py --blockout` meshes this exact field and encodes it, so the
ribs are in what A2 trains against. Anyone eyeballing a vecset decode for "melt" will see them.

**3. The sampled arm was not reproducible.** `Stage3aModel.inference` draws its DDIM start from the
global RNG, and nothing seeded it: two identical runs moved the deployed median 3D IoU by **0.027**.
Seeding **per building** (keyed on the id, so a subset reproduces the full run's rows) cuts that to
0.001. It is deliberately not bit-exact — a ~0.001 residue survives that is *not* cuDNN (same-seed
inference is already bit-identical within a process, and `benchmark=False` does not remove it), and
`deterministic=True` + TF32 off would cost 13× (2 it/s vs 27, ~40 min for this arm). The artifact
carries the measured `noise_floor` instead: fp-IoU ±0.008, missing ±0.016, **extra ±0.040**, 3D IoU
±0.001. `extra` is much the loosest — it is unbounded above and dominated by the over-fill.

## Reading the montage

Arms are columns, buildings are rows, every panel meshed at continuous SDF level 0.0 (never binary
occupancy @0.5). **One fixed camera in the shared world frame, no per-mesh normalisation** — so an arm
that lost volume renders smaller instead of being silently rescaled up to GT's apparent size, which is
what `render_mesh_png` did and what would have hidden criterion 3 entirely.
