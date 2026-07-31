# What our surface corpus can actually produce

Ticket: [Audit what geometric signals our surface corpus can produce](https://github.com/danvisai/SDFusion/issues/74)
· Map: [#69](https://github.com/danvisai/SDFusion/issues/69)

| what | where |
|---|---|
| audit script | `scripts/foundations/audit_surface_corpus.py` |
| artifact | `execution/artifacts/surface_corpus_audit.json` |

Every one of the **35,623** meshes was checked, not a sample — the corpus is 18 MB, so there was no
reason to extrapolate.

## Verdict: proceed. The next training run should not wait for a data change.

Three independent reasons, and the audit only supplies the third:

1. **[#72](https://github.com/danvisai/SDFusion/issues/72)** — no new *input* signal is needed, and
   all of Dora's sharp-detail apparatus is *autoencoder* machinery, inert while we keep the codec
   frozen.
2. **[#73](https://github.com/danvisai/SDFusion/issues/73)** — the diffusion's whole supervision is
   latent-space MSE, now measured as decoupled from decode quality. **Enriching data cannot reach a
   loss that never touches the data.**
3. **This audit** — everything the encoder consumes is present, and verified correct *at the point of
   consumption*.

The one genuinely recoverable extra signal (semantic surface labels) would feed a *decoder* fine-tune
([#77](https://github.com/danvisai/SDFusion/issues/77)), not the diffusion.

## The signal table

Audited against the list [#72](https://github.com/danvisai/SDFusion/issues/72) ruled on.

| signal | status | evidence / cost |
|---|---|---|
| **surface points + normals** (coarse stream) | ✅ **have now** | meshes on disk; normals derived from faces. Orientation verified along the whole consumption path — see below |
| **sharp-edge points + bisector normals** | ✅ **have now** | **100% of meshes** have at least one edge above the 25° dihedral threshold; median **21** sharp edges. The sampler's uniform fallback never fires |
| **inside/outside (occupancy, signed field)** | ✅ **have now** | `real.h5` 64³ SDFs, independent of the mesh |
| **normal maps — as the SNE *metric*** | 🟡 **cheap, no re-ingest** | rendering 22 views from meshes we already hold. ⚠️ **still not built** — see the gap below |
| **semantic surface labels** (roof / wall / ground) | 🟠 **needs a re-ingest** | CityGML LoD2 *has* them and our parser already walks `boundedBy` (`ingest_citygml_lod2.py:67`) — but only to *filter* for LoD2, keeping `posList` geometry and dropping the type. Recovering them costs a re-ingest, ~80 min against #62's parallelised fetch |
| **normal maps — as an input** | ❌ ruled out (#72) | a lossy re-encoding of 3D GT we already hold |
| **height maps** | ❌ ruled out (#52, #72) | breaks editable-SDF carving downstream |
| **UVs** | ❌ not available, and carries no geometry | LoD2 sources ship no parametrisation |
| **ambient occlusion** | ❌ pointless | computed *from* geometry; adds nothing |

## Four things the audit turned up

### 1. ⚠️ The corpus on disk is inward-wound — 35,602 of 35,623

`ingest_surfaces.to_frame_n`'s docstring says winding is repaired, and `fix_winding` exists to do it.
**The stored data says otherwise:** the enclosed volume is negative for 99.94% of meshes, while winding
is *internally consistent* for 100% of them — so they are cleanly wound and globally inverted, which is
the signature of the Frame-N y/z swap (a reflection) being applied after the repair.

It is **not a live bug**, because every consumer repairs it. Traced end to end:

| stage | orientation |
|---|---|
| raw in `surfaces_*.h5` | **inward** |
| after `dora_frozen_gate.load_surfaces` | outward |
| after `scene.surface_sampling.to_array_frame` | outward |
| **what the encoder receives** (`Building.require_mesh` → `ensure_outward`) | **outward ✅** |

It *is* a latent hazard: any new reader that opens the h5 directly gets inside-out surfaces, and the
signed-distance path will not notice — which is exactly how #62 shipped 400/400 inverted meshes past a
check that passed at IoU 1.0000. Cheapest durable fix is to rewrite the corpus outward-wound and keep
`ensure_outward` as the backstop; until then, **go through `load_surfaces`, never through h5py**.

(The 21 meshes measuring outward are all NRW, and all in its non-watertight set, where the volume sign
is not meaningful.)

### 2. 🔑 The meshes are extremely coarse — median 20 faces

| source | n | watertight | has sharp edges | median faces |
|---|---|---|---|---|
| bag3d (NL) | 11,773 | 1.0000 | 1.0000 | 66 |
| nrw (DE) | 11,850 | **0.8927** | 1.0000 | 20 |
| plateau (JP) | 12,000 | 1.0000 | 1.0000 | **12** |
| **all** | **35,623** | 0.9643 | 1.0000 | 20 |

A cube is 12 triangles. **Half the Japanese corpus is at or below box complexity**, and the median
building across all three sources is 20 faces. This is worth stating plainly because it bounds what any
sharpness supervision can ever teach: there is very little geometric detail in the data to learn from.
It bears directly on sizing [#77](https://github.com/danvisai/SDFusion/issues/77) and on how much to
expect from a longer run.

### 3. NRW is 10.7% non-watertight; the other two are clean

1,272 German meshes are not watertight. This does **not** block the frozen-codec path — the encoder
point-samples the surface, and occupancy comes from `real.h5` independently. It *would* matter for
[#77](https://github.com/danvisai/SDFusion/issues/77), where a decoder fine-tune needs `pysdf`-style
watertight targets; budget a repair pass or an NRW exclusion there.

10,003 meshes carry at least one degenerate (zero-area) face, 31,900 faces in total out of ~1.6M — a
rounding error, and surface sampling weights by area, so they contribute nothing rather than corrupting
anything.

### 4. The 153 unrecovered are ordinary buildings, and 150 of them are German

`{bag3d: 3, nrw: 150, plateau: 0}` — the failure is concentrated in one source's parser, not spread.
They are not degenerate: median SDF occupancy 0.18 (against a corpus-typical 0.20), max 0.50, **none
empty**, and every footprint is non-empty. At 0.43% of the corpus they do not matter statistically, and
the held-out split already ignores them.

## ⚠️ Gap: SNE was supposed to land in the harness and did not

[#72](https://github.com/danvisai/SDFusion/issues/72) closed with *"SNE belongs in that harness"*,
pointing at [#71](https://github.com/danvisai/SDFusion/issues/71). The harness shipped **without it** —
the #71 ticket body listed montage, fp-IoU and the missing/extra split, and did not carry the SNE
instruction across. Nothing in this audit blocks it: rendering normal maps from the meshes we hold is
cheap and needs no re-ingest. It remains the only proposed instrument that might separate crisp from
melted, after three scalars failed to.
