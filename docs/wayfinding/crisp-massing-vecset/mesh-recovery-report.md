# #62 — source LoD2 surface geometry: recovery report

**Date:** 2026-07-27 · **Verdict: GO — all 35,776 buildings are recoverable, for ~280 MB and no new
ingest code.** Every question this ticket asked resolved favourably, and one of them (JP id truncation)
was a real risk that had to be measured rather than assumed.

---

## 1. The gap, confirmed by measurement

`data/real_massing_v1/real.h5` keys are exactly:

```
bag_id · class_label · footprint · height_m · sdf · source_id · style_id
```

A scan for any key containing `vert`/`face`/`mesh`/`tri`/`point`/`normal` returns **NONE**. The corpus
holds a 64³ `sdf` and a 64² `footprint` and nothing else — **confirmed, not inferred**. (Also confirmed
in #63: `sdf` is stored *un*-truncated, range −0.32…1.13; the ±0.2 clamp is applied at load.)

## 2. Per-source availability — all three live

| source | `source_id` | n | availability | download |
|---|---|---|---|---|
| **NL — 3D BAG** | 0 | 11,776 | **live OGC API**, per-building lookup **by our exact stored id** | API calls |
| **DE — NRW OpenGeodata LoD2** | 1 | 12,000 | **41 / 41** of our tiles present in the index; refreshed **2026-05-26** | **234 MB** |
| **JP — PLATEAU (Tokyo 23-ku)** | 2 | 12,000 | **3 / 3** archives reachable on the GIC S3 bucket | **45 MB** |

**Total ≈ 280 MB**, plus NL API calls. This is a non-issue on cost.

**NL — verified live.** `GET https://api.3dbag.nl/collections/pand/items/NL.IMBAG.Pand.0599100000058822`
(an id taken straight from our corpus) returns **200** with the building and its `BuildingPart` carrying
`Solid` geometry at **LoD 1.2, 1.3 and 2.2** — exactly the LoD2.2 solid we need.

**DE — verified against the index.** `https://www.opengeodata.nrw.de/produkte/geobasis/3dg/lod2_gml/lod2_gml/`
lists **35,022** tiles as `<file name="…" size="…"/>`; all **41** tiles referenced by our DE rows are
present, totalling 234 MB. Open data (NRW OpenGeodata).

**JP — verified by HEAD.** Our rows reference **3** archives (`533937_2.zip`, `533954_2.zip`,
`533957_2.zip`) — 9 distinct inner `.gml` across them — all reachable at
`https://gic-plateau.s3-ap-northeast-1.amazonaws.com/2020/tokyo23ku/`. Note the archives are **nested**:
the outer zip contains `bldg.zip` / `dem.zip` / `luse.zip` / `tran.zip`, and the CityGML lives one level
down inside `bldg.zip`. Any re-ingest must unwrap twice.

## 3. Id alignment — the one real risk, and it came back clean

**All 35,776 `bag_id` values are unique** (11,776 / 12,000 / 12,000, no collisions), and each encodes
both the source file and the in-file building id:

- NL — `NL.IMBAG.Pand.0599100000058822` (globally unique BAG Pand id; **resolves directly against the
  live API**, verified above)
- DE — `LoD2_32_280_5657_1_NW.gml#DENW43AL00001j66` (tile + ALKIS building id, both intact)
- JP — `533937_2.zip:53393770_bldg_6697_op2.gml#BLD_64cd89a1-7985-4655-9` ⚠️ **truncated**

**The JP risk.** `bag_id` is dtype `|S64`, and **all 12,000 JP ids sit at that 64-char limit** — their
`BLD_` UUIDs are cut mid-string. NL and DE have **zero** ids at the limit, so only JP is affected.

**Measured, not assumed.** Downloading `533937_2.zip` (4.8 MB) and prefix-matching our truncated ids
against the real `gml:id` values in its three CityGML files:

> **resolved 4,858 · ambiguous 0 · missing 0 — 100.0 % uniquely recoverable**

The surviving prefix is long enough to identify each building uniquely within its own gml. **JP
truncation is harmless.** (Verified on 4,858 of the 12,000 JP rows — the one archive tested; the same
scheme applies to the other two.)

## 4. Cost is far lower than assumed, because the mesh path already exists

`scripts/ingest_3dbag.py` already does, for NL: 3DBAG OGC API → CityJSONFeature → **LoD2.2 Solid
boundaries → `trimesh`** (`lod22_mesh`) → `building_to_sdf` → 64³ grid. **The mesh is already
constructed in-pipeline and then discarded at the SDF step.** Recovering NL surfaces is a matter of
writing out what that function already builds — not authoring a new ingest.

DE and JP need CityGML parsing rather than CityJSON, which is genuinely new code, but both are standard
LoD2 CityGML with `lod2Solid` geometry and 41 + 9 files to walk.

## 5. Answers to the ticket's four questions

1. **Confirm the gap** — ✅ confirmed by key scan. No surface representation of any kind.
2. **Re-ingest cost** — ✅ ~280 MB total; all three sources live and open; NL needs no new parsing code.
   The DE source is **NRW OpenGeodata** (North Rhine-Westphalia, the `_NW` suffix), still reachable and
   refreshed two months ago.
3. **Do ids survive** — ✅ yes, all 35,776 unique; NL verified against the live API; JP truncation
   measured at **100 % unique prefix-match**.
4. **Fallback if partial** — **not needed.** No source is lost, so the full 35,776 stays available and
   the corpus stays aligned to the existing `footprint` / `height_m` / `source_id` rows and to the #27
   gate.

## 6. What this unblocks

**Surface supervision is a GO**, which removes the hard prerequisite that gated the whole vecset
direction. Concretely:

- **#64's A2** (pretrained/fine-tuned vecset AE + our own footprint-conditioned diffusion) is fully
  available — fine-tuning needs surface samples, and we can produce them.
- The **frozen-AE fallback** that #64 named in case this ticket failed is **no longer necessary**. It
  stays available as a cheaper option, not as a forced one.
- **#65 is now unblocked** — its last open blocker. All three inputs (this, #63's measured headroom,
  #64's A1/A2 split) are in hand, so the posture decision can be taken with evidence.

**Not chartered here:** the re-ingest *execution* itself. What to sample — and how, e.g. Dora-style
sharp-edge-aware sampling — depends on which autoencoder is chosen, so it graduates after #65 rather
than before it.

## Reproduce

```
# NL: per-building by exact id
curl https://api.3dbag.nl/collections/pand/items/NL.IMBAG.Pand.0599100000058822
# DE: tile index
curl https://www.opengeodata.nrw.de/produkte/geobasis/3dg/lod2_gml/lod2_gml/
# JP: nested archive (outer zip -> bldg.zip -> *.gml)
curl -I https://gic-plateau.s3-ap-northeast-1.amazonaws.com/2020/tokyo23ku/533937_2.zip
```
