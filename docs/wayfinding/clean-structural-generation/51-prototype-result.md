# #51 — map-#24 on the demo's own footprints: do NOT integrate

**Date:** 2026-07-28 (renders produced 2026-07-25) · **Verdict: do not swap map-#24 into the demo.**
The prototype answers its question negatively, and in doing so **inverts map #50's premise**.

Assets: `compare.png` (town, oblique), `compare_topdown.png`, `compare_closeups.png`, plus
`outputs/proto_clean_structure/{compare_oblique,native_model}.png`.

---

## The question, and the answer

#51 asked whether map-#24 + the v1 refiner produces the **clean, coherent simple structures** #50 wants,
on the demo's own synthetic / Munich / Lafayette OSM footprints. **It does not — and the procedural
path's *base* already comes closer.**

### Town scale flatters the model

In `compare.png`, map-#24 looks dramatically cleaner: no spheres, no spike forest, no random towers or
gables. If the demo were only ever seen from above at distance, this would be an easy win.

### Building scale reverses it

`compare_closeups.png` (six Munich footprints, solidity 0.53–0.79, four columns: footprint / procedural
full / procedural **base only** / map-#24 + refiner):

- **map-#24 is lumpy and melted.** Rounded walls, eroded edges. On the most irregular footprint
  (solidity 0.53) it degenerates into a blob with floating debris.
- **The procedural *base only* column is crisp and footprint-exact** — flat walls, sharp vertical
  edges, L- and Z-shapes followed faithfully.
- The town-scale "cleanliness" was distance hiding the defect, not an absence of one.

### Top-down confirms both, and adds a third finding

`compare_topdown.png`:

- map-#24 tracks the footprint **plan** reasonably, but with **wobbly, eroded boundaries**; crisp
  rectangles in `synthetic_blocks` come back rounded and ragged.
- **Height variety collapses.** map-#24 buildings are uniformly low — the town loses its skyline, which
  the procedural path has.

⚠️ **Caveat on the height finding:** this may be a harness gap rather than a model defect. If the
offline prototype did not wire height conditioning through, map-#24 would default to flat. **Confirm
before holding it against the model.**

## Root cause of the spike forest — it is a bug in the base recipe, not the detail composer

`scene/sdf_recipes.py:297`:

```python
def _shrink_polygon(poly, amount):
    c = p.mean(axis=0)                      # centroid
    shrink = (norms - amount).clip(0.1) / np.maximum(norms, 1e-6)
    return c + v * shrink
```

This is a **radial shrink toward the centroid**, not a true polygon offset. For a concave OSM footprint
the centroid frequently lies **outside** the polygon, so the "inner" polygon **self-intersects**, and
the parapet ring built from it at `scene/sdf_recipes.py:80` —
`sdf_subtract(parapet, parapet_inner)` — is left with thin slivers. Those slivers mesh as the picket
fence of vertical spikes.

The evidence matches exactly: **spikes appear on Munich's irregular footprints and never on
`synthetic_blocks`' rectangles.**

Separately, the domes and spires are **style recipes**, not random detail: `recipe_public_civic`
(`sdf_recipes.py:252`, dome + drum) and `recipe_victorian` (`:148`, conical spire). They are selected by
style, so they are a *choice* to revisit, not a bug.

## What this does to map #50

#50 was written to **replace** the procedural per-building generation with map-#24. The evidence says
that swap trades a crisp, footprint-exact base for a lumpy one — and per
[map #52](https://github.com/danvisai/SDFusion/issues/52) /
[#58](https://github.com/danvisai/SDFusion/issues/58), map-#24's lumpiness is **not fixable** in the
dense-grid stack (five documented negatives; a hard 0.0047 roughness wall against a 0.0041 GT floor).

So the demo has two honest paths, and they are not in conflict:

1. **Now:** keep the procedural base and fix its bugs — the `_shrink_polygon` offset, and a decision on
   the dome/spire style recipes. Cheap, and it sidesteps the #48 conditioning-slot incompatibility
   entirely, since no integration is required.
2. **Later:** integrate the **A2 model** from [map #61](https://github.com/danvisai/SDFusion/issues/61)
   — a query-based vecset decoder measured at **0.00328** roughness (#63), i.e. at the GT floor — once
   it is trained. That, not map-#24, is the model worth swapping in.

**Recommendation: do not integrate map-#24.** Fix the base recipe for the demo, and let #61's A2 model
be the eventual replacement.

## Open, not answered here

- Whether the height collapse is a harness gap (see caveat) — cheap to check, worth checking.
- Whether the dome/spire style recipes stay, get restricted to plausible classes, or go — a
  keep/cull judgement, and the same kind of call
  [#49](https://github.com/danvisai/SDFusion/issues/49) exists to make for the demo's features.
