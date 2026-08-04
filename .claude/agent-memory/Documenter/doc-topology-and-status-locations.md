---
name: doc-topology-and-status-locations
description: Where this repo's living status/documentation lives and how to update it during a Documenter sweep
metadata:
  type: reference
---

Documentation topology for SDFusion / GenerativeTowns (learned during the 2026-07-26 doc sweep). There is
**no STATUS.md, CHANGELOG, or single roadmap doc** — do not go looking for one.

- **`docs/wayfinding/` is the living status.** One subdirectory per GitHub issue-map (e.g.
  `solid-massing-generation` = map #24, `massing-surface-fidelity` = #34, `crisp-massing-model` = #52,
  `diffusion-latent-accuracy` = #58). Each carries its own dated result `.md` files **with comparison
  tables + `*.png` montages** and mirrors the GitHub tracker (GitHub is unreachable from the cluster, so
  these docs ARE the source of truth). Treat these as authoritative and current — cross-link them, don't
  rewrite them.
- **`CONTEXT.md`** = the stable conceptual doc: research thesis (C1 transform / C2 compose) + ubiquitous
  language glossary, plus a **`## Project status (updated YYYY-MM-DD)`** section that indexes into the
  wayfinding maps. Update the status section's date + headlines each sweep; leave thesis/glossary alone
  unless a claim is genuinely contradicted.
- **`docs/adr/`** = architecture decision records (stable; don't churn). ADR 0004 fixes detail-scale `s*`
  to the 64³ massing-generator resolution limit — that's a design fact, NOT a codec-roughness claim.
- **`README.md`** = demo/serving user doc (the `main` branch surface). Its old pointers to
  `docs/DEMO_BUILD_PLAN_*` and `docs/HANDOFF_*` were dead (files don't exist) — repointed to
  CONTEXT.md / docs/wayfinding / docs/adr.

**Convention for superseded docs:** annotate with a dated forward-pointer banner ("SUPERSEDED (date) — see
X") rather than deleting or rewriting; preserve the historical record. Applied 2026-07-26 to
`crisp-massing-model/residual-retrain-design.md` and `massing-surface-fidelity/phase2-result.md`.

**Recurring roughness figures** (so a sweep can sanity-check consistency): GT roughness floor ≈ 0.0041;
VQVAE `decode(encode(GT))` round-trip ≈ 0.0044 (codec is NOT the crispness ceiling); every post-hoc
correction so far (SDF refiner #54, latent corrector #59) plateaus at ≈ 0.0047.
