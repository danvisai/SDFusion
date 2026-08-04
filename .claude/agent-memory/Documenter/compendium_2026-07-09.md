---
name: compendium-created-2026-07-09
description: Comprehensive project documentation compiled into a single reference document
metadata:
  type: project
---

# Compendium Created 2026-07-09

**Document:** `docs/PROJECT_COMPENDIUM_2026-07-09.md` (180 lines, ~9 KB)

**Contents:** Single-file reference synthesizing all handoff docs (2026-05-19 through 2026-07-07), build specs, and session chronicles into 8 structured sections:

1. **Project overview** — design doctrine (learned decisions / procedural realization), the two web pages (town page + SDF Sculptor), lossless round-trip editing
2. **Status ledger** — 16-row table covering all shipped/in-progress/parked features (recipe diffusion through sketch relief v2) with commits, source docs, and notes
3. **Data map** — 8 datasets (BuildingNet, real massing v1 cross-cultural, ornaments, element library) → producer/consumer relationships
4. **Checkpoint map** — 10 serving checkpoints (recipe_param_diffusion_b6, Stage3a xcultural warmstart, planners, composer, SDXL/Depth-Anything auto-downloads)
5. **Outputs map** — 9 notable output directories (weathering, sketch_relief_verify, layerA_eval, demo_video, element_library_v1, etc.)
6. **Test/gate reference** — 13 branch + 18 sculpt flow tests; gate coverage; known F16 flake; missing `/paint_relief` test noted
7. **Demo bundle** — 8.24 GB runnable archive; verification against both gate suites; git clone fix (sample images tracked)
8. **Open items & deferred work** — consolidated: non-convex window bug, weathering preview gap, ornament library size, F16 policy, relief stacking contract, post-demo research (Phase G, corpus, disk purge)

**Key decisions embedded:**
- Later-dated docs win on conflicts (e.g., HANDOFF_2026-07-06.md supersedes 2026-07-03 on ornament placement)
- Context-snap (Layer-A/AB) explicitly PARKED with evidence (layerA_eval/ shows no win)
- AI-detailing panel REMOVED 2026-07-08 but planner model retained for Make-it-architecture
- Upstream SDFusion training on `upstream-training` branch; main is serving-only
- Phase G (crop inpainting) conditional on Phase R eval gap; not a blocker for demo

**Inline citations:** All major facts cite source doc name (e.g., `HANDOFF_2026-07-06.md`) so users can deep-dive the original.

**Why useful:** Single read covers the entire project's design, shipped features, data/checkpoint topology, test strategy, demo verification, and unfinished work — no need to hunt across 20+ docs.
