# Progress + Research — Appearance, Engine Export, Cleanup, and the Generative Core

**Date:** 2026-06-14 · **Scope:** the AI-sculpting frontier from 2026-06-11→14 — what was
built, the research that informed it, and the open model-improvement threads.
**Companion docs:** `ARCHITECTURAL_DETAIL_RESEARCH_2026-06-10.md`,
`TRAINING_GAPS_RESEARCH_PLAN_2026-06-09.md`, `TRACK2_part_mixing_design.md`.
**Companion memories:** `project_neural_appearance_v0`, `project_localized_snap`,
`project_smart_add`, `project_geometry_cleanup`, `project_architecture_generation_research`,
`project_detailizer_v1`, `project_detail_layer_geometry_fixes`.

---

## 0. The settled architecture (three layers)

The project converged — empirically, after multiple negative results — on a **hybrid** the same
shape as **Roblox Reality** (their 2026-04 "we shouldn't ask a neural engine to become a game
engine"):

| Layer | What | Generative? |
|---|---|---|
| ① **Symbolic state** | footprint, recipe params, edit list, per-building style image | the data model |
| ② **Crisp geometry** | recipes + `sdf_detail` CSG; learned models DECIDE (params, parts, snap) | decisions learned, **surface procedural** |
| ③ **Neural appearance** | diffusion paints textures/pixels conditioned on G-buffers | fully generative |
| → **Engine** | one textured `.glb` → Unreal (Nanite + PBR) | real-time |

**Why ② renders procedurally:** every attempt to make a net synthesize the final surface came
out soft/blurry — REPA (neg), adaLN (neg), three detailizer variants (L1/GAN/layout-cond, all
failed to carve windows). The procedural render is the only thing that stayed crisp. So:
**learned models decide and texture; CSG renders the crisp geometry; the engine renders
real-time.** The honest user-facing framing: the building's *choices* and *textures* are
generative; its *crisp surface* is procedural by necessity.

---

## 1. What was built (2026-06-11 → 14)

### Sculpt loop (geometry, layer ②)
- **Localized generative snap** (`project_localized_snap`): SDEdit only remolds inside/near the
  placed mass (edit-primitive locality mask); the untouched building stays bit-exact. Empty
  edits = no-op; whole-building re-mold is an explicit opt-in. Fixed "snap melts the whole
  building." **Detail re-applied after snap** so the result stays dressed; **details auto-adjust**
  (resnap → set-refiner re-cohere → row-regularize) on wall-moving snaps.
- **SMART ADD** (`project_smart_add`): `/interpret_mass` types a placed primitive (tower/dormer/
  balcony/bay/wing/window/door) from shape+position and replaces it with a crisp typed
  construction. v2 = **generative**: planner-scored typing + SEEDED sampled construction (spire/
  dome at real per-class rates, window rhythm) → same box + different seed = different
  architecture. Wired into sculpt.html (🧠 Make it architecture / 🎲 Re-roll).
- **Live detail preview** (👁): the full ② treatment composed on the current massing at 96³,
  shown live — "what Bake exports."
- **Four production geometry fixes** (`project_detail_layer_geometry_fixes`): roof bbox-body fill,
  steps apron, bbox band/cornice/plinth plates, minaret cone skirt — all "built on bbox not the
  massing." Windows-on-bbox-faces still open (rotated footprints get none).
- **Geometry cleanup** (`project_geometry_cleanup`, NEW 2026-06-14): `scene/mesh_cleanup.py` —
  SDF floating-debris removal (drop small disconnected occupied blobs PRE-mesh) + mesh
  connected-component/weld/degenerate cleanup, wired into EVERY emission point. Fixes "placed
  primitives produce noise." Conservative (keeps a legit detached wing, kills specks).

### Neural appearance (layer ③) — `project_neural_appearance_v0`
- **v0 still render**: sphere-trace our detailed SDF → depth/edge G-buffers → SDXL +
  ControlNet(+IP-Adapter) re-renders OUR geometry photoreal, zero training (~8-15s).
- **Per-building style embeddings**: instance G-buffer (free from the SDF) → instance-masked
  IP-Adapter → each building wears its own style-image look in one render. `/neural_render`,
  `/neural_render_town`.
- **v2 TEXTURE BAKE → Unreal-ready assets**: xatlas UV unwrap → multi-view stylized render →
  back-project into albedo atlas → textured glb. **PBR**: normal (height-from-luminance) +
  per-style metallic-roughness. **Iterative TEXTure** (inpaint each new view over the texture-
  so-far) = seam-free, crisper (cov 0.85 vs 0.77). **Outward winding** fix (marching-cubes +
  axis-reorder reflection was inward → Unreal showed the inside). `/bake_texture`.
- **Town-wide textured export**: `/export_town` (`textures=true`) → ONE multi-textured glb,
  per-building styles, named nodes + ground, cm scale. Plus the gray geometry-only export.

### Quality gates
`scripts/server/test_sculpt_flows.py` (17 flows) + `scripts/server/test_branches.py` (12
branches) + the 23s `scripts/server/eval_visual.py` photo sheet. **17/17 + 12/12 green** as of
2026-06-14. Server endpoints (18): see `/openapi.json`.

### Closed negative results (don't re-run)
REPA, adaLN, 3 detailizer variants (`project_detailizer_v1`), B+.6 diversity ceiling — all
documented. They are *why* ② stays procedural.

---

## 2. Research findings

### A. Image-conditioned 3D + appearance (the v0→v2 basis)
- Image→3D conditions a 3D generator on a frozen **DINOv2/CLIP embedding** (Hunyuan3D-2,
  TRELLIS); **Isotropic3D** generates from a single CLIP embedding (~1KB) → a style can be a
  vector, not an image. **GeoTexBuild** (2504.08419) does footprint→ControlNet→heightmap→
  geometry→stylize for buildings specifically. **Roblox Reality** (2026-04) validated the
  hybrid (engine state + geometry → video model paints pixels).
- **Texture bake**: xatlas UV + multi-view diffusion back-projection; **TEXTure/Text2Tex**
  iterative inpaint for cross-view consistency (now implemented).

### B. Detailization / coarse→detailed geometry (tried, mostly negative)
- DECOR-GAN / DECOLLAGE / ShaDDR / MARS (voxel detailization), DetailGen3D (data-dependent
  flow), SDF-Diffusion (patch SR), Direct3D-S2 (sparse SDF + sharp-edge supervision). Our
  detailizer attempts (L1/GAN/layout-cond) all failed to render windows at 96³ → confirmed the
  procedural-render decision. A real win here needs a **structured/vec-set latent** (TRELLIS /
  3DShape2VecSet), i.e. the representation upgrade — weeks of work.

### C. Geometry cleanup (the 2026-06-14 "noise" fix)
- **NeuManifold** (2305.17134) watertight manifold reconstruction; **SDF regularization** for
  floating-debris removal; **fast surface mesh denoising**; standard recipe = connected-
  component keep-large + vertex weld + degenerate/dup removal + SDF Gaussian. We had none of
  the component/weld steps → now wired in.

### D. Part-coherence + "placed geometry → architecture" (the OPEN model thread)
See `project_architecture_generation_research` for the full synthesis. Headline:
- **Coherence**: **CoPart** (2507.08772, mutual-guidance joint part denoising), **X-Part**
  (2509.08643, adjust one box → regenerate it AND neighbors), **OmniPart** (SIGGRAPH Asia 2025,
  CODE, layout planner → joint refinement), **FullPart** (2510.26140), **BoxSplitGen** (WACV'26).
- **Architecture-specific (best doctrine fit)**: **FacAID** (2406.01829, neuro-symbolic facade →
  editable *procedural* split grammar), **Pro-DG** (2504.01571, shape grammar + diffusion). A
  **grid of cells makes alignment/rhythm true by construction** — the key to "looks like
  architecture."

---

## 3. Open model-improvement threads (sequenced, with the recommendation)

The split is settled (user 2026-06-14); the work is improving the models. Two named threads:

1. **Rhythm/alignment-aware re-cohere** *(RECOMMENDED next)*. The current `PartSetRefiner` is a
   joint-set DDPM over `[type | axis-box | validity]` with a plain ε-loss — **no orientation, no
   relationships, nothing rewards alignment/rhythm/symmetry**, so it can only nudge boxes and
   drop dupes (why "re-coherence is not good"). Fix: add a learned **floor-grid** parts align to
   (+ uniform spacing + wall-attachment in-model) and **X-Part neighbor-regeneration** edit
   semantics, trained on the 28k part instances. Days. Makes windows snap into rows = reads as
   architecture.
2. **Neuro-symbolic facade-program model** (FacAID/Pro-DG style). Predict a split-grammar facade
   program (floors × bays, window type per cell) from (massing + placed proxy); our `sdf_detail`
   renders it crisply. Alignment/rhythm by construction. Higher ceiling, ~weeks, best doctrine
   fit. #1 is a stepping stone (the "grid parts align to" is the same insight).

Also standing (orthogonal, optional): underside texture coverage; USD instancing + collision
proxies (footprint prisms free); the representation upgrade (structured/vec-set latent) if a
genuinely-generative *geometry* core is ever wanted.

---

## 4. Sources (this session's research)
- Hunyuan3D-2 https://arxiv.org/abs/2501.12202 · Isotropic3D https://arxiv.org/abs/2403.10395 · GeoTexBuild https://arxiv.org/abs/2504.08419 · Roblox Reality https://about.roblox.com/newsroom/2026/04/roblox-reality-hybrid-architecture-democratizing-photorealistic-multiplayer-gaming
- DECOR-GAN https://arxiv.org/abs/2012.09159 · DECOLLAGE https://arxiv.org/abs/2409.06129 · MARS https://arxiv.org/abs/2502.11390 · DetailGen3D https://arxiv.org/abs/2411.16820 · SDF-Diffusion (CVPR'23) · Direct3D-S2 https://arxiv.org/abs/2505.17412
- NeuManifold https://arxiv.org/abs/2305.17134 · TEXTure/Text2Tex (iterative texture)
- OmniPart https://arxiv.org/abs/2507.06165 · CoPart https://arxiv.org/abs/2507.08772 · X-Part https://arxiv.org/abs/2509.08643 · FullPart https://arxiv.org/abs/2510.26140 · BoxSplitGen https://boxsplitgen.github.io/ · FacAID https://arxiv.org/abs/2406.01829 · Pro-DG https://arxiv.org/abs/2504.01571
