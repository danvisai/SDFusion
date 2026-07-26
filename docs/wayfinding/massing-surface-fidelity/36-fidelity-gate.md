# Fidelity Gate — visual-primary (no reliable scalar geometry metric found)

Resolves [Define the surface-fidelity metric and no-regression gate](https://github.com/danvisai/SDFusion/issues/36).

## What we tried and why it failed

Two scalar "crispness" metrics were implemented and calibrated (real LoD2 = positive, current rough samples = negative), and **both failed to separate** the two populations:

| metric | real p10 | sample median | separates? |
|---|---|---|---|
| mesh-vertex local normal-consistency | 0.994 | 0.997 (*higher!*) | ❌ `separation_ok: False` |
| field-gradient normal-consistency (3×3×3) | 0.990 | 0.994 | ❌ `separation_ok: False` |

**Why:** the roughness is **mid-scale waviness** (~10-voxel wobble), but both metrics are *local* (3-voxel), and the VQVAE-decoded fields are all *locally* smooth — so a local measure saturates near 1 for crisp and wavy alike. The property that truly separates crisp buildings from blobs (large flat faces sharing one normal) is *global*, and a cheap scalar for it wasn't worth an open-ended hunt. Artifacts: `execution/artifacts/fidelity_gate_calibration.json`; `scripts/foundations/calibrate_fidelity_gate.py`.

## The gate — visual-primary

- **PRIMARY (required, final arbiter): visual montage sign-off.** Generated held-out output vs real LoD2, judged for crisp flat faces / sharp roof planes vs the current wavy, blobby surfaces. This mirrors #27's own conclusion that scalar proxies mislead on surface judgment and the visual montage is the final arbiter.
- **No-regression guard (automatable): the #27 massing gate must still pass** (collapse ≤1%, ≥85% LCC≥0.90, footprint-IoU median≥0.65/p10≥0.35) — reuse `scripts/foundations/baseline_gate_eval.py`.
- **Optional non-gating diagnostic:** a global normal-concentration ("large-flat-faces") scalar MAY be added later as a screen, but the gate does **not** block on any scalar.

## Reference for the visual gate

The crisp-vs-rough contrast the gate judges: `outputs/surface_roughness/ladder_montage.png` (A real GT & B VQVAE round-trip = crisp; C prior sample = rough) and `slice_montage.png` (crisp vs noisy SDF field).

## Implication for #37

Success for the fix (prior-side, per #35) = **montage shows crisp surfaces (visual sign-off) AND the #27 massing gate still passes.** The tradeoff accepted here: each fix attempt needs a human glance rather than a fully-auto scalar, since no scalar reliably captures the roughness.
