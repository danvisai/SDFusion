"""Stage the keeper checkpoints for transfer off the cluster.

REPRODUCING.md §6 records that the weights are hosted nowhere and exist only on
this filesystem. This strips the optimizer state -- which is 70% of a vecset
checkpoint and 50% of the stage3a baseline, and is only needed to *resume*
training -- and writes the result to transfer/weights/ with a checksum manifest.

Everything else is kept. For the vecset models that means latent_mu/latent_sd,
which are load-bearing: the denoiser trains on globally normalised latents and
decodes to noise without them. For stage3a it means both df and ema_df, since
load_ckpt requires df and defaults to ema_df at inference (stage3a_model.py:893,
:899) -- dropping either changes which model you are scoring.
"""
import hashlib
import pathlib
import sys

import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
DEST = REPO / "transfer" / "weights"
DROP = ("opt", "sched")

# Tier A -- the current line of work. Scored on the 48-id harness, cited in the writeup.
TIER_A = [
    ("logs_building/vecset_v5_surfband/vecset_denoiser_step240000.pth",
     "vecset_v5_surfband_step240000.pth", "band fix -- FINAL, scored (29/48 solid)"),
    ("logs_building/vecset_v5_surfband/vecset_denoiser_step230000.pth",
     "vecset_v5_surfband_step230000.pth", "band fix -- best 3D IoU 0.825, post-recovery"),
    ("logs_building/vecset_v5_surfband/vecset_denoiser_step220000.pth",
     "vecset_v5_surfband_step220000.pth", "band fix -- the collapse, kept as evidence"),
    ("logs_building/vecset_v4_surf/vecset_denoiser.pth",
     "vecset_v4_surf.pth", "surface-loss model, pre-band-fix (+0.029 IoU)"),
    ("logs_building/vecset_v3_pair_long/vecset_denoiser_step180000.pth",
     "vecset_v3_pair_long_step180000.pth", "41-epoch control, no surface loss"),
]

# Tier B -- latest of every other run. Kept so no run is lost, not because each is good.
TIER_B = [
    ("logs_building/2026-07-16-stage3a-lod2-fromscratch-region/ckpt/stage3a_steps-latest.pth",
     "stage3a_lod2_deployed.pth", "deployed map-#24 baseline -- comparison arm"),
    ("logs_building/vecset_v2_pair/vecset_denoiser_step60000.pth",
     "vecset_v2_pair_step60000.pth", "pre-frame-fix pair run (latents were transposed)"),
    ("logs_building/vecset_v2_plain/vecset_denoiser.pth",
     "vecset_v2_plain.pth", "pre-frame-fix plain run"),
    ("logs_building/vecset_v1/vecset_denoiser.pth",
     "vecset_v1.pth", "first vecset run"),
    ("logs_building/vecset_pair_v1/vecset_denoiser.pth",
     "vecset_pair_v1.pth", "first aligned-pair run"),
    ("logs_building/vqvae_clean_ft/vqvae_clean.pth",
     "vqvae_clean_ft.pth", "cleaned VQVAE fine-tune (dense-grid era)"),
    ("logs_building/2025-05-19T19-58-28-vqvae-building-all-res64-LR1e-4-T0.2-release/ckpt/vqvae_steps-latest.pth",
     "vqvae_release_res64.pth", "released 64^3 VQVAE codec"),
    ("logs_building/monolith_v3/ckpt/monolith_steps-latest.pth",
     "monolith_v3.pth", "monolith arm v3 (C2 thesis)"),
    ("logs_building/monolith_v2/ckpt/monolith_steps-latest.pth",
     "monolith_v2.pth", "monolith arm v2"),
    ("logs_building/monolith_v1/ckpt/monolith_steps-latest.pth",
     "monolith_v1.pth", "monolith arm v1"),
]

# Tier C -- documented NEGATIVE results and smoke tests. 14.2 GB each, ~67 GB total.
# The finding is in docs/; the weights add little. Excluded by default; --include-negatives adds them.
TIER_C = [
    ("logs_building/x0sharp-w05-clip/ckpt/stage3a_steps-latest.pth",
     "x0sharp_w05_clip.pth", "NEGATIVE #60: stable but roughness unchanged, footprint eroded"),
    ("logs_building/x0sharp-gradtv-w1-pilot/ckpt/stage3a_steps-latest.pth",
     "x0sharp_gradtv_w1.pth", "NEGATIVE #60: w=0.1 diverged into rubble"),
    ("logs_building/2026-06-08T11-50-42-stage3a-hybrid-clean/ckpt/stage3a_steps-latest.pth",
     "stage3a_hybrid_clean.pth", "superseded hybrid-clean architecture"),
    ("logs_building/continue-stage3a-xcultural-warmstart-ft-final/ckpt/stage3a_steps-3000.pth",
     "stage3a_xcultural_ft_final.pth", "cross-cultural warmstart fine-tune, final"),
    ("logs_building/continue-stage3a-xcultural-warmstart-ft/ckpt/stage3a_steps-3000.pth",
     "stage3a_xcultural_ft.pth", "cross-cultural warmstart fine-tune"),
    ("logs_building/smoke-lod2-fromscratch-region/ckpt/stage3a_steps-latest.pth",
     "smoke_lod2_fromscratch.pth", "SMOKE TEST -- superseded by the real run"),
]

# Tier D -- the demo's snap prior. Exact files resolved from scripts/server/refine.py:457-471:
# main is the -final run's latest, guide is an EARLIER checkpoint of the SAME finetune run used
# for autoguidance. The commented-out hybrid-clean path is pre-2026-07-03 and is NOT needed.
# Staged to demo-serving/ so the paths mirror what refine.py expects.
TIER_D = [
    ("logs_building/continue-stage3a-xcultural-warmstart-ft-final/ckpt/stage3a_steps-latest.pth",
     "demo-serving/logs_building/continue-stage3a-xcultural-warmstart-ft-final/ckpt/stage3a_steps-latest.pth",
     "snap prior MAIN (refine.py:469)"),
    ("logs_building/continue-stage3a-xcultural-warmstart-ft/ckpt/stage3a_steps-1000.pth",
     "demo-serving/logs_building/continue-stage3a-xcultural-warmstart-ft/ckpt/stage3a_steps-1000.pth",
     "snap prior GUIDE for autoguidance (refine.py:471)"),
]

KEEPERS = TIER_A + TIER_B


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main():
    keepers = list(KEEPERS)
    if "--include-negatives" in sys.argv:
        keepers += TIER_C
    if "--demo-only" in sys.argv:
        keepers = list(TIER_D)
    elif "--include-demo" in sys.argv:
        keepers += TIER_D
    if "--list" in sys.argv:
        for rel, out_name, note in keepers:
            src = REPO / rel
            mark = "ok " if src.exists() else "MISSING"
            size = src.stat().st_size / 1e9 if src.exists() else 0.0
            print(f"{mark} {size:7.2f} GB  {out_name:34s} {note}")
        return 0

    DEST.mkdir(parents=True, exist_ok=True)
    rows, total_in, total_out = [], 0, 0

    for rel, out_name, note in keepers:
        src = REPO / rel
        if not src.exists():
            print(f"MISSING  {rel}", flush=True)
            continue
        dst = DEST / out_name
        dst.parent.mkdir(parents=True, exist_ok=True)  # tier D mirrors refine.py's nested paths
        size_in = src.stat().st_size

        state = torch.load(src, map_location="cpu", weights_only=False)
        dropped = [k for k in DROP if k in state]
        for k in dropped:
            del state[k]
        torch.save(state, dst)

        size_out = dst.stat().st_size
        total_in += size_in
        total_out += size_out
        rows.append((out_name, rel, size_out, sha256(dst), note))
        print(f"{out_name:38s} {size_in/1e9:6.2f} -> {size_out/1e9:5.2f} GB  "
              f"(dropped {','.join(dropped) or 'nothing'})", flush=True)

    manifest = DEST / "MANIFEST.txt"
    with open(manifest, "w") as fh:
        fh.write("Keeper checkpoints, optimizer state stripped.\n")
        fh.write("Source: /scratch/gilbreth/dsimhadr/GenerativeTowns/SDFusion\n")
        fh.write("Verify after transfer:  sha256sum -c SHA256SUMS\n\n")
        for name, rel, size, digest, note in rows:
            fh.write(f"{name}\n  from  {rel}\n  size  {size/1e9:.2f} GB\n"
                     f"  what  {note}\n  sha256 {digest}\n\n")
    with open(DEST / "SHA256SUMS", "w") as fh:
        for name, _, _, digest, _ in rows:
            fh.write(f"{digest}  {name}\n")

    print(f"\n{len(rows)} files: {total_in/1e9:.1f} GB -> {total_out/1e9:.1f} GB "
          f"({100*(1-total_out/max(total_in,1)):.0f}% smaller)")
    print(f"staged in {DEST}")


if __name__ == "__main__":
    sys.exit(main())
