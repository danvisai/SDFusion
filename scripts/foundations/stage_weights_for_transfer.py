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

KEEPERS = [
    ("logs_building/vecset_v4_surf/vecset_denoiser.pth",
     "vecset_v4_surf.pth", "best model -- surface-loss fine-tune (#80)"),
    ("logs_building/vecset_v3_pair_long/vecset_denoiser_step180000.pth",
     "vecset_v3_pair_long_step180000.pth", "41-epoch control, surface-loss starting point"),
    ("logs_building/vecset_v5_surfband/vecset_denoiser_step220000.pth",
     "vecset_v5_surfband_step220000.pth", "band fix -- scored, collapse checkpoint"),
    ("logs_building/vecset_v5_surfband/vecset_denoiser_step230000.pth",
     "vecset_v5_surfband_step230000.pth", "band fix -- scored"),
    ("logs_building/vecset_v5_surfband/vecset_denoiser_step240000.pth",
     "vecset_v5_surfband_step240000.pth", "band fix -- scored"),
    ("logs_building/2026-07-16-stage3a-lod2-fromscratch-region/ckpt/stage3a_steps-latest.pth",
     "stage3a_lod2_deployed.pth", "deployed map-#24 baseline -- comparison arm only"),
]


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    rows, total_in, total_out = [], 0, 0

    for rel, out_name, note in KEEPERS:
        src = REPO / rel
        if not src.exists():
            print(f"MISSING  {rel}", flush=True)
            continue
        dst = DEST / out_name
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
