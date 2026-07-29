"""Train the footprint-conditioned token-set denoiser on precomputed vecset latents (spec #67).

Standard noise-prediction training -- the same objective whether the model is later used from noise or
as a projection, which is why aligning with ADR 0003 needed no change here. Inference is where the two
differ, and `SetSDEdit` is the path that ships.

Conditioning is dropped at random during training so classifier-free guidance works at inference
without a second model. Note the drop is PER SAMPLE: `drop_cond` on the denoiser is per-batch, so a
mixed batch is run as two forwards rather than one, and the split is done here rather than pretending
the flag is per-sample.

Held-out rows come from the corpus's own deterministic split, carried in the latent cache, so nothing
the gate scores is ever trained on.

Usage:
    train_vecset.py --latents data/real_massing_v1/vecset_latents.h5 --steps 20000
    train_vecset.py --latents ..._smoke.h5 --steps 200 --width 256 --depth 4   # wiring smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from models.networks.vecset_denoiser import VecsetDenoiser          # noqa: E402
from models.networks.vecset_projection import cosine_alphas          # noqa: E402


class LatentSet(torch.utils.data.Dataset):
    """Precomputed (latent, footprint, height, region), train split only."""

    def __init__(self, path, held_out=False):
        import h5py
        with h5py.File(path, "r") as f:
            m = (f["held_out"][:] == (1 if held_out else 0))
            self.z = f["latent"][:][m]
            self.fp = f["footprint"][:][m]
            self.h = f["height_m"][:][m]
            self.r = f["region"][:][m]
        # normalise the latent to unit scale so the noise schedule is well-posed
        self.mu = float(self.z.astype(np.float32).mean())
        self.sd = float(self.z.astype(np.float32).std()) or 1.0

    def __len__(self):
        return len(self.z)

    def __getitem__(self, i):
        z = (self.z[i].astype(np.float32) - self.mu) / self.sd
        return (torch.from_numpy(z),
                torch.from_numpy(self.fp[i].astype(np.float32))[None],
                torch.tensor(self.h[i], dtype=torch.float32),
                torch.tensor(int(self.r[i]), dtype=torch.long))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", default="data/real_massing_v1/vecset_latents.h5")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--cfg_drop", type=float, default=0.1)
    ap.add_argument("--out", default="logs_building/vecset_v1")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--save_every", type=int, default=2000)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ds = LatentSet(args.latents)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True,
                                     num_workers=2, persistent_workers=True)
    C, FPRES = ds.z.shape[-1], ds.fp.shape[-1]
    print(f"[data] {len(ds)} train latents  tokens={ds.z.shape[1]} ch={C}  fp={FPRES}  "
          f"mu={ds.mu:.3f} sd={ds.sd:.3f}", flush=True)

    net = VecsetDenoiser(latent_channels=C, width=args.width, depth=args.depth,
                         heads=args.heads, footprint_res=FPRES).to(dev)
    n_par = sum(p.numel() for p in net.parameters())
    print(f"[model] {n_par/1e6:.1f}M params", flush=True)

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    ac = cosine_alphas(args.timesteps).to(dev)

    step, t0, hist = 0, time.time(), []
    while step < args.steps:
        for z, fp, h, r in dl:
            if step >= args.steps:
                break
            z, fp, h, r = z.to(dev), fp.to(dev), h.to(dev), r.to(dev)
            t = torch.randint(0, args.timesteps, (z.shape[0],), device=dev)
            a = ac[t].view(-1, 1, 1)
            noise = torch.randn_like(z)
            zt = a.sqrt() * z + (1 - a).sqrt() * noise

            # per-sample CFG dropout, honestly: split the batch rather than pretend the
            # per-batch flag is per-sample
            drop = torch.rand(z.shape[0], device=dev) < args.cfg_drop
            pred = torch.empty_like(z)
            for mask, flag in ((~drop, False), (drop, True)):
                if mask.any():
                    pred[mask] = net(x=zt[mask], t=t[mask], footprint=fp[mask],
                                     height=h[mask], region=r[mask], drop_cond=flag)
            loss = torch.nn.functional.mse_loss(pred, noise)

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            hist.append(loss.item()); step += 1

            if step % args.log_every == 0:
                w = np.mean(hist[-args.log_every:])
                print(f"  step {step:6d}/{args.steps}  loss {w:.4f}  "
                      f"{(time.time()-t0)/step:.2f}s/step", flush=True)
            if step % args.save_every == 0 or step == args.steps:
                torch.save({"model": net.state_dict(), "step": step, "args": vars(args),
                            "latent_mu": ds.mu, "latent_sd": ds.sd,
                            "latent_channels": C, "footprint_res": FPRES},
                           out / "vecset_denoiser.pth")

    json.dump({"steps": step, "final_loss": float(np.mean(hist[-100:])),
               "first_loss": float(np.mean(hist[:100])), "params_M": n_par / 1e6,
               "n_train": len(ds)}, open(out / "train.json", "w"), indent=2)
    print(f"[done] {step} steps  loss {np.mean(hist[:100]):.4f} -> {np.mean(hist[-100:]):.4f}  "
          f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
