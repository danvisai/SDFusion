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
    """Precomputed (latent, footprint, height, region), train split only.

    With `blockout_path`, also serves the ALIGNED PARTNER: the latent of the footprint extrusion the
    generator is actually handed at inference. Training from Gaussian corruption of real latents alone
    left the model unable to start from a blockout -- it denoised in-distribution latents well
    (cos 0.707 -> 0.935 at s=0.5) yet collapsed on blockouts. Pairs close that gap by construction.

    Rows are matched by corpus id, not by position, since the two passes can drop different buildings.
    """

    def __init__(self, path, held_out=False, blockout_path=None):
        import h5py
        with h5py.File(path, "r") as f:
            m = (f["held_out"][:] == (1 if held_out else 0))
            self.z = f["latent"][:][m]
            self.fp = f["footprint"][:][m]
            self.h = f["height_m"][:][m]
            self.r = f["region"][:][m]
            rows = f["row"][:][m]
        self.zb = None
        if blockout_path:
            with h5py.File(blockout_path, "r") as g:
                brow, bz = g["row"][:], g["latent"][:]
            idx = {int(r): i for i, r in enumerate(brow)}
            keep = np.array([i for i, r in enumerate(rows) if int(r) in idx])
            if len(keep) == 0:
                raise SystemExit("no rows shared between the latent and blockout caches")
            self.z, self.fp = self.z[keep], self.fp[keep]
            self.h, self.r = self.h[keep], self.r[keep]
            self.zb = bz[[idx[int(rows[i])] for i in keep]]
            print(f"[pairs] {len(keep)} aligned blockout/real pairs")
        # normalise the latent to unit scale so the noise schedule is well-posed
        self.mu = float(self.z.astype(np.float32).mean())
        self.sd = float(self.z.astype(np.float32).std()) or 1.0

    def __len__(self):
        return len(self.z)

    def __getitem__(self, i):
        z = (self.z[i].astype(np.float32) - self.mu) / self.sd
        zb = ((self.zb[i].astype(np.float32) - self.mu) / self.sd) if self.zb is not None else z
        return (torch.from_numpy(z), torch.from_numpy(zb),
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
    ap.add_argument("--blockouts", default=None,
                    help="aligned blockout latent cache; enables pair training")
    ap.add_argument("--pair_frac", type=float, default=0.8,
                    help="fraction of steps corrupted FROM the blockout rather than from the real "
                         "latent; the remainder keeps a plain denoiser so the manifold is retained")
    ap.add_argument("--pair_t_min", type=float, default=0.35,
                    help="pair steps sample t only ABOVE this fraction of the schedule. Below it the "
                         "target epsilon diverges: it is sqrt(a)/sqrt(1-a)*(blockout-real) + eps, so "
                         "as a->1 the model would be asked to bridge the pair in one unbounded step. "
                         "It also sets the floor for inference strength -- projecting below this is "
                         "asking for a correction the model was never trained to make.")
    ap.add_argument("--surf_weight", type=float, default=0.0,
                    help="#80: weight of the decoded-surface term. 0 disables it and the codec is "
                         "never loaded. The latent eps-loss alone was measured (#76) as unable to "
                         "rank its own candidates -- Spearman +0.12 pooled across error families, "
                         "i.e. mildly WRONG-signed -- so nothing in it reaches the decoded surface.")
    ap.add_argument("--surf_points", type=int, default=8192,
                    help="query points per selected sample. Cheap: cost is dominated by `decode`, "
                         "not by point count (#76 measured 1k -> 32k as only 0.172s -> 0.313s), so "
                         "be generous here and stingy with --surf_bs instead.")
    ap.add_argument("--surf_bs", type=int, default=1,
                    help="how many batch elements get the surface term. This is the real cost knob: "
                         "one element at 8192 points adds ~67%% to a 305ms step.")
    ap.add_argument("--surf_t_max", type=float, default=0.85,
                    help="hard ceiling: samples above this fraction of the schedule are skipped "
                         "entirely. The main protection is not this cutoff but the alpha_bar "
                         "WEIGHTING applied to the term -- x0_hat = (x_t - sqrt(1-a)*eps_hat)/"
                         "sqrt(a), and 1/sqrt(a) amplifies eps-error without bound as t rises, which "
                         "is the mechanism #60 measured diverging into rubble. Weighting by alpha_bar "
                         "makes an unreliable x0_hat contribute ~nothing on its own. ⚠️ Do NOT set "
                         "this at or below --pair_t_min: the two are complementary, and a hard mask "
                         "there silently disables the term on every pair step.")
    ap.add_argument("--out", default="logs_building/vecset_v1")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--save_every", type=int, default=2000)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ds = LatentSet(args.latents, blockout_path=args.blockouts)
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

    # #80: the codec is loaded ONLY when the surface term is on -- it is 191.6M parameters and
    # ~2 GB, and every run that does not use it should not pay for it. `differentiable=True` opens
    # the gradient path; `freeze()` is what keeps the decoder's own weights out of the optimiser.
    codec = None
    if args.surf_weight > 0:
        from models.shape_codec import DoraCodec
        from scripts.foundations.dora_roundtrip_probe import load_dora
        codec = DoraCodec(load_dora(dev), differentiable=True).freeze()
        print(f"[surf] decoded-surface loss ON  w={args.surf_weight}  "
              f"{args.surf_points} pts x {args.surf_bs} sample(s)  t<={args.surf_t_max}", flush=True)

    step, t0, hist, surf_hist = 0, time.time(), [], []
    while step < args.steps:
        for z, zb, fp, h, r in dl:
            if step >= args.steps:
                break
            z, zb = z.to(dev), zb.to(dev)
            fp, h, r = fp.to(dev), h.to(dev), r.to(dev)
            use_pair = ds.zb is not None and np.random.rand() < args.pair_frac
            lo = int(args.pair_t_min * args.timesteps) if use_pair else 0
            t = torch.randint(lo, args.timesteps, (z.shape[0],), device=dev)
            a = ac[t].view(-1, 1, 1)
            noise = torch.randn_like(z)

            # ALIGNED PAIRS: corrupt FROM the blockout, keep the target as the REAL latent, so the
            # implied epsilon is whatever carries blockout -> real. That is exactly what inference
            # does; the earlier model failed only because training corrupted from the real latent
            # while inference started from a blockout.
            #
            # Restricted to high t on purpose. The target is
            #     sqrt(a)/sqrt(1-a) * (blockout - real) + eps
            # which diverges as a -> 1, so at low t the objective is ill-posed rather than merely
            # hard -- the first attempt at this exploded to loss ~40-70 for exactly that reason.
            src = zb if use_pair else z
            zt = a.sqrt() * src + (1 - a).sqrt() * noise
            if use_pair:
                noise = (zt - a.sqrt() * z) / (1 - a).sqrt()

            # per-sample CFG dropout, honestly: split the batch rather than pretend the
            # per-batch flag is per-sample
            drop = torch.rand(z.shape[0], device=dev) < args.cfg_drop
            pred = torch.empty_like(z)
            for mask, flag in ((~drop, False), (drop, True)):
                if mask.any():
                    pred[mask] = net(x=zt[mask], t=t[mask], footprint=fp[mask],
                                     height=h[mask], region=r[mask], drop_cond=flag)
            loss = torch.nn.functional.mse_loss(pred, noise)

            # #80: the decoded-surface term. Supervised against the decode of the TRUE latent rather
            # than against real.h5's field -- the codec round-trips at 0.999 so the true decode is the
            # reachable ceiling anyway, it needs no GT lookup in the loop, and it penalises exactly
            # the failure #73 identified: a latent that decodes differently from the true one. Both
            # sides go through the same frozen decoder, so the codec's own error cancels.
            surf_val = 0.0
            if codec is not None:
                # x0_hat is the same target in both regimes: for pair steps `noise` was redefined so
                # that zt = sqrt(a)*z + sqrt(1-a)*noise, hence x0_hat approximates z either way.
                # Take the LOWEST-t samples in the batch and weight by alpha_bar, rather than hard
                # -masking on t. A hard mask at 0.35 is exactly complementary to `--pair_t_min`
                # (pair steps sample t >= 0.35 by construction), so it silently never fired on pair
                # steps -- 80% of training, and the ones that do the actual task. Weighting keeps
                # #60's protection (a wildly wrong x0_hat contributes ~nothing, since alpha_bar is
                # small exactly where 1/sqrt(alpha_bar) blows the error up) without the blind spot.
                order = torch.argsort(t)[:args.surf_bs]
                sel = order[t[order].float() / args.timesteps <= args.surf_t_max]
                if sel.numel():
                    asel = a[sel]
                    x0 = (zt[sel] - (1 - asel).sqrt() * pred[sel]) / asel.sqrt()
                    pts = torch.rand(sel.numel(), args.surf_points, 3, device=dev) * 2 - 1
                    with torch.no_grad():
                        tgt = codec.query(z[sel] * ds.sd + ds.mu, pts)
                    got = codec.query(x0 * ds.sd + ds.mu, pts)
                    w_t = asel.reshape(sel.numel(), *([1] * (got.dim() - 1)))
                    surf = (w_t * (got - tgt) ** 2).mean()
                    loss = loss + args.surf_weight * surf
                    surf_val = float(surf.detach())
            surf_hist.append(surf_val)

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            hist.append(loss.item()); step += 1

            if step % args.log_every == 0:
                w = np.mean(hist[-args.log_every:])
                sfx = ""
                if codec is not None:
                    sfx = f"  surf {np.mean(surf_hist[-args.log_every:]):.4f}"
                print(f"  step {step:6d}/{args.steps}  loss {w:.4f}{sfx}  "
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
