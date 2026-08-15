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
import secrets
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


def latent_moments(latents, chunk_rows: int = 32,
                   indices: np.ndarray | None = None) -> tuple[float, float]:
    """Compute cache-wide float32 moments with bounded temporary memory.

    A production cache is roughly 10 GB in fp16.  Calling ``latents.astype(float32)`` materialises
    another ~20 GB array and can OOM before the first training step.  Parallel/Welford merging keeps
    the temporary conversion to ``chunk_rows`` while retaining stable corpus-level moments.
    """
    count, mean, m2 = 0, 0.0, 0.0
    rows = latents.shape[0] if indices is None else len(indices)
    for start in range(0, rows, chunk_rows):
        stop = min(start + chunk_rows, rows)
        selection = slice(start, stop) if indices is None else indices[start:stop]
        chunk = np.asarray(latents[selection], dtype=np.float32)
        chunk_count = int(chunk.size)
        chunk_mean = float(chunk.mean(dtype=np.float64))
        chunk_m2 = float(np.square(chunk - chunk_mean, dtype=np.float64).sum(dtype=np.float64))
        delta = chunk_mean - mean
        total = count + chunk_count
        mean += delta * chunk_count / total
        m2 += chunk_m2 + delta * delta * count * chunk_count / total
        count = total
    if count == 0:
        return 0.0, 1.0
    return mean, (m2 / count) ** 0.5 or 1.0


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
        self.real_path = str(path)
        self.blockout_path = str(blockout_path) if blockout_path else None
        self._real_h5 = None
        self._blockout_h5 = None
        with h5py.File(path, "r") as f:
            m = (f["held_out"][:] == (1 if held_out else 0))
            real_indices = np.flatnonzero(m)
            self.latent_shape = tuple(f["latent"].shape[1:])
            self.fp = f["footprint"][real_indices]
            self.h = f["height_m"][real_indices]
            self.r = f["region"][real_indices]
            rows = f["row"][:][m]
        self._real_indices = real_indices
        self._blockout_indices = None
        if blockout_path:
            with h5py.File(blockout_path, "r") as g:
                brow = g["row"][:]
            idx = {int(r): i for i, r in enumerate(brow)}
            keep = np.array([i for i, r in enumerate(rows) if int(r) in idx])
            if len(keep) == 0:
                raise SystemExit("no rows shared between the latent and blockout caches")
            self._real_indices = self._real_indices[keep]
            self.fp = self.fp[keep]
            self.h, self.r = self.h[keep], self.r[keep]
            self._blockout_indices = np.asarray([idx[int(rows[i])] for i in keep], np.int64)
            print(f"[pairs] {len(keep)} aligned blockout/real pairs")
        # Footprint solidity = mask area / convex-hull area. Precomputed ONCE here, not in the
        # training loop: a ConvexHull per batch element per step would dominate a 305 ms denoiser
        # step. 1.0 = convex, lower = re-entrant (courtyards, L-plans, terraced party walls).
        self.solidity = np.ones(len(self._real_indices), np.float32)
        try:
            from scipy.spatial import ConvexHull
            for i in range(len(self._real_indices)):
                ys, xs = np.nonzero(self.fp[i] > 0)
                if len(xs) < 3:
                    continue
                try:
                    hull = ConvexHull(np.c_[xs, ys].astype(float)).volume   # 2-D: .volume is area
                except Exception:
                    continue                                                # degenerate/collinear
                if hull > 0:
                    self.solidity[i] = float((self.fp[i] > 0).sum() / hull)
            self.solidity = np.clip(self.solidity, 0.0, 1.0)
            print(f"[solidity] median {np.median(self.solidity):.3f}  "
                  f"min {self.solidity.min():.3f}  <0.9: {(self.solidity < 0.9).mean()*100:.1f}%")
        except ImportError:
            print("[solidity] scipy unavailable -- solidity fixed at 1.0")

        # normalise the latent to unit scale so the noise schedule is well-posed
        with h5py.File(path, "r") as f:
            self.mu, self.sd = latent_moments(f["latent"], indices=self._real_indices)

    def __len__(self):
        return len(self._real_indices)

    @property
    def has_blockouts(self) -> bool:
        return self._blockout_indices is not None

    @staticmethod
    def _open(path: str):
        import h5py
        return h5py.File(path, "r")

    def _real_latent(self, i: int) -> np.ndarray:
        if self._real_h5 is None:
            self._real_h5 = self._open(self.real_path)
        return self._real_h5["latent"][int(self._real_indices[i])]

    def _blockout_latent(self, i: int) -> np.ndarray:
        if self._blockout_h5 is None:
            assert self.blockout_path is not None and self._blockout_indices is not None
            self._blockout_h5 = self._open(self.blockout_path)
        return self._blockout_h5["latent"][int(self._blockout_indices[i])]

    def __getstate__(self):
        """Never pickle or inherit an open HDF5 handle into a DataLoader worker."""
        state = self.__dict__.copy()
        state["_real_h5"] = None
        state["_blockout_h5"] = None
        return state

    def __del__(self):
        for handle_name in ("_real_h5", "_blockout_h5"):
            handle = getattr(self, handle_name, None)
            if handle is not None:
                handle.close()

    def __getitem__(self, i):
        z = (self._real_latent(i).astype(np.float32) - self.mu) / self.sd
        zb = ((self._blockout_latent(i).astype(np.float32) - self.mu) / self.sd
              if self.has_blockouts else z)
        return (torch.from_numpy(z), torch.from_numpy(zb),
                torch.from_numpy(self.fp[i].astype(np.float32))[None],
                torch.tensor(self.h[i], dtype=torch.float32),
                torch.tensor(int(self.r[i]), dtype=torch.long),
                torch.tensor(float(self.solidity[i]), dtype=torch.float32))


def surface_term(got, tgt, w_t, sample_w=None, norm=None):
    """The decoded-surface reduction. **Extracted so the tests call this, not a copy of it.**

    Returns `(weighted, unweighted)`. `unweighted` is what gets logged, so `surf` stays comparable to
    runs without a weighting flag -- logging the weighted value would let a run look better purely by
    down-weighting its own hard cases.

    `sample_w` is a per-sample weight (footprint solidity, or a per-region constant). `norm` is the
    **corpus-level mean** of that weight.

    ⚠️ `norm` is why this signature exists. The first version divided by `sample_w.mean()` -- the mean
    over the SELECTED window. With `--surf_bs 1` that window holds one element, so the quotient was
    identically 1.0 and **both weighting flags were exact no-ops**; #84's own 3.8-day run trained with
    flat weighting and nobody noticed. Normalising by a fixed corpus mean keeps the intent (total
    pressure preserved, pressure redistributed) while working at any `surf_bs`, including 1.
    """
    per = (w_t * (got - tgt) ** 2).flatten(1).mean(1)          # keep the per-sample dimension alive
    unweighted = per.mean()
    if sample_w is None:
        return unweighted, unweighted
    w = sample_w if norm is None else sample_w / max(float(norm), 1e-8)
    return (per * w).mean(), unweighted


class ExperimentRng:
    """Independent stochastic streams for a controlled training run.

    #92 compares token ordering and the decoded-surface term in a 2x2. Those interventions must not
    also change batch order, pair/plain selection, timesteps, diffusion noise, or CFG dropout. Surface
    queries use their own stream because only two arms draw them; sharing torch's global stream would
    shift every later diffusion draw in those arms and quietly confound the factorial comparison.

    This is training-only experiment control. It does not touch SetSDEdit's inference seed or the town
    demo's per-building seed decorrelation.
    """

    _SEED_MODULUS = 2**63 - 1
    _STREAM_STRIDE = 1_000_003

    def __init__(self, seed: int | None, device: str):
        self.seed = (int(seed) if seed is not None else secrets.randbelow(self._SEED_MODULUS))
        self.seed %= self._SEED_MODULUS
        self.device = device
        self.model_seed = self._stream_seed(0)
        self.pair = np.random.default_rng(self._stream_seed(1))
        self.loader = torch.Generator(device="cpu").manual_seed(self._stream_seed(2))
        self.training = torch.Generator(device=device).manual_seed(self._stream_seed(3))
        self.surface = torch.Generator(device=device).manual_seed(self._stream_seed(4))

    def _stream_seed(self, stream: int) -> int:
        return (self.seed + stream * self._STREAM_STRIDE) % self._SEED_MODULUS

    def pair_random(self) -> float:
        return float(self.pair.random())

    def randint(self, low: int, high: int, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randint(low, high, shape, generator=self.training, device=self.device)

    def randn(self, shape: tuple[int, ...], dtype=torch.float32) -> torch.Tensor:
        return torch.randn(shape, generator=self.training, device=self.device, dtype=dtype)

    def rand(self, shape: tuple[int, ...], dtype=torch.float32) -> torch.Tensor:
        return torch.rand(shape, generator=self.training, device=self.device, dtype=dtype)

    def surface_rand(self, shape: tuple[int, ...], dtype=torch.float32) -> torch.Tensor:
        return torch.rand(shape, generator=self.surface, device=self.device, dtype=dtype)


def build_arg_parser() -> argparse.ArgumentParser:
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
    ap.add_argument("--surf_weight_by", choices=("solidity", "region"), default=None,
                    help="redistribute the decoded-surface term PER SAMPLE (#84). 'solidity' weights "
                         "by footprint area / convex-hull area; 'region' weights by --surf_region_weights. "
                         "Both are normalised by the CORPUS mean of the weight, so total surface "
                         "pressure is preserved and only its distribution changes. ⚠️ Measured: "
                         "region predicts the band-fix collapse ~3.6x more strongly than solidity, and "
                         "only 7.8%% of the corpus has solidity < 0.9, so the solidity variant has "
                         "very little to redistribute across.")
    ap.add_argument("--surf_region_weights", default="0.387,0.574,0.779",
                    help="per-region weights for --surf_weight_by region: the measured SOLID RATE per "
                         "region (BAG/NRW/PLATEAU) on the full 714 held-out set.")
    ap.add_argument("--surf_points", type=int, default=8192,
                    help="query points per selected sample. Cheap: cost is dominated by `decode`, "
                         "not by point count (#76 measured 1k -> 32k as only 0.172s -> 0.313s), so "
                         "be generous here and stingy with --surf_bs instead.")
    ap.add_argument("--surf_bs", type=int, default=1,
                    help="how many batch elements get the surface term. This is the real cost knob: "
                         "one element at 8192 points adds ~67%% to a 305ms step.")
    ap.add_argument("--surf_t_center", type=float, default=0.55,
                    help="put the surface term where INFERENCE runs. Samples nearest this fraction "
                         "of the schedule get the term; 0.0 reproduces the original "
                         "lowest-t selection. ⚠️ That original choice was a measured mistake: it fed "
                         "the term only near-clean latents, where the model has nothing to change, "
                         "so it learned to reproduce its input instead of carving it (#80 run, "
                         "vs-input 0.993). Default matches the s~0.5-0.6 projection band.")
    ap.add_argument("--surf_t_max", type=float, default=0.85,
                    help="hard ceiling: samples above this fraction of the schedule are skipped "
                         "entirely. The main protection is not this cutoff but the alpha_bar "
                         "WEIGHTING applied to the term -- x0_hat = (x_t - sqrt(1-a)*eps_hat)/"
                         "sqrt(a), and 1/sqrt(a) amplifies eps-error without bound as t rises, which "
                         "is the mechanism #60 measured diverging into rubble. Weighting by alpha_bar "
                         "makes an unreliable x0_hat contribute ~nothing on its own. ⚠️ Do NOT set "
                         "this at or below --pair_t_min: the two are complementary, and a hard mask "
                         "there silently disables the term on every pair step.")
    ap.add_argument("--resume", default=None,
                    help="continue from a checkpoint: restores weights, optimizer state and step "
                         "count, so --steps is the TOTAL target, not the additional count")
    ap.add_argument("--archive_every", type=int, default=0,
                    help="also keep a step-tagged copy this often (0 = only the rolling file). #75 "
                         "found the quality curve is non-monotonic, so keeping the trajectory is "
                         "what separates a temporary dip from a real decline.")
    ap.add_argument("--seed", type=int, default=None,
                    help="#92 experiment seed. Replays batch order, pair/plain selection, timesteps, "
                         "noise and CFG dropout; surface queries use an isolated stream. If omitted, "
                         "a random seed is chosen and recorded in the checkpoint.")
    ap.add_argument("--out", default="logs_building/vecset_v1")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--save_every", type=int, default=2000)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = ExperimentRng(args.seed, dev)
    args.seed = rng.seed
    torch.manual_seed(rng.model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rng.model_seed)
    print(f"[rng] experiment seed {rng.seed}  independent train/surface streams", flush=True)

    ds = LatentSet(args.latents, blockout_path=args.blockouts)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True,
                                     num_workers=2, persistent_workers=True, generator=rng.loader)
    C, FPRES = ds.latent_shape[-1], ds.fp.shape[-1]
    print(f"[data] {len(ds)} train latents  tokens={ds.latent_shape[0]} ch={C}  fp={FPRES}  "
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
    region_w, surf_norm = None, None
    if args.surf_weight_by == "region":
        region_w = torch.tensor([float(x) for x in args.surf_region_weights.split(",")],
                                device=dev, dtype=torch.float32)
        n_reg = int(ds.r.max()) + 1
        if region_w.numel() < n_reg:
            raise SystemExit(f"[surf] --surf_region_weights has {region_w.numel()} entries but the "
                             f"corpus has {n_reg} regions; a short list would IndexError mid-run")
        # corpus mean of the weight actually seen, so normalisation does not depend on --surf_bs
        surf_norm = float(region_w[torch.from_numpy(ds.r.astype(np.int64)).to(dev)].mean())
        print(f"[surf] per-REGION weights {region_w.tolist()[:n_reg]}  corpus mean {surf_norm:.4f}",
              flush=True)
    elif args.surf_weight_by == "solidity":
        surf_norm = float(np.mean(ds.solidity))
        print(f"[surf] per-SOLIDITY weighting  corpus mean {surf_norm:.4f}  "
              f"(only {(ds.solidity < 0.9).mean()*100:.1f}% of the corpus is below 0.9, so there is "
              f"little to redistribute across)", flush=True)
    if args.surf_weight_by and args.surf_weight <= 0:
        raise SystemExit("[surf] --surf_weight_by has no effect with --surf_weight 0")
    if args.surf_weight > 0:
        from models.shape_codec import DoraCodec
        from scripts.foundations.dora_roundtrip_probe import load_dora
        codec = DoraCodec(load_dora(dev), differentiable=True).freeze()
        # `surf_t_center` is in this line because it is the variable the #80 band-fix run turns on,
        # and a config knob that decides an experiment should not be invisible in its own log.
        print(f"[surf] decoded-surface loss ON  w={args.surf_weight}  "
              f"{args.surf_points} pts x {args.surf_bs} sample(s)  "
              f"t centred {args.surf_t_center} (max {args.surf_t_max})"
              + (f"  [PER-SAMPLE weighted by {args.surf_weight_by.upper()}]"
                 if args.surf_weight_by else ""), flush=True)

    step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        net.load_state_dict(ck["model"])
        step = int(ck["step"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
            print(f"[resume] from {args.resume} at step {step} (optimizer state restored)", flush=True)
        else:
            # AdamW's moments are gone, so the first few hundred steps re-estimate them and the loss
            # bumps before settling. Harmless, but say so rather than let it look like a regression.
            print(f"[resume] from {args.resume} at step {step} -- ⚠️ no optimizer state in that "
                  f"checkpoint, AdamW moments restart and the loss will bump briefly", flush=True)
        if step >= args.steps:
            raise SystemExit(f"[resume] checkpoint is already at step {step}; --steps must exceed it")

    t0, hist, surf_hist, step0 = time.time(), [], [], step
    while step < args.steps:
        for z, zb, fp, h, r, sol in dl:
            if step >= args.steps:
                break
            z, zb = z.to(dev), zb.to(dev)
            fp, h, r, sol = fp.to(dev), h.to(dev), r.to(dev), sol.to(dev)
            use_pair = ds.has_blockouts and rng.pair_random() < args.pair_frac
            lo = int(args.pair_t_min * args.timesteps) if use_pair else 0
            t = rng.randint(lo, args.timesteps, (z.shape[0],))
            a = ac[t].view(-1, 1, 1)
            noise = rng.randn(tuple(z.shape), dtype=z.dtype)

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
            drop = rng.rand((z.shape[0],)) < args.cfg_drop
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
                # Select the samples nearest `surf_t_center` on the schedule, and weight by
                # alpha_bar.
                #
                # ⚠️ Selecting the LOWEST-t samples (the first version) was a design error, and the
                # #80 run measured its consequence: the term only ever saw near-clean latents, where
                # the model already has almost nothing to change, so it learned "reproduce the input"
                # rather than "carve it" (vs-input rose to 0.993). The edits that matter happen where
                # inference runs, s ~ 0.5-0.6, so that is where the supervision has to sit.
                #
                # This does not give up #60's protection. At t/T ~ 0.55 the cosine schedule puts
                # alpha_bar ~ 0.42, so the 1/sqrt(alpha_bar) amplification of eps-error is ~1.55 --
                # mild. The blow-up #60 measured is at t -> T, where alpha_bar -> 0, and both the
                # alpha_bar weighting and `surf_t_max` still exclude that.
                frac = t.float() / args.timesteps
                order = torch.argsort((frac - args.surf_t_center).abs())[:args.surf_bs]
                sel = order[frac[order] <= args.surf_t_max]
                if sel.numel():
                    asel = a[sel]
                    x0 = (zt[sel] - (1 - asel).sqrt() * pred[sel]) / asel.sqrt()
                    pts = rng.surface_rand((sel.numel(), args.surf_points, 3)) * 2 - 1
                    with torch.no_grad():
                        tgt = codec.query(z[sel] * ds.sd + ds.mu, pts)
                    got = codec.query(x0 * ds.sd + ds.mu, pts)
                    w_t = asel.reshape(sel.numel(), *([1] * (got.dim() - 1)))
                    wsel = (sol[sel] if args.surf_weight_by == "solidity" else
                            region_w[r[sel]] if args.surf_weight_by == "region" else None)
                    surf, surf_unweighted = surface_term(got, tgt, w_t, wsel, surf_norm)
                    loss = loss + args.surf_weight * surf
                    surf_val = float(surf_unweighted.detach())
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
                # divide by steps taken THIS process, not the absolute counter -- after a --resume
                # the latter makes the rate look ~4x better than it is
                print(f"  step {step:6d}/{args.steps}  loss {w:.4f}{sfx}  "
                      f"{(time.time()-t0)/max(step-step0,1):.2f}s/step", flush=True)
            if step % args.save_every == 0 or step == args.steps:
                blob = {"model": net.state_dict(), "step": step, "args": vars(args),
                        "opt": opt.state_dict(),       # so --resume continues rather than restarts
                        "latent_mu": ds.mu, "latent_sd": ds.sd,
                        "latent_channels": C, "footprint_res": FPRES}
                torch.save(blob, out / "vecset_denoiser.pth")
                # Keep periodic step-tagged copies. #75 found the quality curve is NON-MONOTONIC --
                # it fell for three consecutive checkpoints and then rose past all of them -- so a
                # single overwritten file makes the trajectory unrecoverable after the fact, and the
                # trajectory is the only thing that distinguishes a dip from a decline.
                if args.archive_every and step % args.archive_every == 0:
                    torch.save(blob, out / f"vecset_denoiser_step{step}.pth")

    json.dump({"steps": step, "final_loss": float(np.mean(hist[-100:])),
               "first_loss": float(np.mean(hist[:100])), "params_M": n_par / 1e6,
               "n_train": len(ds)}, open(out / "train.json", "w"), indent=2)
    print(f"[done] {step} steps  loss {np.mean(hist[:100]):.4f} -> {np.mean(hist[-100:]):.4f}  "
          f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
