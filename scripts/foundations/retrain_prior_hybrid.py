"""Foundations task #4 — retrain the Stage3a prior on the CLEAN VQVAE + HYBRID conditioning.

New vs the old prior: (1) the gap#6-fixed clean VQVAE (vqvae_clean.pth) as the frozen latent space,
(2) hybrid corpus = real BAG (era/floors labels) + procedural (8 named styles), (3) CFG dropout
(p_uncond, gap#3) so guidance/autoguidance work, (4) EMA (gap#5). Self-contained loop (mirrors the
VQVAE finetune) so we don't fight the train.py harness; nThreads 0 (h5py+fork hangs).

Validate first with --total_iters 200, then launch the full run.
"""
from __future__ import annotations
import argparse, sys, time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from datasets.hybrid_dataset import HybridDataset
from models.stage3a_model import Stage3aModel

VQ_CLEAN = "logs_building/vqvae_clean_ft/vqvae_clean.pth"


def build_opt(args, ckpt_dir):
    # Phase-2 warm-start (map #34): fine-tune an existing checkpoint (weights only; load_ckpt
    # keeps a FRESH optimizer via warm_start) with the SDF-field smoothness regularizer on.
    finetune_ckpt = str(REPO / args.finetune_from) if args.finetune_from else None
    return SimpleNamespace(
        isTrain=True, device=args.device,
        df_cfg=str(REPO / "configs/stage3a_sdf_diffusion.yaml"),
        vq_cfg=str(REPO / "configs/vqvae_bnet.yaml"),
        vq_ckpt=str(REPO / args.vq_ckpt), ckpt=finetune_ckpt,
        warm_start=bool(finetune_ckpt),
        # SDF-field smoothness regularizer (gated; default OFF preserves the from-scratch recipe)
        use_smooth=bool(args.use_smooth), smooth_weight=args.smooth_weight,
        smooth_kind=args.smooth_kind, smooth_sigma=args.smooth_sigma, smooth_every=args.smooth_every,
        ddim_steps=50, debug="0", gpu_ids=[0] if args.device == "cuda" else [],
        ckpt_dir=str(ckpt_dir),
        lr=args.lr, warmup_steps=1000, cosine_total_steps=args.total_iters,
        # hybrid conditioning + training-quality fixes
        use_extra_cond=bool(args.use_extra_cond),
        use_region=bool(args.use_region), num_regions=4, region_emb_dim=16,
        p_uncond=args.p_uncond, use_ema=True, ema_decay=0.999,
        # REPA (training-gaps step 4): DINOv2 alignment, early-stopped. VERDICT NEGATIVE
        # 2026-06-10 (repa20k ≈ hybrid20k) — kept for reference, default OFF.
        use_repa=bool(args.repa), repa_weight=args.repa_weight,
        repa_stop_iter=int(args.repa_stop_frac * args.total_iters),
        # adaLN (gap #1 conditioning fix): cond vector into the time embedding
        use_adaln=bool(args.adaln),
        # dataset
        bag3d_h5=args.bag3d_h5, trunc_thres=0.2, augment=True, bag_ratio=args.bag_ratio,
        bag_labels=str(REPO / "data/bag3d_v1/bag_labels.npz"), seed=0,
        dataroot="data", recipe_aug_root="data/recipe_augmentation_v1",
        heights_csv="outputs/stage3_metadata/asset_dimensions.csv",
        recipe_aug_ratio=1.0, recipe_styles=None,
    )


@torch.no_grad()
def vqvae_recon_iou(model, x):
    z = model.vqvae(x, forward_no_quant=True, encode_only=True)
    r = model.vqvae.decode_no_quant(z)
    oa, ob = (x <= 0), (r <= 0); u = (oa | ob).sum().item()
    return (oa & ob).sum().item() / u if u else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--vq_ckpt", default=VQ_CLEAN)
    ap.add_argument("--bag3d_h5", default="/dev/shm/bag3d_fast.h5")
    ap.add_argument("--total_iters", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--p_uncond", type=float, default=0.1)
    ap.add_argument("--bag_ratio", type=float, default=0.5)
    ap.add_argument("--dataset", default="hybrid", choices=["hybrid", "bag3d"],
                    help="bag3d = LoD2-only (Bag3dDataset direct, no recipe corpus)")
    ap.add_argument("--use_region", type=int, default=0, help="1 = NL/DE/JP culture token")
    ap.add_argument("--use_extra_cond", type=int, default=1, help="0 = drop era/floors")
    ap.add_argument("--save_every", type=int, default=5000)
    ap.add_argument("--name", default=None)
    ap.add_argument("--repa", type=int, default=0, help="1 = REPA DINOv2 alignment (verdict: no gain)")
    ap.add_argument("--repa_weight", type=float, default=0.5)
    ap.add_argument("--repa_stop_frac", type=float, default=0.75,
                    help="stop alignment after this fraction of total iters (2505.16792)")
    ap.add_argument("--adaln", type=int, default=0, help="1 = adaLN cond-into-time-embedding (gap #1)")
    # Phase-2 surface-fidelity fine-tune (map #34)
    ap.add_argument("--finetune_from", default=None,
                    help="warm-start ckpt (repo-relative); fresh optimizer at a constant --lr")
    ap.add_argument("--use_smooth", type=int, default=0, help="1 = SDF-field smoothness regularizer")
    ap.add_argument("--smooth_weight", type=float, default=0.05)
    ap.add_argument("--smooth_kind", default="grad_tv", choices=["grad_tv", "eikonal"])
    ap.add_argument("--smooth_sigma", type=float, default=0.05)
    ap.add_argument("--smooth_every", type=int, default=1, help="apply the smoothness term every K steps")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    name = args.name or (f"{stamp}-stage3a-hybrid-clean"
                         + ("-repa" if args.repa else "") + ("-adaln" if args.adaln else "")
                         + ("-ftsmooth" if args.finetune_from and args.use_smooth else
                            "-ft" if args.finetune_from else ""))
    logdir = REPO / "logs_building" / name; ckpt_dir = logdir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] {logdir}")

    opt = build_opt(args, ckpt_dir)
    if args.dataset == "bag3d":
        from datasets.bag3d_dataset import Bag3dDataset
        ds = Bag3dDataset(); ds.initialize(opt, "train")   # LoD2-only, forwards region_id
    else:
        ds = HybridDataset(); ds.initialize(opt, "train")
    loader = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=0, drop_last=True)
    model = Stage3aModel(); model.initialize(opt)
    model.switch_train()
    if args.finetune_from:
        # Constant low LR: the self-contained loop never steps the LambdaLR scheduler, and a
        # warm-start smoothing fine-tune wants a gentle fixed LR (not the from-scratch cosine).
        for g in model.optimizer.param_groups:
            g["lr"] = args.lr
        print(f"[ft] warm-start from {args.finetune_from}; smooth={bool(args.use_smooth)} "
              f"({args.smooth_kind} w={args.smooth_weight}); constant lr={args.lr}", flush=True)

    logf = open(logdir / "loss_log.txt", "a")
    t0 = time.time(); it = 0
    while it < args.total_iters:
        for batch in loader:
            model.set_input(batch)
            model.optimize_parameters(it)
            if it % 50 == 0:
                e = model.get_current_errors()
                msg = (f"[{name}] it {it:6d}/{args.total_iters}  " +
                       "  ".join(f"{k}:{float(v):.4f}" for k, v in e.items()) +
                       f"  {time.time()-t0:.0f}s")
                print(msg, flush=True); logf.write(msg + "\n"); logf.flush()
            if it > 0 and it % args.save_every == 0:
                model.save(f"steps-{it}", it)
                model.save("steps-latest", it)
                model.switch_eval()
                iou = vqvae_recon_iou(model, model.x[:4])
                print(f"  [ckpt {it}] saved; new-VQVAE recon IoU on batch = {iou:.3f}", flush=True)
                model.switch_train()
            it += 1
            if it >= args.total_iters:
                break
    model.save("steps-latest", it)
    print(f"[done] {it} iters in {time.time()-t0:.0f}s -> {ckpt_dir}")
    logf.close()


if __name__ == "__main__":
    main()
