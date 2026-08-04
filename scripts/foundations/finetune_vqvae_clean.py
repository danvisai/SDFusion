"""Foundations task #2 — finetune the VQVAE on CLEAN SDFs to fix gap #6.

v1 reconstructs real buildings well but catastrophically fails on large cube-filling boxes
(bench_vqvae_recon: box fill0.90 IoU 0.47, vRatio 1.56). Finetune v1 on a clean corpus that
INCLUDES that failure case: 50% real 3D BAG + 50% random analytic boxes (half-extents up to 0.92,
the regime v1 breaks on). SIMPLE losses only (L1 recon + codebook) — NO aux losses (those caused
the v2 failure, project_vqvae_v2_failure). Periodically reports box0.90 + BAG recon IoU.

Output: a new VQVAE ckpt (latent space shifts -> prior must be re-encoded/retrained, task #4).
Does NOT touch the running demo (which keeps using v1 until the prior is retrained).
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import h5py, numpy as np, torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from models.networks.vqvae_networks.network import VQVAE

TRUNC = 0.2
V1 = "logs_building/2025-05-19T19-58-28-vqvae-building-all-res64-LR1e-4-T0.2-release/ckpt/vqvae_steps-latest.pth"
_GRID = {}


def grid(dev, R=64):
    if dev not in _GRID:
        g = torch.linspace(-1, 1, R, device=dev)
        ZZ, YY, XX = torch.meshgrid(g, g, g, indexing="ij")
        _GRID[dev] = torch.stack([ZZ, YY, XX], 0)                 # (3=z,y,x, R,R,R)
    return _GRID[dev]


def boxes(he, dev):                                              # he: (B,3) half-extents (z,y,x)
    G = grid(dev)
    out = []
    for h in he:
        q = G.abs() - h.view(3, 1, 1, 1)
        d = torch.linalg.vector_norm(q.clamp(min=0), dim=0) + q.amax(0).clamp(max=0)
        out.append(d.clamp(-TRUNC, TRUNC))
    return torch.stack(out).unsqueeze(1)


def occ_iou(a, b):
    oa, ob = (a <= 0), (b <= 0); u = (oa | ob).sum().item()
    return (oa & ob).sum().item() / u if u else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--vq_ckpt", default=str(REPO / V1))
    ap.add_argument("--bag_h5", default="/dev/shm/bag3d_fast.h5")
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="logs_building/vqvae_clean_ft/vqvae_clean.pth")
    args = ap.parse_args()
    dev = args.device

    vq_conf = OmegaConf.load(REPO / "configs/vqvae_bnet.yaml").model.params
    vqvae = VQVAE(vq_conf.ddconfig, vq_conf.n_embed, vq_conf.embed_dim).to(dev)
    st = torch.load(args.vq_ckpt, map_location="cpu"); vqvae.load_state_dict(st.get("vqvae", st))
    print(f"[init] loaded v1 from {Path(args.vq_ckpt).name}")
    vqvae.train()
    opt = torch.optim.AdamW(vqvae.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=1e-4)

    h5p = args.bag_h5 if Path(args.bag_h5).exists() else str(REPO / "data/bag3d_v1/bag3d.h5")
    h5 = h5py.File(h5p, "r"); n_bag = h5["sdf"].shape[0]
    print(f"[data] BAG {n_bag} from {h5p} + random boxes (50/50)")

    # fixed eval probes
    box090 = boxes(torch.full((1, 3), 0.9, device=dev), dev)
    bag_eval = torch.from_numpy(h5["sdf"][7].astype(np.float32)).clamp(-TRUNC, TRUNC).view(1, 1, 64, 64, 64).to(dev)

    t0 = time.time()
    for it in range(1, args.iters + 1):
        nb = args.bs // 2
        idx = np.sort(np.random.choice(n_bag, nb, replace=False))
        bag = torch.from_numpy(h5["sdf"][idx].astype(np.float32)).clamp(-TRUNC, TRUNC).unsqueeze(1).to(dev)
        he = torch.empty(args.bs - nb, 3, device=dev).uniform_(0.4, 0.92)
        x = torch.cat([bag, boxes(he, dev)], 0)
        recon, qloss = vqvae(x, verbose=False)
        rec = (recon - x).abs().mean()
        loss = rec + qloss.mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

        if it % 500 == 0 or it == 1:
            vqvae.eval()
            with torch.no_grad():
                rb = vqvae.decode_no_quant(vqvae(box090, forward_no_quant=True, encode_only=True))
                rg = vqvae.decode_no_quant(vqvae(bag_eval, forward_no_quant=True, encode_only=True))
            vqvae.train()
            print(f"  it {it:5d}  loss {loss.item():.4f} rec {rec.item():.4f}  "
                  f"box0.90 IoU {occ_iou(box090, rb):.3f}  BAG IoU {occ_iou(bag_eval, rg):.3f}  "
                  f"{time.time()-t0:.0f}s")

    outp = REPO / args.out; outp.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"vqvae": vqvae.state_dict()}, outp)
    print(f"[saved] {outp}")
    h5.close()


if __name__ == "__main__":
    main()
