"""Train / finetune the CoherentPartRefiner (coherent-add-primitive, spec §5).

No-image conditioning (§2): massing SDF + class + added-primitive MARKER. Edit-pair supervision
(on-the-fly) + relational COHERENCE losses (band-rhythm/size/attach). Two phases:
  pretrain  --source buildingnet   (1847 bldgs, real BuildingNet SDFs)
  finetune  --source lod3 --init <ckpt>   (55 clean LoD3 bldgs, 99% row-aligned; box-prism SDF)
Leaves the deployed PartSetRefiner/refiner.pth untouched; saves coherent_refiner.pth.

  env -u LD_PRELOAD -u LD_LIBRARY_PATH PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
    /tmp/sdfusion_venv/bin/python scripts/foundations/train_coherent_refiner.py \
      --source buildingnet --iters 2500 --cohw 0.3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts" / "foundations"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_layout_planner import build_dataset
from make_part_edit_pairs import make_pair, encode, draw, SLOTS, PART_DIM
from models.networks.part_set_refiner import CoherentPartRefiner, N_TYPES
from models.networks.part_layout_planner import RAW_TYPES

RAW2IDX = {r: i for i, r in enumerate(RAW_TYPES)}
LOD3 = REPO / "data/lod3_tum/lod3_part_instances.npz"
OUT = REPO / "outputs/part_set_refiner"


def box_sdf_64(lo, hi):
    """64^3 signed-distance to a box (axes z,y,x; coords [-1,1]); negative inside."""
    g = np.linspace(-1, 1, 64, dtype=np.float32)
    Z, Y, X = np.meshgrid(g, g, g, indexing="ij")
    c = (np.asarray(lo) + np.asarray(hi)) / 2; h = (np.asarray(hi) - np.asarray(lo)) / 2 + 1e-3
    q = np.stack([np.abs(X - c[0]) - h[0], np.abs(Y - c[1]) - h[1], np.abs(Z - c[2]) - h[2]], 0)
    out = np.sqrt(np.clip(q, 0, None) ** 2).sum(0) ** 0.5
    return (out + np.minimum(q.max(0), 0)).astype(np.float32)


def parts_of(T, Bx, L):
    return [(int(T[j]), np.asarray(Bx[j], np.float32)) for j in range(int(L))]


def load_source(source, dev):
    if source == "buildingnet":
        sdf, cls, T, Bx, lens, names = build_dataset(dev)
        parts = [parts_of(T[i], Bx[i], lens[i]) for i in range(len(names))]
        return sdf.float(), cls, parts, names
    d = np.load(LOD3, allow_pickle=True); r = d["rows"]
    pb = {}
    for row in r:
        b, t = int(row[0]), int(row[1])
        if t in RAW2IDX:
            pb.setdefault(b, []).append((RAW2IDX[t], row[2:8].astype(np.float32)))
    names = [b for b in sorted(pb) if len(pb[b]) >= 4]
    parts, sdfs = [], []
    for b in names:
        ps = pb[b]; parts.append(ps)
        cen = np.array([p[1][:3] for p in ps]); sz = np.array([p[1][3:6] for p in ps])
        sdfs.append(box_sdf_64((cen - sz).min(0), (cen + sz).max(0)))
    sdf = torch.from_numpy(np.stack(sdfs)).unsqueeze(1).float()
    cls = torch.full((len(names),), 3, dtype=torch.long)          # unlabeled -> RESIDENTIAL
    return sdf, cls, parts, names


def batch(idx, parts, rng):
    xc, x0, mk = [], [], []
    for i in idx:
        a, b, m, _ = make_pair(parts[i], rng)
        xc.append(a); x0.append(b); mk.append(m)
    f = lambda a: torch.from_numpy(np.stack(a)).float()
    return f(xc), f(x0), f(mk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="buildingnet", choices=["buildingnet", "lod3"])
    ap.add_argument("--iters", type=int, default=2500)
    ap.add_argument("--cohw", type=float, default=0.3)
    ap.add_argument("--init", default="")
    ap.add_argument("--lr", type=float, default=2e-4)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    sdf, cls, parts, names = load_source(a.source, dev)
    n = len(names); rng = np.random.default_rng(0)
    rs = np.random.RandomState(0).permutation(n); val, tr = rs[:max(n // 10, 4)], rs[max(n // 10, 4):]
    model = CoherentPartRefiner(device=dev)
    if a.init:
        model.net.load_state_dict(torch.load(a.init, map_location=dev)["net"])
        print(f"[init] finetuning from {a.init}")
    opt = torch.optim.AdamW(model.net.parameters(), lr=a.lr, weight_decay=1e-4)
    t0 = time.time()
    print(f"[{a.source}] {n} buildings | iters {a.iters} cohw {a.cohw}")
    for it in range(a.iters):
        idx = tr[np.random.randint(0, len(tr), 32)]
        xc, x0, mk = batch(idx, parts, rng)
        cw = a.cohw * min(1.0, max(0.0, (it - 300) / 500))       # ramp coherence in after warmup
        loss = model.loss(x0.to(dev), xc.to(dev), mk.to(dev),
                          sdf[idx].clamp(-0.2, 0.2).to(dev), cls[idx].to(dev), cohw=cw)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if it % 250 == 0:
            print(f"it {it:5d}  loss {float(loss):.4f}  cohw {cw:.2f}  {time.time()-t0:.0f}s", flush=True)
    torch.save({"net": model.net.state_dict(), "T": model.T, "source": a.source}, OUT / "coherent_refiner.pth")
    print(f"[save] {OUT/'coherent_refiner.pth'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
