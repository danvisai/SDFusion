"""Train the part-LAYOUT PLANNER (detail-plan step 3): massing+class -> part-bbox sequence.

Data: outputs/part_layouts_full/part_instances.npz (28k instances / 1849 buildings) +
BuildingNet 64^3 SDFs (RAM-preloaded, ~1 GB fp16). Small model — coexists with prior training.

Eval: held-out type/box losses + sampled per-class type-count histograms vs GT (the composer's
sanity check, now with WHERE) + a GT-vs-sampled bbox montage.

Run:
  env -u LD_PRELOAD -u LD_LIBRARY_PATH PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
    ./sdfusion/bin/python scripts/foundations/train_layout_planner.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.networks.part_layout_planner import (PartLayoutPlanner, PartLayoutPlannerV2,
                                                 RAW_TYPES, TYPE_NAMES, N_TYPES)

NPZ = REPO / "outputs/part_layouts_full/part_instances.npz"
RES64 = REPO / "data/BuildingNet_dataset_v0_1/resolution_64"
OUT = REPO / "outputs/part_layout_planner"
CLASSES = ["COMMERCIAL", "PUBLIC", "RELIGIOUS", "RESIDENTIAL"]
MAX_LEN = 40
MAX_PER_TYPE = 16          # cap windows etc. (keep largest instances)


CACHE = Path("/tmp/layout_dataset_cache.npz")   # node-local: Lustre stalls kill re-reads


def build_dataset(device):
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        print(f"[data] cache hit {CACHE} ({len(z['names'])} buildings)")
        return (torch.from_numpy(z["sdf"]).unsqueeze(1), torch.from_numpy(z["cls"]),
                torch.from_numpy(z["T"]), torch.from_numpy(z["Bx"]),
                torch.from_numpy(z["lens"]), [str(x) for x in z["names"]])
    d = np.load(NPZ, allow_pickle=True)
    rows, names = d["rows"], [str(x) for x in d["names"]]
    raw2t = {r: i for i, r in enumerate(RAW_TYPES)}
    sdfs, cls_ids, seq_types, seq_boxes, lens, kept_names = [], [], [], [], [], []
    t0 = time.time()
    for bi, aid in enumerate(names):
        h5p = RES64 / aid / "ori_sample_grid.h5"
        if not h5p.exists():
            continue
        r = rows[rows[:, 0] == bi]
        if len(r) == 0:
            continue
        # per-type cap, biggest first
        keep = []
        for t in np.unique(r[:, 1]):
            rt = r[r[:, 1] == t]
            vol = (rt[:, 5] * rt[:, 6] * rt[:, 7])
            keep.append(rt[np.argsort(-vol)][:MAX_PER_TYPE])
        r = np.concatenate(keep)
        # canonical order: type, then y, then z, x
        r = r[np.lexsort((r[:, 2], r[:, 4], r[:, 3], r[:, 1]))][:MAX_LEN]
        try:
            with h5py.File(h5p, "r") as h:
                sdf = h["pc_sdf_sample"][:].reshape(64, 64, 64).astype(np.float16)
        except Exception:
            continue
        types = np.array([raw2t[int(x)] for x in r[:, 1]], np.int64)
        boxes = r[:, 2:8].astype(np.float32)
        sdfs.append(sdf)
        m = re.match(r"^([A-Z]+)", aid)
        cls_ids.append(CLASSES.index(m.group(1)) if m and m.group(1) in CLASSES else 3)
        seq_types.append(types); seq_boxes.append(boxes); lens.append(len(types))
        kept_names.append(aid)
        if bi % 300 == 0:
            print(f"load {bi}/{len(names)} ({time.time()-t0:.0f}s)", flush=True)
    n = len(sdfs)
    print(f"[data] {n} buildings, mean seq len {np.mean(lens):.1f}")
    T = np.zeros((n, MAX_LEN), np.int64)
    Bx = np.zeros((n, MAX_LEN, 6), np.float32)
    for i, (t, b) in enumerate(zip(seq_types, seq_boxes)):
        T[i, :len(t)] = t
        Bx[i, :len(t)] = b
    sdf_arr = np.stack(sdfs)
    np.savez(CACHE, sdf=sdf_arr, cls=np.asarray(cls_ids, np.int64), T=T, Bx=Bx,
             lens=np.asarray(lens, np.int64), names=np.asarray(kept_names))
    print(f"[data] cached -> {CACHE}")
    return (torch.from_numpy(sdf_arr).unsqueeze(1), torch.tensor(cls_ids),
            torch.from_numpy(T), torch.from_numpy(Bx), torch.tensor(lens), kept_names)


def main():
    import argparse
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("iters", type=int, nargs="?", default=800)  # val min ~400-600 for v1
    ap.add_argument("--v2", action="store_true", help="spatial cross-attention planner")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.v2:
        OUT = OUT.parent / "part_layout_planner_v2"
    OUT.mkdir(parents=True, exist_ok=True)
    sdf, cls_id, T, Bx, lens, names = build_dataset(device)
    n = len(names)
    rs = np.random.RandomState(0).permutation(n)
    val_idx, tr_idx = rs[:max(n // 10, 1)], rs[max(n // 10, 1):]
    iters = args.iters
    model = (PartLayoutPlannerV2 if args.v2 else PartLayoutPlanner)(max_len=MAX_LEN).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)
    bs = 32
    t0 = time.time()
    logf = open(OUT / "train_log.txt", "w")
    for it in range(iters):
        idx = tr_idx[np.random.randint(0, len(tr_idx), bs)]
        x = sdf[idx].float().clamp(-0.2, 0.2).to(device)
        tl, bl = model(x, cls_id[idx].to(device), T[idx].to(device),
                       Bx[idx].to(device), lens[idx].to(device))
        loss = tl + 4.0 * bl
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if it % 200 == 0:
            model.eval()
            with torch.no_grad():
                vi = val_idx[:128]
                vt, vb = model(sdf[vi].float().clamp(-0.2, 0.2).to(device),
                               cls_id[vi].to(device), T[vi].to(device),
                               Bx[vi].to(device), lens[vi].to(device))
            model.train()
            msg = (f"it {it:5d}  type:{float(tl):.3f} box:{float(bl):.3f}  "
                   f"VAL type:{float(vt):.3f} box:{float(vb):.3f}  {time.time()-t0:.0f}s")
            print(msg, flush=True); logf.write(msg + "\n"); logf.flush()
    torch.save({"model": model.state_dict(), "max_len": MAX_LEN}, OUT / "planner.pth")

    # ---- eval: per-class sampled type counts vs GT + montage --------------------------
    model.eval()
    stats = {}
    for ci, cname in enumerate(CLASSES):
        vi = [i for i in val_idx if int(cls_id[i]) == ci][:24]
        if not vi:
            continue
        x = sdf[vi].float().clamp(-0.2, 0.2).to(device)
        outs = model.sample(x, cls_id[vi].to(device))
        cnt_s = np.zeros(N_TYPES); cnt_g = np.zeros(N_TYPES)
        for k, i in enumerate(vi):
            for t, _ in outs[k]:
                cnt_s[t] += 1
            for t in T[i, :lens[i]].numpy():
                cnt_g[t] += 1
        stats[cname] = {"sampled": (cnt_s / len(vi)).round(2).tolist(),
                        "gt": (cnt_g / len(vi)).round(2).tolist()}
        print(f"[{cname}] " + " ".join(
            f"{TYPE_NAMES[t]}:{cnt_s[t]/len(vi):.1f}/{cnt_g[t]/len(vi):.1f}"
            for t in range(N_TYPES) if cnt_g[t] > 0 or cnt_s[t] > 0))
    json.dump({"types": TYPE_NAMES, "per_class_sampled_vs_gt": stats},
              open(OUT / "metrics.json", "w"), indent=1)

    # montage: GT vs sampled boxes for 6 val buildings
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    def draw_boxes(ax, items, color):
        for t, b in items:
            c, e = np.asarray(b[:3]), np.asarray(b[3:])
            lo, hi = c - e, c + e
            for a_, b_ in [(0,1),(0,2),(1,3),(2,3),(4,5),(4,6),(5,7),(6,7),(0,4),(1,5),(2,6),(3,7)]:
                pts = [[lo[0],lo[1],lo[2]],[hi[0],lo[1],lo[2]],[lo[0],hi[1],lo[2]],[hi[0],hi[1],lo[2]],
                       [lo[0],lo[1],hi[2]],[hi[0],lo[1],hi[2]],[lo[0],hi[1],hi[2]],[hi[0],hi[1],hi[2]]]
                seg = np.array([pts[a_], pts[b_]])[:, [0, 2, 1]]
                ax.add_collection3d(Line3DCollection([seg], colors=color, linewidths=0.7))
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        ax.set_axis_off(); ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=16, azim=-55)
    vi = val_idx[:6]
    fig, axes = plt.subplots(2, len(vi), figsize=(2.6 * len(vi), 5.4),
                             subplot_kw={"projection": "3d"})
    outs = model.sample(sdf[vi].float().clamp(-0.2, 0.2).to(device), cls_id[vi].to(device))
    for k, i in enumerate(vi):
        gt = [(int(T[i, j]), Bx[i, j].numpy()) for j in range(int(lens[i]))]
        draw_boxes(axes[0, k], gt, "tab:green"); axes[0, k].set_title(f"GT {names[i][:18]}", fontsize=6)
        draw_boxes(axes[1, k], outs[k], "tab:red"); axes[1, k].set_title("sampled", fontsize=6)
    plt.tight_layout(); plt.savefig(OUT / "layout_montage.png", dpi=110)
    print("[done]", OUT)


if __name__ == "__main__":
    main()
