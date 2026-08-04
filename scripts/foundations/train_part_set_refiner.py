"""Train the part-SET refiner (detail-plan step 4) + the ADD-A-PART re-cohere test.

Data: same instance dataset as the planner (reuses train_layout_planner.build_dataset).
Set encoding per slot: [type one-hot ±1 | box6 | validity ±1]; empty slots all-(-1) boxes 0.

Eval (the step-4 acceptance test): take a held-out GT set, INJECT a duplicate window at a
random off-surface position, set-SDEdit at strength 0.3 -> does the refined set (a) keep the
real parts (poses stable), (b) kill or relocate the junk part (validity/pose), measured +
visualized in a montage.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "foundations"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from train_layout_planner import build_dataset, MAX_LEN
from models.networks.part_set_refiner import PartSetRefiner, SLOTS, PART_DIM, N_TYPES

OUT = REPO / "outputs/part_set_refiner"


def encode_sets(T, Bx, lens):
    n = T.shape[0]
    x = torch.full((n, SLOTS, PART_DIM), 0.0)
    x[:, :, :N_TYPES] = -1.0
    x[:, :, -1] = -1.0
    for i in range(n):
        L = int(lens[i])
        x[i, :L, :N_TYPES] = -1.0
        x[i, torch.arange(L), T[i, :L]] = 1.0
        x[i, :L, N_TYPES:N_TYPES + 6] = Bx[i, :L]
        x[i, :L, -1] = 1.0
    return x


def decode_set(x):
    """-> list of (type, box6) for slots with validity > 0."""
    out = []
    for s in range(x.shape[0]):
        if x[s, -1] > 0:
            t = int(x[s, :N_TYPES].argmax())
            out.append((t, x[s, N_TYPES:N_TYPES + 6].cpu().numpy()))
    return out


def draw(ax, items, color, title=""):
    for t, b in items:
        c, e = np.asarray(b[:3]), np.abs(np.asarray(b[3:])) + 1e-3
        lo, hi = c - e, c + e
        pts = [[lo[0],lo[1],lo[2]],[hi[0],lo[1],lo[2]],[lo[0],hi[1],lo[2]],[hi[0],hi[1],lo[2]],
               [lo[0],lo[1],hi[2]],[hi[0],lo[1],hi[2]],[lo[0],hi[1],hi[2]],[hi[0],hi[1],hi[2]]]
        for a_, b_ in [(0,1),(0,2),(1,3),(2,3),(4,5),(4,6),(5,7),(6,7),(0,4),(1,5),(2,6),(3,7)]:
            seg = np.array([pts[a_], pts[b_]])[:, [0, 2, 1]]
            ax.add_collection3d(Line3DCollection([seg], colors=color, linewidths=0.7))
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_axis_off(); ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=16, azim=-55)
    ax.set_title(title, fontsize=6)


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    sdf, cls_id, T, Bx, lens, names = build_dataset(device)
    X = encode_sets(T, Bx, lens)
    n = len(names)
    rs = np.random.RandomState(0).permutation(n)
    val_idx, tr_idx = rs[:n // 10], rs[n // 10:]
    model = PartSetRefiner(device=device)
    opt = torch.optim.AdamW(model.net.parameters(), lr=2e-4, weight_decay=1e-4)
    t0 = time.time()
    def corrupt(xb, lens_b):
        """inject 1-2 junk parts into ~half the batch; CLEAN target marks them empty."""
        xc = xb.clone()
        for bi in range(xb.shape[0]):
            if np.random.rand() < 0.5:
                continue
            L = int(lens_b[bi])
            for _ in range(np.random.randint(1, 3)):
                s_ = min(L, SLOTS - 1)
                junk = torch.zeros(PART_DIM); junk[:N_TYPES] = -1.0
                junk[np.random.randint(0, N_TYPES)] = 1.0
                junk[N_TYPES:N_TYPES + 3] = torch.rand(3) * 1.8 - 0.9
                junk[N_TYPES + 3:N_TYPES + 6] = torch.rand(3) * 0.1 + 0.03
                junk[-1] = 1.0
                xc[bi, s_] = junk
                L = min(L + 1, SLOTS)
        return xc

    for it in range(iters):
        idx = tr_idx[np.random.randint(0, len(tr_idx), 32)]
        xb = X[idx]
        xc = corrupt(xb, lens[idx])
        loss = model.loss(xb.to(device), sdf[idx].float().clamp(-0.2, 0.2).to(device),
                          x_corrupt=xc.to(device))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if it % 250 == 0:
            with torch.no_grad():
                vi = val_idx[:128]
                vl = model.loss(X[vi].to(device), sdf[vi].float().clamp(-0.2, 0.2).to(device))
            print(f"it {it:5d}  eps:{float(loss):.4f}  VAL eps:{float(vl):.4f}  {time.time()-t0:.0f}s",
                  flush=True)
    torch.save({"net": model.net.state_dict(), "T": model.T}, OUT / "refiner.pth")

    # ---- ADD-A-PART re-cohere test ----------------------------------------------------
    model.net.eval()
    vi = val_idx[:6]
    fig, axes = plt.subplots(3, len(vi), figsize=(2.6 * len(vi), 8),
                             subplot_kw={"projection": "3d"})
    kept, junk_killed, moved = [], [], []
    for k, i in enumerate(vi):
        x0 = X[i].clone()
        L = int(lens[i])
        junk = torch.zeros(PART_DIM); junk[:N_TYPES] = -1.0
        junk[0] = 1.0                                          # a "window"
        junk[N_TYPES:N_TYPES + 6] = torch.tensor([0.9, 0.9, 0.9, .05, .07, .05])  # floating corner
        slot = min(L, SLOTS - 1)
        x_pert = x0.clone(); x_pert[slot] = junk
        ref = model.refine(x_pert[None].to(device),
                           sdf[i][None].float().clamp(-0.2, 0.2).to(device),
                           strength=0.2, steps=12)[0].cpu()
        gt_items, pert_items, ref_items = decode_set(x0), decode_set(x_pert), decode_set(ref)
        kept.append(len(ref_items) / max(len(gt_items), 1))
        junk_killed.append(1.0 if ref[slot, -1] <= 0 else 0.0)
        moved.append(float((ref[slot, N_TYPES:N_TYPES + 3] -
                            x_pert[slot, N_TYPES:N_TYPES + 3]).norm()))
        draw(axes[0, k], gt_items, "tab:green", f"GT ({len(gt_items)})")
        draw(axes[1, k], pert_items, "tab:orange", "+ junk part @corner")
        draw(axes[2, k], ref_items, "tab:red", f"refined ({len(ref_items)}) "
             f"{'junk KILLED' if junk_killed[-1] else f'junk moved {moved[-1]:.2f}'}")
    plt.tight_layout(); plt.savefig(OUT / "recohere_montage.png", dpi=110)
    print(f"[add-a-part] parts kept {np.mean(kept):.2f}x of GT · junk killed {np.mean(junk_killed):.0%} "
          f"· junk moved {np.mean(moved):.2f} (cube units)")
    print("[done]", OUT)


if __name__ == "__main__":
    main()
