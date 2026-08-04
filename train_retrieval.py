from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.buildingnet_retrieval_dataset import (
    BuildingNetRetrievalDataset,
    build_label_maps,
    load_split_ids,
)
from models.networks.retrieval import FootprintEmbedNet


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    b = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    logits = (z @ z.T) / temperature
    logits.fill_diagonal_(-1e9)
    targets = torch.cat([
        torch.arange(b, 2 * b, device=z.device),
        torch.arange(0, b, device=z.device),
    ])
    return nn.functional.cross_entropy(logits, targets)


@torch.no_grad()
def eval_retrieval(model, loader, device):
    model.eval()
    embs, class_ids, top_ids, model_ids = [], [], [], []
    for batch in loader:
        fp = batch["fp"].to(device)
        cid = batch["class_id"].to(device)
        emb, logits = model(fp, cid)
        embs.append(emb.cpu())
        class_ids.append(batch["class_id"])
        top_ids.append(batch["top_id"])
        model_ids.extend(batch["id"])
    emb = torch.cat(embs, dim=0)
    cls = torch.cat(class_ids, dim=0)
    top = torch.cat(top_ids, dim=0)
    sim = emb @ emb.T
    sim.fill_diagonal_(-1e9)
    nn_idx = sim.argmax(dim=1)
    same_subtype = (cls[nn_idx] == cls).float().mean().item()
    same_top = (top[nn_idx] == top).float().mean().item()
    return {"nn_same_subtype": same_subtype, "nn_same_top": same_top}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/BuildingNet_dataset_v0_1")
    ap.add_argument("--out_dir", default="Logs_GT/retrieval_footprint")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--embedding_dim", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--ce_weight", type=float, default=0.25)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--max_val_samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=111)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_ids = []
    for phase in ("train", "val", "test"):
        all_ids.extend(load_split_ids(args.data_root, phase))
    subtype_to_idx, top_to_idx = build_label_maps(all_ids)
    (out_dir / "label_maps.json").write_text(json.dumps({
        "subtype_to_idx": subtype_to_idx,
        "top_to_idx": top_to_idx,
    }, indent=2, sort_keys=True))

    train_ds = BuildingNetRetrievalDataset(
        args.data_root, "train", subtype_to_idx, top_to_idx,
        augment=True, max_samples=args.max_train_samples,
    )
    val_ds = BuildingNetRetrievalDataset(
        args.data_root, "val", subtype_to_idx, top_to_idx,
        augment=False, max_samples=args.max_val_samples,
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    device = torch.device(args.device)
    model = FootprintEmbedNet(len(subtype_to_idx), embedding_dim=args.embedding_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_same_top = -1.0
    log_path = out_dir / "loss_log.txt"
    with log_path.open("w") as log:
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            pbar = tqdm(train_dl, desc=f"epoch {epoch}/{args.epochs}")
            for batch in pbar:
                fp_a = batch["fp_a"].to(device)
                fp_b = batch["fp_b"].to(device)
                cid = batch["class_id"].to(device)
                z1, logits1 = model(fp_a, cid)
                z2, logits2 = model(fp_b, cid)
                loss_con = nt_xent_loss(z1, z2, temperature=args.temperature)
                loss_ce = 0.5 * (
                    nn.functional.cross_entropy(logits1, cid)
                    + nn.functional.cross_entropy(logits2, cid)
                )
                loss = loss_con + args.ce_weight * loss_ce

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                losses.append(float(loss.detach().cpu()))
                pbar.set_postfix(loss=f"{np.mean(losses):.4f}")

            metrics = eval_retrieval(model, val_dl, device)
            mean_loss = float(np.mean(losses)) if losses else 0.0
            line = (
                f"epoch {epoch:03d} loss {mean_loss:.6f} "
                f"val_nn_same_top {metrics['nn_same_top']:.4f} "
                f"val_nn_same_subtype {metrics['nn_same_subtype']:.4f}"
            )
            print(line)
            log.write(line + "\n")
            log.flush()

            ckpt = {
                "model": model.state_dict(),
                "args": vars(args),
                "subtype_to_idx": subtype_to_idx,
                "top_to_idx": top_to_idx,
                "epoch": epoch,
                "metrics": metrics,
            }
            torch.save(ckpt, out_dir / "ckpt_latest.pth")
            if metrics["nn_same_top"] > best_same_top:
                best_same_top = metrics["nn_same_top"]
                torch.save(ckpt, out_dir / "ckpt_best.pth")

    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
