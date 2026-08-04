from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.buildingnet_retrieval_dataset import BuildingNetRetrievalDataset
from models.networks.retrieval import FootprintEmbedNet


@torch.no_grad()
def encode_phase(model, dataset, device, batch_size, num_workers):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    embeddings, class_ids, top_ids, ids = [], [], [], []
    model.eval()
    for batch in loader:
        fp = batch["fp"].to(device)
        cid = batch["class_id"].to(device)
        emb, _ = model(fp, cid)
        embeddings.append(emb.cpu().numpy())
        class_ids.append(batch["class_id"].numpy())
        top_ids.append(batch["top_id"].numpy())
        ids.extend(batch["id"])
    return {
        "ids": np.array(ids, dtype=object),
        "embeddings": np.concatenate(embeddings, axis=0).astype(np.float32),
        "class_ids": np.concatenate(class_ids, axis=0).astype(np.int64),
        "top_ids": np.concatenate(top_ids, axis=0).astype(np.int64),
    }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/BuildingNet_dataset_v0_1")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", default="data/BuildingNet_dataset_v0_1/retrieval_index")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_samples", type=int, default=0,
                    help="debug limit per phase; 0 means all")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    subtype_to_idx = ckpt["subtype_to_idx"]
    top_to_idx = ckpt["top_to_idx"]
    model_args = ckpt.get("args", {})
    model = FootprintEmbedNet(
        len(subtype_to_idx),
        embedding_dim=int(model_args.get("embedding_dim", 256)),
    )
    model.load_state_dict(ckpt["model"])
    device = torch.device(args.device)
    model.to(device)

    meta = {
        "ckpt": str(Path(args.ckpt).resolve()),
        "subtype_to_idx": subtype_to_idx,
        "top_to_idx": top_to_idx,
        "idx_to_subtype": {v: k for k, v in subtype_to_idx.items()},
        "idx_to_top": {v: k for k, v in top_to_idx.items()},
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True))

    for phase in ("train", "val", "test"):
        ds = BuildingNetRetrievalDataset(
            args.data_root, phase, subtype_to_idx, top_to_idx,
            augment=False, max_samples=args.max_samples,
        )
        data = encode_phase(model, ds, device, args.batch_size, args.num_workers)
        np.savez_compressed(out_dir / f"{phase}_embeddings.npz", **data)
        print(f"{phase}: {len(data['ids'])} embeddings -> {out_dir / f'{phase}_embeddings.npz'}")

    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
