from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_npz(path: Path):
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", default="data/BuildingNet_dataset_v0_1/retrieval_index")
    ap.add_argument("--phase", default="val", choices=["train", "val", "test"])
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=8)
    return ap.parse_args()


def main():
    args = parse_args()
    index_dir = Path(args.index_dir)
    train = load_npz(index_dir / "train_embeddings.npz")
    query = load_npz(index_dir / f"{args.phase}_embeddings.npz")

    train_emb = train["embeddings"]
    query_emb = query["embeddings"]
    sims = query_emb @ train_emb.T

    for i in range(min(args.limit, len(query["ids"]))):
        same_top = train["top_ids"] == query["top_ids"][i]
        masked = sims[i].copy()
        masked[~same_top] = -1e9
        nn = np.argsort(-masked)[: args.top_k]
        print(f"\nquery {query['ids'][i]} top={query['top_ids'][i]}")
        for rank, j in enumerate(nn, 1):
            print(f"  {rank}. {train['ids'][j]} sim={masked[j]:.4f}")


if __name__ == "__main__":
    main()
