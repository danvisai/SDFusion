"""
Carve a held-out validation split out of BuildingNet's train_split.txt.

The trainer currently only uses train/test phases (datasets/base_dataset.py:85-90),
and val_split.txt does not exist. This script:

  - Stratified-samples ~val_frac of the training ids per top-level category
    (RESIDENTIAL / RELIGIOUS / COMMERCIAL / MILITARY / PUBLIC) using a fixed seed
    so the split is reproducible.
  - Writes data/.../splits/val_split.txt
  - Rewrites data/.../splits/train_split.txt without the val ids
    (the original is preserved as train_split.txt.bak on first run)
  - Moves the corresponding PNGs from footprints_png/train/ to footprints_png/val/
  - Prints before/after stats

Run from repo root:
    python scripts/make_val_split.py --dry_run     # preview, write nothing
    python scripts/make_val_split.py               # commit
"""
import argparse
import os
import random
import re
import shutil
import sys
from collections import defaultdict


CATEGORY_RE = re.compile(r"^([A-Z]+)")


def category_of(model_id):
    m = CATEGORY_RE.match(model_id)
    return m.group(1) if m else "UNKNOWN"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/BuildingNet_dataset_v0_1")
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    splits_dir = os.path.join(args.data_root, "splits")
    train_p = os.path.join(splits_dir, "train_split.txt")
    val_p = os.path.join(splits_dir, "val_split.txt")
    train_bak_p = os.path.join(splits_dir, "train_split.txt.bak")

    if not os.path.exists(train_p):
        sys.exit(f"missing {train_p}")
    if os.path.exists(val_p) and not args.dry_run:
        sys.exit(f"{val_p} already exists; refusing to overwrite. delete it first if you really want to redo this.")

    with open(train_p) as f:
        all_ids = [ln.strip() for ln in f if ln.strip()]

    by_cat = defaultdict(list)
    for mid in all_ids:
        by_cat[category_of(mid)].append(mid)

    rng = random.Random(args.seed)
    val_ids = []
    for cat in sorted(by_cat):
        ids = sorted(by_cat[cat])
        n_val = max(1, round(len(ids) * args.val_frac)) if len(ids) >= 4 else 0
        rng.shuffle(ids)
        val_ids.extend(ids[:n_val])
    val_ids = sorted(val_ids)
    val_set = set(val_ids)
    new_train = [mid for mid in all_ids if mid not in val_set]

    print(f"[*] train ids in {train_p}: {len(all_ids)}")
    print(f"[*] target val_frac        : {args.val_frac:.3f}")
    print(f"[*] new val   count        : {len(val_ids)}")
    print(f"[*] new train count        : {len(new_train)}")
    print()
    print("  by-category:")
    for cat in sorted(by_cat):
        v = sum(1 for x in val_ids if category_of(x) == cat)
        t = sum(1 for x in new_train if category_of(x) == cat)
        print(f"    {cat:12s}  train -> {len(by_cat[cat]):4d}   "
              f"=> new train {t:4d}  /  new val {v:3d}")
    print()

    # Footprint PNG move plan
    png_train = os.path.join(args.data_root, "footprints_png", "train")
    png_val = os.path.join(args.data_root, "footprints_png", "val")

    moves = []
    missing = []
    for mid in val_ids:
        src = os.path.join(png_train, f"{mid}.png")
        dst = os.path.join(png_val, f"{mid}.png")
        if os.path.exists(src):
            moves.append((src, dst))
        else:
            missing.append(mid)

    print(f"[*] PNGs to move train -> val : {len(moves)}")
    if missing:
        print(f"[!] {len(missing)} val ids have no PNG (will skip move): "
              f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
    print()

    if args.dry_run:
        print("[*] DRY RUN — nothing written. First 5 val ids:", val_ids[:5])
        return

    # Commit:
    if not os.path.exists(train_bak_p):
        shutil.copyfile(train_p, train_bak_p)
        print(f"[+] backed up original train_split to {train_bak_p}")
    with open(val_p, "w") as f:
        f.write("\n".join(val_ids) + "\n")
    with open(train_p, "w") as f:
        f.write("\n".join(new_train) + "\n")
    print(f"[+] wrote {val_p}  ({len(val_ids)} ids)")
    print(f"[+] rewrote {train_p}  ({len(new_train)} ids)")

    os.makedirs(png_val, exist_ok=True)
    n_moved = 0
    for src, dst in moves:
        shutil.move(src, dst)
        n_moved += 1
    print(f"[+] moved {n_moved} PNGs to {png_val}")


if __name__ == "__main__":
    main()
