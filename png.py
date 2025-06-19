import os, h5py, numpy as np
from PIL import Image

# ───── Paths ────────────────────────────────────────────────────────────────
DATA_ROOT  = "/mnt/c/Users/Public/generativetowns/sdfusion/SDFusion/data/BuildingNet_dataset_v0_1"
RES64_DIR  = os.path.join(DATA_ROOT, "resolution_64")
SPLITS_DIR = os.path.join(DATA_ROOT, "splits")
OUT_ROOT   = os.path.join(DATA_ROOT, "footprints_png")

H5_NAME    = "ori_sample_grid.h5"   # inside each model folder
FP_KEY     = "footprint"            # HDF5 dataset name

# ───── Process each split ───────────────────────────────────────────────────
for split in ("train", "val", "test"):
    # read model IDs from the split file
    split_file = os.path.join(SPLITS_DIR, f"{split}_split.txt")
    with open(split_file, "r") as f:
        model_ids = [l.strip() for l in f if l.strip()]
    
    # make output dir
    out_dir = os.path.join(OUT_ROOT, split)
    os.makedirs(out_dir, exist_ok=True)
    
    for mid in model_ids:
        h5_path = os.path.join(RES64_DIR, mid, H5_NAME)
        if not os.path.isfile(h5_path):
            print(f"[WARN] missing H5 for {mid}: {h5_path}")
            continue
        
        # load footprint mask
        with h5py.File(h5_path, "r") as hf:
            fp = hf[FP_KEY][()]        # shape = (1, H, W), values 0/1
        
        # squeeze to (H, W), convert to 0–255
        fp = (fp.squeeze(0).astype(np.uint8) * 255)
        
        # save PNG
        png_path = os.path.join(out_dir, f"{mid}.png")
        Image.fromarray(fp, mode="L").save(png_path, optimize=True)
    
    print(f"→ {split}: wrote {len(os.listdir(out_dir))} PNGs into {out_dir}")
