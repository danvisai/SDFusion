"""Recreate /dev/shm/bag3d_fast.h5 (per-sample-chunked RAM copy of the 3D BAG corpus).

The on-disk data/bag3d_v1/bag3d.h5 has ~736-sample chunks -> random per-sample reads starve
the GPU (known gotcha); training reads the /dev/shm copy, which is EPHEMERAL (lost on node
reboot/purge). Sequential block copy + rechunk to (1,64,64,64).
"""
import sys
import time

import h5py

SRC = "data/bag3d_v1/bag3d.h5"
DST = "/dev/shm/bag3d_fast.h5"
BLK = 1024

with h5py.File(SRC, "r") as s, h5py.File(DST, "w") as d:
    n = s["sdf"].shape[0]
    d.create_dataset("sdf", shape=s["sdf"].shape, dtype=s["sdf"].dtype, chunks=(1, 64, 64, 64))
    d.create_dataset("footprint", shape=s["footprint"].shape, dtype=s["footprint"].dtype,
                     chunks=(1, 64, 64))
    for k in ("height_m", "style_id", "bag_id", "class_label"):
        d.create_dataset(k, data=s[k][:])
    t0 = time.time()
    for i in range(0, n, BLK):
        j = min(i + BLK, n)
        d["sdf"][i:j] = s["sdf"][i:j]
        d["footprint"][i:j] = s["footprint"][i:j]
        print(f"{j}/{n}  {time.time()-t0:.0f}s", flush=True)
print("done", DST)
