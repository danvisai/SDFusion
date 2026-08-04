#!/usr/bin/env bash
# Full 15K-step ControlNet training: footprint -> building render.
#
# Run from repo root:
#     ./launchers/train_controlnet_full.sh
# or in background:
#     nohup ./launchers/train_controlnet_full.sh > /tmp/cn_full.log 2>&1 &
set -e

NAME="footprint2view-15k"

PYTHON="${PYTHON:-$(pwd)/sdfusion/bin/python}"
HF_ROOT="$(pwd)/external/hf_cache"
mkdir -p "$HF_ROOT"

cmd="train_controlnet.py
--name ${NAME}
--logs_dir Logs_GT
--data_root data/BuildingNet_dataset_v0_1
--sd_model stable-diffusion-v1-5/stable-diffusion-v1-5
--resolution 512
--batch_size 4
--lr 1e-5
--max_steps 15000
--save_every 1500
--sample_every 500
--print_every 50
--num_workers 4
--mixed_precision fp16
--gradient_checkpointing
"

echo "[full-cn] python:  ${PYTHON}"
echo "[full-cn] command: ${PYTHON} ${cmd}"

env -u LD_PRELOAD -u LD_LIBRARY_PATH \
    HF_HOME="$HF_ROOT" \
    HY3DGEN_MODELS="$HF_ROOT/hy3dgen" \
    CUDA_VISIBLE_DEVICES=0 \
    ${PYTHON} ${cmd}
