#!/usr/bin/env bash
# v2: bf16 throughout (fixes the fp16 dtype-boundary collapse from v1) +
# 500-step LR warmup. Filtered splits: 1091 train / 55 val / 132 test.
set -e

NAME="footprint2view-15k-bf16"

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
--mixed_precision bf16
--warmup_steps 500
--gradient_checkpointing
"

echo "[v2-cn] python:  ${PYTHON}"
echo "[v2-cn] command: ${PYTHON} ${cmd}"

env -u LD_PRELOAD -u LD_LIBRARY_PATH \
    HF_HOME="$HF_ROOT" \
    HY3DGEN_MODELS="$HF_ROOT/hy3dgen" \
    CUDA_VISIBLE_DEVICES=0 \
    ${PYTHON} ${cmd}
