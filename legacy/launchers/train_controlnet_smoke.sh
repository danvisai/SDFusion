#!/usr/bin/env bash
# Smoke train of the footprint -> building-render ControlNet.
# Goal: confirm pipeline runs end-to-end (load SD1.5, init ControlNet, take a
# few hundred steps, sample a (footprint | gen | gt) grid, save weights).
#
# Run from repo root:
#     ./launchers/train_controlnet_smoke.sh
set -e

NAME="footprint2view-smoke"

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
--max_steps 600
--save_every 600
--sample_every 200
--print_every 25
--num_workers 4
--mixed_precision fp16
--gradient_checkpointing
"

echo "[smoke-cn] python:  ${PYTHON}"
echo "[smoke-cn] command: ${PYTHON} ${cmd}"

env -u LD_PRELOAD -u LD_LIBRARY_PATH \
    HF_HOME="$HF_ROOT" \
    HY3DGEN_MODELS="$HF_ROOT/hy3dgen" \
    CUDA_VISIBLE_DEVICES=0 \
    ${PYTHON} ${cmd}
