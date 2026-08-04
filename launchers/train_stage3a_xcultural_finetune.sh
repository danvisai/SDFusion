#!/usr/bin/env bash
# WARM-START finetune of the DEPLOYED massing prior on the broadened cross-cultural corpus.
#
# This is NOT a new model and NOT from-scratch: we load the prior the sculptor actually uses
# (logs_building/2026-06-08T11-50-42-stage3a-hybrid-clean, 20k, era/floors) and continue training
# it on data/real_massing_v1/real.h5 (NL 11776 + DE 12000 + JP 12000) so the EXISTING SDEdit sculpt
# output gets broader / less blobby. Gentle LR + short horizon (massing converges fast). use_region
# OFF, use_extra_cond ON (from df_cfg) so the ckpt loads strict (global_proj in=368).
#
#   env -u LD_PRELOAD -u LD_LIBRARY_PATH bash launchers/train_stage3a_xcultural_finetune.sh
export PYTHONUNBUFFERED=1                 # stream loss lines to the log live (no block-buffering)
logs_dir='logs_building'
gpu_ids=0
lr=3e-5                                   # finetune LR (vs 1e-4 from-scratch) — adapt, don't forget
batch_size=16

model="stage3a"
df_cfg="configs/stage3a_sdf_diffusion.yaml"        # use_extra_cond:True, use_region:False
CKPT="logs_building/2026-06-08T11-50-42-stage3a-hybrid-clean/ckpt/stage3a_steps-latest.pth"
VQ_CKPT="${VQ_CKPT:-logs_building/2025-05-19T19-58-28-vqvae-building-all-res64-LR1e-4-T0.2-release/ckpt/vqvae_steps-latest.pth}"
VQ_CFG="${VQ_CFG:-configs/vqvae_bnet.yaml}"          # load_ckpt overrides vqvae with the ckpt's clean VQVAE

dataset_mode='hybrid'
bag3d_h5="data/real_massing_v1/real.h5"
dataroot="data"; res=64; cat='all'; trunc_thres=0.2

total_iters=6000
cosine_total_steps=6000
warmup_steps=200
display_freq=1000
print_freq=50
save_latest_freq=1000
save_steps_freq=1000

name="stage3a-xcultural-warmstart-ft"

echo "[*] WARM-START finetune on `hostname` GPU#${gpu_ids}"
echo "[*] from ckpt: ${CKPT}"
echo "[*] corpus: ${bag3d_h5} (NL+DE+JP)  lr=${lr}  steps=${total_iters}"

PY="${PY:-sdfusion/bin/python}"
CUDA_VISIBLE_DEVICES=${gpu_ids} ${PY} train.py \
    --name ${name} --logs_dir ${logs_dir} --gpu_ids ${gpu_ids} --lr ${lr} --batch_size ${batch_size} \
    --model ${model} --df_cfg ${df_cfg} --vq_cfg ${VQ_CFG} --vq_ckpt ${VQ_CKPT} --ckpt ${CKPT} \
    --dataset_mode ${dataset_mode} --bag3d_h5 ${bag3d_h5} \
    --cat ${cat} --res ${res} --trunc_thres ${trunc_thres} \
    --display_freq ${display_freq} --print_freq ${print_freq} \
    --save_latest_freq ${save_latest_freq} --save_steps_freq ${save_steps_freq} \
    --total_iters ${total_iters} \
    --augment --use_adamw_cosine --warmup_steps ${warmup_steps} --cosine_total_steps ${cosine_total_steps} \
    --dataroot ${dataroot} --debug 0
