#!/usr/bin/env bash
# Stage 3a LAYER A+B: Layer-A context channels (known_body + edit_mask + primitive) PLUS a
# Layer-B element-type token (window/door/balcony/pilaster/bay), classified from the Layer-A
# primitive's shape (models/stage3a_model.py:_classify_element_type — the voxel-space analog of
# scripts/server/facade_grammar.py:classify_shape). Per the ablation methodology in
# docs/CONTEXT_CONDITIONED_SNAP_BUILD_SPEC_2026-06-30.md (train A, A+B, A+B+C; eval each), this
# is the A+B arm: does telling the model WHAT the added mass should become (right vocabulary) fix
# residual issues on top of A (no more blob/suppression)?
#
# Warm-starts the finished Layer-A ckpt (continue-stage3a-layerA-context, 8000 steps). The new
# element_type_emb + the expanded global_proj input columns are zero-init (identity start);
# load_ckpt skips optimizer restore (same as Layer-A: shapes changed). Self-supervised from
# clean targets, no new data. dataset_mode=hybrid, real.h5 (NL+DE+JP) + recipes.
#
#   env -u LD_PRELOAD -u LD_LIBRARY_PATH bash launchers/train_stage3a_layerAB.sh
export PYTHONUNBUFFERED=1
logs_dir='logs_building'
gpu_ids=0
lr=5e-5                                    # same recipe as Layer-A: new zero-init params must learn
batch_size=16

model="stage3a"
df_cfg="configs/stage3a_sdf_diffusion.yaml"
CKPT="logs_building/continue-stage3a-layerA-context/ckpt/stage3a_steps-latest.pth"
VQ_CKPT="${VQ_CKPT:-logs_building/2025-05-19T19-58-28-vqvae-building-all-res64-LR1e-4-T0.2-release/ckpt/vqvae_steps-latest.pth}"
VQ_CFG="${VQ_CFG:-configs/vqvae_bnet.yaml}"

dataset_mode='hybrid'
bag3d_h5="data/real_massing_v1/real.h5"
dataroot="data"; res=64; cat='all'; trunc_thres=0.2

total_iters=8000
cosine_total_steps=8000
warmup_steps=200
display_freq=4000
print_freq=50
save_latest_freq=4000
save_steps_freq=4000

name="stage3a-layerAB-context-elemtype"

echo "[*] LAYER A+B finetune on `hostname` GPU#${gpu_ids}"
echo "[*] warm-start: ${CKPT}  + --use_context --use_element_type  lr=${lr} steps=${total_iters}"

PY="${PY:-sdfusion/bin/python}"
CUDA_VISIBLE_DEVICES=${gpu_ids} ${PY} train.py \
    --name ${name} --logs_dir ${logs_dir} --gpu_ids ${gpu_ids} --lr ${lr} --batch_size ${batch_size} \
    --model ${model} --df_cfg ${df_cfg} --vq_cfg ${VQ_CFG} --vq_ckpt ${VQ_CKPT} --ckpt ${CKPT} \
    --dataset_mode ${dataset_mode} --bag3d_h5 ${bag3d_h5} --use_context --use_element_type \
    --cat ${cat} --res ${res} --trunc_thres ${trunc_thres} \
    --display_freq ${display_freq} --print_freq ${print_freq} \
    --save_latest_freq ${save_latest_freq} --save_steps_freq ${save_steps_freq} \
    --total_iters ${total_iters} \
    --augment --use_adamw_cosine --warmup_steps ${warmup_steps} --cosine_total_steps ${cosine_total_steps} \
    --dataroot ${dataroot} --debug 0
