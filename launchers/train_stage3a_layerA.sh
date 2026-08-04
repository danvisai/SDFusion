#!/usr/bin/env bash
# Stage 3a LAYER-A context-conditioned finetune.
#
# Warm-starts the 6k cross-cultural prior and adds the Layer-A context channels (known_body +
# edit_mask + primitive) so an added mass integrates coherently with the existing body. The
# context is derived self-supervised from the clean target on-the-fly (no new data). The new
# input-conv channels are zero-init (identity start) and learn from here. dataset_mode=hybrid,
# real.h5 (NL+DE+JP) + recipes.
#
#   env -u LD_PRELOAD -u LD_LIBRARY_PATH bash launchers/train_stage3a_layerA.sh
export PYTHONUNBUFFERED=1
logs_dir='logs_building'
gpu_ids=0
lr=5e-5                                    # a touch higher than the breadth finetune: the zero-init context conv must learn
batch_size=16

model="stage3a"
df_cfg="configs/stage3a_sdf_diffusion.yaml"
CKPT="logs_building/continue-stage3a-xcultural-warmstart-ft-final/ckpt/stage3a_steps-latest.pth"
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

name="stage3a-layerA-context"

echo "[*] LAYER-A finetune on `hostname` GPU#${gpu_ids}"
echo "[*] warm-start: ${CKPT}  + --use_context  lr=${lr} steps=${total_iters}"

PY="${PY:-sdfusion/bin/python}"
CUDA_VISIBLE_DEVICES=${gpu_ids} ${PY} train.py \
    --name ${name} --logs_dir ${logs_dir} --gpu_ids ${gpu_ids} --lr ${lr} --batch_size ${batch_size} \
    --model ${model} --df_cfg ${df_cfg} --vq_cfg ${VQ_CFG} --vq_ckpt ${VQ_CKPT} --ckpt ${CKPT} \
    --dataset_mode ${dataset_mode} --bag3d_h5 ${bag3d_h5} --use_context \
    --cat ${cat} --res ${res} --trunc_thres ${trunc_thres} \
    --display_freq ${display_freq} --print_freq ${print_freq} \
    --save_latest_freq ${save_latest_freq} --save_steps_freq ${save_steps_freq} \
    --total_iters ${total_iters} \
    --augment --use_adamw_cosine --warmup_steps ${warmup_steps} --cosine_total_steps ${cosine_total_steps} \
    --dataroot ${dataroot} --debug 0
