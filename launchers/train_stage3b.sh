#!/usr/bin/env bash
# Stage 3b — SDF -> Gaussian voxel-slot lifter.
#
# Trained on (real BuildingNet SDF, voxelized v2 Gaussians) pairs. The recipe-aug
# corpus is NOT used here — we have no v2 Gaussian targets for synthetic SDFs.
# 60k steps at batch 16 typically suffices; the network is feed-forward and small.
#
# Prerequisites:
#   1. Phase 1c — data/BuildingNet_dataset_v0_1/gsplat_voxelized_32k8/*.npz exists.
#   2. Phase 1b.1 — outputs/stage3_metadata/asset_dimensions.csv exists.
#   3. Logs_GT/retrieval_footprint_full/ckpt_best.pth exists (or Stage 3b silently
#      falls back to a random-init FootprintEmbedNet — fine, just less informed).
#
# Run with:
#   env -u LD_PRELOAD -u LD_LIBRARY_PATH bash launchers/train_stage3b.sh
DATE_WITH_TIME=`date "+%Y-%m-%dT%H-%M-%S"`

logs_dir='logs_building'
gpu_ids=0

lr=2e-4
batch_size=16

model="stage3b"
df_cfg="configs/stage3b_lifter.yaml"
VQ_CFG="${VQ_CFG:-configs/vqvae_bnet.yaml}"   # unused but train.py copies it

dataset_mode='stage3b'
dataroot="data"
res=64
cat='all'
trunc_thres=0.3

total_iters=60000
cosine_total_steps=60000
warmup_steps=500

display_freq=4000
print_freq=50
save_latest_freq=2000
save_steps_freq=10000

note="lifter-v1"
name="${DATE_WITH_TIME}-stage3b-${dataset_mode}-${cat}-res${res}-LR${lr}-${note}"

debug=0
if [ $debug = 1 ]; then
    batch_size=4
    total_iters=200
    cosine_total_steps=200
    warmup_steps=20
    save_steps_freq=50
    save_latest_freq=20
    display_freq=20
    print_freq=5
    name="DEBUG-${name}"
fi

cmd="train.py --name ${name} --logs_dir ${logs_dir} --gpu_ids ${gpu_ids} --lr ${lr} --batch_size ${batch_size} \
                --model ${model} --df_cfg ${df_cfg} --vq_cfg ${VQ_CFG} \
                --dataset_mode ${dataset_mode} --cat ${cat} --res ${res} --trunc_thres ${trunc_thres} \
                --display_freq ${display_freq} --print_freq ${print_freq} \
                --save_latest_freq ${save_latest_freq} --save_steps_freq ${save_steps_freq} \
                --total_iters ${total_iters} \
                --use_adamw_cosine --warmup_steps ${warmup_steps} --cosine_total_steps ${cosine_total_steps} \
                --debug ${debug}"

if [ ! -z "$dataroot" ]; then
    cmd="${cmd} --dataroot ${dataroot}"
fi

echo "[*] Stage 3b training on `hostname`, GPU#: ${gpu_ids}"
echo "[*] CUDA_VISIBLE_DEVICES=${gpu_ids} python ${cmd}"

CUDA_VISIBLE_DEVICES=${gpu_ids} python ${cmd}
