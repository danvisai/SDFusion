#!/usr/bin/env bash
# Stage 3a — conditional SDF latent diffusion on BuildingNet + recipe-aug.
#
# Conditioning: (footprint, class, height, style); see configs/stage3a_sdf_diffusion.yaml.
# Three-phase schedule per the Stage 3 plan (proud-waddling-cocoa.md):
#   - Day 1: 30k steps on recipe-aug only       (set --recipe_aug_ratio 1.0)
#   - Day 2-3: 100k steps on mixed 0.7 recipe   (this default)
#   - Day 4-5: 20k steps on real only           (set --recipe_aug_ratio 0.0)
# We launch the *full* run as a single 150k-step training with --recipe_aug_ratio 0.7;
# the safety net at 50k steps (per plan) is to manually drop ratio if the contact
# sheets are speckle.
#
# Prerequisites (verify before launching):
#   1. Phase 1a — VQVAE-v2 checkpoint trained. Set VQ_CKPT below to the
#      promoted checkpoint (or fall back to v1 if v2 is still training).
#   2. Phase 1b.1 — outputs/stage3_metadata/asset_dimensions.csv exists.
#   3. Phase 1b.3 — data/recipe_augmentation_v1/{8 styles}.h5 exist.
#   4. scripts/compute_scale_factor.py rerun on the v2 latents; configs/stage3a_sdf_diffusion.yaml
#      `scale_factor` updated.
#
# Run with:
#   env -u LD_PRELOAD -u LD_LIBRARY_PATH bash launchers/train_stage3a.sh
RED='\033[0;31m'
NC='\033[0m'
DATE_WITH_TIME=`date "+%Y-%m-%dT%H-%M-%S"`

logs_dir='logs_building'

gpu_ids=0
if [ ${#gpu_ids} -gt 1 ]; then
    NGPU=4
    PORT=11768
fi

lr=1e-4
batch_size=16

model="stage3a"
df_cfg="configs/stage3a_sdf_diffusion.yaml"

# VQ-VAE checkpoint. Using v1 ckpt+config (matched).
# Diagnostic on 2026-05-24 showed v2's aux losses (surface_band + footprint BCE)
# hurt iso=0 IoU (0.33 vs v1's 0.47) and produced fragmented meshes despite
# better L1. v1 round-trips with clean iso=0 surfaces.
# See: outputs/vqvae_ab_diagnostic_t03/visual_sheet.png + per_asset.csv.
# v2 ckpts retained at Saved_Checkpoint/vqvae_v2_20260523/ for later retraining.
VQ_CKPT="${VQ_CKPT:-logs_building/2025-05-19T19-58-28-vqvae-building-all-res64-LR1e-4-T0.2-release/ckpt/vqvae_steps-latest.pth}"
VQ_CFG="${VQ_CFG:-configs/vqvae_bnet.yaml}"

dataset_mode='stage3a'
dataroot="data"
res=64
cat='all'
trunc_thres=0.2   # match v1's training distribution
recipe_aug_ratio=0.7

total_iters=150000
cosine_total_steps=150000
warmup_steps=1000

display_freq=2000
print_freq=50
save_latest_freq=2000
save_steps_freq=10000

note="recipe-aug-0.7-aux-cosine"
name="${DATE_WITH_TIME}-stage3a-${dataset_mode}-${cat}-res${res}-LR${lr}-${note}"

debug=0
if [ $debug = 1 ]; then
    echo -e "${RED}Debugging!${NC}"
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
                --model ${model} --df_cfg ${df_cfg} --vq_cfg ${VQ_CFG} --vq_ckpt ${VQ_CKPT} \
                --dataset_mode ${dataset_mode} --cat ${cat} --res ${res} --trunc_thres ${trunc_thres} \
                --recipe_aug_ratio ${recipe_aug_ratio} \
                --display_freq ${display_freq} --print_freq ${print_freq} \
                --save_latest_freq ${save_latest_freq} --save_steps_freq ${save_steps_freq} \
                --total_iters ${total_iters} \
                --augment --use_adamw_cosine \
                --warmup_steps ${warmup_steps} --cosine_total_steps ${cosine_total_steps} \
                --debug ${debug}"

if [ ! -z "$dataroot" ]; then
    cmd="${cmd} --dataroot ${dataroot}"
fi

multi_gpu=0
if [ ${#gpu_ids} -gt 1 ]; then
    multi_gpu=1
fi

echo "[*] Training is starting on `hostname`, GPU#: ${gpu_ids}, logs_dir: ${logs_dir}"
echo "[*] VQ_CKPT=${VQ_CKPT}"
echo "[*] VQ_CFG=${VQ_CFG}"

if [ $multi_gpu = 1 ]; then
    cmd="-m torch.distributed.launch --nproc_per_node=${NGPU} --master_port=${PORT} ${cmd}"
fi

echo "[*] Training with command:"
echo "CUDA_VISIBLE_DEVICES=${gpu_ids} python ${cmd}"

CUDA_VISIBLE_DEVICES=${gpu_ids} python ${cmd}
