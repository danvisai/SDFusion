#!/usr/bin/env bash
# Stage 3a PILOT — cross-cultural massing (NL + DE + JP) with the region/culture token.
#
# Tests the data-variety hypothesis for Layer 1: a ~947M-param model plateaued on ~1.8k
# BuildingNet + ~11.8k NL recipes; here we add ~12k DE (NRW) + ~12k JP (PLATEAU) real LoD2
# massing and condition on a region token so cultures stay distinct. dataset_mode=hybrid mixes
# the concatenated real corpus (data/real_massing_v1/real.h5, via Bag3dDataset) with the
# procedural recipes by bag_ratio (default 0.5). use_region=True comes from the df_cfg.
#
#   env -u LD_PRELOAD -u LD_LIBRARY_PATH bash launchers/train_stage3a_xcultural.sh
DATE_WITH_TIME=`date "+%Y-%m-%dT%H-%M-%S"`
logs_dir='logs_building'
gpu_ids=0
lr=1e-4
batch_size=16

model="stage3a"
df_cfg="configs/stage3a_sdf_diffusion.yaml"          # use_region: True lives here
VQ_CKPT="${VQ_CKPT:-logs_building/2025-05-19T19-58-28-vqvae-building-all-res64-LR1e-4-T0.2-release/ckpt/vqvae_steps-latest.pth}"
VQ_CFG="${VQ_CFG:-configs/vqvae_bnet.yaml}"           # v1 (matches scale_factor in df_cfg)

dataset_mode='hybrid'
bag3d_h5="data/real_massing_v1/real.h5"              # NL+DE+JP concatenated, carries source_id
dataroot="data"
res=64
cat='all'
trunc_thres=0.2

# PILOT length — long enough to read sample diversity, short enough to turn around (~hrs on A100).
total_iters=30000
cosine_total_steps=30000
warmup_steps=1000
display_freq=2000
print_freq=50
save_latest_freq=2000
save_steps_freq=5000

name="stage3a-hybrid-xcultural-region-pilot"   # fixed (not timestamped) for deterministic monitoring

echo "[*] Pilot training on `hostname`, GPU#${gpu_ids}"
echo "[*] bag3d_h5=${bag3d_h5}  df_cfg=${df_cfg} (use_region)"
echo "[*] VQ_CKPT=${VQ_CKPT}  VQ_CFG=${VQ_CFG}"

PY="${PY:-sdfusion/bin/python}"
CUDA_VISIBLE_DEVICES=${gpu_ids} ${PY} train.py \
    --name ${name} --logs_dir ${logs_dir} --gpu_ids ${gpu_ids} --lr ${lr} --batch_size ${batch_size} \
    --model ${model} --df_cfg ${df_cfg} --vq_cfg ${VQ_CFG} --vq_ckpt ${VQ_CKPT} \
    --dataset_mode ${dataset_mode} --bag3d_h5 ${bag3d_h5} \
    --cat ${cat} --res ${res} --trunc_thres ${trunc_thres} \
    --display_freq ${display_freq} --print_freq ${print_freq} \
    --save_latest_freq ${save_latest_freq} --save_steps_freq ${save_steps_freq} \
    --total_iters ${total_iters} \
    --augment --use_adamw_cosine --warmup_steps ${warmup_steps} --cosine_total_steps ${cosine_total_steps} \
    --dataroot ${dataroot} --debug 0
