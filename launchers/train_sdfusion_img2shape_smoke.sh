#!/usr/bin/env bash
# Short smoke test for SDFusionImageFPShapeModel after F1/F2/F3 + footprint rewrite.
# Goal: verify (a) loss is logged (was empty for 100K+ steps before), (b) generated
# samples are not blank, (c) no NaN/inf on the rescaled latents.
#
# Run from repo root:
#     ./launchers/train_sdfusion_img2shape_smoke.sh
# or with the conda env explicitly:
#     env -u LD_PRELOAD -u LD_LIBRARY_PATH ./launchers/train_sdfusion_img2shape_smoke.sh
set -e

DATE_WITH_TIME=$(date "+%Y-%m-%dT%H-%M-%S")

logs_dir="Logs_GT"
gpu_ids=0

### hyperparameters ###
lr=1e-5
batch_size=16          # smaller than full-run 50, for fast iters
backend='gloo'

### model + ckpts ###
model='sdfusion_model_img2shape'
df_cfg='configs/sdfusion-img2shape.yaml'
vq_model='vqvae'
vq_cfg='configs/vqvae_bnet.yaml'
vq_ckpt="$(pwd)/logs_building/2025-05-19T19-58-28-vqvae-building-all-res64-LR1e-4-T0.2-release/ckpt/vqvae_steps-latest.pth"
vq_dset='bnet'
vq_cat='all'
ckpt=""                # IMPORTANT: do NOT resume — old weights are pre-fix

### dataset ###
max_dataset_size=10000000
dataset_mode='building'
dataroot='data'
res=64
cat='all'
trunc_thres=0.2

### smoke knobs ###
total_iters=2000       # ~8x as many real loss readings as we need
display_freq=250       # 8 visualization snapshots over the run
print_freq=25          # every 25 iters → ~80 loss log entries
save_steps_freq=1000

debug=0                # NOT debug=1; want the real dataset, just short
name="SMOKE-${DATE_WITH_TIME}-${model}-${dataset_mode}-LR${lr}"

cmd="train.py
--name ${name}
--logs_dir ${logs_dir}
--gpu_ids ${gpu_ids}
--lr ${lr}
--batch_size ${batch_size}
--max_dataset_size ${max_dataset_size}
--model ${model}
--df_cfg ${df_cfg}
--vq_model ${vq_model}
--vq_cfg ${vq_cfg}
--vq_ckpt ${vq_ckpt}
--vq_dset ${vq_dset}
--vq_cat ${vq_cat}
--dataset_mode ${dataset_mode}
--dataroot ${dataroot}
--res ${res}
--cat ${cat}
--trunc_thres ${trunc_thres}
--display_freq ${display_freq}
--print_freq ${print_freq}
--total_iters ${total_iters}
--save_steps_freq ${save_steps_freq}
--debug ${debug}
--backend ${backend}
--nThreads 4
"

PYTHON="${PYTHON:-$(pwd)/sdfusion/bin/python}"

echo "[smoke] python:  ${PYTHON}"
echo "[smoke] command: CUDA_VISIBLE_DEVICES=${gpu_ids} ${PYTHON} ${cmd}"

# Strip XALT preload + spack LD_LIBRARY_PATH that break PyTorch+CUDA on Gilbreth
env -u LD_PRELOAD -u LD_LIBRARY_PATH \
    CUDA_VISIBLE_DEVICES=${gpu_ids} ${PYTHON} ${cmd}
