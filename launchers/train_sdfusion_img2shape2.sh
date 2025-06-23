#!/usr/bin/env bash

# Launcher for conditional SDFusionImageFPShapeModel (image + footprint)

RED='\033\[0;31m'
NC='\033\[0m' # No Color
DATE_WITH_TIME=`date "+%Y-%m-%dT%H-%M-%S"`

logs_dir="logs"
if [ `hostname` = az007 ] || [ `hostname` = az006 ]; then
    logs_dir="logs_home"
fi

### GPUs

gpu_ids=0          # single-GPU
echo "Using GPU(s): ${gpu_ids}"

# For multi-GPU (e.g. gpu\_ids=0,1,2,3):

# if \[ \${#gpu\_ids} -gt 1 ]; then

# NGPU=4

# PORT=11768

# fi

### Hyperparameters

lr=1e-5
batch_size=2
backend='gloo'  # 'gloo' for CPU, 'nccl' for multi-GPUs
#######################

### Model and config

model='sdfusion_model_img2shape'
df_cfg='configs/sdfusion-img2shape.yaml'   # your new diffusion config
vq_model='vqvae'
vq_cfg='configs/vqvae_bnet.yaml'
vq_ckpt="/mnt/c/Users/Public/generativetowns/sdfusion/SDFusion/logs_building/2025-05-19T19-58-28-vqvae-building-all-res64-LR1e-4-T0.2-release/ckpt/vqvae_steps-latest.pth"
vq_dset='bnet'
vq_cat='all'

# Optional: resume from a checkpoint

epoch_ckpt=""  # e.g. "--ckpt /path/to/df\_steps-latest.pth"
########################

### Dataset

max_dataset_size=10000000
dataset_mode='building'
dataroot='data'
res=64
cat='all'
trunc_thres=0.2
################

### Logging and debug

display_freq=500
print_freq=50
total_iters=500000
save_steps_freq=2000
########################

debug=0
if [ $debug -eq 1 ]; then
echo -e "${RED}DEBUG MODE ON ${NC}"
batch_size=2
max_dataset_size=200
display_freq=10
print_freq=5
total_iters=20000
save_steps_freq=50
name="DEBUG-${DATE_WITH_TIME}-${model}-${dataset_mode}-LR${lr}"
else
name="${DATE_WITH_TIME}-${model}-${dataset_mode}-LR${lr}"
fi

# Assemble command

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
--res ${res}
--cat ${cat}
--trunc_thres ${trunc_thres}
--display_freq ${display_freq}
--print_freq ${print_freq}
--total_iters ${total_iters}
--save_steps_freq ${save_steps_freq}
--debug ${debug}
--dataroot ${dataroot}
${epoch_ckpt}"

echo "[INFO] Training command:"
echo "CUDA_VISIBLE_DEVICES=${gpu_ids} python ${cmd}"

echo "\[INFO] Starting training..."
CUDA_VISIBLE_DEVICES=${gpu_ids} python ${cmd}
