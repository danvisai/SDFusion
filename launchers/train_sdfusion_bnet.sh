#!/usr/bin/env bash

RED='\033[0;31m'
NC='\033[0m' # No Color
DATE_WITH_TIME=$(date "+%Y-%m-%dT%H-%M-%S")

logs_dir='logs_home'

### set gpus ###
gpu_ids=0          # single‑GPU
# gpu_ids=0,1,2,3  # multi‑GPU

if [ ${#gpu_ids} -gt 1 ]; then
    NGPU=4
    PORT=11768
fi
################

### hyper params ###
lr=1e-4
batch_size=3
####################

### model stuff ###
model='sdfusion'
df_cfg='configs/sdfusion_bnet.yaml'        # if you have a BNet‑specific config, otherwise reuse snet one

vq_model="vqvae"
vq_cfg="configs/vqvae_bnet.yaml"           # your BuildingNet VQ‑VAE config
vq_ckpt="/mnt/c/Users/Public/generativetowns/sdfusion/SDFusion/logs_building/2025-05-19T19-58-28-vqvae-building-all-res64-LR1e-4-T0.2-release/ckpt/vqvae_steps-latest.pth"    # your BNet VQ‑VAE checkpoint
vq_dset='bnet'
vq_cat='all'

ckpt="/mnt/c/Users/Public/generativetowns/sdfusion/SDFusion/logs_home/continue-2025-05-24T20-34-56-sdfusion-building-all-LR1e-4-bnet/ckpt/df_steps-latest.pth"

#C:\Users\Public\generativetowns\sdfusion\SDFusion\saved_ckpt\sdfusion-bnet-all.pth
cmd="${cmd} --ckpt ${ckpt}"
echo "continuing from ckpt=${ckpt}"
####################

### dataset stuff ###
max_dataset_size=10000000
dataset_mode='building'    # <-- this must match your BuildingNetDataset registration
dataroot="data"
res=64
cat='all'
trunc_thres=0.2
#####################

### display & log ###
display_freq=500
print_freq=25
total_iters=100000000
save_steps_freq=3000
#####################


today=$(date '+%m%d')
me=`basename "$0"`
me=$(echo $me | cut -d'.' -f 1)


# build run name
note="bnet"
name="${DATE_WITH_TIME}-${model}-${dataset_mode}-${cat}-LR${lr}-${note}"

debug=0
if [ $debug = 1 ]; then
    printf "${RED}Debugging!${NC}\n"
    batch_size=3
    max_dataset_size=120
    save_steps_freq=3
    display_freq=2
    print_freq=2
    name="DEBUG-${name}"
fi

# assemble python command
cmd="train.py \
    --name ${name} \
    --logs_dir ${logs_dir} \
    --gpu_ids ${gpu_ids} \
    --lr ${lr} \
    --batch_size ${batch_size} \
    --max_dataset_size ${max_dataset_size} \
    --model ${model} \
    --df_cfg ${df_cfg} \
    --vq_model ${vq_model} \
    --vq_cfg ${vq_cfg} \
    --vq_ckpt ${vq_ckpt} \
    --vq_dset ${vq_dset} \
    --vq_cat ${vq_cat} \
    --dataset_mode ${dataset_mode} \
    --res ${res} \
    --cat ${cat} \
    --trunc_thres ${trunc_thres} \
    --display_freq ${display_freq} \
    --print_freq ${print_freq} \
    --total_iters ${total_iters} \
    --save_steps_freq ${save_steps_freq} \
    --debug ${debug}"

# add dataroot if non‑empty
if [ ! -z "$dataroot" ]; then
    cmd="${cmd} --dataroot ${dataroot}"
    echo "setting dataroot to: ${dataroot}"
fi

if [ ! -z "$ckpt" ]; then
    cmd="${cmd} --ckpt ${ckpt}"
    echo "continue training with ckpt=${ckpt}"
fi

multi_gpu=0
if [ ${#gpu_ids} -gt 1 ]; then
    multi_gpu=1
fi

echo "[*] Starting on $(hostname), GPUs: ${gpu_ids}, logs_dir: ${logs_dir}"

# wrap for distributed if needed
if [ $multi_gpu = 1 ]; then
    cmd="-m torch.distributed.launch --nproc_per_node=${NGPU} --master_port=${PORT} ${cmd}"
fi

echo "[*] Final command:"
echo "CUDA_VISIBLE_DEVICES=${gpu_ids} python ${cmd}"

# run it
CUDA_VISIBLE_DEVICES=${gpu_ids} python ${cmd}
