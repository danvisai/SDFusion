#!/usr/bin/env bash
# VQVAE v2 — quality rebuild for Stage 3a's latent space.
#
# Differences from launchers/train_vqvae_bnet.sh:
#   - configs/vqvae_bnet_v2.yaml (ch=96, num_res_blocks=2, attn@16)
#   - trunc_thres 0.2 -> 0.3 (expose more SDF tail)
#   - batch 3 -> 12 (A100 80GB underutilized at batch 3)
#   - --augment (Y-rotations + X/Z flips, train phase only)
#   - --use_aux_losses (surface-band SmoothL1 + footprint BCE on top of L1 + codebook)
#   - --use_adamw_cosine (AdamW + linear warmup + cosine, replaces StepLR)
#   - 80k total iters at batch 12 (~36 h on A100 80GB, target val IoU > 0.90)
#
# Defaults preserve v1 behavior; everything new is opt-in via flags.
#
# Run with:
#   env -u LD_PRELOAD -u LD_LIBRARY_PATH bash launchers/train_vqvae_bnet_v2.sh
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
batch_size=12

model="vqvae"
vq_cfg="configs/vqvae_bnet_v2.yaml"

max_dataset_size=10000000
dataset_mode='building'
dataroot="data"
res=64
cat='all'
trunc_thres=0.3   # v1 was 0.2; bumped to expose more SDF tail

# Training horizon. ~80k steps at batch 12 = ~650 epochs on 1480 train ids.
# Cosine reaches 0 at cosine_total_steps; we stop at total_iters.
total_iters=80000
cosine_total_steps=80000
warmup_steps=1000

# Aux loss weights (verbatim defaults from base_options.py).
aux_band_weight=0.5
aux_fp_weight=0.25
aux_band_sigma=0.05
aux_fp_tau=0.05

display_freq=2000
print_freq=50
save_latest_freq=1000
save_steps_freq=5000

today=$(date '+%m%d')
me=`basename "$0" | cut -d'.' -f 1`

note="v2-aux-aug-cosine"
name="${DATE_WITH_TIME}-${model}-${dataset_mode}-${cat}-res${res}-LR${lr}-T${trunc_thres}-${note}"

debug=0
if [ $debug = 1 ]; then
    echo -e "${RED}Debugging!${NC}"
    batch_size=4
    max_dataset_size=24
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
                --model ${model} --vq_cfg ${vq_cfg} \
                --dataset_mode ${dataset_mode} --cat ${cat} --res ${res} --trunc_thres ${trunc_thres} --max_dataset_size ${max_dataset_size} \
                --display_freq ${display_freq} --print_freq ${print_freq} \
                --save_latest_freq ${save_latest_freq} --save_steps_freq ${save_steps_freq} \
                --total_iters ${total_iters} \
                --augment --use_aux_losses --use_adamw_cosine \
                --warmup_steps ${warmup_steps} --cosine_total_steps ${cosine_total_steps} \
                --aux_band_weight ${aux_band_weight} --aux_fp_weight ${aux_fp_weight} \
                --aux_band_sigma ${aux_band_sigma} --aux_fp_tau ${aux_fp_tau} \
                --debug ${debug}"

if [ ! -z "$dataroot" ]; then
    cmd="${cmd} --dataroot ${dataroot}"
    echo "setting dataroot to: ${dataroot}"
fi

multi_gpu=0
if [ ${#gpu_ids} -gt 1 ]; then
    multi_gpu=1
fi

echo "[*] Training is starting on `hostname`, GPU#: ${gpu_ids}, logs_dir: ${logs_dir}"

if [ $multi_gpu = 1 ]; then
    cmd="-m torch.distributed.launch --nproc_per_node=${NGPU} --master_port=${PORT} ${cmd}"
fi

echo "[*] Training with command:"
echo "CUDA_VISIBLE_DEVICES=${gpu_ids} python ${cmd}"

CUDA_VISIBLE_DEVICES=${gpu_ids} python ${cmd}
