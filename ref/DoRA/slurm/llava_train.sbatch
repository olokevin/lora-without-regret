#!/usr/bin/env bash
#SBATCH --job-name=dora_llava_train
#SBATCH --account=m4788_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=48:00:00
#SBATCH --output=/pscratch/sd/l/liyantan/dora/logs/slurm-%x-%j.out

# LLaVA-1.5-7B DoRA fine-tuning (Table 4/12 reproduction).
# Runs DeepSpeed ZeRO-3 across 4x A100 40GB on a single Perlmutter GPU node.
# Submit:  sbatch ref/DoRA/slurm/llava_train.sbatch
#
# Assumes the following are ready under ${DORA_REPO}/visual_instruction_tuning/:
#   playground/data/llava_v1_5_mix665k.json
#   playground/data/coco/train2017/
#   playground/data/gqa/, textvqa/, ocr_vqa/, vg/
#   checkpoints/llava-v1.5-7b-pretrain/mm_projector.bin

source /global/homes/l/liyantan/lora-without-regret/ref/DoRA/slurm/_common.sh

conda activate /pscratch/sd/l/liyantan/dora/envs/llava

cd ${DORA_REPO}/visual_instruction_tuning

# ---- offline mode: compute nodes have no outbound internet ----
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ---- wandb: log offline, sync after job finishes ----
export WANDB_MODE=offline
export WANDB_PROJECT=dora-llava
export WANDB_DIR=${DATA_ROOT}/wandb
export WANDB_NAME=llava-dora-r128-a256-${SLURM_JOB_ID}
mkdir -p ${WANDB_DIR}

# ---- NCCL: Slingshot-friendly settings (no IB, use hsn socket) ----
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=hsn

# ---- torch / NCCL timeout: give slow dataloader 30 min before erroring ----
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=1800

# ---- CPU threading ----
export OMP_NUM_THREADS=4

# ---- Per-job cache isolation on node-local scratch ----
# Triton / inductor / XDG caches: avoid concurrent rank writes on shared FS.
LOCAL_SCRATCH="${TMPDIR:-/tmp}"
export TRITON_CACHE_DIR="${LOCAL_SCRATCH}/${USER}/triton_${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="${LOCAL_SCRATCH}/${USER}/torchinductor_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${LOCAL_SCRATCH}/${USER}/xdg_${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME"

# Give deepspeed 4 GPUs on this node.
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Dora_7b.sh has gradient_accumulation_steps=8, tuned for 4 GPUs
# (preserving effective batch of 128). See SETUP_GUIDE.md §5.
bash ./Dora_7b.sh

# Sync wandb if this job happens to run from login-reachable network (rare).
# Usually you'll sync manually on login:
#   wandb sync ${DATA_ROOT}/wandb/offline-run-*
echo "To upload metrics to wandb, run on login node:"
echo "  wandb sync ${DATA_ROOT}/wandb/offline-run-*"
