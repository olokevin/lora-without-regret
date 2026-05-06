#!/usr/bin/env bash
#SBATCH --job-name=btt_llava_train
#SBATCH --account=m4788_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=48:00:00
#SBATCH --output=/pscratch/sd/l/liyantan/dora/logs/slurm-%x-%j.out

# LLaVA-1.5-7B BlockTT fine-tuning — comparison against DoRA (Table 4/12).
# rank=full, decomp=output_one_block, train_position=small, s_merged_to=keep_trainable.
# Same LR/schedule/data as Dora_7b.sh. DeepSpeed ZeRO-3 on 4x A100 40GB.
#
# Submit:  sbatch ref/DoRA/slurm/llava_btt_train.sbatch

source /global/homes/l/liyantan/lora-without-regret/ref/DoRA/slurm/_common.sh

conda activate /pscratch/sd/l/liyantan/dora/envs/llava

cd ${DORA_REPO}/visual_instruction_tuning

# ---- offline mode ----
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ---- wandb offline ----
export WANDB_MODE=offline
export WANDB_PROJECT=dora-llava
export WANDB_DIR=${DATA_ROOT}/wandb
export WANDB_NAME=llava-btt-full-ob-small-kt-${SLURM_JOB_ID}
mkdir -p ${WANDB_DIR}

# ---- NCCL / torch ----
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=hsn
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=1800
export OMP_NUM_THREADS=4

# ---- Per-job cache isolation ----
LOCAL_SCRATCH="${TMPDIR:-/tmp}"
export TRITON_CACHE_DIR="${LOCAL_SCRATCH}/${USER}/triton_${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="${LOCAL_SCRATCH}/${USER}/torchinductor_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${LOCAL_SCRATCH}/${USER}/xdg_${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME"

export CUDA_VISIBLE_DEVICES=0,1,2,3

bash ./BTT_7b.sh

echo "To upload metrics to wandb, run on login node:"
echo "  wandb sync ${DATA_ROOT}/wandb/offline-run-*"
