#!/usr/bin/env bash
#SBATCH --job-name=dora_llava_eval
#SBATCH --account=m4788_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=08:00:00
#SBATCH --output=/pscratch/sd/l/liyantan/dora/logs/slurm-%x-%j.out

# Run the LLaVA eval suite (GQA, ScienceQA, TextVQA, POPE local; VQAv2/VizWiz/MMBench upload).
# Submit:  sbatch ref/DoRA/slurm/llava_eval.sbatch
# Assumes training checkpoint in
#   ${DORA_REPO}/visual_instruction_tuning/checkpoints/llava-v1.5-7b-dora-r128-alpha-256/
# and eval data under ${DORA_REPO}/visual_instruction_tuning/playground/data/eval/

source /global/homes/l/liyantan/lora-without-regret/ref/DoRA/slurm/_common.sh

conda activate /pscratch/sd/l/liyantan/dora/envs/llava

BASE=${DORA_REPO}/visual_instruction_tuning
cd ${BASE}

# Offline: compute nodes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# NCCL
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=hsn

export OMP_NUM_THREADS=4

# Per-job cache isolation
LOCAL_SCRATCH="${TMPDIR:-/tmp}"
export TRITON_CACHE_DIR="${LOCAL_SCRATCH}/${USER}/triton_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${LOCAL_SCRATCH}/${USER}/xdg_${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"

export CUDA_VISIBLE_DEVICES=0,1,2,3

CKPT_NAME=llava-v1.5-7b-dora-r128-alpha-256

bash 7B_eval_dora.sh \
    ${CKPT_NAME} \
    ${BASE}/checkpoints/${CKPT_NAME} \
    ${BASE}/playground/data/eval \
    ${BASE}/eval_result/${CKPT_NAME} \
    ${BASE}
