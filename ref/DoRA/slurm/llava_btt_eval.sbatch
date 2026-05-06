#!/usr/bin/env bash
#SBATCH --job-name=btt_llava_eval
#SBATCH --account=m4788_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=08:00:00
#SBATCH --output=/pscratch/sd/l/liyantan/dora/logs/slurm-%x-%j.out

# BTT full eval suite: GQA/SQA/TextVQA/POPE local + VQAv2/VizWiz/MMBench upload files.
# Uses 7B_eval_btt.sh (no --model-base, checkpoint is a full LlavaLlamaForCausalLM
# after BTT materialize_btt_to_linear at save time).
# Submit:  sbatch ref/DoRA/slurm/llava_btt_eval.sbatch

source /global/homes/l/liyantan/lora-without-regret/ref/DoRA/slurm/_common.sh

conda activate /pscratch/sd/l/liyantan/dora/envs/llava

BASE=${DORA_REPO}/visual_instruction_tuning
cd ${BASE}

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=hsn
export OMP_NUM_THREADS=4

LOCAL_SCRATCH="${TMPDIR:-/tmp}"
export TRITON_CACHE_DIR="${LOCAL_SCRATCH}/${USER}/triton_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${LOCAL_SCRATCH}/${USER}/xdg_${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"

export CUDA_VISIBLE_DEVICES=0,1,2,3

CKPT_NAME=llava-v1.5-7b-btt-full-output_one_block-small-keep_trainable

bash 7B_eval_btt.sh \
    ${CKPT_NAME} \
    ${BASE}/checkpoints/${CKPT_NAME} \
    ${BASE}/playground/data/eval \
    ${BASE}/eval_result/${CKPT_NAME} \
    ${BASE}
