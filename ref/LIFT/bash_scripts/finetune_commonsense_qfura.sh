#!/bin/bash

pwd
hostname
date
echo starting job...
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export LIBRARY_PATH="/usr/local/cuda/lib64:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export HF_HOME="${HF_HOME:-/data/yequan/huggingface}"

PROJECT_DIR="${PROJECT_DIR:-/home/yequan/Project/lora/lora-without-regret/.worktrees/qfura}"
SRC_DIR="${SRC_DIR:-${PROJECT_DIR}/ref/LIFT}"
DATA_DIR="${DATA_DIR:-/data/ruijiezhang/llm-adapter_bp/LLM-Adapters}"
OUTPUT_SRC_DIR="${OUTPUT_SRC_DIR:-/data/yequan/fura/lift}"

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B}"
decomp_mode="${decomp_mode:-output_one_block}"
train_position="${train_position:-small}"
s_merged_to="${s_merged_to:-keep_trainable}"
blocktt_rank="${blocktt_rank:-full}"
trainable_type="${trainable_type:-all}"
quant_block_layout="${quant_block_layout:-flat}"
lr="${lr:-2e-4}"
seed="${seed:-43}"
MAX_STEPS="${MAX_STEPS:-0}"
model_tag="${MODEL##*/}"

wandb_project="${wandb_project:-qfura-${model_tag}}"
wandb_run_id="${wandb_run_id:-$(uv run --project ${PROJECT_DIR} python -c 'import wandb; print(wandb.util.generate_id())' 2>/dev/null)}"
no_wandb_flag=""
if [ "${no_wandb:-0}" = "1" ]; then
    no_wandb_flag="--no_wandb"
fi
export WANDB_RUN_ID="${wandb_run_id}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"

echo $MODEL

OUTPUT="${OUTPUT:-${OUTPUT_SRC_DIR}/commonsense/${MODEL}/qfura-layout_${quant_block_layout}-decomp_${decomp_mode}_smerge_${s_merged_to}-lr_${lr}-seed_${seed}}"
run_name="${run_name:-$(basename "$OUTPUT")}"

mkdir -p $OUTPUT

cd ${SRC_DIR}

uv run --project ${PROJECT_DIR} accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    --mixed_precision="bf16" \
    src/finetune_qfura.py \
    --model_name_or_path ${MODEL} \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --max_seq_len 2048 \
    --learning_rate ${lr} \
    --weight_decay 0. \
    --num_train_epochs 3 \
    --mixed_precision bf16 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type linear \
    --num_warmup_steps 0.03 \
    --seed ${seed} \
    --gradient_checkpointing \
    --instruction_type single \
    --decomp_mode ${decomp_mode} \
    --train_position ${train_position} \
    --blocktt_rank ${blocktt_rank} \
    --s_merged_to ${s_merged_to} \
    --trainable_type ${trainable_type} \
    --quant_block_layout ${quant_block_layout} \
    --save_interval 100000 \
    --val_set_size 120 \
    --eval_step 400 \
    --load_last_model \
    --data_path ${DATA_DIR}/ft-training_set/commonsense_170k.json \
    --wandb_project "${wandb_project}" \
    --wandb_run_name "${run_name}" \
    ${no_wandb_flag} \
    --max_steps ${MAX_STEPS} \
    --output_dir $OUTPUT 2> >(tee $OUTPUT/err.log >&2) | tee $OUTPUT/training.log

if [ "${MAX_STEPS}" = "0" ]; then
    bash ./bash_scripts/eval_commonsense.sh \
        CKPT="$OUTPUT" \
        base_model="${MODEL}" \
        wandb_project="${wandb_project}" \
        wandb_run_name="${run_name}" \
        wandb_run_id="${wandb_run_id}"
fi
