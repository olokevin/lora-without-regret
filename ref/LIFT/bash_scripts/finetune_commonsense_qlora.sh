#!/bin/bash

pwd
hostname
date
echo starting job...
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export LIBRARY_PATH="/usr/local/cuda/lib64:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export HF_HOME="${HF_HOME:-/data/yequan/huggingface/cache}"

PROJECT_DIR="${PROJECT_DIR:-/home/yequan/Project/lora/lora-without-regret/.worktrees/qfura}"
SRC_DIR="${SRC_DIR:-${PROJECT_DIR}/ref/LIFT}"
DATA_DIR="${DATA_DIR:-/data/ruijiezhang/llm-adapter_bp/LLM-Adapters}"
OUTPUT_SRC_DIR="${OUTPUT_SRC_DIR:-/data/yequan/fura/lift}"

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B}"
lora_r="${lora_r:-48}"
lora_alpha="${lora_alpha:-96}"
lora_dropout="${lora_dropout:-0.05}"
lr="${lr:-2e-4}"
seed="${seed:-43}"
MAX_STEPS="${MAX_STEPS:-0}"
model_tag="${MODEL##*/}"

wandb_project="${wandb_project:-qlora-commonsense-${model_tag}}"
wandb_run_id="${wandb_run_id:-$(uv run --project ${PROJECT_DIR} python -c 'import wandb; print(wandb.util.generate_id())' 2>/dev/null)}"
no_wandb_flag=""
if [ "${no_wandb:-0}" = "1" ]; then
    no_wandb_flag="--no_wandb"
fi

export WANDB_RUN_ID="${wandb_run_id}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"

echo $MODEL

OUTPUT=${OUTPUT_SRC_DIR}/commonsense/${MODEL}/qlora-r_${lora_r}-alpha_${lora_alpha}-lr_${lr}-seed_${seed}
run_name="${run_name:-$(basename "$OUTPUT")}"

mkdir -p $OUTPUT

cd ${SRC_DIR}

uv run --project ${PROJECT_DIR} accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    --mixed_precision="bf16" \
    src/finetune_qlora.py \
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
    --lora_r ${lora_r} \
    --lora_alpha ${lora_alpha} \
    --lora_dropout ${lora_dropout} \
    --max_steps ${MAX_STEPS} \
    --val_set_size 120 \
    --eval_step 400 \
    --save_interval 100000 \
    --data_path ${DATA_DIR}/ft-training_set/commonsense_170k.json \
    --wandb_project "${wandb_project}" \
    --wandb_run_name "${run_name}" \
    ${no_wandb_flag} \
    --output_dir $OUTPUT 2> >(tee $OUTPUT/err.log >&2) | tee $OUTPUT/training.log

if [ "${MAX_STEPS}" = "0" ]; then
    MERGED="${OUTPUT}-merged"
    if [ ! -f "${MERGED}/config.json" ]; then
        uv run --project ${PROJECT_DIR} python ${PROJECT_DIR}/tools/merge_qlora_for_eval.py \
            --base_model ${MODEL} \
            --adapter_dir "${OUTPUT}" \
            --output_dir "${MERGED}"
    fi
    bash ./bash_scripts/eval_commonsense.sh \
        CKPT="${MERGED}" \
        base_model="${MODEL}" \
        wandb_project="${wandb_project}" \
        wandb_run_name="${run_name}" \
        wandb_run_id="${wandb_run_id}"
fi
