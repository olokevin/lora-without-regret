#!/bin/bash
# Full-FT math launcher with bs=2, grad_accum=8, no gradient checkpointing.
# Mirrors finetune_math_full.sh; differs only in batch/accum/ckpt.

pwd
hostname
date
echo starting job...
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export LIBRARY_PATH="/usr/local/cuda/lib64:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export HF_HOME="${HF_HOME:-/data/yequan/huggingface/cache}"

SRC_DIR="${SRC_DIR:-/home/yequan/Project/lora/lora-without-regret/ref/LIFT}"
DATA_DIR="${DATA_DIR:-LLM-Adapters}"
OUTPUT_SRC_DIR="${OUTPUT_SRC_DIR:-/data/yequan/fura/lift}"

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B}"
lr="${lr:-5e-5}"
seed="${seed:-43}"
model_tag="${MODEL##*/}"
wandb_project="${wandb_project:-math-${model_tag}}"
wandb_run_id="${wandb_run_id:-$(python -c 'import wandb; print(wandb.util.generate_id())' 2>/dev/null || echo "")}"

export WANDB_RUN_ID="${wandb_run_id}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"

echo $MODEL

OUTPUT=${OUTPUT_SRC_DIR}/math/${MODEL}/full-bs2gacc8-noctk-lr_${lr}-seed_${seed}
run_name="${run_name:-$(basename "$OUTPUT")}"
mkdir -p $OUTPUT

cd ${SRC_DIR}

accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    --mixed_precision="bf16" \
    src/finetune_sft.py \
    --model_name_or_path ${MODEL} \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 16 \
    --logging_steps 10 \
    --max_seq_len 2048 \
    --learning_rate ${lr} \
    --weight_decay 0. \
    --num_train_epochs 3 \
    --mixed_precision bf16 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type linear \
    --num_warmup_steps 0.03 \
    --seed ${seed} \
    --instruction_type single \
    --load_last_model \
    --data_path ${DATA_DIR}/ft-training_set/math_10k.json \
    --wandb_project "${wandb_project}" \
    --wandb_run_name "${run_name}" \
    --output_dir $OUTPUT 2> >(tee $OUTPUT/err.log >&2) | tee $OUTPUT/training.log

bash ./bash_scripts/eval_math.sh \
    CKPT="$OUTPUT" \
    base_model="${MODEL}" \
    wandb_project="${wandb_project}" \
    wandb_run_name="${run_name}" \
    wandb_run_id="${wandb_run_id}"
