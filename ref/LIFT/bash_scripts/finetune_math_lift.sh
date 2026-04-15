#!/bin/bash

pwd
hostname
date
echo starting job...
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export LIBRARY_PATH="/usr/local/cuda/lib64:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export HF_HOME="${HF_HOME:-/data/yequan/huggingface/cache}"      # MODIFY THIS LINE

SRC_DIR="${SRC_DIR:-/home/yequan/Project/lora/lora-without-regret/ref/LIFT}"      # MODIFY THIS LINE
DATA_DIR="${DATA_DIR:-LLM-Adapters}"      # MODIFY THIS LINE
OUTPUT_SRC_DIR="${OUTPUT_SRC_DIR:-/data/yequan/fura/lift}"    # MODIFY THIS LINE

# SLURM_ARRAY_TASK_ID=$1
# cfg=$(sed -n "$SLURM_ARRAY_TASK_ID"p ${SRC_DIR}/bash_scripts/slurm_config_lift_math.txt)
# MODEL=$(echo $cfg | cut -f 1 -d ' ')
# no_grad=$(echo $cfg | cut -f 2 -d ' ')
# mask=$(echo $cfg | cut -f 3 -d ' ')
# lr=$(echo $cfg | cut -f 4 -d ' ')
# lora_rank=$(echo $cfg | cut -f 5 -d ' ')
# filter_rank=$(echo $cfg | cut -f 6 -d ' ')
# update_interval=$(echo $cfg | cut -f 7 -d ' ')
# seed=$(echo $cfg | cut -f 8 -d ' ')

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B}"
no_grad="${no_grad:-head}"
mask="${mask:-lift_sparse}"
lr="${lr:-1e-4}"
lora_rank="${lora_rank:-128}"
filter_rank="${filter_rank:-128}"
update_interval="${update_interval:-400}"
seed="${seed:-43}"
model_tag="${MODEL##*/}"

wandb_project="${wandb_project:-math-${model_tag}}"
wandb_run_id="${wandb_run_id:-$(python -c 'import wandb; print(wandb.util.generate_id())')}"

export WANDB_RUN_ID="${wandb_run_id}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"

echo $MODEL

peft_tuner=sparse

OUTPUT=${OUTPUT_SRC_DIR}/math/${MODEL}/lift-${peft_tuner}-no_${no_grad}-mask_${mask}-rank_${lora_rank}-filter_${filter_rank}-interval_${update_interval}-lr_${lr}-seed_${seed}
run_name="${run_name:-$(basename "$OUTPUT")}"

mkdir -p $OUTPUT

cd ${SRC_DIR}

accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    --mixed_precision="bf16" \
    src/finetune_sft.py \
    --model_name_or_path ${MODEL} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 16 \
    --logging_steps 10 \
    --max_seq_len 2048 \
    --learning_rate ${lr} \
    --weight_decay 0. \
    --num_train_epochs 3 \
    --mixed_precision bf16 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type linear \
    --num_warmup_steps 0.03 \
    --seed ${seed} \
    --gradient_checkpointing \
    --instruction_type single \
    --peft_tuner ${peft_tuner} \
    --mask_type ${mask} \
    --lora_rank ${lora_rank} \
    --filter_rank ${filter_rank} \
    --update_interval ${update_interval} \
    --load_last_model \
    --no_grad ${no_grad} \
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
