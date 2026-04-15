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
# cfg=$(sed -n "$SLURM_ARRAY_TASK_ID"p ${SRC_DIR}/bash_scripts/slurm_config_blocktt_math.txt)
# MODEL=$(echo $cfg | cut -f 1 -d ' ')
# decomp_mode=$(echo $cfg | cut -f 2 -d ' ')
# train_position=$(echo $cfg | cut -f 3 -d ' ')
# blocktt_rank=$(echo $cfg | cut -f 4 -d ' ')
# s_merged_to=$(echo $cfg | cut -f 5 -d ' ')
# trainable_type=$(echo $cfg | cut -f 6 -d ' ')
# lr=$(echo $cfg | cut -f 7 -d ' ')
# seed=$(echo $cfg | cut -f 8 -d ' ')

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B}"
decomp_mode="${decomp_mode:-input_one_block}"
train_position="${train_position:-small}"
blocktt_rank="${blocktt_rank:-full}"
s_merged_to="${s_merged_to:-frozen}"
trainable_type="${trainable_type:-all}"
lr="${lr:-1e-4}"
seed="${seed:-43}"
model_tag="${MODEL##*/}"

# --- calibrated BTT knobs (set calib_mode=v2_bp to enable) ---
calib_mode="${calib_mode:-none}"
calib_source="${calib_source:-training_data}"
calib_num_seqs="${calib_num_seqs:-128}"
calib_batch_size="${calib_batch_size:-4}"

wandb_project="${wandb_project:-math-${model_tag}}"
wandb_run_id="${wandb_run_id:-$(python -c 'import wandb; print(wandb.util.generate_id())')}"

export WANDB_RUN_ID="${wandb_run_id}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"

echo $MODEL

OUTPUT=${OUTPUT_SRC_DIR}/math/${MODEL}/blocktt-calib_${calib_mode}-lr_${lr}-decomp_${decomp_mode}_pos_${train_position}-rank_${blocktt_rank}-smerge_${s_merged_to}-type_${trainable_type}-seed_${seed}
run_name="${run_name:-$(basename "$OUTPUT")}"

mkdir -p $OUTPUT

cd ${SRC_DIR}

accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    --mixed_precision="bf16" \
    src/finetune_blocktt.py \
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
    --decomp_mode ${decomp_mode} \
    --train_position ${train_position} \
    --blocktt_rank ${blocktt_rank} \
    --s_merged_to ${s_merged_to} \
    --trainable_type ${trainable_type} \
    --calib_mode ${calib_mode} \
    --calib_source ${calib_source} \
    --calib_num_seqs ${calib_num_seqs} \
    --calib_batch_size ${calib_batch_size} \
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
