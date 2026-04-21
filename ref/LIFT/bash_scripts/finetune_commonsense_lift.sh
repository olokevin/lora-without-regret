#!/bin/bash

pwd
hostname
date
echo starting job...
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export LIBRARY_PATH="/usr/local/cuda/lib64:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export HF_HOME=/your/path/to/huggingface/cache      # MODIFY THIS LINE

SRC_DIR=/home/yequan/Project/lora/lora-without-regret/ref/LIFT      # MODIFY THIS LINE
DATA_DIR=LLM-Adapters      # MODIFY THIS LINE
OUTPUT_SRC_DIR=/data/yequan/fura/lift    # MODIFY THIS LINE

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B}"
no_grad="${no_grad:-0.1}"
mask="${mask:-topk}"
lr="${lr:-2e-4}"
lora_rank="${lora_rank:-32}"
filter_rank="${filter_rank:-${lora_rank}}"
update_interval="${update_interval:-500}"
seed="${seed:-43}"
MAX_STEPS="${MAX_STEPS:-0}"
model_tag="${MODEL##*/}"
wandb_project="${wandb_project:-commonsense-${model_tag}}"

echo $MODEL


peft_tuner=sparse


OUTPUT=${OUTPUT_SRC_DIR}/commonsense/${MODEL}/lift-lr_${lr}-rank_${lora_rank}-seed_${seed}
run_name="${run_name:-lift-lr_${lr}-rank_${lora_rank}-seed_${seed}}"
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./outs/math/s2_llama3
fi
mkdir -p $OUTPUT

cd ${SRC_DIR}

accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    --mixed_precision="bf16" \
    src/finetune_sft.py \
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
    --peft_tuner ${peft_tuner} \
    --mask_type ${mask} \
    --lora_rank ${lora_rank} \
    --filter_rank ${filter_rank} \
    --update_interval ${update_interval} \
    --save_interval 100000 \
    --instruction_type single \
    --val_set_size 120 \
    --eval_step 400 \
    --no_grad ${no_grad} \
    --data_path ${DATA_DIR}/ft-training_set/commonsense_170k.json \
    --wandb_project "${wandb_project}" \
    --wandb_run_name "${run_name}" \
    --max_steps ${MAX_STEPS} \
    --output_dir $OUTPUT 2> >(tee $OUTPUT/err.log >&2) | tee $OUTPUT/training.log

if [ "${MAX_STEPS}" = "0" ]; then
    bash ./bash_scripts/eval_commonsense.sh \
        CKPT="$OUTPUT" \
        base_model="${MODEL}" \
        wandb_project="${wandb_project}" \
        wandb_run_name="${run_name}"
fi
