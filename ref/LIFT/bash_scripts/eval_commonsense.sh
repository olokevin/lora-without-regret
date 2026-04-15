#!/bin/bash

CKPT=""
base_model="meta-llama/Meta-Llama-3-8B"
wandb_project="${wandb_project:-}"
wandb_run_name="${run_name:-}"
wandb_run_id="${wandb_run_id:-${WANDB_RUN_ID:-}}"

usage() {
    echo "Usage: bash bash_scripts/eval_commonsense.sh CKPT=<model_checkpoint_path> [base_model=<hf_model>] [wandb_project=<project>] [wandb_run_name=<name>] [wandb_run_id=<id>]"
}

for arg in "$@"; do
    case "$arg" in
        CKPT=*|ckpt=*)
            CKPT="${arg#*=}"
            ;;
        MODEL=*|model=*)
            CKPT="${arg#*=}"
            ;;
        base_model=*)
            base_model="${arg#*=}"
            ;;
        wandb_project=*)
            wandb_project="${arg#*=}"
            ;;
        wandb_run_name=*|run_name=*)
            wandb_run_name="${arg#*=}"
            ;;
        wandb_run_id=*)
            wandb_run_id="${arg#*=}"
            ;;
        *)
            echo "Unknown argument: $arg"
            usage
            exit 1
            ;;
    esac
done

if [ -z "$CKPT" ]; then
    usage
    exit 1
fi

MODEL="$CKPT"
OUTPUT_DIR="${MODEL}/commonsense"

SRC_DIR=/home/yequan/Project/lora/lora-without-regret/ref/LIFT
DATA_DIR=LLM-Adapters/dataset

# datasets=(boolq)
datasets=(boolq piqa social_i_qa ARC-Challenge ARC-Easy openbookqa hellaswag winogrande)

cd $SRC_DIR

if [ -n "$wandb_run_id" ]; then
    export WANDB_RUN_ID="$wandb_run_id"
    export WANDB_RESUME="${WANDB_RESUME:-must}"
fi

master_port=$((RANDOM % 5000 + 20000))
for dataset in "${datasets[@]}"; do
    OUTPUT=$OUTPUT_DIR/$dataset
    mkdir -p $OUTPUT
    BATCH_SIZE=32
    cmd=(
        accelerate launch --main_process_port "$master_port" ./src/eval/run_commonsense_parallel.py
        --data_path "${DATA_DIR}/$dataset/test.json"
        --model_name_or_path "$MODEL"
        --base_model "$base_model"
        --per_device_eval_batch_size "$BATCH_SIZE"
        --seed 1234
        --dtype bf16
        --dataset "$dataset"
        --output_dir "$OUTPUT"
    )
    if [ -n "$wandb_project" ]; then
        cmd+=(--wandb_project "$wandb_project")
    fi
    if [ -n "$wandb_run_name" ]; then
        cmd+=(--wandb_run_name "$wandb_run_name")
    fi
    if [ -n "$wandb_run_id" ]; then
        cmd+=(--wandb_run_id "$wandb_run_id")
    fi

    "${cmd[@]}" 2> >(tee "$OUTPUT/eval_err.log" >&2) | tee "$OUTPUT/eval.log"
done
