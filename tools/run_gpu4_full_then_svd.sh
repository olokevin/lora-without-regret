#!/bin/bash
# Launch full-FT then SVD-FT on GPU 4 with bsz=1, grad_accum=16, no gradient checkpointing,
# lr=5e-5. If a run OOMs, retry once with --gradient_checkpointing.
set -uo pipefail

export CUDA_VISIBLE_DEVICES=4
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export LIBRARY_PATH="/usr/local/cuda/lib64:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME:-/data/yequan/huggingface}"

MODEL=meta-llama/Llama-3.1-8B
LR=5e-5
SEED=43
SRC_DIR=/home/yequan/Project/lora/lora-without-regret/ref/LIFT
DATA_DIR=LLM-Adapters
OUT_BASE=/data/yequan/fura/lift/commonsense/${MODEL}

cd ${SRC_DIR}

# Common args (no --gradient_checkpointing here; appended on retry).
COMMON=(
    --model_name_or_path "${MODEL}"
    --per_device_train_batch_size 1
    --per_device_eval_batch_size 1
    --gradient_accumulation_steps 16
    --logging_steps 10
    --max_seq_len 2048
    --learning_rate "${LR}"
    --weight_decay 0.
    --num_train_epochs 3
    --mixed_precision bf16
    --lr_scheduler_type linear
    --num_warmup_steps 0.03
    --seed "${SEED}"
    --instruction_type single
    --save_interval 100000
    --val_set_size 120
    --eval_step 400
    --data_path "${DATA_DIR}/ft-training_set/commonsense_170k.json"
    --max_steps 0
)

run_one() {
    # $1 = entrypoint .py, $2 = output_dir, $3 = wandb_run_name, rest = extra args
    local entry=$1; shift
    local outdir=$1; shift
    local rname=$1; shift
    local extra=("$@")
    mkdir -p "${outdir}"
    local tag="${rname}"

    # Attempt 1: no gradient checkpointing.
    echo "=== [$(date)] ${tag} attempt 1 (no grad-ckpt, bsz=1, accum=16) ==="
    accelerate launch \
        --num_machines 1 --num_processes 1 --mixed_precision=bf16 \
        "${entry}" \
        "${COMMON[@]}" "${extra[@]}" \
        --wandb_project "commonsense-Llama-3.1-8B" \
        --wandb_run_name "${rname}" \
        --output_dir "${outdir}" \
        2> >(tee "${outdir}/err.log" >&2) | tee "${outdir}/training.log"
    local rc=${PIPESTATUS[0]}

    if [ $rc -ne 0 ] && grep -qE "out of memory|CUDA out of memory|OutOfMemoryError" "${outdir}/err.log"; then
        echo "=== [$(date)] ${tag} OOMed; retrying with --gradient_checkpointing ==="
        local outdir2="${outdir}-gradckpt"
        local rname2="${rname}-gradckpt"
        mkdir -p "${outdir2}"
        accelerate launch \
            --num_machines 1 --num_processes 1 --mixed_precision=bf16 \
            "${entry}" \
            "${COMMON[@]}" "${extra[@]}" \
            --gradient_checkpointing \
            --wandb_project "commonsense-Llama-3.1-8B" \
            --wandb_run_name "${rname2}" \
            --output_dir "${outdir2}" \
            2> >(tee "${outdir2}/err.log" >&2) | tee "${outdir2}/training.log"
        rc=${PIPESTATUS[0]}
    fi

    if [ $rc -ne 0 ]; then
        echo "=== [$(date)] ${tag} FAILED (rc=$rc) ==="
        return $rc
    fi
    echo "=== [$(date)] ${tag} OK ==="
    return 0
}

# --- Full FT ---
FULL_OUT="${OUT_BASE}/full-lr_${LR}-bsz1_accum16_nockpt-seed_${SEED}"
run_one src/finetune_sft.py "${FULL_OUT}" "full-lr_${LR}-bsz1_accum16_nockpt-seed_${SEED}" || true

# --- SVD FT (train_position=input, s_merged_to=trainable) ---
SVD_OUT="${OUT_BASE}/svd-lr_${LR}-pos_input_smerge_trainable-bsz1_accum16_nockpt-seed_${SEED}"
run_one src/finetune_svd.py "${SVD_OUT}" "svd-lr_${LR}-pos_input_smerge_trainable-bsz1_accum16_nockpt-seed_${SEED}" \
    --train_position input \
    --s_merged_to trainable \
    --trainable_type all || true

echo "=== [$(date)] GPU4 full+svd chain DONE ==="
