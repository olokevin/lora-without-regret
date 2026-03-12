#!/usr/bin/env bash
set -euo pipefail

# Global mode selector (outside LR iteration): full | lora | svd | blocktt
TRAIN_MODE="${TRAIN_MODE:-lora}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-qwen3-4B-SFT}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./sft/lr_search}"

BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACC="${GRAD_ACC:-16}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
SEED="${SEED:-42}"

# Space-separated LR list override, e.g.:
# LR_LIST="5e-6 1e-5 2.5e-5" ./run_lr_search.sh
if [[ -n "${LR_LIST:-}" ]]; then
  read -r -a LRS <<<"${LR_LIST}"
else
  LRS=(5e-5 1e-4 2e-4 3e-4 5e-4)
fi

MODE_ARGS=()
case "${TRAIN_MODE}" in
  full)
    ;;
  lora)
    MODE_ARGS+=(
      --lora-rank "${LORA_RANK:-64}"
      --trainable-type "${TRAINABLE_TYPE:-all}"
    )
    run_name="${TRAIN_MODE}—rank_${LORA_RANK:-64}"
    ;;
  svd)
    MODE_ARGS+=(
      --trainable-type "${TRAINABLE_TYPE:-all}"
      --train-position "${TRAIN_POSITION:-output}"
    )
    run_name="${TRAIN_MODE}—${TRAIN_POSITION:-output}"
    ;;
  blocktt)
    MODE_ARGS+=(
      --trainable-type "${TRAINABLE_TYPE:-all}"
      --decomp-mode "${DECOMP_MODE:-input_one_block}"
      --train-position "${TRAIN_POSITION:-small}"
      --blocktt-rank "${BLOCKTT_RANK:-full}"
    )
    if [[ "${NO_TRAIN_BIAS:-0}" == "1" ]]; then
      MODE_ARGS+=(--no-train-bias)
    fi
    run_name="${TRAIN_MODE}—${DECOMP_MODE}—${TRAIN_POSITION:-small}"
    ;;
  *)
    echo "Unsupported TRAIN_MODE: ${TRAIN_MODE}. Use full|lora|svd|blocktt." >&2
    exit 1
    ;;
esac

WANDB_ARGS=()
if [[ "${NO_WANDB:-0}" == "1" ]]; then
  WANDB_ARGS+=(--no-wandb)
else
  WANDB_ARGS+=(--wandb-project "${WANDB_PROJECT}")
fi

if [[ "${SAVE_CKPT:-0}" == "1" ]]; then
  SAVE_ARGS=(--enable-save-ckpt)
else
  SAVE_ARGS=()
fi

echo "Starting LR sweep"
echo "  TRAIN_MODE=${TRAIN_MODE}"
echo "  MODEL_ID=${MODEL_ID}"
echo "  CUDA_DEVICE=${CUDA_DEVICE}"
echo "  LRs=${LRS[*]}"

for lr in "${LRS[@]}"; do
  safe_lr="${lr//./p}"
  run_name="${run_name}_lr_${safe_lr}"
  output_dir="${OUTPUT_ROOT}/${TRAIN_MODE}/lr_${safe_lr}"

  cmd=(
    uv run run_sft.py
    --train-mode "${TRAIN_MODE}"
    --model-id "${MODEL_ID}"
    --lr "${lr}"
    --batch-size "${BATCH_SIZE}"
    --gradient-accumulation-steps "${GRAD_ACC}"
    --num-epochs "${NUM_EPOCHS}"
    --seed "${SEED}"
    --output-dir "${output_dir}"
    --wandb-run-name "${run_name}"
    "${MODE_ARGS[@]}"
    "${WANDB_ARGS[@]}"
    "${SAVE_ARGS[@]}"
  )

  echo
  echo "[RUN] lr=${lr}"
  echo "      output_dir=${output_dir}"
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${cmd[@]}"
done


echo
echo "LR sweep complete."
