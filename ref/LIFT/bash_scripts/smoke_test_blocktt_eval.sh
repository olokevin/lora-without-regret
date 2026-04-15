#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${LIFT_DIR}/../.." && pwd)"
PYTHON_BIN="${REPO_DIR}/.venv/bin/python"

SMOKE_ROOT="${REPO_DIR}/tests/smoke_runs/blocktt_eval_smoke"
TINY_MODEL_DIR="${SMOKE_ROOT}/tiny_qwen3_model"
CKPT_DIR="${SMOKE_ROOT}/blocktt_ckpt"
EVAL_DIR="${SMOKE_ROOT}/eval_boolq"

TRAIN_JSON="${REPO_DIR}/tests/smoke_data/blocktt_eval/train.json"
BOOLQ_JSON="${REPO_DIR}/tests/smoke_data/blocktt_eval/boolq_test.json"
BASE_MODEL_DIR="${REPO_DIR}/sft/full_model"

mkdir -p "${SMOKE_ROOT}"

echo "[1/4] Building tiny local model for fast smoke testing..."
"${PYTHON_BIN}" "${REPO_DIR}/tests/smoke_data/blocktt_eval/make_tiny_qwen3_model.py" \
  --base-tokenizer-dir "${BASE_MODEL_DIR}" \
  --base-config-dir "${BASE_MODEL_DIR}" \
  --out-dir "${TINY_MODEL_DIR}"

echo "[2/4] Running tiny BlockTT finetune to force checkpoint save..."
"${PYTHON_BIN}" "${LIFT_DIR}/src/finetune_blocktt.py" \
  --model_name_or_path "${TINY_MODEL_DIR}" \
  --data_path "${TRAIN_JSON}" \
  --output_dir "${CKPT_DIR}" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --max_seq_len 128 \
  --logging_steps 1 \
  --learning_rate 5e-4 \
  --val_set_size 0 \
  --trainable_type attn \
  --decomp_mode input_one_block \
  --blocktt_rank full \
  --train_position small \
  --s_merged_to frozen \
  --instruction_type single \
  --no_wandb

test -f "${CKPT_DIR}/config.json"
test -f "${CKPT_DIR}/pytorch_model.bin"
echo "Saved checkpoint files found in ${CKPT_DIR}"

# save_hf_format does not persist tokenizer files; copy them for local eval smoke tests.
for f in tokenizer.json tokenizer_config.json special_tokens_map.json vocab.json merges.txt added_tokens.json chat_template.jinja; do
  if [[ -f "${TINY_MODEL_DIR}/${f}" && ! -f "${CKPT_DIR}/${f}" ]]; then
    cp "${TINY_MODEL_DIR}/${f}" "${CKPT_DIR}/${f}"
  fi
done

echo "[3/4] Running boolq eval on saved BlockTT checkpoint..."
mkdir -p "${EVAL_DIR}"
"${PYTHON_BIN}" "${LIFT_DIR}/src/eval/run_commonsense_parallel.py" \
  --data_path "${BOOLQ_JSON}" \
  --model_name_or_path "${CKPT_DIR}" \
  --per_device_eval_batch_size 1 \
  --seed 1234 \
  --dtype bf16 \
  --dataset boolq \
  --output_dir "${EVAL_DIR}"

test -f "${EVAL_DIR}/model_predictions.jsonl"
echo "Eval output found: ${EVAL_DIR}/model_predictions.jsonl"
echo "Prediction lines: $(wc -l < "${EVAL_DIR}/model_predictions.jsonl")"

echo "[4/4] BlockTT save->eval smoke flow completed successfully."
