#!/usr/bin/env bash
# Convert every legacy calibrated-BTT checkpoint under the LIFT output
# roots (commonsense + math) into HuggingFace-format checkpoints.
#
# A legacy checkpoint is detected as: any directory that contains BOTH
# `btt_topology.json` and `model.safetensors` but no `config.json`.
# The converted HF checkpoint is written to `<ckpt>/hf`.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash ref/LIFT/scripts/convert_all_calib_btt_to_hf.sh
#
# Overrides:
#   LIFT_OUTPUT_ROOT  root dir under which to search     (default: /data/yequan/fura/lift)
#   BASE_MODEL        HF base model id for tokenizer/config (default: meta-llama/Meta-Llama-3-8B)
#   DTYPE             bf16 | fp16 | fp32                 (default: bf16)
#   DRY_RUN           1 to list conversions without running them
#   FORCE             1 to reconvert even when <ckpt>/hf already exists

set -euo pipefail

LIFT_OUTPUT_ROOT="${LIFT_OUTPUT_ROOT:-/data/yequan/fura/lift}"
BASE_MODEL="${BASE_MODEL:-meta-llama/Meta-Llama-3-8B}"
DTYPE="${DTYPE:-bf16}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONVERTER="${REPO_ROOT}/ref/LIFT/scripts/convert_calib_btt_to_hf.py"

if [[ ! -f "${CONVERTER}" ]]; then
    echo "error: converter not found at ${CONVERTER}" >&2
    exit 1
fi

if [[ ! -d "${LIFT_OUTPUT_ROOT}" ]]; then
    echo "error: LIFT_OUTPUT_ROOT does not exist: ${LIFT_OUTPUT_ROOT}" >&2
    exit 1
fi

echo "[convert-all] root=${LIFT_OUTPUT_ROOT} base_model=${BASE_MODEL} dtype=${DTYPE} dry_run=${DRY_RUN} force=${FORCE}"

# Find every dir that looks like a legacy calib-BTT checkpoint.
# Use -print0 / read -d '' to survive spaces in paths.
mapfile -d '' CKPTS < <(
    find "${LIFT_OUTPUT_ROOT}" \
        -type f -name btt_topology.json \
        -printf '%h\0'
)

if [[ ${#CKPTS[@]} -eq 0 ]]; then
    echo "[convert-all] no legacy calib-BTT checkpoints found under ${LIFT_OUTPUT_ROOT}"
    exit 0
fi

converted=0
skipped=0
failed=0
for ckpt in "${CKPTS[@]}"; do
    # Must also have model.safetensors to be a real checkpoint.
    if [[ ! -f "${ckpt}/model.safetensors" ]]; then
        echo "[skip] ${ckpt} (missing model.safetensors)"
        skipped=$((skipped + 1))
        continue
    fi
    # Already HF-format (e.g. a plain non-calib checkpoint) — skip.
    if [[ -f "${ckpt}/config.json" ]]; then
        echo "[skip] ${ckpt} (already HF-format: config.json present)"
        skipped=$((skipped + 1))
        continue
    fi

    out="${ckpt}/hf"
    if [[ -d "${out}" && -f "${out}/config.json" && "${FORCE}" != "1" ]]; then
        echo "[skip] ${ckpt} (already converted at ${out}; set FORCE=1 to redo)"
        skipped=$((skipped + 1))
        continue
    fi

    echo "[convert] ${ckpt} -> ${out}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        continue
    fi

    if uv run python "${CONVERTER}" \
            --ckpt "${ckpt}" \
            --base-model "${BASE_MODEL}" \
            --out "${out}" \
            --dtype "${DTYPE}"; then
        converted=$((converted + 1))
    else
        echo "[fail] ${ckpt}" >&2
        failed=$((failed + 1))
    fi
done

echo "[convert-all] done: converted=${converted} skipped=${skipped} failed=${failed}"
if [[ ${failed} -gt 0 ]]; then
    exit 2
fi
