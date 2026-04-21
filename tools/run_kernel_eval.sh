#!/usr/bin/env bash
# tools/run_kernel_eval.sh
# One command: tests -> microbench -> end-to-end SFT -> auto-report.
#
# Usage:  GPU=0 bash tools/run_kernel_eval.sh
set -euo pipefail

GPU="${GPU:-0}"
OUT_ROOT="${OUT_ROOT:-/data/yequan/fura/sys_eval/kernel}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT="${REPORT:-docs/26_nips_fura_paper/kernel_eval_report.md}"

mkdir -p "${OUT_ROOT}"

# Guard against GPU contention (fail fast)
if nvidia-smi --query-compute-apps=pid -i "${GPU}" 2>/dev/null | grep -q '[0-9]'; then
    echo "[abort] GPU ${GPU} has running compute apps. Pick a free GPU." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU}"

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${GPU}" 2>/dev/null || echo unknown)"

cd "${REPO_ROOT}"

# 1. correctness
echo "== correctness tests =="
uv run pytest -q tests/test_fura_fused_kernel.py tests/test_btt_layer_fused_flag.py \
    | tee "${OUT_ROOT}/test.log"

# 2. microbench
echo "== microbench =="
uv run python tools/bench_fused_kernel.py \
    --out "${OUT_ROOT}/fused_kernel_micro.json"

# 3. end-to-end SFT (300-step cap)
echo "== end-to-end SFT =="
uv run python tools/bench_fused_kernel_sft.py \
    --out_root "${OUT_ROOT}/sft" --max_steps 300

# 4. report
echo "== report =="
uv run python tools/write_kernel_report.py \
    --micro "${OUT_ROOT}/fused_kernel_micro.json" \
    --sft_baseline "${OUT_ROOT}/sft/baseline/sys_metrics.json" \
    --sft_fused    "${OUT_ROOT}/sft/fused/sys_metrics.json" \
    --gpu_name "${GPU_NAME}" \
    --out "${REPORT}"

echo "Report written to ${REPORT}"
