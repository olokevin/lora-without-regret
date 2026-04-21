#!/usr/bin/env bash
# tools/run_sft_matrix.sh
#
# Runs 18 short-horizon (300-step) SFT configurations sequentially on one GPU.
# Skips any configuration whose output dir already contains a complete
# sys_metrics.json, so the script is resumable.
#
# Usage: GPU=0 bash tools/run_sft_matrix.sh

set -euo pipefail

GPU="${GPU:-0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIFT_DIR="${REPO_ROOT}/ref/LIFT"
OUT_ROOT="${OUT_ROOT:-/data/yequan/fura/sys_eval/commonsense}"
MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B}"
MAX_STEPS="${MAX_STEPS:-300}"
SEED="${SEED:-43}"
LR="${LR:-2e-4}"

export CUDA_VISIBLE_DEVICES="${GPU}"

mkdir -p "${OUT_ROOT}"

have_metrics() {
    local dir="$1"
    [ -f "${dir}/sys_metrics.json" ]
}

run_one() {
    local label="$1" out_dir="$2"
    shift 2
    if have_metrics "${out_dir}"; then
        echo "[skip] ${label} — sys_metrics.json already present at ${out_dir}"
        return 0
    fi
    echo "[run ] ${label} — ${out_dir}"
    mkdir -p "${out_dir}"
    OUTPUT="${out_dir}" run_name="${label}" "$@" 2>&1 | tee "${out_dir}/matrix.log" || {
        echo "[fail] ${label} — exit $?"
        return 1
    }
}

cd "${LIFT_DIR}"

# 1. Full FT
run_one "full" "${OUT_ROOT}/full" \
    env MODEL="${MODEL}" lr="${LR}" seed="${SEED}" MAX_STEPS="${MAX_STEPS}" \
    bash bash_scripts/finetune_commonsense_full.sh

# 2. FuRA (BlockTT, default corner)
run_one "fura" "${OUT_ROOT}/fura" \
    env MODEL="${MODEL}" lr="${LR}" seed="${SEED}" MAX_STEPS="${MAX_STEPS}" \
        decomp_mode=input_one_block train_position=small s_merged_to=frozen blocktt_rank=full \
    bash bash_scripts/finetune_commonsense_blocktt.sh

# 3. LoRA rank sweep
for r in 16 32 64 128; do
    alpha=$((2 * r))
    run_one "lora-r${r}" "${OUT_ROOT}/lora/r${r}" \
        env MODEL="${MODEL}" adapter_name=lora lora_r="${r}" lora_alpha="${alpha}" \
            lr="${LR}" seed="${SEED}" MAX_STEPS="${MAX_STEPS}" \
        bash bash_scripts/finetune_commonsense_lora.sh
done

# 4. DoRA rank sweep (reuse LoRA script with adapter_name=dora)
for r in 16 32 64 128; do
    alpha=$((2 * r))
    run_one "dora-r${r}" "${OUT_ROOT}/dora/r${r}" \
        env MODEL="${MODEL}" adapter_name=dora lora_r="${r}" lora_alpha="${alpha}" \
            lr="${LR}" seed="${SEED}" MAX_STEPS="${MAX_STEPS}" \
        bash bash_scripts/finetune_commonsense_lora.sh
done

# 5. RandLoRA rank sweep (alpha = 20r per spec)
for r in 16 32 64 128; do
    alpha=$((20 * r))
    run_one "randlora-r${r}" "${OUT_ROOT}/randlora/r${r}" \
        env MODEL="${MODEL}" adapter_name=randlora lora_r="${r}" lora_alpha="${alpha}" \
            lr="${LR}" seed="${SEED}" MAX_STEPS="${MAX_STEPS}" \
        bash bash_scripts/finetune_commonsense_randlora.sh
done

# 6. LIFT rank sweep
for r in 16 32 64 128; do
    run_one "lift-r${r}" "${OUT_ROOT}/lift/r${r}" \
        env MODEL="${MODEL}" lora_rank="${r}" filter_rank="${r}" \
            lr="${LR}" seed="${SEED}" MAX_STEPS="${MAX_STEPS}" \
        bash bash_scripts/finetune_commonsense_lift.sh
done

echo "== matrix complete =="
ls -R "${OUT_ROOT}"
