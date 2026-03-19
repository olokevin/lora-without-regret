############ RL ############

# New factorized-training config notes (svd / blocktt):
# --trainable-type: all | mlp | attn
# --s-merged-to: frozen | trainable | output | input | split
# SVD defaults:     --train-position output, --s-merged-to frozen
# BlockTT defaults: --train-position small,  --s-merged-to frozen
# BlockTT special:  --train-position both => default --s-merged-to split
# BlockTT constraint: with --train-position both, --s-merged-to frozen/trainable is invalid.
# BlockTT side map: output -> btt_l, input -> btt_r
# Fine-grained BlockTT decomp example:
# DECOMP_MODE='{qkv:input,o:output,mlp_upgate:output,mlp_down:output}'

# For LORA: 
# - this script requires a vllm instance to be run on the same node with --enable-lora flag
# - in another terminal, install vllm and then run the following commands
# ```
# export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
# source .venv/bin/activate
# CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-1.7B --enable-lora --max-lora-rank 64
# ```
# - otherwise it falls back to local in-process vLLM rollout (no vllm serve)


run_full()
{
  local train_mode="full"
  local lr="${LR:-1e-5}"
  local optimizer="${OPTIMIZER:-adamw}"
  local run_name="${train_mode}-${optimizer}-lr_${lr}"
  local device="${DEVICE:-2}"

  CUDA_VISIBLE_DEVICES="$device" uv run run_rl.py \
    --train-mode "$train_mode" \
    --lr "$lr" \
    --optimizer "$optimizer" \
    --model-id Qwen/Qwen3-1.7B \
    --wandb-project qwen3-1_7B-RL \
    --wandb-run-name "$run_name"
}

run_lora()
{
  local train_mode="lora"
  local lr="${LR:-2e-4}"
  local optimizer="${OPTIMIZER:-adamw}"
  local lora_rank="${LORA_RANK:-64}"
  local trainable_type="${TRAINABLE_TYPE:-all}"
  local name_suffix="${NAME_SUFFIX:-}"
  local run_name="${train_mode}-${optimizer}-lr_${lr}-rank_${lora_rank}${name_suffix}"
  local device="${DEVICE:-2}"
  local vllm_url="${VLLM_URL:-http://localhost:8000}"
  local -a cfg_suffix_args=()
  if [[ -n "${CFG_SUFFIX:-}" ]]; then
    # Intended for trusted local overrides, e.g. CFG_SUFFIX="--flag --arg value".
    read -r -a cfg_suffix_args <<< "${CFG_SUFFIX}"
  fi

  CUDA_VISIBLE_DEVICES="$device" uv run run_rl.py \
    --train-mode "$train_mode" \
    --lr "$lr" \
    --optimizer "$optimizer" \
    --lora-rank "$lora_rank" \
    --trainable-type "$trainable_type" \
    --vllm-url "$vllm_url" \
    --model-id Qwen/Qwen3-1.7B \
    --wandb-project qwen3-1_7B-RL \
    --wandb-run-name "$run_name" \
    "${cfg_suffix_args[@]}"
}

run_svd()
{
  local train_mode="svd"
  local lr="${LR:-8e-5}"
  local optimizer="${OPTIMIZER:-adamw}"
  local train_position="${1:-${TRAIN_POSITION:-output}}"
  local s_merged_to="${2:-${S_MERGED_TO:-frozen}}"
  local device="${DEVICE:-2}"
  local name_suffix="${NAME_SUFFIX:-}"
  local run_name="${train_mode}-${optimizer}-lr_${lr}-s_to_${s_merged_to}-train_${train_position}${name_suffix}"
  local -a cfg_suffix_args=()
  if [[ -n "${CFG_SUFFIX:-}" ]]; then
    # Intended for trusted local overrides, e.g. CFG_SUFFIX="--flag --arg value".
    read -r -a cfg_suffix_args <<< "${CFG_SUFFIX}"
  fi

  if [[ "$train_position" != "output" && "$train_position" != "input" ]]; then
    echo "Invalid SVD train position: $train_position (expected: output|input)"
    return 1
  fi

  CUDA_VISIBLE_DEVICES="$device" uv run run_rl.py \
    --train-mode "$train_mode" \
    --lr "$lr" \
    --optimizer "$optimizer" \
    --trainable-type all \
    --train-position "$train_position" \
    --s-merged-to "$s_merged_to" \
    --model-id Qwen/Qwen3-1.7B \
    --wandb-project qwen3-1_7B-RL \
    --wandb-run-name "$run_name" \
    "${cfg_suffix_args[@]}"
}

run_blocktt()
{
  local train_mode="blocktt"
  local lr="${LR:-8e-5}"
  local optimizer="${OPTIMIZER:-adamw}"
  local decomp_mode="${DECOMP_MODE:-input_one_block}"
  local train_position="${TRAIN_POSITION:-small}"
  local s_merged_to="${S_MERGED_TO:-frozen}"
  local device="${DEVICE:-2}"
  local name_suffix="${NAME_SUFFIX:-}"
  local run_name="${train_mode}-${optimizer}-lr_${lr}-${decomp_mode}-s_to_${s_merged_to}-train_${train_position}${name_suffix}"
  local -a cfg_suffix_args=()
  if [[ -n "${CFG_SUFFIX:-}" ]]; then
    # Intended for trusted local overrides, e.g. CFG_SUFFIX="--flag --arg value".
    read -r -a cfg_suffix_args <<< "${CFG_SUFFIX}"
  fi

  if [[ "$train_position" != "small" && "$train_position" != "large" && "$train_position" != "both" ]]; then
    echo "Invalid BlockTT train position: $train_position (expected: small|large|both)"
    return 1
  fi

  CUDA_VISIBLE_DEVICES="$device" uv run run_rl.py \
    --train-mode "$train_mode" \
    --lr "$lr" \
    --optimizer "$optimizer" \
    --trainable-type all \
    --decomp-mode "$decomp_mode" \
    --s-merged-to "$s_merged_to" \
    --train-position "$train_position" \
    --model-id Qwen/Qwen3-1.7B \
    --wandb-project qwen3-1_7B-RL \
    --wandb-run-name "$run_name" \
    "${cfg_suffix_args[@]}"
}

run_blocktt_muon()
{
  local train_mode="blocktt"
  local lr="${LR:-8e-5}"
  local optimizer="muon"
  local decomp_mode="${DECOMP_MODE:-input_one_block}"
  local train_position="${TRAIN_POSITION:-small}"
  local s_merged_to="${S_MERGED_TO:-frozen}"
  local device="${DEVICE:-2}"
  local name_suffix="${NAME_SUFFIX:-}"
  local run_name="${train_mode}-${optimizer}-lr_${lr}-${decomp_mode}-s_to_${s_merged_to}-train_${train_position}-warmup_0.1-minlr_0.01${name_suffix}"
  local -a cfg_suffix_args=()
  if [[ -n "${CFG_SUFFIX:-}" ]]; then
    # Intended for trusted local overrides, e.g. CFG_SUFFIX="--flag --arg value".
    read -r -a cfg_suffix_args <<< "${CFG_SUFFIX}"
  fi

  if [[ "$train_position" != "small" && "$train_position" != "large" && "$train_position" != "both" ]]; then
    echo "Invalid BlockTT train position: $train_position (expected: small|large|both)"
    return 1
  fi

  CUDA_VISIBLE_DEVICES="$device" uv run run_rl.py \
    --train-mode "$train_mode" \
    --lr "$lr" \
    --optimizer "$optimizer" \
    --trainable-type all \
    --decomp-mode "$decomp_mode" \
    --s-merged-to "$s_merged_to" \
    --train-position "$train_position" \
    --warmup-ratio 0.1 \
    --min-lr-ratio 0.01 \
    --model-id Qwen/Qwen3-1.7B \
    --wandb-project qwen3-1_7B-RL \
    --wandb-run-name "$run_name" \
    "${cfg_suffix_args[@]}"
}

### lora/blocktt does not update embedding / lm_head

run_sequential()
{
  ### baseline
  LR=3e-5 run_full
  LR=5e-5 run_full

  # LR=1e-5 run_full
  # LR=1e-5 run_full
  # LR=8e-5 LORA_RANK=64 run_lora

  ### input one block ablation
  # LR=1e-5 DECOMP_MODE=input_one_block TRAIN_POSITION=both S_MERGED_TO=split run_blocktt
  # LR=8e-5 DECOMP_MODE=input_one_block TRAIN_POSITION=small S_MERGED_TO=frozen run_blocktt
  # LR=8e-5 DECOMP_MODE=input_one_block TRAIN_POSITION=small S_MERGED_TO=trainable run_blocktt

  ### output one block ablation
  # LR=8e-5 DECOMP_MODE=output_one_block TRAIN_POSITION=small S_MERGED_TO=frozen run_blocktt
  # LR=1e-5 DECOMP_MODE=output_one_block TRAIN_POSITION=both S_MERGED_TO=split run_blocktt
  # LR=1e-5 DECOMP_MODE=output_one_block TRAIN_POSITION=both S_MERGED_TO=keep run_blocktt

  # LR=1e-4 DECOMP_MODE=output_one_block TRAIN_POSITION=small S_MERGED_TO=frozen run_blocktt
  # LR=2e-4 DECOMP_MODE=output_one_block TRAIN_POSITION=small S_MERGED_TO=frozen run_blocktt

  # LR=6e-5 DECOMP_MODE=output_one_block TRAIN_POSITION=small S_MERGED_TO=frozen run_blocktt
  # LR=5e-5 DECOMP_MODE=output_one_block TRAIN_POSITION=small S_MERGED_TO=frozen run_blocktt

  ### fine-grained block ablation
  # LR=8e-5 DECOMP_MODE='{qkv:input,o:output,mlp_upgate:output,mlp_down:output}'  TRAIN_POSITION=small S_MERGED_TO=frozen run_blocktt
  # LR=1e-5 DECOMP_MODE='{qkv:input,o:output,mlp_upgate:output,mlp_down:output}'  TRAIN_POSITION=both S_MERGED_TO=keep run_blocktt

  ### muon ablation
  # LR=1e-4 OPTIMIZER=muon DECOMP_MODE=input_one_block TRAIN_POSITION=both S_MERGED_TO=keep run_blocktt
  # LR=1e-3 OPTIMIZER=muon DECOMP_MODE=input_one_block TRAIN_POSITION=both S_MERGED_TO=keep run_blocktt
  # LR=1e-3 OPTIMIZER=muon DECOMP_MODE=input_one_block TRAIN_POSITION=small S_MERGED_TO=frozen run_blocktt
  # LR=1e-2 OPTIMIZER=muon DECOMP_MODE=input_one_block TRAIN_POSITION=small S_MERGED_TO=frozen run_blocktt

  ### muon lr decay
  # LR=1e-3 OPTIMIZER=muon DECOMP_MODE=input_one_block TRAIN_POSITION=small S_MERGED_TO=frozen run_blocktt_muon
  # LR=5e-4 OPTIMIZER=muon DECOMP_MODE=input_one_block TRAIN_POSITION=both S_MERGED_TO=keep run_blocktt_muon

  ### normalize after update
  # LR=8e-5 DECOMP_MODE=input_one_block TRAIN_POSITION=small S_MERGED_TO=frozen \
  # CFG_SUFFIX="--blocktt-normalize-after-update" NAME_SUFFIX="-norm" run_blocktt
  # LR=1e-3 OPTIMIZER=muon DECOMP_MODE=input_one_block TRAIN_POSITION=small S_MERGED_TO=frozen \
  # CFG_SUFFIX="--blocktt-normalize-after-update" NAME_SUFFIX="-norm" run_blocktt_muon

  ### RECORD CKPT
  # LR=1e-5 TRAIN_POSITION=input S_MERGED_TO=keep CFG_SUFFIX="--enable-save-ckpt" run_svd
  # LR=1e-4 DECOMP_MODE=output_one_block TRAIN_POSITION=small S_MERGED_TO=frozen CFG_SUFFIX="--enable-save-ckpt" run_blocktt
  # LR=1e-5 DECOMP_MODE=output_one_block TRAIN_POSITION=both S_MERGED_TO=keep CFG_SUFFIX="--enable-save-ckpt" run_blocktt


}


if [[ "$TRAIN_MODE" == "full" ]]; then
    run_full
elif [[ "$TRAIN_MODE" == "lora" ]]; then
    run_lora
elif [[ "$TRAIN_MODE" == "svd" ]]; then
    run_svd
elif [[ "$TRAIN_MODE" == "blocktt" ]]; then
    run_blocktt
elif [[ "$TRAIN_MODE" == "blocktt_muon" ]]; then
    run_blocktt_muon
elif [[ "$TRAIN_MODE" == "sequential" ]]; then
    run_sequential
else
    echo "Unsupported train mode: $TRAIN_MODE"
    echo "Use TRAIN_MODE=full|lora|svd|blocktt|sequential"
    exit 1
fi


### shell scripts
# DEVICE=4 LR=1e-5 TRAIN_MODE=full bash run_rl.sh >/dev/null 2>&1 &
# DEVICE=0 LR=8e-5 TRAIN_MODE=lora LORA_RANK=16 bash run_rl.sh >/dev/null 2>&1 &
# DEVICE=1 LR=8e-5 TRAIN_MODE=svd bash run_rl.sh >/dev/null 2>&1 &
# DEVICE=6 LR=8e-5 TRAIN_MODE=blocktt bash run_rl.sh >/dev/null 2>&1 &

# DEVICE=1 TRAIN_MODE=sequential bash run_rl.sh >logs/ckpt_runs.log 2>&1 &

# DECOMP_MODE='{qkv:input,o:output,mlp_upgate:output,mlp_down:output}' \
# CFG_SUFFIX="--enable-save-ckpt"

# DEVICE=1 LR=2e-5 TRAIN_MODE=full CFG_SUFFIX="--enable-save-ckpt" bash run_rl.sh >/dev/null 2>&1 &
# DEVICE=1 LR=1e-5 TRAIN_MODE=svd TRAIN_POSITION=input S_MERGED_TO=keep CFG_SUFFIX="--enable-save-ckpt" bash run_rl.sh >/dev/null 2>&1 &
# DEVICE=2 LR=1e-4 TRAIN_MODE=blocktt DECOMP_MODE=output_one_block TRAIN_POSITION=small S_MERGED_TO=frozen CFG_SUFFIX="--enable-save-ckpt" bash run_rl.sh >/dev/null 2>&1 &
# DEVICE=3 LR=1e-5 TRAIN_MODE=blocktt DECOMP_MODE=output_one_block TRAIN_POSITION=both S_MERGED_TO=keep CFG_SUFFIX="--enable-save-ckpt" bash run_rl.sh >/dev/null 2>&1 &

