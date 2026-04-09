############ SFT ############

# New factorized-training config notes (svd / blocktt):
# --trainable-type: all | mlp | attn
# --s-merged-to: frozen | trainable | output | input | split
# SVD defaults:     --train-position output, --s-merged-to frozen
# BlockTT defaults: --train-position small,  --s-merged-to frozen
# BlockTT special:  --train-position both => default --s-merged-to split
# BlockTT constraint: with --train-position both, --s-merged-to frozen/trainable is invalid.
# BlockTT side map: output -> btt_l, input -> btt_r

run_full()
{
  local train_mode="full"
  local model_id="${MODEL_ID:-Qwen/Qwen3-4B}"
  local lr="${LR:-3e-5}"
  local optimizer="${OPTIMIZER:-adamw}"
  local base_dir="${BASE_DIR:-/data/yequan/fura/sft_runs}"
  local wandb_project="${WANDB_PROJECT:-qwen3-4B-SFT}"
  local name_suffix="${NAME_SUFFIX:-}"
  local run_name="${train_mode}-${optimizer}-lr_${lr}${name_suffix}"
  local device="${DEVICE:-2}"
  local -a cfg_suffix_args=()
  if [[ -n "${CFG_SUFFIX:-}" ]]; then
    # Intended for trusted local overrides, e.g. CFG_SUFFIX="--flag --arg value".
    read -r -a cfg_suffix_args <<< "${CFG_SUFFIX}"
  fi

  CUDA_VISIBLE_DEVICES="$device" uv run run_sft.py \
    --train-mode "$train_mode" \
    --model-id "$model_id" \
    --lr "$lr" \
    --optimizer "$optimizer" \
    --base-dir "$base_dir" \
    --wandb-project "$wandb_project" \
    --wandb-run-name "$run_name" \
    "${cfg_suffix_args[@]}"
}

run_lora()
{
  local train_mode="lora"
  local model_id="${MODEL_ID:-Qwen/Qwen3-4B}"
  local lr="${LR:-2e-4}"
  local optimizer="${OPTIMIZER:-adamw}"
  local lora_rank="${LORA_RANK:-64}"
  local trainable_type="${TRAINABLE_TYPE:-all}"
  local base_dir="${BASE_DIR:-/data/yequan/fura/sft_runs}"
  local wandb_project="${WANDB_PROJECT:-qwen3-4B-SFT}"
  local name_suffix="${NAME_SUFFIX:-}"
  local run_name="${train_mode}-${optimizer}-lr_${lr}-rank_${lora_rank}${name_suffix}"
  local device="${DEVICE:-2}"
  local -a cfg_suffix_args=()
  if [[ -n "${CFG_SUFFIX:-}" ]]; then
    # Intended for trusted local overrides, e.g. CFG_SUFFIX="--flag --arg value".
    read -r -a cfg_suffix_args <<< "${CFG_SUFFIX}"
  fi

  CUDA_VISIBLE_DEVICES="$device" uv run run_sft.py \
    --train-mode "$train_mode" \
    --model-id "$model_id" \
    --lr "$lr" \
    --optimizer "$optimizer" \
    --lora-rank "$lora_rank" \
    --trainable-type "$trainable_type" \
    --base-dir "$base_dir" \
    --wandb-project "$wandb_project" \
    --wandb-run-name "$run_name" \
    "${cfg_suffix_args[@]}"
}

run_svd()
{
  local train_mode="svd"
  local model_id="${MODEL_ID:-Qwen/Qwen3-4B}"
  local lr="${LR:-2e-4}"
  local optimizer="${OPTIMIZER:-adamw}"
  local trainable_type="${TRAINABLE_TYPE:-all}"
  local train_position="${TRAIN_POSITION:-output}"
  local s_merged_to="${S_MERGED_TO:-frozen}"
  local base_dir="${BASE_DIR:-/data/yequan/fura/sft_runs}"
  local wandb_project="${WANDB_PROJECT:-qwen3-4B-SFT}"
  local name_suffix="${NAME_SUFFIX:-}"
  local run_name="${train_mode}-${optimizer}-lr_${lr}-s_to_${s_merged_to}-train_${train_position}${name_suffix}"
  local device="${DEVICE:-2}"
  local -a cfg_suffix_args=()
  if [[ -n "${CFG_SUFFIX:-}" ]]; then
    # Intended for trusted local overrides, e.g. CFG_SUFFIX="--flag --arg value".
    read -r -a cfg_suffix_args <<< "${CFG_SUFFIX}"
  fi

  if [[ "$train_position" != "output" && "$train_position" != "input" ]]; then
    echo "Invalid SVD train position: $train_position (expected: output|input)"
    return 1
  fi

  CUDA_VISIBLE_DEVICES="$device" uv run run_sft.py \
    --train-mode "$train_mode" \
    --model-id "$model_id" \
    --lr "$lr" \
    --optimizer "$optimizer" \
    --trainable-type "$trainable_type" \
    --train-position "$train_position" \
    --s-merged-to "$s_merged_to" \
    --base-dir "$base_dir" \
    --wandb-project "$wandb_project" \
    --wandb-run-name "$run_name" \
    "${cfg_suffix_args[@]}"
}

run_blocktt()
{
  local train_mode="blocktt"
  local model_id="${MODEL_ID:-Qwen/Qwen3-4B}"
  local lr="${LR:-2e-4}"
  local optimizer="${OPTIMIZER:-adamw}"
  local trainable_type="${TRAINABLE_TYPE:-all}"
  local decomp_mode="${DECOMP_MODE:-input_one_block}"
  local train_position="${TRAIN_POSITION:-small}"
  local s_merged_to="${S_MERGED_TO:-frozen}"
  local base_dir="${BASE_DIR:-/data/yequan/fura/sft_runs}"
  local wandb_project="${WANDB_PROJECT:-qwen3-4B-SFT}"
  local name_suffix="${NAME_SUFFIX:-}"
  local run_name="${train_mode}-${optimizer}-lr_${lr}-${decomp_mode}-s_to_${s_merged_to}-train_${train_position}${name_suffix}"
  local device="${DEVICE:-2}"
  local -a cfg_suffix_args=()
  if [[ -n "${CFG_SUFFIX:-}" ]]; then
    # Intended for trusted local overrides, e.g. CFG_SUFFIX="--flag --arg value".
    read -r -a cfg_suffix_args <<< "${CFG_SUFFIX}"
  fi

  if [[ "$train_position" != "small" && "$train_position" != "large" && "$train_position" != "both" ]]; then
    echo "Invalid BlockTT train position: $train_position (expected: small|large|both)"
    return 1
  fi

  CUDA_VISIBLE_DEVICES="$device" uv run run_sft.py \
    --train-mode "$train_mode" \
    --model-id "$model_id" \
    --lr "$lr" \
    --optimizer "$optimizer" \
    --trainable-type "$trainable_type" \
    --decomp-mode "$decomp_mode" \
    --train-position "$train_position" \
    --s-merged-to "$s_merged_to" \
    --base-dir "$base_dir" \
    --wandb-project "$wandb_project" \
    --wandb-run-name "$run_name" \
    "${cfg_suffix_args[@]}"
}

run_blocktt_muon()
{
  local train_mode="blocktt"
  local model_id="${MODEL_ID:-Qwen/Qwen3-4B}"
  local lr="${LR:-2e-4}"
  local optimizer="muon"
  local trainable_type="${TRAINABLE_TYPE:-all}"
  local decomp_mode="${DECOMP_MODE:-input_one_block}"
  local train_position="${TRAIN_POSITION:-small}"
  local s_merged_to="${S_MERGED_TO:-frozen}"
  local base_dir="${BASE_DIR:-/data/yequan/fura/sft_runs}"
  local wandb_project="${WANDB_PROJECT:-qwen3-4B-SFT}"
  local name_suffix="${NAME_SUFFIX:-}"
  local run_name="${train_mode}-${optimizer}-lr_${lr}-${decomp_mode}-s_to_${s_merged_to}-train_${train_position}-warmup_0.1-minlr_0.01${name_suffix}"
  local device="${DEVICE:-2}"
  local -a cfg_suffix_args=()
  if [[ -n "${CFG_SUFFIX:-}" ]]; then
    # Intended for trusted local overrides, e.g. CFG_SUFFIX="--flag --arg value".
    read -r -a cfg_suffix_args <<< "${CFG_SUFFIX}"
  fi

  if [[ "$train_position" != "small" && "$train_position" != "large" && "$train_position" != "both" ]]; then
    echo "Invalid BlockTT train position: $train_position (expected: small|large|both)"
    return 1
  fi

  CUDA_VISIBLE_DEVICES="$device" uv run run_sft.py \
    --train-mode "$train_mode" \
    --model-id "$model_id" \
    --lr "$lr" \
    --optimizer "$optimizer" \
    --trainable-type "$trainable_type" \
    --decomp-mode "$decomp_mode" \
    --train-position "$train_position" \
    --s-merged-to "$s_merged_to" \
    --warmup-ratio 0.1 \
    --min-lr-ratio 0.01 \
    --base-dir "$base_dir" \
    --wandb-project "$wandb_project" \
    --wandb-run-name "$run_name" \
    "${cfg_suffix_args[@]}"
}

run_sequential()
{
  ### wider test
  LR=2e-4 DECOMP_MODE=input_one_block TRAIN_POSITION=small S_MERGED_TO=trainable run_blocktt
  LR=4e-4 DECOMP_MODE=input_one_block TRAIN_POSITION=small S_MERGED_TO=trainable run_blocktt

  LR=2e-4 DECOMP_MODE=output_one_block TRAIN_POSITION=small S_MERGED_TO=trainable run_blocktt
  LR=4e-4 DECOMP_MODE=output_one_block TRAIN_POSITION=small S_MERGED_TO=trainable run_blocktt

  # LR=3e-5 TRAIN_POSITION=output S_MERGED_TO=output run_svd
  # LR=3e-5 TRAIN_POSITION=output S_MERGED_TO=input run_svd

  # LR=3e-5 TRAIN_POSITION=input S_MERGED_TO=input run_svd
  # LR=3e-5 TRAIN_POSITION=input S_MERGED_TO=output run_svd
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
    echo "Use TRAIN_MODE=full|lora|svd|blocktt|blocktt_muon|sequential"
    exit 1
fi


### shell scripts
# DEVICE=3 LR=3e-5 TRAIN_MODE=full bash run_sft.sh >/dev/null 2>&1 &
# DEVICE=3 LR=2e-4 TRAIN_MODE=lora LORA_RANK=64 bash run_sft.sh >/dev/null 2>&1 &
# DEVICE=2 LR=2e-4 TRAIN_MODE=svd TRAIN_POSITION=input bash run_sft.sh >/dev/null 2>&1 &
# DEVICE=2 LR=2e-4 TRAIN_MODE=blocktt DECOMP_MODE=output_one_block TRAIN_POSITION=small S_MERGED_TO=frozen CFG_SUFFIX="--enable-save-ckpt --save-grads-steps=0,10,30" bash run_sft.sh >/dev/null 2>&1 &

# DEVICE=7 TRAIN_MODE=sequential bash run_sft.sh >/dev/null 2>&1 &

# CFG_SUFFIX examples:
# CFG_SUFFIX="--no-wandb"
# CFG_SUFFIX="--enable-save-ckpt --lr-scheduler cosine --cycle-length 200 --warmup-ratio 0.1"
# CFG_SUFFIX="--weight-decay 0.0 --lr-adam 3e-4 --lr-embedding 1e-4 --norm-method shape"
