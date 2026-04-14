############ KD ############

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
  local kd_loss_type="${KD_LOSS_TYPE:-sft}"
  local student_model_id="${STUDENT_MODEL_ID:-Qwen/Qwen2.5-0.5B}"
  local teacher_data_dir="${TEACHER_DATA_DIR:-/data/yequan/fura/kd_data/DeepSeek-R1-Distill-Qwen-7B-competition_math}"
  local lr="${LR:-5e-6}"
  local optimizer="${OPTIMIZER:-adamw}"
  local name_suffix="${NAME_SUFFIX:-}"
  local run_name="${train_mode}-${kd_loss_type}-${optimizer}-lr_${lr}${name_suffix}"
  local device="${DEVICE:-2}"
  local -a cfg_suffix_args=()
  if [[ -n "${CFG_SUFFIX:-}" ]]; then
    read -r -a cfg_suffix_args <<< "${CFG_SUFFIX}"
  fi

  CUDA_VISIBLE_DEVICES="$device" uv run run_kd.py \
    --kd-loss-type "$kd_loss_type" \
    --train-mode "$train_mode" \
    --student-model-id "$student_model_id" \
    --teacher-data-dir "$teacher_data_dir" \
    --lr "$lr" \
    --optimizer "$optimizer" \
    --wandb-project qwen2_5-0_5B-KD \
    --wandb-run-name "$run_name" \
    "${cfg_suffix_args[@]}"
}

run_lora()
{
  local train_mode="lora"
  local kd_loss_type="${KD_LOSS_TYPE:-sft}"
  local student_model_id="${STUDENT_MODEL_ID:-Qwen/Qwen2.5-0.5B}"
  local teacher_data_dir="${TEACHER_DATA_DIR:-/data/yequan/fura/kd_data/DeepSeek-R1-Distill-Qwen-7B-competition_math}"
  local lr="${LR:-1e-4}"
  local optimizer="${OPTIMIZER:-adamw}"
  local lora_rank="${LORA_RANK:-128}"
  local trainable_type="${TRAINABLE_TYPE:-all}"
  local name_suffix="${NAME_SUFFIX:-}"
  local run_name="${train_mode}-${kd_loss_type}-${optimizer}-lr_${lr}-rank_${lora_rank}${name_suffix}"
  local device="${DEVICE:-2}"
  local -a cfg_suffix_args=()
  if [[ -n "${CFG_SUFFIX:-}" ]]; then
    read -r -a cfg_suffix_args <<< "${CFG_SUFFIX}"
  fi

  CUDA_VISIBLE_DEVICES="$device" uv run run_kd.py \
    --kd-loss-type "$kd_loss_type" \
    --train-mode "$train_mode" \
    --student-model-id "$student_model_id" \
    --teacher-data-dir "$teacher_data_dir" \
    --lr "$lr" \
    --optimizer "$optimizer" \
    --lora-rank "$lora_rank" \
    --trainable-type "$trainable_type" \
    --wandb-project qwen2_5-0_5B-KD \
    --wandb-run-name "$run_name" \
    "${cfg_suffix_args[@]}"
}

run_svd()
{
  local train_mode="svd"
  local kd_loss_type="${KD_LOSS_TYPE:-sft}"
  local student_model_id="${STUDENT_MODEL_ID:-Qwen/Qwen2.5-0.5B}"
  local teacher_data_dir="${TEACHER_DATA_DIR:-/data/yequan/fura/kd_data/DeepSeek-R1-Distill-Qwen-7B-competition_math}"
  local lr="${LR:-1e-4}"
  local optimizer="${OPTIMIZER:-adamw}"
  local train_position="${TRAIN_POSITION:-output}"
  local s_merged_to="${S_MERGED_TO:-frozen}"
  local name_suffix="${NAME_SUFFIX:-}"
  local run_name="${train_mode}-${kd_loss_type}-${optimizer}-lr_${lr}-s_to_${s_merged_to}-train_${train_position}${name_suffix}"
  local device="${DEVICE:-2}"
  local -a cfg_suffix_args=()
  if [[ -n "${CFG_SUFFIX:-}" ]]; then
    read -r -a cfg_suffix_args <<< "${CFG_SUFFIX}"
  fi

  if [[ "$train_position" != "output" && "$train_position" != "input" ]]; then
    echo "Invalid SVD train position: $train_position (expected: output|input)"
    return 1
  fi

  CUDA_VISIBLE_DEVICES="$device" uv run run_kd.py \
    --kd-loss-type "$kd_loss_type" \
    --train-mode "$train_mode" \
    --student-model-id "$student_model_id" \
    --teacher-data-dir "$teacher_data_dir" \
    --lr "$lr" \
    --optimizer "$optimizer" \
    --trainable-type all \
    --train-position "$train_position" \
    --s-merged-to "$s_merged_to" \
    --wandb-project qwen2_5-0_5B-KD \
    --wandb-run-name "$run_name" \
    "${cfg_suffix_args[@]}"
}

run_blocktt()
{
  local train_mode="blocktt"
  local kd_loss_type="${KD_LOSS_TYPE:-sft}"
  local student_model_id="${STUDENT_MODEL_ID:-Qwen/Qwen2.5-0.5B}"
  local teacher_data_dir="${TEACHER_DATA_DIR:-/data/yequan/fura/kd_data/DeepSeek-R1-Distill-Qwen-7B-competition_math}"
  local lr="${LR:-1e-4}"
  local optimizer="${OPTIMIZER:-adamw}"
  local decomp_mode="${DECOMP_MODE:-input_one_block}"
  local train_position="${TRAIN_POSITION:-small}"
  local s_merged_to="${S_MERGED_TO:-frozen}"
  local name_suffix="${NAME_SUFFIX:-}"
  local run_name="${train_mode}-${kd_loss_type}-${optimizer}-lr_${lr}-${decomp_mode}-s_to_${s_merged_to}-train_${train_position}${name_suffix}"
  local device="${DEVICE:-2}"
  local -a cfg_suffix_args=()
  if [[ -n "${CFG_SUFFIX:-}" ]]; then
    read -r -a cfg_suffix_args <<< "${CFG_SUFFIX}"
  fi

  if [[ "$train_position" != "small" && "$train_position" != "large" && "$train_position" != "both" ]]; then
    echo "Invalid BlockTT train position: $train_position (expected: small|large|both)"
    return 1
  fi

  CUDA_VISIBLE_DEVICES="$device" uv run run_kd.py \
    --kd-loss-type "$kd_loss_type" \
    --train-mode "$train_mode" \
    --student-model-id "$student_model_id" \
    --teacher-data-dir "$teacher_data_dir" \
    --lr "$lr" \
    --optimizer "$optimizer" \
    --trainable-type all \
    --decomp-mode "$decomp_mode" \
    --s-merged-to "$s_merged_to" \
    --train-position "$train_position" \
    --wandb-project qwen2_5-0_5B-KD \
    --wandb-run-name "$run_name" \
    "${cfg_suffix_args[@]}"
}

run_blocktt_muon()
{
  local train_mode="blocktt"
  local kd_loss_type="${KD_LOSS_TYPE:-sft}"
  local student_model_id="${STUDENT_MODEL_ID:-Qwen/Qwen2.5-0.5B}"
  local teacher_data_dir="${TEACHER_DATA_DIR:-/data/yequan/fura/kd_data/DeepSeek-R1-Distill-Qwen-7B-competition_math}"
  local lr="${LR:-1e-4}"
  local optimizer="muon"
  local decomp_mode="${DECOMP_MODE:-input_one_block}"
  local train_position="${TRAIN_POSITION:-small}"
  local s_merged_to="${S_MERGED_TO:-frozen}"
  local name_suffix="${NAME_SUFFIX:-}"
  local run_name="${train_mode}-${kd_loss_type}-${optimizer}-lr_${lr}-${decomp_mode}-s_to_${s_merged_to}-train_${train_position}-warmup_0.1-minlr_0.01${name_suffix}"
  local device="${DEVICE:-2}"
  local -a cfg_suffix_args=()
  if [[ -n "${CFG_SUFFIX:-}" ]]; then
    read -r -a cfg_suffix_args <<< "${CFG_SUFFIX}"
  fi

  if [[ "$train_position" != "small" && "$train_position" != "large" && "$train_position" != "both" ]]; then
    echo "Invalid BlockTT train position: $train_position (expected: small|large|both)"
    return 1
  fi

  CUDA_VISIBLE_DEVICES="$device" uv run run_kd.py \
    --kd-loss-type "$kd_loss_type" \
    --train-mode "$train_mode" \
    --student-model-id "$student_model_id" \
    --teacher-data-dir "$teacher_data_dir" \
    --lr "$lr" \
    --optimizer "$optimizer" \
    --trainable-type all \
    --decomp-mode "$decomp_mode" \
    --s-merged-to "$s_merged_to" \
    --train-position "$train_position" \
    --warmup-ratio 0.1 \
    --min-lr-ratio 0.01 \
    --wandb-project qwen2_5-0_5B-KD \
    --wandb-run-name "$run_name" \
    "${cfg_suffix_args[@]}"
}

run_sequential()
{
  :
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
# DEVICE=3 LR=5e-6 TRAIN_MODE=full KD_LOSS_TYPE=sft bash run_kd.sh >/dev/null 2>&1 &
# DEVICE=3 LR=1e-4 TRAIN_MODE=lora KD_LOSS_TYPE=kl LORA_RANK=128 bash run_kd.sh >/dev/null 2>&1 &
# DEVICE=2 LR=1e-4 TRAIN_MODE=svd KD_LOSS_TYPE=sft TRAIN_POSITION=output bash run_kd.sh >/dev/null 2>&1 &
# DEVICE=2 LR=1e-4 TRAIN_MODE=blocktt KD_LOSS_TYPE=sft DECOMP_MODE=input_one_block TRAIN_POSITION=small bash run_kd.sh >/dev/null 2>&1 &

# CFG_SUFFIX examples:
# CFG_SUFFIX="--no-wandb"
# CFG_SUFFIX="--enable-save-ckpt --top-k 128"
# CFG_SUFFIX="--enable-save-ckpt --save-grads-steps 0,10,30"
