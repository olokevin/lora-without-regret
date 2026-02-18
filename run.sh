############ SFT ############

### LR search
CUDA_DEVICE=2 TRAIN_MODE=blocktt DECOMP_MODE=input_one_block ./run_lr_search.sh >/dev/null 2>&1 &
CUDA_DEVICE=3 TRAIN_MODE=blocktt DECOMP_MODE=output_one_block ./run_lr_search.sh >/dev/null 2>&1 &

### Full
CUDA_VISIBLE_DEVICES=3 uv run run_sft.py \
  --train-mode full \
  --model-id Qwen/Qwen3-4B \
  --lr 3e-5 \
  --output-dir ./sft/full_model \
  --wandb-project qwen3-4B-SFT \
  --wandb-run-name full_lr_3e-5

### LoRA
CUDA_VISIBLE_DEVICES=3 uv run run_sft.py \
  --train-mode lora \
  --model-id Qwen/Qwen3-4B \
  --lr 2e-4 \
  --lora-rank 64 \
  --lora-type all \
  --output-dir ./sft/lora_model \
  --wandb-project qwen3-4B-SFT \
  --wandb-run-name lora_lr_2e-4

### BTT
CUDA_VISIBLE_DEVICES=2 uv run run_sft.py \
  --train-mode blocktt \
  --model-id Qwen/Qwen3-4B \
  --lr 2e-4 \
  --blocktt-type all \
  --decomp-mode input_one_block \
  --output-dir ./sft/blocktt_model \
  --wandb-project qwen3-4B-SFT \
  --wandb-run-name btt_lr_2e-4

# --disable_wandb

############ RL ############

# this script requires a vllm instance to be run on the same node with --enable-lora flag in another terminal
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-1.7B --enable-lora

# then run this training script after the vllm instance is set up
CUDA_VISIBLE_DEVICES=2 uv run rl_lora.py \
  --lr 1e-4 \
  --lora_r 1 \
  --model_id Qwen/Qwen3-1.7B \
  --wandb-project qwen3-1_7B-RL \
  --wandb-run-name lora_rank-1_lr_1e-4




uv run rl_blocktt.py --lr 1e-4 --blocktt-type all --decomp-mode input_one_block
