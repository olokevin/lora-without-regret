# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase reproducing experiments from the "LoRA without Regret" blog post (Schulman & Thinking Machines). Compares full fine-tuning, LoRA, BlockTT decomposition, and SVD decomposition for SFT, GRPO-based RL, and knowledge distillation on Qwen3 models.

## Build & Run Commands

```bash
uv sync                                    # install dependencies (Python 3.13+, uses uv)

# SFT (supervised fine-tuning) - runs on single GPU
CUDA_VISIBLE_DEVICES=0 uv run run_sft.py --train-mode lora --lr 2e-4 --lora-rank 1 --trainable-type all --no-wandb

# RL (GRPO reinforcement learning) - requires vLLM server for LoRA mode
# Terminal 1: start vLLM server
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-1.7B --enable-lora --max-lora-rank 64

# Terminal 2: run RL training
CUDA_VISIBLE_DEVICES=0 uv run run_rl.py --train-mode lora --lr 1e-4 --lora-rank 1 --trainable-type all --no-wandb

# Knowledge distillation - first generate teacher data, then train
CUDA_VISIBLE_DEVICES=0 uv run generate_teacher_data.py --teacher-model-id deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --top-k 256
CUDA_VISIBLE_DEVICES=0 uv run run_kd.py --kd-loss-type sft --train-mode svd --train-position output --no-wandb
CUDA_VISIBLE_DEVICES=0 uv run run_kd.py --kd-loss-type kl --train-mode blocktt --train-position small --no-wandb

# Batch experiments via shell scripts
DEVICE=2 LR=8e-5 TRAIN_MODE=blocktt bash run_rl.sh
```

## Testing

```bash
python -m py_compile *.py optim/*.py       # syntax check
python -m unittest tests/test_btt_pipeline_compat.py   # BlockTT compatibility
python -m unittest tests/test_svd_pipeline_compat.py   # SVD compatibility
python -m unittest tests/test_run_sft_cli.py           # SFT CLI arg validation
python -m unittest tests/test_run_rl_cli.py            # RL CLI arg validation
python -m unittest tests/test_run_rl_optimizer.py      # RL optimizer tests
python -m unittest tests/test_run_kd_cli.py            # KD CLI arg validation
python -m unittest tests/test_generate_teacher_data.py # Teacher data generation
python -m unittest tests/test_analyze_weights.py       # Weight analysis
```

## Architecture

### Training entrypoints
- `run_sft.py` — Unified SFT script. `--train-mode full|lora|blocktt|svd` selects the method. Compatibility wrappers (`sft_full.py`, `sft_lora.py`, `sft_blocktt.py`, `sft_svd.py`) delegate to it. Also exports shared utilities (`build_collate_fn`, `build_lr_scheduler`, `build_optimizer`, `prepare_model`, etc.) used by other scripts.
- `run_rl.py` — Unified RL (GRPO) script with same `--train-mode` flag. Uses vLLM for rollout generation. Legacy scripts (`rl_full.py`, `rl_lora.py`, `rl_blocktt.py`) still exist.
- `run_kd.py` — Knowledge distillation script. `--kd-loss-type sft|kl` selects SFT cross-entropy or KL divergence loss. Trains on precomputed teacher data. Imports shared utilities from `run_sft.py`.
- `generate_teacher_data.py` — Generates teacher model completions and top-K logits for knowledge distillation. Outputs safetensors files consumed by `run_kd.py`.

### Custom layer decompositions
- `btt_layer.py` — `BTTLayer` replaces `nn.Linear` with Block Tensor-Train factorization. Key functions: `convert_linear_to_btt()` replaces matching layers in-place, `configure_blocktt_trainability()` freezes/unfreezes cores. Cores are `btt_l` (left, output-side) and `btt_r` (right, input-side). The `--train-position small|large|both` flag controls which core trains; `--s-merged-to` controls where singular values are absorbed.
- `svd_layer.py` — `SVDLayer` replaces `nn.Linear` with full-rank SVD factorization (`svd_a @ svd_b`). Same `convert_linear_to_svd()` / `configure_svd_trainability()` pattern. `--train-position output|input` selects trainable factor.

### Optimizers
- `optim/muon.py` — Muon optimizer (from Moonlight/Muon repos). Uses orthogonalization via `optim/polar_express.py`. Selected via `--optimizer muon`.

### Shared utilities
- `math_utils.py` — Extracts `\boxed{}` answers from model outputs and checks mathematical equivalence. Used by RL reward computation.
- `boxed.prompt` — Prompt template for math RL rollouts.

### Analysis
- `analysis/` — Weight analysis and reporting tools (`analyze_weights.py`, `build_report.py`, `plot_singular_relationship.py`).

### Reference implementations
- `ref/spectral_adapter/llama3_tune/` — Llama 3 BlockTT tuning experiments. Uses `btt_layer.py` indirectly via a local `blocktt_utils.py` wrapper that provides higher-level convenience functions (`apply_blocktt_to_model`, `write_blocktt_metadata`).

### Key design patterns
- Both `run_sft.py` and `run_rl.py` share the same argparse structure with mode-specific flag validation (flags for one mode raise errors if used with another). `run_kd.py` reuses shared utilities from `run_sft.py` rather than duplicating them.
- RL rollout in LoRA mode: exports LoRA adapter to a temp dir, uploads to vLLM server via API, generates completions, then updates weights after each GRPO step.
- RL rollout in BlockTT/SVD mode: materializes dense weights from factored layers, patches them into the model, runs local vLLM generation, then restores factored layers.
- Checkpoints saved as `step={N}/model.safetensors` under the run output directory.

## Hardware Assumptions

Experiments designed for H100 (94GB). SFT uses Qwen3-4B, RL uses Qwen3-1.7B. Adjust batch size or add gradient checkpointing for smaller GPUs.
