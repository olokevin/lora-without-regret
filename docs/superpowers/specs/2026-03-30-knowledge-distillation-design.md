# Knowledge Distillation Pipeline Design

## Overview

Add a two-script knowledge distillation pipeline: `generate_teacher_data.py` (run once to cache teacher outputs) and `run_kd.py` (train student model on cached data). Supports all four training modes (full/lora/blocktt/svd) and two KD loss types. Checkpoints are compatible with `plot_singular_relationship.py` for analyzing singular vector updates.

## Models

- **Teacher:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` (Qwen2-based, 3584 hidden dim)
- **Student:** `Qwen/Qwen2.5-0.5B` (Qwen2-based, 896 hidden dim)
- **Tokenizer compatibility:** Both share the same base BPE vocabulary (151,643 tokens, identical merges). 7 special token IDs (151643-151649) differ but are irrelevant for math content. Embedding matrix sizes differ (151,936 vs 152,064 — padding rows). For KL loss, student logits are sliced to `[:shared_vocab_size]` where `shared_vocab_size = min(teacher_vocab, student_vocab)`.

## Dataset

- `qwedsacf/competition_math` from HuggingFace
- Training split: `train[:7500]`
- Validation split: `train[-5000:]`
- Prompt template: `boxed.prompt`
- Max generation length: 1024 tokens

## Script 1: `generate_teacher_data.py`

### Purpose

Load teacher model, generate completions for MATH dataset with top-K logits, save to disk. Run once; reuse across student training runs.

### Output directory

`/data/yequan/fura/kd/{teacher_name}-{dataset}/`

Example: `/data/yequan/fura/kd/DeepSeek-R1-Distill-Qwen-7B-competition_math/`

### Saved artifacts

- `config.json` — metadata: teacher model ID, dataset name/split, K, max_tokens, shared_vocab_size, num_examples, prompt template path, generation temperature
- `completions.jsonl` — one line per example: `{"index": int, "question": str, "ground_truth": str, "prompt": str, "completion": str, "token_ids": list[int]}`
- `logits/chunk_{i}.safetensors` — top-K logit data per chunk:
  - `topk_values`: shape `[num_seqs_in_chunk, max_seq_len, K]`, bfloat16
  - `topk_indices`: shape `[num_seqs_in_chunk, max_seq_len, K]`, int32
  - `seq_lengths`: shape `[num_seqs_in_chunk]`, int32 (actual response token count per sequence)

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--teacher-model-id` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | Teacher model HF ID |
| `--dataset` | `qwedsacf/competition_math` | Dataset name |
| `--dataset-split` | `train[:7500]` | Dataset split |
| `--output-dir` | `/data/yequan/fura/kd/` | Base output directory |
| `--max-tokens` | `1024` | Max generation length |
| `--top-k` | `256` | Number of top logits to save per token |
| `--prompt-template` | `boxed.prompt` | Path to prompt template file |
| `--batch-size` | `16` | Batch size for vLLM generation |
| `--chunk-size` | `500` | Examples per logit chunk file |

### Generation

Uses vLLM locally with greedy decoding (`temperature=0`) and `logprobs=top_k`. Processes the full dataset, saves completions and logits in chunks.

## Script 2: `run_kd.py`

### Purpose

Train student model on precomputed teacher data. Mirrors `run_sft.py` structure, reusing shared functions.

### KD loss types

Selected via `--kd-loss-type sft|kl`:

1. **SFT loss:** Teacher's completion text is tokenized as ground-truth sequence using student tokenizer. Standard cross-entropy loss, masked to response tokens only (prompt tokens get `-100` labels). Same loss as `run_sft.py`.

2. **KL loss:** Token-wise `KL(student || teacher)` over teacher's top-K logit positions. For each token position:
   - Load teacher's top-K indices and values
   - Extract student logits at those K positions
   - Softmax both to get distributions (renormalized over K tokens)
   - Compute `KL(student || teacher)` per token
   - Masked to response tokens only

### CLI flags

Inherits all flags from `run_sft.py` (train-mode, optimizer, scheduler, lora/blocktt/svd options, wandb, etc.) with these changes:

| Flag | Default | Description |
|------|---------|-------------|
| `--kd-loss-type` | required | `sft` or `kl` |
| `--teacher-data-dir` | `/data/yequan/fura/kd/DeepSeek-R1-Distill-Qwen-7B-competition_math/` | Path to precomputed teacher data |
| `--top-k` | `256` | Top-K for KL loss (must match generated data) |
| `--save-steps` | `1,10,final` | Comma-separated optimizer steps at which to save checkpoints |
| `--student-model-id` replaces `--model-id` | `Qwen/Qwen2.5-0.5B` | Student model HF ID |

Removed flags vs `run_sft.py`: no dataset selection flags (data comes from `--teacher-data-dir`).

### Training mode support

All four modes from `run_sft.py`: `full`, `lora`, `blocktt`, `svd`. Uses the same `prepare_model()` function, same flag validation, same optimizer/scheduler building.

### Data loading

- Reads `config.json` to validate compatibility (top-k, vocab size)
- Loads `completions.jsonl` into a dataset
- For SFT loss: tokenizes teacher completions using student tokenizer, creates `(input_ids, labels, attention_mask)` tensors with prompt masked
- For KL loss: additionally memory-maps logit chunks, serves `(input_ids, attention_mask, topk_values, topk_indices)` per batch

### Checkpoint saving

- Format: `{output_dir}/step={N}/model.safetensors`
- Saves full model state via `safetensors.torch.save_file()` with all named parameters (including factored weights like `svd_a`, `svd_b`, `svd_s`, `btt_l`, `btt_r`, `btt_s`)
- Default save points: step 1 (after first optimizer update), step 10, and final step after training completes
- Configurable via `--save-steps` (e.g., `--save-steps 1,5,10,20,final`)

### Evaluation

Computes validation loss on the last portion of teacher data (using the same loss type as training). Runs every 10 steps, matching `run_sft.py` cadence.

### Code reuse from `run_sft.py`

Imported directly:
- `set_seed`
- `prepare_model`
- `validate_trainable_params`
- `build_optimizer`
- `build_lr_scheduler`
- `compute_num_training_steps`
- `resolve_warmup_steps`

KD-specific (new code):
- Arg parsing (adapted flags)
- Teacher data loading and dataset construction
- SFT loss on teacher completions
- KL loss on teacher logits
- Checkpoint saving in `step={N}/model.safetensors` format
- Training loop (adapted from `run_sft.py` with checkpoint hooks)

## Example usage

```bash
# Step 1: Generate teacher data (run once)
CUDA_VISIBLE_DEVICES=0 uv run generate_teacher_data.py \
  --teacher-model-id deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --top-k 256

# Step 2: Train student with SFT loss + SVD decomposition
CUDA_VISIBLE_DEVICES=0 uv run run_kd.py \
  --kd-loss-type sft \
  --train-mode svd \
  --train-position output \
  --s-merged-to frozen \
  --lr 1e-4 \
  --save-steps 1,10,final \
  --enable-save-ckpt \
  --no-wandb

# Step 3: Train student with KL loss + BlockTT decomposition
CUDA_VISIBLE_DEVICES=0 uv run run_kd.py \
  --kd-loss-type kl \
  --train-mode blocktt \
  --decomp-mode input_one_block \
  --train-position small \
  --s-merged-to frozen \
  --lr 8e-5 \
  --save-steps 1,10,final \
  --enable-save-ckpt \
  --no-wandb

# Step 4: Plot singular vector analysis
python analysis/plot_singular_relationship.py --ckpt-dir ./kd_model/
```
