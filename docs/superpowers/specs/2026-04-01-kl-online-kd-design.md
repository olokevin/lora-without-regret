# Design: `kl_online` KD Loss Type

**Date:** 2026-04-01  
**File:** `run_kd.py`

## Context

The existing `run_kd.py` supports two KD loss types:
- `sft`: CE loss on teacher completions (precomputed)
- `kl`: KL divergence against precomputed teacher top-K logits

`kl_online` adds a third mode that runs the teacher model live during training — no precomputed logits needed. This enables distillation from any compatible model without a separate data generation step for logits.

## Data Source

Reuses `completions.jsonl` from `generate_teacher_data.py` for prompts and token IDs (same teacher data directory). The saved logit chunks are ignored. `--teacher-data-dir` is still required to locate `completions.jsonl` and `config.json` (for `shared_vocab_size`, `prompt_template`, ground truth answers).

## New Argument

Add `--teacher-model-id` to `run_kd.py` argparse:
- Required when `--kd-loss-type kl_online`, error otherwise if not provided
- Ignored for `sft` and `kl` modes
- Example: `--teacher-model-id deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`

## Dataset: `KDOnlineDataset`

New dataset class parallel to `KDSftDataset`. Returns per-example:
- `input_ids`: completion token IDs (from `token_ids` in completions.jsonl)
- `attention_mask`: all ones
- `response_mask`: all ones (all tokens are completion tokens)

Collate function: `build_kd_online_collate_fn(pad_token_id)` — pads `input_ids`, `attention_mask`, `response_mask`.

## Teacher Loading: `load_teacher_model`

```python
def load_teacher_model(teacher_model_id, device):
    model = AutoModelForCausalLM.from_pretrained(teacher_model_id, torch_dtype=torch.bfloat16)
    model.requires_grad_(False)
    model.eval()
    return model.to(device)
```

Called once in `main()` after student model is prepared, only when `kd_loss_type == "kl_online"`. Otherwise `teacher_model = None`.

## Loss: `compute_online_kl_loss`

```python
def compute_online_kl_loss(student_logits, teacher_logits, response_mask, shared_vocab_size=None):
```

1. If `shared_vocab_size` is not None, slice both logits: `[:, :, :shared_vocab_size]`
2. `teacher_log_probs = F.log_softmax(teacher_logits.float(), dim=-1)`
3. `student_log_probs = F.log_softmax(student_logits.float(), dim=-1)`
4. `kl_per_token = F.kl_div(teacher_log_probs, student_log_probs, log_target=True, reduction="none").sum(dim=-1)`
5. Mask and average over response tokens (same pattern as `compute_kl_loss`)

Unlike the offline `kl` mode this operates over the full shared vocab, not just top-K.

## Training Loop Changes

In the training step, when `kd_loss_type == "kl_online"`:

```python
with torch.no_grad():
    teacher_outputs = teacher_model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )
student_outputs = model(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
)
loss = compute_online_kl_loss(
    student_outputs.logits,
    teacher_outputs.logits,
    batch["response_mask"],
    shared_vocab_size,
)
```

## Validation Loss

`eval_model` Part 1 gains a `kl_online` branch: runs teacher + student forward passes on CE val batches, reports `val/loss (KL online)` using `compute_online_kl_loss`. The CE val dataloader (already built from `KDSftDataset`) provides `input_ids`, `attention_mask`, `labels` — `response_mask` for the val KL pass is derived as `(labels != -100).long()`.

## Validation for `top-k` flag

Add a check: if `kd_loss_type == "kl_online"` and `--top-k` is explicitly passed, raise a warning (it is ignored in this mode).

## Example Usage

```bash
CUDA_VISIBLE_DEVICES=0 uv run run_kd.py \
  --kd-loss-type kl_online \
  --teacher-model-id deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --teacher-data-dir /data/yequan/fura/kd_data/DeepSeek-R1-Distill-Qwen-7B-competition_math \
  --train-mode lora --lora-rank 16 \
  --no-wandb
```
