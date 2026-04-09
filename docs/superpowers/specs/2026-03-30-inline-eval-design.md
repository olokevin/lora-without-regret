# Inline Evaluation for llama_tune.py

## Goal

Add periodic GSM8K evaluation during training in `ref/spectral_adapter/llama3_tune/llama_tune.py`, logging accuracy to wandb under `val/`.

## Argument

- `--inline-eval-steps`: int, default 100. Run GSM8K eval every N steps. 0 disables inline eval.

## Implementation

### InlineEvalCallback

A `TrainerCallback` that on `on_step_end`:

1. Checks `state.global_step % inline_eval_steps == 0` (and `> 0`).
2. Sets model to `eval()` mode.
3. Wraps the in-memory model+tokenizer with `lm_eval.models.huggingface.HFLM` (no disk save).
4. Calls `lm_eval.simple_evaluate(model=hflm, tasks=["gsm8k"], num_fewshot=5, batch_size=8)`.
5. Extracts `exact_match,flexible-extract` score.
6. Logs `{"val/gsm8k_exact_match": score}` via the trainer's logger at the current global step.
7. Restores model to `train()` mode.

### Model compatibility

- **full/lora/spectral**: Model passed directly to HFLM. PEFT wrappers work transparently.
- **blocktt**: Evaluate with factored layers as-is (identical forward pass).

### wandb

Metric logged as `val/gsm8k_exact_match` in the existing wandb run. No additional wandb configuration needed.

### Edge cases

- First eval at step N (not step 0).
- `--inline-eval-steps 0` disables eval entirely.
- `model.eval()` during evaluation, `model.train()` after.
