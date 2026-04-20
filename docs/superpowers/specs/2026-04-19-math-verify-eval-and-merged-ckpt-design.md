# Math-verify post-training eval + merged checkpoints

**Status:** Approved (2026-04-19)
**Scope:** Spec 1 of 3 from the "expand `run_rl.py` usage" planning thread. Specs 2 (DoRA / PiSSA / MiLoRA / LIFT adapters in `run_rl.py`) and 3 (RandLoRA / HiRA + cross-entrypoint usability with `ref/LIFT`) are deferred to follow-up specs.

## 1. Scope and goals

Add two capabilities to `run_rl.py` and one new entrypoint:

1. **`--enable-merged-ckpt` (default `true`)** — every checkpoint save goes through merge / materialize so the on-disk format is plain HuggingFace (`config.json` + `model.safetensors`, only `nn.Linear` layers). Loadable with vanilla `AutoModelForCausalLM.from_pretrained`. Setting `false` preserves today's adapter-only / factored save behavior.
2. **`--enable-math-verify` (default `true`)** — after the GRPO loop completes, evaluate the final merged checkpoint on a fixed set of math reasoning benchmarks using the HuggingFace `math-verify` library. Default datasets: `MATH-500`, `AIME-24`, `AIME-25`, `AMC23`, `Minerva`. Override via `--math-verify-datasets`. Results logged to wandb (`eval/{dataset}/accuracy`) and written to `eval_results.json` next to the checkpoint.
3. **`eval_rl.py`** — standalone script that takes `--checkpoint PATH_OR_HF_ID` and runs the same math-verify eval against any merged checkpoint (or a base model from HF Hub).

**Out of scope:**

- Changes to the GRPO reward grader (still uses `math_utils.is_equiv`).
- Changes to in-loop training-time validation (still uses `is_equiv` on `qwedsacf/competition_math`).
- Loading legacy adapter-only / factored checkpoints from `eval_rl.py`.
- Fixing the `lora_full` adapter-only save bug as a separate change. The merged-ckpt path captures backbone deltas as a side effect; the legacy non-merged path remains broken for `lora_full` and we do not address it here.

## 2. Architecture and component layout

```
lora-without-regret/
├── run_rl.py                  # modified: --enable-merged-ckpt, --enable-math-verify, post-loop hook
├── eval_rl.py                 # NEW: standalone eval entrypoint
├── math_verify_eval.py        # NEW: shared eval module (used by both run_rl.py and eval_rl.py)
└── eval_datasets.py           # NEW: dataset registry
```

### `math_verify_eval.py`

Single source of truth for grading. One public function:

```python
def math_verify_eval(
    model,                  # already-loaded HF model on GPU
    tokenizer,
    datasets: list[str],    # e.g. ["MATH-500", "AIME-24"]
    *,
    n_samples_override: int | None = None,
    temperature_override: float | None = None,
    max_tokens: int = 2048,
    prompt_template_path: str = "boxed.prompt",
    vllm_kwargs: dict | None = None,
) -> dict[str, dict]:
    """
    Returns {dataset_name: {"accuracy": float, "n_correct": int,
                             "n_total": int, "n_samples_per_problem": int,
                             "temperature": float, "max_tokens": int,
                             "wall_time_sec": float, "n_unparseable": int,
                             "n_grader_errors": int}}.
    """
```

Internally:

- Builds (or accepts) a vLLM `LLM` instance. Pass `vllm_kwargs={"llm": existing_llm}` to reuse an in-process LLM from the training loop.
- For each requested dataset, calls `eval_datasets.load_eval_dataset(...)` to get prompt-wrapped problems, runs vLLM generation with that dataset's sampling params, calls `math_verify.parse(model_output)` and `math_verify.verify(gold, pred)` on each rollout.
- For multi-sample datasets, accuracy is averaged across `n_problems × n_samples` rollouts (avg@k convention; this is what published AIME numbers report).
- Per-dataset failures (HF download error, etc.) are collected in a returned `errors` dict; the function returns successfully with whatever datasets succeeded.

### `eval_datasets.py`

Dataset registry mapping name → loader + sampling defaults:

```python
@dataclass(frozen=True)
class DatasetSpec:
    hf_id: str
    split: str
    problem_field: str
    answer_field: str
    n_samples: int          # 1 for greedy datasets, 8 for AIME
    temperature: float      # 0.0 for greedy, 0.6 for AIME
    top_p: float            # 1.0 for greedy, 0.95 for AIME

REGISTRY = {
    "MATH-500":  DatasetSpec("HuggingFaceH4/MATH-500",  "test",  "problem",  "answer", 1, 0.0, 1.0),
    "AIME-24":   DatasetSpec("HuggingFaceH4/aime_2024", "train", "problem",  "answer", 8, 0.6, 0.95),
    "AIME-25":   DatasetSpec("yentinglin/aime_2025",    "train", "problem",  "answer", 8, 0.6, 0.95),
    "AMC23":     DatasetSpec("math-ai/amc23",           "test",  "question", "answer", 1, 0.0, 1.0),
    "Minerva":   DatasetSpec("math-ai/minerva-math",    "test",  "question", "answer", 1, 0.0, 1.0),
}
```

`load_eval_dataset(name, tokenizer, prompt_template) -> list[dict]` returns `[{prompt, gold_answer, n_samples, temperature, top_p}, ...]` where `prompt` is already wrapped in `boxed.prompt` + chat template — the same wrapping used by `run_rl.py` training. The exact field names above will be confirmed against each HF mirror at implementation time; the registry is the only place that needs updating if a mirror's schema drifts.

### Merge helper in `run_rl.py`

```python
def save_merged_checkpoint(model, tokenizer, ckpt_dir: str, train_mode: str, args):
    """Save model in plain HF format. Model object is restored to its
    factored/adapter state before returning so training can resume."""
    # full           → model.save_pretrained(ckpt_dir)
    # lora/lora_full → merge_adapter() → get_base_model().save_pretrained(...) → unmerge_adapter()
    # blocktt/svd    → build a state_dict where each BTTLayer/SVDLayer is replaced by its
    #                  materialize_dense_weight() output, save_pretrained(state_dict=...).
    #                  Model object never mutated; training resumes immediately.
    # blocktt+calib  → existing save_calibrated_btt_hf_pretrained (already produces dense)
    tokenizer.save_pretrained(ckpt_dir)
```

When `--enable-merged-ckpt false`, `save_checkpoint` falls back to today's exact code path (adapter-only / factored). Behavior is byte-identical to current `main`.

### `eval_rl.py`

Thin: argparse → `AutoModelForCausalLM.from_pretrained(checkpoint)` → `math_verify_eval(...)` → write `eval_results.json`. Pre-flight checks reject legacy adapter-only / factored paths with a clear error.

## 3. Data flow and CLI surface

### `run_rl.py` end-of-training flow (both flags default on)

```
GRPO loop completes
    ↓
final_ckpt_dir = run_dir/step={N_FINAL}/
                 (always created when math-verify or merged-ckpt is enabled,
                  regardless of --enable-save-ckpt)
    ↓
save_merged_checkpoint(model, tokenizer, final_ckpt_dir, args.train_mode, args)
    ↓                                       (model object restored to factored/adapter state)
if args.enable_math_verify:
    results = math_verify_eval(
        model,                              # reuse in-memory model where possible
        tokenizer,
        datasets=args.math_verify_datasets,
        n_samples_override=args.math_verify_n_samples,
        temperature_override=args.math_verify_temperature,
        max_tokens=args.math_verify_max_tokens,
        vllm_kwargs={"llm": existing_vllm_llm}  # reuse rollout LLM where possible
    )
    log to wandb under eval/{dataset}/accuracy
    write final_ckpt_dir/eval_results.json
```

**vLLM reuse logic.** Training-time vLLM is bound to either an HTTP server (`lora` mode with reachable server) or an in-process `LLM`. For HTTP-lora training, eval needs a fresh local in-process LLM, because the HTTP server holds the *base* model + LoRA adapter, which won't match the merged checkpoint we just wrote. So:

- HTTP-lora training → spin up a fresh local `LLM(model=final_ckpt_dir)` after training (loaded from the just-written merged checkpoint on disk), evaluate, tear down. The HF model in Python memory is not reused for generation here.
- All other modes → reuse the existing local `vllm_model` instance, hot-swap its weights from the in-memory HF model via the existing `load_weights` path used by `build_local_vllm_generators` / `build_lora_local_generators`. The merged checkpoint on disk exists for `eval_rl.py` reuse later, but eval generation reads weights from the in-memory model, not from disk.

**Mid-training checkpoints.** When `--enable-save-ckpt` is set, `save_checkpoint` (run_rl.py:681) calls `save_merged_checkpoint` instead of `model.save_pretrained` directly. The HTTP-lora rollout path's `save_lora` (run_rl.py:815) is **not** changed — it must continue saving adapter-only because vLLM's `/v1/load_lora_adapter` endpoint requires that format. To eliminate the directory collision between rollout adapter saves and checkpoint saves, the rollout path is moved to `{run_dir}/lora_adapters/step={N}/` (new subdir). This makes ownership unambiguous: `step={N}/` always means "merged checkpoint", `lora_adapters/step={N}/` always means "rollout adapter".

### `run_rl.py` new flags

```
--enable-merged-ckpt / --no-enable-merged-ckpt   (BooleanOptionalAction, default True)
--enable-math-verify / --no-enable-math-verify   (BooleanOptionalAction, default True)
--math-verify-datasets STR                       (default "MATH-500,AIME-24,AIME-25,AMC23,Minerva")
--math-verify-n-samples INT                      (default None → use registry per-dataset)
--math-verify-temperature FLOAT                  (default None → use registry per-dataset)
--math-verify-max-tokens INT                     (default 2048)
```

`--math-verify-datasets` is parsed at argparse-validation time: split on comma, strip whitespace, every name must be in `eval_datasets.REGISTRY` or argparse fails fast (before training starts).

### `eval_rl.py` CLI

```
eval_rl.py --checkpoint PATH_OR_HF_ID
           [--math-verify-datasets MATH-500,AIME-24,...]
           [--math-verify-n-samples INT]
           [--math-verify-temperature FLOAT]
           [--math-verify-max-tokens 2048]
           [--prompt-template boxed.prompt]
           [--max-model-len 2048]
           [--gpu-memory-utilization 0.4]
           [--output-json PATH]                  # default: {checkpoint}/eval_results.json
                                                 # if checkpoint is HF ID, default ./eval_results.json
           [--seed 42]
```

`--checkpoint` accepts either a local directory or a HuggingFace ID; both are passed verbatim to `AutoModelForCausalLM.from_pretrained`. Base-model eval is `eval_rl.py --checkpoint Qwen/Qwen3-1.7B`.

### `eval_results.json` schema

```json
{
  "checkpoint": "/data/.../step=50",
  "model_id_at_train_time": "Qwen/Qwen3-1.7B",
  "datasets": {
    "MATH-500": {
      "accuracy": 0.612,
      "n_correct": 306,
      "n_total": 500,
      "n_samples_per_problem": 1,
      "temperature": 0.0,
      "max_tokens": 2048,
      "wall_time_sec": 84.3,
      "n_unparseable": 12,
      "n_grader_errors": 0
    },
    "AIME-24": {
      "accuracy": 0.167,
      "n_correct": 40,
      "n_total": 240,
      "n_samples_per_problem": 8,
      "temperature": 0.6,
      "max_tokens": 2048,
      "wall_time_sec": 312.1,
      "n_unparseable": 5,
      "n_grader_errors": 1
    }
  },
  "errors": {},
  "math_verify_version": "0.5.2",
  "timestamp": "2026-04-19T10:23:00Z"
}
```

For multi-sample datasets, `accuracy = mean(per_sample_correct)` across all `n_problems × n_samples` rollouts (avg@k convention). `n_total` is `n_problems × n_samples_per_problem`. `pass@1` and `pass@k` are out of scope for this spec but the JSON layout leaves room to add them later.

## 4. Error handling and edge cases

### Save-time merge

- `lora_full`: `merge_adapter` folds the LoRA delta into base; the additionally trainable backbone deltas are already in `base_model.named_parameters()`, so `get_base_model().save_pretrained()` captures them. This is the "free side effect" — the merged-ckpt path solves the existing `lora_full` adapter-only save bug as a byproduct without any special handling.
- `blocktt` / `svd`: the merge helper builds a fresh `state_dict` (factored layer name `foo.bar` → dense `foo.bar.weight` via `materialize_dense_weight()`) and calls `save_pretrained(state_dict=...)` rather than swapping modules. The model object is never mutated, so training can resume without try/finally.
- `blocktt` with `--calib-mode != none`: routes to existing `save_calibrated_btt_hf_pretrained`, unchanged.
- `--enable-merged-ckpt false`: falls back to today's exact code path. Behavioral parity with current `main`.

### Eval-time

Two distinct failure tiers:

- **Tier 1 — eval cannot run at all** (vLLM startup failure, e.g. CUDA OOM after training): log error, skip math-verify, exit 0 (training still succeeded). `eval_results.json` is **not** written; wandb gets a single `eval/error` field. Nothing partial.
- **Tier 2 — per-dataset failure** during a successful eval run (HF download error, network blip, etc.): log warning, skip that dataset, continue with the rest. `eval_results.json` **is** written, omits the failed dataset under `datasets`, and records its reason under the top-level `errors` field. Wandb gets `eval/{dataset}/error` for each failed dataset.
- `math_verify.parse` returns empty (no `\boxed{}` in output): counted as incorrect, tallied per-dataset in `n_unparseable`. Not an error.
- `math_verify.verify` raises on a single problem: counted as incorrect, tallied per-dataset in `n_grader_errors`. One bad sample never crashes the whole run.

### `eval_rl.py`

- Checkpoint path doesn't exist and isn't an HF ID: `AutoModelForCausalLM.from_pretrained` raises; let it propagate (clear default error).
- Legacy adapter-only / factored format: pre-flight detects it before calling `from_pretrained`. If the local path contains `adapter_config.json` or its `model.safetensors` contains keys matching `*.btt_l`, `*.btt_r`, `*.svd_a`, `*.svd_b`, error out with: `"Checkpoint at {path} is a legacy {adapter|factored} checkpoint. eval_rl.py only supports merged checkpoints; re-run training with --enable-merged-ckpt true or use the in-loop --enable-math-verify path."`

### Boundary checks

- `--math-verify-datasets` parsing: split on comma, strip whitespace, every name must be in `REGISTRY`. Unknown name → fail fast at argparse-validation time, before training starts.
- `--math-verify-n-samples 0` → reject.
- `--math-verify-max-tokens` ≤ prompt length: not pre-validated; vLLM will error (depends on tokenizer).

### Accepted limitation

`--no-enable-merged-ckpt --enable-math-verify` is valid but eval will likely fail at end-of-training for non-`full` modes (the checkpoint isn't loadable by vanilla `from_pretrained`). We do not block this combination; argparse prints a warning at startup: `"--enable-math-verify with --no-enable-merged-ckpt may fail at eval time for non-full modes."`

### Default-on impact on existing scripts and tests

With `--enable-math-verify` defaulting to `true`, every existing invocation of `run_rl.py` that does not pass `--no-enable-math-verify` will run a multi-dataset eval at the end of training and download the five HF datasets the first time. This is intentional for normal usage but undesirable for fast smoke runs and existing test scripts. Mitigations:

- Existing shell scripts (`run_rl.sh`, etc.) and any short-step debug invocations should add `--no-enable-math-verify` explicitly, the same way they currently add `--no-wandb`. The implementation plan includes a sweep of existing `*.sh` scripts under the repo to add the flag where appropriate.
- `tests/test_run_rl_cli.py` and other existing argparse tests already pass `--no-wandb`; they will be updated to also pass `--no-enable-math-verify` and `--no-enable-merged-ckpt` to keep their behavior identical to today.

## 5. Testing

The repo's existing tests are CLI / argparse smoke tests + a few pipeline-compat unittests (`tests/test_run_rl_cli.py`, `tests/test_btt_pipeline_compat.py`). Small, fast, no GPU required. We follow that pattern.

### New test files

**`tests/test_run_rl_merged_ckpt.py`** — argparse + merge-helper unit tests:

- `--enable-merged-ckpt` / `--no-enable-merged-ckpt` parse correctly; default is `True`.
- `save_merged_checkpoint` for a tiny `nn.Linear`-wrapped module in each mode produces a directory containing `config.json` + `model.safetensors` + tokenizer files; the saved state_dict has zero LoRA / `btt_l` / `btt_r` / `svd_a` / `svd_b` keys.
- After `save_merged_checkpoint`, the in-memory model is unchanged: same parameter ids, same `requires_grad` flags. (Confirms training can resume.)
- Round-trip: save merged → `AutoModelForCausalLM.from_pretrained(saved_dir)` succeeds and produces forward outputs numerically close (atol=1e-4) to the pre-save factored model on a small input. One test per mode (`full`, `lora`, `lora_full`, `blocktt`, `svd`). Tiny model (2-layer toy config), CPU OK.

**`tests/test_run_rl_math_verify_cli.py`** — argparse-only, no model load:

- `--enable-math-verify` parses; defaults `True`.
- `--math-verify-datasets MATH-500,AIME-24` parses; unknown name (`BOGUS`) fails with a clear error citing valid names.
- `--math-verify-n-samples 0` rejected.
- The "warn but don't block" combination `--no-enable-merged-ckpt --enable-math-verify` runs without raising and prints the warning string.

**`tests/test_eval_rl_cli.py`** — argparse + pre-flight legacy-format detection:

- Required `--checkpoint` enforced.
- Adapter-only directory (write a fake `adapter_config.json` to a tmpdir) → pre-flight raises with the legacy-format message.
- Factored directory (write a fake state_dict containing `*.btt_l` keys) → pre-flight raises.
- Plain HF directory (`config.json` only) → passes pre-flight (we do not actually load the model in this test).

**`tests/test_eval_datasets.py`** — registry sanity:

- `REGISTRY` covers exactly the five names from the spec.
- `load_eval_dataset` against each name returns a non-empty list of `{prompt, gold_answer, ...}` dicts; each prompt contains the chat template's assistant marker. **This test hits HF Hub** — gate behind an env var (`RUN_HF_TESTS=1`) so default CI does not depend on network.

**`tests/test_math_verify_eval.py`** — grader smoke test (no GPU):

- Mock the vLLM call so it returns canned `\boxed{...}` strings; assert `math_verify_eval` returns the expected accuracy on a 4-problem hand-built mini dataset. Validates the parse / verify integration and the multi-sample averaging math.

### Not tested

- Actual end-to-end training + eval (slow, needs GPU; validated by user runs).
- Real vLLM startup / weight hot-swap (covered by existing manual smoke tests).

### Dependency change

Add `math-verify>=0.5` to `pyproject.toml`. Run `uv sync` once after the change lands.

### Manual validation steps (called out in the implementation plan, not automated)

1. `uv run run_rl.py --train-mode lora --n-grpo-steps 1 --no-wandb` — confirms merged-ckpt save works end-to-end, eval runs after the 1-step training, `eval_results.json` written.
2. `uv run eval_rl.py --checkpoint Qwen/Qwen3-1.7B --math-verify-datasets MATH-500` — base-model eval from HF Hub, no checkpoint needed.
3. Repeat (1) with `--train-mode blocktt --decomp-mode input_one_block --train-position small` and `--train-mode svd --train-position output` — confirms merge for factored modes.
