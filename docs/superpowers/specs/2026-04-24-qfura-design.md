# qfura: Quantized BlockTT Fine-Tuning — Design Spec

**Date:** 2026-04-24
**Status:** Draft pending implementation plan
**Scope:** SFT-only, commonsense-reasoning, Llama-3-70B on a single H100 (94 GB)

## 1. Summary

qfura is a quantized variant of BlockTT (BTT) fine-tuning modeled on QLoRA. Starting from an existing BTT factorization `W ≈ btt_l ⊗ btt_r` (produced by the repo's existing SVD-based BTT conversion), qfura quantizes the **larger, frozen** core to 4-bit NF4 and keeps the smaller core plus the singular-value vector `btt_s` trainable in bf16. The headline target is fine-tuning Llama-3-70B on commonsense-reasoning data on a single H100, matching the QLoRA paper's demonstration that a 65B+ model can be tuned on a single consumer/data-center GPU.

qfura does not introduce a new initialization scheme. BTT's existing SVD init is reused as-is; quantization is applied to whichever core is frozen after `configure_blocktt_trainability(..., train_position="small")`.

The integration sits alongside the existing `ref/LIFT/src/finetune_blocktt.py` path — a new sibling script `finetune_qfura.py` that reuses LIFT's commonsense data loader, eval harness, and `save_hf_format` pipeline.

## 2. Contribution and research questions

qfura is a straightforward application of QLoRA-style 4-bit quantization to BTT's frozen core. The research axis exposed in this spec is the **quantization block layout** for the 3D BTT core:

- **`flat`** — flatten the 3D core to a single 2D tensor, quantize with one block-wise NF4 pass. Minimal deviation from standard QLoRA; block-wise statistics are computed over a layout that has no physical meaning relative to the BTT core structure.
- **`per_core_block`** — quantize each outermost-dimension BTT block independently. For `btt_l` of shape `(m, rank*n, a)`, this produces `m` separate NF4 tensors with `m` independent quant_states. The quantization grid aligns with the natural block structure of BTT.

A dedicated forward + backward quantization-error benchmark measures both layouts and selects the default. The default is recorded in this spec once the benchmark report lands.

## 3. Non-goals

- RL (GRPO, DAPO) integration with qfura. `run_rl.py` is not touched.
- KD integration. `legacy/run_kd.py` is not touched.
- Math-reasoning SFT via qfura. LIFT's math path is not wired up; only commonsense.
- NF4-preserving checkpoint format and NF4-preserving resume. Checkpoints are dequanted to bf16 at save time, which introduces a one-shot round-trip error noted as a caveat.
- A new eval harness. The existing `ref/LIFT/bash_scripts/eval_commonsense.sh` is reused without modification.
- Training entrypoints outside `ref/LIFT/src/`. `run_sft.py` remains unchanged.
- Additional `--train-mode` selections beyond `full|lora|blocktt|svd` in the unified scripts.
- CLI flags for NF4 vs FP4, double-quant on/off, paged-optimizer on/off, and compute dtype. All are hardcoded to QLoRA defaults (NF4, double-quant on, paged AdamW 8-bit, bf16 compute dtype).

## 4. Design

### 4.1 `QBTTLayer` class and conversion path

All code lives in `btt_layer.py` alongside the existing `BTTLayer`. No new file.

**`QBTTLayer(BTTLayer)`** — subclass that stores the frozen BTT core as 4-bit NF4 using `bitsandbytes`' `Params4bit`. The trainable core and `btt_s` remain standard bf16 `nn.Parameter`s.

Storage layout depends on `layout`:

- **`flat`:** a single `bnb.nn.Params4bit` of shape `(frozen_numel, 1)`. One `quant_state`. Shape metadata (original 3D shape, which core is frozen) stored on the module.
- **`per_core_block`:** for frozen `btt_l` of shape `(m, rank*n, a)`, a list of `m` `Params4bit` tensors each of shape `(rank*n, a)` with independent `quant_state`s. For frozen `btt_r` of shape `(n, b, m*rank)`, a list of `n` `Params4bit` tensors each of shape `(b, m*rank)`. The "block" dimension is the outermost BTT dimension.

**Conversion API (added to `btt_layer.py`):**

```python
def quantize_frozen_core_(
    btt_layer: BTTLayer,
    layout: str,                       # "flat" | "per_core_block"
    compute_dtype=torch.bfloat16,
    double_quant: bool = True,
    quant_type: str = "nf4",
) -> QBTTLayer: ...

def convert_btt_to_qbtt_(model: nn.Module, layout: str) -> dict: ...
```

`quantize_frozen_core_` returns a `QBTTLayer` instance that replaces the original `BTTLayer` in the parent module. `convert_btt_to_qbtt_` walks the model, finds every `BTTLayer` whose trainability has been configured with `train_position="small"`, and applies `quantize_frozen_core_` to each. Returns per-model stats (number of layers converted, bytes saved, list of layer names).

**Call order in the training script:**

```
convert_linear_to_btt(model, ...)                    # existing: SVD init → BTT factors
configure_blocktt_trainability(model, train_position="small", ...)  # existing
convert_btt_to_qbtt_(model, layout=args.quant_block_layout)         # new
```

Trainability resolution must happen before quantization so the conversion knows which side is frozen. If `train_position != "small"`, conversion raises `ValueError`.

### 4.2 Forward path

`QBTTLayer.forward` overrides `BTTLayer.forward`. The body is identical to the existing forward except for one step at the top: dequant the frozen core back to its original 3D bf16 shape and bind to a local. The trainable core reads directly from `self.btt_l` or `self.btt_r` as before.

For `flat` layout, one `bnb.functional.dequantize_nf4` call + one reshape produces the 3D bf16 core.

For `per_core_block` layout, iterate over the stored list, dequant each block, `torch.stack` along the block axis to reconstruct the 3D core.

The trainable `btt_s` path is untouched.

**Fused Step-2 Triton kernel compatibility:** the existing `FURA_FUSED_STEP2=1` path reads `self.btt_l` as a bf16 tensor. In qfura, when `btt_l` is frozen, the dequanted local is passed to the kernel instead of `self.btt_l`. Single-line patch at the kernel call site; no kernel changes. When `btt_r` is frozen (less common with square decomp), the fused kernel still operates on the bf16 `btt_l`, so the kernel path is unchanged for that case.

### 4.3 Gradient checkpointing

`Params4bit` is a proper `nn.Parameter` subclass and survives re-forward during backward recomputation. `QBTTLayer.forward` dequants into a local on every call — identical between the initial forward and the recomputation forward. This matches QLoRA's working pattern; no additional integration work is required.

### 4.4 Checkpoint save

Uses LIFT's existing `save_hf_format` path. At save time, qfura dequants the frozen core back to bf16, reconstructs the full bf16 `BTTLayer` state dict, then materializes to dense `nn.Linear.weight` as the existing blocktt path already does. The resulting HF-format checkpoint is indistinguishable from a regular BTT checkpoint and is consumed by `eval_commonsense.sh` without change.

**Round-trip caveat:** the dequanted save introduces a one-shot quantization round-trip error (saved weights differ from training-time effective weights by the quantization error). Magnitude matches the forward-error numbers reported in the benchmark (Section 4.6). Eval accuracy on the saved checkpoint is what is reported in `docs/exp_results/qfura.md`.

### 4.5 Resume

Not supported in v1. If training crashes, restart from the most recent saved HF-format checkpoint by reconverting through the full pipeline (`convert_linear_to_btt` → `configure_blocktt_trainability` → `convert_btt_to_qbtt_`). LIFT's 3-epoch commonsense runs complete in under 24 hours on H100, so a restart-from-scratch is tolerable.

### 4.6 Quantization-error test and benchmark

Two artifacts. The unit test runs in CI; the benchmark is a manual script whose output is committed as a report.

**Unit test: `tests/test_qbtt_quant_error.py`.**

Per layout in `{flat, per_core_block}`:

1. Construct `nn.Linear(4096, 4096)` with a seeded Gaussian weight.
2. Convert to `BTTLayer` via `convert_linear_to_btt` with `decomp_mode="square"`, `train_position="small"`, `blocktt_rank="full"`, `s_merged_to="frozen"`. This runs SVD init.
3. Snapshot reference forward `y_ref = btt(x_ref)` on a seeded Gaussian input of shape `(4, 128, 4096)`. Snapshot reference gradient by taking `(y_ref - target).pow(2).mean().backward()` and collecting `.grad` from the trainable core and `btt_s`.
4. Clone the BTT layer; run `quantize_frozen_core_(clone, layout=layout)`; rerun forward and backward on the same input and target.
5. Assert relative errors:
   - Forward: `‖y_ref − y_q‖ / ‖y_ref‖ < 0.05` for `flat`, `< 0.03` for `per_core_block`.
   - Backward: `‖g_ref − g_q‖ / ‖g_ref‖ < 0.10` for both layouts.

Thresholds are guardrails that catch regressions. They will be tightened or relaxed after the benchmark produces real numbers. The benchmark, not the unit test, is the scientific instrument.

**Benchmark: `analysis/bench_qbtt_quant_error.py` → `docs/reports/qfura-quant-error.md`.**

Single-entry script. Default model `meta-llama/Meta-Llama-3-8B`; `--model meta-llama/Meta-Llama-3-70B` optional.

Per layout × per target-linear-name (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`):

- Load the full model in bf16.
- Capture 32 real activations by running a forward pass on 32 commonsense-reasoning prompts sampled from `ref/LIFT/LLM-Adapters/ft-training_set/commonsense_170k.json`, using hooks at the input of each targeted linear.
- For each linear module, convert to `BTTLayer` via SVD init, measure:
  - **Layer-level forward error:** `‖W_bf16(x) − W_qfura(x)‖ / ‖W_bf16(x)‖` averaged over the 32 activation batches.
  - **Layer-level backward error:** `‖g_bf16 − g_qfura‖ / ‖g_bf16‖` where `g` is the gradient of a seeded MSE proxy loss with respect to the trainable core plus `btt_s`.
- Aggregate per-linear-type: mean, p50, p95 relative forward + backward error.
- Per-layer-depth aggregation: mean + p95 error broken down by transformer layer index (to catch depth-specific patterns).
- **Model-level forward error:** run one full model forward in bf16 vs qfura on the same 32 prompts. Report top-1 logit-match rate, `KL(bf16 || qfura)` averaged over tokens, and logit relative error.

The committed report `docs/reports/qfura-quant-error.md` contains:

- Command to reproduce.
- Per-layer-type table (flat vs per_core_block side-by-side).
- Per-layer-depth plot saved to `figures/qfura_quant_error_by_depth.png`.
- Model-level summary row.
- A concluding paragraph stating the default layout chosen and the rationale (lower model-level KL wins; ties broken in favor of `flat` for simplicity).

**Default for `--quant_block_layout`:** placeholder `flat` until the benchmark report is committed. After the report lands, this spec and `finetune_commonsense_qfura.sh`'s default are updated to match.

### 4.7 Training entrypoint

**New file: `ref/LIFT/src/finetune_qfura.py`.** Derived from `finetune_blocktt.py`. Delta from the baseline:

1. Import `bitsandbytes as bnb` and the qfura additions from `btt_layer` (`QBTTLayer`, `quantize_frozen_core_`, `convert_btt_to_qbtt_`).
2. Assert `args.train_position == "small"`. Raise `ValueError` otherwise.
3. Assert `args.gradient_checkpointing` is set. Raise `ValueError` otherwise.
4. After `configure_blocktt_trainability(..., train_position="small")`, call `convert_btt_to_qbtt_(model, layout=args.quant_block_layout)`.
5. Replace `optimizer = torch.optim.AdamW(...)` with `optimizer = bnb.optim.PagedAdamW8bit(...)` using identical param groups.
6. Log `stats` returned by `convert_btt_to_qbtt_` — number of layers quantized, total bytes saved, layout used — via the existing `print_rank_0` helper, and log the same as a W&B summary.

Everything else — data loading, collator, eval loop, `save_hf_format`, calib-mode hooks, Accelerate prepare — is inherited unchanged.

**Calibration compatibility:** `--calib_mode v2_bp` is accepted and runs on the BTT factorization before quantization. A calibrated BTT factorization produces a frozen core that may be better conditioned for NF4 quantization. Not a hard requirement; exposed for ablation.

### 4.8 CLI surface

**New flag on `finetune_qfura.py`:**

- `--quant_block_layout {flat, per_core_block}` — required. Default set post-benchmark (placeholder `flat`).

**Flags inherited and constrained:**

- `--train_position` — must equal `small`.
- `--gradient_checkpointing` — must be set.

**Hardcoded (not exposed):** NF4 quant type, double quantization enabled, bf16 compute dtype, `PagedAdamW8bit`, gradient checkpointing required.

**New shell script: `ref/LIFT/bash_scripts/finetune_commonsense_qfura.sh`.**

Copy of `finetune_commonsense_blocktt.sh`, changing:

- The `accelerate launch` target to `src/finetune_qfura.py`.
- Default `MODEL="${MODEL:-meta-llama/Meta-Llama-3-70B}"` (qfura's headline is 70B; blocktt's is 8B).
- New env var `quant_block_layout="${quant_block_layout:-flat}"` forwarded as `--quant_block_layout`.
- `train_position` forced to `small` regardless of env var.
- `per_device_train_batch_size=1`, `gradient_accumulation_steps=16` (match blocktt's effective batch size while fitting 70B activations).
- Output dir template: `commonsense/${MODEL}/qfura-layout_${quant_block_layout}-lr_${lr}-decomp_${decomp_mode}-seed_${seed}`.
- Terminal eval via `bash_scripts/eval_commonsense.sh` is inherited unchanged.

### 4.9 Memory budget

Llama-3-70B qfura on H100 (94 GB):

| Component | Size |
|---|---|
| Frozen large core, NF4 | ~35 GB |
| Trainable small core + `btt_s`, bf16 | ~2.8 GB |
| Gradients on trainable only, bf16 | ~2.8 GB |
| `PagedAdamW8bit` state on trainable only | ~2.8 GB |
| Activations with gradient checkpointing | ~6–10 GB |
| CUDA/cuBLAS/misc | ~2–4 GB |
| **Total** | **~52–56 GB** |

Leaves 35+ GB of headroom for transient allocations during optimizer step. QLoRA's precedent of tuning a 65B model on a single 48 GB GPU is the anchor — qfura has both a similar-sized model and a substantially larger GPU.

### 4.10 Dependency changes

- Add `bitsandbytes>=0.43` to `pyproject.toml`. Tested against `bitsandbytes==0.43.x` on CUDA 12.1 with a compatible `torch`. `uv sync` must succeed after the addition.

## 5. Test matrix

**Unit tests** (all run in CI, no large-model load):

1. `tests/test_qbtt_quant_error.py` — synthetic-layer forward and backward error thresholds per 4.6.
2. `tests/test_qbtt_forward_shape.py` — `QBTTLayer.forward` output shape matches `BTTLayer.forward` output shape; values match within quantization tolerance on a small random model.
3. `tests/test_qbtt_conversion.py` — `convert_btt_to_qbtt_` round-trip: the dequanted frozen core matches the pre-quantization bf16 core within tolerance.
4. `tests/test_qbtt_gradient_flow.py` — after conversion, only the small core and `btt_s` receive gradients; the frozen core's gradient is `None`.
5. `tests/test_qbtt_fused_step2_compat.py` — with `FURA_FUSED_STEP2=1`, `QBTTLayer.forward` matches the non-fused qfura path within tolerance.
6. `tests/test_finetune_qfura_cli.py` — asserts `--train_position != "small"` raises; asserts missing `--gradient_checkpointing` raises; asserts unknown `--quant_block_layout` raises.

**Benchmark artifacts** (manual run, committed outputs):

7. `analysis/bench_qbtt_quant_error.py` → `docs/reports/qfura-quant-error.md`. Default 8B run; 70B optional follow-up.

**Integration artifacts** (manual run, committed outputs):

8. `docs/exp_results/qfura.md` — full Llama-3-70B qfura commonsense run per layout (flat + per_core_block). Includes 8 commonsense-reasoning benchmark accuracies, wall-clock training time, peak GPU memory, W&B run IDs.

**Sanity-check execution order:**

1. Implement `QBTTLayer`, `quantize_frozen_core_`, `convert_btt_to_qbtt_` in `btt_layer.py`. Get unit tests 1–5 green.
2. Implement `ref/LIFT/src/finetune_qfura.py` and its shell script. Get unit test 6 green.
3. Run `analysis/bench_qbtt_quant_error.py` on Llama-3-8B. Commit `docs/reports/qfura-quant-error.md`. Select default layout; update this spec and the shell script.
4. Smoke-run `finetune_commonsense_qfura.sh` on Llama-3-8B for 100 steps; verify loss decreases and GPU memory is stable.
5. Full Llama-3-70B qfura commonsense runs (one per layout). Write `docs/exp_results/qfura.md`.

## 6. Open items deferred to implementation plan

- Exact NF4 block_size choice for `per_core_block` layout when `rank*n*a` (or `b*m*rank`) does not cleanly divide bnb's default block_size of 64. Pad-and-mask vs. shrink-block-size is a plan-level detail.
- Whether to fold `btt_s` into the frozen core before quantization (when `s_merged_to="frozen"`) or keep `btt_s` as a separate trainable vector. Current design keeps `btt_s` trainable always; the `s_merged_to="frozen"` case means `btt_s` is still present as a vector but initialized to ones. Plan should verify this matches existing `BTTLayer` behavior.
- Whether the per-core-block layout needs a custom Triton kernel for the dequant loop, or if a Python `for` + `torch.stack` is acceptable. Benchmark wall-clock at smoke-test stage will decide.
