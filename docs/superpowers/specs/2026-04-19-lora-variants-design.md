# LoRA Variants: DoRA, PiSSA, MiLoRA, RandLoRA, LIFT

Date: 2026-04-19

## Problem

`run_rl.py` and `run_sft.py` only support vanilla LoRA, BlockTT, SVD, and full fine-tuning. We want to add five additional methods so we can compare them on the same RL/SFT loop:

- **DoRA** — weight-decomposed LoRA (`use_dora=True`).
- **PiSSA** — LoRA initialized from the **top-r** singular components of the base weight, with the residual installed as the new base.
- **MiLoRA** — same structure as PiSSA but uses the **bottom-r** singular components instead. Encourages learning in the minor subspace, preserves world knowledge.
- **RandLoRA** — full-rank update via learned diagonal scalings of fixed shared random bases.
- **LIFT** — sparse fine-tuning of the dense model: a top-5%-magnitude mask per `nn.Linear`, recomputed every `update_interval` steps from a low-rank approximation of the running weights. Implemented as a custom `SparseAdamW` optimizer.

`ref/LIFT/src/finetune_lora.py` already supports `lora`/`dora`/`pissa` via PEFT but is missing `milora` and `randlora`, and has a dead `hira` branch that calls an undefined function.

## Non-goals

- HiRA (dropped during brainstorming).
- Unifying the LIFT script's LoRA-variant code with the main repo's. The two surfaces evolve independently.
- KD (`legacy/run_kd.py`) — same patterns will apply but not in this PR.
- `run_rl_dapo.py` — referenced in CLAUDE.md but does not exist on disk; not creating it.
- Changing existing `lora` / `lora_full` / `blocktt` / `svd` / `full` semantics.
- Saving LIFT optimizer state across checkpoint restarts (matches LIFT upstream's behavior).

## Architecture

Two independent surfaces, no shared module between them:

1. **Main repo (`run_rl.py` + `run_sft.py`)**: five new `--train-mode` values, dispatched in the existing `prepare_model` / equivalent block. Helpers (e.g. `apply_milora_init_`) live next to `prepare_model` in `run_sft.py`.
2. **`ref/LIFT/src/finetune_lora.py`**: two new branches (`milora`, `randlora`) added to the existing `adapter_name` cascade. `apply_milora_init_` is **duplicated** here as a small local helper — drift risk is low (~30 lines, deterministic math) and it keeps LIFT vendoring-clean.
3. **`SparseAdamW` vendoring**: copied from `ref/LIFT/src/sparseAdam.py` to `optim/sparse_adam.py`. The original LIFT file stays put; the copy carries a provenance header.

## Main-repo changes

### New `--train-mode` values

`MODE_DEFAULTS` gains five entries:

| mode | default lr | wandb_project | micro_batch_size | grad_accum |
|---|---|---|---|---|
| `dora` | `9e-5` | `math-grpo-dora` | 2 | 128 |
| `pissa` | `9e-5` | `math-grpo-pissa` | 2 | 128 |
| `milora` | `9e-5` | `math-grpo-milora` | 2 | 128 |
| `randlora` | `9e-5` | `math-grpo-randlora` | 2 | 128 |
| `lift` | `1e-4` | `math-grpo-lift` | 4 | 64 |

The four LoRA-family modes mirror `lora`'s defaults; `lift` mirrors `full`'s (it trains the dense model). `--train-mode` `choices` extends to:

`["full","lora","lora_full","dora","pissa","milora","randlora","lift","blocktt","svd"]`

### New CLI flags

| flag | default | valid for |
|---|---|---|
| `--randlora-projection-prng-key` | `0` | `randlora` only |
| `--lift-lora-rank` | `128` | `lift` only |
| `--lift-filter-rank` | `128` | `lift` only |
| `--lift-update-interval` | `400` | `lift` only |

`--lora-rank`, `--lora-alpha`, `--trainable-type` become valid for the full LoRA family: `{lora, lora_full, dora, pissa, milora, randlora}`. `--vllm-url` is **rejected** for `pissa` and `milora` (forced local rollout, see below). `--optimizer muon` is **rejected** for `lift` (LIFT *is* the optimizer choice).

### Branching in `prepare_model` (run_sft.py)

Inserted alongside the existing `elif args.train_mode == "lora":` block:

```python
elif args.train_mode == "dora":
    target_modules = get_lora_target_modules(args.trainable_type)
    peft_config = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha,
                             use_dora=True, target_modules=target_modules)
    model = get_peft_model(model, peft_config)

elif args.train_mode == "pissa":
    target_modules = get_lora_target_modules(args.trainable_type)
    peft_config = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha,
                             init_lora_weights="pissa_niter_4",
                             lora_dropout=0, target_modules=target_modules)
    model = get_peft_model(model, peft_config)

elif args.train_mode == "milora":
    target_modules = get_lora_target_modules(args.trainable_type)
    peft_config = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha,
                             lora_dropout=0, target_modules=target_modules)
    model = get_peft_model(model, peft_config)
    apply_milora_init_(model, rank=args.lora_rank)

elif args.train_mode == "randlora":
    from peft import RandLoraConfig
    target_modules = get_lora_target_modules(args.trainable_type)
    peft_config = RandLoraConfig(r=args.lora_rank, randlora_alpha=args.lora_alpha,
                                 projection_prng_key=args.randlora_projection_prng_key,
                                 target_modules=target_modules)
    model = get_peft_model(model, peft_config)

elif args.train_mode == "lift":
    # Dense model, all params trainable. No PEFT wrapping.
    pass
```

The same five-branch dispatch is mirrored in `run_rl.py` at the model-construction site (`run_rl.py` does not currently import `prepare_model` from `run_sft.py` for LoRA; both files have parallel construction blocks).

### MiLoRA init (`apply_milora_init_`)

PEFT 0.17.1 has no `init_lora_weights="milora"` so we run a one-shot SVD pass after `get_peft_model()`.

For each target `nn.Linear` with weight `W ∈ R^{out×in}`:

1. SVD: `W = U Σ V^T`, singular values descending (torch default).
2. Slice the **last r**: `U_r = U[:, -r:]`, `S_r = Σ[-r:]`, `V_r = V[:, -r:]`.
3. Build LoRA factors so `(α/r) · lora_B @ lora_A = U_r diag(S_r) V_r^T`:
   - `lora_A = sqrt(S_r) · V_r^T · (r/α)^{1/2}` — shape `(r, in)`
   - `lora_B = U_r · sqrt(S_r) · (r/α)^{1/2}` — shape `(out, r)`
   - The `(r/α)^{1/2}` factor on each side absorbs PEFT's internal `α/r` scaling.
4. Replace base: `W ← W − U_r diag(S_r) V_r^T` (the top-`(n−r)` residual).

This is PiSSA's structure exactly, with bottom-r slicing instead of top-r.

```python
@torch.no_grad()
def apply_milora_init_(peft_model, *, rank: int) -> None:
    from peft.tuners.lora import LoraLayer
    first_check = True
    for name, module in peft_model.named_modules():
        if not isinstance(module, LoraLayer):
            continue
        base = module.get_base_layer()
        W = base.weight.data
        dtype, device = W.dtype, W.device
        U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
        r = rank
        U_r, S_r, Vh_r = U[:, -r:], S[-r:], Vh[-r:, :]
        sqrt_S = S_r.sqrt()
        adapter_name = list(module.lora_A.keys())[0]
        alpha = module.lora_alpha[adapter_name]
        scale_correction = (r / alpha) ** 0.5
        lora_A = (sqrt_S.unsqueeze(1) * Vh_r) * scale_correction
        lora_B = (U_r * sqrt_S.unsqueeze(0)) * scale_correction
        residual = W.float() - U_r @ torch.diag(S_r) @ Vh_r
        module.lora_A[adapter_name].weight.data.copy_(lora_A.to(dtype=dtype, device=device))
        module.lora_B[adapter_name].weight.data.copy_(lora_B.to(dtype=dtype, device=device))
        base.weight.data.copy_(residual.to(dtype=dtype, device=device))
        if first_check:
            reconstructed = (alpha / r) * lora_B.to(dtype=torch.float32) @ lora_A.to(dtype=torch.float32) + residual
            rel_err = torch.linalg.norm(reconstructed - W.float()) / torch.linalg.norm(W.float())
            assert rel_err < 1e-3, f"MiLoRA init reconstruction error too high: {rel_err:.2e}"
            first_check = False
```

**Why full SVD, not randomized:** randomized SVD (PiSSA's `pissa_niter_k`) recovers top components well but is dominated by error in the bottom — exactly the components MiLoRA needs. Full SVD is the only correct choice. Cost: ~50ms per 4096×4096 layer on H100 in fp32, ~15s total for ~300 layers. One-shot at init.

### Rollout backend routing (run_rl.py)

```python
def resolve_lora_rollout_backend(train_mode, vllm_url):
    if train_mode == "lora_full":            return "local_inproc"
    if train_mode in {"pissa", "milora"}:    return "local_inproc"   # forced
    if train_mode in {"lora", "dora", "randlora"}:
        return "http" if is_vllm_http_available(vllm_url) else "local_inproc"
    return None
```

PiSSA and MiLoRA modify the base weight at init, so the running vLLM server (loaded with the unmodified base) is out of sync with the trainer. Rather than wire up PEFT's `path_initial_model_for_weight_conversion` save trick, we force these two modes to local in-process rollout, which calls `merge_adapter()` on the (modified-base + adapter) model and pushes the resulting dense weights into vLLM via `load_weights`. This produces the correct effective weight transparently.

`build_lora_local_generators` already handles DoRA's magnitude vector and any PEFT layer that supports `merge_adapter()`. RandLoRA's `merge()` materializes its delta via `get_delta_weight()` — supported. No code change inside the local generator beyond accepting the new modes.

For the HTTP path (used by `lora`, `dora`, `randlora`), `model.save_pretrained()` writes a standard PEFT adapter that vLLM's `/v1/load_lora_adapter` endpoint accepts. `normalize_lora_merged_weight_name`'s skip-list adds `randlora_lambda`, `randlora_gamma`, `randlora_m` for safety, even though the HTTP path doesn't go through it.

The main loop's `if train_mode in {"full"}: build_local_vllm_generators(...)` extends to `if train_mode in {"full", "lift"}:`. LIFT trains the dense model, so it uses the same dense-rollout path as `full`.

### LIFT: `--train-mode lift`

`SparseAdamW` is vendored from `ref/LIFT/src/sparseAdam.py` to `optim/sparse_adam.py` with a provenance header recording the source path and the commit SHA of `ref/LIFT` at the time of vendoring (filled in by the implementer via `git -C ref/LIFT rev-parse HEAD`).

`build_optimizer` adds a branch keyed on `args.train_mode == "lift"` that overrides `args.optimizer`:

```python
if args.train_mode == "lift":
    from optim.sparse_adam import SparseAdamW
    weights_with_mask, decay_ids = [], []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and "lm_head" not in name and mod.weight.requires_grad:
            weights_with_mask.append(mod.weight)
            decay_ids.append(id(mod.weight))
    other_decay, other_nodecay = [], []
    no_decay_names = ()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in decay_ids:
            continue
        (other_nodecay if any(nd in name for nd in no_decay_names) else other_decay).append(p)

    param_groups = [
        {"params": weights_with_mask, "weight_decay": 0.0,
         "rank": args.lift_lora_rank, "filter_rank": args.lift_filter_rank,
         "update_proj_gap": args.lift_update_interval, "group_name": "weights_with_mask"},
        {"params": other_decay,    "weight_decay": 0.0, "group_name": "other_params_w_decay"},
        {"params": other_nodecay,  "weight_decay": 0.0, "group_name": "other_params"},
    ]
    optimizer = SparseAdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
```

LIFT-specific wandb logging: every `update_interval` steps, log mean/max sparsity across the `weights_with_mask` group and number of mask flips since last update. Confirms the mask is being recomputed.

**Memory:** LIFT trains all `nn.Linear` weights with full-precision optimizer states for the masked subset (5%). Per the LIFT paper, comparable to LoRA r=128. Fits H100 for both Qwen3-1.7B (RL) and Qwen3-4B (SFT); confirm via smoke test.

### Flag-validation table

`mode_to_flag_sets` in `run_rl.py` extends:

- LoRA-family flag set (`--lora-rank`, `--lora-alpha`, `--trainable-type`) becomes valid for `{lora, lora_full, dora, pissa, milora, randlora}`. `--vllm-url` valid for `{lora, dora, randlora}` only — rejected for `pissa`/`milora`.
- New `randlora` set: `["--randlora-projection-prng-key"]`.
- New `lift` set: `["--lift-lora-rank", "--lift-filter-rank", "--lift-update-interval"]`.
- `--train-position`, `--s-merged-to`, `--decomp-mode`, `--blocktt-rank`, `--blocktt-normalize-after-update`, `--blocktt-factorize-by-head` continue to be rejected for the new modes.

### `compute_run_name` additions

Each new mode gets a name template:

- `dora`: `f"{model_id}_{lr:.1e}_r{lora_rank}_dora"`
- `pissa`: `f"{model_id}_{lr:.1e}_r{lora_rank}_pissa"`
- `milora`: `f"{model_id}_{lr:.1e}_r{lora_rank}_milora"`
- `randlora`: `f"{model_id}_{lr:.1e}_r{lora_rank}_a{lora_alpha}_randlora"`
- `lift`: `f"{model_id}_{lr:.1e}_r{lift_lora_rank}_int{lift_update_interval}_lift"`

### Checkpointing — integration with `save_merged_checkpoint`

Recent commits added `save_merged_checkpoint` (run_rl.py:799) which produces a **plain HuggingFace checkpoint** (no LoRA adapters, no factored cores) so `eval_rl.py` and `math_verify_eval` can load it via vanilla `AutoModelForCausalLM.from_pretrained`. The function currently dispatches on `train_mode` with branches for `full`, `{lora, lora_full}`, `{blocktt, svd}`, and raises on anything else. The new modes must be added:

- `dora`, `pissa`, `milora`, `randlora` → reuse the existing `{lora, lora_full}` branch (`merge_adapter()` → `get_base_model().save_pretrained()`). This is correct for **all four**: PEFT's `merge_adapter` produces the right effective dense weight in every case (DoRA folds in the magnitude vector; PiSSA/MiLoRA's `merge_adapter` adds the adapter delta to the residual base, recovering the full effective `W`; RandLoRA's `merge` materializes its delta via `get_delta_weight`). The on-disk result is a vanilla HF model, so no separate "save base + adapter" wart is needed for PiSSA/MiLoRA. **This supersedes the earlier draft of this section.**
- `lift` → reuse the `full` branch (`model.save_pretrained(ckpt_dir)`). LIFT trains the dense model; no adapters to merge.

Concretely:

```python
def save_merged_checkpoint(model, tokenizer, ckpt_dir, train_mode, args):
    os.makedirs(ckpt_dir, exist_ok=True)
    if train_mode in {"full", "lift"}:
        model.save_pretrained(ckpt_dir)
    elif train_mode in {"lora", "lora_full", "dora", "pissa", "milora", "randlora"}:
        model.merge_adapter()
        try:
            base = model.get_base_model()
            base.save_pretrained(ckpt_dir)
        finally:
            model.unmerge_adapter()
    elif train_mode in {"blocktt", "svd"}:
        # unchanged
        ...
    else:
        raise ValueError(f"Unknown train_mode for save_merged_checkpoint: {train_mode}")
    tokenizer.save_pretrained(ckpt_dir)
```

`save_checkpoint` (the wrapper) already routes through `save_merged_checkpoint` when `--enable-merged-ckpt` is set (default true), so no change needed there.

LIFT optimizer mask state is **not** saved; restart reinitializes the mask on the first `update_interval` step. Documented limitation.

### Math-verify post-training eval — in-memory weight export

The `--enable-math-verify` block (run_rl.py:1867) currently has explicit `train_mode` branches for in-memory hot-swap into the existing vLLM:

- `{blocktt, svd}` → `export_weights_for_vllm(model)`
- `{lora, lora_full}` → `merge_adapter()` + iterate `get_base_model().named_parameters()` with `normalize_lora_merged_weight_name`
- else → raw `named_parameters()`

Updated dispatch:

- `{dora, pissa, milora, randlora}` → reuse the `{lora, lora_full}` branch (same `merge_adapter` + `normalize_lora_merged_weight_name` logic). For RandLoRA, the `normalize_lora_merged_weight_name` skip-list is extended (see Rollout backend section) so its trainable buffers/params are not pushed as base-model weights.
- `lift` → reuse the `else` branch (raw `named_parameters()` — dense model).

The HTTP-trained fallback (line 1912, "spin up a fresh in-process LLM from disk") works for all new modes as long as `save_merged_checkpoint` produced a valid HF checkpoint. The check on line 559 (`--enable-math-verify` without `--enable-merged-ckpt` warns for `train_mode != "full"`) is updated to allow `lift` too: `train_mode not in {"full", "lift"}`. PiSSA/MiLoRA stay flagged because their HTTP path is forbidden anyway, but the warning is still accurate (without merged ckpt, there's no loadable on-disk model).

### `eval_rl.py` (standalone post-training eval)

`eval_rl.py` (added in commit 5d8632e) loads a checkpoint dir via `AutoModelForCausalLM.from_pretrained` and runs `math_verify_eval`. It is **mode-agnostic** — it never inspects `train_mode`, only the on-disk checkpoint. Once `save_merged_checkpoint` correctly handles the five new modes, `eval_rl.py` works for all of them with no code change.

### HTTP-lora adapter save directory

`build_lora_http_generators` (run_rl.py:935) writes adapters to `{run_dir}/lora_adapters/step={step}` (commit f240721). DoRA and RandLoRA save standard PEFT adapter directories that vLLM's `/v1/load_lora_adapter` accepts, so they reuse this path unchanged. PiSSA and MiLoRA never reach this code (forced to local rollout).

## `ref/LIFT/src/finetune_lora.py` changes

**Existing `lora` / `dora` / `pissa` branches are not touched.**

### New `milora` branch

Inserted after the `pissa` branch:

```python
elif args.adapter_name == "milora":
    print("MiLoRA Init")
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        target_modules=args.target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    apply_milora_init_(model, rank=args.lora_r)
    model.print_trainable_parameters()
```

`apply_milora_init_` is a local helper inside `finetune_lora.py` (~30 lines, identical math to the main-repo copy). Duplicated rather than imported — keeps LIFT vendoring-clean.

### New `randlora` branch

```python
elif args.adapter_name == "randlora":
    print("RandLoRA Init")
    from peft import RandLoraConfig
    config = RandLoraConfig(
        r=args.lora_r,
        randlora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        projection_prng_key=args.randlora_projection_prng_key,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
```

### CLI flag additions

```python
parser.add_argument("--randlora_projection_prng_key", type=int, default=0,
                    help="Seed for RandLoRA's shared random bases (default: 0)")
```

Reuses existing `--lora_r` and `--lora_alpha`.

### Cascade list updates

Two existing lines extend:

- Line 390: `if args.adapter_name in ["lora", "dora", "pissa", "milora", "randlora"]:`
- Line 651: same list (save/eval routing).

### Dead code removal

Lines 425–431 (the `elif args.adapter_name == "hira":` branch) call `convert_layer_to_hira`, which is undefined and unimported in this repo. Delete the entire branch — it would crash at runtime, and we explicitly dropped HiRA.

### New bash scripts

Four new scripts under `ref/LIFT/bash_scripts/`, copied from the existing `finetune_math_lora.sh` / `finetune_commonsense_lora.sh` and adjusted:

| script | adapter_name | lora_r | lora_alpha | extra |
|---|---|---|---|---|
| `finetune_math_milora.sh` | `milora` | 128 | 128 | — |
| `finetune_math_randlora.sh` | `randlora` | 32 | 640 | `--randlora_projection_prng_key=${seed}` |
| `finetune_commonsense_milora.sh` | `milora` | 128 | 128 | — |
| `finetune_commonsense_randlora.sh` | `randlora` | 32 | 640 | `--randlora_projection_prng_key=${seed}` |

PiSSA-like defaults for MiLoRA (shared structure). RandLoRA uses PEFT's published defaults (`r=32`, `randlora_alpha=640`).

## Testing

### Unit tests

- `tests/test_milora_init.py`: build a tiny model (one `nn.Linear`), run `apply_milora_init_`, assert `‖(α/r)·B@A + W_residual − W_original‖_F / ‖W_original‖_F < 1e-4`. Same test runs against the main-repo copy and the LIFT-local copy (independent functions, identical behavior expected).
- `tests/test_run_rl_cli.py`: extend with one case per new `--train-mode` value verifying flag validation (correct flags pass, mode-incompatible flags raise).
- `tests/test_run_sft_cli.py`: same pattern (if it exists; otherwise add).
- `tests/test_sparse_adam_smoke.py`: instantiate `SparseAdamW` on a 2-layer MLP, run one step, verify only the masked entries update.

### Smoke runs (manual, not CI)

- `uv run run_sft.py --train-mode dora --lora-rank 8 --no-wandb` (5 steps), repeat for `pissa`, `milora`, `randlora`, `lift`.
- `uv run run_rl.py --train-mode milora --lora-rank 8 --no-wandb --enable-save-ckpt --enable-merged-ckpt --enable-math-verify --n-grpo-steps 2` — verifies (a) forced local rollout, (b) `save_merged_checkpoint` produces a loadable HF checkpoint, (c) in-memory math-verify hot-swap works after `merge_adapter`.
- Same for `dora`, `pissa`, `randlora` (each must produce a vanilla HF checkpoint loadable by `eval_rl.py`).
- `uv run run_rl.py --train-mode lift --no-wandb --enable-save-ckpt --enable-merged-ckpt --enable-math-verify --n-grpo-steps 2` — verifies LIFT routes through the dense `full`-style branches in both `save_merged_checkpoint` and the math-verify hot-swap.
- `uv run run_rl.py --train-mode dora --lora-rank 8 --no-wandb` against a running vLLM server — verifies HTTP path works for DoRA, including the `lora_adapters/step=*` save directory.
- `uv run eval_rl.py --checkpoint <run-dir>/step=2` for one checkpoint per mode — verifies mode-agnostic eval works end-to-end.
- `bash ref/LIFT/bash_scripts/finetune_math_milora.sh` with reduced epochs — verifies LIFT-side init.

## Open risks

- **vLLM RandLoRA HTTP support**: vLLM's `/v1/load_lora_adapter` endpoint may not yet recognize RandLoRA adapter weights. Mitigation: smoke-test the HTTP path; if it fails, fall back to `local_inproc` for `randlora` too (one-line change to `resolve_lora_rollout_backend`).
- **MiLoRA scaling correction**: the `(r/α)^{1/2}` factor depends on PEFT's `LoraLayer` applying `lora_alpha / r` as the runtime scaling. PEFT's default `use_rslora=False` does this; the assert in `apply_milora_init_` catches any drift.
- **LIFT optimizer state on resume**: not saved. If we ever need true resume, that's a follow-up.
