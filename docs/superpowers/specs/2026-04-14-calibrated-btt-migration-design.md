# Calibrated BTT Migration Design

**Date:** 2026-04-14
**Status:** Draft for review

## Summary

Enable the calibrated BTT decomposition methods from `src/compress` (`btt_llm_v2`, `btt_llm_v2_bp`, `btt_llm_v2_combined`, `btt_twosteps`) as an additional path inside the existing `--train-mode blocktt` flow in `run_sft.py`, `run_rl.py`, `run_rl_dapo.py`, and `ref/LIFT/src/finetune_blocktt.py`. Legacy `btt_layer.py` and `svd_layer.py` paths are left untouched. A new `--calib-mode` flag selects the calibrated variant; when it is set to any non-`none` value, decomposition is routed through `compress.decompose_with_loader` with a calibration loader built to match real training batches (forward activations and backward gradients during calibration mirror the real training distribution).

## Non-goals

- SVD calibrated variants (`svd_llm`, `svd_llm_v2`, `svd_llm_v2_bp`, `svd_llm_v2_combined`, `svd_als`, `svd_twosteps`).
- Any changes to the legacy `btt_layer.py` / `svd_layer.py` code paths beyond one minor extension (see §3.6).
- `legacy/run_kd.py`.
- LIFT's `finetune_lora.py`, `finetune_sft.py`, `finetune_s2ft.py`.

## 1. CLI surface

### 1.1 New flags

Added to `run_sft.py`, `run_rl.py`, `run_rl_dapo.py` (hyphen style) and `ref/LIFT/src/finetune_blocktt.py` (underscore style). Same semantics across all four.

| Flag | Type | Default | Description |
|---|---|---|---|
| `--calib-mode` | choice | `none` | `none` keeps legacy BTT path. `v2` → `btt_llm_v2`; `v2_bp` → `btt_llm_v2_bp`; `v2_combined` → `btt_llm_v2_combined`; `twosteps` → `btt_twosteps`. |
| `--calib-source` | choice | `c4` | `c4` / `traces` / `training_data`. Only meaningful when `--calib-mode != none`. |
| `--calib-traces-path` | str | `None` | Required when `--calib-source=traces`. |
| `--calib-num-seqs` | int | `128` | Passed to `DecompositionConfig`. |
| `--calib-max-length` | int | `2048` | Passed to `DecompositionConfig`. |
| `--calib-seed` | int | `3` | Passed to `DecompositionConfig`. |
| `--calib-batch-size` | int | `8` | Batch size for the calibration DataLoader. |

### 1.2 Modified flag: `--blocktt-rank`

Extended to accept a float.

- `--calib-mode=none` (legacy path): `"full"` or positive int. Float raises.
- `--calib-mode != none` (calibrated path): `"full"` → ratio `1.0`; float in `(0.0, 1.0]` → ratio passthrough; int raises with a clear message.

### 1.3 Flag compatibility when `--calib-mode != none`

Valid and routed through to `DecompositionConfig`:

- `--train-position {small, large, both}` → `DecompositionConfig.train_position`.
- `--trainable-type {all, mlp, attn}` → translated to `skip_layers` (inverse of `get_blocktt_target_module_names()`); `lm_head` always included in skip list.
- `--decomp-mode {square, input_one_block, output_one_block}` → `DecompositionConfig.decomp_mode`.
- `--blocktt-rank full|<float>` per §1.2.
- `--s-merged-to` → passed through (see §6).
- `--blocktt-factorize-by-head` → passed through (see §6).
- `--blocktt-normalize-after-update` → unchanged (optimizer-level; see §3.6).

### 1.4 Validation precedence

1. Existing `--train-mode`-level validation unchanged.
2. In main-repo scripts: `--calib-mode != none` with `--train-mode != blocktt` raises ("`--calib-mode` only valid with `--train-mode blocktt`"). LIFT's `finetune_blocktt.py` is BTT-only, so this check is trivially satisfied.
3. `--calib-mode != none`: `--calib-source=traces` requires `--calib-traces-path`; integer `--blocktt-rank` is rejected.
4. `--calib-mode=none`: any of the new `--calib-*` flags being explicitly passed raises (reuses existing `_flag_was_passed` pattern).

## 2. Shared helper module: `compress_integration.py`

New file at repo root. On import, prepends `<repo_root>/src` to `sys.path` if not already present so that `from compress...` works.

### 2.1 Exported surface

```python
# CLI wiring
def add_calibrated_btt_args(parser, *, hyphen_style: bool = True) -> None
def validate_calibrated_btt_args(args, *, argv: list[str], hyphen_style: bool = True) -> None

# Config translation
def build_decomposition_config(args, *, hyphen_style: bool = True) -> DecompositionConfig

# Calibration loader builders
def build_training_data_calib_loader(
    dataset, collate_fn, *, num_seqs: int, batch_size: int, seed: int,
) -> DataLoader

def build_rl_rollout_calib_loader(
    *, rl_rollout_fn, tokenizer,
    num_seqs: int, batch_size: int, max_length: int, seed: int,
) -> DataLoader
# rl_rollout_fn is a zero-arg-or-seed-arg closure that returns a list of
# (prompt_text, completion_text) pairs. The caller (run_rl.py) constructs it
# with the existing training rollout config captured inside.

def build_calib_loader(
    args, *, tokenizer, training_dataset=None, training_collate_fn=None,
    rl_rollout_fn=None,
) -> DataLoader | None
# Dispatches on args.calib_source. For training_data: prefers
# (training_dataset, training_collate_fn) when given (SFT/LIFT);
# otherwise uses rl_rollout_fn (RL).

# Decomposition wire-up
def apply_calibrated_btt(
    model, args, *, calib_loader, device: str | None = None,
) -> tuple[nn.Module, dict]

# RL materialization
def materialize_calibrated_btt_weights(model) -> dict[str, torch.Tensor]
def restore_calibrated_btt_weights(model, saved_state: dict) -> None

# Checkpoint save/load for eval
def save_calibrated_btt_checkpoint(model, out_dir: str) -> None
def load_calibrated_btt_for_eval(model, checkpoint_dir: str) -> nn.Module
```

### 2.2 Key translation logic (`build_decomposition_config`)

- `--calib-mode` → `train_mode`: `{v2, v2_bp, v2_combined, twosteps}` → `{btt_llm_v2, btt_llm_v2_bp, btt_llm_v2_combined, btt_twosteps}`.
- `--blocktt-rank` → `compression_ratio` per §1.2.
- `--trainable-type` → `skip_layers`: call `get_blocktt_target_module_names(trainable_type)` to get the *include* set, invert against all `nn.Linear` leaf names in the model to produce a comma-separated *exclude* string. `lm_head` always in skip list.
- `--s-merged-to`, `--blocktt-factorize-by-head` → passed through on the new `DecompositionConfig` fields added in §6.
- `--calib-num-seqs`, `--calib-max-length`, `--calib-seed`, `--decomp-mode`, `--train-position` → direct passthrough.

### 2.3 `build_calib_loader` dispatch

- `calib_source=c4` → `compress.build_c4_calib_loader`.
- `calib_source=traces` → `compress.build_traces_jsonl_calib_loader`.
- `calib_source=training_data`:
  - If `training_dataset` + `training_collate_fn` given (SFT, LIFT): `build_training_data_calib_loader`.
  - Else if `rl_rollout_fn` given (RL): call it to produce `(prompt, completion)` pairs; build batches with label-masked prompt tokens.
  - Else: raise.

### 2.4 Underscore variant for LIFT

`add_calibrated_btt_args(parser, hyphen_style=False)` registers underscore-named flags. `validate_calibrated_btt_args` and `build_decomposition_config` look up attributes accordingly. Single code path; style flag only affects arg-name registration and attribute lookup.

## 3. SFT integration (`run_sft.py`)

### 3.1 Argparse

Call `add_calibrated_btt_args(parser, hyphen_style=True)` near the existing `--blocktt-*` flag block.

### 3.2 Post-parse validation

Call `validate_calibrated_btt_args(args, argv=argv, hyphen_style=True)` inside the existing `_flag_was_passed` validation block.

### 3.3 Dataset ordering

`load_dataset("HuggingFaceH4/no_robots", ...)` and `build_collate_fn(tokenizer)` must run *before* `prepare_model` so the calib loader can draw real training batches. Current code loads dataset after `prepare_model`; reorder to load first. `prepare_model` gains new kwargs `train_dataset=None` and `collate_fn=None` which are only used when `args.calib_mode != "none"`. No behavior change for `--calib-mode=none`.

### 3.4 Decomposition branch

Inside `prepare_model`, before the legacy `blocktt` branch:

```python
if args.train_mode == "blocktt" and args.calib_mode != "none":
    calib_loader = build_calib_loader(
        args, tokenizer=tokenizer,
        training_dataset=train_dataset,
        training_collate_fn=collate_fn,
    )
    model, stats = apply_calibrated_btt(model, args, calib_loader=calib_loader)
else:
    # existing blocktt / svd / lora / full logic unchanged
```

### 3.5 Calib loader shape

`build_training_data_calib_loader` takes a `Subset` of the first `calib_num_seqs` indices from `train_dataset` (deterministic under `calib_seed`), wraps it in `DataLoader(subset, batch_size=calib_batch_size, collate_fn=collate_fn)`. The existing `collate_fn` produces `input_ids`/`attention_mask`/`labels`; contract satisfied with no reformatting. This is the "calibration batches look exactly like training batches" principle.

### 3.6 `--blocktt-normalize-after-update`

Extend `normalize_trainable_blocktt_cores_` in `btt_layer.py` to also recognize `compress.btt.btt_linear.BTTLinear` via duck-typing on `btt_l` / `btt_r` attributes. This is the one touch-point in legacy files; semantics for legacy `BTTLayer` are unchanged. Shared across SFT, RL, and LIFT.

### 3.7 Checkpointing

SFT already saves `step={N}/model.safetensors` via `{n: p.detach().cpu() for n, p in model.named_parameters()}`. Works for calibrated BTT since `btt_l` / `btt_r` are registered parameters.

Additionally, when `args.calib_mode != "none"`, call `save_calibrated_btt_checkpoint(model, step_dir)` which also writes `btt_topology.json` alongside `model.safetensors`. See §6.3 for topology format.

## 4. RL integration (`run_rl.py`, `run_rl_dapo.py`)

### 4.1 Argparse and validation

Same as SFT: `add_calibrated_btt_args(parser, hyphen_style=True)` + `validate_calibrated_btt_args(...)`.

### 4.2 Rollout-based calibration loader

RL training prompts have no paired completions at decomposition time. `build_rl_rollout_calib_loader` performs a one-time HF `model.generate()` pass on the dense base model before decomposition, using the **exact same rollout configuration** as the RL training loop (sampling params, `max_new_tokens`, temperature, top-p, stop tokens, prompt-template). This shared config is captured via a closure passed into `build_calib_loader` as `rl_rollout_fn`; the implementation locates and reuses the existing training rollout config object rather than duplicating it.

For each (prompt, completion) pair, the loader tokenizes identically to how training tokenizes its policy inputs, and builds `input_ids` / `attention_mask` / `labels` where prompt tokens are masked to `-100`. This mirrors the RL loss mask so backward-calibration gradients match real training gradients.

### 4.3 Decomposition branch

After loading the dense base model and before any factored-layer conversion:

```python
if args.train_mode == "blocktt" and args.calib_mode != "none":
    rollout_fn = make_rl_calib_rollout_fn(base_model, tokenizer, train_dataset, rollout_cfg)
    calib_loader = build_calib_loader(
        args, tokenizer=tokenizer, rl_rollout_fn=rollout_fn,
    )
    model, stats = apply_calibrated_btt(model, args, calib_loader=calib_loader)
else:
    # existing logic unchanged
```

Order: load dense model → run calib rollouts → decompose.

### 4.4 vLLM materialize / restore

Extend the existing `materialize` / `restore` helpers in `run_rl.py` to also recognize `compress.btt.btt_linear.BTTLinear`, via the helpers exported from `compress_integration.py` (§2.1). The existing flow (materialize dense → patch into model → local vLLM generation → restore factored layers) is unchanged for calibrated BTT.

If `BTTLinear` does not already expose `materialize_dense_weight()`, add one as part of the `src/compress` additions in §6. Implementation-time verification.

**vLLM adapter-upload path:** N/A for calibrated BTT (as for legacy BTT). Calibrated BTT always takes the materialize-and-generate-locally path.

### 4.5 Training loop, checkpointing

Training loop is unchanged; GRPO/DAPO operate on `requires_grad=True` parameters, which `configure_btt_trainability` has set on `btt_l`/`btt_r` per `--train-position`. Checkpointing uses the same topology-metadata mechanism as SFT (§3.7).

### 4.6 `run_rl_dapo.py`

Receives identical integration to `run_rl.py`. Kept in sync to avoid drift.

## 5. LIFT integration (`ref/LIFT/src/finetune_blocktt.py`)

### 5.1 `sys.path` setup

Near the existing `btt_layer` path injection:

```python
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if os.path.join(REPO_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
```

After which `from compress...` imports work.

### 5.2 Argparse and validation (underscore style)

```python
add_calibrated_btt_args(parser, hyphen_style=False)
validate_calibrated_btt_args(args, argv=argv, hyphen_style=False)
```

Registers `--calib_mode`, `--calib_source`, `--calib_traces_path`, `--calib_num_seqs`, `--calib_max_length`, `--calib_seed`, `--calib_batch_size`.

### 5.3 Dataset ordering

Move `SupervisedDataset` construction and `DataCollatorForSupervisedDataset` instantiation *before* the decomposition/model-prep block so the calib loader can draw real training batches.

### 5.4 Decomposition branch

```python
if args.calib_mode != "none":
    calib_loader = build_calib_loader(
        args, tokenizer=tokenizer,
        training_dataset=train_dataset,
        training_collate_fn=data_collator,
    )
    model, stats = apply_calibrated_btt(model, args, calib_loader=calib_loader)
else:
    # existing legacy convert_linear_to_btt + configure_blocktt_trainability
```

### 5.5 Calib loader shape

`SupervisedDataset.__getitem__` already returns `{input_ids, labels}` with prompt tokens masked to `IGNORE_INDEX` (= `-100`). `build_training_data_calib_loader` wraps a `Subset` with `DataCollatorForSupervisedDataset`. Backward-calibration gradients match real training gradients.

### 5.6 Training loop, normalize-after-update, checkpointing

Training loop unchanged. `normalize_trainable_blocktt_cores_` extension (§3.6) covers both legacy `BTTLayer` and new `BTTLinear`. Checkpointing gains the same topology-metadata mechanism as SFT (§3.7).

## 6. `src/compress` package changes

### 6.1 New `DecompositionConfig` fields

Added to `src/compress/decomposition.py`:

```python
s_merged_to: Optional[str] = field(
    default=None,
    metadata={
        "help": (
            "BTT SVD-init: where singular values are absorbed during the SVD "
            "step of decomposition. None preserves package default. "
            "Legacy values: 'frozen', 'trainable', 'both'."
        ),
    },
)
factorize_by_head: bool = field(
    default=True,
    metadata={
        "help": (
            "For attention projections, align BTT block shapes with "
            "attention head structure (default True matches legacy path)."
        ),
    },
)
```

`__post_init__` ignores these fields (with a warning) for non-BTT `train_mode` values, matching the existing pattern.

### 6.2 Plumbing through `compress_model_with_loader`

`decompose_with_loader` passes the two new kwargs through to `compress_model_with_loader`:

```python
compress_model_with_loader(
    ...,
    btt_decomp_mode=config.decomp_mode,
    btt_s_merged_to=config.s_merged_to,
    btt_factorize_by_head=config.factorize_by_head,
)
```

**Best-effort rule** (per user instruction 3b): each calibrated variant inside `src/compress/btt/` consumes these kwargs if it has a natural hook. Variants that lack a hook accept the kwarg and log a note that it was ignored. A per-variant honoring table is populated during implementation and included in the PR description.

### 6.3 `BTTLinear` topology metadata

Added to `src/compress/btt/btt_linear.py`:

```python
class BTTLinear(nn.Module):
    def topology_spec(self) -> dict:
        """JSON-serializable dict with: in_features, out_features, left_shape,
        right_shape, rank_l, rank_r, decomp_mode, has_bias, s_merged_to,
        factorize_by_head. Sufficient to reconstruct an empty BTTLinear with
        load_state_dict-compatible parameter shapes (no calibration needed)."""

    @classmethod
    def from_topology_spec(cls, spec: dict) -> "BTTLinear":
        """Construct an empty BTTLinear matching the spec. Weights are
        uninitialized; caller load_state_dict's."""
```

Module-level helpers (exported via `compress.__all__`):

```python
def export_btt_topology(model: nn.Module) -> dict[str, dict]:
    """Walk model; return {module_name: topology_spec} for every BTTLinear."""

def rebuild_btt_from_topology(model: nn.Module, topology: dict) -> nn.Module:
    """In-place: replace nn.Linear modules named in topology with empty
    BTTLinear constructed via from_topology_spec. Caller load_state_dict's."""
```

Implementation-time verification: confirm `BTTLinear.__init__` accepts the fields in the spec. If any field is missing, add a minimal constructor path as part of this task.

### 6.4 Checkpoint helpers in `compress_integration.py`

```python
def save_calibrated_btt_checkpoint(model, out_dir: str) -> None:
    """Save {out_dir}/model.safetensors (existing state_dict) and
    {out_dir}/btt_topology.json (from export_btt_topology)."""

def load_calibrated_btt_for_eval(model, checkpoint_dir: str) -> nn.Module:
    """Read btt_topology.json, rebuild BTTLinear modules via
    rebuild_btt_from_topology, load_state_dict from model.safetensors.
    No calibration required."""
```

All four entrypoints call `save_calibrated_btt_checkpoint` at checkpoint time when `args.calib_mode != "none"`. Legacy-path checkpoints use the existing code unchanged.

### 6.5 Namespace

`BTTLinear` (in `compress.btt.btt_linear`) and `BTTLayer` (in repo-root `btt_layer`) are distinct classes in distinct namespaces. `compress_integration.py` imports the former; legacy paths import the latter.

## 7. Testing

### 7.1 New unit tests

- `tests/test_calibrated_btt_cli.py`: argparse validation for all four entrypoints.
  - Each new flag parses with correct defaults.
  - `--calib-mode != none` with `--train-mode != blocktt` raises (main-repo scripts).
  - Integer `--blocktt-rank` with `--calib-mode != none` raises.
  - `--calib-mode=none` with any `--calib-*` flag explicitly passed raises.
  - `--calib-source=traces` without `--calib-traces-path` raises.
  - Underscore-style flags work for LIFT.

- `tests/test_compress_integration.py`: pure-Python logic.
  - `build_decomposition_config`: rank translation (`"full"` → 1.0; float passthrough; int → raise when calibrated).
  - `--trainable-type` → `skip_layers` inversion produces expected exclude sets for `all`/`mlp`/`attn` against a stub model.
  - `--calib-mode` → `train_mode` mapping table.
  - `build_training_data_calib_loader`: yields contract-compliant dicts, respects `num_seqs`, deterministic under fixed seed.

- `tests/test_calibrated_btt_pipeline.py`: end-to-end smoke on a tiny model.
  - `apply_calibrated_btt` with `train_mode=btt_llm_v2`, ratio 0.7, fake 4-batch calib loader: BTT layers replace linears, `stats["num_btt_layers"] > 0`, `train_position=small` gives correct `requires_grad`, forward pass works.
  - `materialize_calibrated_btt_weights` + `restore_calibrated_btt_weights` preserves forward outputs within tolerance.
  - Topology export → rebuild → `load_state_dict` preserves forward outputs exactly.

- `tests/test_calibrated_btt_checkpoint.py`: save/load round trip.
  - Decompose tiny model, run one fake optimizer step, `save_calibrated_btt_checkpoint`, rebuild in a fresh model via `load_calibrated_btt_for_eval`, assert outputs match.

### 7.2 Extensions to existing tests

- `tests/test_btt_pipeline_compat.py`: add a case with `--calib-mode=none` through the new CLI; confirm bit-identical behavior to pre-change.
- `tests/test_svd_pipeline_compat.py`: verify SVD path unaffected.
- `tests/test_run_rl_cli.py`, `tests/test_run_rl_dapo_cli.py`: extend with calib flag parsing assertions.

### 7.3 Manual smoke checklist

- SFT legacy: `--train-mode blocktt --calib-mode none` matches current behavior.
- SFT calibrated: `--train-mode blocktt --calib-mode v2 --calib-source training_data --calib-num-seqs 32`.
- RL calibrated: `run_rl.py` with calibrated BTT; verify calib rollout runs on dense base model and vLLM rollout works post-decomposition.
- LIFT calibrated: `finetune_blocktt.py --calib_mode v2_bp --calib_source training_data`.
- Eval: `load_calibrated_btt_for_eval` on a trained calibrated-BTT checkpoint produces working inference.

## 8. Implementation order

1. `src/compress` additions (§6.1, §6.2, §6.3) with unit tests for topology export/rebuild.
2. `compress_integration.py` core (§2, excluding RL rollout) with unit tests for config translation and SFT/LIFT loader builders.
3. Extend `normalize_trainable_blocktt_cores_` in `btt_layer.py` (§3.6).
4. SFT integration (§3). CLI and pipeline tests.
5. LIFT integration (§5). CLI and pipeline tests.
6. RL rollout calibration helper (§4.2) with mock-model unit test.
7. RL integration (§4), including `materialize` / `restore` extensions. Manual smoke.
8. `run_rl_dapo.py` integration (§4.6).
9. Checkpoint save/load wiring (§6.4) across all four entrypoints. End-to-end eval test.

## 9. Implementation-time verifications

- Does `compress.btt.btt_linear.BTTLinear` expose `materialize_dense_weight()`? If not, add it in step 1.
- For each of `btt_llm_v2`, `btt_llm_v2_bp`, `btt_llm_v2_combined`, `btt_twosteps`: which honor `s_merged_to` and which honor `factorize_by_head`? Table populated during implementation and included in PR description.
- Confirm `BTTLinear.__init__` accepts all fields needed by `from_topology_spec`. Add any missing fields in step 1.
- Locate the RL training rollout-config object / function in `run_rl.py` and share it with the calibration rollout (no duplication).
