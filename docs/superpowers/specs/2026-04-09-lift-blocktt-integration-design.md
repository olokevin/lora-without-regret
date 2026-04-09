# LIFT Reproduction + BlockTT Integration

## Goal

1. Make the LIFT codebase (`ref/LIFT/`) runnable under the main project's `.venv/` (Python 3.13+, torch 2.8+, transformers 4.57+).
2. Add a `finetune_blocktt.py` training script to LIFT that uses our Block Tensor-Train decomposition (`btt_layer.py`) as an alternative PEFT method, enabling direct comparison with LIFT, LoRA, S2FT, and full FT on the same arithmetic/commonsense reasoning benchmarks.

## Part 1: Venv Compatibility Patches

### Problem

LIFT's `requirements.txt` pins old deps (torch 2.1, transformers 4.45, peft 0.14) and imports `deepspeed` which is not installed. Compatibility analysis shows only two actual blockers:

- `src/utils/utils.py:9` imports `from deepspeed.accelerator import get_accelerator` (used only for `manual_seed_all`)
- `src/finetune_lora.py:16-17` imports `deepspeed` (used only for `deepspeed.add_config_arguments(parser)`)

All other APIs (peft, transformers, accelerate, torch.svd) are compatible with the main project's versions.

### Changes

**`src/utils/utils.py`** — Replace deepspeed seeding with torch.cuda:
```python
# Before
from deepspeed.accelerator import get_accelerator
def set_random_seed(seed):
    ...
    get_accelerator().manual_seed_all(seed)

# After
def set_random_seed(seed):
    ...
    torch.cuda.manual_seed_all(seed)
```

**`src/finetune_lora.py`** — Remove deepspeed import and arg parsing:
```python
# Remove: import deepspeed
# Remove: parser = deepspeed.add_config_arguments(parser)
```

No other files need modification.

## Part 2: `finetune_blocktt.py`

### Architecture

New file: `ref/LIFT/src/finetune_blocktt.py`

Follows the same structure as `finetune_sft.py` (the LIFT/full-FT script), with these differences:

1. **Model preparation**: After loading the HF model, applies `convert_linear_to_btt()` and `configure_blocktt_trainability()` from the main project's `btt_layer.py`.
2. **Optimizer**: Standard `torch.optim.AdamW` on trainable TT cores (dense updates). No `SparseAdamW`.
3. **Post-step normalization**: Optionally calls `normalize_trainable_blocktt_cores_(model)` after each optimizer step (controlled by `--blocktt_normalize_after_update`).
4. **Model saving**: Before saving, materializes BTT layers back to dense `nn.Linear` so the output is a standard HF checkpoint compatible with LIFT's existing eval scripts.

### CLI Arguments

BlockTT-specific arguments (mirroring `run_sft.py`):

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--trainable_type` | str | `all` | Which modules to convert: `all`, `mlp`, `attn` |
| `--decomp_mode` | str | `input_one_block` | BTT decomposition mode |
| `--blocktt_rank` | str | `full` | BTT rank; `full` for lossless init |
| `--train_position` | str | `small` | Which core to train: `small`, `large`, `both` |
| `--s_merged_to` | str | `frozen` | Where to merge singular values |
| `--blocktt_normalize_after_update` | flag | False | Normalize cores after optimizer step |
| `--blocktt_factorize_by_head` | flag | True | Align attention BTT blocks with head structure |
| `--no_train_bias` | flag | False | Freeze BTT biases |

Standard training arguments are inherited from the existing LIFT scripts (model_name_or_path, lr, epochs, batch size, etc.).

### Import Strategy

`btt_layer.py` lives at the repo root. The script adds the repo root to `sys.path` (LIFT scripts already do this for their parent dir) so it can `from btt_layer import ...`.

### Save/Load for Eval

At save time, iterate all `BTTLayer` modules and replace them with materialized `nn.Linear`:
```python
for name, module in model.named_modules():
    if isinstance(module, BTTLayer):
        dense = module.to_linear()  # materialize btt_l @ btt_r -> weight
        # replace in parent
```
This ensures eval scripts (`run_math_parallel.py`, `run_commonsense_parallel.py`) work without modification.

## Part 3: Bash Scripts

### Training Scripts

**`bash_scripts/finetune_math_blocktt.sh`** — Mirrors `finetune_math_lift.sh`:
- Reads config from `slurm_config_blocktt_math.txt`
- Fields: `MODEL`, `decomp_mode`, `train_position`, `blocktt_rank`, `s_merged_to`, `trainable_type`, `lr`, `seed`
- Calls `accelerate launch src/finetune_blocktt.py` with appropriate flags
- Runs `eval_math.sh` on the output

**`bash_scripts/finetune_commonsense_blocktt.sh`** — Same pattern for commonsense.

### Config Files

**`bash_scripts/slurm_config_blocktt_math.txt`** — Initial configs matching the models used by LIFT:
```
MODEL                          decomp_mode        train_position  blocktt_rank  s_merged_to  trainable_type  lr    seed
meta-llama/Llama-3.2-3B       input_one_block    small           full          frozen       all             2e-4  43
meta-llama/Llama-3.2-1B       input_one_block    small           full          frozen       all             2e-4  43
meta-llama/Llama-2-7b-hf      input_one_block    small           full          frozen       all             2e-4  43
meta-llama/Meta-Llama-3-8B    input_one_block    small           full          frozen       all             1e-4  43
```

**`bash_scripts/slurm_config_blocktt_commonsense.txt`** — Similar, matching LIFT's commonsense configs.

### Eval Scripts

No changes needed. The existing `eval_math.sh` and `eval_commonsense.sh` work on any standard HF checkpoint.

## Data Setup

The LLM-Adapters repo must be cloned to provide training and eval data:
```bash
git clone https://github.com/AGI-Edgerunners/LLM-Adapters.git
```

Datasets used:
- **Math training**: `LLM-Adapters/ft-training_set/math_10k.json`
- **Math eval**: `{MultiArith,gsm8k,AddSub,AQuA,SingleEq,SVAMP,mawps}/test.json`
- **Commonsense training**: `LLM-Adapters/ft-training_set/commonsense_170k.json`
- **Commonsense eval**: `{boolq,piqa,social_i_qa,ARC-Challenge,ARC-Easy,openbookqa,hellaswag,winogrande}/test.json`

## Files Modified/Created

| File | Action |
|------|--------|
| `ref/LIFT/src/utils/utils.py` | Modify: replace deepspeed import |
| `ref/LIFT/src/finetune_lora.py` | Modify: remove deepspeed import |
| `ref/LIFT/src/finetune_blocktt.py` | **Create**: new training script |
| `ref/LIFT/bash_scripts/finetune_math_blocktt.sh` | **Create** |
| `ref/LIFT/bash_scripts/finetune_commonsense_blocktt.sh` | **Create** |
| `ref/LIFT/bash_scripts/slurm_config_blocktt_math.txt` | **Create** |
| `ref/LIFT/bash_scripts/slurm_config_blocktt_commonsense.txt` | **Create** |
