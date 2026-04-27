# qfura vs QLoRA: Fine-Tuning Results on Llama-3-8B

**Last updated:** 2026-04-27

This document tracks qfura's fine-tuning quality against QLoRA baselines on two LIFT benchmark suites: math reasoning (`math_10k.json`) and commonsense reasoning (`commonsense_170k.json`). All runs use Llama-3-8B with 3 training epochs.

## Methods

### qfura

NF4-quantized BTT fine-tuning with the project-locked defaults (see `CLAUDE.md`):

- `--blocktt_rank full`
- `--decomp_mode output_one_block`
- `--train_position small` (small core trainable, large core frozen + NF4-quantized)
- `--s_merged_to keep_trainable` (singular values held in a separate trainable `btt_s`)
- `--quant_block_layout flat`
- `bnb.optim.PagedAdamW8bit`, `--gradient_checkpointing`

**Trainable parameters:** 118,685,696 (1.46% of 8.15B). 224 frozen BTT cores quantized to NF4.

### QLoRA (two ranks reported)

Standard QLoRA: 4-bit NF4 base + double-quant + bf16 compute, LoRA adapters on the 7 leaves (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`), `bnb.optim.PagedAdamW8bit`.

| QLoRA variant | rank | alpha | Trainable params | Trainable % | Notes |
|---|---|---|---|---|---|
| QLoRA r=48 | 48 | 96 | 125,829,120 | **1.54%** | Param-parity vs qfura (+6% trainable) |
| QLoRA r=64 | 64 | 128 | 167,772,160 | 2.05% | Standard QLoRA preset (+41% trainable) |

After training, QLoRA's PEFT adapter is merged into the bf16 base via `tools/merge_qlora_for_eval.py` to produce a full HF model checkpoint (the eval scripts expect a full model).

## Math Results

`LLM-Adapters/ft-training_set/math_10k.json` (9819 examples, 3 epochs, lr 1e-4, batch 1×16 accum, 1842 optimizer steps). Eval harness: `ref/LIFT/bash_scripts/eval_math.sh` → `run_math_parallel.py` (beam=4, top_k=40, top_p=0.75, temp=0.1).

| Dataset | n | qfura (1.46%) | QLoRA r=48 (1.54%) | QLoRA r=64 (2.05%) |
|---|---:|---:|---:|---:|
| MultiArith | 600 | 95.67 | 98.83 | 99.00 |
| GSM8K | 1319 | 66.72 | 70.43 | 70.43 |
| AddSub | 395 | 91.90 | 92.15 | 92.66 |
| AQuA | 254 | 26.38 | 27.56 | 27.17 |
| SingleEq | 508 | 95.47 | 96.06 | 96.65 |
| SVAMP | 1000 | 76.50 | 76.90 | 82.20 |
| mawps | 238 | 92.02 | **91.18** | 92.02 |
| **Average (unweighted)** | | **77.81** | **79.02** | **79.88** |
| **Average (n-weighted)** | 4314 | **75.98** | **78.00** | **79.46** |

### Math observations

- **At parameter parity (qfura 1.46% vs QLoRA r=48 1.54%), QLoRA still wins overall by +1.21 unweighted / +2.02 n-weighted points** — about half the gap of the over-parameterized QLoRA r=64 run.
- **SVAMP is the parameter-capacity discriminator.** At r=64 QLoRA hits 82.20% — at r=48 it collapses to 76.90%, statistically tied with qfura's 76.50%. The earlier "QLoRA dominates SVAMP" finding was largely a budget effect, not a method effect.
- **qfura wins mawps (92.02 vs 91.18) at param parity** — first dataset where qfura beats QLoRA at any tested rank. mawps is the smallest test set (n=238), so the difference is within ±2-point noise.
- **GSM8K is parameter-insensitive within QLoRA but rank-insensitive** — r=48 and r=64 both at 70.43%; both ahead of qfura by +3.71. Hard arithmetic seems to require either more capacity (r=64) or a different adapter structure than BTT.
- **MultiArith, AddSub, SingleEq, AQuA are near-saturated** — methods within 1-3 points of each other, gaps small relative to per-dataset noise.

### Math training time

| Method | Wall clock (3 epochs) | Final epoch-3 train loss |
|---|---|---|
| qfura | 1h 26m | 0.0009 |
| QLoRA r=48 | 2h 13m | 0.0011 |
| QLoRA r=64 | 2h 09m | 0.0003 |

Both QLoRA runs are slower per-step than qfura because the 4-bit dequant happens on every linear matmul of the full forward (vs qfura's per-BTT-layer dequant).

## Commonsense Results

`LLM-Adapters/ft-training_set/commonsense_170k.json` (170k examples, 3 epochs, lr 2e-4 commonsense / 1e-4 math, batch 8×2 accum, 31932 optimizer steps). Eval harness: `ref/LIFT/bash_scripts/eval_commonsense.sh` over 8 commonsense reasoning datasets.

**Status:** Runs in progress as of 2026-04-27. Both qfura (GPU 2) and QLoRA r=48 (GPU 3) launched at 01:23. Expected completion ~10 hours per run with eval following.

Results table to be filled in when both runs + evals complete:

| Dataset | n | qfura (1.46%) | QLoRA r=48 (1.54%) |
|---|---:|---:|---:|
| BoolQ | TBD | TBD | TBD |
| PIQA | TBD | TBD | TBD |
| SIQA (social_i_qa) | TBD | TBD | TBD |
| HellaSwag | TBD | TBD | TBD |
| WinoGrande | TBD | TBD | TBD |
| ARC-Easy | TBD | TBD | TBD |
| ARC-Challenge | TBD | TBD | TBD |
| OBQA (openbookqa) | TBD | TBD | TBD |
| **Average (unweighted)** | | TBD | TBD |

## Reproducibility

```bash
# qfura math (defaults baked in CLAUDE.md)
CUDA_VISIBLE_DEVICES=4 \
HF_HOME=/data/yequan/huggingface \
no_wandb=1 \
bash ref/LIFT/bash_scripts/finetune_math_qfura.sh

# qfura commonsense
CUDA_VISIBLE_DEVICES=2 \
HF_HOME=/data/yequan/huggingface \
no_wandb=1 \
bash ref/LIFT/bash_scripts/finetune_commonsense_qfura.sh

# QLoRA at param-equivalent rank=48 (math)
CUDA_VISIBLE_DEVICES=0 \
HF_HOME=/data/yequan/huggingface \
no_wandb=1 \
lora_r=48 lora_alpha=96 \
bash ref/LIFT/bash_scripts/finetune_math_qlora.sh

# QLoRA at param-equivalent rank=48 (commonsense)
CUDA_VISIBLE_DEVICES=3 \
HF_HOME=/data/yequan/huggingface \
no_wandb=1 \
lora_r=48 lora_alpha=96 \
bash ref/LIFT/bash_scripts/finetune_commonsense_qlora.sh

# QLoRA at default rank=64 (math)
CUDA_VISIBLE_DEVICES=5 \
HF_HOME=/data/yequan/huggingface \
no_wandb=1 \
bash ref/LIFT/bash_scripts/finetune_math_qlora.sh
```

The QLoRA runners auto-merge their PEFT adapter via `tools/merge_qlora_for_eval.py` after training completes, then invoke the existing eval shell scripts on the merged checkpoint.

## Output paths

- qfura math: `/data/yequan/fura/lift/math/meta-llama/Meta-Llama-3-8B/qfura-layout_flat-decomp_output_one_block_smerge_keep_trainable-lr_1e-4-seed_43/`
- qfura commonsense: `/data/yequan/fura/lift/commonsense/meta-llama/Meta-Llama-3-8B/qfura-layout_flat-decomp_output_one_block_smerge_keep_trainable-lr_2e-4-seed_43/`
- qlora math r=48 (merged): `/data/yequan/fura/lift/math/meta-llama/Meta-Llama-3-8B/qlora-r_48-alpha_96-lr_1e-4-seed_43-merged/`
- qlora math r=64 (merged): `/data/yequan/fura/lift/math/meta-llama/Meta-Llama-3-8B/qlora-r_64-alpha_128-lr_1e-4-seed_43-merged/`
- qlora commonsense r=48 (merged): `/data/yequan/fura/lift/commonsense/meta-llama/Meta-Llama-3-8B/qlora-r_48-alpha_96-lr_2e-4-seed_43-merged/`

## Quantization-error references

For pre-training-state error analysis (no fine-tuning):
- `docs/reports/qfura-quant-error.md` — qfura post-conversion error sweep over decomp_mode × layout. Best config: `decomp_mode=output_one_block, layout=flat, s_merged_to=keep_trainable` → model-level KL = 0.31 vs bf16 base.
- `docs/reports/qlora-quant-error.md` — QLoRA post-conversion error (NF4 of the original `nn.Linear.weight`, no LoRA adapters). Model-level KL = 0.19.

The pre-training KL gap (qfura 0.31 vs QLoRA 0.19, ~1.6×) tracks the trained-state n-weighted accuracy gap (qfura 75.98 vs QLoRA r=48 78.00, ~2.7%-pt; vs QLoRA r=64 79.46, ~4.6%-pt). Closer initial states yield slightly better fine-tuned models — but the gap shrinks dramatically when QLoRA is constrained to qfura's parameter budget.
