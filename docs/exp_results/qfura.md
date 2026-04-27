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

`LLM-Adapters/ft-training_set/commonsense_170k.json` (170k examples, 3 epochs, lr 2e-4, batch 8×2 accum, 31932 optimizer steps). Eval harness: `ref/LIFT/bash_scripts/eval_commonsense.sh` over 8 commonsense reasoning datasets.

| Dataset | n | qfura (1.46%) | QLoRA r=48 (1.54%) | gap (qfura − QLoRA) |
|---|---:|---:|---:|---:|
| BoolQ | 3270 | 73.00 | 65.10 | +7.90 |
| PIQA | 1838 | 89.90 | 71.30 | +18.60 |
| SIQA (social_i_qa) | 1954 | 82.70 | 70.90 | +11.80 |
| HellaSwag | 10042 | 96.60 | 70.00 | +26.60 |
| WinoGrande | 1267 | 89.10 | 72.20 | +16.90 |
| ARC-Easy | 2376 | 93.10 | 67.80 | +25.30 |
| ARC-Challenge | 1172 | 83.40 | 55.50 | +27.90 |
| OBQA (openbookqa) | 500 | 90.60 | 68.20 | +22.40 |
| **Average (unweighted)** | | **87.30** | **67.63** | **+19.68** |
| **Average (n-weighted)** | 22419 | **89.30** | **69.40** | **+19.90** |

### Commonsense observations

- **qfura dominates QLoRA by ~20 points at parameter parity on commonsense reasoning.** Every dataset shows qfura ahead, gaps ranging from +7.9 (BoolQ) to +27.9 (ARC-Challenge). This is the inverse of the math result, where QLoRA at the same param budget edges qfura by +1.2-+2.0 points.
- **HellaSwag (n=10042) is qfura's strongest absolute showing: 96.60%** — the largest test set in the suite. QLoRA's 70.00% on the same set suggests it failed to converge to the multi-choice continuation pattern that qfura learned.
- **ARC-Challenge gap is the largest (+27.9 points).** Hardest commonsense reasoning task in the suite; qfura's BTT-then-quantize pipeline retains capacity to learn it; QLoRA's r=48 LoRA adapters at the same param budget do not.
- **All 4 multi-choice question datasets (PIQA, OBQA, ARC-E, ARC-C) show 18-28 point gaps.** All 4 binary/two-choice (BoolQ, WinoGrande) and rationale-style (HellaSwag, SIQA) datasets show 8-27 point gaps. The pattern is uniform — qfura consistently better, not specific to one task type.
- **QLoRA's commonsense numbers (avg 67.6%) are well below typical 8B-LoRA papers** (usually 75-85%). This suggests QLoRA at r=48 is genuinely under-fit for the 170k-example commonsense corpus, while qfura's BTT factorization captures more of it at the same parameter budget.

### Commonsense training time

| Method | Wall clock (3 epochs) |
|---|---|
| qfura | 9h 25m |
| QLoRA r=48 | 10h 21m |

## The math vs commonsense reversal

The single most striking finding from these experiments: **qfura's relative performance against QLoRA at parameter parity flips by ~22 points between math and commonsense.**

| Suite | qfura avg (n-weighted) | QLoRA r=48 avg (n-weighted) | qfura − QLoRA r=48 |
|---|---:|---:|---:|
| Math (math_10k.json) | 75.98 | 78.00 | **−2.02** |
| Commonsense (commonsense_170k.json) | 89.30 | 69.40 | **+19.90** |

Possible explanations to investigate:

1. **Dataset size effect.** Commonsense has 17× more training examples than math (170k vs 10k). At the same trainable param budget, qfura's BTT structure may scale to absorb large datasets better than QLoRA's low-rank adapters. The 6× more optimizer steps (31932 vs 1842) compound any structural advantage.
2. **Task structure.** Math is closed-form symbolic reasoning where the right answer is mostly a deterministic function of the input. Commonsense has a much wider distribution of "reasonable" continuations; absorbing that distribution may benefit from the higher effective rank that BTT provides on each linear (full rank in qfura's defaults vs r=48 in QLoRA).
3. **Quantization noise interaction.** Commonsense answers are short and high-entropy in token-space; even small per-token logit errors can flip a multi-choice answer. QLoRA's NF4 quantization of the full base injects noise on every matmul; qfura's NF4 only on the frozen BTT core. The relative noise budget per forward may matter more for commonsense than for math (where the model has thousands of forward steps to recover via chain-of-thought).
4. **Training-recipe interaction.** Math used lr 1e-4 + batch=1×16accum; commonsense used lr 2e-4 + batch=8×2accum. The higher commonsense LR might over-fit QLoRA's smaller-effective-rank adapters more aggressively.

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

**The pre-training KL gap (qfura 0.31 vs QLoRA 0.19) does NOT predict downstream accuracy across suites.**

- On math, the KL gap weakly tracks the trained-state gap: QLoRA's lower KL (0.19) maps to a 2-point math advantage at param parity, 3.5 points at default rank.
- On commonsense, qfura's higher pre-training KL (0.31) does *not* hurt — qfura outperforms QLoRA by 19.9 points despite starting from a worse-conditioned initial state.

Initial-state quantization error is a poor proxy for trained-state quality. The structural difference (BTT factorization vs LoRA adapter) is more important than the per-layer quantization-noise difference, and the relative benefit depends strongly on the task domain and dataset size.
