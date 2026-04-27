# qfura vs QLoRA: Math Fine-Tuning Results

**Date:** 2026-04-26
**Base model:** `meta-llama/Meta-Llama-3-8B`
**Training data:** `LLM-Adapters/ft-training_set/math_10k.json` (9819 examples, 3 epochs)
**Hardware:** 1× H100 NVL 95 GB per run, runs in parallel on GPU 4 (qfura) and GPU 5 (qlora)
**Evaluation harness:** `ref/LIFT/bash_scripts/eval_math.sh` → `run_math_parallel.py` (beam=4, top_k=40, top_p=0.75, temp=0.1)

## Methods

### qfura (this work)

NF4-quantized BTT fine-tuning with these defaults (also locked in `CLAUDE.md`):

- `--blocktt_rank full`
- `--decomp_mode output_one_block`
- `--train_position small` (small core trainable, large core frozen + NF4-quantized)
- `--s_merged_to keep_trainable` (singular values held in a separate trainable `btt_s`)
- `--quant_block_layout flat`
- `--learning_rate 1e-4`
- `--per_device_train_batch_size 1`, `--gradient_accumulation_steps 16` (effective batch 16)
- `bnb.optim.PagedAdamW8bit`, `--gradient_checkpointing`

**Trainable parameters:** 118,685,696 (1.46% of 8.15B)
**Frozen large cores quantized to NF4:** 224 layers, 10.47 GB saved vs bf16 BTT
**Training time:** 1h 26m
**Final epoch-3 average loss:** 0.0009

### QLoRA (baseline)

Standard QLoRA setup using PEFT + bitsandbytes:

- 4-bit NF4 base model with double-quant, bf16 compute dtype
- LoRA adapters on the same 7 leaves: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- `--lora_r 64`, `--lora_alpha 128`, `--lora_dropout 0.05`
- Same `--learning_rate 1e-4`, batch sizes, optimizer, schedule as qfura

**Trainable parameters:** 167,772,160 (2.05% of 8.20B with rounded vocab)
**Training time:** 2h 09m
**Final epoch-3 average loss:** 0.0003

QLoRA's checkpoint is a PEFT adapter; for evaluation it was merged into the bf16 base via `tools/merge_qlora_for_eval.py` to produce a full HF-format checkpoint that the existing `run_math_parallel.py` eval script can load.

## Results

Pass rate (%) on each math dataset, all 7 from the LIFT/LLM-Adapters benchmark.

| Dataset | n | qfura | QLoRA | QLoRA − qfura |
|---|---:|---:|---:|---:|
| MultiArith | 600 | 95.67 | 99.00 | +3.33 |
| GSM8K | 1319 | 66.72 | 70.43 | +3.71 |
| AddSub | 395 | 91.90 | 92.66 | +0.76 |
| AQuA | 254 | 26.38 | 27.17 | +0.79 |
| SingleEq | 508 | 95.47 | 96.65 | +1.18 |
| SVAMP | 1000 | 76.50 | 82.20 | +5.70 |
| mawps | 238 | 92.02 | 92.02 | 0.00 |
| **Average (unweighted)** | | **77.81** | **79.88** | **+2.07** |
| **Average (n-weighted)** | 4314 | **75.98** | **79.46** | **+3.48** |

## Observations

- **QLoRA ahead overall, but small gap.** ~2 points unweighted, ~3.5 points n-weighted. qfura uses 30% fewer trainable parameters (1.46% vs 2.05%) and a structurally more constrained adapter (BTT factorization), so a 2-3 point gap is the cost of that compression.
- **mawps tied at 92.02%.** Both methods saturate this dataset, suggesting the ceiling is set by base-model capability rather than fine-tuning method.
- **GSM8K (+3.71) and SVAMP (+5.70) are where QLoRA pulls ahead most.** Both are larger, more linguistically varied. QLoRA's higher parameter capacity helps with disambiguation and longer chain-of-thought arithmetic.
- **AddSub, SingleEq, MultiArith are near-ceiling.** Both methods within 1-3 points; the gap is small because the problems are templated.
- **AQuA is hard for both** (~26-27%). Multiple-choice algebra needs more than `math_10k.json` provides — both methods are bottlenecked by training-data coverage, not parameter count.
- **Pre-training-state distance from base predicted the trained gap.** Per `docs/reports/qfura-quant-error.md` and `docs/reports/qlora-quant-error.md`, the post-conversion (no training) model has KL=0.31 against the bf16 base for qfura, KL=0.19 for QLoRA. After 3 epochs, the gap on training loss is similar (qfura epoch-3 0.0009 vs QLoRA epoch-3 0.0003) and translates to a 2-3 point eval gap. The trainable cores cannot fully close the initial-state gap from pure quantization noise.

## Reproducibility

```bash
# qfura
CUDA_VISIBLE_DEVICES=4 \
HF_HOME=/data/yequan/huggingface \
no_wandb=1 \
bash ref/LIFT/bash_scripts/finetune_math_qfura.sh

# qlora (training writes a PEFT adapter)
CUDA_VISIBLE_DEVICES=5 \
HF_HOME=/data/yequan/huggingface \
no_wandb=1 \
bash ref/LIFT/bash_scripts/finetune_math_qlora.sh

# qlora needs an extra step before its built-in eval can run:
# the eval script expects a full HF-format model checkpoint, but PEFT only
# wrote adapter files. Merge the adapter into the bf16 base:
HF_HOME=/data/yequan/huggingface \
uv run python tools/merge_qlora_for_eval.py \
  --base_model meta-llama/Meta-Llama-3-8B \
  --adapter_dir <qlora_output_dir> \
  --output_dir <qlora_output_dir>-merged

# Then run eval against the merged dir:
CUDA_VISIBLE_DEVICES=4 \
HF_HOME=/data/yequan/huggingface \
WANDB_DISABLED=true \
bash ref/LIFT/bash_scripts/eval_math.sh \
    CKPT=<qlora_output_dir>-merged \
    base_model=meta-llama/Meta-Llama-3-8B
```

## Output paths

- qfura ckpt + per-dataset eval logs:
  `/data/yequan/fura/lift/math/meta-llama/Meta-Llama-3-8B/qfura-layout_flat-decomp_output_one_block_smerge_keep_trainable-lr_1e-4-seed_43/`
- qlora adapter:
  `/data/yequan/fura/lift/math/meta-llama/Meta-Llama-3-8B/qlora-r_64-alpha_128-lr_1e-4-seed_43/`
- qlora merged ckpt + per-dataset eval logs:
  `/data/yequan/fura/lift/math/meta-llama/Meta-Llama-3-8B/qlora-r_64-alpha_128-lr_1e-4-seed_43-merged/`
