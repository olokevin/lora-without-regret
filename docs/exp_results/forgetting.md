# Catastrophic Forgetting Analysis: Math Fine-tuned Models on Commonsense Tasks

Evaluating whether math-finetuned models lose commonsense reasoning ability
compared to the base model (Meta-Llama-3-8B).

## Setup

- **Base model**: `meta-llama/Meta-Llama-3-8B`
- **Commonsense eval tasks**: BoolQ, PIQA, Social IQa, ARC-Challenge, ARC-Easy, OpenBookQA, HellaSwag, WinoGrande
- **Eval protocol**: LIFT commonsense eval (beam search, num_beams=4, max_new_tokens=256)
- **Math training data**: MATH-10K
- **Source root**: `/data/yequan/fura/lift/math/meta-llama/Meta-Llama-3-8B`

## Math Performance (MATH-10K eval)

| Task | BlockTT | LIFT | LoRA r=128 | LIFT paper |
| --------- | ------: | ------: | ---------: | ---------: |
| MultiArith | 99.50 | 99.33 | **99.67** | 99.33 |
| gsm8k | **71.87** | 68.84 | 63.99 | 72.40 |
| AddSub | 93.16 | 93.42 | **94.94** | 93.42 |
| AQuA | 27.56 | **29.13** | 27.17 | 34.65 |
| SingleEQ | 97.64 | 97.24 | **97.83** | 98.03 |
| SVAMP | **79.00** | 78.80 | 76.50 | 80.90 |
| MAWPS | **94.54** | 91.60 | 91.18 | 93.70 |
| **Avg** | **80.47** | 79.77 | 78.75 | 81.78 |

- BlockTT: `blocktt-lr_2e-4-decomp_output_one_block_pos_small-rank_full-smerge_trainable-type_all-seed_43`
  - Trainable params: 1.44% (117M / 8.1B)
- LIFT: `lift-sparse-no_head-mask_lift_sparse-rank_128-filter_128-interval_400-lr_1e-4-seed_43`
  - Config: rank=128, filter_rank=128, update_interval=400, lr=1e-4
- LoRA: `lora-lr_2e-4-rank_128-alpha_256-seed_43`
  - Config: rank=128, alpha=256, lr=2e-4

## Commonsense Performance (cross-task forgetting)

| Task | Base Model | Math BlockTT | Math LIFT | Math LoRA r=128 | BTT vs Base | LIFT vs Base | LoRA vs Base |
| ------------- | ---------: | -----------: | --------: | --------------: | ----------: | -----------: | -----------: |
| BoolQ | 54.9 | 57.1 | **58.0** | 52.5 | +2.2 | +3.1 | -2.4 |
| PIQA | 67.4 | **67.0** | 66.8 | 51.3 | -0.4 | -0.6 | -16.1 |
| Social IQa | 32.6 | **42.9** | 16.9 | 28.2 | +10.3 | -15.7 | -4.4 |
| ARC-Challenge | 23.5 | **48.9** | 40.5 | 29.7 | +25.4 | +17.0 | +6.2 |
| ARC-Easy | 25.3 | **55.3** | 50.3 | 43.0 | +30.0 | +25.0 | +17.7 |
| OpenBookQA | 27.6 | **36.4** | 36.0 | 35.6 | +8.8 | +8.4 | +8.0 |
| HellaSwag | 25.0 | **18.3** | 14.0 | 21.8 | -6.7 | -11.0 | -3.2 |
| WinoGrande | 44.1 | **38.0** | 23.0 | 23.0 | -6.1 | -21.1 | -21.1 |
| **Average** | **37.6** | **45.5** | **38.2** | **35.6** | **+7.9** | **+0.6** | **-1.9** |

## Observations

1. **BlockTT preserves commonsense best after math SFT.** BlockTT averages +7.9 over base model on commonsense. LIFT is +0.6 (roughly neutral). LoRA is -1.9 (slight net forgetting).
2. **Forgetting ranking: BlockTT > LIFT > LoRA.** Despite all three achieving similar math performance (78.8--80.5), their impact on commonsense differs markedly. BlockTT actually improves commonsense; LoRA degrades it.
3. **LoRA suffers severe PIQA forgetting (-16.1).** This is the worst single-task drop across all methods. PIQA measures physical intuition -- LoRA appears to overwrite this knowledge more than the others.
4. **All methods degrade on HellaSwag and WinoGrande.** These pattern-completion tasks rely on surface-level language modeling that math SFT disrupts regardless of method.
5. **All methods gain on ARC tasks.** The instruction-following format learned from math SFT transfers positively. BlockTT gains most (+25-30), LoRA least (+6-18).
6. **Math performance is comparable.** BlockTT 80.47, LIFT 79.77, LoRA 78.75 -- all within ~2 points. The forgetting differences are not explained by math quality.

## Interpretation

BlockTT's full-rank factorization (training only the small core while keeping the large core frozen) constrains weight updates to a low-dimensional subspace, acting as a natural regularizer that preserves pre-trained capabilities. LIFT's sparse mask modifies a broader set of weight entries, causing moderate forgetting. LoRA's low-rank additive updates, while parameter-efficient, modify the model's behavior more globally across tasks, leading to the most forgetting.

The key difference: BlockTT freezes the principal components (large core) and only trains the residual (small core), explicitly preserving the dominant weight structure. LoRA and LIFT lack this structural constraint.

## Reference: Commonsense-finetuned BlockTT (from lift_commonsense.md)

| Task | Commonsense BlockTT |
| ------------- | ------------------: |
| BoolQ | 76.6 |
| PIQA | 89.9 |
| Social IQa | 83.2 |
| ARC-Challenge | 84.1 |
| ARC-Easy | 93.6 |
| OpenBookQA | 89.4 |
| HellaSwag | 96.8 |
| WinoGrande | 89.7 |
| **Average** | **87.9** |
