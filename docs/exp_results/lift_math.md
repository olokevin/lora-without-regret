# LIFT Math Eval Results

- Source root: `/data/yequan/fura/lift/math/meta-llama/Meta-Llama-3-8B`
- Metric: task accuracy (`Result` from each `eval.log`), averaged across 7 tasks.

| Run | MultiArith | GSM8K | AddSub | AQuA | SingleEQ | SVAMP | MAWPS | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `output_one_block + smerge_trainable` | **99.50** | **71.87** | **93.16** | 27.56 | **97.64** | 79.00 | **94.54** | **80.47** |
| `output_one_block + smerge_keep_trainable` | 98.67 | 70.28 | 91.65 | **29.53** | 96.06 | **80.20** | 92.02 | 79.77 |

## Paper Table 2 (Copied)

Arithmetic reasoning, fine-tuned on the MATH-10K dataset (from `docs/25_ICML_Principal Weights Emerge after Rank Reduction for Reasoning-Focused Supervised Fine-Tuning.pdf`).

| Model | Method | Best Rank | MultiArith | GSM8K | AddSub | AQuA | SingleEQ | SVAMP | MAWPS | Avg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LLaMA-3.2-1B | Full FT | - | 98.50 | 33.13 | 92.15 | 24.02 | 94.29 | 51.10 | 87.40 | 68.66 |
| LLaMA-3.2-1B | LoRA | 128 | 98.50 | 30.48 | 89.87 | 24.41 | 92.91 | 53.60 | 86.13 | 67.98 |
| LLaMA-3.2-1B | DoRA | 128 | 98.50 | 30.48 | 89.87 | 24.41 | 92.91 | 53.60 | 86.13 | 67.98 |
| LLaMA-3.2-1B | PiSSA | 128 | 97.33 | 32.15 | 91.90 | 23.23 | 94.09 | 52.30 | 86.97 | 68.28 |
| LLaMA-3.2-1B | S2FT | 128 | 96.17 | 30.17 | 90.38 | 23.62 | 92.13 | 49.40 | 81.93 | 66.26 |
| LLaMA-3.2-1B | LIFT | 128 | 98.17 | 32.37 | 90.89 | 26.38 | 92.91 | 56.30 | 86.13 | 69.02 |
| LLaMA-3.2-3B | Full FT | - | 97.83 | 56.71 | 92.66 | 33.46 | 95.08 | 69.10 | 90.34 | 76.45 |
| LLaMA-3.2-3B | LoRA | 128 | 98.50 | 55.95 | 91.39 | 27.95 | 94.29 | 70.30 | 91.60 | 75.71 |
| LLaMA-3.2-3B | DoRA | 128 | 98.33 | 55.12 | 91.65 | 28.35 | 94.88 | 70.90 | 89.92 | 75.59 |
| LLaMA-3.2-3B | PiSSA | 128 | 98.33 | 59.51 | 91.90 | 25.20 | 95.47 | 69.90 | 89.50 | 75.69 |
| LLaMA-3.2-3B | S2FT | 128 | 98.33 | 55.65 | 91.90 | 29.53 | 96.06 | 68.90 | 88.66 | 75.58 |
| LLaMA-3.2-3B | LIFT | 128 | 99.17 | 57.92 | 94.17 | 28.74 | 96.85 | 71.00 | 91.60 | 77.06 |
| LLaMA-2-7B | Full FT | - | 98.17 | 46.55 | 93.67 | 22.05 | 96.85 | 63.20 | 89.08 | 72.79 |
| LLaMA-2-7B | LoRA | 128 | 98.00 | 47.76 | 92.41 | 23.62 | 95.08 | 62.90 | 90.76 | 72.93 |
| LLaMA-2-7B | DoRA | 64 | 98.00 | 47.38 | 92.41 | 21.26 | 96.06 | 62.30 | 89.50 | 72.42 |
| LLaMA-2-7B | PiSSA | 128 | 98.83 | 48.45 | 92.66 | 21.26 | 95.87 | 63.40 | 90.76 | 73.03 |
| LLaMA-2-7B | S2FT | 128 | 99.17 | 44.43 | 91.39 | 29.13 | 95.47 | 62.60 | 89.50 | 73.10 |
| LLaMA-2-7B | LIFT | 128 | 98.67 | 47.31 | 92.66 | 26.77 | 96.85 | 63.60 | 90.34 | 73.74 |
| LLaMA-3-8B | Full FT | - | 99.00 | 69.83 | 93.42 | 28.74 | 97.83 | 79.60 | 92.86 | 80.18 |
| LLaMA-3-8B | LoRA | 64 | 99.17 | 71.57 | 92.15 | 24.41 | 96.26 | 80.50 | 92.02 | 79.44 |
| LLaMA-3-8B | DoRA | 64 | 98.83 | 70.96 | 90.89 | 29.53 | 96.65 | 81.80 | 90.76 | 79.92 |
| LLaMA-3-8B | PiSSA | 128 | 99.00 | 71.27 | 93.67 | 28.74 | 97.64 | 80.60 | 92.02 | 80.42 |
| LLaMA-3-8B | S2FT | 64 | 99.67 | 70.89 | 92.91 | 32.68 | 97.64 | 78.20 | 94.12 | 80.87 |
| LLaMA-3-8B | LIFT | 128 | 99.33 | 72.40 | 93.42 | 34.65 | 98.03 | 80.90 | 93.70 | 81.78 |
| LLaMA-3-8B | **Our best eval run** (`output_one_block + smerge_trainable`) | - | **99.50** | 71.87 | 93.16 | 27.56 | 97.64 | 79.00 | **94.54** | 80.47 |

Runs with missing or incomplete math eval outputs (not included above):
- `blocktt-lr_2e-4-decomp_input_one_block_pos_small-rank_full-smerge_keep_trainable-type_all-seed_43`
- `blocktt-lr_2e-4-decomp_output_one_block_pos_small-rank_full-smerge_frozen-type_all-seed_43`
- `blocktt-lr_1e-4-decomp_output_one_block_pos_small-rank_full-smerge_trainable-type_all-seed_43` (no eval outputs)
- `blocktt-lr_3e-4-decomp_output_one_block_pos_small-rank_full-smerge_trainable-type_all-seed_43` (no eval outputs)
- `lora-lr_2e-4-rank_64-alpha_128-seed_43` (no math eval outputs)
