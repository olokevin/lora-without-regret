# LIFT Math Eval Results

- Source root:`/data/yequan/fura/lift/math/meta-llama/Meta-Llama-3-8B`
- Metric: task accuracy (`Result` from each`eval.log`), averaged across 7 tasks.

| Run                                          |      MultiArith |           GSM8K |          AddSub |            AQuA |        SingleEQ |           SVAMP |           MAWPS |             Avg |
| -------------------------------------------- | --------------: | --------------: | --------------: | --------------: | --------------: | --------------: | --------------: | --------------: |
| `output_one_block + smerge_trainable`      | **99.50** | **71.87** | **93.16** |           27.56 | **97.64** |           79.00 | **94.54** | **80.47** |
| `output_one_block + smerge_keep_trainable` |           98.67 |           70.28 |           91.65 | **29.53** |           96.06 | **80.20** |           92.02 |           79.77 |

## Paper Table 2 (Copied)

| Model      | Method                                                                | Best Rank |      MultiArith | GSM8K | AddSub |  AQuA | SingleEQ | SVAMP |           MAWPS |   Avg |
| ---------- | --------------------------------------------------------------------- | --------: | --------------: | ----: | -----: | ----: | -------: | ----: | --------------: | ----: |
|            |                                                                       |           |                 |       |        |       |          |       |                 |       |
| LLaMA-3-8B | Full FT                                                               |         - |           99.00 | 69.83 |  93.42 | 28.74 |    97.83 | 79.60 |           92.86 | 80.18 |
| LLaMA-3-8B | LoRA                                                                  |        64 |           99.17 | 71.57 |  92.15 | 24.41 |    96.26 | 80.50 |           92.02 | 79.44 |
| LLaMA-3-8B | DoRA                                                                  |        64 |           98.83 | 70.96 |  90.89 | 29.53 |    96.65 | 81.80 |           90.76 | 79.92 |
| LLaMA-3-8B | PiSSA                                                                 |       128 |           99.00 | 71.27 |  93.67 | 28.74 |    97.64 | 80.60 |           92.02 | 80.42 |
| LLaMA-3-8B | S2FT                                                                  |        64 |           99.67 | 70.89 |  92.91 | 32.68 |    97.64 | 78.20 |           94.12 | 80.87 |
| LLaMA-3-8B | LIFT                                                                  |       128 |           99.33 | 72.40 |  93.42 | 34.65 |    98.03 | 80.90 |           93.70 | 81.78 |
| LLaMA-3-8B | **Our best eval run** (`output_one_block + smerge_trainable`) |         - | **99.50** | 71.87 |  93.16 | 27.56 |    97.64 | 79.00 | **94.54** | 80.47 |

Arithmetic reasoning, fine-tuned on the MATH-10K dataset (from `docs/25_ICML_Principal Weights Emerge after Rank Reduction for Reasoning-Focused Supervised Fine-Tuning.pdf`).

Runs with missing or incomplete math eval outputs (not included above):

- `blocktt-lr_2e-4-decomp_input_one_block_pos_small-rank_full-smerge_keep_trainable-type_all-seed_43`
- `blocktt-lr_2e-4-decomp_output_one_block_pos_small-rank_full-smerge_frozen-type_all-seed_43`
- `blocktt-lr_1e-4-decomp_output_one_block_pos_small-rank_full-smerge_trainable-type_all-seed_43` (no eval outputs)
- `blocktt-lr_3e-4-decomp_output_one_block_pos_small-rank_full-smerge_trainable-type_all-seed_43` (no eval outputs)
- `lora-lr_2e-4-rank_64-alpha_128-seed_43` (no math eval outputs)
