# LIFT Commonsense Results

## Our Runs (`/data/yequan/fura/lift`)

- Source root:`/data/yequan/fura/lift/commonsense/meta-llama/Meta-Llama-3-8B`
- Metric: task accuracy (`Result` from each `eval.log`), averaged across 8 tasks.
- Bold indicates the best value among our runs for that column.

| Run                                              |           BoolQ |            PIQA |            SIQA |       HellaSwag |            Wino |           ARC-e |           ARC-c |            OBQA |             Avg |
| ------------------------------------------------ | --------------: | --------------: | --------------: | --------------: | --------------: | --------------: | --------------: | --------------: | --------------: |
| `output_one_block + smerge_keep_trainable`     | **76.60** |           89.90 | **83.20** | **96.80** | **89.70** | **93.60** | **84.10** | **89.40** | **87.91** |
| `output_one_block + rank_full + smerge_frozen` |           74.00 | **90.80** |           81.90 |           95.90 |           89.40 |           93.40 |           84.00 |           87.60 |           87.12 |
| `input_one_block + rank_full + smerge_frozen`  |           66.30 |           85.60 |           79.50 |           94.70 |           86.10 |           90.50 |           80.50 |           86.80 |           83.75 |
| `input_one_block + smerge_trainable`           |           67.00 |           85.60 |           80.60 |           93.50 |           86.30 |           90.20 |           78.40 |           84.80 |           83.30 |

Runs with missing or incomplete commonsense eval outputs (not included above):

- `blocktt-lr_2e-4-decomp_output_one_block_pos_small_smerge_trainable-seed_43` (no task eval logs)
- `full-lr_5e-5-seed_43` (no task eval logs found)
- `lora-lr_2e-4-rank_128-seed_43` (task eval logs missing/incomplete)
- `meta-llama/Meta-Llama-3-8B/lora/commonsense/lr_2e-4/rank_128_alpha_256/seed_43` (partial task logs only)

## Paper Table 1 (Copied)

Commonsense reasoning, fine-tuning on Commonsense-170K (from `docs/25_ICML_Principal Weights Emerge after Rank Reduction for Reasoning-Focused Supervised Fine-Tuning.pdf`).

### LLaMA-2-7B

| Method  | Best Rank | BoolQ | PIQA | SIQA | HellaSwag | Wino | ARC-e | ARC-c | OBQA |   Avg |
| ------- | --------: | ----: | ---: | ---: | --------: | ---: | ----: | ----: | ---: | ----: |
| Full FT |         - |  73.8 | 84.2 | 81.0 |      94.7 | 85.2 |  88.9 |  75.6 | 84.8 | 83.53 |
| LoRA    |       128 |  70.8 | 82.8 | 79.4 |      92.9 | 83.4 |  86.3 |  71.6 | 82.8 | 81.25 |
| DoRA    |       128 |  71.3 | 83.4 | 80.1 |      92.3 | 84.0 |  86.1 |  71.4 | 85.8 | 81.80 |
| PiSSA   |       128 |  72.5 | 85.3 | 80.8 |      87.2 | 86.1 |  87.1 |  74.3 | 85.6 | 82.36 |
| S2FT    |       128 |  73.3 | 83.7 | 81.0 |      94.3 | 84.6 |  88.3 |  75.8 | 84.8 | 83.22 |
| LIFT    |       128 |  74.8 | 84.7 | 82.2 |      94.4 | 86.0 |  89.2 |  76.4 | 89.6 | 84.66 |

### LLaMA-3-8B

| Method                                       | Best Rank |           BoolQ |  PIQA |  SIQA | HellaSwag |  Wino |           ARC-e | ARC-c |  OBQA |   Avg |
| -------------------------------------------- | --------: | --------------: | ----: | ----: | --------: | ----: | --------------: | ----: | ----: | ----: |
| Full FT                                      |         - |            75.4 |  88.0 |  81.8 |      96.5 |  89.3 |            93.1 |  83.0 |  86.0 | 86.64 |
| LoRA                                         |        64 |            71.8 |  85.3 |  80.9 |      93.4 |  84.5 |            90.0 |  77.0 |  84.8 | 83.46 |
| DoRA                                         |        64 |            74.6 |  87.4 |  81.2 |      94.7 |  87.1 |            89.4 |  79.5 |  86.4 | 85.04 |
| S2FT                                         |        64 |            67.7 |  89.8 |  82.5 |      95.2 |  87.8 |            93.1 |  84.6 |  88.6 | 86.16 |
| LIFT                                         |        32 |            75.7 |  90.5 |  83.2 |      96.5 |  89.4 |            93.6 |  83.9 |  90.2 | 87.88 |
| `output_one_block + smerge_keep_trainable` |         - | **76.60** | 89.90 | 83.20 |     96.80 | 89.70 | **93.60** | 84.10 | 89.40 | 87.91 |

### Best Of Our Runs (LLaMA-3-8B)

| Run                                          |           BoolQ |  PIQA |            SIQA |       HellaSwag |            Wino |           ARC-e |           ARC-c |            OBQA |             Avg |
| :------------------------------------------- | --------------: | ----: | --------------: | --------------: | --------------: | --------------: | --------------: | --------------: | --------------: |
| `output_one_block + smerge_keep_trainable` | **76.60** | 89.90 | **83.20** | **96.80** | **89.70** | **93.60** | **84.10** | **89.40** | **87.91** |
