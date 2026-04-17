# RL Experiment Results

- Source: `/data/yequan/fura/rl_runs`
- Scope: non-debug runs only (`blocktt`, `full`, `svd`, `lora_full`).
- Metric source: final values in each run's `wandb-summary.json`.
- Primary metric reported: `eval/accuracy`.

## Group Summary

| Group | Runs | Completed | Best eval/accuracy | Mean eval/accuracy | Best run |
|---|---:|---:|---:|---:|---|
| blocktt | 17 | 15 | 0.895 | 0.814 | `blocktt-adamw-lr_1e-5-output_one_block-s_to_keep-train_both-0317-155422` |
| full | 5 | 5 | 0.886 | 0.801 | `full-adamw-lr_2e-5-0325-215533` |
| lora_full | 1 | 1 | 0.856 | 0.856 | `lora_full-adamw-lr_1e-5-rank_64-0319-140945` |
| svd | 5 | 5 | 0.891 | 0.874 | `svd-adamw-lr_1e-5-s_to_keep-train_input-0317-141139` |

## Completed Runs (Sorted by eval/accuracy)

| Rank | Group | Run | eval/accuracy | train/accuracy | reward_mean | approx_kl | runtime (min) | step |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | blocktt | `blocktt-adamw-lr_1e-5-output_one_block-s_to_keep-train_both-0317-155422` | 0.895 | 0.703 | 0.703 | 0.000 | 51.8 | 50 |
| 2 | svd | `svd-adamw-lr_1e-5-s_to_keep-train_input-0317-141139` | 0.891 | 0.711 | 0.711 | 0.000 | 51.8 | 50 |
| 3 | blocktt | `blocktt-adamw-lr_1e-5-output_one_block-s_to_keep_trainable-train_small-calib_v2_bp-0415-103823` | 0.886 | 0.719 | 0.719 | 0.000 | 111.7 | 50 |
| 4 | full | `full-adamw-lr_2e-5-0325-215533` | 0.886 | 0.715 | 0.715 | 0.000 | 45.7 | 50 |
| 5 | blocktt | `blocktt-adamw-lr_1e-4-input_one_block-s_to_keep_trainable-train_small-0413-184101` | 0.883 | 0.703 | 0.703 | 0.000 | 50.1 | 50 |
| 6 | blocktt | `blocktt-adamw-lr_1e-4-output_one_block-s_to_trainable-train_small-0403-105745` | 0.881 | 0.707 | 0.707 | 0.000 | 52.5 | 50 |
| 7 | blocktt | `blocktt-adamw-lr_1e-4-output_one_block-s_to_keep_trainable-train_small-0413-184109` | 0.880 | 0.688 | 0.688 | 0.000 | 52.9 | 50 |
| 8 | blocktt | `blocktt-adamw-lr_1e-4-input_one_block-s_to_trainable-train_small-0403-105737` | 0.879 | 0.699 | 0.699 | 0.000 | 50.3 | 50 |
| 9 | svd | `svd-adamw-lr_1e-5-s_to_output-train_output-0403-110126` | 0.879 | 0.680 | 0.680 | 0.000 | 51.6 | 50 |
| 10 | blocktt | `blocktt-adamw-lr_2e-4-output_one_block-s_to_trainable-train_small-0403-140625` | 0.878 | 0.641 | 0.641 | 0.000 | 42.3 | 50 |
| 11 | blocktt | `blocktt-adamw-lr_1e-4-{qkv:input,o:output,mlp_upgate:output,mlp_down:output}-s_to_frozen-train_small-0331-180927` | 0.872 | 0.664 | 0.664 | 0.000 | 61.5 | 50 |
| 12 | svd | `svd-adamw-lr_1e-5-s_to_input-train_input-0403-110133` | 0.871 | 0.695 | 0.695 | 0.000 | 53.9 | 50 |
| 13 | svd | `svd-adamw-lr_1e-5-s_to_input-train_output-0403-115314` | 0.869 | 0.691 | 0.691 | 0.000 | 50.6 | 50 |
| 14 | blocktt | `blocktt-adamw-lr_1e-4-output_one_block-s_to_frozen-train_small-0317-150342` | 0.868 | 0.676 | 0.676 | 0.000 | 50.4 | 50 |
| 15 | full | `full-adamw-lr_3e-5-0317-212057` | 0.867 | 0.680 | 0.680 | 0.000 | 39.1 | 50 |
| 16 | svd | `svd-adamw-lr_1e-5-s_to_output-train_input-0403-115542` | 0.858 | 0.660 | 0.660 | 0.000 | 53.2 | 50 |
| 17 | lora_full | `lora_full-adamw-lr_1e-5-rank_64-0319-140945` | 0.856 | 0.676 | 0.676 | 0.000 | 65.6 | 50 |
| 18 | blocktt | `blocktt-adamw-lr_1e-4-input_one_block-s_to_frozen-train_small-0331-171704` | 0.854 | 0.660 | 0.660 | 0.000 | 52.1 | 50 |
| 19 | blocktt | `blocktt-adamw-lr_1e-4-output_one_block-s_to_keep_trainable-train_small-norm-0413-193414` | 0.852 | 0.691 | 0.691 | 0.000 | 55.5 | 50 |
| 20 | full | `full-adamw-lr_2e-5-0324-223524` | 0.847 | 0.641 | 0.641 | 0.000 | 43.5 | 50 |
| 21 | full | `full-adamw-lr_2e-5-0318-160547` | 0.842 | 0.703 | 0.703 | 0.000 | 51.7 | 50 |
| 22 | blocktt | `blocktt-adamw-lr_2e-4-input_one_block-s_to_trainable-train_small-0403-135431` | 0.824 | 0.578 | 0.578 | 0.000 | 37.2 | 50 |
| 23 | blocktt | `blocktt-adamw-lr_5e-5-input_one_block-s_to_trainable-train_small-0403-114806` | 0.801 | 0.625 | 0.625 | 0.000 | 55.3 | 50 |
| 24 | blocktt | `blocktt-adamw-lr_5e-5-output_one_block-s_to_trainable-train_small-0403-115030` | 0.782 | 0.609 | 0.609 | 0.000 | 55.7 | 50 |
| 25 | full | `full-adamw-lr_5e-5-0317-220017` | 0.564 | 0.426 | 0.426 | 0.000 | 43.4 | 50 |
| 26 | blocktt | `blocktt-adamw-lr_1e-4-output_one_block-s_to_keep_trainable-train_small-norm-0413-193319` | 0.179 | - | - | - | 4.0 | 0 |

## Incomplete Runs (No eval/accuracy in summary)

| Group | Run | Summary keys | runtime (s) |
|---|---|---:|---:|
| blocktt | `blocktt-adamw-lr_1e-4-input_one_block-s_to_keep_trainable-train_small-0413-183110` | 2 | 133 |
| blocktt | `blocktt-adamw-lr_1e-4-output_one_block-s_to_keep_trainable-train_small-0413-193132` | 2 | 97 |
