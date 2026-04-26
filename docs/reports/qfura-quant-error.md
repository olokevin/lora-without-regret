# qfura Quantization-Error Report

**Model:** `meta-llama/Meta-Llama-3-8B`
**Num prompts:** 32
**Max seq len:** 256

**Sweep:** `train_position=small`, `s_merged_to=keep_trainable`, `blocktt_rank=full`,
decomp_mode ∈ {input_one_block, output_one_block}, layout ∈ {flat, per_core_block}.

Note: with `input_one_block`, the frozen core has outer dim 1, so
`per_core_block` degenerates to `flat` — these two cells will report
identical numbers.

## Reproduce

```bash
HF_HOME=/data/yequan/huggingface uv run python analysis/bench_qbtt_quant_error.py \\
    --model meta-llama/Meta-Llama-3-8B \\
    --data-path /data/ruijiezhang/llm-adapter_bp/LLM-Adapters/ft-training_set/commonsense_170k.json \\
    --num-prompts 32
```

## Model-level error

| decomp_mode | layout | top1 match | KL(bf16 ‖ qfura) | logit rel err |
|---|---|---|---|---|
| `input_one_block` | `flat` | 0.2764 | 0.9823 | 0.4713 |
| `input_one_block` | `per_core_block` | 0.2764 | 0.9823 | 0.4713 |
| `output_one_block` | `flat` | 0.3317 | 0.3057 | 0.2471 |
| `output_one_block` | `per_core_block` | 0.3317 | 0.3057 | 0.2471 |

## Per-linear-type error (averaged over all transformer layers)

| Layer | decomp_mode | layout | fwd mean | fwd p95 | bwd mean | n layers |
|---|---|---|---|---|---|---|
| `q_proj` | `input_one_block` | `flat` | 0.0536 | 0.0643 | 0.0655 | 32 |
| `q_proj` | `input_one_block` | `per_core_block` | 0.0536 | 0.0643 | 0.0655 | 32 |
| `q_proj` | `output_one_block` | `flat` | 0.0581 | 0.0700 | 0.0156 | 32 |
| `q_proj` | `output_one_block` | `per_core_block` | 0.0581 | 0.0700 | 0.0156 | 32 |
| `k_proj` | `input_one_block` | `flat` | 0.0596 | 0.0744 | 0.0767 | 32 |
| `k_proj` | `input_one_block` | `per_core_block` | 0.0596 | 0.0744 | 0.0767 | 32 |
| `k_proj` | `output_one_block` | `flat` | 0.0605 | 0.0727 | 0.0323 | 32 |
| `k_proj` | `output_one_block` | `per_core_block` | 0.0605 | 0.0727 | 0.0323 | 32 |
| `v_proj` | `input_one_block` | `flat` | 0.0831 | 0.0949 | 0.1024 | 32 |
| `v_proj` | `input_one_block` | `per_core_block` | 0.0831 | 0.0949 | 0.1024 | 32 |
| `v_proj` | `output_one_block` | `flat` | 0.0824 | 0.0943 | 0.1024 | 32 |
| `v_proj` | `output_one_block` | `per_core_block` | 0.0824 | 0.0943 | 0.1024 | 32 |
| `o_proj` | `input_one_block` | `flat` | 0.1052 | 0.1523 | 0.1028 | 32 |
| `o_proj` | `input_one_block` | `per_core_block` | 0.1052 | 0.1523 | 0.1029 | 32 |
| `o_proj` | `output_one_block` | `flat` | 0.1142 | 0.1567 | 0.0620 | 32 |
| `o_proj` | `output_one_block` | `per_core_block` | 0.1142 | 0.1567 | 0.0624 | 32 |
| `gate_proj` | `input_one_block` | `flat` | 0.0709 | 0.0795 | 0.0957 | 32 |
| `gate_proj` | `input_one_block` | `per_core_block` | 0.0709 | 0.0795 | 0.0957 | 32 |
| `gate_proj` | `output_one_block` | `flat` | 0.0731 | 0.0805 | 0.0196 | 32 |
| `gate_proj` | `output_one_block` | `per_core_block` | 0.0731 | 0.0805 | 0.0196 | 32 |
| `up_proj` | `input_one_block` | `flat` | 0.0851 | 0.0931 | 0.1176 | 32 |
| `up_proj` | `input_one_block` | `per_core_block` | 0.0851 | 0.0931 | 0.1176 | 32 |
| `up_proj` | `output_one_block` | `flat` | 0.0859 | 0.0935 | 0.0445 | 32 |
| `up_proj` | `output_one_block` | `per_core_block` | 0.0859 | 0.0935 | 0.0445 | 32 |
| `down_proj` | `input_one_block` | `flat` | 0.0949 | 0.1055 | 0.1237 | 32 |
| `down_proj` | `input_one_block` | `per_core_block` | 0.0949 | 0.1055 | 0.1235 | 32 |
| `down_proj` | `output_one_block` | `flat` | 0.0885 | 0.0949 | 0.0825 | 32 |
| `down_proj` | `output_one_block` | `per_core_block` | 0.0885 | 0.0949 | 0.0825 | 32 |

## Default recommendation

Lowest model-level KL: `decomp_mode=output_one_block`, `layout=flat` (KL = 0.3057).

Pick this combination as the qfura training default for matching settings.
