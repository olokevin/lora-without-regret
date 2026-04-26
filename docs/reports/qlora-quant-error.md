# QLoRA Quantization-Error Report

**Model:** `meta-llama/Meta-Llama-3-8B`
**Num prompts:** 32
**Max seq len:** 256

**Setup:** standard QLoRA NF4 + double-quant + blocksize=64 on the original
`nn.Linear.weight` of every target leaf. No LoRA adapters added (a real QLoRA
training would add zero-init adapters, which contribute nothing at step 0).
Backward error is N/A in this setting because the only trainable params
(the absent adapters) would be zero, so gradients through them are
degenerate.

**Companion report:** `docs/reports/qfura-quant-error.md` measures the
qfura post-conversion error on the same model with the same prompts.
Numbers below should be compared against qfura's `output_one_block` cells
(qfura's recommended default).

## Reproduce

```bash
HF_HOME=/data/yequan/huggingface uv run python analysis/bench_qlora_quant_error.py \\
    --model meta-llama/Meta-Llama-3-8B \\
    --data-path /data/ruijiezhang/llm-adapter_bp/LLM-Adapters/ft-training_set/commonsense_170k.json \\
    --num-prompts 32
```

## Model-level error

| top1 match | KL(bf16 ‖ qlora) | logit rel err |
|---|---|---|
| 0.4547 | 0.1933 | 0.2026 |

## Per-linear-type forward error (averaged over all transformer layers)

| Layer | fwd mean | fwd p95 | n layers |
|---|---|---|---|
| `q_proj` | 0.0501 | 0.0602 | 32 |
| `k_proj` | 0.0485 | 0.0549 | 32 |
| `v_proj` | 0.0884 | 0.1102 | 32 |
| `o_proj` | 0.1051 | 0.1516 | 32 |
| `gate_proj` | 0.0704 | 0.0786 | 32 |
| `up_proj` | 0.0856 | 0.0945 | 32 |
| `down_proj` | 0.0865 | 0.0948 | 32 |

## Comparison hint

If qfura's KL is materially higher than QLoRA's (e.g., >2×), qfura needs
the trainable cores to absorb more error than QLoRA's adapters do. If
they're comparable, qfura's pre-training state is no worse than QLoRA's,
and the question becomes whether qfura's tighter param budget can match
QLoRA's downstream accuracy.
