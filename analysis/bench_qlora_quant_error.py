"""Quantization-error benchmark for QLoRA (NF4 of nn.Linear.weight, no BTT).

Companion to analysis/bench_qbtt_quant_error.py. Measures the apples-to-apples
QLoRA-style NF4 quantization error for the same Llama-3-8B forward pass, so
qfura's post-conversion error can be compared directly to QLoRA's.

Sweep:
  Just one cell — vanilla NF4 + double-quant + blocksize=64 (the standard
  QLoRA paper config). No LoRA adapters added (zero-init contributes nothing).
  This is "QLoRA at step 0" base error.

Backward error is N/A here because the trainable params would be the
zero-init LoRA adapters; gradient through zero adapters is degenerate. The
report explains this and only reports forward + model-level metrics.

Usage:
  HF_HOME=/data/yequan/huggingface uv run python analysis/bench_qlora_quant_error.py \\
      --model meta-llama/Meta-Llama-3-8B \\
      --data-path /data/ruijiezhang/llm-adapter_bp/LLM-Adapters/ft-training_set/commonsense_170k.json \\
      --num-prompts 32 \\
      --output docs/reports/qlora-quant-error.md
"""

import argparse
import copy
import gc
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


TARGET_NAMES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Meta-Llama-3-8B")
    p.add_argument(
        "--data-path",
        default="/data/ruijiezhang/llm-adapter_bp/LLM-Adapters/ft-training_set/commonsense_170k.json",
    )
    p.add_argument("--num-prompts", type=int, default=32)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument(
        "--output", default="docs/reports/qlora-quant-error.md"
    )
    return p.parse_args()


def load_prompts(data_path, num_prompts):
    with open(data_path) as f:
        data = json.load(f)
    prompts = []
    for entry in data[:num_prompts]:
        text = entry.get("instruction", "") + "\n" + entry.get("input", "")
        prompts.append(text.strip())
    return prompts


def tokenize_batch(tokenizer, prompts, max_seq_len):
    return tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_seq_len,
    )


@torch.no_grad()
def capture_layer_inputs(model, prompt_batch, target_names):
    captures = {}

    def make_hook(name):
        def hook(module, inputs, output):
            captures.setdefault(name, []).append(inputs[0].detach().cpu())
        return hook

    handles = []
    for name, module in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in target_names and isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(make_hook(name)))
    model(**prompt_batch)
    for h in handles:
        h.remove()
    return captures


def make_qlora_linear(linear):
    """Return a new nn.Module whose forward dequantizes NF4 weight then runs linear.

    Uses Params4bit + dequantize for the forward path (matches QLoRA's
    Linear4bit at step 0 with zero-init adapters absent).
    """
    device = linear.weight.device
    out_features = linear.out_features
    in_features = linear.in_features
    has_bias = linear.bias is not None

    p4 = bnb.nn.Params4bit(
        linear.weight.data.contiguous(),
        requires_grad=False,
        quant_type="nf4",
        compress_statistics=True,
        quant_storage=torch.uint8,
    )
    p4 = p4.to(device=device)

    class _QLoRALinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight_4bit = p4
            self.in_features = in_features
            self.out_features = out_features
            if has_bias:
                self.bias = nn.Parameter(linear.bias.data.detach().clone())
            else:
                self.bias = None

        def forward(self, x):
            w = bnb.functional.dequantize_4bit(
                self.weight_4bit.data,
                quant_state=self.weight_4bit.quant_state,
            ).to(x.dtype)
            out = torch.nn.functional.linear(x, w, self.bias)
            return out

    m = _QLoRALinear()
    m.to(device=device)
    return m


def per_layer_error(linear, layer_inputs):
    device = linear.weight.device
    dtype = linear.weight.dtype

    qlora = make_qlora_linear(linear)

    fwd_errs = []
    for xin in layer_inputs:
        xin = xin.to(device=device, dtype=dtype)
        with torch.no_grad():
            y_ref = linear(xin)
            y_q = qlora(xin)
        denom = y_ref.float().norm().clamp_min(1e-12)
        fwd_errs.append(
            ((y_ref.float() - y_q.float()).norm() / denom).item()
        )

    return {
        "fwd_err_mean": sum(fwd_errs) / len(fwd_errs),
        "fwd_err_p50": sorted(fwd_errs)[len(fwd_errs) // 2],
        "fwd_err_p95": sorted(fwd_errs)[int(len(fwd_errs) * 0.95)],
    }


@torch.no_grad()
def model_level_error(model_bf16, model_qlora, prompt_batch):
    out_bf16 = model_bf16(**prompt_batch).logits
    out_q = model_qlora(**prompt_batch).logits
    top1_match = (out_bf16.argmax(-1) == out_q.argmax(-1)).float().mean().item()
    log_p = torch.log_softmax(out_bf16.float(), dim=-1)
    log_q = torch.log_softmax(out_q.float(), dim=-1)
    kl = (log_p.exp() * (log_p - log_q)).sum(-1).mean().item()
    logit_rel = (
        (out_bf16.float() - out_q.float()).norm()
        / out_bf16.float().norm().clamp_min(1e-12)
    ).item()
    return {"top1_match": top1_match, "kl": kl, "logit_rel_err": logit_rel}


def aggregate_by_leaf(per_layer_results):
    by_leaf = {}
    for full_name, metrics in per_layer_results.items():
        leaf = full_name.split(".")[-1]
        by_leaf.setdefault(leaf, []).append(metrics)
    agg = {}
    for leaf, rows in by_leaf.items():
        fwd = [r["fwd_err_mean"] for r in rows]
        agg[leaf] = {
            "fwd_mean": sum(fwd) / len(fwd),
            "fwd_p95": sorted(fwd)[int(len(fwd) * 0.95)] if len(fwd) > 1 else fwd[0],
            "n_layers": len(rows),
        }
    return agg


def replace_linear_with_qlora(model, target_names):
    """Walk the model; replace each nn.Linear with leaf name in target_names
    with a QLoRA-style NF4-quantized linear."""
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.split(".")[-1] in target_names:
            replacements.append((name, module))
    for full_name, linear in replacements:
        new_module = make_qlora_linear(linear)
        parent = model
        path = full_name.split(".")
        for key in path[:-1]:
            parent = getattr(parent, key)
        setattr(parent, path[-1], new_module)
    return len(replacements)


def format_report(args, model_level, per_leaf):
    lines = [
        "# QLoRA Quantization-Error Report",
        "",
        f"**Model:** `{args.model}`",
        f"**Num prompts:** {args.num_prompts}",
        f"**Max seq len:** {args.max_seq_len}",
        "",
        "**Setup:** standard QLoRA NF4 + double-quant + blocksize=64 on the original",
        "`nn.Linear.weight` of every target leaf. No LoRA adapters added (a real QLoRA",
        "training would add zero-init adapters, which contribute nothing at step 0).",
        "Backward error is N/A in this setting because the only trainable params",
        "(the absent adapters) would be zero, so gradients through them are",
        "degenerate.",
        "",
        "**Companion report:** `docs/reports/qfura-quant-error.md` measures the",
        "qfura post-conversion error on the same model with the same prompts.",
        "Numbers below should be compared against qfura's `output_one_block` cells",
        "(qfura's recommended default).",
        "",
        "## Reproduce",
        "",
        "```bash",
        f"HF_HOME=/data/yequan/huggingface uv run python analysis/bench_qlora_quant_error.py \\\\",
        f"    --model {args.model} \\\\",
        f"    --data-path {args.data_path} \\\\",
        f"    --num-prompts {args.num_prompts}",
        "```",
        "",
        "## Model-level error",
        "",
        "| top1 match | KL(bf16 ‖ qlora) | logit rel err |",
        "|---|---|---|",
        f"| {model_level['top1_match']:.4f} | {model_level['kl']:.4f} | {model_level['logit_rel_err']:.4f} |",
        "",
        "## Per-linear-type forward error (averaged over all transformer layers)",
        "",
        "| Layer | fwd mean | fwd p95 | n layers |",
        "|---|---|---|---|",
    ]
    for leaf in TARGET_NAMES:
        agg = per_leaf.get(leaf)
        if agg is None:
            continue
        lines.append(
            f"| `{leaf}` | {agg['fwd_mean']:.4f} | {agg['fwd_p95']:.4f} | {agg['n_layers']} |"
        )
    lines += [
        "",
        "## Comparison hint",
        "",
        "If qfura's KL is materially higher than QLoRA's (e.g., >2×), qfura needs",
        "the trainable cores to absorb more error than QLoRA's adapters do. If",
        "they're comparable, qfura's pre-training state is no worse than QLoRA's,",
        "and the question becomes whether qfura's tighter param budget can match",
        "QLoRA's downstream accuracy.",
        "",
    ]
    return "\n".join(lines)


def main():
    args = parse_args()
    prompts = load_prompts(args.data_path, args.num_prompts)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    batch = tokenize_batch(tokenizer, prompts, args.max_seq_len)
    batch = {k: v.cuda() for k, v in batch.items()}

    print(f"Loading {args.model} in bf16 on cuda...")
    model_bf16 = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).cuda()
    model_bf16.eval()

    print("Capturing layer inputs (one forward pass)...")
    layer_captures = capture_layer_inputs(model_bf16, batch, TARGET_NAMES)
    print(f"Captured inputs for {len(layer_captures)} linear layers")

    per_layer_results = {}
    for name, xins in layer_captures.items():
        mod = dict(model_bf16.named_modules())[name]
        metrics = per_layer_error(mod, xins[:4])
        per_layer_results[name] = metrics
        print(f"  {name}: fwd_mean={metrics['fwd_err_mean']:.4f}")

    per_leaf = aggregate_by_leaf(per_layer_results)

    print("Building full QLoRA model for model-level error...")
    model_q = copy.deepcopy(model_bf16)
    n_replaced = replace_linear_with_qlora(model_q, TARGET_NAMES)
    print(f"Replaced {n_replaced} linears with QLoRA NF4 versions")
    model_q.eval()
    model_level = model_level_error(model_bf16, model_q, batch)
    print(f"Model-level: {model_level}")

    del model_q
    gc.collect()
    torch.cuda.empty_cache()

    report = format_report(args, model_level, per_leaf)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(report)
    print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
