"""Quantization-error benchmark for qfura.

Measures forward + backward error for a sweep of (decomp_mode, layout) cells
across every target linear in a model, plus a model-level logit error.

Sweep (per user direction 2026-04-26):
  train_position=small, s_merged_to=keep_trainable, blocktt_rank=full
  decomp_mode in {input_one_block, output_one_block}
  quant_block_layout in {flat, per_core_block}

Usage:
  HF_HOME=/data/yequan/huggingface uv run python analysis/bench_qbtt_quant_error.py \\
      --model meta-llama/Meta-Llama-3-8B \\
      --data-path ref/LIFT/LLM-Adapters/ft-training_set/commonsense_170k.json \\
      --num-prompts 32 \\
      --output docs/reports/qfura-quant-error.md
"""

import argparse
import copy
import gc
import itertools
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from btt_layer import (
    BTTLayer,
    configure_blocktt_trainability,
    convert_btt_to_qbtt_,
    convert_linear_to_btt,
    quantize_frozen_core_,
)


TARGET_NAMES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)
DECOMP_MODES = ("input_one_block", "output_one_block")
LAYOUTS = ("flat", "per_core_block")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Meta-Llama-3-8B")
    p.add_argument(
        "--data-path",
        default="ref/LIFT/LLM-Adapters/ft-training_set/commonsense_170k.json",
    )
    p.add_argument("--num-prompts", type=int, default=32)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument(
        "--output", default="docs/reports/qfura-quant-error.md"
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


def per_layer_error(linear, layer_inputs, decomp_mode, layout):
    """Convert a single nn.Linear -> BTT (s_merged_to=keep_trainable) -> QBTT.

    Returns dict with fwd/bwd relative errors.
    """
    device = linear.weight.device
    dtype = linear.weight.dtype
    parent = nn.Module()
    parent.add_module("target", copy.deepcopy(linear))
    convert_linear_to_btt(
        parent,
        btt_rank="full",
        decomp_mode=decomp_mode,
        include_names=("target",),
        train_position="small",
        s_merged_to="keep_trainable",
    )
    configure_blocktt_trainability(
        parent,
        train_position="small",
        train_singular_values=True,
    )
    btt_ref = copy.deepcopy(parent.target)
    qbtt = quantize_frozen_core_(parent.target, layout=layout)

    fwd_errs = []
    for xin in layer_inputs:
        xin = xin.to(device=device, dtype=dtype)
        y_ref = btt_ref(xin)
        y_q = qbtt(xin)
        denom = y_ref.float().norm().clamp_min(1e-12)
        fwd_errs.append(
            ((y_ref.float() - y_q.float()).norm() / denom).item()
        )

    xin = layer_inputs[0].to(device=device, dtype=dtype)
    target = torch.randn_like(btt_ref(xin))

    def grad_of(module):
        for p in module.parameters():
            if p.grad is not None:
                p.grad = None
        loss = (module(xin) - target).pow(2).mean()
        loss.backward()
        trainable = [p for p in module.parameters() if p.requires_grad]
        return [p.grad.detach().clone() for p in trainable]

    g_ref = grad_of(btt_ref)
    g_q = grad_of(qbtt)
    bwd_errs = []
    for a, b in zip(g_ref, g_q):
        denom = a.float().norm().clamp_min(1e-12)
        bwd_errs.append(((a.float() - b.float()).norm() / denom).item())

    return {
        "fwd_err_mean": sum(fwd_errs) / len(fwd_errs),
        "fwd_err_p50": sorted(fwd_errs)[len(fwd_errs) // 2],
        "fwd_err_p95": sorted(fwd_errs)[int(len(fwd_errs) * 0.95)],
        "bwd_err_max": max(bwd_errs),
    }


@torch.no_grad()
def model_level_error(model_bf16, model_qfura, prompt_batch):
    out_bf16 = model_bf16(**prompt_batch).logits
    out_q = model_qfura(**prompt_batch).logits
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
        bwd = [r["bwd_err_max"] for r in rows]
        agg[leaf] = {
            "fwd_mean": sum(fwd) / len(fwd),
            "fwd_p95": sorted(fwd)[int(len(fwd) * 0.95)] if len(fwd) > 1 else fwd[0],
            "bwd_mean": sum(bwd) / len(bwd),
            "n_layers": len(rows),
        }
    return agg


def format_report(args, model_level, per_leaf):
    """per_leaf: dict keyed by (decomp_mode, layout) tuple."""
    lines = [
        "# qfura Quantization-Error Report",
        "",
        f"**Model:** `{args.model}`",
        f"**Num prompts:** {args.num_prompts}",
        f"**Max seq len:** {args.max_seq_len}",
        "",
        "**Sweep:** `train_position=small`, `s_merged_to=keep_trainable`, `blocktt_rank=full`,",
        "decomp_mode ∈ {input_one_block, output_one_block}, layout ∈ {flat, per_core_block}.",
        "",
        "Note: with `input_one_block`, the frozen core has outer dim 1, so",
        "`per_core_block` degenerates to `flat` — these two cells will report",
        "identical numbers.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "HF_HOME=/data/yequan/huggingface uv run python analysis/bench_qbtt_quant_error.py \\\\",
        f"    --model {args.model} \\\\",
        f"    --data-path {args.data_path} \\\\",
        f"    --num-prompts {args.num_prompts}",
        "```",
        "",
        "## Model-level error",
        "",
        "| decomp_mode | layout | top1 match | KL(bf16 ‖ qfura) | logit rel err |",
        "|---|---|---|---|---|",
    ]
    for (decomp, layout), m in model_level.items():
        lines.append(
            f"| `{decomp}` | `{layout}` | {m['top1_match']:.4f} | {m['kl']:.4f} | {m['logit_rel_err']:.4f} |"
        )
    lines += [
        "",
        "## Per-linear-type error (averaged over all transformer layers)",
        "",
        "| Layer | decomp_mode | layout | fwd mean | fwd p95 | bwd mean | n layers |",
        "|---|---|---|---|---|---|---|",
    ]
    for leaf in TARGET_NAMES:
        for (decomp, layout) in itertools.product(DECOMP_MODES, LAYOUTS):
            agg = per_leaf.get((decomp, layout), {}).get(leaf)
            if agg is None:
                continue
            lines.append(
                f"| `{leaf}` | `{decomp}` | `{layout}` | "
                f"{agg['fwd_mean']:.4f} | {agg['fwd_p95']:.4f} | "
                f"{agg['bwd_mean']:.4f} | {agg['n_layers']} |"
            )
    # Pick the lowest-KL (decomp, layout) cell as the recommendation.
    best = min(model_level.items(), key=lambda kv: kv[1]["kl"])
    best_key, best_metrics = best
    lines += [
        "",
        "## Default recommendation",
        "",
        f"Lowest model-level KL: `decomp_mode={best_key[0]}`, `layout={best_key[1]}` "
        f"(KL = {best_metrics['kl']:.4f}).",
        "",
        "Pick this combination as the qfura training default for matching settings.",
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

    per_layer_results = {}  # keyed by (decomp_mode, layout) -> {full_name -> metrics}
    model_level = {}        # keyed by (decomp_mode, layout) -> metrics
    per_leaf = {}           # keyed by (decomp_mode, layout) -> {leaf -> agg}

    for decomp_mode in DECOMP_MODES:
        for layout in LAYOUTS:
            cell = (decomp_mode, layout)
            print(f"\n=== Cell: decomp={decomp_mode}, layout={layout} ===")
            cell_per_layer = {}
            for name, xins in layer_captures.items():
                mod = dict(model_bf16.named_modules())[name]
                metrics = per_layer_error(mod, xins[:4], decomp_mode, layout)
                cell_per_layer[name] = metrics
                print(
                    f"  {name}: fwd_mean={metrics['fwd_err_mean']:.4f} "
                    f"bwd_max={metrics['bwd_err_max']:.4f}"
                )
            per_layer_results[cell] = cell_per_layer
            per_leaf[cell] = aggregate_by_leaf(cell_per_layer)

            # Build a full qfura model for model-level error.
            print(f"  Building full qfura model...")
            model_q = copy.deepcopy(model_bf16)
            convert_linear_to_btt(
                model_q,
                btt_rank="full",
                decomp_mode=decomp_mode,
                include_names=TARGET_NAMES,
                train_position="small",
                s_merged_to="keep_trainable",
            )
            configure_blocktt_trainability(
                model_q,
                train_position="small",
                train_singular_values=True,
            )
            convert_btt_to_qbtt_(model_q, layout=layout)
            model_q.eval()
            model_level[cell] = model_level_error(model_bf16, model_q, batch)
            print(f"  model-level: {model_level[cell]}")
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
