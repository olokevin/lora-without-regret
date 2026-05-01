"""Batch-size probe for Llama-3-8B finetuning on a single H100.

For each (method, gradient_checkpointing) cell, try batch sizes in a doubling
schedule and report the largest BS that completes one forward+backward+opt step
at max_seq_len=2048 in bf16 with AdamW. OOMs are caught and reported.

This uses the same modules and settings the LIFT trainers use:
- Full-FT: AutoModelForCausalLM with all params trainable
- SVD: convert_linear_to_svd over q/k/v/o/gate/up/down with
  train_position=input, s_merged_to=keep_trainable (matches the user's runs)
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import traceback

import torch

# Make repo root importable for svd_layer.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from svd_layer import (
    convert_linear_to_svd,
    configure_svd_trainability,
    get_svd_target_module_names,
)
from transformers import AutoConfig, AutoModelForCausalLM


def _human_gb(b):
    return f"{b / (1024**3):.1f} GB"


def _free_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _new_model(method: str, model_path: str, dtype, device):
    cfg = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=cfg, dtype=dtype
    ).to(device)
    if method == "svd":
        targets = get_svd_target_module_names("all")
        convert_linear_to_svd(
            model,
            skip_names=("lm_head",),
            include_names=targets,
            s_merged_to="keep_trainable",
            train_position="input",
        )
        configure_svd_trainability(
            model,
            train_position="input",
            train_bias=True,
            train_embed_lm_head=False,
            train_singular_values=True,
        )
    # else: full FT — all params trainable by default
    return model


def _try_one_step(model, bs, seq_len, device, dtype, use_ckpt):
    if use_ckpt:
        model.gradient_checkpointing_enable()
        # Required so HF can backprop into inputs when ckpting embeddings.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    else:
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            pass

    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=1e-5)

    # Synthetic input: (bs, seq_len) random ids; labels = input_ids for CLM loss.
    vocab = model.config.vocab_size
    input_ids = torch.randint(0, vocab, (bs, seq_len), device=device, dtype=torch.long)
    labels = input_ids.clone()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = model(input_ids=input_ids, labels=labels, use_cache=False)
    loss = out.loss
    loss.backward()
    optim.step()
    optim.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated()
    return dt, peak


def probe_cell(method, ckpt, model_path, seq_len, device, candidates, dtype):
    print(f"\n=== {method.upper()}  ckpt={ckpt}  seq_len={seq_len}  dtype={dtype} ===")
    print(f"loading model ({method}) …", flush=True)
    _free_mem()
    model = _new_model(method, model_path, dtype, device)
    base_alloc = torch.cuda.memory_allocated()
    print(f"  base allocated after load+convert: {_human_gb(base_alloc)}")

    last_ok = None
    for bs in candidates:
        # Reset only optimizer + grads between attempts (keep model in mem).
        try:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad = None
            _free_mem()
            dt, peak = _try_one_step(model, bs, seq_len, device, dtype, ckpt)
            print(f"  bs={bs:>3}: OK  step={dt:.2f}s  peak={_human_gb(peak)}", flush=True)
            last_ok = (bs, dt, peak)
        except torch.cuda.OutOfMemoryError as e:
            print(f"  bs={bs:>3}: OOM  ({str(e).splitlines()[0][:120]})", flush=True)
            _free_mem()
            break
        except Exception as e:
            print(f"  bs={bs:>3}: ERROR  {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            _free_mem()
            break

    del model
    _free_mem()

    if last_ok is None:
        print(f"  RESULT: no batch size fit for {method} ckpt={ckpt}")
    else:
        bs, dt, peak = last_ok
        print(f"  RESULT: max BS = {bs}  (step {dt:.2f}s, peak {_human_gb(peak)})")
    return last_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B")
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--candidates", type=str, default="1,2,4,8,16,32",
                    help="Comma-separated batch sizes to try in order.")
    ap.add_argument("--methods", type=str, default="full,svd")
    ap.add_argument("--ckpt", type=str, default="on,off",
                    help="Comma-separated: on,off — gradient checkpointing modes")
    ap.add_argument("--dtype", type=str, default="bf16",
                    choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    device = torch.device("cuda")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    candidates = [int(x) for x in args.candidates.split(",")]

    print(f"Device: {torch.cuda.get_device_name(0)}  total={_human_gb(torch.cuda.get_device_properties(0).total_memory)}")
    print(f"Model: {args.model}  seq_len={args.seq_len}  dtype={args.dtype}")
    print(f"Candidates: {candidates}")

    summary = {}
    for method in args.methods.split(","):
        for ckpt_str in args.ckpt.split(","):
            ckpt = ckpt_str.strip().lower() == "on"
            res = probe_cell(method.strip(), ckpt, args.model, args.seq_len,
                             device, candidates, dtype)
            summary[(method.strip(), ckpt)] = res

    print("\n========== SUMMARY ==========")
    print(f"{'method':<6}  {'ckpt':<4}  {'max BS':<7}  {'step':<8}  {'peak':<10}")
    for (m, c), r in summary.items():
        if r is None:
            print(f"{m:<6}  {'on' if c else 'off':<4}  {'-':<7}  {'-':<8}  {'-':<10}")
        else:
            bs, dt, peak = r
            print(f"{m:<6}  {'on' if c else 'off':<4}  {bs:<7}  {dt:<7.2f}s  {_human_gb(peak):<10}")


if __name__ == "__main__":
    main()
