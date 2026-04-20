"""Standalone math-verify evaluation entrypoint.

Loads a merged checkpoint (or HuggingFace model ID) and evaluates it on a
fixed set of math reasoning benchmarks. Does not support legacy adapter-only
or factored checkpoints — for those, re-run training with --enable-merged-ckpt
or use the in-loop --enable-math-verify path in run_rl.py.

Examples:
  uv run eval_rl.py --checkpoint Qwen/Qwen3-1.7B
  uv run eval_rl.py --checkpoint /path/to/runs/lora/run-name/step=50
  uv run eval_rl.py --checkpoint <path> --math-verify-datasets MATH-500,AIME-24
"""

import argparse
import json
import os
import sys
import time

import torch

from eval_datasets import REGISTRY, known_dataset_names
from math_verify_eval import math_verify_eval


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate a merged checkpoint on math reasoning benchmarks."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a merged checkpoint directory or a HuggingFace model ID.",
    )
    parser.add_argument(
        "--math-verify-datasets",
        type=str,
        default=",".join(known_dataset_names()),
        help="Comma-separated dataset names. Default: all five.",
    )
    parser.add_argument(
        "--math-verify-n-samples",
        type=int,
        default=None,
        help="Override per-dataset n_samples. Default: registry per-dataset.",
    )
    parser.add_argument(
        "--math-verify-temperature",
        type=float,
        default=None,
        help="Override per-dataset sampling temperature.",
    )
    parser.add_argument(
        "--math-verify-max-tokens",
        type=int,
        default=2048,
        help="Max tokens per generation. Default: 2048.",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="boxed.prompt",
        help="Path to the prompt template. Default: boxed.prompt.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="vLLM max_model_len. Default: 2048.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.4,
        help="vLLM gpu_memory_utilization. Default: 0.4.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help=(
            "Where to write eval_results.json. Default: "
            "{checkpoint}/eval_results.json for local paths, "
            "./eval_results.json for HF IDs."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42.",
    )

    args = parser.parse_args(argv)
    args.math_verify_datasets = [
        s.strip() for s in args.math_verify_datasets.split(",") if s.strip()
    ]
    return args


def validate_args(args):
    unknown = [d for d in args.math_verify_datasets if d not in REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown --math-verify-datasets entries: {unknown}. "
            f"Known names: {sorted(REGISTRY.keys())}"
        )
    if args.math_verify_n_samples is not None and args.math_verify_n_samples <= 0:
        raise ValueError("--math-verify-n-samples must be > 0")
    if args.math_verify_max_tokens <= 0:
        raise ValueError("--math-verify-max-tokens must be > 0")


def preflight_checkpoint(path: str) -> None:
    """Reject legacy adapter-only / factored checkpoints with a clear message.

    HF model IDs and plain HF directories pass through.
    """
    if not os.path.isdir(path):
        return  # HF ID or unknown path → defer to from_pretrained

    if os.path.exists(os.path.join(path, "adapter_config.json")):
        raise ValueError(
            f"Checkpoint at {path} is a legacy adapter-only checkpoint "
            f"(found adapter_config.json). eval_rl.py only supports merged "
            f"checkpoints; re-run training with --enable-merged-ckpt true "
            f"or use the in-loop --enable-math-verify path."
        )

    safetensors_path = os.path.join(path, "model.safetensors")
    if os.path.exists(safetensors_path):
        from safetensors import safe_open

        with safe_open(safetensors_path, framework="pt") as f:
            for key in f.keys():
                if any(
                    marker in key
                    for marker in (".btt_l", ".btt_r", ".btt_s", ".svd_a", ".svd_b", ".svd_s")
                ):
                    raise ValueError(
                        f"Checkpoint at {path} is a legacy factored checkpoint "
                        f"(found key {key!r}). eval_rl.py only supports merged "
                        f"checkpoints; re-run training with --enable-merged-ckpt true "
                        f"or use the in-loop --enable-math-verify path."
                    )


def default_output_json_path(checkpoint: str) -> str:
    if os.path.isdir(checkpoint):
        return os.path.join(checkpoint, "eval_results.json")
    return os.path.join(os.getcwd(), "eval_results.json")


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    preflight_checkpoint(args.checkpoint)

    output_json = args.output_json or default_output_json_path(args.checkpoint)

    import random

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading model: {args.checkpoint}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Running math-verify on: {args.math_verify_datasets}")
    results = math_verify_eval(
        model=None,
        tokenizer=tokenizer,
        datasets=args.math_verify_datasets,
        n_samples_override=args.math_verify_n_samples,
        temperature_override=args.math_verify_temperature,
        max_tokens=args.math_verify_max_tokens,
        prompt_template_path=args.prompt_template,
        vllm_kwargs={
            "model": args.checkpoint,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": 4096,
        },
    )
    results["checkpoint"] = args.checkpoint
    results["model_id_at_train_time"] = args.checkpoint  # best-effort

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {output_json}")

    print("Math-verify results:")
    for ds_name, ds_result in results["datasets"].items():
        print(
            f"  {ds_name}: {ds_result['accuracy']:.2%} "
            f"({ds_result['n_correct']}/{ds_result['n_total']})"
        )
    if results.get("errors"):
        print("Errors:")
        for ds_name, reason in results["errors"].items():
            print(f"  {ds_name}: {reason}")


if __name__ == "__main__":
    main()
