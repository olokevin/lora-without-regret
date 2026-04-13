"""
Unified RL training entrypoint with DAPO support.

Examples:
  CUDA_VISIBLE_DEVICES=6 uv run run_rl_dapo.py --train-mode full --loss-type dapo --lr 1e-5 
  CUDA_VISIBLE_DEVICES=6 uv run run_rl_dapo.py --train-mode lora --loss-type dapo --lr 1e-4 --lora-rank 1 --trainable-type all
  
  CUDA_VISIBLE_DEVICES=6 uv run run_rl_dapo.py --train-mode blocktt --loss-type dapo --lr 1e-4 --blocktt-rank full --decomp-mode input_one_block --s-merged-to frozen --train-position small
  CUDA_VISIBLE_DEVICES=6 uv run run_rl_dapo.py --train-mode blocktt --loss-type dapo --lr 1e-4 --blocktt-rank full --decomp-mode input_one_block --s-merged-to trainable --train-position small
  CUDA_VISIBLE_DEVICES=6 uv run run_rl_dapo.py --train-mode blocktt --loss-type dapo --lr 1e-4 --blocktt-rank full --decomp-mode input_one_block --s-merged-to split --train-position both
"""

import argparse
import math
import os
import random
import sys
import time
from collections import Counter
from functools import partial
from pathlib import Path

import requests
import torch
import wandb
from safetensors.torch import save_file as save_safetensors_file
from btt_layer import (
    BTTLayer,
    BLOCKTT_DECOMP_GROUP_TO_MODULES,
    configure_blocktt_trainability,
    convert_linear_to_btt,
    get_blocktt_target_module_names,
    normalize_trainable_blocktt_cores_,
    resolve_blocktt_decomp_modes,
)
from datasets import load_dataset
from math_utils import is_equiv, last_boxed_only_string, remove_boxed
from optim.muon import Muon
from svd_layer import (
    SVDLayer,
    configure_svd_trainability,
    convert_linear_to_svd,
    get_svd_target_module_names,
)
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

MODE_DEFAULTS = {
    "full": {
        "lr": 1e-5,
        "wandb_project": "math_grpo_full",
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 128,
    },
    "lora": {
        "lr": 9e-5,
        "wandb_project": "math-grpo",
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 128,
    },
    "lora_full": {
        "lr": 9e-5,
        "wandb_project": "math-grpo-lora-full",
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 128,
    },
    "blocktt": {
        "lr": 9e-5,
        "wandb_project": "math-grpo-blocktt",
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 128,
    },
    "svd": {
        "lr": 9e-5,
        "wandb_project": "math-grpo-svd",
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 128,
    },
}

DEFAULT_TRAIN_DATASET_ID = "BytedTsinghua-SIA/DAPO-Math-17k"
DEFAULT_TRAIN_SPLIT = "train"
DEFAULT_MATH500_DATASET_ID = "HuggingFaceH4/MATH-500"
DEFAULT_MATH500_SPLIT = "test"
DEFAULT_AIME24_DATASET_ID = "BytedTsinghua-SIA/AIME-2024"
DEFAULT_AIME24_SPLIT = "train"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Unified RL training script (GRPO + DAPO)")
    parser.add_argument(
        "--train-mode",
        type=str,
        required=True,
        choices=["full", "lora", "lora_full", "blocktt", "svd"],
        help="Training mode: full, lora, lora_full, blocktt, or svd",
    )

    # Shared configuration
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="HuggingFace model ID to use",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default depends on train mode)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "muon"],
        help="Optimizer to use: adamw or muon (default: adamw)",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="none",
        choices=["none", "linear", "cosine"],
        help="Learning rate scheduler to use (default: none)",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.0,
        help="Warmup ratio in [0, 1] over total update steps (default: 0.0)",
    )
    parser.add_argument(
        "--cycle-length",
        type=int,
        default=None,
        help="Cycle length for cosine scheduler (default: full training length)",
    )
    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=0.1,
        help="Minimum LR ratio for cosine scheduler (default: 0.1)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay for optimizer (default: 0.0)",
    )
    parser.add_argument(
        "--lr-adam",
        type=float,
        default=None,
        help="Muon-only: AdamW fallback learning rate (default: inherit --lr)",
    )
    parser.add_argument(
        "--lr-embedding",
        type=float,
        default=None,
        help="Muon-only: embedding AdamW learning rate (default: inherit AdamW lr)",
    )
    parser.add_argument(
        "--norm-method",
        type=str,
        default=None,
        choices=["row", "col", "row-col", "col-row", "shape"],
        help="Muon-only: optional update normalization method",
    )
    parser.add_argument(
        "--n-grpo-steps",
        type=int,
        default=50,
        help="Number of GRPO training steps",
    )
    parser.add_argument(
        "--n-prompts-per-step",
        type=int,
        default=32,
        help="Number of prompts to sample per step",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=8,
        help="Number of rollouts per prompt",
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        default="grpo",
        choices=["grpo", "dapo"],
        help="Policy loss type: grpo (current behavior) or dapo",
    )
    parser.add_argument(
        "--clip-ratio-low",
        type=float,
        default=0.2,
        help="Lower clip ratio for DAPO decoupled clipping (default: 0.2)",
    )
    parser.add_argument(
        "--clip-ratio-high",
        type=float,
        default=0.28,
        help="Upper clip ratio for DAPO decoupled clipping (default: 0.28)",
    )
    parser.add_argument(
        "--mask-truncated-completions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mask truncated completions in DAPO loss (default: enabled)",
    )
    parser.add_argument(
        "--dynamic-sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable DAPO dynamic group filtering and refill (default: enabled)",
    )
    parser.add_argument(
        "--dynamic-sampling-max-rounds",
        type=int,
        default=4,
        help="Max refill rounds for DAPO dynamic sampling (default: 4)",
    )
    parser.add_argument(
        "--dynamic-sampling-oversample",
        type=float,
        default=1.5,
        help="Oversampling factor per refill round for DAPO dynamic sampling (default: 1.5)",
    )
    parser.add_argument(
        "--epochs-per-step",
        type=int,
        default=1,
        help="Number of epochs to train per step",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=None,
        help="Micro batch size (default depends on train mode)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Gradient accumulation steps (default depends on train mode)",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/data/yequan/fura/rl_runs",
        help="Base directory for saving runs",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="boxed.prompt",
        help="Path to prompt template file",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="vLLM max_model_len for local rollout backends (default: 2048)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.4,
        help="vLLM GPU memory utilization fraction for local rollout backends (default: 0.4)",
    )
    parser.add_argument(
        "--train-dataset-id",
        type=str,
        default=DEFAULT_TRAIN_DATASET_ID,
        help="Training dataset ID (default: official DAPO-Math-17k)",
    )
    parser.add_argument(
        "--train-split",
        type=str,
        default=DEFAULT_TRAIN_SPLIT,
        help="Training split name (default: train)",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=1000,
        help="Validation subset size sampled from the tail of the train split (default: 1000)",
    )
    parser.add_argument(
        "--eval-math500-dataset-id",
        type=str,
        default=DEFAULT_MATH500_DATASET_ID,
        help="Final-eval dataset ID for MATH-500",
    )
    parser.add_argument(
        "--eval-math500-split",
        type=str,
        default=DEFAULT_MATH500_SPLIT,
        help="Final-eval split for MATH-500",
    )
    parser.add_argument(
        "--eval-aime24-dataset-id",
        type=str,
        default=DEFAULT_AIME24_DATASET_ID,
        help="Final-eval dataset ID for AIME-24",
    )
    parser.add_argument(
        "--eval-aime24-split",
        type=str,
        default=DEFAULT_AIME24_SPLIT,
        help="Final-eval split for AIME-24",
    )
    parser.add_argument(
        "--final-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run final evaluation on MATH-500 and AIME-24 (default: enabled)",
    )
    parser.add_argument(
        "--eval-pass1",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log pass@1 in final evaluation (default: enabled)",
    )
    parser.add_argument(
        "--eval-majority",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log majority@k in final evaluation (default: enabled)",
    )
    parser.add_argument(
        "--eval-majority-k",
        type=int,
        default=32,
        help="Number of samples per prompt for majority@k final eval (default: 32)",
    )
    parser.add_argument(
        "--eval-temperature",
        type=float,
        default=1.0,
        help="Temperature for majority@k final eval samples (default: 1.0)",
    )
    parser.add_argument(
        "--eval-top-p",
        type=float,
        default=0.7,
        help="Top-p for majority@k final eval samples (default: 0.7)",
    )
    parser.add_argument(
        "--eval-max-tokens",
        type=int,
        default=1024,
        help="Max completion tokens for final evaluation (default: 1024)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--enable-save-ckpt",
        action="store_true",
        help="Save checkpoints at step 1, step 10, and final step (default: disabled)",
    )
    parser.add_argument(
        "--save-first-step-grads-path",
        type=str,
        default=None,
        help=(
            "Optional path to save raw gradients at the first optimizer step "
            "(safetensors)."
        ),
    )
    parser.add_argument(
        "--save-first-step-grads-prefixes",
        type=str,
        default=None,
        help=(
            "Comma-separated parameter-name prefixes to include when saving first-step "
            "grads. If omitted, all trainable grads are saved."
        ),
    )
    parser.add_argument(
        "--stop-after-first-step",
        action="store_true",
        help="Stop training immediately after the first optimizer step.",
    )
    parser.add_argument(
        "--save-grads-steps",
        type=str,
        default=None,
        help=(
            "Comma-separated optimizer steps to save gradients "
            "(e.g. '0,10,30'). Step 0 = before first update. Default: disabled."
        ),
    )

    # LoRA-family args (lora / lora_full)
    parser.add_argument(
        "--lora-rank", type=int, default=1, help="LoRA rank (default: 1)"
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        default="http://localhost:8000",
        help="URL for vLLM API server (lora mode only; lora_full uses local in-process rollout)",
    )

    # BlockTT-only args
    parser.add_argument(
        "--decomp-mode",
        type=str,
        default="input_one_block",
        help=(
            "BlockTT decomposition mode: scalar input_one_block|output_one_block "
            "or dict literal with keys qkv,o,mlp_upgate,mlp_down "
            "(values: input|output|input_one_block|output_one_block)"
        ),
    )
    parser.add_argument(
        "--blocktt-rank",
        type=str,
        default="full",
        help="BTT rank; default full for lossless decomposition",
    )
    parser.add_argument(
        "--no-train-bias",
        action="store_true",
        help="Freeze BTT biases; by default biases are trainable",
    )
    parser.add_argument(
        "--blocktt-normalize-after-update",
        action="store_true",
        help="Normalize trainable BTT cores after each optimizer update",
    )
    parser.add_argument(
        "--blocktt-factorize-by-head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Align attention layer BTT blocks with head structure (default: enabled)",
    )

    # Shared trainable module selector for LoRA/BlockTT/SVD
    parser.add_argument(
        "--trainable-type",
        type=str,
        default="all",
        choices=["all", "mlp", "attn"],
        help="Trainable target modules: all, mlp, or attn (default: all)",
    )
    parser.add_argument(
        "--train-position",
        type=str,
        default=None,
        choices=["output", "input", "small", "large", "both"],
        help="Trainable side selector: svd uses output|input, blocktt uses small|large|both",
    )
    parser.add_argument(
        "--s-merged-to",
        type=str,
        default=None,
        choices=["frozen", "trainable", "output", "input", "split", "keep"],
        help=(
            "Where to merge S during SVD init for svd/blocktt: "
            "frozen, trainable, output, input, split, or keep"
        ),
    )

    parser.add_argument(
        "--wandb-project",
        type=str,
        default="RL-DAPO",
        help="W&B project name (default depends on train mode)",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name (default: mode-specific auto-generated name)",
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")

    return parser.parse_args(argv)


def apply_mode_defaults(args):
    defaults = MODE_DEFAULTS[args.train_mode]
    if args.lr is None:
        args.lr = defaults["lr"]
    if args.wandb_project is None:
        args.wandb_project = defaults["wandb_project"]
    if args.micro_batch_size is None:
        args.micro_batch_size = defaults["micro_batch_size"]
    if args.gradient_accumulation_steps is None:
        args.gradient_accumulation_steps = defaults["gradient_accumulation_steps"]
    if args.train_mode == "blocktt" and args.train_position is None:
        args.train_position = "small"
    if args.train_mode == "svd" and args.train_position is None:
        args.train_position = "output"
    if args.train_mode in {"blocktt", "svd"} and args.s_merged_to is None:
        if args.train_position == "both":
            args.s_merged_to = "split"
        else:
            args.s_merged_to = "frozen"


def require_cuda_for_structured_conversion(train_mode, entrypoint):
    if train_mode not in {"blocktt", "svd"}:
        return
    if torch.cuda.is_available():
        return
    raise RuntimeError(
        f"{entrypoint}: --train-mode {train_mode} requires CUDA so SVD/BTT conversion "
        "runs on GPU. No CUDA device is available."
    )


def get_local_cuda_device_id():
    return int(os.environ.get("LOCAL_RANK", "0"))


def _flag_was_passed(argv, flag_name):
    return any(
        token == flag_name or token.startswith(f"{flag_name}=") for token in argv
    )


def validate_mode_specific_flags(args, argv):
    mode_to_flag_sets = {
        "lora": ["--lora-rank", "--vllm-url"],
        "blocktt": [
            "--decomp-mode",
            "--blocktt-rank",
            "--no-train-bias",
            "--blocktt-normalize-after-update",
            "--blocktt-factorize-by-head",
            "--no-blocktt-factorize-by-head",
        ],
        "svd": [],
    }

    if args.train_mode not in {"lora", "lora_full"}:
        passed = [f for f in mode_to_flag_sets["lora"] if _flag_was_passed(argv, f)]
        if passed:
            raise ValueError(
                f"{', '.join(passed)} is only valid when --train-mode lora or lora_full"
            )

    if args.train_mode != "blocktt":
        passed = [f for f in mode_to_flag_sets["blocktt"] if _flag_was_passed(argv, f)]
        if passed:
            raise ValueError(
                f"{', '.join(passed)} is only valid when --train-mode blocktt"
            )
    else:
        blocktt_targets = get_blocktt_target_module_names(args.trainable_type)
        decomp_mode, module_decomp_modes = resolve_blocktt_decomp_modes(
            args.decomp_mode,
            include_names=blocktt_targets,
            default_mode="input_one_block",
        )
        args.decomp_mode = decomp_mode
        args.blocktt_module_decomp_modes = module_decomp_modes
        args.decomp_mode_display = format_blocktt_decomp_mode(decomp_mode)

    if args.train_mode == "full" and _flag_was_passed(argv, "--trainable-type"):
        raise ValueError(
            "--trainable-type is only valid when --train-mode "
            "lora, lora_full, blocktt, or svd"
        )

    train_position_passed = _flag_was_passed(argv, "--train-position")
    if args.train_mode in {"full", "lora", "lora_full"} and train_position_passed:
        raise ValueError("--train-position is only valid when --train-mode blocktt or svd")
    if args.train_mode == "blocktt" and train_position_passed:
        if args.train_position not in {"small", "large", "both"}:
            raise ValueError("--train-position for blocktt must be one of: small, large, both")
    if args.train_mode == "svd" and train_position_passed:
        if args.train_position not in {"output", "input", "both"}:
            raise ValueError("--train-position for svd must be one of: output, input, both")

    s_merged_to_passed = _flag_was_passed(argv, "--s-merged-to")
    if args.train_mode in {"full", "lora", "lora_full"} and s_merged_to_passed:
        raise ValueError("--s-merged-to is only valid when --train-mode blocktt or svd")
    if (
        args.train_mode == "blocktt"
        and s_merged_to_passed
        and args.train_position == "both"
        and args.s_merged_to in {"frozen", "trainable"}
    ):
        raise ValueError(
            "--s-merged-to frozen/trainable is invalid when blocktt --train-position is both; "
            "use output, input, or split"
        )


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def resolve_blocktt_rank(rank_arg: str):
    if rank_arg == "full":
        return "full"
    try:
        rank = int(rank_arg)
    except ValueError as exc:
        raise ValueError("--blocktt-rank must be 'full' or a positive integer") from exc
    if rank <= 0:
        raise ValueError("--blocktt-rank must be > 0")
    return rank


def format_blocktt_decomp_mode(decomp_mode):
    if isinstance(decomp_mode, str):
        return decomp_mode
    if isinstance(decomp_mode, dict):
        order = tuple(BLOCKTT_DECOMP_GROUP_TO_MODULES.keys())
        return ",".join(f"{group}={decomp_mode[group]}" for group in order)
    return str(decomp_mode)


def get_lora_target_modules(trainable_type):
    if trainable_type == "all":
        return [
            "gate_proj",
            "up_proj",
            "down_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ]
    if trainable_type == "mlp":
        return ["gate_proj", "up_proj", "down_proj"]
    return ["q_proj", "k_proj", "v_proj", "o_proj"]


@torch.no_grad()
def materialize_btt_weight(layer: BTTLayer) -> torch.Tensor:
    if getattr(layer, "lr_act", False):
        raise ValueError("BlockTT rollout export requires lr_act=False")
    return layer.materialize_dense_weight()


@torch.no_grad()
def materialize_svd_weight(layer: SVDLayer) -> torch.Tensor:
    return layer.materialize_dense_weight()


@torch.no_grad()
def export_weights_for_vllm(model: torch.nn.Module):
    factorized_module_names = []
    weight_tuples = []

    for module_name, module in model.named_modules():
        if isinstance(module, BTTLayer):
            factorized_module_names.append(module_name)
            dense_weight = materialize_btt_weight(module)
            weight_tuples.append((f"{module_name}.weight", dense_weight))
            if module.bias is not None:
                weight_tuples.append((f"{module_name}.bias", module.bias))
        elif isinstance(module, SVDLayer):
            factorized_module_names.append(module_name)
            dense_weight = materialize_svd_weight(module)
            weight_tuples.append((f"{module_name}.weight", dense_weight))
            if module.bias is not None:
                weight_tuples.append((f"{module_name}.bias", module.bias))

    if factorized_module_names:
        factorized_prefixes = tuple(f"{name}." for name in factorized_module_names)
    else:
        factorized_prefixes = ()

    for name, param in model.named_parameters():
        if factorized_prefixes and name.startswith(factorized_prefixes):
            continue
        weight_tuples.append((name, param))

    return weight_tuples


def canonicalize_answer(value):
    if value is None:
        return None
    if isinstance(value, list):
        if len(value) == 0:
            return None
        value = value[0]
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    if len(value) == 0:
        return None
    boxed = remove_boxed(last_boxed_only_string(value))
    return boxed if boxed is not None else value


def parse_ground_truth_from_example(example):
    reward_model = example.get("reward_model")
    if isinstance(reward_model, dict) and "ground_truth" in reward_model:
        return canonicalize_answer(reward_model["ground_truth"])
    for key in ("ground_truth", "answer", "final_answer", "solution"):
        if key in example:
            return canonicalize_answer(example[key])
    return None


def build_prompt_from_example(example, tokenizer, template):
    if "prompt" in example:
        prompt = example["prompt"]
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            return tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
        if isinstance(prompt, str):
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )

    question = None
    for key in ("problem", "question", "query", "input"):
        if key in example and example[key] is not None:
            question = str(example[key])
            break
    if question is None:
        raise ValueError(
            "Unable to build prompt: expected one of prompt/problem/question/query/input."
        )
    prompt_text = template.replace("{question}", question)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False,
        add_generation_prompt=True,
    )


def process_math_example(example, tokenizer, template):
    return {
        "prompt": build_prompt_from_example(example, tokenizer, template),
        "answer": parse_ground_truth_from_example(example),
    }


def _dataset_tail_splits(dataset, val_size):
    if val_size <= 0 or len(dataset) <= val_size:
        n = min(1000, len(dataset))
        return dataset, dataset.select(range(n))
    split_at = len(dataset) - val_size
    train = dataset.select(range(0, split_at))
    val = dataset.select(range(split_at, len(dataset)))
    return train, val


def load_datasets_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    with open(args.prompt_template, "r", encoding="utf-8") as f:
        template = f.read().strip()

    base_dataset = load_dataset(args.train_dataset_id, split=args.train_split)
    train_dataset, val_dataset = _dataset_tail_splits(base_dataset, args.val_size)
    process_fn = lambda example: process_math_example(example, tokenizer, template)
    train_dataset = train_dataset.map(process_fn)
    val_dataset = val_dataset.map(process_fn)
    return train_dataset, val_dataset, tokenizer, template


def load_eval_dataset(dataset_id, split, tokenizer, template):
    dataset = load_dataset(dataset_id, split=split)
    return dataset.map(lambda example: process_math_example(example, tokenizer, template))


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_completion_tokens=None,
    mask_truncated=False,
) -> dict[str, torch.Tensor]:
    prompt_t = [tokenizer.encode(p) for p in prompt_strs]
    output_t = [tokenizer.encode(o) for o in output_strs]
    truncated_flags = []
    full = []
    max_len = 0

    for i in range(len(prompt_t)):
        row_len = len(prompt_t[i]) + len(output_t[i])
        truncated_flags.append(
            max_completion_tokens is not None and len(output_t[i]) >= max_completion_tokens
        )
        max_len = max(max_len, row_len)

    for i in range(len(prompt_t)):
        padding_size = max_len - len(prompt_t[i]) - len(output_t[i])
        padding = [tokenizer.pad_token_id] * padding_size
        row = torch.tensor(prompt_t[i] + output_t[i] + padding, dtype=torch.long)
        full.append(row.unsqueeze(0))

    f2 = torch.cat(full)
    input_ids = f2[:, :-1]
    labels = f2[:, 1:]

    response_mask = torch.zeros(len(prompt_strs), max_len - 1)
    for i in range(len(prompt_t)):
        if mask_truncated and truncated_flags[i]:
            continue
        response_mask[
            i, len(prompt_t[i]) - 1 : len(prompt_t[i]) + len(output_t[i]) - 1
        ] = 1

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask.bool(),
        "is_truncated": torch.tensor(truncated_flags, dtype=torch.bool),
    }


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    logits = model(input_ids).logits
    z = logits - logits.max(dim=-1, keepdim=True).values
    denom = torch.exp(z).sum(dim=-1, keepdim=True)
    logprobs = z - torch.log(denom)
    return torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def extract_generated_answer(text):
    if text is None:
        return None
    return remove_boxed(last_boxed_only_string(text))


def compute_rewards_from_outputs(outputs, answers, group_size):
    generated_answers = [extract_generated_answer(o) for o in outputs]
    raw_reward = [
        ans is not None and is_equiv(ans, answers[j // group_size])
        for j, ans in enumerate(generated_answers)
    ]
    return torch.tensor(raw_reward, dtype=torch.float).reshape((-1, group_size))


def compute_advantages(raw_reward_tensor):
    means = raw_reward_tensor.mean(dim=-1).unsqueeze(1)
    return (raw_reward_tensor - means).reshape(-1)


def select_informative_groups(raw_reward_tensor):
    reward_std = raw_reward_tensor.std(dim=-1)
    return (reward_std > 0).tolist()


def filter_rollouts_by_group_mask(prompts, answers, outputs, group_size, keep_mask):
    kept_prompts = []
    kept_answers = []
    kept_outputs = []
    for idx, keep in enumerate(keep_mask):
        if not keep:
            continue
        kept_prompts.append(prompts[idx])
        kept_answers.append(answers[idx])
        start = idx * group_size
        end = start + group_size
        kept_outputs.extend(outputs[start:end])
    return kept_prompts, kept_answers, kept_outputs


def majority_vote_answer(candidate_answers):
    valid = [a for a in candidate_answers if a is not None]
    if len(valid) == 0:
        return None
    return Counter(valid).most_common(1)[0][0]


def unique_by_problem(dataset):
    seen = set()
    unique_rows = []
    for row in dataset:
        key = (
            row.get("problem_id")
            or row.get("question_id")
            or row.get("id")
            or row.get("problem")
            or row.get("question")
            or row.get("prompt")
        )
        if isinstance(key, (list, dict)):
            key = str(key)
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)
    return unique_rows


def compute_policy_loss(
    loss_type,
    ratio,
    advantages,
    mask,
    *,
    clip_ratio_low,
    clip_ratio_high,
):
    adv = advantages.unsqueeze(-1)
    if loss_type == "dapo":
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high)
        objective = torch.minimum(ratio * adv, clipped_ratio * adv)
        per_token_loss = -objective
        masked_loss = per_token_loss * mask
        loss = masked_loss.sum() / mask.sum().clamp_min(1)
        clip_fraction = ((ratio != clipped_ratio).float() * mask).sum() / mask.sum().clamp_min(1)
        return loss, clip_fraction.item()

    per_token_loss = -ratio * adv
    masked_loss = per_token_loss * mask
    denom = mask.sum(dim=-1).clamp_min(1)
    loss_per_prompt = masked_loss.sum(dim=-1) / denom
    return loss_per_prompt.mean(), 0.0


def compute_run_name(args, mode_info: dict) -> str:
    """Compute a human-readable run name (used for both directory and W&B)."""
    loss_tag = args.loss_type
    if args.wandb_run_name is not None:
        return args.wandb_run_name
    if args.train_mode == "full":
        return f"{args.model_id}_{args.lr:.1e}_full_{loss_tag}"
    if args.train_mode == "lora":
        return f"{args.model_id}_{args.lr:.1e}_r{args.lora_rank}_{loss_tag}"
    if args.train_mode == "lora_full":
        return f"{args.model_id}_{args.lr:.1e}_r{args.lora_rank}_lora_full_{loss_tag}"
    if args.train_mode == "blocktt":
        decomp_mode_name = mode_info.get("decomp_mode_display", args.decomp_mode)
        return (
            f"{args.model_id}_{args.lr:.1e}_{decomp_mode_name}_"
            f"{args.train_position}_{args.trainable_type}_{loss_tag}"
        )
    return (
        f"{args.model_id}_{args.lr:.1e}_{args.train_position}_{args.trainable_type}_"
        f"{loss_tag}"
    )


def create_run_dir(base_dir: str, train_mode: str, run_name: str) -> str:
    """Create run directory at runs/{train_mode}/{run_name}-{timestamp}."""
    timestamp = time.strftime("%m%d-%H%M%S")
    run_dir = os.path.join(base_dir, train_mode, f"{run_name}-{timestamp}")
    os.makedirs(run_dir)
    return run_dir


def should_save_checkpoint(step_num: int, total_steps: int) -> bool:
    return step_num == 10 or step_num == 30 or step_num == total_steps


def save_checkpoint(model, tokenizer, run_dir: str, step_num: int):
    ckpt_dir = os.path.join(run_dir, f"step={step_num}")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Saving checkpoint to {ckpt_dir}")
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    print(f"Checkpoint saved to {ckpt_dir}")


def parse_save_grads_steps(s):
    """Parse comma-separated step numbers, or return empty set if None."""
    if s is None:
        return set()
    return {int(x.strip()) for x in s.split(",")}


def save_gradients(trainable_named_params, base_dir, step, subdir="step"):
    """Save all trainable gradients to {base_dir}/{subdir}={step}/grads.safetensors."""
    grad_payload = {}
    for name, p in trainable_named_params:
        if p.grad is None:
            continue
        grad_payload[name] = p.grad.detach().cpu()
    if not grad_payload:
        print(f"Warning: no gradients to save at step {step}")
        return
    ckpt_dir = os.path.join(base_dir, f"{subdir}={step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    save_safetensors_file(grad_payload, os.path.join(ckpt_dir, "grads.safetensors"))
    print(f"Saved gradients ({len(grad_payload)} tensors) to {ckpt_dir}")


def maybe_init_wandb(args, run_dir, run_name, mode_info, num_training_steps):
    if args.no_wandb:
        return

    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "train_mode": args.train_mode,
            "model_id": args.model_id,
            "learning_rate": args.lr,
            "optimizer": args.optimizer,
            "lr_scheduler": args.lr_scheduler,
            "warmup_ratio": args.warmup_ratio,
            "warmup_steps": resolve_warmup_steps(args.warmup_ratio, num_training_steps),
            "cycle_length": args.cycle_length,
            "min_lr_ratio": args.min_lr_ratio,
            "num_training_steps": num_training_steps,
            "weight_decay": args.weight_decay,
            "lr_adam": args.lr_adam,
            "lr_embedding": args.lr_embedding,
            "norm_method": args.norm_method,
            "n_grpo_steps": args.n_grpo_steps,
            "n_prompts_per_step": args.n_prompts_per_step,
            "group_size": args.group_size,
            "loss_type": args.loss_type,
            "clip_ratio_low": args.clip_ratio_low,
            "clip_ratio_high": args.clip_ratio_high,
            "mask_truncated_completions": args.mask_truncated_completions,
            "dynamic_sampling": args.dynamic_sampling,
            "epochs_per_step": args.epochs_per_step,
            "micro_batch_size": args.micro_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "seed": args.seed,
            "prompt_template": args.prompt_template,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "train_dataset_id": args.train_dataset_id,
            "train_split": args.train_split,
            "val_size": args.val_size,
            "eval_math500_dataset_id": args.eval_math500_dataset_id,
            "eval_math500_split": args.eval_math500_split,
            "eval_aime24_dataset_id": args.eval_aime24_dataset_id,
            "eval_aime24_split": args.eval_aime24_split,
            "final_eval": args.final_eval,
            "eval_pass1": args.eval_pass1,
            "eval_majority": args.eval_majority,
            "eval_majority_k": args.eval_majority_k,
            "eval_temperature": args.eval_temperature,
            "eval_top_p": args.eval_top_p,
            "eval_max_tokens": args.eval_max_tokens,
            "enable_save_ckpt": args.enable_save_ckpt,
            "run_dir": run_dir,
            **mode_info,
        },
        dir=run_dir,
    )


def is_vllm_http_available(vllm_url: str, timeout_sec: float = 2.0) -> bool:
    try:
        response = requests.get(f"{vllm_url.rstrip('/')}/v1/models", timeout=timeout_sec)
    except requests.RequestException:
        return False
    return response.ok


def resolve_lora_rollout_backend(train_mode: str, vllm_url: str) -> str | None:
    if train_mode == "lora_full":
        return "local_inproc"
    if train_mode == "lora":
        return "http" if is_vllm_http_available(vllm_url) else "local_inproc"
    return None


def normalize_lora_merged_weight_name(name: str) -> str | None:
    # Skip adapter-only tensors; local vLLM wants dense base-model parameter names.
    lora_markers = (
        ".lora_A.",
        ".lora_B.",
        ".lora_embedding_A.",
        ".lora_embedding_B.",
        ".lora_magnitude_vector",
    )
    if any(marker in name for marker in lora_markers):
        return None
    return name.replace(".base_layer.", ".")


def build_lora_http_generators(args, model, run_dir):
    loaded_loras = []

    def generate_http(
        prompts: list[str],
        vllm_model_id: str,
        *,
        temperature=0,
        responses_per_prompt=1,
        top_p=1.0,
        max_tokens=1024,
    ):
        api_url = f"{args.vllm_url}/v1/completions"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": vllm_model_id,
            "prompt": prompts,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "n": responses_per_prompt,
        }
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return [choice["text"] for choice in result["choices"]]

    def load_lora(lora_name):
        if lora_name in loaded_loras:
            return
        api_url = f"{args.vllm_url}/v1/load_lora_adapter"
        headers = {"Content-Type": "application/json"}
        lora_path = str(Path.cwd() / lora_name)
        payload = {"lora_name": lora_name, "lora_path": lora_path}
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        loaded_loras.append(lora_name)

    def save_lora(step):
        lora_name = f"{run_dir}/step={step}"
        if not os.path.exists(lora_name):
            model.save_pretrained(lora_name)
        return lora_name

    def generate_with_params(
        prompts: list[str],
        step: int,
        *,
        temperature,
        responses_per_prompt,
        top_p,
        max_tokens,
    ):
        vllm_model_id = save_lora(step)
        load_lora(vllm_model_id)
        return generate_http(
            prompts,
            vllm_model_id=vllm_model_id,
            temperature=temperature,
            responses_per_prompt=responses_per_prompt,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    def generate_for_train(prompts: list[str], step: int):
        return generate_with_params(
            prompts,
            step,
            temperature=1.0,
            responses_per_prompt=args.group_size,
            top_p=1.0,
            max_tokens=args.eval_max_tokens,
        )

    def generate_for_eval(prompts: list[str], step: int):
        return generate_with_params(
            prompts,
            step,
            temperature=0.0,
            responses_per_prompt=1,
            top_p=1.0,
            max_tokens=args.eval_max_tokens,
        )

    return generate_for_train, generate_for_eval, generate_with_params


def build_lora_local_generators(args, model):
    os.environ["VLLM_USE_V1"] = "0"
    from vllm import LLM, SamplingParams

    if not all(hasattr(model, attr) for attr in ("merge_adapter", "unmerge_adapter", "get_base_model")):
        raise RuntimeError(
            "Local LoRA fallback requires PEFT model methods: "
            "merge_adapter, unmerge_adapter, get_base_model."
        )

    vllm_model = LLM(
        model=args.model_id,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=4096,
        logprobs_mode="processed_logprobs",
    )

    def export_lora_merged_weights():
        with torch.no_grad():
            model.merge_adapter()
            try:
                base_model = model.get_base_model()
                weight_tuples = []
                seen = set()
                for name, param in base_model.named_parameters():
                    normalized = normalize_lora_merged_weight_name(name)
                    if normalized is None:
                        continue
                    if normalized in seen:
                        raise RuntimeError(f"Duplicate normalized LoRA weight name: {normalized}")
                    seen.add(normalized)
                    weight_tuples.append((normalized, param))
                return weight_tuples
            finally:
                model.unmerge_adapter()

    def generate(
        prompts: list[str],
        *,
        temperature=0,
        responses_per_prompt=1,
        top_p=1.0,
        max_tokens=1024,
    ):
        weight_tuples = export_lora_merged_weights()
        vllm_internal_model = (
            vllm_model.llm_engine.model_executor.driver_worker.model_runner.model
        )
        vllm_internal_model.load_weights(weight_tuples)

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            n=responses_per_prompt,
            top_p=top_p,
        )
        outputs = vllm_model.generate(prompts, sampling_params)
        return [o.text for output in outputs for o in output.outputs]

    def generate_for_train(prompts: list[str], _step: int):
        return generate(
            prompts,
            temperature=1,
            responses_per_prompt=args.group_size,
            top_p=1.0,
            max_tokens=args.eval_max_tokens,
        )

    def generate_for_eval(prompts: list[str], _step: int):
        return generate(
            prompts,
            temperature=0.0,
            responses_per_prompt=1,
            top_p=1.0,
            max_tokens=args.eval_max_tokens,
        )

    def generate_with_params(
        prompts: list[str],
        _step: int,
        *,
        temperature,
        responses_per_prompt,
        top_p,
        max_tokens,
    ):
        return generate(
            prompts,
            temperature=temperature,
            responses_per_prompt=responses_per_prompt,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    return generate_for_train, generate_for_eval, generate_with_params


def build_local_vllm_generators(args, model):
    os.environ["VLLM_USE_V1"] = "0"
    from vllm import LLM, SamplingParams

    vllm_model = LLM(
        model=args.model_id,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=4096,
        logprobs_mode="processed_logprobs",
    )

    def generate(
        prompts: list[str],
        *,
        temperature=0,
        responses_per_prompt=1,
        top_p=1.0,
        max_tokens=1024,
    ):
        if args.train_mode in {"blocktt", "svd"}:
            weight_tuples = export_weights_for_vllm(model)
        else:
            weight_tuples = [(name, param) for name, param in model.named_parameters()]

        vllm_internal_model = (
            vllm_model.llm_engine.model_executor.driver_worker.model_runner.model
        )
        vllm_internal_model.load_weights(weight_tuples)

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            n=responses_per_prompt,
            top_p=top_p,
        )
        outputs = vllm_model.generate(prompts, sampling_params)
        return [o.text for output in outputs for o in output.outputs]

    def generate_for_train(prompts: list[str], _step: int):
        return generate(
            prompts,
            temperature=1,
            responses_per_prompt=args.group_size,
            top_p=1.0,
            max_tokens=args.eval_max_tokens,
        )

    def generate_for_eval(prompts: list[str], _step: int):
        return generate(
            prompts,
            temperature=0.0,
            responses_per_prompt=1,
            top_p=1.0,
            max_tokens=args.eval_max_tokens,
        )

    def generate_with_params(
        prompts: list[str],
        _step: int,
        *,
        temperature,
        responses_per_prompt,
        top_p,
        max_tokens,
    ):
        return generate(
            prompts,
            temperature=temperature,
            responses_per_prompt=responses_per_prompt,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    return generate_for_train, generate_for_eval, generate_with_params


def validate_trainable_params(trainable_params):
    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters found.")
    if any(not p.requires_grad for p in trainable_params):
        raise ValueError("Found non-trainable parameter in optimizer parameter list.")
    if len({id(p) for p in trainable_params}) != len(trainable_params):
        raise ValueError("Duplicate trainable parameters found.")


def _flatten_param_ids(params_by_group):
    return {id(p) for group in params_by_group for p in group}


def assert_muon_routing(train_mode, trainable_named_params, optimizer):
    if not isinstance(optimizer, Muon):
        return

    matrix_ids = _flatten_param_ids(optimizer._bucket_muon_matrix_by_group)
    cola_ids = _flatten_param_ids(optimizer._bucket_muon_cola_a_by_group)
    cola_ids |= _flatten_param_ids(optimizer._bucket_muon_cola_b_by_group)
    cola_ids |= _flatten_param_ids(optimizer._bucket_muon_cola_g_by_group)
    btt_ids = _flatten_param_ids(optimizer._bucket_btt_by_group)

    def _param_ids_by_suffix(suffixes):
        ids = set()
        for name, p in trainable_named_params:
            if any(s in name for s in suffixes):
                ids.add(id(p))
        return ids

    if train_mode == "svd":
        svd_ids = _param_ids_by_suffix((".svd_a", ".svd_b"))
        if len(svd_ids) == 0:
            raise ValueError(
                "Muon routing check failed: no trainable SVD factors were found."
            )
        missing = sorted(pid for pid in svd_ids if pid not in matrix_ids)
        if missing:
            raise ValueError(
                "Muon routing check failed: some SVD factors are not in matrix-Muon buckets."
            )
        if svd_ids & cola_ids:
            raise ValueError(
                "Muon routing check failed: SVD factors were routed to CoLA buckets."
            )
        return

    if train_mode == "blocktt":
        btt_trainable_ids = _param_ids_by_suffix((".btt_l", ".btt_r"))
        if len(btt_trainable_ids) == 0:
            raise ValueError(
                "Muon routing check failed: no trainable BlockTT factors were found."
            )
        missing = sorted(pid for pid in btt_trainable_ids if pid not in btt_ids)
        if missing:
            raise ValueError(
                "Muon routing check failed: some BlockTT factors are not in BTT-Muon buckets."
            )
        return

    if train_mode in {"lora", "lora_full"}:
        lora_ids = _param_ids_by_suffix(("lora_A", "lora_B"))
        if len(lora_ids) == 0:
            raise ValueError(
                "Muon routing check failed: no trainable LoRA factors were found."
            )
        missing = sorted(pid for pid in lora_ids if pid not in matrix_ids)
        if missing:
            raise ValueError(
                "Muon routing check failed: some LoRA factors are not in matrix-Muon buckets."
            )
        return

    if train_mode == "full":
        if len(matrix_ids) == 0:
            raise ValueError(
                "Muon routing check failed: no matrix-Muon parameters were detected in full mode."
            )


def build_optimizer(args, trainable_params, trainable_named_params):
    if args.optimizer == "adamw":
        return torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    if args.optimizer == "muon":
        return Muon(
            trainable_named_params,
            lr=args.lr,
            lr_adam=args.lr_adam,
            lr_embedding=args.lr_embedding,
            weight_decay=args.weight_decay,
            norm_method=args.norm_method,
        )
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def _get_cyclical_cosine_schedule_with_min_lr_lambda(
    current_step,
    *,
    num_warmup_steps,
    cycle_length,
    min_lr_ratio,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))

    progress = float((current_step - num_warmup_steps) % cycle_length) / float(
        max(1, cycle_length - num_warmup_steps)
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def get_cyclical_cosine_schedule_with_min_lr(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    cycle_length,
    min_lr_ratio=0.1,
    last_epoch=-1,
):
    assert (
        cycle_length is not None or num_training_steps is not None
    ), "You must specify either cycle_length or num_training_steps"

    if cycle_length is None:
        cycle_length = num_training_steps

    if num_training_steps % cycle_length != 0:
        raise ValueError(
            f"num_training_steps ({num_training_steps}) must be divisible by cycle_length ({cycle_length})"
        )

    lr_lambda = partial(
        _get_cyclical_cosine_schedule_with_min_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        cycle_length=cycle_length,
        min_lr_ratio=min_lr_ratio,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def build_lr_scheduler(args, optimizer, num_training_steps):
    warmup_steps = resolve_warmup_steps(args.warmup_ratio, num_training_steps)

    if args.lr_scheduler == "none":
        return None

    if num_training_steps <= 0:
        raise ValueError(
            f"num_training_steps must be > 0 when using lr scheduler, got {num_training_steps}"
        )

    if args.lr_scheduler == "linear":
        return transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            last_epoch=-1,
        )
    if args.lr_scheduler == "cosine":
        return get_cyclical_cosine_schedule_with_min_lr(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            cycle_length=args.cycle_length,
            min_lr_ratio=args.min_lr_ratio,
            last_epoch=-1,
        )
    raise ValueError(f"Unsupported lr scheduler: {args.lr_scheduler}")


def compute_num_training_steps(args):
    if args.micro_batch_size <= 0:
        raise ValueError("--micro-batch-size must be > 0")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be > 0")

    samples_per_step = args.n_prompts_per_step * args.group_size
    micro_batches_per_epoch = math.ceil(samples_per_step / args.micro_batch_size)
    optimizer_steps_per_epoch = math.ceil(
        micro_batches_per_epoch / args.gradient_accumulation_steps
    )
    return args.n_grpo_steps * args.epochs_per_step * optimizer_steps_per_epoch


def resolve_warmup_steps(warmup_ratio, num_training_steps):
    if not 0.0 <= warmup_ratio <= 1.0:
        raise ValueError(f"--warmup-ratio must be in [0, 1], got {warmup_ratio}")
    if num_training_steps < 0:
        raise ValueError(f"num_training_steps must be >= 0, got {num_training_steps}")
    return int(math.ceil(num_training_steps * warmup_ratio))


def collect_rollouts_for_step(args, train_dataset, generate_for_train, step_idx):
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty.")

    target_prompts = args.n_prompts_per_step
    if args.loss_type != "dapo" or not args.dynamic_sampling:
        sample_n = min(len(train_dataset), target_prompts)
        sample_indices = random.sample(range(0, len(train_dataset)), sample_n)
        batch = train_dataset[sample_indices]
        outputs = generate_for_train(batch["prompt"], step_idx)
        return batch["prompt"], batch["answer"], outputs, 1.0

    kept_prompts = []
    kept_answers = []
    kept_outputs = []
    sampled_groups = 0
    kept_groups = 0

    for _ in range(args.dynamic_sampling_max_rounds):
        if len(kept_prompts) >= target_prompts:
            break

        missing = target_prompts - len(kept_prompts)
        sampled = int(math.ceil(missing * args.dynamic_sampling_oversample))
        sample_n = max(missing, sampled)
        sample_n = min(sample_n, len(train_dataset))
        sample_indices = random.sample(range(0, len(train_dataset)), sample_n)
        batch = train_dataset[sample_indices]

        outputs = generate_for_train(batch["prompt"], step_idx)
        raw_reward_tensor = compute_rewards_from_outputs(
            outputs, batch["answer"], args.group_size
        )
        keep_mask = select_informative_groups(raw_reward_tensor)
        prompts, answers, outputs = filter_rollouts_by_group_mask(
            batch["prompt"],
            batch["answer"],
            outputs,
            args.group_size,
            keep_mask,
        )
        kept_prompts.extend(prompts)
        kept_answers.extend(answers)
        kept_outputs.extend(outputs)
        sampled_groups += len(keep_mask)
        kept_groups += sum(keep_mask)

    if len(kept_prompts) < target_prompts:
        missing = target_prompts - len(kept_prompts)
        sample_n = min(len(train_dataset), missing)
        sample_indices = random.sample(range(0, len(train_dataset)), sample_n)
        batch = train_dataset[sample_indices]
        outputs = generate_for_train(batch["prompt"], step_idx)
        kept_prompts.extend(batch["prompt"])
        kept_answers.extend(batch["answer"])
        kept_outputs.extend(outputs)
        sampled_groups += sample_n
        kept_groups += sample_n

    kept_prompts = kept_prompts[:target_prompts]
    kept_answers = kept_answers[:target_prompts]
    kept_outputs = kept_outputs[: target_prompts * args.group_size]
    keep_ratio = kept_groups / max(1, sampled_groups)
    return kept_prompts, kept_answers, kept_outputs, keep_ratio


def evaluate_rows(rows, generate_with_params, step, args, prefix):
    prompts = [row["prompt"] for row in rows]
    answers = [row["answer"] for row in rows]
    metrics = {f"eval/{prefix}/total": len(rows)}
    logs = []

    if len(rows) == 0:
        return metrics

    if args.eval_pass1:
        outputs = generate_with_params(
            prompts,
            step,
            temperature=0.0,
            responses_per_prompt=1,
            top_p=1.0,
            max_tokens=args.eval_max_tokens,
        )
        correct = sum(
            1
            for i, output in enumerate(outputs)
            if (
                (pred := extract_generated_answer(output)) is not None
                and is_equiv(pred, answers[i])
            )
        )
        acc = correct / len(rows)
        metrics[f"eval/{prefix}/pass@1"] = acc
        logs.append(f"pass@1={acc:.2%}")

    if args.eval_majority:
        k = args.eval_majority_k
        outputs = generate_with_params(
            prompts,
            step,
            temperature=args.eval_temperature,
            responses_per_prompt=k,
            top_p=args.eval_top_p,
            max_tokens=args.eval_max_tokens,
        )
        correct = 0
        for i in range(len(rows)):
            start = i * k
            end = start + k
            candidates = [extract_generated_answer(t) for t in outputs[start:end]]
            voted = majority_vote_answer(candidates)
            if voted is not None and is_equiv(voted, answers[i]):
                correct += 1
        acc = correct / len(rows)
        metrics[f"eval/{prefix}/majority@{k}"] = acc
        logs.append(f"majority@{k}={acc:.2%}")

    print(f"Final eval {prefix}: " + ", ".join(logs))
    return metrics


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    validate_mode_specific_flags(args, argv)
    apply_mode_defaults(args)
    require_cuda_for_structured_conversion(args.train_mode, entrypoint="run_rl_dapo.py")
    if args.loss_type == "dapo":
        if args.clip_ratio_low < 0 or args.clip_ratio_high < 0:
            raise ValueError("--clip-ratio-low/high must be non-negative.")
    if args.eval_majority_k <= 0:
        raise ValueError("--eval-majority-k must be > 0.")
    if args.eval_max_tokens <= 0:
        raise ValueError("--eval-max-tokens must be > 0.")
    if args.max_model_len <= 0:
        raise ValueError("--max-model-len must be > 0.")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        raise ValueError("--gpu-memory-utilization must be in (0, 1].")
    if args.dynamic_sampling_max_rounds <= 0:
        raise ValueError("--dynamic-sampling-max-rounds must be > 0.")
    if args.dynamic_sampling_oversample <= 0:
        raise ValueError("--dynamic-sampling-oversample must be > 0.")
    grad_prefixes = None
    if args.save_first_step_grads_prefixes is not None:
        grad_prefixes = [
            p.strip()
            for p in args.save_first_step_grads_prefixes.split(",")
            if p.strip()
        ]

    set_seed(args.seed)
    lora_rollout_backend = resolve_lora_rollout_backend(args.train_mode, args.vllm_url)

    mode_info = {}
    if args.train_mode in {"lora", "lora_full"}:
        lora_target_modules = get_lora_target_modules(args.trainable_type)
        mode_info.update(
            {
                "lora_rank": args.lora_rank,
                "lora_full_backbone_trainable": args.train_mode == "lora_full",
                "trainable_type": args.trainable_type,
                "target_modules": lora_target_modules,
                "vllm_url": args.vllm_url,
                "rollout_backend": lora_rollout_backend,
            }
        )
    elif args.train_mode == "blocktt":
        blocktt_rank = resolve_blocktt_rank(args.blocktt_rank)
        blocktt_targets = get_blocktt_target_module_names(args.trainable_type)
        train_bias = not args.no_train_bias
        train_position = args.train_position
        decomp_mode = args.decomp_mode
        decomp_mode_display = getattr(
            args, "decomp_mode_display", format_blocktt_decomp_mode(decomp_mode)
        )
        module_decomp_modes = getattr(args, "blocktt_module_decomp_modes", None)
        if not isinstance(decomp_mode, dict):
            module_decomp_modes = None
        mode_info.update(
            {
                "blocktt_rank": blocktt_rank,
                "trainable_type": args.trainable_type,
                "decomp_mode": decomp_mode,
                "decomp_mode_by_module": module_decomp_modes,
                "decomp_mode_display": decomp_mode_display,
                "train_position": train_position,
                "s_merged_to": args.s_merged_to,
                "target_modules": blocktt_targets,
                "train_bias": train_bias,
                "blocktt_normalize_after_update": args.blocktt_normalize_after_update,
                "blocktt_factorize_by_head": args.blocktt_factorize_by_head,
            }
        )
    elif args.train_mode == "svd":
        svd_targets = get_svd_target_module_names(args.trainable_type)
        mode_info.update(
            {
                "trainable_type": args.trainable_type,
                "train_position": args.train_position,
                "s_merged_to": args.s_merged_to,
                "target_modules": svd_targets,
            }
        )

    print("Training configuration:")
    print(f"  Train mode: {args.train_mode}")
    print(f"  Model ID: {args.model_id}")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Learning rate: {args.lr}")
    print(f"  vLLM max model len: {args.max_model_len}")
    print(f"  vLLM GPU memory utilization: {args.gpu_memory_utilization}")
    print(f"  Loss type: {args.loss_type}")
    if args.loss_type == "dapo":
        print(f"  Clip ratio low/high: {args.clip_ratio_low}/{args.clip_ratio_high}")
        print(f"  Mask truncated completions: {args.mask_truncated_completions}")
        print(f"  Dynamic sampling: {args.dynamic_sampling}")
    print(f"  LR scheduler: {args.lr_scheduler}")
    print(f"  Warmup ratio: {args.warmup_ratio}")
    if args.lr_scheduler == "cosine":
        print(f"  Cycle length: {args.cycle_length}")
        print(f"  Min LR ratio: {args.min_lr_ratio}")
    print(f"  Weight decay: {args.weight_decay}")
    if args.optimizer == "muon":
        print(f"  Muon lr_adam: {args.lr_adam}")
        print(f"  Muon lr_embedding: {args.lr_embedding}")
        print(f"  Muon norm_method: {args.norm_method}")
    print(f"  RL steps: {args.n_grpo_steps}")
    print(f"  Prompts per step: {args.n_prompts_per_step}")
    print(f"  Group size: {args.group_size}")
    print(f"  Epochs per step: {args.epochs_per_step}")
    print(f"  Micro batch size: {args.micro_batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Train dataset: {args.train_dataset_id} [{args.train_split}]")
    print(f"  Final eval datasets: {args.eval_math500_dataset_id}, {args.eval_aime24_dataset_id}")
    if args.train_mode in {"lora", "lora_full"}:
        print(f"  LoRA rank: {args.lora_rank}")
        print(f"  Trainable type: {args.trainable_type}")
        print(
            "  Backbone trainable: "
            f"{'yes' if args.train_mode == 'lora_full' else 'no'}"
        )
        print(f"  Target modules: {mode_info['target_modules']}")
        print(f"  Rollout backend: {lora_rollout_backend}")
        print(f"  vLLM URL: {args.vllm_url}")
    if args.train_mode == "blocktt":
        print(f"  BlockTT rank: {mode_info['blocktt_rank']}")
        print(f"  Trainable type: {args.trainable_type}")
        print(f"  Decomp mode: {mode_info['decomp_mode_display']}")
        print(f"  Train position: {mode_info['train_position']}")
        print(f"  S merged to: {mode_info['s_merged_to']}")
        print(f"  Train BTT bias: {mode_info['train_bias']}")
        print(
            "  Normalize BTT after update: "
            f"{mode_info['blocktt_normalize_after_update']}"
        )
        print(f"  Factorize by head: {mode_info['blocktt_factorize_by_head']}")
        print(f"  Target modules: {mode_info['target_modules']}")
    if args.train_mode == "svd":
        print(f"  Trainable type: {args.trainable_type}")
        print(f"  Train position: {args.train_position}")
        print(f"  S merged to: {mode_info['s_merged_to']}")
        print(f"  Target modules: {mode_info['target_modules']}")
    print(f"  Save checkpoint: {'enabled' if args.enable_save_ckpt else 'disabled'}")
    print(f"  W&B logging: {'enabled' if not args.no_wandb else 'disabled'}")
    print()

    num_training_steps = compute_num_training_steps(args)
    warmup_steps = resolve_warmup_steps(args.warmup_ratio, num_training_steps)
    print(f"  Optimizer update steps: {num_training_steps}")
    print(f"  Warmup steps (derived): {warmup_steps}")

    run_name = compute_run_name(args, mode_info)
    run_dir = create_run_dir(args.base_dir, args.train_mode, run_name)
    print(f"Created: {run_dir}")

    maybe_init_wandb(args, run_dir, run_name, mode_info, num_training_steps)

    train_dataset, val_dataset, tokenizer, template = load_datasets_and_tokenizer(args)

    device_id = get_local_cuda_device_id()
    device = f"cuda:{device_id}"
    model_kwargs = dict(
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        use_cache=False,
        device_map={"": device_id},
    )
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)

    if args.train_mode in {"lora", "lora_full"}:
        from peft import LoraConfig, get_peft_model

        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=32,
            target_modules=mode_info["target_modules"],
        )
        model = get_peft_model(model, peft_config)
        if args.train_mode == "lora_full":
            for p in model.parameters():
                p.requires_grad = True
        model.print_trainable_parameters()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    elif args.train_mode == "blocktt":
        decomp_mode_for_convert = (
            mode_info["decomp_mode_by_module"]
            if mode_info["decomp_mode_by_module"] is not None
            else mode_info["decomp_mode"]
        )
        converted_modules = convert_linear_to_btt(
            model,
            btt_rank=mode_info["blocktt_rank"],
            decomp_mode=decomp_mode_for_convert,
            init_mode="default",
            include_names=mode_info["target_modules"],
            skip_names=("lm_head",),
            lr_act=False,
            s_merged_to=mode_info["s_merged_to"],
            train_position=mode_info["train_position"],
            factorize_by_head=mode_info.get("blocktt_factorize_by_head", False),
            model_config=model.config,
        )
        stats = configure_blocktt_trainability(
            model,
            train_bias=mode_info["train_bias"],
            train_position=mode_info["train_position"],
        )
        if stats["num_btt_layers"] == 0:
            raise ValueError(
                "No layers were converted to BTT; check --trainable-type selection."
            )
        trainable_params = stats["trainable_params"]
        print(f"Converted modules: {len(converted_modules)}")
        print(
            f"Trainable params: {stats['trainable_param_count']:,} / "
            f"{stats['total_param_count']:,} "
            f"({100 * stats['trainable_param_count'] / stats['total_param_count']:.4f}%)"
        )
        print(
            f"Tuned cores: left={stats['tuned_left_cores']}, "
            f"right={stats['tuned_right_cores']}, biases={stats['tuned_biases']}"
        )
    elif args.train_mode == "svd":
        converted_modules = convert_linear_to_svd(
            model,
            include_names=mode_info["target_modules"],
            skip_names=("lm_head",),
            s_merged_to=mode_info["s_merged_to"],
            train_position=args.train_position,
        )
        stats = configure_svd_trainability(
            model,
            train_position=args.train_position,
            train_bias=True,
            train_embed_lm_head=(args.train_position == "both"),
        )
        if stats["num_svd_layers"] == 0:
            raise ValueError("No layers were converted to SVD; check --trainable-type selection.")
        trainable_params = stats["trainable_params"]
        print(f"Converted modules: {len(converted_modules)}")
        print(
            f"Trainable params: {stats['trainable_param_count']:,} / "
            f"{stats['total_param_count']:,} "
            f"({100 * stats['trainable_param_count'] / stats['total_param_count']:.4f}%)"
        )
        print(
            f"Tuned factors: output={stats['tuned_output_cores']}, "
            f"input={stats['tuned_input_cores']}, biases={stats['tuned_biases']}"
        )
    else:
        trainable_params = list(model.parameters())

    trainable_named_params = [
        (name, p) for name, p in model.named_parameters() if p.requires_grad
    ]
    trainable_params = [p for _, p in trainable_named_params]
    validate_trainable_params(trainable_params)

    if args.train_mode in {"lora", "lora_full"}:
        if lora_rollout_backend == "http":
            generate_for_train, generate_for_eval, generate_with_params = build_lora_http_generators(
                args,
                model,
                run_dir,
            )
        else:
            generate_for_train, generate_for_eval, generate_with_params = build_lora_local_generators(
                args,
                model,
            )
    else:
        generate_for_train, generate_for_eval, generate_with_params = build_local_vllm_generators(args, model)

    optimizer = build_optimizer(args, trainable_params, trainable_named_params)
    if args.optimizer == "muon":
        assert_muon_routing(args.train_mode, trainable_named_params, optimizer)
    scheduler = build_lr_scheduler(args, optimizer, num_training_steps)
    normalize_blocktt_after_update = (
        args.train_mode == "blocktt" and args.blocktt_normalize_after_update
    )
    first_step_grads_saved = False
    optimizer_steps = 0
    save_grads_steps = parse_save_grads_steps(args.save_grads_steps)
    stop_requested = False

    def eval_model(step):
        n = min(len(val_dataset), 1000)
        val_prompts = val_dataset[:n]["prompt"]
        val_answers = val_dataset[:n]["answer"]

        eval_start = time.time()
        outputs = generate_for_eval(val_prompts, step)
        eval_time = time.time() - eval_start

        correct = 0
        for i in range(len(outputs)):
            correct_answer = val_answers[i]
            generated_answer = extract_generated_answer(outputs[i])
            if generated_answer is not None and is_equiv(generated_answer, correct_answer):
                correct += 1

        accuracy = correct / len(outputs)
        print(f"step={step}, correct: {correct} / {len(outputs)} ({accuracy:.2%})")

        if not args.no_wandb:
            wandb.log(
                {
                    "eval/accuracy": accuracy,
                    "eval/correct": correct,
                    "eval/total": len(outputs),
                    "eval/time_seconds": eval_time,
                },
                step=step,
            )

    print("Starting initial evaluation...")
    model = model.to(device)
    eval_model(0)
    model.train()

    for i in range(args.n_grpo_steps):
        step_start_time = time.time()

        generation_start = time.time()
        prompts, answers, outputs, dynamic_keep_ratio = collect_rollouts_for_step(
            args,
            train_dataset,
            generate_for_train,
            i,
        )
        generation_time = time.time() - generation_start

        raw_reward_tensor = compute_rewards_from_outputs(outputs, answers, args.group_size)
        advantages = compute_advantages(raw_reward_tensor)

        train_accuracy = raw_reward_tensor.mean().item()
        reward_std = raw_reward_tensor.std().item()
        reward_max = raw_reward_tensor.max().item()
        reward_min = raw_reward_tensor.min().item()
        advantage_mean = advantages.mean().item()
        advantage_std = advantages.std().item()
        advantage_max = advantages.max().item()
        advantage_min = advantages.min().item()

        prompts_expanded = [x for x in prompts for _ in range(args.group_size)]
        data = tokenize_prompt_and_output(
            prompts_expanded,
            outputs,
            tokenizer,
            max_completion_tokens=args.eval_max_tokens,
            mask_truncated=(args.loss_type == "dapo" and args.mask_truncated_completions),
        )
        input_ids = data["input_ids"].to(device)
        labels = data["labels"].to(device)
        response_mask = data["response_mask"].to(device)
        truncated_fraction = data["is_truncated"].float().mean().item()

        with torch.inference_mode():
            old_logprobs_all = []
            for idx in range(0, len(input_ids), args.micro_batch_size):
                end = min(idx + args.micro_batch_size, len(input_ids))
                old_logprobs_all.append(
                    get_response_log_probs(model, input_ids[idx:end], labels[idx:end]).detach()
                )
            old_logprobs_all = torch.cat(old_logprobs_all, dim=0)
            assert old_logprobs_all.shape == labels.shape

        epoch_grad_norms = []
        epoch_policy_ratios = []
        epoch_kl_divs = []
        epoch_clip_fracs = []

        training_start = time.time()
        for epoch in range(args.epochs_per_step):
            batch_starts = list(range(0, len(input_ids), args.micro_batch_size))
            for b in tqdm(
                range(len(batch_starts)),
                desc=(
                    f"Step {i + 1}/{args.n_grpo_steps}, "
                    f"Epoch {epoch + 1}/{args.epochs_per_step}"
                ),
            ):
                idx = batch_starts[b]
                end = min(idx + args.micro_batch_size, len(input_ids))
                x = input_ids[idx:end]
                y = labels[idx:end]
                mask = response_mask[idx:end]
                micro_batch_adv = advantages[idx:end].to(device)

                policy_logprobs = get_response_log_probs(model, x, y)
                old_logprobs = old_logprobs_all[idx:end]

                ratio = torch.exp(policy_logprobs - old_logprobs)
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)) * mask
                    approx_kl_mean = approx_kl.sum() / mask.sum().clamp_min(1)
                    epoch_kl_divs.append(approx_kl_mean.item())

                    policy_ratio_mean = (ratio * mask).sum() / mask.sum().clamp_min(1)
                    epoch_policy_ratios.append(policy_ratio_mean.item())

                loss_value, clip_frac = compute_policy_loss(
                    args.loss_type,
                    ratio,
                    micro_batch_adv,
                    mask,
                    clip_ratio_low=args.clip_ratio_low,
                    clip_ratio_high=args.clip_ratio_high,
                )
                epoch_clip_fracs.append(clip_frac)
                loss = loss_value / args.gradient_accumulation_steps

                loss.backward()

                should_step = (
                    ((b + 1) % args.gradient_accumulation_steps == 0)
                    or (b + 1 == len(batch_starts))
                )
                if should_step:
                    if (
                        args.save_first_step_grads_path is not None
                        and not first_step_grads_saved
                    ):
                        grad_payload = {}
                        for name, p in trainable_named_params:
                            if p.grad is None:
                                continue
                            if grad_prefixes is not None and not any(
                                name.startswith(prefix) for prefix in grad_prefixes
                            ):
                                continue
                            grad_payload[name] = p.grad.detach().cpu()
                        if len(grad_payload) == 0:
                            raise ValueError(
                                "No gradients matched --save-first-step-grads-prefixes "
                                f"for path {args.save_first_step_grads_path}"
                            )
                        grad_dir = os.path.dirname(args.save_first_step_grads_path)
                        if grad_dir:
                            os.makedirs(grad_dir, exist_ok=True)
                        save_safetensors_file(
                            grad_payload,
                            args.save_first_step_grads_path,
                        )
                        first_step_grads_saved = True
                        print(
                            "Saved first-step gradients to "
                            f"{args.save_first_step_grads_path} "
                            f"({len(grad_payload)} tensors)"
                        )
                    if optimizer_steps in save_grads_steps:
                        save_gradients(
                            trainable_named_params, run_dir, optimizer_steps,
                            subdir="optim_step",
                        )
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    epoch_grad_norms.append(grad_norm.item())
                    optimizer.step()
                    if normalize_blocktt_after_update:
                        normalize_trainable_blocktt_cores_(model)
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()
                    optimizer_steps += 1
                    if args.stop_after_first_step and optimizer_steps >= 1:
                        stop_requested = True
                        break
            if stop_requested:
                break
        if stop_requested:
            print("Stopping early after first optimizer step (--stop-after-first-step).")

        training_time = time.time() - training_start
        step_time = time.time() - step_start_time

        mean_grad_norm = (
            sum(epoch_grad_norms) / len(epoch_grad_norms) if epoch_grad_norms else 0
        )
        mean_policy_ratio = (
            sum(epoch_policy_ratios) / len(epoch_policy_ratios)
            if epoch_policy_ratios
            else 0
        )
        mean_kl = sum(epoch_kl_divs) / len(epoch_kl_divs) if epoch_kl_divs else 0
        mean_clip_frac = (
            sum(epoch_clip_fracs) / len(epoch_clip_fracs) if epoch_clip_fracs else 0
        )

        total_tokens = response_mask.sum().item()
        tokens_per_sec = total_tokens / generation_time if generation_time > 0 else 0
        avg_generation_len = int(response_mask.sum(dim=-1).float().mean().item())

        if not args.no_wandb:
            wandb.log(
                {
                    "train/accuracy": train_accuracy,
                    "train/reward_mean": train_accuracy,
                    "train/reward_std": reward_std,
                    "train/reward_max": reward_max,
                    "train/reward_min": reward_min,
                    "train/advantage_mean": advantage_mean,
                    "train/advantage_std": advantage_std,
                    "train/advantage_max": advantage_max,
                    "train/advantage_min": advantage_min,
                    "train/grad_norm": mean_grad_norm,
                    "train/policy_ratio": mean_policy_ratio,
                    "train/approx_kl": mean_kl,
                    "train/loss_type": 1 if args.loss_type == "dapo" else 0,
                    "train/clip_frac": mean_clip_frac,
                    "train/active_tokens": total_tokens,
                    "train/dynamic_keep_ratio": dynamic_keep_ratio,
                    "train/truncated_fraction": truncated_fraction,
                    "train/avg_gen_length": avg_generation_len,
                    "time/step_time": step_time,
                    "time/generation_time": generation_time,
                    "time/training_time": training_time,
                    "time/tokens_per_sec": tokens_per_sec,
                },
                step=i + 1,
            )

        print(
            f"Step {i + 1}/{args.n_grpo_steps} | "
            f"Train Acc: {train_accuracy:.2%} | KL: {mean_kl:.4f} | "
            f"Time: {step_time:.1f}s"
        )

        if (i + 1) % 5 == 0:
            eval_model(i + 1)

        # Save checkpoints after first update, at step 10, and at final step
        if args.enable_save_ckpt:
            step_num = i + 1
            if should_save_checkpoint(step_num, args.n_grpo_steps):
                save_checkpoint(model, tokenizer, run_dir, step_num)
        if stop_requested:
            break

    if args.final_eval:
        print("Running final benchmark evaluation (MATH-500 + AIME-24)...")

        math500_rows = [
            row
            for row in load_eval_dataset(
                args.eval_math500_dataset_id,
                args.eval_math500_split,
                tokenizer,
                template,
            )
            if row["answer"] is not None
        ]

        aime24_raw = load_dataset(
            args.eval_aime24_dataset_id,
            split=args.eval_aime24_split,
        )
        aime24_unique = unique_by_problem(aime24_raw)
        aime24_rows = []
        for row in aime24_unique:
            processed = process_math_example(row, tokenizer, template)
            if processed["answer"] is not None:
                aime24_rows.append(processed)

        final_step = args.n_grpo_steps
        math500_metrics = evaluate_rows(
            math500_rows,
            generate_with_params,
            final_step,
            args,
            prefix="math500",
        )
        aime24_metrics = evaluate_rows(
            aime24_rows,
            generate_with_params,
            final_step,
            args,
            prefix="aime24",
        )
        dedup_ratio = len(aime24_unique) / max(1, len(aime24_raw))
        extra_metrics = {"eval/aime24/dedup_ratio": dedup_ratio}

        if not args.no_wandb:
            wandb.log({**math500_metrics, **aime24_metrics, **extra_metrics}, step=final_step)

    if not args.enable_save_ckpt:
        print("Checkpoint saving disabled (--enable-save-ckpt not set).")

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
