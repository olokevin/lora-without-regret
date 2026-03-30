# run_kd.py
"""
Knowledge distillation training script.

Trains a student model on precomputed teacher data (from generate_teacher_data.py).

Examples:
  CUDA_VISIBLE_DEVICES=0 uv run run_kd.py --kd-loss-type sft --train-mode svd --train-position output --no-wandb
  CUDA_VISIBLE_DEVICES=0 uv run run_kd.py --kd-loss-type kl --train-mode blocktt --train-position small --no-wandb
"""

import argparse
import json
import math
import os
import sys
from functools import partial

import torch
import wandb
from btt_layer import (
    BLOCKTT_DECOMP_GROUP_TO_MODULES,
    configure_blocktt_trainability,
    convert_linear_to_btt,
    get_blocktt_target_module_names,
    normalize_trainable_blocktt_cores_,
    resolve_blocktt_decomp_modes,
)
from safetensors.torch import save_file as save_safetensors_file
from svd_layer import (
    configure_svd_trainability,
    convert_linear_to_svd,
    get_svd_target_module_names,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from run_sft import (
    build_collate_fn,
    build_lr_scheduler,
    build_optimizer,
    compute_num_training_steps,
    format_blocktt_decomp_mode,
    get_lora_target_modules,
    prepare_model,
    resolve_blocktt_rank,
    resolve_warmup_steps,
    set_seed,
    validate_trainable_params,
)


MODE_DEFAULTS = {
    "full": {
        "lr": 5e-6,
        "wandb_project": "kd-full",
        "output_dir": "./kd_full_model",
    },
    "lora": {
        "lr": 1e-4,
        "wandb_project": "kd-lora",
        "output_dir": "./kd_lora_model",
    },
    "blocktt": {
        "lr": 1e-4,
        "wandb_project": "kd-blocktt",
        "output_dir": "./kd_blocktt_model",
    },
    "svd": {
        "lr": 1e-4,
        "wandb_project": "kd-svd",
        "output_dir": "./kd_svd_model",
    },
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Knowledge distillation training script")

    # KD-specific args
    parser.add_argument(
        "--kd-loss-type",
        type=str,
        required=True,
        choices=["sft", "kl"],
        help="KD loss type: sft (train on teacher completions) or kl (KL divergence on logits)",
    )
    parser.add_argument(
        "--teacher-data-dir",
        type=str,
        default="/data/yequan/fura/kd/DeepSeek-R1-Distill-Qwen-7B-competition_math",
        help="Path to precomputed teacher data directory",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=256,
        help="Top-K for KL loss; must match generated data (default: 256)",
    )
    parser.add_argument(
        "--save-steps",
        type=str,
        default="1,10,final",
        help="Comma-separated optimizer steps to save checkpoints (default: 1,10,final)",
    )

    # Model args
    parser.add_argument(
        "--train-mode",
        type=str,
        required=True,
        choices=["full", "lora", "blocktt", "svd"],
        help="Training mode: full, lora, blocktt, or svd",
    )
    parser.add_argument(
        "--student-model-id",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Student model ID (default: Qwen/Qwen2.5-0.5B)",
    )
    # Alias so prepare_model can use args.model_id
    parser.add_argument("--model-id", type=str, default=None, help=argparse.SUPPRESS)

    # Optimizer args (same as run_sft.py)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "muon"])
    parser.add_argument("--lr-scheduler", type=str, default="none", choices=["none", "linear", "cosine"])
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--cycle-length", type=int, default=None)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lr-adam", type=float, default=None)
    parser.add_argument("--lr-embedding", type=float, default=None)
    parser.add_argument("--norm-method", type=str, default=None, choices=["row", "col", "row-col", "col-row", "shape"])

    # LoRA-only
    parser.add_argument("--lora-rank", type=int, default=128)

    # BlockTT-only
    parser.add_argument("--decomp-mode", type=str, default="input_one_block")
    parser.add_argument("--blocktt-rank", type=str, default="full")
    parser.add_argument("--no-train-bias", action="store_true")
    parser.add_argument("--blocktt-normalize-after-update", action="store_true")

    # Shared trainable module selector
    parser.add_argument("--trainable-type", type=str, default="all", choices=["all", "mlp", "attn"])
    parser.add_argument(
        "--train-position", type=str, default=None,
        choices=["output", "input", "small", "large", "both"],
    )
    parser.add_argument(
        "--s-merged-to", type=str, default=None,
        choices=["frozen", "trainable", "output", "input", "split", "keep"],
    )

    # Training args
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--enable-save-ckpt", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # Wandb
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")

    args = parser.parse_args(argv)
    # Sync model_id for prepare_model compatibility
    if args.model_id is None:
        args.model_id = args.student_model_id
    return args


def apply_mode_defaults(args):
    defaults = MODE_DEFAULTS[args.train_mode]
    if args.lr is None:
        args.lr = defaults["lr"]
    if args.wandb_project is None:
        args.wandb_project = defaults["wandb_project"]
    if args.output_dir is None:
        args.output_dir = defaults["output_dir"]
    if args.train_mode == "blocktt" and args.train_position is None:
        args.train_position = "small"
    if args.train_mode == "svd" and args.train_position is None:
        args.train_position = "output"
    if args.train_mode in {"blocktt", "svd"} and args.s_merged_to is None:
        if args.train_mode == "blocktt" and args.train_position == "both":
            args.s_merged_to = "split"
        else:
            args.s_merged_to = "frozen"


def _flag_was_passed(argv, flag_name):
    return any(
        token == flag_name or token.startswith(f"{flag_name}=") for token in argv
    )


def validate_mode_specific_flags(args, argv):
    """Same validation logic as run_sft.py."""
    mode_to_flag_sets = {
        "lora": ["--lora-rank"],
        "blocktt": [
            "--decomp-mode",
            "--blocktt-rank",
            "--no-train-bias",
            "--blocktt-normalize-after-update",
        ],
        "svd": [],
    }

    if args.train_mode != "lora":
        passed = [f for f in mode_to_flag_sets["lora"] if _flag_was_passed(argv, f)]
        if passed:
            raise ValueError(f"{', '.join(passed)} is only valid when --train-mode lora")

    if args.train_mode != "blocktt":
        passed = [f for f in mode_to_flag_sets["blocktt"] if _flag_was_passed(argv, f)]
        if passed:
            raise ValueError(f"{', '.join(passed)} is only valid when --train-mode blocktt")
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
        raise ValueError("--trainable-type is only valid when --train-mode lora, blocktt, or svd")

    train_position_passed = _flag_was_passed(argv, "--train-position")
    if args.train_mode in {"full", "lora"} and train_position_passed:
        raise ValueError("--train-position is only valid when --train-mode blocktt or svd")
    if args.train_mode == "blocktt" and train_position_passed:
        if args.train_position not in {"small", "large", "both"}:
            raise ValueError("--train-position for blocktt must be one of: small, large, both")
    if args.train_mode == "svd" and train_position_passed:
        if args.train_position not in {"output", "input"}:
            raise ValueError("--train-position for svd must be one of: output, input")

    s_merged_to_passed = _flag_was_passed(argv, "--s-merged-to")
    if args.train_mode in {"full", "lora"} and s_merged_to_passed:
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


def parse_save_steps(save_steps_str, total_steps):
    steps = set()
    for part in save_steps_str.split(","):
        part = part.strip()
        if part == "final":
            steps.add(total_steps)
        else:
            steps.add(int(part))
    return steps
