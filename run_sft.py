"""
Unified SFT training entrypoint.

Examples:
  CUDA_VISIBLE_DEVICES=0 uv run run_sft.py --train-mode full --no-wandb
  CUDA_VISIBLE_DEVICES=0 uv run run_sft.py --train-mode lora --lora-rank 64 --trainable-type all --no-wandb
  CUDA_VISIBLE_DEVICES=0 uv run run_sft.py --train-mode blocktt --trainable-type all --decomp-mode input_one_block --train-position small --no-wandb
  CUDA_VISIBLE_DEVICES=0 uv run run_sft.py --train-mode svd --trainable-type all --train-position output --no-wandb
"""

import argparse
import math
import random
import sys
from functools import partial

import numpy as np
import os
import time
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
from datasets import load_dataset
from optim.muon import Muon
from peft import LoraConfig, get_peft_model
from svd_layer import (
    configure_svd_trainability,
    convert_linear_to_svd,
    get_svd_target_module_names,
)
from safetensors.torch import save_file as save_safetensors_file
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

MODE_DEFAULTS = {
    "full": {
        "lr": 5e-6,
        "wandb_project": "full-finetuning",
    },
    "lora": {
        "lr": 1e-4,
        "wandb_project": "lora-finetuning",
    },
    "blocktt": {
        "lr": 1e-4,
        "wandb_project": "blocktt-finetuning",
    },
    "svd": {
        "lr": 1e-4,
        "wandb_project": "svd-finetuning",
    },
}


def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Unified SFT training script")
    parser.add_argument(
        "--train-mode",
        type=str,
        required=True,
        choices=["full", "lora", "blocktt", "svd"],
        help="Training mode: full, lora, blocktt, or svd",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen3-4B",
        help="Model ID from HuggingFace Hub (default: Qwen/Qwen3-4B)",
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
        default=0.01,
        help="Weight decay for optimizer (default: 0.01)",
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

    # LoRA-only args
    parser.add_argument(
        "--lora-rank", type=int, default=128, help="LoRA rank (default: 128)"
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
        help="BTT rank; default full for lossless initialization",
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
        default=None,
        help="W&B project name (default depends on train mode)",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name (default: auto-generated)",
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Training batch size (default: 2)"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=16,
        help="Gradient accumulation steps (default: 16)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1)",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/data/yequan/fura/sft_runs",
        help="Base directory for saving runs",
    )
    parser.add_argument(
        "--enable-save-ckpt",
        action="store_true",
        help="Save checkpoints at step 10, 30, and final step (default: disabled)",
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args(argv)


def apply_mode_defaults(args):
    defaults = MODE_DEFAULTS[args.train_mode]
    if args.lr is None:
        args.lr = defaults["lr"]
    if args.wandb_project is None:
        args.wandb_project = defaults["wandb_project"]
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
    mode_to_flag_sets = {
        "lora": ["--lora-rank"],
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

    if args.train_mode != "lora":
        passed = [f for f in mode_to_flag_sets["lora"] if _flag_was_passed(argv, f)]
        if passed:
            raise ValueError(
                f"{', '.join(passed)} is only valid when --train-mode lora"
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


def resolve_blocktt_rank(rank_arg):
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


def build_tokenize_function(tokenizer, max_length=2048):
    gen_prompt_len = len(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "test"}], add_generation_prompt=True
        )
    ) - len(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "test"}], add_generation_prompt=False
        )
    )

    gen_prompt_tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": "test"}], add_generation_prompt=True
    )[-gen_prompt_len:]

    def tokenize_function(examples):
        all_input_ids = []
        all_labels = []
        all_attention_masks = []

        for messages in examples["messages"]:
            current_length = 0
            input_ids = []
            labels = []

            for i, message in enumerate(messages):
                text_so_far = tokenizer.apply_chat_template(
                    messages[: i + 1], tokenize=False
                )
                tokens_so_far = tokenizer(
                    text_so_far,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_length,
                )["input_ids"]

                new_tokens = tokens_so_far[current_length:]
                input_ids.extend(new_tokens)

                if message["role"] == "assistant":
                    if new_tokens[:gen_prompt_len] == gen_prompt_tokens:
                        new_tokens[:gen_prompt_len] = [-100] * gen_prompt_len
                    labels.extend(new_tokens)
                else:
                    labels.extend([-100] * len(new_tokens))

                current_length = len(tokens_so_far)

            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                labels = labels[:max_length]

            attention_mask = [1] * len(input_ids)
            all_input_ids.append(input_ids)
            all_labels.append(labels)
            all_attention_masks.append(attention_mask)

        return {
            "input_ids": all_input_ids,
            "labels": all_labels,
            "attention_mask": all_attention_masks,
        }

    return tokenize_function


def build_collate_fn(tokenizer):
    def collate_fn(batch):
        max_len = max(len(item["input_ids"]) for item in batch)

        input_ids = []
        labels = []
        attention_mask = []

        for item in batch:
            padding_len = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [tokenizer.pad_token_id] * padding_len)
            attention_mask.append(item["attention_mask"] + [0] * padding_len)
            labels.append(item["labels"] + [-100] * padding_len)

        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attention_mask),
        }

    return collate_fn


def prepare_model(args):
    model_kwargs = dict(
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        use_cache=False,
        device_map="auto",
    )
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)

    mode_info = {}

    if args.train_mode == "full":
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        trainable_count = sum(p.numel() for p in trainable_params)
        total_count = sum(p.numel() for p in model.parameters())
        print(
            f"Trainable params: {trainable_count:,} || All params: {total_count:,} || "
            f"Trainable%: {100 * trainable_count / total_count:.2f}"
        )
        mode_info["wandb_extra"] = {}
        mode_info["print_lines"] = []

    elif args.train_mode == "lora":
        target_modules = get_lora_target_modules(args.trainable_type)
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=32,
            target_modules=target_modules,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        mode_info["wandb_extra"] = {
            "lora_rank": args.lora_rank,
            "lora_alpha": 32,
            "trainable_type": args.trainable_type,
            "target_modules": target_modules,
        }
        mode_info["print_lines"] = [
            f"  LoRA rank: {args.lora_rank}",
            f"  Trainable type: {args.trainable_type}",
            f"  Target modules: {target_modules}",
        ]

    elif args.train_mode == "blocktt":
        blocktt_rank = resolve_blocktt_rank(args.blocktt_rank)
        target_modules = get_blocktt_target_module_names(args.trainable_type)
        train_bias = not args.no_train_bias
        train_position = args.train_position
        decomp_mode = args.decomp_mode
        decomp_mode_display = getattr(
            args, "decomp_mode_display", format_blocktt_decomp_mode(decomp_mode)
        )
        module_decomp_modes = getattr(args, "blocktt_module_decomp_modes", None)
        if not isinstance(decomp_mode, dict):
            module_decomp_modes = None

        converted_modules = convert_linear_to_btt(
            model,
            btt_rank=blocktt_rank,
            decomp_mode=module_decomp_modes if module_decomp_modes is not None else decomp_mode,
            init_mode="default",
            include_names=target_modules,
            skip_names=("lm_head",),
            lr_act=False,
            s_merged_to=args.s_merged_to,
            train_position=train_position,
            factorize_by_head=args.blocktt_factorize_by_head,
            model_config=model.config,
        )
        stats = configure_blocktt_trainability(
            model,
            train_bias=train_bias,
            train_position=train_position,
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

        mode_info["wandb_extra"] = {
            "blocktt_rank": blocktt_rank,
            "trainable_type": args.trainable_type,
            "decomp_mode": decomp_mode,
            "decomp_mode_by_module": module_decomp_modes,
            "decomp_mode_display": decomp_mode_display,
            "train_position": train_position,
            "s_merged_to": args.s_merged_to,
            "train_bias": train_bias,
            "blocktt_normalize_after_update": args.blocktt_normalize_after_update,
            "blocktt_factorize_by_head": args.blocktt_factorize_by_head,
            "target_modules": target_modules,
        }
        mode_info["print_lines"] = [
            f"  BlockTT rank: {blocktt_rank}",
            f"  Trainable type: {args.trainable_type}",
            f"  Decomp mode: {decomp_mode_display}",
            f"  Train position: {train_position}",
            f"  S merged to: {args.s_merged_to}",
            f"  Train BTT bias: {train_bias}",
            f"  Normalize BTT after update: {args.blocktt_normalize_after_update}",
            f"  Factorize by head: {args.blocktt_factorize_by_head}",
            f"  Target modules: {target_modules}",
        ]

    else:
        target_modules = get_svd_target_module_names(args.trainable_type)
        converted_modules = convert_linear_to_svd(
            model,
            include_names=target_modules,
            skip_names=("lm_head",),
            s_merged_to=args.s_merged_to,
            train_position=args.train_position,
        )
        stats = configure_svd_trainability(
            model,
            train_position=args.train_position,
            train_bias=True,
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

        mode_info["wandb_extra"] = {
            "trainable_type": args.trainable_type,
            "train_position": args.train_position,
            "s_merged_to": args.s_merged_to,
            "target_modules": target_modules,
        }
        mode_info["print_lines"] = [
            f"  Trainable type: {args.trainable_type}",
            f"  Train position: {args.train_position}",
            f"  S merged to: {args.s_merged_to}",
            f"  Target modules: {target_modules}",
        ]

    trainable_named_params = [
        (name, p) for name, p in model.named_parameters() if p.requires_grad
    ]
    trainable_params = [p for _, p in trainable_named_params]
    return model, trainable_params, trainable_named_params, mode_info


def validate_trainable_params(trainable_params):
    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters found.")
    if any(not p.requires_grad for p in trainable_params):
        raise ValueError("Found non-trainable parameter in optimizer parameter list.")
    if len({id(p) for p in trainable_params}) != len(trainable_params):
        raise ValueError("Duplicate trainable parameters found.")


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


def compute_run_name(args, mode_info: dict) -> str:
    """Compute a human-readable run name (used for both directory and W&B)."""
    if args.wandb_run_name is not None:
        return args.wandb_run_name
    if args.train_mode == "full":
        return f"{args.model_id}_{args.lr:.1e}_full"
    if args.train_mode == "lora":
        return f"{args.model_id}_{args.lr:.1e}_r{args.lora_rank}"
    if args.train_mode == "blocktt":
        decomp_mode_name = mode_info.get("decomp_mode_display", args.decomp_mode)
        return f"{args.model_id}_{args.lr:.1e}_{decomp_mode_name}_{args.train_position}_{args.trainable_type}"
    return f"{args.model_id}_{args.lr:.1e}_{args.train_position}_{args.trainable_type}"


def create_run_dir(base_dir: str, train_mode: str, run_name: str) -> str:
    """Create run directory at {base_dir}/{train_mode}/{run_name}-{timestamp}."""
    timestamp = time.strftime("%m%d-%H%M%S")
    run_dir = os.path.join(base_dir, train_mode, f"{run_name}-{timestamp}")
    os.makedirs(run_dir)
    return run_dir


def save_sft_checkpoint(model, tokenizer, run_dir, step):
    """Save model checkpoint to {run_dir}/step={step}/."""
    ckpt_dir = os.path.join(run_dir, f"step={step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Saving checkpoint to {ckpt_dir}")
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    print(f"Checkpoint saved to {ckpt_dir}")


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


def compute_num_training_steps(num_batches, num_epochs, gradient_accumulation_steps):
    if gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be > 0")
    return num_epochs * (num_batches // gradient_accumulation_steps)


def resolve_warmup_steps(warmup_ratio, num_training_steps):
    if not 0.0 <= warmup_ratio <= 1.0:
        raise ValueError(f"--warmup-ratio must be in [0, 1], got {warmup_ratio}")
    if num_training_steps < 0:
        raise ValueError(f"num_training_steps must be >= 0, got {num_training_steps}")
    return int(math.ceil(num_training_steps * warmup_ratio))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    validate_mode_specific_flags(args, argv)
    apply_mode_defaults(args)

    set_seed(args.seed)

    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    model, trainable_params, trainable_named_params, mode_info = prepare_model(args)
    validate_trainable_params(trainable_params)

    run_name = compute_run_name(args, mode_info)
    run_dir = create_run_dir(args.base_dir, args.train_mode, run_name)
    print(f"Created: {run_dir}")

    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "train_mode": args.train_mode,
                "model_id": args.model_id,
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "effective_batch_size": effective_batch_size,
                "num_epochs": args.num_epochs,
                "max_length": 2048,
                "optimizer": args.optimizer,
                "lr_scheduler": args.lr_scheduler,
                "warmup_ratio": args.warmup_ratio,
                "cycle_length": args.cycle_length,
                "min_lr_ratio": args.min_lr_ratio,
                "weight_decay": args.weight_decay,
                "lr_adam": args.lr_adam,
                "lr_embedding": args.lr_embedding,
                "norm_method": args.norm_method,
                "seed": args.seed,
                "enable_save_ckpt": args.enable_save_ckpt,
                **mode_info["wandb_extra"],
            },
            tags=["sft", args.train_mode],
        )

    print("Training configuration:")
    print(f"  Train mode: {args.train_mode}")
    print(f"  Model ID: {args.model_id}")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  LR scheduler: {args.lr_scheduler}")
    print(f"  Warmup ratio: {args.warmup_ratio}")
    if args.optimizer == "muon":
        print(f"  Muon Adam LR: {args.lr_adam if args.lr_adam is not None else args.lr}")
        if args.lr_embedding is None:
            embedding_lr = args.lr_adam if args.lr_adam is not None else args.lr
            print(f"  Muon Embedding LR: {embedding_lr}")
        else:
            print(f"  Muon Embedding LR: {args.lr_embedding}")
        print(f"  Muon norm method: {args.norm_method}")
    if args.lr_scheduler == "cosine":
        print(f"  Cycle length: {args.cycle_length}")
        print(f"  Min LR ratio: {args.min_lr_ratio}")
    for line in mode_info["print_lines"]:
        print(line)
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"  Save checkpoint: {'enabled' if args.enable_save_ckpt else 'disabled'}")
    print(f"  W&B logging: {'enabled' if use_wandb else 'disabled'}")
    print()

    dataset = load_dataset("HuggingFaceH4/no_robots", split="train[:6400]")
    val_dataset = load_dataset("HuggingFaceH4/no_robots", split="test[:100]")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    tokenize_function = build_tokenize_function(tokenizer)

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )

    tokenized_val_dataset = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing dataset",
    )

    collate_fn = build_collate_fn(tokenizer)
    train_dataloader = DataLoader(
        tokenized_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_dataloader = DataLoader(
        tokenized_val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    gradient_accumulation_steps = args.gradient_accumulation_steps
    device = model.device
    num_training_steps = compute_num_training_steps(
        num_batches=len(train_dataloader),
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    warmup_steps = resolve_warmup_steps(args.warmup_ratio, num_training_steps)
    ckpt_save_steps = {10, 30, num_training_steps}
    save_grads_steps = parse_save_grads_steps(args.save_grads_steps)
    print(f"  Optimizer update steps: {num_training_steps}")
    print(f"  Warmup steps (derived): {warmup_steps}")
    optimizer = build_optimizer(args, trainable_params, trainable_named_params)
    scheduler = build_lr_scheduler(args, optimizer, num_training_steps)

    def eval_model(step=None):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = model(**batch).loss
                total_loss += loss.item()
        val_loss = total_loss / len(val_dataloader)
        print(f"val_loss: {val_loss:.4f} at step {step}")

        if use_wandb and step is not None:
            wandb.log({"val/loss": val_loss}, step=step)

        model.train()
        return val_loss

    print("Starting initial evaluation...")
    eval_model()
    model.train()
    global_step = 0
    total_loss = 0
    prev_step_loss_acc = 0
    prev_step_loss = 0
    normalize_blocktt_after_update = (
        args.train_mode == "blocktt" and args.blocktt_normalize_after_update
    )

    for epoch in range(args.num_epochs):
        progress_bar = tqdm(
            train_dataloader, desc=f"Epoch {epoch + 1}/{args.num_epochs}"
        )

        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / gradient_accumulation_steps
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                if global_step in save_grads_steps:
                    save_gradients(trainable_named_params, run_dir, global_step)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                if normalize_blocktt_after_update:
                    normalize_trainable_blocktt_cores_(model)
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if args.enable_save_ckpt and global_step in ckpt_save_steps:
                    save_sft_checkpoint(model, tokenizer, run_dir, global_step)

                if use_wandb:
                    wandb.log(
                        {
                            "train/loss": prev_step_loss_acc,
                            "train/epoch": epoch + (step + 1) / len(train_dataloader),
                            "train/grad_norm": grad_norm,
                        },
                        step=global_step,
                    )

                if global_step % 10 == 0:
                    eval_model(step=global_step)

                prev_step_loss = prev_step_loss_acc
                prev_step_loss_acc = 0

            prev_step_loss_acc += loss.item()
            total_loss += loss.item() * gradient_accumulation_steps
            avg_loss = total_loss / (step + 1)

            progress_bar.set_postfix(
                {
                    "avg_loss": f"{avg_loss:.4f}",
                    "step": global_step,
                    "prev_step_loss": prev_step_loss,
                }
            )

    print("Training complete!")

    print("Running final evaluation...")
    final_val_loss = eval_model(step=global_step)

    if use_wandb:
        wandb.summary["final_val_loss"] = final_val_loss
        wandb.summary["total_steps"] = global_step

    if not args.enable_save_ckpt:
        print("Checkpoint saving disabled (--enable-save-ckpt not set).")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
