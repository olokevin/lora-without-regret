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
import time
from functools import partial

import torch
import torch.nn.functional as F
import wandb
from btt_layer import (
    BLOCKTT_DECOMP_GROUP_TO_MODULES,
    configure_blocktt_trainability,
    convert_linear_to_btt,
    get_blocktt_target_module_names,
    normalize_trainable_blocktt_cores_,
    resolve_blocktt_decomp_modes,
)
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file
from transformers import AutoTokenizer
from svd_layer import (
    configure_svd_trainability,
    convert_linear_to_svd,
    get_svd_target_module_names,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from math_utils import is_equiv, last_boxed_only_string, remove_boxed
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
from run_rl import export_weights_for_vllm


MODE_DEFAULTS = {
    "full": {
        "lr": 5e-6,
        "wandb_project": "kd-full",
    },
    "lora": {
        "lr": 1e-4,
        "wandb_project": "kd-lora",
    },
    "blocktt": {
        "lr": 1e-4,
        "wandb_project": "kd-blocktt",
    },
    "svd": {
        "lr": 1e-4,
        "wandb_project": "kd-svd",
    },
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Knowledge distillation training script")

    # KD-specific args
    parser.add_argument(
        "--kd-loss-type",
        type=str,
        required=True,
        choices=["sft", "kl", "kl_online"],
        help="KD loss type: sft (CE on teacher completions), kl (offline KL on top-K logits), or kl_online (live KL against frozen teacher)",
    )
    parser.add_argument(
        "--teacher-data-dir",
        type=str,
        default="/data/yequan/fura/kd_data/DeepSeek-R1-Distill-Qwen-7B-competition_math",
        help="Path to precomputed teacher data directory",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=256,
        help="Top-K for KL loss; must match generated data (default: 256)",
    )
    parser.add_argument(
        "--teacher-model-id",
        type=str,
        default=None,
        help="Teacher model ID for kl_online mode (e.g. deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)",
    )
    parser.add_argument(
        "--save-steps",
        type=str,
        default="10,30,final",
        help="Comma-separated optimizer steps to save checkpoints (default: 10,30,final)",
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
    parser.add_argument(
        "--blocktt-factorize-by-head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Align attention layer BTT blocks with head structure (default: enabled)",
    )

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
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/data/yequan/fura/kd_runs",
        help="Base directory for saving runs",
    )
    parser.add_argument("--enable-save-ckpt", action="store_true")
    parser.add_argument(
        "--save-grads-steps",
        type=str,
        default=None,
        help=(
            "Comma-separated optimizer steps to save gradients "
            "(e.g. '0,10,30'). Step 0 = before first update. Default: disabled."
        ),
    )
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
    if args.train_mode == "blocktt" and args.train_position is None:
        args.train_position = "small"
    if args.train_mode == "svd" and args.train_position is None:
        args.train_position = "output"
    if args.train_mode in {"blocktt", "svd"} and args.s_merged_to is None:
        if args.train_position == "both":
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
            "--blocktt-factorize-by-head",
            "--no-blocktt-factorize-by-head",
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
        if args.train_position not in {"output", "input", "both"}:
            raise ValueError("--train-position for svd must be one of: output, input, both")

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

    if args.kd_loss_type == "kl_online" and not getattr(args, "teacher_model_id", None):
        raise ValueError("--teacher-model-id is required when --kd-loss-type kl_online")


def parse_save_steps(save_steps_str, total_steps):
    steps = set()
    for part in save_steps_str.split(","):
        part = part.strip()
        if part == "final":
            steps.add(total_steps)
        else:
            steps.add(int(part))
    return steps


def load_teacher_config(teacher_data_dir):
    config_path = os.path.join(teacher_data_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Teacher data config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_completions(teacher_data_dir):
    completions_path = os.path.join(teacher_data_dir, "completions.jsonl")
    completions = []
    with open(completions_path, "r", encoding="utf-8") as f:
        for line in f:
            completions.append(json.loads(line))
    return completions


class KDSftDataset(Dataset):
    """Dataset for SFT-style KD: train on teacher completions."""

    def __init__(self, completions, max_length=2048):
        self.completions = completions
        self.max_length = max_length

    def __len__(self):
        return len(self.completions)

    def __getitem__(self, idx):
        entry = self.completions[idx]
        prompt_text = entry["prompt"]
        completion_text = entry["completion"]

        # Use token_ids directly (already tokenized by teacher's tokenizer,
        # which shares the same base vocab as student)
        prompt_ids = entry.get("prompt_ids")
        if prompt_ids is None:
            # Fallback: the full sequence is prompt + completion token_ids
            # We need to figure out where prompt ends.
            # Since token_ids are completion-only from vLLM, we construct:
            #   input_ids = prompt_token_ids + completion_token_ids
            #   labels = [-100]*len(prompt) + completion_token_ids
            # But we don't have prompt_token_ids stored separately.
            # The token_ids field contains completion tokens only.
            completion_ids = entry["token_ids"]
        else:
            completion_ids = entry["token_ids"]

        # For SFT, input_ids = completion token_ids (teacher's output)
        # Labels = same (next-token prediction on teacher's output)
        # We train the student to reproduce the teacher's output autoregressively
        input_ids = completion_ids[: self.max_length]
        labels = input_ids.copy()
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


class KDKlDataset(Dataset):
    """Dataset for KL-divergence KD: match teacher logits."""

    def __init__(self, completions, teacher_data_dir, top_k, max_length=2048, index_offset=0):
        self.completions = completions
        self.top_k = top_k
        self.max_length = max_length
        self.index_offset = index_offset

        # Load all logit chunks
        logits_dir = os.path.join(teacher_data_dir, "logits")
        chunk_files = sorted(
            f for f in os.listdir(logits_dir) if f.startswith("chunk_") and f.endswith(".safetensors")
        )
        self.all_topk_values = []
        self.all_topk_indices = []
        self.all_seq_lengths = []

        for chunk_file in chunk_files:
            chunk = load_safetensors_file(os.path.join(logits_dir, chunk_file))
            n = chunk["seq_lengths"].shape[0]
            for i in range(n):
                self.all_topk_values.append(chunk["topk_values"][i])
                self.all_topk_indices.append(chunk["topk_indices"][i])
                self.all_seq_lengths.append(chunk["seq_lengths"][i].item())

    def __len__(self):
        return len(self.completions)

    def __getitem__(self, idx):
        entry = self.completions[idx]
        completion_ids = entry["token_ids"]
        # Use index_offset to align with original logit indices after train/val split
        logit_idx = idx + self.index_offset
        seq_len = self.all_seq_lengths[logit_idx]

        input_ids = completion_ids[: self.max_length]
        actual_len = len(input_ids)
        attention_mask = [1] * actual_len

        # response_mask: all tokens are response tokens (completion only)
        response_mask = [1] * actual_len

        topk_values = self.all_topk_values[logit_idx][:actual_len]
        topk_indices = self.all_topk_indices[logit_idx][:actual_len]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "teacher_topk_values": topk_values,
            "teacher_topk_indices": topk_indices,
        }


class KDOnlineDataset(Dataset):
    """Dataset for online KL KD: provides token_ids for live teacher inference."""

    def __init__(self, completions, max_length=2048):
        self.completions = completions
        self.max_length = max_length

    def __len__(self):
        return len(self.completions)

    def __getitem__(self, idx):
        entry = self.completions[idx]
        input_ids = entry["token_ids"][: self.max_length]
        attention_mask = [1] * len(input_ids)
        response_mask = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
        }


def compute_kl_loss(student_logits, teacher_topk_values, teacher_topk_indices, response_mask):
    """
    Compute token-wise KL(student || teacher) over teacher's top-K positions.

    Args:
        student_logits: [batch, seq_len, vocab_size]
        teacher_topk_values: [batch, seq_len, K] - teacher log-probs from vLLM
        teacher_topk_indices: [batch, seq_len, K] - token indices for top-K
        response_mask: [batch, seq_len] - 1 for response tokens, 0 for padding

    Returns:
        Scalar KL loss averaged over response tokens.
    """
    batch, seq_len, K = teacher_topk_values.shape
    vocab_size = student_logits.shape[2]

    # Teacher may have larger vocab than student (e.g. DeepSeek 152064 vs Qwen 151936).
    # Both share identical token mappings for real tokens; the difference is only in
    # padding slots beyond ~151650. Clamp indices for safe gather, then zero out KL
    # contributions from out-of-bound entries.
    oob_mask = teacher_topk_indices >= vocab_size  # [batch, seq_len, K]
    if oob_mask.any():
        teacher_topk_indices = teacher_topk_indices.clone()
        teacher_topk_indices[oob_mask] = 0  # safe index for gather

    # Gather student logits at teacher's top-K positions
    student_topk_logits = torch.gather(student_logits, dim=2, index=teacher_topk_indices.long())

    # Student: log-softmax over K positions (from raw logits)
    student_log_probs = F.log_softmax(student_topk_logits, dim=-1)

    # Teacher: values are already log-probs from vLLM over full vocab.
    # Re-normalize over the top-K subset to get a valid distribution over K positions.
    teacher_log_probs = F.log_softmax(teacher_topk_values.float(), dim=-1)

    # KL(student || teacher) = sum_k student_prob * (log_student_prob - log_teacher_prob)
    # F.kl_div(input=log_teacher, target=log_student, log_target=True) gives KL(student || teacher)
    kl_per_k = F.kl_div(
        teacher_log_probs,
        student_log_probs,
        log_target=True,
        reduction="none",
    )  # [batch, seq_len, K]

    # Zero out contributions from out-of-vocab teacher entries
    if oob_mask.any():
        kl_per_k = kl_per_k.masked_fill(oob_mask, 0.0)

    kl_per_token = kl_per_k.sum(dim=-1)  # [batch, seq_len]

    # Mask and average
    masked_kl = kl_per_token * response_mask
    num_tokens = response_mask.sum().clamp(min=1)
    return masked_kl.sum() / num_tokens


def build_vllm_generator(args, model):
    """Build a greedy generation function backed by a local vLLM engine."""
    os.environ["VLLM_USE_V1"] = "0"
    from vllm import LLM, SamplingParams

    vllm_model = LLM(
        model=args.model_id,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.4,
        max_model_len=2048,
        max_num_batched_tokens=4096,
        logprobs_mode="processed_logprobs",
    )
    sampling_params = SamplingParams(max_tokens=1024, temperature=0, n=1)

    def generate(prompts: list[str]) -> list[str]:
        if args.train_mode in {"blocktt", "svd"}:
            weight_tuples = export_weights_for_vllm(model)
        else:
            weight_tuples = list(model.named_parameters())
        vllm_internal_model = (
            vllm_model.llm_engine.model_executor.driver_worker.model_runner.model
        )
        vllm_internal_model.load_weights(weight_tuples)
        outputs = vllm_model.generate(prompts, sampling_params)
        return [o.text for output in outputs for o in output.outputs]

    return generate


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


def save_kd_checkpoint(model, run_dir, step):
    """Save model checkpoint in step={N}/model.safetensors format."""
    ckpt_dir = os.path.join(run_dir, f"step={step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    state_dict = {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
    }
    save_safetensors_file(state_dict, os.path.join(ckpt_dir, "model.safetensors"))
    print(f"Saved checkpoint to {ckpt_dir}")


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


def build_kd_sft_collate_fn(pad_token_id):
    """Collate for SFT KD: pad input_ids, labels, attention_mask."""
    def collate_fn(batch):
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids = []
        labels = []
        attention_mask = []

        for item in batch:
            padding_len = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_token_id] * padding_len)
            labels.append(item["labels"] + [-100] * padding_len)
            attention_mask.append(item["attention_mask"] + [0] * padding_len)

        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attention_mask),
        }
    return collate_fn


def build_kd_kl_collate_fn(pad_token_id, top_k):
    """Collate for KL KD: pad input_ids, attention_mask, response_mask, teacher logits."""
    def collate_fn(batch):
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids = []
        attention_mask = []
        response_mask = []
        teacher_topk_values = []
        teacher_topk_indices = []

        for item in batch:
            actual_len = len(item["input_ids"])
            padding_len = max_len - actual_len
            input_ids.append(item["input_ids"] + [pad_token_id] * padding_len)
            attention_mask.append(item["attention_mask"] + [0] * padding_len)
            response_mask.append(item["response_mask"] + [0] * padding_len)

            # Pad teacher logits
            tv = item["teacher_topk_values"]  # [actual_len, K]
            ti = item["teacher_topk_indices"]  # [actual_len, K]
            if padding_len > 0:
                tv = torch.cat([tv, torch.zeros(padding_len, top_k, dtype=tv.dtype)])
                ti = torch.cat([ti, torch.zeros(padding_len, top_k, dtype=ti.dtype)])
            teacher_topk_values.append(tv)
            teacher_topk_indices.append(ti)

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "response_mask": torch.stack(response_mask) if isinstance(response_mask[0], torch.Tensor) else torch.tensor(response_mask, dtype=torch.float32),
            "teacher_topk_values": torch.stack(teacher_topk_values),
            "teacher_topk_indices": torch.stack(teacher_topk_indices),
        }
    return collate_fn


def build_kd_online_collate_fn(pad_token_id):
    """Collate for online KL KD: pad input_ids, attention_mask, response_mask."""
    def collate_fn(batch):
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids = []
        attention_mask = []
        response_mask = []
        for item in batch:
            padding_len = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_token_id] * padding_len)
            attention_mask.append(item["attention_mask"] + [0] * padding_len)
            response_mask.append(item["response_mask"] + [0] * padding_len)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "response_mask": torch.tensor(response_mask, dtype=torch.float32),
        }
    return collate_fn


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    validate_mode_specific_flags(args, argv)
    apply_mode_defaults(args)

    set_seed(args.seed)

    # Load teacher data
    teacher_config = load_teacher_config(args.teacher_data_dir)
    if args.kd_loss_type == "kl" and args.top_k > teacher_config["top_k"]:
        raise ValueError(
            f"--top-k ({args.top_k}) exceeds teacher data top_k ({teacher_config['top_k']})"
        )

    completions = load_completions(args.teacher_data_dir)
    num_total = len(completions)
    # Use last 20% for validation, rest for training
    val_size = max(1, num_total // 5)
    train_completions = completions[:-val_size]
    val_completions = completions[-val_size:]
    print(f"Loaded {num_total} completions: {len(train_completions)} train, {len(val_completions)} val")

    # Prepare model
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    model, trainable_params, trainable_named_params, mode_info = prepare_model(args)
    validate_trainable_params(trainable_params)
    vllm_generate = build_vllm_generator(args, model)

    run_name = compute_run_name(args, mode_info)
    run_dir = create_run_dir(args.base_dir, args.train_mode, run_name)
    print(f"Created: {run_dir}")

    # Prepare tokenizer for padding
    from transformers import AutoTokenizer as _AT
    tokenizer = _AT.from_pretrained(args.student_model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Build datasets
    if args.kd_loss_type == "sft":
        train_dataset = KDSftDataset(train_completions)
        val_dataset = KDSftDataset(val_completions)
        collate_fn = build_kd_sft_collate_fn(tokenizer.pad_token_id)
    else:
        train_dataset = KDKlDataset(train_completions, args.teacher_data_dir, args.top_k)
        val_dataset = KDKlDataset(val_completions, args.teacher_data_dir, args.top_k, index_offset=len(train_completions))
        collate_fn = build_kd_kl_collate_fn(tokenizer.pad_token_id, args.top_k)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Unified CE val dataloader (for comparable val/loss across sft and kl modes)
    if args.kd_loss_type == "sft":
        ce_val_dataloader = val_dataloader
    else:
        ce_val_dataset = KDSftDataset(val_completions)
        ce_val_dataloader = DataLoader(
            ce_val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=build_kd_sft_collate_fn(tokenizer.pad_token_id),
        )

    gradient_accumulation_steps = args.gradient_accumulation_steps
    device = model.device
    num_training_steps = compute_num_training_steps(
        num_batches=len(train_dataloader),
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    warmup_steps = resolve_warmup_steps(args.warmup_ratio, num_training_steps)

    save_steps = parse_save_steps(args.save_steps, num_training_steps)
    save_grads_steps = parse_save_grads_steps(args.save_grads_steps)

    optimizer = build_optimizer(args, trainable_params, trainable_named_params)
    scheduler = build_lr_scheduler(args, optimizer, num_training_steps)

    # Wandb
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "train_mode": args.train_mode,
                "kd_loss_type": args.kd_loss_type,
                "student_model_id": args.student_model_id,
                "teacher_data_dir": args.teacher_data_dir,
                "top_k": args.top_k,
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "effective_batch_size": effective_batch_size,
                "num_epochs": args.num_epochs,
                "optimizer": args.optimizer,
                "lr_scheduler": args.lr_scheduler,
                "warmup_ratio": args.warmup_ratio,
                "seed": args.seed,
                **mode_info["wandb_extra"],
            },
            tags=["kd", args.train_mode, args.kd_loss_type],
        )

    shared_vocab_size = teacher_config.get("shared_vocab_size")

    print("Training configuration:")
    print(f"  KD loss type: {args.kd_loss_type}")
    print(f"  Train mode: {args.train_mode}")
    print(f"  Student model: {args.student_model_id}")
    print(f"  Teacher data: {args.teacher_data_dir}")
    print(f"  Top-K: {args.top_k}")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Learning rate: {args.lr}")
    print(f"  LR scheduler: {args.lr_scheduler}")
    for line in mode_info["print_lines"]:
        print(line)
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {gradient_accumulation_steps}")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"  Optimizer update steps: {num_training_steps}")
    print(f"  Save steps: {sorted(save_steps)}")
    print()

    # Build val prompts for accuracy evaluation
    prompt_template_path = teacher_config.get("prompt_template", "boxed.prompt")
    if os.path.exists(prompt_template_path):
        with open(prompt_template_path, "r", encoding="utf-8") as f:
            _prompt_template = f.read().strip()
    else:
        _prompt_template = "{question}"

    val_prompts_for_gen = []
    val_ground_truths = []
    has_chat_template = hasattr(tokenizer, "apply_chat_template")
    for c in val_completions:
        content = _prompt_template.replace("{question}", c["question"])
        if has_chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = content
        val_prompts_for_gen.append(prompt)
        val_ground_truths.append(c["ground_truth"])

    # Eval function
    def eval_model(step=None, compute_accuracy=False):
        model.eval()

        # Part 1: CE loss (unified for both sft and kl)
        total_ce_loss = 0
        with torch.no_grad():
            for batch in ce_val_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                total_ce_loss += model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                ).loss.item()
        val_loss = total_ce_loss / max(1, len(ce_val_dataloader))
        print(f"val/loss (CE): {val_loss:.4f} at step {step}")

        log_dict = {"val/loss": val_loss}

        # Part 2: Accuracy via generation (only when requested — generation is slow)
        if compute_accuracy and has_chat_template:
            eval_start = time.time()
            outputs = vllm_generate(val_prompts_for_gen)
            eval_time = time.time() - eval_start

            correct = 0
            for i, output in enumerate(outputs):
                gen_ans = remove_boxed(last_boxed_only_string(output))
                if gen_ans is not None and is_equiv(gen_ans, val_ground_truths[i]):
                    correct += 1
            accuracy = correct / len(outputs) if outputs else 0
            print(f"val/accuracy: {correct}/{len(outputs)} ({accuracy:.2%}) at step {step}")
            log_dict.update({
                "val/accuracy": accuracy,
                "val/correct": correct,
                "val/total": len(outputs),
                "val/eval_time_seconds": eval_time,
            })

        if use_wandb and step is not None:
            wandb.log(log_dict, step=step)

        model.train()
        return val_loss

    # Training loop
    print("Starting initial evaluation...")
    eval_model(compute_accuracy=False)
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

            if args.kd_loss_type == "sft":
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
            else:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                logits = outputs.logits
                if shared_vocab_size is not None:
                    logits = logits[:, :, :shared_vocab_size]
                loss = compute_kl_loss(
                    logits,
                    batch["teacher_topk_values"],
                    batch["teacher_topk_indices"],
                    batch["response_mask"],
                )

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

                # Checkpoint saving (with accuracy eval)
                if args.enable_save_ckpt and global_step in save_steps:
                    eval_model(step=global_step, compute_accuracy=False)
                    save_kd_checkpoint(model, run_dir, global_step)

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
    final_val_loss = eval_model(step=global_step, compute_accuracy=True)

    if use_wandb:
        wandb.summary["final_val_loss"] = final_val_loss
        wandb.summary["total_steps"] = global_step

    # Save final checkpoint if "final" was in save_steps and not already saved
    if args.enable_save_ckpt and num_training_steps in save_steps and global_step not in save_steps:
        save_kd_checkpoint(model, run_dir, global_step)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
