"""
Unified RL training entrypoint.

Examples:
  uv run run_rl.py --train-mode full --lr 1e-5 --no-wandb
  uv run run_rl.py --train-mode full --optimizer muon --lr 1e-5 --no-wandb
  uv run run_rl.py --train-mode lora --lr 1e-4 --lora-rank 1 --trainable-type all --no-wandb
  uv run run_rl.py --train-mode blocktt --lr 1e-4 --trainable-type all --decomp-mode input_one_block --train-position small --no-wandb
  uv run run_rl.py --train-mode svd --lr 1e-4 --trainable-type all --train-position output --no-wandb
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

import requests
import torch
import wandb
from btt_layer import (
    BTTLayer,
    configure_blocktt_trainability,
    convert_linear_to_btt,
    get_blocktt_target_module_names,
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
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

MODE_DEFAULTS = {
    "full": {
        "lr": 1e-5,
        "wandb_project": "math_grpo_full",
        "micro_batch_size": 4,
        "gradient_accumulation_steps": 64,
    },
    "lora": {
        "lr": 9e-5,
        "wandb_project": "math-grpo",
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Unified RL training script")
    parser.add_argument(
        "--train-mode",
        type=str,
        required=True,
        choices=["full", "lora", "blocktt", "svd"],
        help="Training mode: full, lora, blocktt, or svd",
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
        default="runs",
        help="Base directory for saving runs",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="boxed.prompt",
        help="Path to prompt template file",
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
        help="Save final checkpoint after training (default: disabled)",
    )

    # LoRA-only args
    parser.add_argument(
        "--lora-rank", type=int, default=1, help="LoRA rank (default: 1)"
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        default="http://localhost:8000",
        help="URL for vLLM API server (lora mode only)",
    )

    # BlockTT-only args
    parser.add_argument(
        "--decomp-mode",
        type=str,
        default="input_one_block",
        choices=["input_one_block", "output_one_block"],
        help="BlockTT decomposition mode determining which core is trainable",
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
        "lora": ["--lora-rank", "--vllm-url"],
        "blocktt": [
            "--decomp-mode",
            "--blocktt-rank",
            "--no-train-bias",
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


def load_datasets_and_tokenizer(model_id, prompt_template_path):
    train_dataset = load_dataset("qwedsacf/competition_math", split="train[:7500]")
    val_dataset = load_dataset("qwedsacf/competition_math", split="train[-5000:]")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    with open(prompt_template_path, "r", encoding="utf-8") as f:
        template = f.read().strip()

    def process_data(example):
        with_template = template.replace("{question}", example["problem"])
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": with_template}],
            tokenize=False,
            add_generation_prompt=True,
        )
        answer = remove_boxed(last_boxed_only_string(example["solution"]))
        return {"prompt": prompt, "answer": answer}

    train_dataset = train_dataset.map(process_data)
    val_dataset = val_dataset.map(process_data)
    return train_dataset, val_dataset, tokenizer


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    prompt_t = [tokenizer.encode(p) for p in prompt_strs]
    output_t = [tokenizer.encode(o) for o in output_strs]
    full = []
    max_len = 0

    for i in range(len(prompt_t)):
        row_len = len(prompt_t[i]) + len(output_t[i])
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
        response_mask[
            i, len(prompt_t[i]) - 1 : len(prompt_t[i]) + len(output_t[i]) - 1
        ] = 1

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask.bool(),
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


def create_run_dir(base_dir: str) -> str:
    os.makedirs(base_dir, exist_ok=True)
    i = 1
    while os.path.exists(f"{base_dir}/{i}"):
        i += 1
    run_name = f"{base_dir}/{i}"
    os.makedirs(run_name)
    return run_name


def maybe_init_wandb(args, run_name, mode_info):
    if args.no_wandb:
        return

    if args.wandb_run_name is not None:
        wandb_run_name = args.wandb_run_name
    elif args.train_mode == "full":
        wandb_run_name = f"{args.model_id}_{args.lr:.1e}_full"
    elif args.train_mode == "lora":
        wandb_run_name = f"{args.model_id}_{args.lr:.1e}_r{args.lora_rank}"
    elif args.train_mode == "blocktt":
        wandb_run_name = (
            f"{args.model_id}_{args.lr:.1e}_{args.decomp_mode}_{args.train_position}_{args.trainable_type}"
        )
    else:
        wandb_run_name = (
            f"{args.model_id}_{args.lr:.1e}_{args.train_position}_{args.trainable_type}"
        )

    wandb.init(
        project=args.wandb_project,
        name=wandb_run_name,
        config={
            "train_mode": args.train_mode,
            "model_id": args.model_id,
            "learning_rate": args.lr,
            "optimizer": args.optimizer,
            "weight_decay": args.weight_decay,
            "lr_adam": args.lr_adam,
            "lr_embedding": args.lr_embedding,
            "norm_method": args.norm_method,
            "n_grpo_steps": args.n_grpo_steps,
            "n_prompts_per_step": args.n_prompts_per_step,
            "group_size": args.group_size,
            "epochs_per_step": args.epochs_per_step,
            "micro_batch_size": args.micro_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "seed": args.seed,
            "prompt_template": args.prompt_template,
            "enable_save_ckpt": args.enable_save_ckpt,
            "run_dir": run_name,
            **mode_info,
        },
        dir=run_name,
    )


def is_vllm_http_available(vllm_url: str, timeout_sec: float = 2.0) -> bool:
    try:
        response = requests.get(f"{vllm_url.rstrip('/')}/v1/models", timeout=timeout_sec)
    except requests.RequestException:
        return False
    return response.ok


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


def build_lora_http_generators(args, model, run_name):
    loaded_loras = []

    def generate_http(prompts: list[str], vllm_model_id: str, temperature=0, responses_per_prompt=1):
        api_url = f"{args.vllm_url}/v1/completions"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": vllm_model_id,
            "prompt": prompts,
            "max_tokens": 1024,
            "temperature": temperature,
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
        lora_name = f"{run_name}/step={step}"
        if not os.path.exists(lora_name):
            model.save_pretrained(lora_name)
        return lora_name

    def generate_for_train(prompts: list[str], step: int):
        vllm_model_id = save_lora(step)
        load_lora(vllm_model_id)
        return generate_http(
            prompts,
            vllm_model_id=vllm_model_id,
            temperature=1,
            responses_per_prompt=args.group_size,
        )

    def generate_for_eval(prompts: list[str], step: int):
        vllm_model_id = save_lora(step)
        load_lora(vllm_model_id)
        return generate_http(
            prompts,
            vllm_model_id=vllm_model_id,
            temperature=0,
            responses_per_prompt=1,
        )

    return generate_for_train, generate_for_eval


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
        gpu_memory_utilization=0.4,
        max_model_len=2048,
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

    def generate(prompts: list[str], temperature=0, responses_per_prompt=1):
        weight_tuples = export_lora_merged_weights()
        vllm_internal_model = (
            vllm_model.llm_engine.model_executor.driver_worker.model_runner.model
        )
        vllm_internal_model.load_weights(weight_tuples)

        sampling_params = SamplingParams(
            max_tokens=1024,
            temperature=temperature,
            n=responses_per_prompt,
        )
        outputs = vllm_model.generate(prompts, sampling_params)
        return [o.text for output in outputs for o in output.outputs]

    def generate_for_train(prompts: list[str], _step: int):
        return generate(
            prompts,
            temperature=1,
            responses_per_prompt=args.group_size,
        )

    def generate_for_eval(prompts: list[str], _step: int):
        return generate(prompts, temperature=0, responses_per_prompt=1)

    return generate_for_train, generate_for_eval


def build_local_vllm_generators(args, model):
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

    def generate(prompts: list[str], temperature=0, responses_per_prompt=1):
        if args.train_mode in {"blocktt", "svd"}:
            weight_tuples = export_weights_for_vllm(model)
        else:
            weight_tuples = [(name, param) for name, param in model.named_parameters()]

        vllm_internal_model = (
            vllm_model.llm_engine.model_executor.driver_worker.model_runner.model
        )
        vllm_internal_model.load_weights(weight_tuples)

        sampling_params = SamplingParams(
            max_tokens=1024,
            temperature=temperature,
            n=responses_per_prompt,
        )
        outputs = vllm_model.generate(prompts, sampling_params)
        return [o.text for output in outputs for o in output.outputs]

    def generate_for_train(prompts: list[str], _step: int):
        return generate(
            prompts,
            temperature=1,
            responses_per_prompt=args.group_size,
        )

    def generate_for_eval(prompts: list[str], _step: int):
        return generate(prompts, temperature=0, responses_per_prompt=1)

    return generate_for_train, generate_for_eval


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

    if train_mode == "lora":
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


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    validate_mode_specific_flags(args, argv)
    apply_mode_defaults(args)

    set_seed(args.seed)
    lora_rollout_backend = None
    if args.train_mode == "lora":
        lora_rollout_backend = (
            "http" if is_vllm_http_available(args.vllm_url) else "local_inproc"
        )

    mode_info = {}
    if args.train_mode == "lora":
        lora_target_modules = get_lora_target_modules(args.trainable_type)
        mode_info.update(
            {
                "lora_rank": args.lora_rank,
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
        mode_info.update(
            {
                "blocktt_rank": blocktt_rank,
                "trainable_type": args.trainable_type,
                "decomp_mode": args.decomp_mode,
                "train_position": train_position,
                "s_merged_to": args.s_merged_to,
                "target_modules": blocktt_targets,
                "train_bias": train_bias,
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
    print(f"  Weight decay: {args.weight_decay}")
    if args.optimizer == "muon":
        print(f"  Muon lr_adam: {args.lr_adam}")
        print(f"  Muon lr_embedding: {args.lr_embedding}")
        print(f"  Muon norm_method: {args.norm_method}")
    print(f"  GRPO steps: {args.n_grpo_steps}")
    print(f"  Prompts per step: {args.n_prompts_per_step}")
    print(f"  Group size: {args.group_size}")
    print(f"  Epochs per step: {args.epochs_per_step}")
    print(f"  Micro batch size: {args.micro_batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    if args.train_mode == "lora":
        print(f"  LoRA rank: {args.lora_rank}")
        print(f"  Trainable type: {args.trainable_type}")
        print(f"  Target modules: {mode_info['target_modules']}")
        print(f"  Rollout backend: {lora_rollout_backend}")
        print(f"  vLLM URL: {args.vllm_url}")
    if args.train_mode == "blocktt":
        print(f"  BlockTT rank: {mode_info['blocktt_rank']}")
        print(f"  Trainable type: {args.trainable_type}")
        print(f"  Decomp mode: {args.decomp_mode}")
        print(f"  Train position: {mode_info['train_position']}")
        print(f"  S merged to: {mode_info['s_merged_to']}")
        print(f"  Train BTT bias: {mode_info['train_bias']}")
        print(f"  Target modules: {mode_info['target_modules']}")
    if args.train_mode == "svd":
        print(f"  Trainable type: {args.trainable_type}")
        print(f"  Train position: {args.train_position}")
        print(f"  S merged to: {mode_info['s_merged_to']}")
        print(f"  Target modules: {mode_info['target_modules']}")
    print(f"  Save checkpoint: {'enabled' if args.enable_save_ckpt else 'disabled'}")
    print(f"  W&B logging: {'enabled' if not args.no_wandb else 'disabled'}")
    print()

    run_name = create_run_dir(args.base_dir)
    print(f"Created: {run_name}")

    maybe_init_wandb(args, run_name, mode_info)

    train_dataset, val_dataset, tokenizer = load_datasets_and_tokenizer(
        args.model_id,
        args.prompt_template,
    )

    device = "cuda:0"
    model_kwargs = dict(
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        use_cache=False,
        device_map=device,
    )
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)

    if args.train_mode == "lora":
        from peft import LoraConfig, get_peft_model

        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=32,
            target_modules=mode_info["target_modules"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    elif args.train_mode == "blocktt":
        converted_modules = convert_linear_to_btt(
            model,
            btt_rank=mode_info["blocktt_rank"],
            decomp_mode=args.decomp_mode,
            init_mode="default",
            include_names=mode_info["target_modules"],
            skip_names=("lm_head",),
            lr_act=False,
            s_merged_to=mode_info["s_merged_to"],
            train_position=mode_info["train_position"],
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

    if args.train_mode == "lora":
        if lora_rollout_backend == "http":
            generate_for_train, generate_for_eval = build_lora_http_generators(
                args,
                model,
                run_name,
            )
        else:
            generate_for_train, generate_for_eval = build_lora_local_generators(
                args,
                model,
            )
    else:
        generate_for_train, generate_for_eval = build_local_vllm_generators(args, model)

    optimizer = build_optimizer(args, trainable_params, trainable_named_params)
    if args.optimizer == "muon":
        assert_muon_routing(args.train_mode, trainable_named_params, optimizer)

    def eval_model(step):
        val_prompts = val_dataset[:1000]["prompt"]

        eval_start = time.time()
        outputs = generate_for_eval(val_prompts, step)
        eval_time = time.time() - eval_start

        correct = 0
        for i in range(len(outputs)):
            correct_answer = val_dataset[i]["answer"]
            generated_answer = remove_boxed(last_boxed_only_string(outputs[i]))
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

        sample_indices = random.sample(
            range(0, len(train_dataset)),
            args.n_prompts_per_step,
        )
        batch = train_dataset[sample_indices]

        generation_start = time.time()
        outputs = generate_for_train(batch["prompt"], i)
        generation_time = time.time() - generation_start

        generated_answers = [remove_boxed(last_boxed_only_string(o)) for o in outputs]
        raw_reward = [
            ans is not None and is_equiv(ans, batch["answer"][j // args.group_size])
            for j, ans in enumerate(generated_answers)
        ]
        raw_reward_tensor = torch.tensor(raw_reward, dtype=torch.float).reshape(
            (args.n_prompts_per_step, args.group_size)
        )
        means = raw_reward_tensor.mean(dim=-1).unsqueeze(1)
        advantages = (raw_reward_tensor - means).reshape(
            (args.n_prompts_per_step * args.group_size,)
        )

        train_accuracy = raw_reward_tensor.mean().item()
        reward_std = raw_reward_tensor.std().item()
        reward_max = raw_reward_tensor.max().item()
        reward_min = raw_reward_tensor.min().item()
        advantage_mean = advantages.mean().item()
        advantage_std = advantages.std().item()
        advantage_max = advantages.max().item()
        advantage_min = advantages.min().item()

        prompts_expanded = [x for x in batch["prompt"] for _ in range(args.group_size)]
        data = tokenize_prompt_and_output(prompts_expanded, outputs, tokenizer)
        input_ids = data["input_ids"].to(device)
        labels = data["labels"].to(device)
        response_mask = data["response_mask"].to(device)

        with torch.inference_mode():
            old_logprobs_all = []
            for b in range(len(input_ids) // args.micro_batch_size):
                idx = b * args.micro_batch_size
                end = idx + args.micro_batch_size
                old_logprobs_all.append(
                    get_response_log_probs(model, input_ids[idx:end], labels[idx:end]).detach()
                )
            old_logprobs_all = torch.cat(old_logprobs_all, dim=0)
            assert old_logprobs_all.shape == labels.shape

        epoch_grad_norms = []
        epoch_policy_ratios = []
        epoch_kl_divs = []

        training_start = time.time()
        for epoch in range(args.epochs_per_step):
            for b in tqdm(
                range(len(input_ids) // args.micro_batch_size),
                desc=(
                    f"Step {i + 1}/{args.n_grpo_steps}, "
                    f"Epoch {epoch + 1}/{args.epochs_per_step}"
                ),
            ):
                idx = b * args.micro_batch_size
                end = idx + args.micro_batch_size
                x = input_ids[idx:end]
                y = labels[idx:end]
                mask = response_mask[idx:end]
                micro_batch_adv = advantages[idx:end].unsqueeze(-1).to(device)

                policy_logprobs = get_response_log_probs(model, x, y)
                old_logprobs = old_logprobs_all[idx:end]

                ratio = torch.exp(policy_logprobs - old_logprobs)
                per_token_loss = -ratio * micro_batch_adv

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)) * mask
                    approx_kl_mean = approx_kl.sum() / mask.sum()
                    epoch_kl_divs.append(approx_kl_mean.item())

                    policy_ratio_mean = (ratio * mask).sum() / mask.sum()
                    epoch_policy_ratios.append(policy_ratio_mean.item())

                masked_loss = per_token_loss * mask
                denom = mask.sum(dim=-1).clamp_min(1)
                loss_per_prompt = masked_loss.sum(dim=-1) / denom
                loss = loss_per_prompt.mean() / args.gradient_accumulation_steps

                loss.backward()

                if (b + 1) % args.gradient_accumulation_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    epoch_grad_norms.append(grad_norm.item())
                    optimizer.step()
                    optimizer.zero_grad()

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

    if args.enable_save_ckpt:
        ckpt_dir = os.path.join(run_name, "final_ckpt")
        print(f"Saving model checkpoint to {ckpt_dir}")
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        print(f"Checkpoint saved to {ckpt_dir}")
    else:
        print("Final checkpoint save disabled (--enable-save-ckpt not set).")

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
