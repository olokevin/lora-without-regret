"""
Unified SFT training entrypoint.

Examples:
  CUDA_VISIBLE_DEVICES=0 uv run run_sft.py --train-mode full --no-wandb
  CUDA_VISIBLE_DEVICES=0 uv run run_sft.py --train-mode lora --lora-rank 64 --lora-type all --no-wandb
  CUDA_VISIBLE_DEVICES=0 uv run run_sft.py --train-mode blocktt --blocktt-type all --decomp-mode input_one_block --no-wandb
"""

import argparse
import random
import sys

import numpy as np
import torch
import wandb
from btt_layer import (
    configure_blocktt_trainability,
    convert_linear_to_btt,
    get_blocktt_target_module_names,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODE_DEFAULTS = {
    "full": {
        "lr": 5e-6,
        "wandb_project": "full-finetuning",
        "output_dir": "./finetuned_model",
    },
    "lora": {
        "lr": 1e-4,
        "wandb_project": "lora-finetuning",
        "output_dir": "./lora_model",
    },
    "blocktt": {
        "lr": 1e-4,
        "wandb_project": "blocktt-finetuning",
        "output_dir": "./blocktt_model",
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
        choices=["full", "lora", "blocktt"],
        help="Training mode: full, lora, or blocktt",
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

    # LoRA-only args
    parser.add_argument(
        "--lora-rank", type=int, default=128, help="LoRA rank (default: 128)"
    )
    parser.add_argument(
        "--lora-type",
        type=str,
        default="all",
        choices=["all", "mlp", "attn"],
        help="LoRA target modules: all, mlp, or attn (default: all)",
    )

    # BlockTT-only args
    parser.add_argument(
        "--blocktt-type",
        type=str,
        default="all",
        choices=["all", "mlp", "attn"],
        help="BlockTT target modules: all, mlp, or attn (default: all)",
    )
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
        help="BTT rank; default full for lossless initialization",
    )
    parser.add_argument(
        "--no-train-bias",
        action="store_true",
        help="Freeze BTT biases; by default biases are trainable",
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
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save the model (default depends on train mode)",
    )
    parser.add_argument(
        "--enable-save-ckpt",
        action="store_true",
        help="Save final checkpoint after training (default: disabled)",
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
    if args.output_dir is None:
        args.output_dir = defaults["output_dir"]


def _flag_was_passed(argv, flag_name):
    return any(
        token == flag_name or token.startswith(f"{flag_name}=") for token in argv
    )


def validate_mode_specific_flags(args, argv):
    mode_to_flag_sets = {
        "lora": ["--lora-rank", "--lora-type"],
        "blocktt": [
            "--blocktt-type",
            "--decomp-mode",
            "--blocktt-rank",
            "--no-train-bias",
        ],
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


def get_lora_target_modules(lora_type):
    if lora_type == "all":
        return [
            "gate_proj",
            "up_proj",
            "down_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ]
    if lora_type == "mlp":
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
        target_modules = get_lora_target_modules(args.lora_type)
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
            "lora_type": args.lora_type,
            "target_modules": target_modules,
        }
        mode_info["print_lines"] = [
            f"  LoRA rank: {args.lora_rank}",
            f"  LoRA type: {args.lora_type}",
            f"  Target modules: {target_modules}",
        ]

    else:
        blocktt_rank = resolve_blocktt_rank(args.blocktt_rank)
        target_modules = get_blocktt_target_module_names(args.blocktt_type)
        train_bias = not args.no_train_bias

        converted_modules = convert_linear_to_btt(
            model,
            btt_rank=blocktt_rank,
            decomp_mode=args.decomp_mode,
            init_mode="default",
            include_names=target_modules,
            skip_names=("lm_head",),
            lr_act=False,
        )
        stats = configure_blocktt_trainability(model, train_bias=train_bias)
        if stats["num_btt_layers"] == 0:
            raise ValueError(
                "No layers were converted to BTT; check --blocktt-type selection."
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
            "blocktt_type": args.blocktt_type,
            "decomp_mode": args.decomp_mode,
            "train_bias": train_bias,
            "target_modules": target_modules,
        }
        mode_info["print_lines"] = [
            f"  BlockTT rank: {blocktt_rank}",
            f"  BlockTT type: {args.blocktt_type}",
            f"  Decomp mode: {args.decomp_mode}",
            f"  Train BTT bias: {train_bias}",
            f"  Target modules: {target_modules}",
        ]

    return model, trainable_params, mode_info


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    validate_mode_specific_flags(args, argv)
    apply_mode_defaults(args)

    set_seed(args.seed)

    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    model, trainable_params, mode_info = prepare_model(args)

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
                "optimizer": "AdamW",
                "seed": args.seed,
                "enable_save_ckpt": args.enable_save_ckpt,
                **mode_info["wandb_extra"],
            },
            tags=["sft", args.train_mode],
        )

    print("Training configuration:")
    print(f"  Train mode: {args.train_mode}")
    print(f"  Model ID: {args.model_id}")
    print(f"  Learning rate: {args.lr}")
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
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

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
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
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

    if args.enable_save_ckpt:
        print(f"Saving model to {args.output_dir}")
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Model saved to {args.output_dir}")
    else:
        print("Final checkpoint save disabled (--enable-save-ckpt not set).")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
