# LIFT BlockTT Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the LIFT codebase runnable under the main `.venv/` and add a `finetune_blocktt.py` training script using our BTT decomposition.

**Architecture:** Patch two files to remove deepspeed imports (the only compatibility blocker). Create `finetune_blocktt.py` following LIFT's `finetune_sft.py` structure — loads HF model, applies `convert_linear_to_btt()` from `btt_layer.py`, trains TT cores with AdamW, materializes back to dense for saving. Add bash scripts and slurm configs matching LIFT's existing patterns.

**Tech Stack:** PyTorch 2.8+, transformers 4.57+, accelerate, peft (for LoRA baseline only), `btt_layer.py` from the main repo root.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `ref/LIFT/src/utils/utils.py` | Modify | Replace deepspeed seeding with `torch.cuda` |
| `ref/LIFT/src/finetune_lora.py` | Modify | Remove deepspeed import and arg parsing |
| `ref/LIFT/src/finetune_blocktt.py` | Create | BlockTT training script |
| `ref/LIFT/bash_scripts/finetune_math_blocktt.sh` | Create | Math training launcher |
| `ref/LIFT/bash_scripts/finetune_commonsense_blocktt.sh` | Create | Commonsense training launcher |
| `ref/LIFT/bash_scripts/slurm_config_blocktt_math.txt` | Create | Hyperparams for math experiments |
| `ref/LIFT/bash_scripts/slurm_config_blocktt_commonsense.txt` | Create | Hyperparams for commonsense experiments |

---

### Task 1: Patch deepspeed imports in LIFT

**Files:**
- Modify: `ref/LIFT/src/utils/utils.py:9,44`
- Modify: `ref/LIFT/src/finetune_lora.py:16,307`

- [ ] **Step 1: Patch `src/utils/utils.py`**

Remove the deepspeed import at line 9 and replace the `get_accelerator()` call at line 44:

```python
# Line 9: DELETE this line
from deepspeed.accelerator import get_accelerator

# Line 38-44: Replace set_random_seed function
def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
```

- [ ] **Step 2: Patch `src/finetune_lora.py`**

Remove the deepspeed import at line 16 and the arg parser line at 307:

```python
# Line 16: DELETE this line
import deepspeed

# Line 307: DELETE this line
    parser = deepspeed.add_config_arguments(parser)
```

- [ ] **Step 3: Verify all LIFT scripts import cleanly**

Run:
```bash
cd /home/yequan/Project/lora/lora-without-regret
.venv/bin/python -c "
import sys, os
sys.path.insert(0, os.path.join('ref/LIFT/src'))
sys.path.insert(0, os.path.join('ref/LIFT'))
from utils.utils import print_rank_0, set_random_seed, int_or_float
from utils.model_utils import load_hf_tokenizer, save_hf_format, get_optimizer_grouped_parameters
from utils.data_utils import SupervisedDataset, DataCollatorForSupervisedDataset
print('All LIFT imports OK')
"
```
Expected: `All LIFT imports OK`

- [ ] **Step 4: Commit**

```bash
git add ref/LIFT/src/utils/utils.py ref/LIFT/src/finetune_lora.py
git commit -m "fix: remove deepspeed imports from LIFT for venv compat"
```

---

### Task 2: Create `finetune_blocktt.py`

**Files:**
- Create: `ref/LIFT/src/finetune_blocktt.py`

This is the main deliverable. It follows `finetune_sft.py`'s structure (accelerate-based, single-GPU, gradient checkpointing, eval/best-model tracking) but replaces the model preparation and optimizer with BlockTT-specific logic.

- [ ] **Step 1: Create `ref/LIFT/src/finetune_blocktt.py`**

```python
import sys
import os

# Add LIFT parent to path (same pattern as other LIFT scripts)
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
# Add repo root to path for btt_layer.py
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir))
)

import copy
import torch
import json
import random
import math
import argparse
from tqdm.auto import tqdm

from torch.utils.data import DataLoader
import torch.nn as nn

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    SchedulerType,
    get_scheduler,
)

from utils.utils import (
    print_rank_0,
    get_all_reduce_mean,
    int_or_float,
)

from accelerate import Accelerator
from accelerate.utils import set_seed

from utils.model_utils import (
    load_hf_tokenizer,
    save_hf_format,
    make_model_gradient_checkpointing_compatible,
)

from utils.data_utils import SupervisedDataset, DataCollatorForSupervisedDataset

from btt_layer import (
    BTTLayer,
    convert_linear_to_btt,
    configure_blocktt_trainability,
    get_blocktt_target_module_names,
    normalize_trainable_blocktt_cores_,
    resolve_blocktt_decomp_modes,
    format_blocktt_decomp_mode,
)


def resolve_blocktt_rank(rank_arg):
    """Parse --blocktt_rank: 'full' or a positive integer."""
    if rank_arg == "full":
        return "full"
    try:
        rank = int(rank_arg)
    except ValueError as exc:
        raise ValueError("--blocktt_rank must be 'full' or a positive integer") from exc
    if rank <= 0:
        raise ValueError("--blocktt_rank must be > 0")
    return rank


def materialize_btt_to_linear(model):
    """Replace all BTTLayer modules with nn.Linear containing materialized dense weights.

    This makes the model saveable/loadable as a standard HF checkpoint.
    """
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, BTTLayer):
            replacements.append((name, module))

    for name, btt_module in replacements:
        dense_weight = btt_module.materialize_dense_weight()
        linear = nn.Linear(
            btt_module.in_features,
            btt_module.out_features,
            bias=btt_module.bias is not None,
            device=dense_weight.device,
            dtype=dense_weight.dtype,
        )
        linear.weight.data.copy_(dense_weight)
        if btt_module.bias is not None:
            linear.bias.data.copy_(btt_module.bias.data)

        # Navigate to parent and replace the child
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], linear)

    print(f"Materialized {len(replacements)} BTTLayer modules to nn.Linear")
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="BlockTT Fine-Tuning (LIFT benchmark)")
    parser.add_argument(
        "--data_path",
        nargs="*",
        default=["./LLM-Adapters/ft-training_set/commonsense_170k.json"],
        help="Path to the training dataset (json).",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--per_device_train_batch_size", type=int, default=16,
        help="Batch size (per device) for training.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size", type=int, default=16,
        help="Batch size (per device) for evaluation.",
    )
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--val_set_size", type=int, default=100,
        help="Size of the validation set. If 0, no validation set is used.")
    parser.add_argument("--load_last_model", action="store_true",
        help="Skip best-model tracking, save only the last model.")
    parser.add_argument("--eval_step", type=int, default=80)
    parser.add_argument("--eval_delay", type=int_or_float, default=0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--lr_scheduler_type", type=SchedulerType, default="linear",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial",
                 "constant", "constant_with_warmup"],
    )
    parser.add_argument("--num_warmup_steps", type=float, default=0.03)
    parser.add_argument(
        "--mixed_precision", type=str, default="bf16",
        choices=["fp16", "bf16", "fp32"],
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument(
        "--instruction_type", type=str, choices=["single", "multi"], default="single",
    )
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument(
        "--use_flash_attn", type=str, default="False",
    )

    # BlockTT-specific arguments
    parser.add_argument("--trainable_type", type=str, default="all",
        choices=["all", "mlp", "attn"],
        help="Which modules to convert to BTT: all, mlp, attn")
    parser.add_argument("--decomp_mode", type=str, default="input_one_block",
        help="BTT decomposition mode: input_one_block, output_one_block, or dict literal")
    parser.add_argument("--blocktt_rank", type=str, default="full",
        help="BTT rank: 'full' for lossless or a positive integer")
    parser.add_argument("--train_position", type=str, default="small",
        choices=["small", "large", "both"],
        help="Which TT core to train: small, large, both")
    parser.add_argument("--s_merged_to", type=str, default="frozen",
        help="Where to merge singular values during SVD init")
    parser.add_argument("--blocktt_normalize_after_update", action="store_true",
        help="Normalize trainable BTT cores after each optimizer step")
    parser.add_argument("--blocktt_factorize_by_head", action="store_true", default=True,
        help="Align attention BTT blocks with head structure")
    parser.add_argument("--no_blocktt_factorize_by_head", action="store_false",
        dest="blocktt_factorize_by_head")
    parser.add_argument("--no_train_bias", action="store_true",
        help="Freeze BTT biases")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )

    set_seed(args.seed)
    args.global_rank = 1

    # Load tokenizer
    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    tokenizer.model_max_length = args.max_seq_len

    # Load model
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model_kwargs = {"torch_dtype": torch.bfloat16}
    if args.use_flash_attn == "True":
        model_kwargs["use_flash_attention_2"] = True
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        **model_kwargs,
    )
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))

    # --- BlockTT conversion ---
    blocktt_rank = resolve_blocktt_rank(args.blocktt_rank)
    target_modules = get_blocktt_target_module_names(args.trainable_type)
    train_bias = not args.no_train_bias

    # Resolve decomp mode (may be scalar or per-module dict)
    decomp_mode, module_decomp_modes = resolve_blocktt_decomp_modes(
        args.decomp_mode,
        include_names=target_modules,
        default_mode="input_one_block",
    )

    converted_modules = convert_linear_to_btt(
        model,
        btt_rank=blocktt_rank,
        decomp_mode=module_decomp_modes if module_decomp_modes is not None else decomp_mode,
        init_mode="default",
        include_names=target_modules,
        skip_names=("lm_head",),
        lr_act=False,
        s_merged_to=args.s_merged_to,
        train_position=args.train_position,
        factorize_by_head=args.blocktt_factorize_by_head,
        model_config=model.config,
    )
    stats = configure_blocktt_trainability(
        model,
        train_bias=train_bias,
        train_position=args.train_position,
    )
    if stats["num_btt_layers"] == 0:
        raise ValueError("No layers were converted to BTT; check --trainable_type.")

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

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"param {name} is trainable")

    if args.gradient_checkpointing:
        model = make_model_gradient_checkpointing_compatible(model)
        model.gradient_checkpointing_enable()

    # --- Dataset ---
    if len(args.data_path) == 1 and ".json" in args.data_path[0]:
        train_dataset = SupervisedDataset(
            data_path=args.data_path[0],
            tokenizer=tokenizer,
            instruction_type=args.instruction_type,
            args=args,
        )
        if args.val_set_size > 0:
            train_dataset, eval_dataset = torch.utils.data.random_split(
                train_dataset,
                [len(train_dataset) - args.val_set_size, args.val_set_size],
            )
    else:
        raise ValueError("Only json format is supported for now.")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=DataCollatorForSupervisedDataset(tokenizer=tokenizer),
    )
    if args.val_set_size > 0:
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=DataCollatorForSupervisedDataset(tokenizer=tokenizer),
        )

    # --- Optimizer: standard AdamW on trainable TT cores ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    if args.num_warmup_steps < 1:
        args.num_warmup_steps = int(args.num_warmup_steps * max_train_steps)
    else:
        args.num_warmup_steps = int(args.num_warmup_steps)

    print(f"max trainable steps: {max_train_steps}, warmup steps: {args.num_warmup_steps}")
    total_batch_size = (
        args.per_device_train_batch_size * args.gradient_accumulation_steps
    )

    print("***** Running BlockTT training *****")
    print(f"  Num examples = {len(train_dataloader)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    print(f"  Total train batch size (w. accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {max_train_steps}")

    progress_bar = tqdm(
        range(max_train_steps), disable=not accelerator.is_local_main_process
    )
    args.completed_steps = 0

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    # Prepare with accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    if args.val_set_size > 0:
        eval_dataloader = accelerator.prepare(eval_dataloader)

    best_model = None

    def train_epoch(epoch):
        nonlocal best_model, best_eval_loss
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                total_loss += loss.detach().float()

            if accelerator.sync_gradients:
                optimizer.step()
                if args.blocktt_normalize_after_update:
                    unwrapped = accelerator.unwrap_model(model)
                    normalize_trainable_blocktt_cores_(unwrapped)
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                args.completed_steps += 1

                if (
                    args.logging_steps
                    and args.completed_steps % args.logging_steps == 0
                ):
                    divisor = args.gradient_accumulation_steps * args.logging_steps
                    avg_loss = (
                        accelerator.gather(total_loss).mean().item() / divisor
                    )
                    print(
                        f"  Step: {args.completed_steps}, "
                        f"LR: {lr_scheduler.get_last_lr()[0]:.8f}, "
                        f"Loss: {avg_loss:.6f}"
                    )
                    accelerator.log(
                        {
                            "learning_rate": lr_scheduler.get_last_lr()[0],
                            "train_loss": avg_loss,
                        },
                        step=args.completed_steps,
                    )
                    total_loss = 0

                if (
                    args.completed_steps % args.eval_step == 0
                    and args.val_set_size > 0
                    and not args.load_last_model
                ):
                    perplexity, eval_loss = evaluate(model)
                    accelerator.print(
                        f"Epoch {epoch+1} Step {args.completed_steps}: "
                        f"Eval perplexity = {perplexity:.4f}, Eval loss = {eval_loss:.4f}"
                    )
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        if accelerator.is_main_process and args.output_dir:
                            accelerator.wait_for_everyone()
                            unwrapped_model = accelerator.unwrap_model(model)
                            best_model = copy.deepcopy(unwrapped_model).to("cpu")
                            print("New best model")

        return total_loss / len(train_dataloader)

    def evaluate(model):
        model.eval()
        losses = 0
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            losses += loss.float()
        losses = losses / (step + 1)
        try:
            losses = get_all_reduce_mean(losses)
        except Exception:
            pass
        try:
            perplexity = torch.exp(losses).item()
        except OverflowError:
            perplexity = float("inf")
        model.train()
        return perplexity, losses.item()

    # --- Training loop ---
    best_eval_loss = float("inf")
    for epoch in range(args.num_train_epochs):
        train_loss = train_epoch(epoch)
        accelerator.print(f"Epoch {epoch+1}: Average loss = {train_loss:.4f}")

    # Save final model if no validation
    if args.val_set_size == 0 and accelerator.is_main_process and args.output_dir:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        materialize_btt_to_linear(unwrapped_model)
        save_hf_format(unwrapped_model, tokenizer, args)

    if args.output_dir is not None:
        # Evaluate last model
        if args.val_set_size > 0 and not args.load_last_model:
            ppl, val_loss = evaluate(model)
            print_rank_0(
                f"Validation perplexity: {ppl}, Validation loss: {val_loss}",
                args.global_rank,
            )
            if val_loss < best_eval_loss:
                best_eval_loss = val_loss
                if args.global_rank == 0:
                    best_model = copy.deepcopy(model.module).to("cpu")

        model = best_model if best_model is not None else model
        materialize_btt_to_linear(model)
        save_hf_format(model, tokenizer, args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script parses args without error**

Run:
```bash
cd /home/yequan/Project/lora/lora-without-regret/ref/LIFT
../../.venv/bin/python src/finetune_blocktt.py --help
```
Expected: prints usage with all blocktt-specific args listed.

- [ ] **Step 3: Commit**

```bash
git add ref/LIFT/src/finetune_blocktt.py
git commit -m "feat: add finetune_blocktt.py for BTT training in LIFT benchmark"
```

---

### Task 3: Create bash scripts and slurm configs for math experiments

**Files:**
- Create: `ref/LIFT/bash_scripts/finetune_math_blocktt.sh`
- Create: `ref/LIFT/bash_scripts/slurm_config_blocktt_math.txt`

- [ ] **Step 1: Create `slurm_config_blocktt_math.txt`**

```
MODEL                             decomp_mode        train_position  blocktt_rank  s_merged_to  trainable_type  lr    seed
meta-llama/Llama-3.2-1B           input_one_block    small           full          frozen       all             2e-4  43
meta-llama/Llama-3.2-3B           input_one_block    small           full          frozen       all             2e-4  43
meta-llama/Llama-2-7b-hf          input_one_block    small           full          frozen       all             2e-4  43
meta-llama/Meta-Llama-3-8B        input_one_block    small           full          frozen       all             1e-4  43
```

- [ ] **Step 2: Create `finetune_math_blocktt.sh`**

```bash
#!/bin/bash

pwd
hostname
date
echo starting job...
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export LIBRARY_PATH="/usr/local/cuda/lib64:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export HF_HOME=/your/path/to/huggingface/cache      # MODIFY THIS LINE

SRC_DIR=/enter/your/path/to/the/repo      # MODIFY THIS LINE
DATA_DIR=/enter/your/data/dir      # MODIFY THIS LINE
OUTPUT_SRC_DIR=/enter/your/output/dir      # MODIFY THIS LINE

SLURM_ARRAY_TASK_ID=$1
cfg=$(sed -n "$SLURM_ARRAY_TASK_ID"p ${SRC_DIR}/bash_scripts/slurm_config_blocktt_math.txt)
MODEL=$(echo $cfg | cut -f 1 -d ' ')
decomp_mode=$(echo $cfg | cut -f 2 -d ' ')
train_position=$(echo $cfg | cut -f 3 -d ' ')
blocktt_rank=$(echo $cfg | cut -f 4 -d ' ')
s_merged_to=$(echo $cfg | cut -f 5 -d ' ')
trainable_type=$(echo $cfg | cut -f 6 -d ' ')
lr=$(echo $cfg | cut -f 7 -d ' ')
seed=$(echo $cfg | cut -f 8 -d ' ')

echo $MODEL

OUTPUT=${OUTPUT_SRC_DIR}/${MODEL}/blocktt/math/decomp_${decomp_mode}_pos_${train_position}_rank_${blocktt_rank}_smerge_${s_merged_to}_type_${trainable_type}/lr_${lr}/seed_${seed}

mkdir -p $OUTPUT

cd ${SRC_DIR}

accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    --mixed_precision="bf16" \
    src/finetune_blocktt.py \
    --model_name_or_path ${MODEL} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 16 \
    --logging_steps 10 \
    --max_seq_len 2048 \
    --learning_rate ${lr} \
    --weight_decay 0. \
    --num_train_epochs 3 \
    --mixed_precision bf16 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type linear \
    --num_warmup_steps 0.03 \
    --seed ${seed} \
    --gradient_checkpointing \
    --instruction_type single \
    --decomp_mode ${decomp_mode} \
    --train_position ${train_position} \
    --blocktt_rank ${blocktt_rank} \
    --s_merged_to ${s_merged_to} \
    --trainable_type ${trainable_type} \
    --load_last_model \
    --data_path ${DATA_DIR}/LLM-Adapters/ft-training_set/math_10k.json \
    --output_dir $OUTPUT 2> >(tee $OUTPUT/err.log >&2) | tee $OUTPUT/training.log

bash ./bash_scripts/eval_math.sh $OUTPUT
```

- [ ] **Step 3: Commit**

```bash
git add ref/LIFT/bash_scripts/finetune_math_blocktt.sh ref/LIFT/bash_scripts/slurm_config_blocktt_math.txt
git commit -m "feat: add math experiment scripts for blocktt in LIFT"
```

---

### Task 4: Create bash scripts and slurm configs for commonsense experiments

**Files:**
- Create: `ref/LIFT/bash_scripts/finetune_commonsense_blocktt.sh`
- Create: `ref/LIFT/bash_scripts/slurm_config_blocktt_commonsense.txt`

- [ ] **Step 1: Create `slurm_config_blocktt_commonsense.txt`**

```
MODEL                               decomp_mode        train_position  blocktt_rank  s_merged_to  trainable_type  lr    seed
meta-llama/Llama-2-7b-hf            input_one_block    small           full          frozen       all             1e-4  43
meta-llama/Meta-Llama-3-8B          input_one_block    small           full          frozen       all             5e-5  43
```

- [ ] **Step 2: Create `finetune_commonsense_blocktt.sh`**

```bash
#!/bin/bash

pwd
hostname
date
echo starting job...
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export LIBRARY_PATH="/usr/local/cuda/lib64:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export HF_HOME=/your/path/to/huggingface/cache      # MODIFY THIS LINE

SRC_DIR=/enter/your/path/to/the/repo      # MODIFY THIS LINE
DATA_DIR=/enter/your/data/dir      # MODIFY THIS LINE
OUTPUT_SRC_DIR=/enter/your/output/dir      # MODIFY THIS LINE

SLURM_ARRAY_TASK_ID=$1
cfg=$(sed -n "$SLURM_ARRAY_TASK_ID"p ${SRC_DIR}/bash_scripts/slurm_config_blocktt_commonsense.txt)
MODEL=$(echo $cfg | cut -f 1 -d ' ')
decomp_mode=$(echo $cfg | cut -f 2 -d ' ')
train_position=$(echo $cfg | cut -f 3 -d ' ')
blocktt_rank=$(echo $cfg | cut -f 4 -d ' ')
s_merged_to=$(echo $cfg | cut -f 5 -d ' ')
trainable_type=$(echo $cfg | cut -f 6 -d ' ')
lr=$(echo $cfg | cut -f 7 -d ' ')
seed=$(echo $cfg | cut -f 8 -d ' ')

echo $MODEL

OUTPUT=${OUTPUT_SRC_DIR}/${MODEL}/blocktt/commonsense/decomp_${decomp_mode}_pos_${train_position}_rank_${blocktt_rank}_smerge_${s_merged_to}_type_${trainable_type}/lr_${lr}/seed_${seed}

mkdir -p $OUTPUT

cd ${SRC_DIR}

accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    --mixed_precision="bf16" \
    src/finetune_blocktt.py \
    --model_name_or_path ${MODEL} \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --max_seq_len 2048 \
    --learning_rate ${lr} \
    --weight_decay 0. \
    --num_train_epochs 3 \
    --mixed_precision bf16 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type linear \
    --num_warmup_steps 0.03 \
    --seed ${seed} \
    --gradient_checkpointing \
    --instruction_type single \
    --decomp_mode ${decomp_mode} \
    --train_position ${train_position} \
    --blocktt_rank ${blocktt_rank} \
    --s_merged_to ${s_merged_to} \
    --trainable_type ${trainable_type} \
    --save_interval 5000 \
    --val_set_size 120 \
    --eval_step 400 \
    --data_path ${DATA_DIR}/LLM-Adapters/ft-training_set/commonsense_170k.json \
    --output_dir $OUTPUT 2> >(tee $OUTPUT/err.log >&2) | tee $OUTPUT/training.log

bash ./bash_scripts/eval_commonsense.sh $OUTPUT
```

- [ ] **Step 3: Commit**

```bash
git add ref/LIFT/bash_scripts/finetune_commonsense_blocktt.sh ref/LIFT/bash_scripts/slurm_config_blocktt_commonsense.txt
git commit -m "feat: add commonsense experiment scripts for blocktt in LIFT"
```

---

### Task 5: Smoke test end-to-end

- [ ] **Step 1: Verify `finetune_blocktt.py` imports resolve correctly**

Run from the repo root:
```bash
cd /home/yequan/Project/lora/lora-without-regret/ref/LIFT
../../.venv/bin/python -c "
import sys, os
sys.path.insert(0, 'src')
sys.path.insert(0, '.')
sys.path.insert(0, '../..')
from finetune_blocktt import parse_args, materialize_btt_to_linear, resolve_blocktt_rank
print('finetune_blocktt imports OK')
print('resolve_blocktt_rank(\"full\") =', resolve_blocktt_rank('full'))
print('resolve_blocktt_rank(\"64\") =', resolve_blocktt_rank('64'))
"
```
Expected:
```
finetune_blocktt imports OK
resolve_blocktt_rank("full") = full
resolve_blocktt_rank("64") = 64
```

- [ ] **Step 2: Verify all other LIFT scripts still work**

```bash
cd /home/yequan/Project/lora/lora-without-regret/ref/LIFT
../../.venv/bin/python -c "
import sys; sys.path.insert(0, 'src'); sys.path.insert(0, '.')
# finetune_sft (LIFT/full FT)
import finetune_sft
print('finetune_sft OK')
# finetune_s2ft
import finetune_s2ft
print('finetune_s2ft OK')
"
```
Expected: both print OK. (`finetune_lora.py` may have peft/other import issues unrelated to our changes — that's fine, it's not our scope.)

- [ ] **Step 3: Verify bash scripts are executable**

```bash
chmod +x ref/LIFT/bash_scripts/finetune_math_blocktt.sh
chmod +x ref/LIFT/bash_scripts/finetune_commonsense_blocktt.sh
```

- [ ] **Step 4: Final commit if any fixups needed**

```bash
git add -A ref/LIFT/
git commit -m "chore: smoke test fixups for LIFT blocktt integration"
```
(Skip this step if no fixups were needed.)
