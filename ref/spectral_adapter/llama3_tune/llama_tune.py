import torch
import os
from datasets import load_dataset, Dataset
from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from argparse import ArgumentParser
from pathlib import Path

from blocktt_utils import apply_blocktt_to_model, write_blocktt_metadata

import lm_eval
from lm_eval.models.huggingface import HFLM

"""login for model download permission"""
# login()


def run_gsm8k_eval(model, tokenizer, num_fewshot):
    """Run GSM8K evaluation and return the flexible-extract exact_match score."""
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=8)
    results = lm_eval.simple_evaluate(
        model=hflm,
        tasks=["gsm8k"],
        num_fewshot=num_fewshot,
        batch_size=8,
    )
    return results["results"]["gsm8k"]["exact_match,flexible-extract"]


class InlineEvalCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_steps, num_fewshot):
        self.tokenizer = tokenizer
        self.eval_steps = eval_steps
        self.num_fewshot = num_fewshot

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.eval_steps <= 0 or state.global_step % self.eval_steps != 0:
            return
        print(f"\n[InlineEval] Running GSM8K {self.num_fewshot}-shot evaluation at step {state.global_step}...")
        model.eval()
        try:
            score = run_gsm8k_eval(model, self.tokenizer, self.num_fewshot)
            print(f"[InlineEval] Step {state.global_step} — gsm8k exact_match (flexible-extract, {self.num_fewshot}-shot): {score:.4f}")
            import wandb
            if wandb.run is not None:
                wandb.log({f"val/gsm8k_exact_match_{self.num_fewshot}shot": score}, step=state.global_step)
        except Exception as e:
            print(f"[InlineEval] Evaluation failed at step {state.global_step}: {e}")
        finally:
            model.train()


class BlockTTMetadataCallback(TrainerCallback):
    def __init__(self, metadata):
        self.metadata = metadata

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        write_blocktt_metadata(checkpoint_dir, self.metadata)
        return control


def format_lr(lr: float) -> str:
    return f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def build_wandb_run_name(args, blocktt_metadata):
    parts = [args.model, f"lr{format_lr(args.learning_rate)}"]
    if args.model == "lora":
        parts.append(f"r{args.lora_rank}")
    elif args.model == "spectral":
        parts.extend([f"r{args.lora_rank}", "spectral_top"])
    elif args.model == "blocktt":
        parts.extend(
            [
                f"rank{blocktt_metadata['blocktt_rank']}",
                blocktt_metadata["blocktt_type"],
                str(blocktt_metadata["decomp_mode_resolved"]).replace(" ", ""),
                f"tp{blocktt_metadata['train_position']}",
            ]
        )
        if blocktt_metadata["s_merged_to"] is not None:
            parts.append(f"s{blocktt_metadata['s_merged_to']}")
    return "-".join(parts)


parser = ArgumentParser()
parser.add_argument("--model", type=str, default="full", choices=["full", "lora", "spectral", "blocktt"])
parser.add_argument(
    "--learning-rate",
    "--lr",
    type=float,
    default=1e-5,
    dest="learning_rate",
    help="Learning rate.",
)
parser.add_argument(
    "--lora-rank",
    type=int,
    default=8,
    help="LoRA rank for lora/spectral modes.",
)
parser.add_argument(
    "--blocktt-rank",
    type=str,
    default="full",
    help="BlockTT rank as a positive integer or 'full'.",
)
parser.add_argument(
    "--blocktt-type",
    type=str,
    default="all",
    choices=["all", "mlp", "attn"],
    help="Target module family for BlockTT conversion.",
)
parser.add_argument(
    "--decomp-mode",
    type=str,
    default="input_one_block",
    help="BlockTT decomposition mode (scalar or mapping literal).",
)
parser.add_argument(
    "--train-position",
    type=str,
    default="small",
    choices=["small", "large", "both"],
    help="Select which BlockTT core to train.",
)
parser.add_argument(
    "--s-merged-to",
    type=str,
    default=None,
    help="Optional singular-value merge target for BlockTT init.",
)
parser.add_argument(
    "--blocktt-train-bias",
    action="store_true",
    default=True,
    help="Train biases for BlockTT-converted layers.",
)
parser.add_argument(
    "--no-blocktt-train-bias",
    action="store_false",
    dest="blocktt_train_bias",
    help="Do not train biases for BlockTT-converted layers.",
)
parser.add_argument(
    "--enable-eco-ckpt",
    action="store_true",
    default=True,
    dest="enable_eco_ckpt",
    help="Save only the final checkpoint (model only, no optimizer state).",
)
parser.add_argument(
    "--no-eco-ckpt",
    action="store_false",
    dest="enable_eco_ckpt",
    help="Disable eco checkpoint mode (use default HF saving behavior).",
)
parser.add_argument(
    "--ckpt-dir",
    type=str,
    default="/data/yequan/fura/spectral_llama",
    help="Directory to save the final checkpoint.",
)
parser.add_argument(
    "--wandb-project",
    type=str,
    default="fura_sft_llama3_8B",
    help="Weights & Biases project name.",
)
parser.add_argument(
    "--suffix",
    type=str,
    default="",
    help="Optional suffix appended to the auto-generated W&B run name.",
)
parser.add_argument(
    "--inline-eval-steps",
    type=int,
    default=100,
    dest="inline_eval_steps",
    help="Run GSM8K eval every N steps and log to wandb under val/. 0 disables.",
)
parser.add_argument(
    "--inline-eval-nshot",
    type=int,
    default=0,
    dest="inline_eval_nshot",
    help="Number of few-shot examples for inline GSM8K eval (default: 0-shot).",
)
args = parser.parse_args()

torch.manual_seed(0)
model_checkpoint = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

if args.model in {"lora", "spectral"}:
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "gate_proj"],
        spectral_top=True if args.model == "spectral" else False,
    )

model = AutoModelForCausalLM.from_pretrained(
    model_checkpoint,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)


blocktt_metadata = None
if args.model in {"lora", "spectral"}:
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
elif args.model == "blocktt":
    blocktt_metadata, blocktt_stats = apply_blocktt_to_model(
        model,
        blocktt_rank=args.blocktt_rank,
        blocktt_type=args.blocktt_type,
        decomp_mode=args.decomp_mode,
        train_position=args.train_position,
        s_merged_to=args.s_merged_to,
        train_bias=args.blocktt_train_bias,
        set_trainability=True,
    )
    print("BlockTT configuration:")
    print(f"  Rank: {blocktt_metadata['blocktt_rank']}")
    print(f"  Type: {blocktt_metadata['blocktt_type']}")
    print(f"  Train position: {blocktt_metadata['train_position']}")
    print(f"  S merged to: {blocktt_metadata['s_merged_to']}")
    print(f"  Converted layers: {len(blocktt_metadata['converted_modules'])}")
    print(
        "  Trainable params: "
        f"{blocktt_stats['trainable_param_count']}/{blocktt_stats['total_param_count']}"
    )

for n, p in model.named_parameters():
    print(n, p.shape)

wandb_run_name = build_wandb_run_name(args, blocktt_metadata)
suffix = args.suffix.strip()
if suffix:
    wandb_run_name = f"{wandb_run_name}-{suffix}"
os.environ["WANDB_PROJECT"] = args.wandb_project


dataset = load_dataset("microsoft/orca-math-word-problems-200k")["train"].shuffle(seed=0)
dataset = dataset.select(range(0, 10000))
data = dataset.to_pandas()
data["text"] = data[["question", "answer"]].apply(lambda x: "question: " + x["question"] + " answer: " + x["answer"], axis=1)
data = Dataset.from_pandas(data)


def tokenize(sample):
    model_inps = tokenizer(sample["text"], padding=True, truncation=True, max_length=512)
    return model_inps


tokenized_data = data.map(tokenize, batched=True, desc="Tokenizing data", remove_columns=data.column_names)

batch_size = 8
save_kwargs = {}
if args.enable_eco_ckpt:
    save_kwargs["save_strategy"] = "no"
else:
    save_kwargs["save_strategy"] = "steps"
    save_kwargs["save_steps"] = 100

training_arguments = TrainingArguments(
    output_dir=args.ckpt_dir,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=1,
    learning_rate=args.learning_rate,
    weight_decay=0.1,
    logging_steps=1,
    num_train_epochs=1,
    push_to_hub=False,
    seed=0,
    lr_scheduler_type="constant",
    report_to=["wandb"],
    run_name=wandb_run_name,
    **save_kwargs,
)

print(f"W&B project: {args.wandb_project}")
print(f"W&B run name: {wandb_run_name}")


callbacks = []
if args.inline_eval_steps > 0:
    callbacks.append(InlineEvalCallback(tokenizer, args.inline_eval_steps, args.inline_eval_nshot))
if blocktt_metadata is not None:
    callbacks.append(BlockTTMetadataCallback(blocktt_metadata))

trainer = Trainer(
    model=model,
    train_dataset=tokenized_data,
    args=training_arguments,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    callbacks=callbacks,
)
trainer.train()

# Final 5-shot GSM8K evaluation (matches llama_test.py)
print("\n[FinalEval] Running GSM8K 5-shot evaluation after training...")
model.eval()
try:
    final_score = run_gsm8k_eval(model, tokenizer, num_fewshot=5)
    print(f"[FinalEval] gsm8k exact_match (flexible-extract, 5-shot): {final_score:.4f}")
    import wandb
    if wandb.run is not None:
        wandb.log({"val/gsm8k_exact_match_5shot_final": final_score})
except Exception as e:
    print(f"[FinalEval] Evaluation failed: {e}")
finally:
    model.train()

if args.enable_eco_ckpt:
    save_dir = Path(args.ckpt_dir) / wandb_run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving final model-only checkpoint to {save_dir}")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    if blocktt_metadata is not None:
        write_blocktt_metadata(save_dir, blocktt_metadata)
else:
    if blocktt_metadata is not None:
        output_dir = Path(training_arguments.output_dir)
        for checkpoint_dir in output_dir.glob("checkpoint-*"):
            write_blocktt_metadata(checkpoint_dir, blocktt_metadata)
