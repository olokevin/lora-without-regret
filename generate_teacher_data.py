# generate_teacher_data.py
"""
Generate teacher model completions and top-K logits for knowledge distillation.

Example:
  CUDA_VISIBLE_DEVICES=0 uv run generate_teacher_data.py \
    --teacher-model-id deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --top-k 256
"""

import argparse
import json
import os
import sys

import torch
from datasets import load_dataset
from math_utils import last_boxed_only_string, remove_boxed
from safetensors.torch import save_file as save_safetensors_file
from transformers import AutoTokenizer
from tqdm import tqdm


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate teacher completions and top-K logits for KD"
    )
    parser.add_argument(
        "--teacher-model-id",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        help="Teacher model HuggingFace ID",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="qwedsacf/competition_math",
        help="Dataset name (default: qwedsacf/competition_math)",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train[:7500]",
        help="Dataset split (default: train[:7500])",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/data/yequan/fura/kd/",
        help="Base output directory",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Max generation length (default: 1024)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=256,
        help="Number of top logits to save per token (default: 256)",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="boxed.prompt",
        help="Path to prompt template file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for vLLM generation (default: 16)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Examples per logit chunk file (default: 500)",
    )
    return parser.parse_args(argv)


def resolve_output_path(args):
    model_name = args.teacher_model_id.split("/")[-1]
    dataset_name = args.dataset.split("/")[-1]
    return os.path.join(args.output_dir, f"{model_name}-{dataset_name}")


def load_and_prepare_dataset(args):
    dataset = load_dataset(args.dataset, split=args.dataset_split)

    with open(args.prompt_template, "r", encoding="utf-8") as f:
        template = f.read().strip()

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prompts = []
    ground_truths = []
    questions = []
    for example in dataset:
        question = example["problem"]
        with_template = template.replace("{question}", question)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": with_template}],
            tokenize=False,
            add_generation_prompt=True,
        )
        answer = remove_boxed(last_boxed_only_string(example["solution"]))
        prompts.append(prompt)
        ground_truths.append(answer)
        questions.append(question)

    return prompts, questions, ground_truths, tokenizer
