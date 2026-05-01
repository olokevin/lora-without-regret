"""Standalone commonsense evaluation script.

Loads a merged checkpoint (or HF model ID) and evaluates on 8 commonsense tasks
using the same prompt format and answer extraction as the LIFT eval.

Usage:
    CUDA_VISIBLE_DEVICES=0 python eval_commonsense.py \
        --model /path/to/checkpoint \
        --tokenizer meta-llama/Meta-Llama-3-8B \
        --output-dir /path/to/output
"""

import argparse
import json
import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

PROMPT_TEMPLATE = """<s> Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

DATASETS = [
    "boolq", "piqa", "social_i_qa",
    "ARC-Challenge", "ARC-Easy", "openbookqa",
    "hellaswag", "winogrande",
]

DATA_ROOT = os.path.join(
    os.path.dirname(__file__),
    "ref/LIFT/LLM-Adapters/dataset",
)


def extract_answer(dataset: str, sentence: str) -> str:
    sentence = sentence.lower().strip()
    if dataset == "boolq":
        m = re.findall(r"true|false", sentence)
    elif dataset == "piqa":
        m = re.findall(r"solution1|solution2", sentence)
    elif dataset in ("social_i_qa", "ARC-Challenge", "ARC-Easy", "openbookqa"):
        m = re.findall(r"answer1|answer2|answer3|answer4|answer5", sentence)
    elif dataset == "hellaswag":
        m = re.findall(r"ending1|ending2|ending3|ending4", sentence)
    elif dataset == "winogrande":
        m = re.findall(r"option1|option2", sentence)
    else:
        m = []
    return m[0] if m else ""


@torch.no_grad()
def evaluate_dataset(model, tokenizer, dataset_name, data_path, batch_size=16):
    data = json.load(open(data_path))
    prompts = [PROMPT_TEMPLATE.format_map(ex) for ex in data]

    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=256,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    all_outputs = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)
        output_ids = model.generate(
            **inputs,
            generation_config=generation_config,
        )
        # Decode only the new tokens
        for j, ids in enumerate(output_ids):
            new_ids = ids[inputs["input_ids"].shape[1] :]
            text = tokenizer.decode(new_ids, skip_special_tokens=True)
            all_outputs.append(text)
        if (i // batch_size) % 10 == 0:
            print(f"  [{dataset_name}] {min(i + batch_size, len(prompts))}/{len(prompts)}")

    correct = 0
    results = []
    for ex, output in zip(data, all_outputs):
        target = ex["answer"].lower()
        pred = extract_answer(dataset_name, output)
        is_correct = target == pred
        if is_correct:
            correct += 1
        results.append({
            "instruction": ex.get("instruction", ""),
            "answer": ex["answer"],
            "raw_output": output,
            "prediction": pred,
            "correct": is_correct,
        })

    accuracy = correct / len(data) * 100
    print(f"Result {accuracy:.1f}%, total: {len(data)}")
    return accuracy, len(data), results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model path or HF ID")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path (defaults to --model)")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset names")
    parser.add_argument("--dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    args = parser.parse_args()

    tokenizer_path = args.tokenizer or args.model
    output_dir = args.output_dir or os.path.join(args.model, "commonsense")
    datasets = args.datasets.split(",") if args.datasets else DATASETS
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading model from {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model device: {model.device}, dtype: {model.dtype}")

    summary = {}
    for ds in datasets:
        data_path = os.path.join(DATA_ROOT, ds, "test.json")
        if not os.path.exists(data_path):
            print(f"Skipping {ds}: {data_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating {ds}")
        print(f"{'='*60}")

        acc, total, results = evaluate_dataset(
            model, tokenizer, ds, data_path, args.batch_size
        )
        summary[ds] = {"accuracy": acc, "total": total}

        ds_out = os.path.join(output_dir, ds)
        os.makedirs(ds_out, exist_ok=True)
        with open(os.path.join(ds_out, "eval.log"), "w") as f:
            f.write(f"Result {acc:.1f}, total: {total}\n")
        with open(os.path.join(ds_out, "model_predictions.jsonl"), "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    total_acc = 0
    for ds, info in summary.items():
        print(f"  {ds:20s}: {info['accuracy']:.1f}% ({info['total']} samples)")
        total_acc += info["accuracy"]
    if summary:
        avg = total_acc / len(summary)
        print(f"  {'Average':20s}: {avg:.1f}%")

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
