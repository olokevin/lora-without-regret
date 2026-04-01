# generate_teacher_data.py
"""
Generate teacher model completions and top-K logits for knowledge distillation.

Example:
  CUDA_VISIBLE_DEVICES=2 uv run generate_teacher_data.py \
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
        default="/data/yequan/fura/kd_data/",
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


def save_config(
    output_path,
    teacher_model_id,
    dataset,
    dataset_split,
    top_k,
    max_tokens,
    shared_vocab_size,
    num_examples,
    prompt_template,
):
    os.makedirs(output_path, exist_ok=True)
    config = {
        "teacher_model_id": teacher_model_id,
        "dataset": dataset,
        "dataset_split": dataset_split,
        "top_k": top_k,
        "max_tokens": max_tokens,
        "shared_vocab_size": shared_vocab_size,
        "num_examples": num_examples,
        "prompt_template": prompt_template,
        "temperature": 0,
    }
    with open(os.path.join(output_path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def save_logit_chunk(output_path, chunk_idx, topk_values, topk_indices, seq_lengths):
    logits_dir = os.path.join(output_path, "logits")
    os.makedirs(logits_dir, exist_ok=True)
    save_safetensors_file(
        {
            "topk_values": topk_values,
            "topk_indices": topk_indices,
            "seq_lengths": seq_lengths,
        },
        os.path.join(logits_dir, f"chunk_{chunk_idx}.safetensors"),
    )


def save_completion(output_path, index, question, ground_truth, prompt, completion, token_ids):
    os.makedirs(output_path, exist_ok=True)
    entry = {
        "index": index,
        "question": question,
        "ground_truth": ground_truth,
        "prompt": prompt,
        "completion": completion,
        "token_ids": token_ids,
    }
    with open(os.path.join(output_path, "completions.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    output_path = resolve_output_path(args)

    if os.path.exists(os.path.join(output_path, "config.json")):
        print(f"Output directory {output_path} already contains data. Skipping.")
        print("Delete the directory to regenerate.")
        return

    print(f"Loading dataset and preparing prompts...")
    prompts, questions, ground_truths, tokenizer = load_and_prepare_dataset(args)
    num_examples = len(prompts)
    print(f"Prepared {num_examples} prompts")

    print(f"Loading teacher model via vLLM: {args.teacher_model_id}")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.teacher_model_id,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
        max_logprobs=args.top_k,
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.max_tokens,
        logprobs=args.top_k,
    )

    print(f"Generating completions...")
    outputs = llm.generate(prompts, sampling_params)

    # Determine shared vocab size
    teacher_vocab_size = llm.llm_engine.model_config.hf_config.vocab_size
    student_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    student_vocab_size = student_tokenizer.vocab_size
    shared_vocab_size = min(teacher_vocab_size, student_vocab_size)
    del student_tokenizer

    save_config(
        output_path=output_path,
        teacher_model_id=args.teacher_model_id,
        dataset=args.dataset,
        dataset_split=args.dataset_split,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        shared_vocab_size=shared_vocab_size,
        num_examples=num_examples,
        prompt_template=args.prompt_template,
    )

    chunk_topk_values = []
    chunk_topk_indices = []
    chunk_seq_lengths = []
    chunk_idx = 0

    for i, output in enumerate(tqdm(outputs, desc="Processing outputs")):
        completion = output.outputs[0]
        completion_text = completion.text
        token_ids = list(completion.token_ids)
        seq_len = len(token_ids)

        # Extract top-K logprobs per token
        topk_vals = torch.zeros(args.max_tokens, args.top_k, dtype=torch.bfloat16)
        topk_idxs = torch.zeros(args.max_tokens, args.top_k, dtype=torch.int32)

        for t, token_logprobs in enumerate(completion.logprobs):
            if t >= args.max_tokens:
                break
            sorted_items = sorted(token_logprobs.items(), key=lambda x: x[1].logprob, reverse=True)
            for k_idx, (tok_id, logprob_obj) in enumerate(sorted_items[:args.top_k]):
                topk_vals[t, k_idx] = logprob_obj.logprob
                topk_idxs[t, k_idx] = tok_id

        chunk_topk_values.append(topk_vals)
        chunk_topk_indices.append(topk_idxs)
        chunk_seq_lengths.append(seq_len)

        save_completion(
            output_path=output_path,
            index=i,
            question=questions[i],
            ground_truth=ground_truths[i],
            prompt=prompts[i],
            completion=completion_text,
            token_ids=token_ids,
        )

        if len(chunk_topk_values) >= args.chunk_size or i == num_examples - 1:
            save_logit_chunk(
                output_path=output_path,
                chunk_idx=chunk_idx,
                topk_values=torch.stack(chunk_topk_values),
                topk_indices=torch.stack(chunk_topk_indices),
                seq_lengths=torch.tensor(chunk_seq_lengths, dtype=torch.int32),
            )
            print(f"Saved logit chunk {chunk_idx} ({len(chunk_topk_values)} examples)")
            chunk_topk_values = []
            chunk_topk_indices = []
            chunk_seq_lengths = []
            chunk_idx += 1

    print(f"Done! Saved {num_examples} examples to {output_path}")


if __name__ == "__main__":
    main()
