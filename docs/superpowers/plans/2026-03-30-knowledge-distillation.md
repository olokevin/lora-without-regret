# Knowledge Distillation Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `generate_teacher_data.py` to cache teacher model outputs (completions + top-K logits) and `run_kd.py` to train student models via knowledge distillation, supporting all four training modes and compatible with `plot_singular_relationship.py`.

**Architecture:** Two-script pipeline. `generate_teacher_data.py` uses vLLM to generate teacher completions with top-K logprobs, saving to disk as JSONL + safetensors chunks. `run_kd.py` loads cached data and trains the student using either SFT loss (on teacher completions) or KL divergence loss (on teacher logits), reusing `run_sft.py`'s model preparation, optimizer, and scheduler infrastructure. Checkpoints saved in `step={N}/model.safetensors` format for singular vector analysis.

**Tech Stack:** PyTorch, vLLM, safetensors, HuggingFace transformers/datasets, wandb

---

### Task 1: `generate_teacher_data.py` — CLI and data preparation

**Files:**
- Create: `generate_teacher_data.py`
- Test: `tests/test_generate_teacher_data.py`

- [ ] **Step 1: Write CLI parsing tests**

```python
# tests/test_generate_teacher_data.py
import unittest


class TestGenerateTeacherDataCli(unittest.TestCase):
    def test_defaults(self):
        import generate_teacher_data

        args = generate_teacher_data.parse_args([])
        self.assertEqual(args.teacher_model_id, "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
        self.assertEqual(args.dataset, "qwedsacf/competition_math")
        self.assertEqual(args.dataset_split, "train[:7500]")
        self.assertEqual(args.output_dir, "/data/yequan/fura/kd/")
        self.assertEqual(args.max_tokens, 1024)
        self.assertEqual(args.top_k, 256)
        self.assertEqual(args.prompt_template, "boxed.prompt")
        self.assertEqual(args.batch_size, 16)
        self.assertEqual(args.chunk_size, 500)

    def test_custom_args(self):
        import generate_teacher_data

        args = generate_teacher_data.parse_args([
            "--teacher-model-id", "some/model",
            "--top-k", "64",
            "--max-tokens", "512",
            "--chunk-size", "100",
        ])
        self.assertEqual(args.teacher_model_id, "some/model")
        self.assertEqual(args.top_k, 64)
        self.assertEqual(args.max_tokens, 512)
        self.assertEqual(args.chunk_size, 100)

    def test_resolve_output_path(self):
        import generate_teacher_data

        args = generate_teacher_data.parse_args([
            "--teacher-model-id", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "--dataset", "qwedsacf/competition_math",
        ])
        path = generate_teacher_data.resolve_output_path(args)
        self.assertEqual(
            path,
            "/data/yequan/fura/kd/DeepSeek-R1-Distill-Qwen-7B-competition_math",
        )

    def test_resolve_output_path_strips_org(self):
        import generate_teacher_data

        args = generate_teacher_data.parse_args([
            "--teacher-model-id", "org/ModelName",
            "--dataset", "user/dataset_name",
        ])
        path = generate_teacher_data.resolve_output_path(args)
        self.assertEqual(path, "/data/yequan/fura/kd/ModelName-dataset_name")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_generate_teacher_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'generate_teacher_data'`

- [ ] **Step 3: Write `generate_teacher_data.py` — CLI parsing and output path resolution**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_generate_teacher_data.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add generate_teacher_data.py tests/test_generate_teacher_data.py
git commit -m "feat: add generate_teacher_data.py CLI parsing and dataset preparation"
```

---

### Task 2: `generate_teacher_data.py` — vLLM generation and saving

**Files:**
- Modify: `generate_teacher_data.py`
- Modify: `tests/test_generate_teacher_data.py`

- [ ] **Step 1: Write test for save_chunk and save_config**

```python
# Add to tests/test_generate_teacher_data.py
import os
import tempfile

import torch
from safetensors.torch import load_file


class TestSaveArtifacts(unittest.TestCase):
    def test_save_config_writes_valid_json(self):
        import generate_teacher_data

        with tempfile.TemporaryDirectory() as tmp:
            generate_teacher_data.save_config(
                output_path=tmp,
                teacher_model_id="org/Model",
                dataset="org/data",
                dataset_split="train[:100]",
                top_k=64,
                max_tokens=512,
                shared_vocab_size=151643,
                num_examples=100,
                prompt_template="boxed.prompt",
            )
            with open(os.path.join(tmp, "config.json")) as f:
                cfg = json.load(f)
            self.assertEqual(cfg["teacher_model_id"], "org/Model")
            self.assertEqual(cfg["top_k"], 64)
            self.assertEqual(cfg["max_tokens"], 512)
            self.assertEqual(cfg["shared_vocab_size"], 151643)
            self.assertEqual(cfg["num_examples"], 100)

    def test_save_logit_chunk_roundtrips(self):
        import generate_teacher_data

        topk_values = torch.randn(3, 10, 64, dtype=torch.bfloat16)
        topk_indices = torch.randint(0, 1000, (3, 10, 64), dtype=torch.int32)
        seq_lengths = torch.tensor([8, 10, 5], dtype=torch.int32)

        with tempfile.TemporaryDirectory() as tmp:
            generate_teacher_data.save_logit_chunk(
                output_path=tmp,
                chunk_idx=0,
                topk_values=topk_values,
                topk_indices=topk_indices,
                seq_lengths=seq_lengths,
            )
            chunk_path = os.path.join(tmp, "logits", "chunk_0.safetensors")
            self.assertTrue(os.path.exists(chunk_path))

            loaded = load_file(chunk_path)
            torch.testing.assert_close(loaded["topk_values"], topk_values)
            torch.testing.assert_close(loaded["topk_indices"], topk_indices)
            torch.testing.assert_close(loaded["seq_lengths"], seq_lengths)

    def test_save_completion_appends_jsonl(self):
        import generate_teacher_data

        with tempfile.TemporaryDirectory() as tmp:
            generate_teacher_data.save_completion(
                output_path=tmp,
                index=0,
                question="What is 1+1?",
                ground_truth="2",
                prompt="prompt text",
                completion="The answer is 2",
                token_ids=[1, 2, 3],
            )
            generate_teacher_data.save_completion(
                output_path=tmp,
                index=1,
                question="What is 2+2?",
                ground_truth="4",
                prompt="prompt text 2",
                completion="The answer is 4",
                token_ids=[4, 5, 6],
            )
            with open(os.path.join(tmp, "completions.jsonl")) as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 2)
            entry = json.loads(lines[0])
            self.assertEqual(entry["index"], 0)
            self.assertEqual(entry["question"], "What is 1+1?")
            self.assertEqual(entry["token_ids"], [1, 2, 3])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_generate_teacher_data.py::TestSaveArtifacts -v`
Expected: FAIL with `AttributeError: module 'generate_teacher_data' has no attribute 'save_config'`

- [ ] **Step 3: Implement save_config, save_logit_chunk, save_completion**

Add to `generate_teacher_data.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_generate_teacher_data.py::TestSaveArtifacts -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Implement `main()` with vLLM generation**

Add to `generate_teacher_data.py`:

```python
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
```

- [ ] **Step 6: Verify syntax**

Run: `python -m py_compile generate_teacher_data.py`
Expected: No output (success)

- [ ] **Step 7: Commit**

```bash
git add generate_teacher_data.py tests/test_generate_teacher_data.py
git commit -m "feat: add vLLM generation and artifact saving to generate_teacher_data.py"
```

---

### Task 3: `run_kd.py` — CLI parsing and flag validation

**Files:**
- Create: `run_kd.py`
- Create: `tests/test_run_kd_cli.py`

- [ ] **Step 1: Write CLI parsing tests**

```python
# tests/test_run_kd_cli.py
import unittest


class TestRunKdCli(unittest.TestCase):
    def test_kd_loss_type_required(self):
        import run_kd

        with self.assertRaises(SystemExit):
            run_kd.parse_args(["--train-mode", "full"])

    def test_defaults_sft(self):
        import run_kd

        args = run_kd.parse_args(["--kd-loss-type", "sft", "--train-mode", "full"])
        run_kd.apply_mode_defaults(args)
        self.assertEqual(args.kd_loss_type, "sft")
        self.assertEqual(args.student_model_id, "Qwen/Qwen2.5-0.5B")
        self.assertEqual(
            args.teacher_data_dir,
            "/data/yequan/fura/kd/DeepSeek-R1-Distill-Qwen-7B-competition_math",
        )
        self.assertEqual(args.top_k, 256)
        self.assertEqual(args.save_steps, "1,10,final")

    def test_defaults_kl(self):
        import run_kd

        args = run_kd.parse_args(["--kd-loss-type", "kl", "--train-mode", "svd"])
        run_kd.apply_mode_defaults(args)
        self.assertEqual(args.kd_loss_type, "kl")

    def test_parse_save_steps_default(self):
        import run_kd

        steps = run_kd.parse_save_steps("1,10,final", total_steps=50)
        self.assertEqual(steps, {1, 10, 50})

    def test_parse_save_steps_custom(self):
        import run_kd

        steps = run_kd.parse_save_steps("1,5,20,final", total_steps=100)
        self.assertEqual(steps, {1, 5, 20, 100})

    def test_parse_save_steps_no_final(self):
        import run_kd

        steps = run_kd.parse_save_steps("1,10", total_steps=50)
        self.assertEqual(steps, {1, 10})

    def test_mode_defaults_lr(self):
        import run_kd

        for mode in ["full", "lora", "blocktt", "svd"]:
            args = run_kd.parse_args(["--kd-loss-type", "sft", "--train-mode", mode])
            run_kd.apply_mode_defaults(args)
            self.assertIsNotNone(args.lr, f"lr should be set for mode {mode}")

    def test_reject_lora_flags_for_non_lora_mode(self):
        import run_kd

        argv = ["--kd-loss-type", "sft", "--train-mode", "full", "--lora-rank", "4"]
        args = run_kd.parse_args(argv)
        with self.assertRaises(ValueError):
            run_kd.validate_mode_specific_flags(args, argv)

    def test_reject_blocktt_flags_for_non_blocktt_mode(self):
        import run_kd

        argv = ["--kd-loss-type", "sft", "--train-mode", "lora", "--decomp-mode", "input_one_block"]
        args = run_kd.parse_args(argv)
        with self.assertRaises(ValueError):
            run_kd.validate_mode_specific_flags(args, argv)

    def test_train_position_defaults(self):
        import run_kd

        args = run_kd.parse_args(["--kd-loss-type", "sft", "--train-mode", "blocktt"])
        run_kd.apply_mode_defaults(args)
        self.assertEqual(args.train_position, "small")

        args = run_kd.parse_args(["--kd-loss-type", "sft", "--train-mode", "svd"])
        run_kd.apply_mode_defaults(args)
        self.assertEqual(args.train_position, "output")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_run_kd_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'run_kd'`

- [ ] **Step 3: Implement `run_kd.py` CLI parsing**

Create `run_kd.py` with arg parsing that mirrors `run_sft.py` but replaces dataset flags with KD-specific ones. Import shared functions from `run_sft.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_run_kd_cli.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add run_kd.py tests/test_run_kd_cli.py
git commit -m "feat: add run_kd.py CLI parsing and flag validation"
```

---

### Task 4: `run_kd.py` — Teacher data loading

**Files:**
- Modify: `run_kd.py`
- Modify: `tests/test_run_kd_cli.py`

- [ ] **Step 1: Write tests for teacher data loading**

```python
# Add to tests/test_run_kd_cli.py
import os
import json
import tempfile

import torch
from safetensors.torch import save_file as save_safetensors_file


class TestTeacherDataLoading(unittest.TestCase):
    def _create_teacher_data(self, tmp_dir, num_examples=4, top_k=8, max_tokens=16):
        """Create minimal teacher data directory for testing."""
        config = {
            "teacher_model_id": "test/teacher",
            "dataset": "test/data",
            "dataset_split": "train[:4]",
            "top_k": top_k,
            "max_tokens": max_tokens,
            "shared_vocab_size": 100,
            "num_examples": num_examples,
            "prompt_template": "boxed.prompt",
            "temperature": 0,
        }
        with open(os.path.join(tmp_dir, "config.json"), "w") as f:
            json.dump(config, f)

        with open(os.path.join(tmp_dir, "completions.jsonl"), "w") as f:
            for i in range(num_examples):
                entry = {
                    "index": i,
                    "question": f"Q{i}",
                    "ground_truth": str(i),
                    "prompt": f"prompt {i}",
                    "completion": f"answer is {i}",
                    "token_ids": list(range(max_tokens)),
                }
                f.write(json.dumps(entry) + "\n")

        logits_dir = os.path.join(tmp_dir, "logits")
        os.makedirs(logits_dir)
        save_safetensors_file(
            {
                "topk_values": torch.randn(num_examples, max_tokens, top_k, dtype=torch.bfloat16),
                "topk_indices": torch.randint(0, 100, (num_examples, max_tokens, top_k), dtype=torch.int32),
                "seq_lengths": torch.full((num_examples,), max_tokens, dtype=torch.int32),
            },
            os.path.join(logits_dir, "chunk_0.safetensors"),
        )

    def test_load_teacher_config(self):
        import run_kd

        with tempfile.TemporaryDirectory() as tmp:
            self._create_teacher_data(tmp)
            config = run_kd.load_teacher_config(tmp)
            self.assertEqual(config["top_k"], 8)
            self.assertEqual(config["shared_vocab_size"], 100)

    def test_load_teacher_config_missing_dir_raises(self):
        import run_kd

        with self.assertRaises(FileNotFoundError):
            run_kd.load_teacher_config("/nonexistent/path")

    def test_load_completions(self):
        import run_kd

        with tempfile.TemporaryDirectory() as tmp:
            self._create_teacher_data(tmp, num_examples=3)
            completions = run_kd.load_completions(tmp)
            self.assertEqual(len(completions), 3)
            self.assertEqual(completions[0]["question"], "Q0")
            self.assertEqual(completions[2]["index"], 2)

    def test_build_sft_dataset(self):
        import run_kd

        completions = [
            {"prompt": "What is 1+1?", "completion": "2", "token_ids": [1, 2, 3]},
            {"prompt": "What is 2+2?", "completion": "4", "token_ids": [4, 5, 6]},
        ]
        dataset = run_kd.KDSftDataset(completions, max_length=32)
        self.assertEqual(len(dataset), 2)
        item = dataset[0]
        self.assertIn("input_ids", item)
        self.assertIn("labels", item)
        self.assertIn("attention_mask", item)
        # prompt tokens should be masked with -100
        # completion tokens should have real labels

    def test_build_kl_dataset(self):
        import run_kd

        with tempfile.TemporaryDirectory() as tmp:
            self._create_teacher_data(tmp, num_examples=2, top_k=8, max_tokens=16)
            completions = run_kd.load_completions(tmp)
            dataset = run_kd.KDKlDataset(completions, tmp, top_k=8, max_length=32)
            self.assertEqual(len(dataset), 2)
            item = dataset[0]
            self.assertIn("input_ids", item)
            self.assertIn("attention_mask", item)
            self.assertIn("teacher_topk_values", item)
            self.assertIn("teacher_topk_indices", item)
            self.assertIn("response_mask", item)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_run_kd_cli.py::TestTeacherDataLoading -v`
Expected: FAIL with `AttributeError: module 'run_kd' has no attribute 'load_teacher_config'`

- [ ] **Step 3: Implement teacher data loading functions**

Add to `run_kd.py`:

```python
from safetensors.torch import load_file as load_safetensors_file
from transformers import AutoTokenizer


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

    def __init__(self, completions, teacher_data_dir, top_k, max_length=2048):
        self.completions = completions
        self.top_k = top_k
        self.max_length = max_length

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
        seq_len = self.all_seq_lengths[idx]

        input_ids = completion_ids[: self.max_length]
        actual_len = len(input_ids)
        attention_mask = [1] * actual_len

        # response_mask: all tokens are response tokens (completion only)
        response_mask = [1] * actual_len

        topk_values = self.all_topk_values[idx][:actual_len]
        topk_indices = self.all_topk_indices[idx][:actual_len]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "teacher_topk_values": topk_values,
            "teacher_topk_indices": topk_indices,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_run_kd_cli.py::TestTeacherDataLoading -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add run_kd.py tests/test_run_kd_cli.py
git commit -m "feat: add teacher data loading and dataset classes for run_kd.py"
```

---

### Task 5: `run_kd.py` — KL loss computation

**Files:**
- Modify: `run_kd.py`
- Modify: `tests/test_run_kd_cli.py`

- [ ] **Step 1: Write tests for KL loss**

```python
# Add to tests/test_run_kd_cli.py

class TestKLLoss(unittest.TestCase):
    def test_kl_loss_shape(self):
        import run_kd

        # student_logits: [batch, seq_len, vocab_size]
        student_logits = torch.randn(2, 4, 100)
        # teacher top-K
        teacher_topk_values = torch.randn(2, 4, 8)
        teacher_topk_indices = torch.randint(0, 100, (2, 4, 8))
        response_mask = torch.ones(2, 4)

        loss = run_kd.compute_kl_loss(
            student_logits, teacher_topk_values, teacher_topk_indices, response_mask
        )
        self.assertEqual(loss.shape, ())
        self.assertFalse(torch.isnan(loss))

    def test_kl_loss_zero_when_identical(self):
        import run_kd

        # If student logits at teacher's top-K positions match teacher logprobs,
        # KL should be ~0
        teacher_topk_values = torch.tensor([[[2.0, 1.0, 0.5]]])  # [1, 1, 3]
        teacher_topk_indices = torch.tensor([[[0, 1, 2]]])  # [1, 1, 3]

        # Student logits: set positions 0,1,2 to match teacher values
        student_logits = torch.full((1, 1, 100), -1000.0)
        student_logits[0, 0, 0] = 2.0
        student_logits[0, 0, 1] = 1.0
        student_logits[0, 0, 2] = 0.5
        response_mask = torch.ones(1, 1)

        loss = run_kd.compute_kl_loss(
            student_logits, teacher_topk_values, teacher_topk_indices, response_mask
        )
        self.assertAlmostEqual(loss.item(), 0.0, places=4)

    def test_kl_loss_respects_response_mask(self):
        import run_kd

        student_logits = torch.randn(1, 4, 100)
        teacher_topk_values = torch.randn(1, 4, 8)
        teacher_topk_indices = torch.randint(0, 100, (1, 4, 8))

        mask_all = torch.ones(1, 4)
        mask_half = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

        loss_all = run_kd.compute_kl_loss(
            student_logits, teacher_topk_values, teacher_topk_indices, mask_all
        )
        loss_half = run_kd.compute_kl_loss(
            student_logits, teacher_topk_values, teacher_topk_indices, mask_half
        )
        # Different masks should generally give different losses
        # (unless by coincidence, but extremely unlikely with random data)
        self.assertFalse(torch.isnan(loss_all))
        self.assertFalse(torch.isnan(loss_half))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_run_kd_cli.py::TestKLLoss -v`
Expected: FAIL with `AttributeError: module 'run_kd' has no attribute 'compute_kl_loss'`

- [ ] **Step 3: Implement compute_kl_loss**

Add to `run_kd.py`:

```python
import torch.nn.functional as F


def compute_kl_loss(student_logits, teacher_topk_values, teacher_topk_indices, response_mask):
    """
    Compute token-wise KL(student || teacher) over teacher's top-K positions.

    Args:
        student_logits: [batch, seq_len, vocab_size]
        teacher_topk_values: [batch, seq_len, K] - teacher logits (NOT log-probs)
        teacher_topk_indices: [batch, seq_len, K] - token indices for top-K
        response_mask: [batch, seq_len] - 1 for response tokens, 0 for padding

    Returns:
        Scalar KL loss averaged over response tokens.
    """
    batch, seq_len, K = teacher_topk_values.shape

    # Gather student logits at teacher's top-K positions
    # teacher_topk_indices: [batch, seq_len, K]
    student_topk_logits = torch.gather(student_logits, dim=2, index=teacher_topk_indices.long())
    # student_topk_logits: [batch, seq_len, K]

    # Compute log-softmax over K positions for both
    student_log_probs = F.log_softmax(student_topk_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_topk_values.float(), dim=-1)

    # KL(student || teacher) = sum_k student_prob * (log_student_prob - log_teacher_prob)
    # Using F.kl_div: expects input=log_probs, target=log_probs with log_target=True
    # kl_div computes: target * (log_target - input) when log_target=True
    # This gives KL(target || input), so we swap: input=teacher, target=student
    # Actually: KL(student || teacher) = sum student * (log_student - log_teacher)
    # F.kl_div(input=log_teacher, target=log_student, log_target=True) gives KL(student || teacher)
    kl_per_token = F.kl_div(
        teacher_log_probs,
        student_log_probs,
        log_target=True,
        reduction="none",
    ).sum(dim=-1)  # [batch, seq_len]

    # Mask and average
    masked_kl = kl_per_token * response_mask
    num_tokens = response_mask.sum().clamp(min=1)
    return masked_kl.sum() / num_tokens
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_run_kd_cli.py::TestKLLoss -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add run_kd.py tests/test_run_kd_cli.py
git commit -m "feat: add KL divergence loss computation for run_kd.py"
```

---

### Task 6: `run_kd.py` — Checkpoint saving

**Files:**
- Modify: `run_kd.py`
- Modify: `tests/test_run_kd_cli.py`

- [ ] **Step 1: Write tests for checkpoint saving**

```python
# Add to tests/test_run_kd_cli.py

class TestCheckpointSaving(unittest.TestCase):
    def test_save_kd_checkpoint_creates_safetensors(self):
        import run_kd

        model = torch.nn.Linear(4, 2)
        with tempfile.TemporaryDirectory() as tmp:
            run_kd.save_kd_checkpoint(model, tmp, step=1)
            ckpt_dir = os.path.join(tmp, "step=1")
            self.assertTrue(os.path.isdir(ckpt_dir))
            self.assertTrue(os.path.exists(os.path.join(ckpt_dir, "model.safetensors")))

    def test_save_kd_checkpoint_roundtrips_weights(self):
        import run_kd
        from safetensors.torch import load_file

        model = torch.nn.Linear(4, 2, bias=True)
        with tempfile.TemporaryDirectory() as tmp:
            run_kd.save_kd_checkpoint(model, tmp, step=5)
            loaded = load_file(os.path.join(tmp, "step=5", "model.safetensors"))
            torch.testing.assert_close(loaded["weight"], model.weight.data.cpu())
            torch.testing.assert_close(loaded["bias"], model.bias.data.cpu())

    def test_parse_save_steps_with_final(self):
        import run_kd

        steps = run_kd.parse_save_steps("1,10,final", total_steps=50)
        self.assertEqual(steps, {1, 10, 50})

    def test_parse_save_steps_deduplicates(self):
        import run_kd

        steps = run_kd.parse_save_steps("1,10,10,final", total_steps=10)
        self.assertEqual(steps, {1, 10})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_run_kd_cli.py::TestCheckpointSaving -v`
Expected: FAIL with `AttributeError: module 'run_kd' has no attribute 'save_kd_checkpoint'`

- [ ] **Step 3: Implement save_kd_checkpoint**

Add to `run_kd.py`:

```python
def save_kd_checkpoint(model, output_dir, step):
    """Save model checkpoint in step={N}/model.safetensors format."""
    ckpt_dir = os.path.join(output_dir, f"step={step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    state_dict = {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
    }
    save_safetensors_file(state_dict, os.path.join(ckpt_dir, "model.safetensors"))
    print(f"Saved checkpoint to {ckpt_dir}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_run_kd_cli.py::TestCheckpointSaving -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add run_kd.py tests/test_run_kd_cli.py
git commit -m "feat: add checkpoint saving in step={N}/model.safetensors format for run_kd.py"
```

---

### Task 7: `run_kd.py` — Collate functions and training loop

**Files:**
- Modify: `run_kd.py`

- [ ] **Step 1: Implement collate functions for SFT and KL datasets**

Add to `run_kd.py`:

```python
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
```

- [ ] **Step 2: Implement `main()` training loop**

Add to `run_kd.py`:

```python
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
        val_dataset = KDKlDataset(val_completions, args.teacher_data_dir, args.top_k)
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

    gradient_accumulation_steps = args.gradient_accumulation_steps
    device = model.device
    num_training_steps = compute_num_training_steps(
        num_batches=len(train_dataloader),
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    warmup_steps = resolve_warmup_steps(args.warmup_ratio, num_training_steps)

    save_steps = parse_save_steps(args.save_steps, num_training_steps)

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

    # Eval function
    def eval_model(step=None):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                if args.kd_loss_type == "sft":
                    loss = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    ).loss
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
                total_loss += loss.item()
        val_loss = total_loss / max(1, len(val_dataloader))
        print(f"val_loss: {val_loss:.4f} at step {step}")

        if use_wandb and step is not None:
            wandb.log({"val/loss": val_loss}, step=step)

        model.train()
        return val_loss

    # Training loop
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

                # Checkpoint saving
                if args.enable_save_ckpt and global_step in save_steps:
                    save_kd_checkpoint(model, args.output_dir, global_step)

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

    # Save final checkpoint if "final" was in save_steps and not already saved
    if args.enable_save_ckpt and num_training_steps in save_steps and global_step not in save_steps:
        save_kd_checkpoint(model, args.output_dir, global_step)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify syntax**

Run: `python -m py_compile run_kd.py`
Expected: No output (success)

- [ ] **Step 4: Commit**

```bash
git add run_kd.py
git commit -m "feat: add training loop, collate functions, and main() to run_kd.py"
```

---

### Task 8: Integration test — `run_kd.py` end-to-end with mocked model

**Files:**
- Modify: `tests/test_run_kd_cli.py`

- [ ] **Step 1: Write integration test**

```python
# Add to tests/test_run_kd_cli.py
from unittest.mock import patch, Mock


class _DummyModel(torch.nn.Module):
    def __init__(self, vocab_size=100):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(4, 4))
        self._vocab_size = vocab_size
        self.device = torch.device("cpu")

    def forward(self, input_ids, attention_mask=None, labels=None):
        batch, seq_len = input_ids.shape
        logits = torch.randn(batch, seq_len, self._vocab_size)

        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, self._vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        return type("Out", (), {"logits": logits, "loss": loss})()

    def named_parameters(self, recurse=True):
        yield "weight", self.weight

    def parameters(self, recurse=True):
        yield self.weight

    def train(self, mode=True):
        return self

    def eval(self):
        return self

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)


class TestRunKdIntegration(unittest.TestCase):
    def _create_teacher_data(self, tmp_dir, num_examples=8, top_k=8, max_tokens=16):
        """Create minimal teacher data directory for testing."""
        config = {
            "teacher_model_id": "test/teacher",
            "dataset": "test/data",
            "dataset_split": "train[:8]",
            "top_k": top_k,
            "max_tokens": max_tokens,
            "shared_vocab_size": 100,
            "num_examples": num_examples,
            "prompt_template": "boxed.prompt",
            "temperature": 0,
        }
        with open(os.path.join(tmp_dir, "config.json"), "w") as f:
            json.dump(config, f)

        with open(os.path.join(tmp_dir, "completions.jsonl"), "w") as f:
            for i in range(num_examples):
                entry = {
                    "index": i,
                    "question": f"Q{i}",
                    "ground_truth": str(i),
                    "prompt": f"prompt {i}",
                    "completion": f"answer {i}",
                    "token_ids": list(range(max_tokens)),
                }
                f.write(json.dumps(entry) + "\n")

        logits_dir = os.path.join(tmp_dir, "logits")
        os.makedirs(logits_dir)
        save_safetensors_file(
            {
                "topk_values": torch.randn(num_examples, max_tokens, top_k, dtype=torch.bfloat16),
                "topk_indices": torch.randint(0, 100, (num_examples, max_tokens, top_k), dtype=torch.int32),
                "seq_lengths": torch.full((num_examples,), max_tokens, dtype=torch.int32),
            },
            os.path.join(logits_dir, "chunk_0.safetensors"),
        )

    def test_sft_mode_runs(self):
        import run_kd

        with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as out_dir:
            self._create_teacher_data(data_dir)
            model = _DummyModel(vocab_size=100)

            argv = [
                "--kd-loss-type", "sft",
                "--train-mode", "full",
                "--teacher-data-dir", data_dir,
                "--student-model-id", "test/student",
                "--output-dir", out_dir,
                "--batch-size", "2",
                "--gradient-accumulation-steps", "1",
                "--num-epochs", "1",
                "--no-wandb",
                "--enable-save-ckpt",
                "--save-steps", "1,final",
            ]

            with (
                patch("run_kd.prepare_model", return_value=(
                    model,
                    list(model.parameters()),
                    list(model.named_parameters()),
                    {"wandb_extra": {}, "print_lines": []},
                )),
                patch("run_kd.AutoTokenizer.from_pretrained", return_value=type(
                    "Tok", (), {"pad_token_id": 0, "eos_token_id": 0}
                )()),
            ):
                run_kd.main(argv)

            # Check that step=1 checkpoint was saved
            self.assertTrue(os.path.isdir(os.path.join(out_dir, "step=1")))

    def test_kl_mode_runs(self):
        import run_kd

        with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as out_dir:
            self._create_teacher_data(data_dir, top_k=8)
            model = _DummyModel(vocab_size=100)

            argv = [
                "--kd-loss-type", "kl",
                "--train-mode", "full",
                "--teacher-data-dir", data_dir,
                "--student-model-id", "test/student",
                "--output-dir", out_dir,
                "--top-k", "8",
                "--batch-size", "2",
                "--gradient-accumulation-steps", "1",
                "--num-epochs", "1",
                "--no-wandb",
            ]

            with (
                patch("run_kd.prepare_model", return_value=(
                    model,
                    list(model.parameters()),
                    list(model.named_parameters()),
                    {"wandb_extra": {}, "print_lines": []},
                )),
                patch("run_kd.AutoTokenizer.from_pretrained", return_value=type(
                    "Tok", (), {"pad_token_id": 0, "eos_token_id": 0}
                )()),
            ):
                run_kd.main(argv)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it passes**

Run: `python -m unittest tests/test_run_kd_cli.py::TestRunKdIntegration -v`
Expected: Both tests PASS

- [ ] **Step 3: Run all KD tests together**

Run: `python -m unittest tests/test_run_kd_cli.py -v`
Expected: All tests PASS

- [ ] **Step 4: Run all existing tests to ensure no regressions**

Run: `python -m unittest tests/test_run_rl_cli.py tests/test_btt_pipeline_compat.py tests/test_svd_pipeline_compat.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_run_kd_cli.py
git commit -m "test: add integration tests for run_kd.py SFT and KL modes"
```

---

### Task 9: Final syntax check and full test run

**Files:** (no new files)

- [ ] **Step 1: Syntax check all new files**

Run: `python -m py_compile generate_teacher_data.py && python -m py_compile run_kd.py`
Expected: No output (success)

- [ ] **Step 2: Run all tests**

Run: `python -m unittest tests/test_generate_teacher_data.py tests/test_run_kd_cli.py -v`
Expected: All tests PASS

- [ ] **Step 3: Verify checkpoint compatibility with plot_singular_relationship.py**

This is a manual verification step. The checkpoint format `step={N}/model.safetensors` with named parameters (including `svd_a`, `svd_b`, `svd_s`, `btt_l`, `btt_r`, `btt_s`) matches what `analysis/plot_singular_relationship.py:resolve_step_model_path()` expects (line 170: `run_dir / f"step={step}" / "model.safetensors"`). The `load_tensor_from_safetensor()` function (line 176) loads individual tensors by key name, which matches our `save_kd_checkpoint()` output.

- [ ] **Step 4: Final commit if any fixes were needed**

```bash
git add -u
git commit -m "fix: address any issues found during final verification"
```
