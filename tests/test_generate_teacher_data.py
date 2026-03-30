# tests/test_generate_teacher_data.py
import json
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


if __name__ == "__main__":
    unittest.main()
