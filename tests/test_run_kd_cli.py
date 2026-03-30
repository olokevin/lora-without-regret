# tests/test_run_kd_cli.py
import unittest
import os
import json
import tempfile

import torch
from safetensors.torch import save_file as save_safetensors_file


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
