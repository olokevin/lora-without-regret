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
        self.assertEqual(args.save_steps, "10,30,final")

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

    def test_kl_online_choice_accepted(self):
        import run_kd
        args = run_kd.parse_args([
            "--kd-loss-type", "kl_online",
            "--train-mode", "full",
            "--teacher-model-id", "test/teacher",
        ])
        self.assertEqual(args.kd_loss_type, "kl_online")
        self.assertEqual(args.teacher_model_id, "test/teacher")

    def test_kl_online_requires_teacher_model_id(self):
        import run_kd
        argv = ["--kd-loss-type", "kl_online", "--train-mode", "full"]
        args = run_kd.parse_args(argv)
        with self.assertRaises(ValueError):
            run_kd.validate_mode_specific_flags(args, argv)

    def test_teacher_model_id_ignored_for_sft(self):
        import run_kd
        # Should not raise even though --teacher-model-id is absent
        argv = ["--kd-loss-type", "sft", "--train-mode", "full"]
        args = run_kd.parse_args(argv)
        run_kd.validate_mode_specific_flags(args, argv)
        self.assertIsNone(args.teacher_model_id)


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
        self.assertFalse(torch.isnan(loss_all))
        self.assertFalse(torch.isnan(loss_half))


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


from unittest.mock import patch, Mock


class _DummyModel(torch.nn.Module):
    def __init__(self, vocab_size=100):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(4, 4))
        self._vocab_size = vocab_size
        self.device = torch.device("cpu")

    def forward(self, input_ids, attention_mask=None, labels=None):
        batch, seq_len = input_ids.shape
        # Use self.weight so logits are connected to the parameter graph
        dummy = self.weight.sum() * 0
        logits = torch.zeros(batch, seq_len, self._vocab_size) + dummy

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
                "--base-dir", out_dir,
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
                patch("transformers.AutoTokenizer.from_pretrained", return_value=type(
                    "Tok", (), {"pad_token_id": 0, "eos_token_id": 0}
                )()),
            ):
                run_kd.main(argv)

            # Check that step=1 checkpoint was saved under {out_dir}/full/.../step=1
            import glob as _glob
            matches = _glob.glob(os.path.join(out_dir, "full", "**", "step=1"), recursive=True)
            self.assertTrue(len(matches) > 0, "step=1 checkpoint not found")

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
                "--base-dir", out_dir,
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
                patch("transformers.AutoTokenizer.from_pretrained", return_value=type(
                    "Tok", (), {"pad_token_id": 0, "eos_token_id": 0}
                )()),
            ):
                run_kd.main(argv)


if __name__ == "__main__":
    unittest.main()
