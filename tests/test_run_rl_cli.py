import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch


class _TinyDataset:
    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._rows[key]
        if isinstance(key, slice):
            rows = self._rows[key]
            return {
                "prompt": [r["prompt"] for r in rows],
                "answer": [r["answer"] for r in rows],
            }
        if isinstance(key, list):
            rows = [self._rows[i] for i in key]
            return {
                "prompt": [r["prompt"] for r in rows],
                "answer": [r["answer"] for r in rows],
            }
        raise TypeError(f"Unsupported key type: {type(key)!r}")


class _DummyTokenizer:
    pad_token_id = 0

    def encode(self, _text):
        return [1, 2, 3]

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "tokenizer.json"), "w", encoding="utf-8") as f:
            f.write("{}")


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1, 1))

    def to(self, _device):
        return self

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros(
            batch_size,
            seq_len,
            8,
            device=input_ids.device,
            dtype=torch.float32,
        )
        logits[..., 1] = 1.0
        return type("_Out", (), {"logits": logits})()

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save({"weight": self.weight.detach().cpu()}, os.path.join(path, "model.pt"))


def _build_fake_generators(args, _model):
    def generate_for_train(prompts, _step):
        return [r"\boxed{1}"] * (len(prompts) * args.group_size)

    def generate_for_eval(prompts, _step):
        return [r"\boxed{1}"] * len(prompts)

    return generate_for_train, generate_for_eval


class TestRunRLCli(unittest.TestCase):
    def test_mode_defaults_applied(self):
        import run_rl

        args = run_rl.parse_args(["--train-mode", "full"])
        run_rl.apply_mode_defaults(args)
        self.assertEqual(args.lr, 1e-5)
        self.assertEqual(args.wandb_project, "math_grpo_full")

        args = run_rl.parse_args(["--train-mode", "lora"])
        run_rl.apply_mode_defaults(args)
        self.assertEqual(args.lr, 9e-5)
        self.assertEqual(args.wandb_project, "math-grpo")

        args = run_rl.parse_args(["--train-mode", "lora_full"])
        run_rl.apply_mode_defaults(args)
        self.assertEqual(args.lr, 9e-5)
        self.assertEqual(args.wandb_project, "math-grpo-lora-full")

        args = run_rl.parse_args(["--train-mode", "blocktt"])
        run_rl.apply_mode_defaults(args)
        self.assertEqual(args.lr, 9e-5)
        self.assertEqual(args.wandb_project, "math-grpo-blocktt")

        args = run_rl.parse_args(["--train-mode", "svd"])
        run_rl.apply_mode_defaults(args)
        self.assertEqual(args.lr, 9e-5)
        self.assertEqual(args.wandb_project, "math-grpo-svd")

    def test_reject_lora_flags_for_non_lora_mode(self):
        import run_rl

        argv = ["--train-mode", "full", "--lora-rank", "4"]
        args = run_rl.parse_args(argv)
        with self.assertRaises(ValueError):
            run_rl.validate_mode_specific_flags(args, argv)

    def test_accept_lora_flags_for_lora_full_mode(self):
        import run_rl

        argv = ["--train-mode", "lora_full", "--lora-rank", "4", "--vllm-url", "http://localhost:8000"]
        args = run_rl.parse_args(argv)
        run_rl.validate_mode_specific_flags(args, argv)

    def test_reject_blocktt_flags_for_non_blocktt_mode(self):
        import run_rl

        argv = ["--train-mode", "lora", "--decomp-mode", "input_one_block"]
        args = run_rl.parse_args(argv)
        with self.assertRaises(ValueError):
            run_rl.validate_mode_specific_flags(args, argv)

        argv = ["--train-mode", "full", "--blocktt-normalize-after-update"]
        args = run_rl.parse_args(argv)
        with self.assertRaises(ValueError):
            run_rl.validate_mode_specific_flags(args, argv)

    def test_reject_trainable_type_for_full_mode(self):
        import run_rl

        argv = ["--train-mode", "full", "--trainable-type", "mlp"]
        args = run_rl.parse_args(argv)
        with self.assertRaises(ValueError):
            run_rl.validate_mode_specific_flags(args, argv)

    def test_reject_lora_and_blocktt_flags_for_svd_mode(self):
        import run_rl

        argv = ["--train-mode", "svd", "--lora-rank", "4"]
        args = run_rl.parse_args(argv)
        with self.assertRaises(ValueError):
            run_rl.validate_mode_specific_flags(args, argv)

        argv = ["--train-mode", "svd", "--decomp-mode", "input_one_block"]
        args = run_rl.parse_args(argv)
        with self.assertRaises(ValueError):
            run_rl.validate_mode_specific_flags(args, argv)

    def test_blocktt_decomp_mode_accepts_json_dict(self):
        import run_rl

        argv = [
            "--train-mode",
            "blocktt",
            "--decomp-mode",
            '{"qkv":"input","o":"output","mlp_upgate":"output","mlp_down":"output"}',
        ]
        args = run_rl.parse_args(argv)
        run_rl.validate_mode_specific_flags(args, argv)

        self.assertIsInstance(args.decomp_mode, dict)
        self.assertEqual(args.decomp_mode["qkv"], "input_one_block")
        self.assertEqual(args.decomp_mode["o"], "output_one_block")
        self.assertEqual(args.blocktt_module_decomp_modes["v_proj"], "input_one_block")
        self.assertEqual(args.blocktt_module_decomp_modes["o_proj"], "output_one_block")
        self.assertEqual(args.blocktt_module_decomp_modes["up_proj"], "output_one_block")

    def test_blocktt_decomp_mode_accepts_python_like_dict(self):
        import run_rl

        argv = [
            "--train-mode",
            "blocktt",
            "--decomp-mode",
            "{qkv: input, o: output, mlp_upgate: output, mlp_down: output}",
        ]
        args = run_rl.parse_args(argv)
        run_rl.validate_mode_specific_flags(args, argv)
        self.assertEqual(args.decomp_mode["qkv"], "input_one_block")
        self.assertEqual(args.decomp_mode["o"], "output_one_block")

    def test_blocktt_decomp_mode_dict_missing_keys_fall_back_to_input(self):
        import run_rl

        argv = ["--train-mode", "blocktt", "--decomp-mode", '{"o":"output"}']
        args = run_rl.parse_args(argv)
        run_rl.validate_mode_specific_flags(args, argv)
        self.assertEqual(args.decomp_mode["qkv"], "input_one_block")
        self.assertEqual(args.decomp_mode["o"], "output_one_block")
        self.assertEqual(args.decomp_mode["mlp_upgate"], "input_one_block")
        self.assertEqual(args.decomp_mode["mlp_down"], "input_one_block")

    def test_blocktt_decomp_mode_rejects_invalid_dict_key(self):
        import run_rl

        argv = ["--train-mode", "blocktt", "--decomp-mode", '{"qkvo":"input"}']
        args = run_rl.parse_args(argv)
        with self.assertRaises(ValueError):
            run_rl.validate_mode_specific_flags(args, argv)

    def test_blocktt_decomp_mode_rejects_invalid_value(self):
        import run_rl

        argv = ["--train-mode", "blocktt", "--decomp-mode", '{"qkv":"diag"}']
        args = run_rl.parse_args(argv)
        with self.assertRaises(ValueError):
            run_rl.validate_mode_specific_flags(args, argv)

    def test_blocktt_normalize_after_update_flag_parses_for_blocktt(self):
        import run_rl

        argv = ["--train-mode", "blocktt", "--blocktt-normalize-after-update"]
        args = run_rl.parse_args(argv)
        run_rl.validate_mode_specific_flags(args, argv)
        self.assertTrue(args.blocktt_normalize_after_update)

    def test_train_position_defaults(self):
        import run_rl

        args = run_rl.parse_args(["--train-mode", "blocktt"])
        run_rl.apply_mode_defaults(args)
        self.assertEqual(args.train_position, "small")
        self.assertEqual(args.s_merged_to, "frozen")

        args = run_rl.parse_args(["--train-mode", "svd"])
        run_rl.apply_mode_defaults(args)
        self.assertEqual(args.train_position, "output")
        self.assertEqual(args.s_merged_to, "frozen")

    def test_train_position_mode_validation(self):
        import run_rl

        for mode in ["full", "lora", "lora_full"]:
            argv = ["--train-mode", mode, "--train-position", "small"]
            args = run_rl.parse_args(argv)
            with self.assertRaises(ValueError):
                run_rl.validate_mode_specific_flags(args, argv)

        argv = ["--train-mode", "blocktt", "--train-position", "output"]
        args = run_rl.parse_args(argv)
        with self.assertRaises(ValueError):
            run_rl.validate_mode_specific_flags(args, argv)

        argv = ["--train-mode", "svd", "--train-position", "small"]
        args = run_rl.parse_args(argv)
        with self.assertRaises(ValueError):
            run_rl.validate_mode_specific_flags(args, argv)

    def test_accept_train_position_values_by_mode(self):
        import run_rl

        for pos in ["small", "large", "both"]:
            argv = ["--train-mode", "blocktt", "--train-position", pos]
            args = run_rl.parse_args(argv)
            run_rl.validate_mode_specific_flags(args, argv)

        for pos in ["output", "input"]:
            argv = ["--train-mode", "svd", "--train-position", pos]
            args = run_rl.parse_args(argv)
            run_rl.validate_mode_specific_flags(args, argv)

    def test_s_merged_to_mode_validation(self):
        import run_rl

        for mode in ["full", "lora", "lora_full"]:
            argv = ["--train-mode", mode, "--s-merged-to", "split"]
            args = run_rl.parse_args(argv)
            with self.assertRaises(ValueError):
                run_rl.validate_mode_specific_flags(args, argv)

        for value in ["frozen", "trainable", "output", "input", "split", "keep"]:
            argv = ["--train-mode", "svd", "--s-merged-to", value]
            args = run_rl.parse_args(argv)
            run_rl.validate_mode_specific_flags(args, argv)

            argv = ["--train-mode", "blocktt", "--s-merged-to", value]
            args = run_rl.parse_args(argv)
            run_rl.validate_mode_specific_flags(args, argv)

    def test_s_merged_to_reject_aliases_for_blocktt_both(self):
        import run_rl

        for value in ["frozen", "trainable"]:
            argv = [
                "--train-mode",
                "blocktt",
                "--train-position",
                "both",
                "--s-merged-to",
                value,
            ]
            args = run_rl.parse_args(argv)
            with self.assertRaises(ValueError):
                run_rl.validate_mode_specific_flags(args, argv)

    def test_s_merged_to_default_is_split_for_blocktt_both(self):
        import run_rl

        args = run_rl.parse_args(["--train-mode", "blocktt", "--train-position", "both"])
        run_rl.apply_mode_defaults(args)
        self.assertEqual(args.s_merged_to, "split")

    def test_accept_trainable_type_for_trainable_modes(self):
        import run_rl

        for mode in ["lora", "lora_full", "blocktt", "svd"]:
            argv = ["--train-mode", mode, "--trainable-type", "attn"]
            args = run_rl.parse_args(argv)
            run_rl.validate_mode_specific_flags(args, argv)

    def test_old_type_flags_are_rejected(self):
        import run_rl

        with self.assertRaises(SystemExit):
            run_rl.parse_args(["--train-mode", "lora", "--lora-type", "all"])
        with self.assertRaises(SystemExit):
            run_rl.parse_args(["--train-mode", "blocktt", "--blocktt-type", "all"])
        with self.assertRaises(SystemExit):
            run_rl.parse_args(["--train-mode", "svd", "--svd-type", "all"])

    def test_vllm_http_available_true_on_ok_response(self):
        import run_rl

        resp = Mock()
        resp.ok = True
        with patch("run_rl.requests.get", return_value=resp):
            self.assertTrue(run_rl.is_vllm_http_available("http://localhost:8000"))

    def test_vllm_http_available_false_on_non_ok_response(self):
        import run_rl

        resp = Mock()
        resp.ok = False
        with patch("run_rl.requests.get", return_value=resp):
            self.assertFalse(run_rl.is_vllm_http_available("http://localhost:8000"))

    def test_vllm_http_available_false_on_request_exception(self):
        import run_rl

        with patch(
            "run_rl.requests.get",
            side_effect=run_rl.requests.RequestException("down"),
        ):
            self.assertFalse(run_rl.is_vllm_http_available("http://localhost:8000"))

    def test_resolve_lora_rollout_backend_forces_local_for_lora_full(self):
        import run_rl

        with patch("run_rl.is_vllm_http_available", return_value=True):
            self.assertEqual(
                run_rl.resolve_lora_rollout_backend("lora_full", "http://localhost:8000"),
                "local_inproc",
            )

    def test_resolve_lora_rollout_backend_prefers_http_for_lora(self):
        import run_rl

        with patch("run_rl.is_vllm_http_available", return_value=True):
            self.assertEqual(
                run_rl.resolve_lora_rollout_backend("lora", "http://localhost:8000"),
                "http",
            )

    def test_normalize_lora_merged_weight_name_rewrites_base_layer(self):
        import run_rl

        self.assertEqual(
            run_rl.normalize_lora_merged_weight_name(
                "layers.0.self_attn.qkv_proj.base_layer.weight"
            ),
            "layers.0.self_attn.qkv_proj.weight",
        )

    def test_normalize_lora_merged_weight_name_drops_lora_tensors(self):
        import run_rl

        self.assertIsNone(
            run_rl.normalize_lora_merged_weight_name(
                "layers.0.self_attn.qkv_proj.lora_A.default.weight"
            )
        )
        self.assertIsNone(
            run_rl.normalize_lora_merged_weight_name(
                "layers.0.self_attn.qkv_proj.lora_B.default.weight"
            )
        )

    def test_first_step_grad_dump_flags_parse(self):
        import run_rl

        args = run_rl.parse_args(
            [
                "--train-mode",
                "svd",
                "--save-first-step-grads-path",
                "tmp/first_step_grads.safetensors",
                "--save-first-step-grads-prefixes",
                "model.layers.0.self_attn.q_proj",
                "--stop-after-first-step",
            ]
        )
        self.assertEqual(
            args.save_first_step_grads_path, "tmp/first_step_grads.safetensors"
        )
        self.assertEqual(
            args.save_first_step_grads_prefixes,
            "model.layers.0.self_attn.q_proj",
        )
        self.assertTrue(args.stop_after_first_step)


class TestRunRLCheckpointing(unittest.TestCase):
    def _run_main(self, base_dir, *, enable_save_ckpt):
        import run_rl

        rows = [{"prompt": "q", "answer": "1"} for _ in range(2)]
        train_dataset = _TinyDataset(rows)
        val_dataset = _TinyDataset(rows)
        tokenizer = _DummyTokenizer()
        model = _DummyModel()

        argv = [
            "--train-mode",
            "full",
            "--model-id",
            "dummy-model",
            "--base-dir",
            base_dir,
            "--no-wandb",
            "--n-grpo-steps",
            "2",
            "--n-prompts-per-step",
            "1",
            "--group-size",
            "1",
            "--epochs-per-step",
            "0",
            "--micro-batch-size",
            "1",
            "--gradient-accumulation-steps",
            "1",
        ]
        if enable_save_ckpt:
            argv.append("--enable-save-ckpt")

        with (
            patch(
                "run_rl.load_datasets_and_tokenizer",
                return_value=(train_dataset, val_dataset, tokenizer),
            ),
            patch(
                "run_rl.AutoModelForCausalLM.from_pretrained",
                return_value=model,
            ),
            patch(
                "run_rl.build_local_vllm_generators",
                side_effect=_build_fake_generators,
            ),
        ):
            run_rl.main(argv)

    def _single_run_dir(self, base_dir):
        full_dir = os.path.join(base_dir, "full")
        entries = os.listdir(full_dir)
        self.assertEqual(len(entries), 1)
        return os.path.join(full_dir, entries[0])

    def test_full_mode_enable_save_ckpt_writes_final(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._run_main(tmp, enable_save_ckpt=True)
            run_dir = self._single_run_dir(tmp)
            # With 2 GRPO steps, only the final step (step=2) is saved (save at 10, 30, final)
            self.assertFalse(os.path.exists(os.path.join(run_dir, "step=1")))
            self.assertTrue(os.path.isdir(os.path.join(run_dir, "step=2")))

    def test_full_mode_without_enable_save_ckpt_writes_no_step_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._run_main(tmp, enable_save_ckpt=False)
            run_dir = self._single_run_dir(tmp)
            self.assertFalse(os.path.exists(os.path.join(run_dir, "step=1")))
            self.assertFalse(os.path.exists(os.path.join(run_dir, "step=2")))


class _DapoDummyTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        assert add_generation_prompt is True
        return " || ".join(f"{m['role']}:{m['content']}" for m in messages)

    def encode(self, _text):
        return [1, 2, 3]


class TestRunRLDapoHelpers(unittest.TestCase):
    def test_dapo_parse_defaults(self):
        import run_rl_dapo

        args = run_rl_dapo.parse_args(["--train-mode", "full"])
        run_rl_dapo.apply_mode_defaults(args)
        self.assertEqual(args.loss_type, "grpo")
        self.assertEqual(args.train_dataset_id, run_rl_dapo.DEFAULT_TRAIN_DATASET_ID)
        self.assertTrue(args.final_eval)
        self.assertEqual(args.eval_majority_k, 32)

    def test_dapo_loss_clipping(self):
        import run_rl_dapo

        ratio = torch.tensor([[1.5, 0.7]], dtype=torch.float32)
        adv = torch.tensor([1.0], dtype=torch.float32)
        mask = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
        loss, clip_frac = run_rl_dapo.compute_policy_loss(
            "dapo",
            ratio,
            adv,
            mask,
            clip_ratio_low=0.2,
            clip_ratio_high=0.1,
        )
        self.assertAlmostEqual(loss.item(), -0.9, places=5)
        self.assertAlmostEqual(clip_frac, 1.0, places=5)

    def test_unique_by_problem(self):
        import run_rl_dapo

        rows = [
            {"problem": "A", "answer": "1"},
            {"problem": "A", "answer": "1"},
            {"problem": "B", "answer": "2"},
        ]
        unique = run_rl_dapo.unique_by_problem(rows)
        self.assertEqual(len(unique), 2)

    def test_process_math_example_dapo_schema(self):
        import run_rl_dapo

        tokenizer = _DapoDummyTokenizer()
        example = {
            "prompt": [{"role": "user", "content": "2+2?"}],
            "reward_model": {"ground_truth": r"\boxed{4}"},
        }
        processed = run_rl_dapo.process_math_example(example, tokenizer, "{question}")
        self.assertIn("user:2+2?", processed["prompt"])
        self.assertEqual(processed["answer"], "4")


if __name__ == "__main__":
    unittest.main()
