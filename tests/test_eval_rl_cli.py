import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


class TestEvalRlCli(unittest.TestCase):
    def test_checkpoint_required(self):
        import eval_rl

        with self.assertRaises(SystemExit):
            eval_rl.parse_args([])

    def test_defaults(self):
        import eval_rl

        args = eval_rl.parse_args(["--checkpoint", "Qwen/Qwen3-1.7B"])
        self.assertEqual(
            args.math_verify_datasets,
            ["MATH-500", "AIME-24", "AIME-25", "AMC23", "Minerva"],
        )
        self.assertEqual(args.math_verify_max_tokens, 2048)
        self.assertEqual(args.prompt_template, "boxed.prompt")
        self.assertEqual(args.max_model_len, 2048)


class TestEvalRlPreflight(unittest.TestCase):
    def test_adapter_only_dir_rejected(self):
        import eval_rl

        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "adapter_config.json"), "w") as f:
                f.write("{}")
            with self.assertRaises(ValueError) as cm:
                eval_rl.preflight_checkpoint(d)
            self.assertIn("adapter", str(cm.exception).lower())

    def test_factored_btt_dir_rejected(self):
        import eval_rl

        from safetensors.torch import save_file
        import torch

        with tempfile.TemporaryDirectory() as d:
            save_file(
                {"layer.btt_l": torch.zeros(2, 2), "layer.btt_r": torch.zeros(2, 2)},
                os.path.join(d, "model.safetensors"),
            )
            with self.assertRaises(ValueError) as cm:
                eval_rl.preflight_checkpoint(d)
            self.assertIn("factored", str(cm.exception).lower())

    def test_factored_svd_dir_rejected(self):
        import eval_rl

        from safetensors.torch import save_file
        import torch

        with tempfile.TemporaryDirectory() as d:
            save_file(
                {"layer.svd_a": torch.zeros(2, 2), "layer.svd_b": torch.zeros(2, 2)},
                os.path.join(d, "model.safetensors"),
            )
            with self.assertRaises(ValueError):
                eval_rl.preflight_checkpoint(d)

    def test_plain_hf_dir_passes(self):
        import eval_rl

        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "config.json"), "w") as f:
                f.write("{}")
            # No exception expected.
            eval_rl.preflight_checkpoint(d)

    def test_hf_id_passes(self):
        import eval_rl

        # Anything not a local directory just passes (will fail later in
        # from_pretrained if invalid, but pre-flight is happy).
        eval_rl.preflight_checkpoint("Qwen/Qwen3-1.7B")


if __name__ == "__main__":
    unittest.main()
