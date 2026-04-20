import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


class TestRunRlCliFlagValidation(unittest.TestCase):
    """--train-mode-specific flag validation in run_rl.py."""

    def _parse(self, argv):
        import run_rl
        args = run_rl.parse_args(argv)
        run_rl.apply_mode_defaults(args)
        run_rl.validate_mode_specific_flags(args, argv)
        return args

    # Each new mode parses with --no-wandb (smoke).
    def test_dora_parses(self):
        args = self._parse(["--train-mode", "dora", "--lora-rank", "8", "--no-wandb"])
        self.assertEqual(args.train_mode, "dora")
        self.assertEqual(args.lora_rank, 8)

    def test_pissa_parses(self):
        args = self._parse(["--train-mode", "pissa", "--lora-rank", "8", "--no-wandb"])
        self.assertEqual(args.train_mode, "pissa")

    def test_milora_parses(self):
        args = self._parse(["--train-mode", "milora", "--lora-rank", "8", "--no-wandb"])
        self.assertEqual(args.train_mode, "milora")

    def test_randlora_parses_with_prng_key(self):
        args = self._parse([
            "--train-mode", "randlora", "--lora-rank", "8",
            "--randlora-projection-prng-key", "42", "--no-wandb",
        ])
        self.assertEqual(args.randlora_projection_prng_key, 42)

    def test_lift_parses_with_lift_flags(self):
        args = self._parse([
            "--train-mode", "lift",
            "--lift-lora-rank", "64",
            "--lift-filter-rank", "64",
            "--lift-update-interval", "200",
            "--no-wandb",
        ])
        self.assertEqual(args.lift_lora_rank, 64)
        self.assertEqual(args.lift_filter_rank, 64)
        self.assertEqual(args.lift_update_interval, 200)

    # Mode-incompatible flags raise.
    def test_pissa_rejects_vllm_url(self):
        with self.assertRaises(ValueError) as cm:
            self._parse([
                "--train-mode", "pissa", "--lora-rank", "8",
                "--vllm-url", "http://localhost:8000", "--no-wandb",
            ])
        self.assertIn("vllm-url", str(cm.exception).lower())

    def test_milora_rejects_vllm_url(self):
        with self.assertRaises(ValueError) as cm:
            self._parse([
                "--train-mode", "milora", "--lora-rank", "8",
                "--vllm-url", "http://localhost:8000", "--no-wandb",
            ])
        self.assertIn("vllm-url", str(cm.exception).lower())

    def test_lora_rejects_lift_flags(self):
        with self.assertRaises(ValueError) as cm:
            self._parse([
                "--train-mode", "lora", "--lora-rank", "8",
                "--lift-lora-rank", "64", "--no-wandb",
            ])
        self.assertIn("lift", str(cm.exception).lower())

    def test_lora_rejects_randlora_prng_key(self):
        with self.assertRaises(ValueError) as cm:
            self._parse([
                "--train-mode", "lora", "--lora-rank", "8",
                "--randlora-projection-prng-key", "7", "--no-wandb",
            ])
        self.assertIn("randlora", str(cm.exception).lower())

    def test_lift_rejects_muon(self):
        with self.assertRaises(ValueError) as cm:
            self._parse([
                "--train-mode", "lift", "--optimizer", "muon", "--no-wandb",
            ])
        self.assertIn("muon", str(cm.exception).lower())

    def test_blocktt_still_works(self):
        # Regression: existing modes unaffected.
        args = self._parse([
            "--train-mode", "blocktt", "--decomp-mode", "input_one_block",
            "--train-position", "small", "--no-wandb",
        ])
        self.assertEqual(args.train_mode, "blocktt")


class TestRunSftCliFlagValidation(unittest.TestCase):
    """run_sft.py mirrors but does not have --vllm-url."""

    def _parse(self, argv):
        import run_sft
        args = run_sft.parse_args(argv)
        run_sft.apply_mode_defaults(args)
        if hasattr(run_sft, "validate_mode_specific_flags"):
            run_sft.validate_mode_specific_flags(args, argv)
        return args

    def test_lift_parses(self):
        args = self._parse([
            "--train-mode", "lift",
            "--lift-lora-rank", "64",
            "--no-wandb",
        ])
        self.assertEqual(args.train_mode, "lift")

    def test_milora_parses(self):
        args = self._parse([
            "--train-mode", "milora", "--lora-rank", "8", "--no-wandb",
        ])
        self.assertEqual(args.train_mode, "milora")


if __name__ == "__main__":
    unittest.main()
