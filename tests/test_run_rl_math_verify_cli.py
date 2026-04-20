import io
import os
import sys
import unittest
from contextlib import redirect_stderr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


class TestRunRlMathVerifyCli(unittest.TestCase):
    def test_defaults_are_on(self):
        import run_rl

        args = run_rl.parse_args(["--train-mode", "full"])
        self.assertTrue(args.enable_merged_ckpt)
        self.assertTrue(args.enable_math_verify)
        self.assertEqual(
            args.math_verify_datasets,
            ["MATH-500", "AIME-24", "AIME-25", "AMC23", "Minerva"],
        )
        self.assertIsNone(args.math_verify_n_samples)
        self.assertIsNone(args.math_verify_temperature)
        self.assertEqual(args.math_verify_max_tokens, 2048)

    def test_no_flags_disable_features(self):
        import run_rl

        args = run_rl.parse_args(
            ["--train-mode", "full", "--no-enable-merged-ckpt", "--no-enable-math-verify"]
        )
        self.assertFalse(args.enable_merged_ckpt)
        self.assertFalse(args.enable_math_verify)

    def test_math_verify_datasets_parses_csv(self):
        import run_rl

        args = run_rl.parse_args(
            ["--train-mode", "full", "--math-verify-datasets", "MATH-500,AIME-24"]
        )
        self.assertEqual(args.math_verify_datasets, ["MATH-500", "AIME-24"])

    def test_math_verify_datasets_rejects_unknown_name(self):
        import run_rl

        argv = ["--train-mode", "full", "--math-verify-datasets", "BOGUS,MATH-500"]
        args = run_rl.parse_args(argv)
        with self.assertRaises(ValueError) as cm:
            run_rl.validate_mode_specific_flags(args, argv)
        self.assertIn("BOGUS", str(cm.exception))
        self.assertIn("MATH-500", str(cm.exception))  # known names cited

    def test_math_verify_n_samples_zero_rejected(self):
        import run_rl

        argv = ["--train-mode", "full", "--math-verify-n-samples", "0"]
        args = run_rl.parse_args(argv)
        with self.assertRaises(ValueError):
            run_rl.validate_mode_specific_flags(args, argv)

    def test_no_merge_with_eval_emits_warning(self):
        import run_rl

        argv = [
            "--train-mode",
            "blocktt",
            "--no-enable-merged-ckpt",
            "--enable-math-verify",
        ]
        args = run_rl.parse_args(argv)

        buf = io.StringIO()
        with redirect_stderr(buf):
            # validate should not raise; warning goes to stderr.
            run_rl.validate_mode_specific_flags(args, argv)

        self.assertIn("--no-enable-merged-ckpt", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
