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
