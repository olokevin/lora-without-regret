import os, sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class TestCompressIntegrationImport(unittest.TestCase):
    def test_module_imports(self):
        import compress_integration  # noqa: F401

    def test_public_surface_exists(self):
        import compress_integration as ci
        names = [
            "add_calibrated_btt_args",
            "validate_calibrated_btt_args",
            "build_decomposition_config",
            "build_training_data_calib_loader",
            "build_rl_rollout_calib_loader",
            "build_calib_loader",
            "apply_calibrated_btt",
            "materialize_calibrated_btt_weights",
            "restore_calibrated_btt_weights",
            "save_calibrated_btt_checkpoint",
            "save_calibrated_btt_hf_pretrained",
            "load_calibrated_btt_for_eval",
            "materialize_calibrated_btt_to_linear",
        ]
        for n in names:
            self.assertTrue(hasattr(ci, n), f"missing {n}")

    def test_add_calibrated_btt_args_registers_flags(self):
        import argparse
        import compress_integration as ci
        parser = argparse.ArgumentParser()
        ci.add_calibrated_btt_args(parser)
        args = parser.parse_args([
            "--calib-mode", "v2",
            "--calib-source", "traces",
            "--calib-traces-path", "/tmp/x.jsonl",
            "--calib-num-seqs", "16",
            "--calib-max-length", "512",
            "--calib-seed", "7",
            "--calib-batch-size", "4",
        ])
        self.assertEqual(args.calib_mode, "v2")
        self.assertEqual(args.calib_source, "traces")
        self.assertEqual(args.calib_traces_path, "/tmp/x.jsonl")
        self.assertEqual(args.calib_num_seqs, 16)
        self.assertEqual(args.calib_max_length, 512)
        self.assertEqual(args.calib_seed, 7)
        self.assertEqual(args.calib_batch_size, 4)

        # Also verify defaults.
        default_args = argparse.ArgumentParser()
        ci.add_calibrated_btt_args(default_args)
        d = default_args.parse_args([])
        self.assertEqual(d.calib_mode, "none")
        self.assertEqual(d.calib_source, "c4")
        self.assertIsNone(d.calib_traces_path)
        self.assertEqual(d.calib_num_seqs, 128)
        self.assertEqual(d.calib_max_length, 2048)
        self.assertEqual(d.calib_seed, 3)
        self.assertEqual(d.calib_batch_size, 8)


if __name__ == "__main__":
    unittest.main()
