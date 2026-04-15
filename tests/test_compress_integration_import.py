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
            "load_calibrated_btt_for_eval",
        ]
        for n in names:
            self.assertTrue(hasattr(ci, n), f"missing {n}")


if __name__ == "__main__":
    unittest.main()
