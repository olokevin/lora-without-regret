import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


class _Tokenizer:
    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "tokenizer.json"), "w", encoding="utf-8") as f:
            f.write("{}")


class TestSaveMergedCheckpointPeftFamily(unittest.TestCase):
    """dora/pissa/milora/randlora must reuse the lora merge_adapter path."""

    def _run(self, train_mode):
        import run_rl
        base = MagicMock()
        model = MagicMock()
        model.get_base_model.return_value = base
        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "step=1")
            args = MagicMock()
            args.calib_mode = "none"
            args.enable_merged_ckpt = True
            run_rl.save_merged_checkpoint(model, _Tokenizer(), ckpt, train_mode, args)
            model.merge_adapter.assert_called_once()
            if train_mode == "randlora":
                # randlora_A is shared across layers; safetensors refuses shared
                # storages, so we must save via torch.save.
                base.save_pretrained.assert_called_once_with(
                    ckpt, safe_serialization=False
                )
            else:
                base.save_pretrained.assert_called_once_with(ckpt)
            model.unmerge_adapter.assert_called_once()
            model.save_pretrained.assert_not_called()
            self.assertTrue(os.path.exists(os.path.join(ckpt, "tokenizer.json")))

    def test_dora(self):
        self._run("dora")

    def test_pissa(self):
        self._run("pissa")

    def test_milora(self):
        self._run("milora")

    def test_randlora(self):
        self._run("randlora")

    def test_unmerges_even_if_save_raises(self):
        import run_rl
        base = MagicMock()
        base.save_pretrained.side_effect = RuntimeError("disk full")
        model = MagicMock()
        model.get_base_model.return_value = base
        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "step=1")
            args = MagicMock()
            args.calib_mode = "none"
            args.enable_merged_ckpt = True
            with self.assertRaises(RuntimeError):
                run_rl.save_merged_checkpoint(model, _Tokenizer(), ckpt, "dora", args)
            model.merge_adapter.assert_called_once()
            model.unmerge_adapter.assert_called_once()


class TestSaveMergedCheckpointLift(unittest.TestCase):
    """lift uses the dense full-style save path."""

    def test_lift_calls_save_pretrained(self):
        import run_rl
        model = MagicMock()
        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "step=1")
            args = MagicMock()
            args.calib_mode = "none"
            args.enable_merged_ckpt = True
            run_rl.save_merged_checkpoint(model, _Tokenizer(), ckpt, "lift", args)
            model.save_pretrained.assert_called_once_with(ckpt)
            model.merge_adapter.assert_not_called()
            self.assertTrue(os.path.exists(os.path.join(ckpt, "tokenizer.json")))


class TestResolveLoraRolloutBackend(unittest.TestCase):
    def test_pissa_forced_local(self):
        import run_rl
        self.assertEqual(
            run_rl.resolve_lora_rollout_backend("pissa", "http://localhost:8000"),
            "local_inproc",
        )

    def test_milora_forced_local(self):
        import run_rl
        self.assertEqual(
            run_rl.resolve_lora_rollout_backend("milora", "http://localhost:8000"),
            "local_inproc",
        )

    def test_dora_can_use_local_fallback(self):
        import run_rl
        with patch.object(run_rl, "is_vllm_http_available", return_value=False):
            self.assertEqual(
                run_rl.resolve_lora_rollout_backend("dora", "http://localhost:8000"),
                "local_inproc",
            )

    def test_randlora_can_use_local_fallback(self):
        import run_rl
        with patch.object(run_rl, "is_vllm_http_available", return_value=False):
            self.assertEqual(
                run_rl.resolve_lora_rollout_backend("randlora", "http://localhost:8000"),
                "local_inproc",
            )

    def test_lift_returns_none(self):
        import run_rl
        self.assertIsNone(
            run_rl.resolve_lora_rollout_backend("lift", "http://localhost:8000"),
        )


class TestNormalizeLoraMergedWeightName(unittest.TestCase):
    def test_skips_randlora_lambda(self):
        import run_rl
        self.assertIsNone(
            run_rl.normalize_lora_merged_weight_name("foo.randlora_lambda")
        )

    def test_skips_randlora_gamma(self):
        import run_rl
        self.assertIsNone(
            run_rl.normalize_lora_merged_weight_name("foo.randlora_gamma")
        )

    def test_skips_randlora_m(self):
        import run_rl
        self.assertIsNone(
            run_rl.normalize_lora_merged_weight_name("foo.randlora_m")
        )

    def test_skips_dora_magnitude_vector(self):
        import run_rl
        self.assertIsNone(
            run_rl.normalize_lora_merged_weight_name(
                "foo.lora_magnitude_vector.default.weight"
            )
        )

    def test_passes_normal_param_through(self):
        import run_rl
        self.assertEqual(
            run_rl.normalize_lora_merged_weight_name("model.layers.0.weight"),
            "model.layers.0.weight",
        )


if __name__ == "__main__":
    unittest.main()
