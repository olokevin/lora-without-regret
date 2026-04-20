import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import torch


class _Tokenizer:
    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "tokenizer.json"), "w", encoding="utf-8") as f:
            f.write("{}")


class TestSaveMergedCheckpointFull(unittest.TestCase):
    def test_full_mode_calls_save_pretrained(self):
        import run_rl

        model = MagicMock()
        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "step=1")
            args = MagicMock()
            args.calib_mode = "none"
            args.enable_merged_ckpt = True
            run_rl.save_merged_checkpoint(model, _Tokenizer(), ckpt, "full", args)
            model.save_pretrained.assert_called_once_with(ckpt)
            self.assertTrue(os.path.exists(os.path.join(ckpt, "tokenizer.json")))


class TestSaveMergedCheckpointLora(unittest.TestCase):
    def test_lora_calls_merge_then_base_save_then_unmerge(self):
        import run_rl

        base = MagicMock()
        model = MagicMock()
        model.get_base_model.return_value = base

        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "step=1")
            args = MagicMock()
            args.calib_mode = "none"
            args.enable_merged_ckpt = True
            run_rl.save_merged_checkpoint(model, _Tokenizer(), ckpt, "lora", args)

            model.merge_adapter.assert_called_once()
            base.save_pretrained.assert_called_once_with(ckpt)
            model.unmerge_adapter.assert_called_once()
            model.save_pretrained.assert_not_called()

    def test_lora_unmerges_even_if_save_raises(self):
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
                run_rl.save_merged_checkpoint(model, _Tokenizer(), ckpt, "lora", args)
            model.merge_adapter.assert_called_once()
            model.unmerge_adapter.assert_called_once()

    def test_lora_full_mode_uses_same_path(self):
        import run_rl

        base = MagicMock()
        model = MagicMock()
        model.get_base_model.return_value = base

        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "step=1")
            args = MagicMock()
            args.calib_mode = "none"
            args.enable_merged_ckpt = True
            run_rl.save_merged_checkpoint(model, _Tokenizer(), ckpt, "lora_full", args)

            model.merge_adapter.assert_called_once()
            base.save_pretrained.assert_called_once_with(ckpt)
            model.unmerge_adapter.assert_called_once()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for BTT conversion")
class TestBuildFactoredDenseStateDict(unittest.TestCase):
    """Exercises the factored->dense conversion using real BTTLayer instances."""

    def _make_btt_module(self):
        import torch.nn as nn

        torch.manual_seed(0)
        in_features, out_features = 8, 8

        class _Wrap(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(in_features, out_features, bias=True)

        wrap = _Wrap().cuda()

        from btt_layer import convert_linear_to_btt

        convert_linear_to_btt(
            wrap,
            btt_rank="full",
            decomp_mode="input_one_block",
            init_mode="default",
            include_names=("linear",),
            skip_names=(),
            lr_act=False,
            s_merged_to="frozen",
            train_position="small",
            factorize_by_head=False,
            model_config=None,
        )
        return wrap

    def test_btt_factored_state_dict_matches_dense(self):
        import run_rl

        model = self._make_btt_module()

        with torch.no_grad():
            expected = model.linear.materialize_dense_weight().detach().clone()

        sd = run_rl._build_factored_dense_state_dict(model)

        # No factored core keys leaked into the new state_dict.
        for k in sd.keys():
            self.assertNotIn(".btt_l", k)
            self.assertNotIn(".btt_r", k)
            self.assertNotIn(".btt_s", k)

        # The dense weight in the state_dict matches what the BTTLayer currently materializes.
        self.assertIn("linear.weight", sd)
        torch.testing.assert_close(
            sd["linear.weight"].float(), expected.float(), atol=1e-5, rtol=1e-5
        )

        # Model object is unchanged: the BTTLayer is still in place.
        from btt_layer import BTTLayer
        self.assertIsInstance(model.linear, BTTLayer)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for SVD conversion")
class TestBuildFactoredDenseStateDictSVD(unittest.TestCase):
    def test_svd_factored_state_dict_matches_dense(self):
        import torch.nn as nn
        from svd_layer import SVDLayer, convert_linear_to_svd

        torch.manual_seed(0)
        in_features, out_features = 8, 8

        class _Wrap(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(in_features, out_features, bias=True)

            def save_pretrained(self, path, state_dict=None):
                os.makedirs(path, exist_ok=True)
                from safetensors.torch import save_file

                save_file(state_dict or self.state_dict(), os.path.join(path, "model.safetensors"))
                with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
                    f.write("{}")

        wrap = _Wrap().cuda()

        convert_linear_to_svd(
            wrap,
            include_names=("linear",),
            skip_names=(),
            s_merged_to="frozen",
            train_position="output",
        )

        with torch.no_grad():
            expected = wrap.linear.materialize_dense_weight().detach().clone()

        import run_rl

        sd = run_rl._build_factored_dense_state_dict(wrap)
        for k in sd.keys():
            self.assertNotIn(".svd_a", k)
            self.assertNotIn(".svd_b", k)
            self.assertNotIn(".svd_s", k)

        self.assertIn("linear.weight", sd)
        torch.testing.assert_close(
            sd["linear.weight"].float(), expected.float(), atol=1e-5, rtol=1e-5
        )
        self.assertIsInstance(wrap.linear, SVDLayer)


if __name__ == "__main__":
    unittest.main()
