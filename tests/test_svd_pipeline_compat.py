import unittest

import torch
import torch.nn as nn

from svd_layer import SVDLayer, configure_svd_trainability, convert_linear_to_svd


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(16, 16, bias=True)
        self.other = nn.Linear(16, 16, bias=True)


class TestSVDPipelineCompat(unittest.TestCase):
    def test_convert_linear_to_svd_targets_requested_modules(self):
        model = _ToyModel()
        original_weight_params = model.q_proj.weight.numel()
        converted = convert_linear_to_svd(
            model,
            include_names=("q_proj",),
            skip_names=("lm_head",),
        )
        self.assertIn("q_proj", converted)
        self.assertIsInstance(model.q_proj, SVDLayer)
        self.assertIsInstance(model.other, nn.Linear)
        self.assertEqual(model.q_proj.svd_a.numel() + model.q_proj.svd_b.numel(), 2 * original_weight_params)

    def test_materialized_dense_weight_matches_svd_forward(self):
        torch.manual_seed(0)
        layer = SVDLayer(in_features=12, out_features=18, bias=True)
        x = torch.randn(4, 12)

        y_svd = layer(x)
        dense_weight = layer.materialize_dense_weight()
        y_dense = torch.nn.functional.linear(x, dense_weight, layer.bias)

        self.assertTrue(torch.allclose(y_svd, y_dense, atol=1e-5, rtol=1e-5))

    def test_configure_trainability_output_side(self):
        model = _ToyModel()
        convert_linear_to_svd(model, include_names=("q_proj",))
        stats = configure_svd_trainability(model, train_position="output", train_bias=True)

        self.assertEqual(stats["num_svd_layers"], 1)
        self.assertTrue(model.q_proj.svd_a.requires_grad)
        self.assertFalse(model.q_proj.svd_b.requires_grad)
        self.assertTrue(model.q_proj.bias.requires_grad)

    def test_configure_trainability_input_side(self):
        model = _ToyModel()
        convert_linear_to_svd(model, include_names=("q_proj",))
        stats = configure_svd_trainability(model, train_position="input", train_bias=True)

        self.assertEqual(stats["num_svd_layers"], 1)
        self.assertFalse(model.q_proj.svd_a.requires_grad)
        self.assertTrue(model.q_proj.svd_b.requires_grad)
        self.assertTrue(model.q_proj.bias.requires_grad)

    def test_init_s_merged_to_alias_matches_concrete_side(self):
        torch.manual_seed(0)
        weight = torch.randn(18, 12)

        layer_alias = SVDLayer(in_features=12, out_features=18, bias=False)
        layer_alias.init_from_linear_weight(
            weight,
            s_merged_to="frozen",
            train_position="output",
        )

        layer_concrete = SVDLayer(in_features=12, out_features=18, bias=False)
        layer_concrete.init_from_linear_weight(
            weight,
            s_merged_to="input",
            train_position="output",
        )

        self.assertTrue(torch.allclose(layer_alias.svd_a, layer_concrete.svd_a))
        self.assertTrue(torch.allclose(layer_alias.svd_b, layer_concrete.svd_b))

    def test_init_s_merged_to_default_uses_frozen_side_for_single_trainable(self):
        torch.manual_seed(0)
        weight = torch.randn(18, 12)

        layer_default = SVDLayer(in_features=12, out_features=18, bias=False)
        layer_default.init_from_linear_weight(
            weight,
            train_position="output",
        )

        layer_expected = SVDLayer(in_features=12, out_features=18, bias=False)
        layer_expected.init_from_linear_weight(
            weight,
            s_merged_to="frozen",
            train_position="output",
        )

        self.assertTrue(torch.allclose(layer_default.svd_a, layer_expected.svd_a))
        self.assertTrue(torch.allclose(layer_default.svd_b, layer_expected.svd_b))

    def test_keep_preserves_weight_and_keeps_s_frozen(self):
        torch.manual_seed(0)
        model = _ToyModel()
        original_weight = model.q_proj.weight.detach().clone()

        convert_linear_to_svd(
            model,
            include_names=("q_proj",),
            s_merged_to="keep",
            train_position="output",
        )
        self.assertIsNotNone(model.q_proj.svd_s)
        self.assertTrue(
            torch.allclose(
                model.q_proj.materialize_dense_weight(),
                original_weight,
                atol=1e-5,
                rtol=1e-5,
            )
        )

        configure_svd_trainability(model, train_position="output", train_bias=False)
        self.assertFalse(model.q_proj.svd_s.requires_grad)


if __name__ == "__main__":
    unittest.main()
