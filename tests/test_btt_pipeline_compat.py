import copy
import unittest

import torch
import torch.nn as nn

from btt_layer import (
    BTTLayer,
    configure_blocktt_trainability,
    convert_linear_to_btt,
    normalize_trainable_blocktt_cores_,
    resolve_blocktt_s_merged_to,
)
from optim.muon import Muon
from legacy.rl_blocktt import materialize_btt_weight


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(16, 16, bias=True)
        self.k_proj = nn.Linear(16, 16, bias=True)
        self.v_proj = nn.Linear(16, 16, bias=True)
        self.o_proj = nn.Linear(16, 16, bias=True)
        self.gate_proj = nn.Linear(16, 16, bias=True)
        self.up_proj = nn.Linear(16, 16, bias=True)
        self.down_proj = nn.Linear(16, 16, bias=True)
        self.other = nn.Linear(16, 16, bias=True)


class TestBTTPipelineCompat(unittest.TestCase):
    def test_convert_linear_to_btt_accepts_legacy_forward_impl_flag(self):
        model = _ToyModel()
        converted = convert_linear_to_btt(
            model,
            btt_rank="full",
            include_names=("q_proj",),
            decomp_mode="input_one_block",
            lr_act=False,
            forward_impl="einsum",
        )
        self.assertIn("q_proj", converted)
        self.assertIsInstance(model.q_proj, BTTLayer)
        self.assertIsInstance(model.other, nn.Linear)

    def test_materialized_dense_weight_matches_btt_forward(self):
        torch.manual_seed(0)
        layer = BTTLayer(
            in_features=12,
            out_features=18,
            rank=3,
            bias=True,
            lr_act=False,
            decomp_mode="square",
        )
        x = torch.randn(4, 12)

        y_btt = layer(x)
        dense_weight = layer.materialize_dense_weight()
        y_dense = torch.nn.functional.linear(x, dense_weight, layer.bias)

        self.assertTrue(torch.allclose(y_btt, y_dense, atol=1e-5, rtol=1e-5))

    def test_rl_blocktt_materialize_uses_canonical_layout(self):
        layer = BTTLayer(
            in_features=16,
            out_features=20,
            rank=2,
            bias=False,
            lr_act=False,
            decomp_mode="output_one_block",
        )
        from_rl = materialize_btt_weight(layer)
        from_layer = layer.materialize_dense_weight()
        self.assertEqual(from_rl.shape, (20, 16))
        self.assertTrue(torch.allclose(from_rl, from_layer, atol=1e-6, rtol=1e-6))

    def test_muon_updates_canonical_btt_params(self):
        torch.manual_seed(0)
        layer = BTTLayer(
            in_features=16,
            out_features=16,
            rank=2,
            bias=False,
            lr_act=False,
            decomp_mode="input_one_block",
        )
        params = [
            ("layer.btt_r", layer.btt_r),
            ("layer.btt_l", layer.btt_l),
        ]
        opt = Muon(params, lr=1e-3, ns_steps=2)

        before_r = layer.btt_r.detach().clone()
        before_l = layer.btt_l.detach().clone()
        layer.btt_r.grad = torch.randn_like(layer.btt_r)
        layer.btt_l.grad = torch.randn_like(layer.btt_l)
        opt.step()

        self.assertFalse(torch.equal(before_r, layer.btt_r))
        self.assertFalse(torch.equal(before_l, layer.btt_l))

    def test_blocktt_train_position_small_large_both(self):
        class _Container(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = BTTLayer(
                    in_features=16,
                    out_features=16,
                    rank=2,
                    bias=False,
                    lr_act=False,
                    decomp_mode="input_one_block",
                )

        model = _Container()
        stats = configure_blocktt_trainability(model, train_position="small", train_bias=False)
        self.assertEqual(stats["num_btt_layers"], 1)
        self.assertTrue(model.layer.btt_l.requires_grad)
        self.assertFalse(model.layer.btt_r.requires_grad)

        stats = configure_blocktt_trainability(model, train_position="large", train_bias=False)
        self.assertEqual(stats["num_btt_layers"], 1)
        self.assertFalse(model.layer.btt_l.requires_grad)
        self.assertTrue(model.layer.btt_r.requires_grad)

        stats = configure_blocktt_trainability(model, train_position="both", train_bias=False)
        self.assertEqual(stats["num_btt_layers"], 1)
        self.assertTrue(model.layer.btt_l.requires_grad)
        self.assertTrue(model.layer.btt_r.requires_grad)

    def test_blocktt_train_position_tie_prefers_left(self):
        class _Container(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = BTTLayer(
                    in_features=16,
                    out_features=16,
                    rank=2,
                    bias=False,
                    lr_act=False,
                    decomp_mode="square",
                )

        model = _Container()
        configure_blocktt_trainability(model, train_position="small", train_bias=False)
        self.assertTrue(model.layer.btt_l.requires_grad)
        self.assertFalse(model.layer.btt_r.requires_grad)

        configure_blocktt_trainability(model, train_position="large", train_bias=False)
        self.assertTrue(model.layer.btt_l.requires_grad)
        self.assertFalse(model.layer.btt_r.requires_grad)

    def test_normalize_trainable_blocktt_cores_output_mode_normalizes_btt_r(self):
        class _Container(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = BTTLayer(
                    in_features=16,
                    out_features=16,
                    rank=2,
                    bias=False,
                    lr_act=False,
                    decomp_mode="output_one_block",
                )

        torch.manual_seed(0)
        model = _Container()
        configure_blocktt_trainability(model, train_position="small", train_bias=False)
        btt_l_before = model.layer.btt_l.detach().clone()

        normalize_trainable_blocktt_cores_(model)

        btt_r_norms = torch.linalg.vector_norm(model.layer.btt_r, dim=1)
        self.assertTrue(torch.allclose(btt_r_norms, torch.ones_like(btt_r_norms), atol=1e-5))
        self.assertTrue(torch.equal(model.layer.btt_l, btt_l_before))

    def test_normalize_trainable_blocktt_cores_input_mode_normalizes_btt_l(self):
        class _Container(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = BTTLayer(
                    in_features=16,
                    out_features=16,
                    rank=2,
                    bias=False,
                    lr_act=False,
                    decomp_mode="input_one_block",
                )

        torch.manual_seed(0)
        model = _Container()
        configure_blocktt_trainability(model, train_position="small", train_bias=False)
        btt_r_before = model.layer.btt_r.detach().clone()

        normalize_trainable_blocktt_cores_(model)

        btt_l_norms = torch.linalg.vector_norm(model.layer.btt_l, dim=2)
        self.assertTrue(torch.allclose(btt_l_norms, torch.ones_like(btt_l_norms), atol=1e-5))
        self.assertTrue(torch.equal(model.layer.btt_r, btt_r_before))

    def test_normalize_trainable_blocktt_cores_both_trainable(self):
        class _Container(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = BTTLayer(
                    in_features=16,
                    out_features=16,
                    rank=2,
                    bias=False,
                    lr_act=False,
                    decomp_mode="input_one_block",
                )

        torch.manual_seed(0)
        model = _Container()
        configure_blocktt_trainability(model, train_position="both", train_bias=False)

        normalize_trainable_blocktt_cores_(model)

        btt_r_norms = torch.linalg.vector_norm(model.layer.btt_r, dim=1)
        btt_l_norms = torch.linalg.vector_norm(model.layer.btt_l, dim=2)
        self.assertTrue(torch.allclose(btt_r_norms, torch.ones_like(btt_r_norms), atol=1e-5))
        self.assertTrue(torch.allclose(btt_l_norms, torch.ones_like(btt_l_norms), atol=1e-5))

    def test_init_s_merged_to_alias_matches_expected_side_for_small_position(self):
        torch.manual_seed(0)
        weight = torch.randn(16, 16)

        # input_one_block with square dims makes left core smaller for this shape.
        layer_alias = BTTLayer(
            in_features=16,
            out_features=16,
            rank=2,
            bias=False,
            lr_act=False,
            decomp_mode="input_one_block",
        )
        layer_alias.init_from_linear_weight(
            weight,
            s_merged_to="frozen",
            train_position="small",
        )

        layer_concrete = BTTLayer(
            in_features=16,
            out_features=16,
            rank=2,
            bias=False,
            lr_act=False,
            decomp_mode="input_one_block",
        )
        layer_concrete.init_from_linear_weight(
            weight,
            s_merged_to="input",
            train_position="small",
        )

        self.assertTrue(torch.allclose(layer_alias.btt_l, layer_concrete.btt_l))
        self.assertTrue(torch.allclose(layer_alias.btt_r, layer_concrete.btt_r))

    def test_init_s_merged_to_default_is_split_for_both_trainable(self):
        torch.manual_seed(0)
        weight = torch.randn(16, 16)

        layer_default = BTTLayer(
            in_features=16,
            out_features=16,
            rank=2,
            bias=False,
            lr_act=False,
            decomp_mode="input_one_block",
        )
        layer_default.init_from_linear_weight(
            weight,
            train_position="both",
        )

        layer_expected = BTTLayer(
            in_features=16,
            out_features=16,
            rank=2,
            bias=False,
            lr_act=False,
            decomp_mode="input_one_block",
        )
        layer_expected.init_from_linear_weight(
            weight,
            s_merged_to="split",
            train_position="both",
        )

        self.assertTrue(torch.allclose(layer_default.btt_l, layer_expected.btt_l))
        self.assertTrue(torch.allclose(layer_default.btt_r, layer_expected.btt_r))

    def test_reject_alias_s_merged_to_when_both_trainable(self):
        layer = BTTLayer(
            in_features=16,
            out_features=16,
            rank=2,
            bias=False,
            lr_act=False,
            decomp_mode="input_one_block",
        )
        weight = torch.randn(16, 16)

        with self.assertRaises(ValueError):
            layer.init_from_linear_weight(
                weight,
                s_merged_to="frozen",
                train_position="both",
            )

    def test_keep_preserves_weight_and_keeps_s_frozen(self):
        torch.manual_seed(0)
        model = _ToyModel()
        original_weight = model.q_proj.weight.detach().clone()

        convert_linear_to_btt(
            model,
            btt_rank="full",
            include_names=("q_proj",),
            decomp_mode="input_one_block",
            lr_act=False,
            s_merged_to="keep",
            train_position="small",
        )
        self.assertIsNotNone(model.q_proj.btt_s)
        self.assertTrue(
            torch.allclose(
                model.q_proj.materialize_dense_weight(),
                original_weight,
                atol=1e-5,
                rtol=1e-5,
            )
        )

        configure_blocktt_trainability(model, train_position="small", train_bias=False)
        self.assertFalse(model.q_proj.btt_s.requires_grad)

    def test_convert_linear_to_btt_supports_per_module_decomp_mode_dict(self):
        model = _ToyModel()
        decomp_mode_by_module = {
            "q_proj": "input_one_block",
            "k_proj": "input_one_block",
            "v_proj": "input_one_block",
            "o_proj": "output_one_block",
            "gate_proj": "output_one_block",
            "up_proj": "output_one_block",
            "down_proj": "output_one_block",
        }
        convert_linear_to_btt(
            model,
            btt_rank="full",
            include_names=tuple(decomp_mode_by_module),
            decomp_mode=decomp_mode_by_module,
            lr_act=False,
        )

        self.assertEqual(model.q_proj.decomp_mode, "input_one_block")
        self.assertEqual(model.v_proj.decomp_mode, "input_one_block")
        self.assertEqual(model.o_proj.decomp_mode, "output_one_block")
        self.assertEqual(model.up_proj.decomp_mode, "output_one_block")
        self.assertIsInstance(model.other, nn.Linear)

    def test_frozen_alias_maps_to_correct_side_with_mixed_decomp_modes(self):
        torch.manual_seed(0)
        model = _ToyModel()
        decomp_mode_by_module = {
            "q_proj": "input_one_block",
            "k_proj": "input_one_block",
            "v_proj": "input_one_block",
            "o_proj": "output_one_block",
            "gate_proj": "output_one_block",
            "up_proj": "output_one_block",
            "down_proj": "output_one_block",
        }
        include_names = tuple(decomp_mode_by_module)
        convert_linear_to_btt(
            model,
            btt_rank="full",
            include_names=include_names,
            decomp_mode=decomp_mode_by_module,
            lr_act=False,
            s_merged_to="frozen",
            train_position="small",
        )

        model_for_trainability = copy.deepcopy(model)
        configure_blocktt_trainability(
            model_for_trainability,
            train_position="small",
            train_bias=False,
        )

        for module_name in include_names:
            layer = getattr(model, module_name)
            layer_for_trainability = getattr(model_for_trainability, module_name)
            merge_target = resolve_blocktt_s_merged_to(
                train_position="small",
                s_merged_to="frozen",
                left_size=layer.btt_l.numel(),
                right_size=layer.btt_r.numel(),
            )
            if layer_for_trainability.btt_l.requires_grad:
                self.assertEqual(merge_target, "input")
            else:
                self.assertEqual(merge_target, "output")


    def test_output_factorization_override(self):
        layer = BTTLayer(
            in_features=64,
            out_features=128,
            rank=2,
            bias=False,
            lr_act=False,
            decomp_mode="square",
            output_factorization=(16, 8),
        )
        self.assertEqual(layer.m, 16)
        self.assertEqual(layer.a, 8)

    def test_input_factorization_override(self):
        layer = BTTLayer(
            in_features=128,
            out_features=64,
            rank=2,
            bias=False,
            lr_act=False,
            decomp_mode="square",
            input_factorization=(16, 8),
        )
        self.assertEqual(layer.n, 16)
        self.assertEqual(layer.b, 8)

    def test_output_factorization_validation(self):
        with self.assertRaises(ValueError):
            BTTLayer(
                in_features=64,
                out_features=128,
                rank=2,
                bias=False,
                lr_act=False,
                decomp_mode="square",
                output_factorization=(10, 10),
            )

    def test_input_factorization_validation(self):
        with self.assertRaises(ValueError):
            BTTLayer(
                in_features=64,
                out_features=128,
                rank=2,
                bias=False,
                lr_act=False,
                decomp_mode="square",
                input_factorization=(10, 10),
            )

    def test_output_one_block_ignores_output_factorization(self):
        layer = BTTLayer(
            in_features=64,
            out_features=128,
            rank=2,
            bias=False,
            lr_act=False,
            decomp_mode="output_one_block",
            output_factorization=(16, 8),
        )
        self.assertEqual(layer.m, 1)
        self.assertEqual(layer.a, 128)

    def test_input_one_block_ignores_input_factorization(self):
        layer = BTTLayer(
            in_features=128,
            out_features=64,
            rank=2,
            bias=False,
            lr_act=False,
            decomp_mode="input_one_block",
            input_factorization=(16, 8),
        )
        self.assertEqual(layer.n, 1)
        self.assertEqual(layer.b, 128)

    def test_factorize_by_head_q_proj(self):
        class _AttnModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 128, bias=False)

        class _FakeConfig:
            num_attention_heads = 16
            num_key_value_heads = 4
            head_dim = 8
            hidden_size = 128

        model = _AttnModel()
        convert_linear_to_btt(
            model,
            btt_rank="full",
            include_names=("q_proj",),
            decomp_mode="square",
            lr_act=False,
            factorize_by_head=True,
            model_config=_FakeConfig(),
        )
        self.assertEqual(model.q_proj.m, 16)
        self.assertEqual(model.q_proj.a, 8)

    def test_factorize_by_head_kv_proj_uses_kv_heads(self):
        class _AttnModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.k_proj = nn.Linear(128, 32, bias=False)

        class _FakeConfig:
            num_attention_heads = 16
            num_key_value_heads = 4
            head_dim = 8
            hidden_size = 128

        model = _AttnModel()
        convert_linear_to_btt(
            model,
            btt_rank="full",
            include_names=("k_proj",),
            decomp_mode="square",
            lr_act=False,
            factorize_by_head=True,
            model_config=_FakeConfig(),
        )
        self.assertEqual(model.k_proj.m, 4)
        self.assertEqual(model.k_proj.a, 8)

    def test_factorize_by_head_o_proj(self):
        class _AttnModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.o_proj = nn.Linear(128, 64, bias=False)

        class _FakeConfig:
            num_attention_heads = 16
            num_key_value_heads = 4
            head_dim = 8
            hidden_size = 128

        model = _AttnModel()
        convert_linear_to_btt(
            model,
            btt_rank="full",
            include_names=("o_proj",),
            decomp_mode="square",
            lr_act=False,
            factorize_by_head=True,
            model_config=_FakeConfig(),
        )
        self.assertEqual(model.o_proj.n, 16)
        self.assertEqual(model.o_proj.b, 8)

    def test_factorize_by_head_disabled_uses_closest_factor(self):
        class _AttnModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 128, bias=False)

        class _FakeConfig:
            num_attention_heads = 16
            num_key_value_heads = 4
            head_dim = 8
            hidden_size = 128

        model = _AttnModel()
        convert_linear_to_btt(
            model,
            btt_rank="full",
            include_names=("q_proj",),
            decomp_mode="square",
            lr_act=False,
            factorize_by_head=False,
            model_config=_FakeConfig(),
        )
        # _closest_factor_pair(128) = (8, 16), not (16, 8)
        self.assertNotEqual(model.q_proj.m, 16)

    def test_factorize_by_head_mlp_unaffected(self):
        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(64, 128, bias=False)

        class _FakeConfig:
            num_attention_heads = 16
            num_key_value_heads = 4
            head_dim = 8
            hidden_size = 128

        model = _Model()
        convert_linear_to_btt(
            model,
            btt_rank="full",
            include_names=("gate_proj",),
            decomp_mode="square",
            lr_act=False,
            factorize_by_head=True,
            model_config=_FakeConfig(),
        )
        # gate_proj should use _closest_factor_pair, not head-based
        from btt_layer import _closest_factor_pair
        expected_m, expected_a = _closest_factor_pair(128)
        self.assertEqual(model.gate_proj.m, expected_m)
        self.assertEqual(model.gate_proj.a, expected_a)


if __name__ == "__main__":
    unittest.main()
