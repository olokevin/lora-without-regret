import unittest

import torch
import torch.nn as nn

from btt_layer import (
    BTTLayer,
    QBTTLayer,
    configure_blocktt_trainability,
    convert_btt_to_qbtt_,
    convert_linear_to_btt,
    quantize_frozen_core_,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for bnb NF4")
class TestQBTTConversionFlat(unittest.TestCase):
    def _make_btt(self):
        torch.manual_seed(0)
        linear = nn.Linear(64, 64, bias=True).cuda().to(torch.bfloat16)
        model = nn.Module()
        model.add_module("q_proj", linear)
        convert_linear_to_btt(
            model,
            btt_rank="full",
            decomp_mode="square",
            include_names=("q_proj",),
            train_position="small",
            s_merged_to="frozen",
        )
        configure_blocktt_trainability(model, train_position="small")
        return model.q_proj

    def test_flat_layout_replaces_btt_with_qbtt(self):
        btt = self._make_btt()
        self.assertIsInstance(btt, BTTLayer)
        qbtt = quantize_frozen_core_(btt, layout="flat")
        self.assertIsInstance(qbtt, QBTTLayer)
        # Frozen side must have been removed from the module's parameter list.
        # Trainable side retains its btt_* metadata for the Muon optimizer.
        trainable = qbtt.btt_r if qbtt._qfura_frozen_side == "btt_l" else qbtt.btt_l
        self.assertTrue(trainable.requires_grad)
        self.assertTrue(hasattr(trainable, "btt_layout"))
        self.assertEqual(trainable.btt_layout, "new")
        # Flat frozen core must be registered as a parameter so model.to() and
        # state_dict() work correctly.
        self.assertIn("_qfura_frozen_flat", dict(qbtt.named_parameters()))

    def test_flat_layout_dequant_roundtrip_within_tolerance(self):
        btt = self._make_btt()
        frozen_side = "btt_l" if not btt.btt_l.requires_grad else "btt_r"
        frozen_before = getattr(btt, frozen_side).detach().clone()
        qbtt = quantize_frozen_core_(btt, layout="flat")
        frozen_after = qbtt._dequantize_frozen_core()
        self.assertEqual(frozen_after.shape, frozen_before.shape)
        rel_err = (
            (frozen_after.float() - frozen_before.float()).norm()
            / frozen_before.float().norm().clamp_min(1e-12)
        )
        self.assertLess(rel_err.item(), 0.15)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for bnb NF4")
class TestQBTTConversionPerCoreBlock(unittest.TestCase):
    def _make_btt(self):
        torch.manual_seed(0)
        linear = nn.Linear(64, 64, bias=True).cuda().to(torch.bfloat16)
        model = nn.Module()
        model.add_module("q_proj", linear)
        convert_linear_to_btt(
            model,
            btt_rank="full",
            decomp_mode="square",
            include_names=("q_proj",),
            train_position="small",
            s_merged_to="frozen",
        )
        configure_blocktt_trainability(model, train_position="small")
        return model.q_proj

    def test_per_core_block_produces_one_params4bit_per_outer_block(self):
        btt = self._make_btt()
        frozen_side = "btt_l" if not btt.btt_l.requires_grad else "btt_r"
        frozen_outer = getattr(btt, frozen_side).shape[0]
        qbtt = quantize_frozen_core_(btt, layout="per_core_block")
        self.assertEqual(len(qbtt._qfura_frozen_blocks), frozen_outer)

    def test_per_core_block_dequant_roundtrip_within_tolerance(self):
        btt = self._make_btt()
        frozen_side = "btt_l" if not btt.btt_l.requires_grad else "btt_r"
        frozen_before = getattr(btt, frozen_side).detach().clone()
        qbtt = quantize_frozen_core_(btt, layout="per_core_block")
        frozen_after = qbtt._dequantize_frozen_core()
        self.assertEqual(frozen_after.shape, frozen_before.shape)
        rel_err = (
            (frozen_after.float() - frozen_before.float()).norm()
            / frozen_before.float().norm().clamp_min(1e-12)
        )
        self.assertLess(rel_err.item(), 0.15)

    def test_per_core_block_registers_each_block_as_parameter(self):
        # Per the Task 2 fix-up commit, per_core_block blocks must be
        # registered via register_parameter so they appear in named_parameters
        # and get moved by model.to(device).
        btt = self._make_btt()
        qbtt = quantize_frozen_core_(btt, layout="per_core_block")
        param_names = set(dict(qbtt.named_parameters()).keys())
        n_blocks = len(qbtt._qfura_frozen_blocks)
        for i in range(n_blocks):
            self.assertIn(f"_qfura_frozen_block_{i}", param_names)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for bnb NF4")
class TestQBTTConversionModelWide(unittest.TestCase):
    def test_convert_btt_to_qbtt_counts_all_layers(self):
        torch.manual_seed(0)

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64, bias=False)
                self.k_proj = nn.Linear(64, 64, bias=False)

        model = Block().cuda().to(torch.bfloat16)
        convert_linear_to_btt(
            model,
            btt_rank="full",
            decomp_mode="square",
            include_names=("q_proj", "k_proj"),
            train_position="small",
            s_merged_to="frozen",
        )
        configure_blocktt_trainability(model, train_position="small")
        stats = convert_btt_to_qbtt_(model, layout="flat")
        self.assertEqual(stats["num_converted"], 2)
        self.assertGreater(stats["bytes_saved"], 0)
        self.assertIsInstance(model.q_proj, QBTTLayer)
        self.assertIsInstance(model.k_proj, QBTTLayer)


if __name__ == "__main__":
    unittest.main()
