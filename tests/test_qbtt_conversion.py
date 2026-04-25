import unittest

import torch
import torch.nn as nn

from btt_layer import (
    BTTLayer,
    QBTTLayer,
    configure_blocktt_trainability,
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


if __name__ == "__main__":
    unittest.main()
