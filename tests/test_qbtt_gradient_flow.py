import unittest

import torch
import torch.nn as nn

from btt_layer import (
    QBTTLayer,
    configure_blocktt_trainability,
    convert_linear_to_btt,
    quantize_frozen_core_,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for bnb NF4")
class TestQBTTGradientFlow(unittest.TestCase):
    def _make_qbtt(self, s_merged_to, layout):
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
            s_merged_to=s_merged_to,
        )
        configure_blocktt_trainability(
            model,
            train_position="small",
            train_singular_values=(s_merged_to == "keep_trainable"),
        )
        return quantize_frozen_core_(model.q_proj, layout=layout)

    def test_only_trainable_core_and_btt_s_receive_gradients(self):
        qbtt = self._make_qbtt(s_merged_to="keep_trainable", layout="flat")
        x = torch.randn(2, 8, 64, device="cuda", dtype=torch.bfloat16, requires_grad=False)
        y = qbtt(x)
        loss = y.pow(2).mean()
        loss.backward()

        # Trainable core should have a grad.
        trainable = qbtt.btt_r if qbtt._qfura_frozen_side == "btt_l" else qbtt.btt_l
        self.assertIsNotNone(trainable.grad)
        self.assertEqual(trainable.grad.shape, trainable.shape)

        # btt_s is trainable under keep_trainable; should also have grad.
        self.assertIsNotNone(qbtt.btt_s)
        self.assertIsNotNone(qbtt.btt_s.grad)

        # Bias should have a grad (default train_bias=True).
        self.assertIsNotNone(qbtt.bias.grad)

    def test_frozen_params4bit_has_no_grad(self):
        qbtt = self._make_qbtt(s_merged_to="frozen", layout="flat")
        # The frozen Params4bit is stored as _qfura_frozen_flat; requires_grad=False.
        self.assertFalse(qbtt._qfura_frozen_flat.requires_grad)


if __name__ == "__main__":
    unittest.main()
