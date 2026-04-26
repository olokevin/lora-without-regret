import copy
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
class TestQBTTForward(unittest.TestCase):
    def _make_pair(self, layout):
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
        btt_ref = copy.deepcopy(model.q_proj)
        qbtt = quantize_frozen_core_(model.q_proj, layout=layout)
        return btt_ref, qbtt

    def _check_forward(self, layout):
        btt, qbtt = self._make_pair(layout)
        torch.manual_seed(1)
        x = torch.randn(4, 8, 64, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            y_ref = btt(x)
            y_q = qbtt(x)
        self.assertEqual(y_ref.shape, y_q.shape)
        rel_err = (
            (y_ref.float() - y_q.float()).norm()
            / y_ref.float().norm().clamp_min(1e-12)
        )
        self.assertLess(rel_err.item(), 0.15)

    def test_forward_flat(self):
        self._check_forward("flat")

    def test_forward_per_core_block(self):
        self._check_forward("per_core_block")


if __name__ == "__main__":
    unittest.main()
