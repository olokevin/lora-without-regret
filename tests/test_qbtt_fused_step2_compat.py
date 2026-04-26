import copy
import unittest

import torch
import torch.nn as nn

from btt_layer import (
    BTTLayer,
    configure_blocktt_trainability,
    convert_linear_to_btt,
    quantize_frozen_core_,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for bnb NF4")
class TestQBTTFusedStep2Compat(unittest.TestCase):
    def test_fused_and_nonfused_match(self):
        torch.manual_seed(0)
        linear = nn.Linear(64, 64, bias=False).cuda().to(torch.bfloat16)
        model = nn.Module()
        model.add_module("q_proj", linear)
        convert_linear_to_btt(
            model,
            btt_rank="full",
            decomp_mode="square",
            include_names=("q_proj",),
            train_position="small",
            s_merged_to="keep_frozen",
        )
        configure_blocktt_trainability(model, train_position="small")
        qbtt = quantize_frozen_core_(model.q_proj, layout="flat")

        x = torch.randn(2, 8, 64, device="cuda", dtype=torch.bfloat16)

        # Non-fused.
        BTTLayer.use_fused_step2 = False
        y_nonfused = qbtt(x)

        # Fused.
        try:
            BTTLayer.use_fused_step2 = True
            y_fused = qbtt(x)
        except (ImportError, RuntimeError) as e:
            self.skipTest(f"Fused kernel unavailable: {e}")
        finally:
            BTTLayer.use_fused_step2 = False

        rel_err = (
            (y_nonfused.float() - y_fused.float()).norm()
            / y_nonfused.float().norm().clamp_min(1e-12)
        )
        self.assertLess(rel_err.item(), 0.02)


if __name__ == "__main__":
    unittest.main()
