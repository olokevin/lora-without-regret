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


FORWARD_THRESHOLDS = {"flat": 0.11, "per_core_block": 0.11}  # calibrated 2026-04-26
BACKWARD_THRESHOLD = 0.13  # calibrated 2026-04-26


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for bnb NF4")
class TestQBTTQuantError(unittest.TestCase):
    def _build_reference(self):
        torch.manual_seed(0)
        linear = nn.Linear(4096, 4096, bias=False).cuda().to(torch.bfloat16)
        with torch.no_grad():
            linear.weight.normal_(mean=0.0, std=0.02)
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

    def _check_layout(self, layout):
        ref = self._build_reference()
        cloned = copy.deepcopy(ref)
        q = quantize_frozen_core_(cloned, layout=layout)

        torch.manual_seed(1)
        x = torch.randn(4, 128, 4096, device="cuda", dtype=torch.bfloat16)
        target = torch.randn(4, 128, 4096, device="cuda", dtype=torch.bfloat16)

        y_ref = ref(x)
        y_q = q(x)
        fwd_err = (
            (y_ref.float() - y_q.float()).norm()
            / y_ref.float().norm().clamp_min(1e-12)
        ).item()

        def grad_of(module):
            for p in module.parameters():
                if p.grad is not None:
                    p.grad = None
            loss = (module(x) - target).pow(2).mean()
            loss.backward()
            trainable_params = [p for p in module.parameters() if p.requires_grad]
            return [p.grad.detach().clone() for p in trainable_params]

        g_ref = grad_of(ref)
        g_q = grad_of(q)

        bwd_errs = []
        for a, b in zip(g_ref, g_q):
            denom = a.float().norm().clamp_min(1e-12)
            bwd_errs.append(((a.float() - b.float()).norm() / denom).item())
        bwd_err = max(bwd_errs)

        self.assertLess(
            fwd_err,
            FORWARD_THRESHOLDS[layout],
            f"{layout}: forward error {fwd_err:.4f} exceeds {FORWARD_THRESHOLDS[layout]}",
        )
        self.assertLess(
            bwd_err,
            BACKWARD_THRESHOLD,
            f"{layout}: backward error {bwd_err:.4f} exceeds {BACKWARD_THRESHOLD}",
        )

    def test_quant_error_flat(self):
        self._check_layout("flat")

    def test_quant_error_per_core_block(self):
        self._check_layout("per_core_block")


if __name__ == "__main__":
    unittest.main()
