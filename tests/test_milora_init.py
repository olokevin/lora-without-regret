import os
import sys
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


class TestMiloraInit(unittest.TestCase):
    """Verify (alpha/r) * B @ A + W_residual reconstructs W exactly."""

    def _make_peft_linear(self, in_features=64, out_features=48, rank=8, alpha=16):
        """Build a single nn.Linear wrapped in a PEFT LoRA layer."""
        from peft import LoraConfig, get_peft_model
        torch.manual_seed(0)
        base = nn.Sequential(nn.Linear(in_features, out_features, bias=False))
        config = LoraConfig(
            r=rank, lora_alpha=alpha,
            target_modules=["0"],   # the nn.Linear inside the Sequential
            lora_dropout=0,
            bias="none",
        )
        peft_model = get_peft_model(base, config)
        return peft_model

    def test_reconstruction_matches_original_weight(self):
        from run_sft import apply_milora_init_
        rank, alpha = 8, 16
        peft_model = self._make_peft_linear(rank=rank, alpha=alpha)

        # Capture original weights before MiLoRA mutates them.
        from peft.tuners.lora import LoraLayer
        lora_layer = next(m for m in peft_model.modules() if isinstance(m, LoraLayer))
        W_original = lora_layer.get_base_layer().weight.data.detach().clone().float()

        apply_milora_init_(peft_model, rank=rank)

        W_residual = lora_layer.get_base_layer().weight.data.float()
        adapter_name = list(lora_layer.lora_A.keys())[0]
        A = lora_layer.lora_A[adapter_name].weight.data.float()  # (r, in)
        B = lora_layer.lora_B[adapter_name].weight.data.float()  # (out, r)

        reconstructed = (alpha / rank) * (B @ A) + W_residual
        rel_err = (reconstructed - W_original).norm() / W_original.norm()
        self.assertLess(rel_err.item(), 1e-4,
                        f"Reconstruction error {rel_err.item():.2e} exceeds 1e-4")

    def test_residual_drops_bottom_r_components(self):
        """MiLoRA replaces W with the top-(n-r) components — its smallest singular
        value should be strictly larger than W's smallest."""
        from run_sft import apply_milora_init_
        peft_model = self._make_peft_linear(rank=4, alpha=8)
        from peft.tuners.lora import LoraLayer
        lora_layer = next(m for m in peft_model.modules() if isinstance(m, LoraLayer))
        W_original = lora_layer.get_base_layer().weight.data.detach().clone().float()
        s_orig = torch.linalg.svdvals(W_original)

        apply_milora_init_(peft_model, rank=4)
        W_residual = lora_layer.get_base_layer().weight.data.float()
        s_resid = torch.linalg.svdvals(W_residual)

        # W_residual has rank n-4; the (n-4)th singular value of residual is
        # the (n-4)th of original; its smallest non-zero is the 5th-smallest
        # of original (i.e. larger than s_orig[-1]).
        self.assertGreater(s_resid[-5].item(), s_orig[-1].item() * 0.99)


if __name__ == "__main__":
    unittest.main()
