"""Correctness tests for Qdora4bitLinear vs PEFT's Linear4bit + use_dora=True.

Each test gated on CUDA + bnb (4-bit quantization needs both).
"""
import unittest

import torch
import torch.nn as nn

# Tests live at the worktree root via the venv; no sys.path tweaks needed.


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for bnb 4-bit")
class TestQdora4bitLinearForwardEquivalence(unittest.TestCase):
    """Forward output of Qdora4bitLinear (K=1) should match PEFT's
    Linear4bit + use_dora=True within bf16 numerical tolerance, when both
    modules are initialized from the same lora_A / lora_B / magnitude /
    base_weight state.
    """

    @torch.no_grad()
    def _build_pair(self, in_features=512, out_features=384, r=16, alpha=32):
        """Build a (peft_module, fast_module) pair sharing all state."""
        import bitsandbytes as bnb
        from peft.tuners.lora.bnb import Linear4bit as PeftLoraLinear4bit

        from qdora_fast import Qdora4bitLinear

        torch.manual_seed(0)

        # Build a base Linear4bit. Initial weights drawn from a normal dist.
        base = nn.Linear(in_features, out_features, bias=False)
        base.weight.data.normal_(mean=0.0, std=0.02)

        # Convert to bnb Linear4bit by re-creating the layer.
        bnb_base = bnb.nn.Linear4bit(
            in_features, out_features, bias=False, compute_dtype=torch.bfloat16,
            quant_type="nf4", compress_statistics=True,
        )
        # Init with the dense weight, then move to CUDA to trigger quant.
        bnb_base.weight = bnb.nn.Params4bit(
            base.weight.data.contiguous(),
            requires_grad=False,
            quant_type="nf4",
            compress_statistics=True,
            quant_storage=torch.uint8,
        )
        bnb_base = bnb_base.cuda()

        # Make a deep-equivalent copy for the PEFT path. PEFT will wrap this.
        bnb_base_for_peft = bnb.nn.Linear4bit(
            in_features, out_features, bias=False, compute_dtype=torch.bfloat16,
            quant_type="nf4", compress_statistics=True,
        )
        bnb_base_for_peft.weight = bnb.nn.Params4bit(
            base.weight.data.contiguous(),
            requires_grad=False,
            quant_type="nf4",
            compress_statistics=True,
            quant_storage=torch.uint8,
        )
        bnb_base_for_peft = bnb_base_for_peft.cuda()

        # Build PEFT DoRA wrapper.
        peft_mod = PeftLoraLinear4bit(
            base_layer=bnb_base_for_peft,
            adapter_name="default",
            r=r, lora_alpha=alpha,
            lora_dropout=0.0,
            init_lora_weights=True,
            use_rslora=False,
            use_dora=True,
        )

        # Build our fast wrapper, K=1 (recompute every step → matches PEFT).
        fast_mod = Qdora4bitLinear(
            base_layer=bnb_base, r=r, lora_alpha=alpha,
            lora_dropout=0.0, norm_cache_steps=1,
        )

        # Sync state: copy lora_A, lora_B, magnitude from PEFT into fast.
        peft_lora_A = peft_mod.lora_A["default"].weight.data.to(torch.bfloat16)
        peft_lora_B = peft_mod.lora_B["default"].weight.data.to(torch.bfloat16)
        # PEFT's DoraLinearLayer stores magnitude under .lora_magnitude_vector["default"].weight
        peft_magnitude = (
            peft_mod.lora_magnitude_vector["default"].weight.data.to(torch.bfloat16)
        )
        fast_mod.lora_A.weight.data.copy_(peft_lora_A)
        fast_mod.lora_B.weight.data.copy_(peft_lora_B)
        fast_mod.magnitude.data.copy_(peft_magnitude)
        # Refresh fast's norm cache to reflect the synced LoRA weights (LoRA-B
        # is zero by PEFT default init, so it should match the base norm
        # already, but sync explicitly to be safe).
        fast_mod._recompute_norm_cache()

        return peft_mod, fast_mod

    def test_forward_matches_peft_with_K1(self):
        peft_mod, fast_mod = self._build_pair()
        peft_mod.train()
        fast_mod.train()

        torch.manual_seed(1)
        x = torch.randn(4, 16, 512, device="cuda", dtype=torch.bfloat16)

        with torch.no_grad():
            y_peft = peft_mod(x)
            y_fast = fast_mod(x)

        self.assertEqual(y_peft.shape, y_fast.shape)
        rel_err = (
            (y_peft.float() - y_fast.float()).norm()
            / y_peft.float().norm().clamp_min(1e-8)
        )
        # bf16 + NF4 round-trip + bnb's matmul kernel rounding: 1e-2 is generous
        # but realistic. PEFT's path uses F.linear on dequanted bf16; we use
        # bnb.matmul_4bit which is fused. Outputs should be very close but not
        # bitwise identical due to different kernel paths.
        self.assertLess(rel_err.item(), 0.02, f"rel_err={rel_err.item():.4f}")

    def test_backward_grads_match_peft_with_K1(self):
        peft_mod, fast_mod = self._build_pair()
        peft_mod.train()
        fast_mod.train()

        torch.manual_seed(2)
        x = torch.randn(2, 8, 512, device="cuda", dtype=torch.bfloat16)
        target = torch.randn(2, 8, 384, device="cuda", dtype=torch.bfloat16)

        # PEFT backward
        for p in peft_mod.parameters():
            if p.grad is not None:
                p.grad = None
        loss_peft = (peft_mod(x) - target).pow(2).mean()
        loss_peft.backward()
        peft_grads = {
            "lora_A": peft_mod.lora_A["default"].weight.grad.detach().clone(),
            "lora_B": peft_mod.lora_B["default"].weight.grad.detach().clone(),
            "magnitude": peft_mod.lora_magnitude_vector["default"].weight.grad.detach().clone(),
        }

        # Fast backward
        for p in fast_mod.parameters():
            if p.grad is not None:
                p.grad = None
        loss_fast = (fast_mod(x) - target).pow(2).mean()
        loss_fast.backward()
        fast_grads = {
            "lora_A": fast_mod.lora_A.weight.grad.detach().clone(),
            "lora_B": fast_mod.lora_B.weight.grad.detach().clone(),
            "magnitude": fast_mod.magnitude.grad.detach().clone(),
        }

        for name in ("lora_A", "lora_B", "magnitude"):
            g_peft = peft_grads[name].float()
            g_fast = fast_grads[name].float()
            denom = g_peft.norm().clamp_min(1e-8)
            rel_err = ((g_peft - g_fast).norm() / denom).item()
            self.assertLess(
                rel_err, 0.05,
                f"{name} grad rel_err={rel_err:.4f} (g_peft.norm={g_peft.norm().item():.4f})",
            )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for bnb 4-bit")
class TestQdora4bitLinearCacheBehavior(unittest.TestCase):
    """Verify the column-norm cache is refreshed exactly every K steps."""

    def test_cache_refreshes_every_K_steps(self):
        import bitsandbytes as bnb
        from qdora_fast import Qdora4bitLinear

        torch.manual_seed(0)
        base = nn.Linear(256, 128, bias=False)
        base.weight.data.normal_(0.0, 0.02)
        bnb_base = bnb.nn.Linear4bit(
            256, 128, bias=False, compute_dtype=torch.bfloat16,
            quant_type="nf4", compress_statistics=True,
        )
        bnb_base.weight = bnb.nn.Params4bit(
            base.weight.data.contiguous(),
            requires_grad=False, quant_type="nf4",
            compress_statistics=True, quant_storage=torch.uint8,
        )
        bnb_base = bnb_base.cuda()

        K = 4
        mod = Qdora4bitLinear(base_layer=bnb_base, r=8, lora_alpha=16, norm_cache_steps=K)
        mod.train()

        x = torch.randn(2, 4, 256, device="cuda", dtype=torch.bfloat16)

        # Mutate lora_B (not lora_A — B is zero-initialized; A alone with B=0
        # leaves the merged norm unchanged). Adding to B makes B@A non-zero so
        # the merged norm shifts.
        with torch.no_grad():
            mod.lora_B.weight.add_(0.1)

        cache_before = mod.norm_cache.detach().clone()

        # Step 1 (counter=0 → recompute happens).
        _ = mod(x)
        cache_after_1 = mod.norm_cache.detach().clone()
        # Counter is now 1.
        self.assertEqual(mod.step_counter.item(), 1)
        # Cache should have refreshed because counter was 0 % K == 0.
        # It must differ from before (we just added 0.1 to lora_B).
        diff = (cache_after_1.float() - cache_before.float()).abs().max().item()
        self.assertGreater(diff, 0.0, "first call should have refreshed cache")

        # Now mutate lora_B again so further refreshes would change the cache.
        with torch.no_grad():
            mod.lora_B.weight.add_(0.5)

        # Steps 2, 3 (counter goes 1→2, 2→3): NO recompute (1%4!=0, 2%4!=0).
        for _ in range(2):
            _ = mod(x)
        cache_after_3 = mod.norm_cache.detach().clone()
        self.assertEqual(mod.step_counter.item(), 3)
        same = (cache_after_3.float() - cache_after_1.float()).abs().max().item()
        self.assertEqual(same, 0.0, "cache must not have changed between K-aligned steps")

        # Step 4 (counter=3 → 3%4!=0; recompute does NOT trigger).
        # Wait: condition is `step_counter % K == 0` BEFORE incrementing.
        # counter=3 → 3%4=3 → no recompute. counter then becomes 4.
        _ = mod(x)
        self.assertEqual(mod.step_counter.item(), 4)
        # Step 5: counter=4 → 4%4=0 → recompute.
        _ = mod(x)
        self.assertEqual(mod.step_counter.item(), 5)
        cache_after_5 = mod.norm_cache.detach().clone()
        diff2 = (cache_after_5.float() - cache_after_3.float()).abs().max().item()
        self.assertGreater(diff2, 0.0, "cache must have refreshed at step 5 (counter was 4)")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for bnb 4-bit")
class TestQdora4bitLinearMergeRoundtrip(unittest.TestCase):
    """merge_to_dense_bf16 should produce a bf16 nn.Linear whose forward
    matches the un-merged Qdora4bitLinear forward (eval mode) within bf16
    tolerance.
    """

    def test_merge_matches_unmerged_forward(self):
        import bitsandbytes as bnb
        from qdora_fast import Qdora4bitLinear

        torch.manual_seed(0)
        base = nn.Linear(256, 128, bias=False)
        base.weight.data.normal_(0.0, 0.02)
        bnb_base = bnb.nn.Linear4bit(
            256, 128, bias=False, compute_dtype=torch.bfloat16,
            quant_type="nf4", compress_statistics=True,
        )
        bnb_base.weight = bnb.nn.Params4bit(
            base.weight.data.contiguous(),
            requires_grad=False, quant_type="nf4",
            compress_statistics=True, quant_storage=torch.uint8,
        )
        bnb_base = bnb_base.cuda()

        mod = Qdora4bitLinear(base_layer=bnb_base, r=16, lora_alpha=32, norm_cache_steps=1)
        # Make LoRA non-trivial so merge actually does something.
        with torch.no_grad():
            mod.lora_A.weight.normal_(0, 0.02)
            mod.lora_B.weight.normal_(0, 0.02)
            mod.magnitude.add_(0.05)
            mod._recompute_norm_cache()

        mod.eval()
        merged = mod.merge_to_dense_bf16()

        x = torch.randn(2, 4, 256, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            y_unmerged = mod(x)
            y_merged = merged(x)

        rel_err = (
            (y_unmerged.float() - y_merged.float()).norm()
            / y_unmerged.float().norm().clamp_min(1e-8)
        )
        self.assertLess(rel_err.item(), 0.02, f"merge rel_err={rel_err.item():.4f}")


if __name__ == "__main__":
    unittest.main()
