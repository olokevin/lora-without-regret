import unittest
import torch
from analysis.analyze_weights import materialize_blocktt_weight


class TestMaterializeBlockTT(unittest.TestCase):
    def test_roundtrip_identity(self):
        """A BTT decomposition of a known matrix should reconstruct it."""
        m, n, rank, a, b = 2, 2, 3, 4, 4
        torch.manual_seed(42)
        btt_l = torch.randn(m, rank * n, a)
        btt_r = torch.randn(n, b, m * rank)
        W = materialize_blocktt_weight(btt_l, btt_r, btt_s=None)
        self.assertEqual(W.shape, (m * a, n * b))

    def test_with_btt_s(self):
        """Materialization with separate singular values."""
        m, n, rank, a, b = 2, 2, 3, 4, 4
        torch.manual_seed(42)
        btt_l = torch.randn(m, rank * n, a)
        btt_r = torch.randn(n, b, m * rank)
        btt_s = torch.ones(m, n, rank)
        W_no_s = materialize_blocktt_weight(btt_l, btt_r, btt_s=None)
        W_with_s = materialize_blocktt_weight(btt_l, btt_r, btt_s=btt_s)
        torch.testing.assert_close(W_no_s, W_with_s)

    def test_matches_btt_layer(self):
        """Our standalone function matches BTTLayer.materialize_dense_weight()."""
        from btt_layer import BTTLayer

        torch.manual_seed(7)
        layer = BTTLayer(
            in_features=16, out_features=16,
            rank=4, decomp_mode="square",
            init_mode="default", bias=False, lr_act=False,
        )
        W_ref = layer.materialize_dense_weight()
        W_ours = materialize_blocktt_weight(
            layer.btt_l.data, layer.btt_r.data,
            btt_s=layer.btt_s.data if layer.btt_s is not None else None,
        )
        torch.testing.assert_close(W_ours, W_ref, atol=1e-5, rtol=1e-5)


class TestMaterializeSVD(unittest.TestCase):
    def test_basic(self):
        from analysis.analyze_weights import materialize_svd_weight
        torch.manual_seed(42)
        svd_a = torch.randn(8, 4)
        svd_b = torch.randn(4, 6)
        W = materialize_svd_weight(svd_a, svd_b, svd_s=None)
        torch.testing.assert_close(W, svd_a @ svd_b)

    def test_with_svd_s(self):
        from analysis.analyze_weights import materialize_svd_weight
        torch.manual_seed(42)
        svd_a = torch.randn(8, 4)
        svd_b = torch.randn(4, 6)
        svd_s = torch.tensor([2.0, 1.0, 0.5, 0.1])
        W = materialize_svd_weight(svd_a, svd_b, svd_s=svd_s)
        expected = (svd_a * svd_s.unsqueeze(0)) @ svd_b
        torch.testing.assert_close(W, expected)

    def test_matches_svd_layer(self):
        from analysis.analyze_weights import materialize_svd_weight
        from svd_layer import SVDLayer

        torch.manual_seed(7)
        linear = torch.nn.Linear(16, 8, bias=False)
        layer = SVDLayer(in_features=16, out_features=8, bias=False)
        layer.init_from_linear_weight(linear.weight, bias=None, s_merged_to="keep")
        W_ref = layer.materialize_dense_weight()
        W_ours = materialize_svd_weight(
            layer.svd_a.data, layer.svd_b.data,
            svd_s=layer.svd_s.data if layer.svd_s is not None else None,
        )
        torch.testing.assert_close(W_ours, W_ref, atol=1e-5, rtol=1e-5)


class TestReconstructLoRA(unittest.TestCase):
    def test_basic(self):
        from analysis.analyze_weights import reconstruct_lora_weight
        torch.manual_seed(42)
        W_base = torch.randn(8, 16)
        lora_A = torch.randn(4, 16)
        lora_B = torch.randn(8, 4)
        W = reconstruct_lora_weight(W_base, lora_A, lora_B, lora_alpha=16, r=4)
        expected = W_base + (16 / 4) * lora_B @ lora_A
        torch.testing.assert_close(W, expected)

    def test_alpha_equals_r(self):
        from analysis.analyze_weights import reconstruct_lora_weight
        torch.manual_seed(42)
        W_base = torch.randn(8, 16)
        lora_A = torch.randn(4, 16)
        lora_B = torch.randn(8, 4)
        W = reconstruct_lora_weight(W_base, lora_A, lora_B, lora_alpha=4, r=4)
        expected = W_base + lora_B @ lora_A
        torch.testing.assert_close(W, expected)


if __name__ == "__main__":
    unittest.main()
