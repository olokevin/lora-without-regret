import os, sys
import unittest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from compress.btt.btt_linear import BTTLinear


def _make_btt(d_in=8, d_out=6, m=2, a=3, n=2, b=4, rank=2, with_bias=True):
    btt_l = torch.randn(m, n * rank, a)
    btt_r = torch.randn(n, b, m * rank)
    bias = torch.randn(d_out) if with_bias else None
    return BTTLinear(btt_l, btt_r, bias=bias, m=m, a=a, n=n, b=b, rank=rank)


class TestBTTLinearMaterialize(unittest.TestCase):
    def test_materialize_matches_forward(self):
        torch.manual_seed(0)
        layer = _make_btt()
        x = torch.randn(5, layer.in_features)
        expected = layer(x)

        dense = layer.materialize_dense_weight()
        self.assertEqual(dense.shape, (layer.out_features, layer.in_features))
        out = x @ dense.T
        if layer.bias is not None:
            out = out + layer.bias
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_materialize_matches_forward_no_bias(self):
        torch.manual_seed(1)
        layer = _make_btt(with_bias=False)
        x = torch.randn(5, layer.in_features)
        expected = layer(x)
        dense = layer.materialize_dense_weight()
        self.assertEqual(dense.shape, (layer.out_features, layer.in_features))
        out = x @ dense.T
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
