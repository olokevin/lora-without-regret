import os, sys
import unittest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from btt_layer import normalize_trainable_blocktt_cores_
from compress.btt.btt_linear import BTTLinear


class TestNormalizeBTTLinear(unittest.TestCase):
    def test_normalizes_btt_linear_cores(self):
        torch.manual_seed(0)
        m, a, n, b, rank = 2, 3, 2, 4, 2
        btt_l = torch.randn(m, n * rank, a) * 5.0
        btt_r = torch.randn(n, b, m * rank) * 5.0
        layer = BTTLinear(btt_l, btt_r, m=m, a=a, n=n, b=b, rank=rank)
        layer.btt_l.requires_grad = True
        layer.btt_r.requires_grad = True

        model = nn.Module()
        model.inner = layer

        result = normalize_trainable_blocktt_cores_(model)
        self.assertEqual(result["normalized_left_cores"], 1)
        self.assertEqual(result["normalized_right_cores"], 1)

        # btt_r: each (n, m, rank) vector over b should be unit norm.
        norms_r = torch.linalg.vector_norm(layer.btt_r, dim=1)
        self.assertTrue(torch.allclose(norms_r, torch.ones_like(norms_r), atol=1e-5))
        norms_l = torch.linalg.vector_norm(layer.btt_l, dim=2)
        self.assertTrue(torch.allclose(norms_l, torch.ones_like(norms_l), atol=1e-5))

    def test_skips_frozen_cores(self):
        torch.manual_seed(1)
        m, a, n, b, rank = 2, 3, 2, 4, 2
        layer = BTTLinear(
            torch.randn(m, n * rank, a) * 5.0,
            torch.randn(n, b, m * rank) * 5.0,
            m=m, a=a, n=n, b=b, rank=rank,
        )
        layer.btt_l.requires_grad = True
        layer.btt_r.requires_grad = False

        model = nn.Module()
        model.inner = layer

        result = normalize_trainable_blocktt_cores_(model)
        self.assertEqual(result["normalized_left_cores"], 1)
        self.assertEqual(result["normalized_right_cores"], 0)


if __name__ == "__main__":
    unittest.main()
