import os, sys
import unittest
import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import compress_integration as ci
from compress.btt.btt_linear import BTTLinear


def _make_btt(m=2, a=3, n=2, b=4, rank=2):
    btt_l = torch.randn(m, n * rank, a)
    btt_r = torch.randn(n, b, m * rank)
    bias = torch.randn(m * a)
    return BTTLinear(btt_l, btt_r, bias=bias, m=m, a=a, n=n, b=b, rank=rank)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = _make_btt()


class TestMaterialize(unittest.TestCase):
    def test_materialize_returns_dense_tuples(self):
        torch.manual_seed(0)
        model = ToyModel()
        pairs = ci.materialize_calibrated_btt_weights(model)
        names = [n for n, _ in pairs]
        self.assertIn("fc.weight", names)
        self.assertIn("fc.bias", names)
        weight = dict(pairs)["fc.weight"]
        self.assertEqual(weight.shape, (model.fc.out_features, model.fc.in_features))

    def test_materialize_preserves_forward_behavior(self):
        torch.manual_seed(0)
        model = ToyModel()
        pairs = dict(ci.materialize_calibrated_btt_weights(model))
        x = torch.randn(3, model.fc.in_features)
        out = x @ pairs["fc.weight"].T + pairs["fc.bias"]
        self.assertTrue(torch.allclose(out, model.fc(x), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
