import os, sys
import tempfile
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
    def __init__(self, with_btt=False):
        super().__init__()
        self.fc = _make_btt() if with_btt else nn.Linear(8, 6)


class TestCheckpoint(unittest.TestCase):
    def test_save_and_reload_preserves_forward(self):
        torch.manual_seed(0)
        trained = ToyModel(with_btt=True)
        x = torch.randn(3, trained.fc.in_features)
        expected = trained.fc(x)

        with tempfile.TemporaryDirectory() as tmp:
            ci.save_calibrated_btt_checkpoint(trained, tmp)
            self.assertTrue(os.path.exists(os.path.join(tmp, "model.safetensors")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "btt_topology.json")))

            fresh = ToyModel(with_btt=False)
            ci.load_calibrated_btt_for_eval(fresh, tmp)
            self.assertIsInstance(fresh.fc, BTTLinear)
            out = fresh.fc(x)
            self.assertTrue(torch.allclose(out, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
