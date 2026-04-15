import os, sys
import unittest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from compress.btt.btt_linear import BTTLinear
from compress.topology import export_btt_topology, rebuild_btt_from_topology


def _make_btt(m=2, a=3, n=2, b=4, rank=2):
    btt_l = torch.randn(m, n * rank, a)
    btt_r = torch.randn(n, b, m * rank)
    bias = torch.randn(m * a)
    return BTTLinear(btt_l, btt_r, bias=bias, m=m, a=a, n=n, b=b, rank=rank)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 6)
        self.fc2 = nn.Linear(6, 4)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class TestCompressTopology(unittest.TestCase):
    def test_export_walks_module_tree(self):
        torch.manual_seed(0)
        model = ToyModel()
        model.fc1 = _make_btt(m=2, a=3, n=2, b=4, rank=2)
        topology = export_btt_topology(model)
        self.assertIn("fc1", topology)
        self.assertNotIn("fc2", topology)
        self.assertEqual(topology["fc1"]["m"], 2)

    def test_rebuild_reinstates_btt_modules(self):
        torch.manual_seed(0)
        model = ToyModel()
        original_btt = _make_btt(m=2, a=3, n=2, b=4, rank=2)
        model.fc1 = original_btt

        topology = export_btt_topology(model)
        state = {k: v.clone() for k, v in model.state_dict().items()}

        fresh = ToyModel()
        rebuild_btt_from_topology(fresh, topology)
        self.assertIsInstance(fresh.fc1, BTTLinear)

        fresh.load_state_dict(state)
        x = torch.randn(3, 8)
        self.assertTrue(torch.allclose(fresh(x), model(x), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
