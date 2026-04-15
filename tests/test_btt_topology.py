import os, sys
import unittest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from compress.btt.btt_linear import BTTLinear


def _make_btt(m=2, a=3, n=2, b=4, rank=2, with_bias=True):
    btt_l = torch.randn(m, n * rank, a)
    btt_r = torch.randn(n, b, m * rank)
    bias = torch.randn(m * a) if with_bias else None
    return BTTLinear(btt_l, btt_r, bias=bias, m=m, a=a, n=n, b=b, rank=rank)


class TestBTTLinearTopology(unittest.TestCase):
    def test_topology_spec_roundtrip_with_bias(self):
        torch.manual_seed(0)
        layer = _make_btt(with_bias=True)
        spec = layer.topology_spec()

        for key in ("m", "a", "n", "b", "rank", "has_bias",
                    "in_features", "out_features"):
            self.assertIn(key, spec)

        new_layer = BTTLinear.from_topology_spec(spec)
        self.assertEqual(new_layer.btt_l.shape, layer.btt_l.shape)
        self.assertEqual(new_layer.btt_r.shape, layer.btt_r.shape)
        self.assertIsNotNone(new_layer.bias)
        self.assertEqual(new_layer.bias.shape, layer.bias.shape)

        new_layer.load_state_dict(layer.state_dict())
        x = torch.randn(5, layer.in_features)
        self.assertTrue(torch.allclose(new_layer(x), layer(x), atol=1e-5))

    def test_topology_spec_no_bias(self):
        torch.manual_seed(1)
        layer = _make_btt(with_bias=False)
        spec = layer.topology_spec()
        self.assertFalse(spec["has_bias"])
        new_layer = BTTLinear.from_topology_spec(spec)
        self.assertIsNone(new_layer.bias)


if __name__ == "__main__":
    unittest.main()
