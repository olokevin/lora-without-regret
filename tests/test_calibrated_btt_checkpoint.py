import os, sys
import tempfile
import unittest
import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

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


class _ToyConfig(PretrainedConfig):
    model_type = "toy_btt"

    def __init__(self, in_features: int = 8, out_features: int = 6, **kw):
        super().__init__(**kw)
        self.in_features = in_features
        self.out_features = out_features


class ToyModel(PreTrainedModel):
    """Minimal PreTrainedModel wrapping one linear layer, so
    `model.save_pretrained` / `from_pretrained` work without requiring a
    full HF model definition."""

    config_class = _ToyConfig

    def __init__(self, config: _ToyConfig, with_btt: bool = False):
        super().__init__(config)
        if with_btt:
            self.fc = _make_btt()
        else:
            self.fc = nn.Linear(config.in_features, config.out_features)

    def forward(self, x):
        return self.fc(x)


class TestSaveLiftFormat(unittest.TestCase):
    """save_calibrated_btt_checkpoint must mirror
    ref/LIFT/src/utils/model_utils.py:save_hf_format byte-for-byte:
    exactly `pytorch_model.bin` + `config.json`, nothing else."""

    def test_writes_only_bin_and_config(self):
        torch.manual_seed(0)
        config = _ToyConfig(in_features=8, out_features=6)
        trained = ToyModel(config, with_btt=True)
        x = torch.randn(3, trained.fc.in_features)
        expected = trained.fc(x).detach().clone()

        with tempfile.TemporaryDirectory() as tmp:
            ci.save_calibrated_btt_checkpoint(trained, tmp)
            contents = set(os.listdir(tmp))
            self.assertEqual(contents, {"pytorch_model.bin", "config.json"},
                             f"unexpected files: {contents}")

            self.assertIsInstance(trained.fc, nn.Linear)
            self.assertTrue(torch.allclose(trained.fc(x), expected, atol=1e-5))

            # Round-trip: state_dict from the .bin loads into a fresh skeleton
            # and reproduces the forward.
            state = torch.load(os.path.join(tmp, "pytorch_model.bin"), weights_only=True)
            fresh = ToyModel(config, with_btt=False)
            fresh.load_state_dict(state)
            self.assertTrue(torch.allclose(fresh.fc(x), expected, atol=1e-5))


class TestSaveHfPretrainedFormat(unittest.TestCase):
    """save_calibrated_btt_hf_pretrained must produce a directory
    byte-for-byte equivalent to `model.save_pretrained(dir)` — the shape
    used by run_sft / run_rl / run_rl_dapo non-calib branches."""

    def test_matches_save_pretrained(self):
        torch.manual_seed(0)
        config = _ToyConfig(in_features=8, out_features=6)
        trained = ToyModel(config, with_btt=True)
        x = torch.randn(3, trained.fc.in_features)
        expected = trained.fc(x).detach().clone()

        with tempfile.TemporaryDirectory() as tmp_calib, \
             tempfile.TemporaryDirectory() as tmp_dense:
            # Dense reference: materialize by hand, then save_pretrained.
            trained_ref = ToyModel(config, with_btt=True)
            trained_ref.load_state_dict(trained.state_dict())
            ci.materialize_calibrated_btt_to_linear(trained_ref)
            trained_ref.save_pretrained(tmp_dense)

            # Calib save under test.
            ci.save_calibrated_btt_hf_pretrained(trained, tmp_calib)

            self.assertEqual(set(os.listdir(tmp_calib)), set(os.listdir(tmp_dense)))

            reloaded = ToyModel.from_pretrained(tmp_calib)
            self.assertIsInstance(reloaded.fc, nn.Linear)
            self.assertTrue(torch.allclose(reloaded.fc(x), expected, atol=1e-5))


class TestLegacyConverter(unittest.TestCase):
    """Legacy checkpoints (btt_topology.json + model.safetensors without
    config.json) must still be loadable via load_calibrated_btt_for_eval
    — the converter in ref/LIFT/scripts/convert_calib_btt_to_hf.py
    depends on this path."""

    def test_roundtrip_through_legacy_loader(self):
        import json
        from safetensors.torch import save_file
        from compress.topology import export_btt_topology

        torch.manual_seed(1)
        config = _ToyConfig(in_features=8, out_features=6)
        trained = ToyModel(config, with_btt=True)
        x = torch.randn(3, trained.fc.in_features)
        expected = trained.fc(x).detach().clone()

        with tempfile.TemporaryDirectory() as tmp:
            state = {n: p.detach().cpu().contiguous() for n, p in trained.state_dict().items()}
            save_file(state, os.path.join(tmp, "model.safetensors"))
            with open(os.path.join(tmp, "btt_topology.json"), "w") as f:
                json.dump(export_btt_topology(trained), f)

            fresh = ToyModel(config, with_btt=False)
            ci.load_calibrated_btt_for_eval(fresh, tmp)
            self.assertIsInstance(fresh.fc, BTTLinear)
            self.assertTrue(torch.allclose(fresh.fc(x), expected, atol=1e-6))

            ci.materialize_calibrated_btt_to_linear(fresh)
            self.assertIsInstance(fresh.fc, nn.Linear)
            self.assertTrue(torch.allclose(fresh.fc(x), expected, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
