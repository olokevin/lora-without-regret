import argparse
import os, sys
import unittest
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import compress_integration as ci


def _parse(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--train-mode", default="blocktt")
    p.add_argument("--blocktt-rank", default="full")
    p.add_argument("--trainable-type", default="all")
    p.add_argument("--decomp-mode", default="square")
    p.add_argument("--train-position", default="small")
    p.add_argument("--s-merged-to", default=None)
    p.add_argument("--blocktt-factorize-by-head", action="store_true", default=True)
    p.add_argument("--no-blocktt-factorize-by-head", dest="blocktt_factorize_by_head",
                   action="store_false")
    ci.add_calibrated_btt_args(p, hyphen_style=True)
    return p.parse_args(argv)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # mimic transformer naming
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "gate_proj": nn.Linear(8, 8),
                "up_proj": nn.Linear(8, 8),
                "down_proj": nn.Linear(8, 8),
                "q_proj": nn.Linear(8, 8),
                "k_proj": nn.Linear(8, 8),
                "v_proj": nn.Linear(8, 8),
                "o_proj": nn.Linear(8, 8),
                "other": nn.Linear(8, 8),
            })
        ])
        self.lm_head = nn.Linear(8, 32)


class TestBuildConfig(unittest.TestCase):
    def test_rank_full_becomes_ratio_one(self):
        args = _parse(["--calib-mode", "v2", "--calib-source", "c4"])
        cfg = ci.build_decomposition_config(args, model=ToyModel())
        self.assertEqual(cfg.train_mode, "btt_llm_v2")
        self.assertEqual(cfg.compression_ratio, 1.0)

    def test_rank_float_passthrough(self):
        args = _parse(["--calib-mode", "v2_bp", "--calib-source", "c4", "--blocktt-rank", "0.5"])
        cfg = ci.build_decomposition_config(args, model=ToyModel())
        self.assertEqual(cfg.train_mode, "btt_llm_v2_bp")
        self.assertAlmostEqual(cfg.compression_ratio, 0.5)

    def test_rank_float_range_validated(self):
        args = _parse(["--calib-mode", "v2", "--calib-source", "c4", "--blocktt-rank", "1.5"])
        with self.assertRaisesRegex(ValueError, "in \\(0, 1\\]"):
            ci.build_decomposition_config(args, model=ToyModel())

    def test_calib_mode_to_train_mode_mapping(self):
        for calib, train in [("v2", "btt_llm_v2"), ("v2_bp", "btt_llm_v2_bp"),
                              ("v2_combined", "btt_llm_v2_combined"),
                              ("twosteps", "btt_twosteps")]:
            args = _parse(["--calib-mode", calib, "--calib-source", "c4"])
            cfg = ci.build_decomposition_config(args, model=ToyModel())
            self.assertEqual(cfg.train_mode, train)

    def test_trainable_type_all_skips_only_non_target(self):
        args = _parse(["--calib-mode", "v2", "--calib-source", "c4", "--trainable-type", "all"])
        cfg = ci.build_decomposition_config(args, model=ToyModel())
        skip = set(s.strip() for s in cfg.skip_layers.split(","))
        self.assertIn("lm_head", skip)
        self.assertIn("other", skip)
        self.assertNotIn("gate_proj", skip)
        self.assertNotIn("q_proj", skip)

    def test_trainable_type_mlp_skips_attn(self):
        args = _parse(["--calib-mode", "v2", "--calib-source", "c4", "--trainable-type", "mlp"])
        cfg = ci.build_decomposition_config(args, model=ToyModel())
        skip = set(s.strip() for s in cfg.skip_layers.split(","))
        for n in ("q_proj", "k_proj", "v_proj", "o_proj", "other", "lm_head"):
            self.assertIn(n, skip)
        for n in ("gate_proj", "up_proj", "down_proj"):
            self.assertNotIn(n, skip)

    def test_trainable_type_attn_skips_mlp(self):
        args = _parse(["--calib-mode", "v2", "--calib-source", "c4", "--trainable-type", "attn"])
        cfg = ci.build_decomposition_config(args, model=ToyModel())
        skip = set(s.strip() for s in cfg.skip_layers.split(","))
        for n in ("gate_proj", "up_proj", "down_proj", "other", "lm_head"):
            self.assertIn(n, skip)
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
            self.assertNotIn(n, skip)

    def test_passthrough_fields(self):
        args = _parse(["--calib-mode", "v2", "--calib-source", "c4",
                       "--train-position", "small", "--decomp-mode", "input_one_block",
                       "--calib-num-seqs", "64", "--calib-max-length", "1024",
                       "--calib-seed", "7", "--s-merged-to", "trainable"])
        cfg = ci.build_decomposition_config(args, model=ToyModel())
        self.assertEqual(cfg.train_position, "small")
        self.assertEqual(cfg.decomp_mode, "input_one_block")
        self.assertEqual(cfg.calib_num_seqs, 64)
        self.assertEqual(cfg.calib_max_length, 1024)
        self.assertEqual(cfg.calib_seed, 7)
        self.assertEqual(cfg.s_merged_to, "trainable")
        self.assertTrue(cfg.factorize_by_head)

    def test_integer_rank_rejected(self):
        # Direct call (bypassing validate) should still reject integer ranks.
        args = _parse(["--calib-mode", "v2", "--calib-source", "c4", "--blocktt-rank", "4"])
        with self.assertRaisesRegex(ValueError, "integer"):
            ci.build_decomposition_config(args, model=ToyModel())

    def test_nan_rank_rejected(self):
        args = _parse(["--calib-mode", "v2", "--calib-source", "c4", "--blocktt-rank", "nan"])
        with self.assertRaisesRegex(ValueError, "finite"):
            ci.build_decomposition_config(args, model=ToyModel())

    def test_zero_rank_rejected(self):
        args = _parse(["--calib-mode", "v2", "--calib-source", "c4", "--blocktt-rank", "0"])
        # "0" is an integer-string — caught by integer check.
        with self.assertRaisesRegex(ValueError, "integer"):
            ci.build_decomposition_config(args, model=ToyModel())

    def test_missing_calib_mode_raises(self):
        # Parser without add_calibrated_btt_args — args has no calib_mode attr.
        import argparse as _ap
        p = _ap.ArgumentParser()
        p.add_argument("--train-mode", default="blocktt")
        p.add_argument("--blocktt-rank", default="full")
        p.add_argument("--trainable-type", default="all")
        p.add_argument("--decomp-mode", default="square")
        p.add_argument("--train-position", default="small")
        p.add_argument("--s-merged-to", default=None)
        p.add_argument("--blocktt-factorize-by-head", action="store_true", default=True)
        args = p.parse_args([])
        with self.assertRaisesRegex(ValueError, "calib_mode"):
            ci.build_decomposition_config(args, model=ToyModel())

    def test_default_train_position_both(self):
        # Parser that leaves train_position attr as default "both" rather than "small".
        import argparse as _ap
        p = _ap.ArgumentParser()
        p.add_argument("--train-mode", default="blocktt")
        p.add_argument("--blocktt-rank", default="full")
        p.add_argument("--trainable-type", default="all")
        p.add_argument("--decomp-mode", default="square")
        p.add_argument("--train-position", default="both")  # intentionally default to "both"
        p.add_argument("--s-merged-to", default=None)
        p.add_argument("--blocktt-factorize-by-head", action="store_true", default=True)
        ci.add_calibrated_btt_args(p, hyphen_style=True)
        args = p.parse_args(["--calib-mode", "v2", "--calib-source", "c4"])
        cfg = ci.build_decomposition_config(args, model=ToyModel())
        self.assertEqual(cfg.train_position, "both")


if __name__ == "__main__":
    unittest.main()
