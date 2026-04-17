import argparse
import os, sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import compress_integration as ci


def _make_parser(hyphen_style=True):
    p = argparse.ArgumentParser()
    p.add_argument("--train-mode" if hyphen_style else "--train_mode", default="blocktt")
    p.add_argument("--blocktt-rank" if hyphen_style else "--blocktt_rank", default="full")
    ci.add_calibrated_btt_args(p, hyphen_style=hyphen_style)
    return p


class TestValidate(unittest.TestCase):
    def test_none_mode_with_no_calib_flags_ok(self):
        p = _make_parser()
        argv = ["--train-mode", "blocktt"]
        args = p.parse_args(argv)
        ci.validate_calibrated_btt_args(args, argv=argv)

    def test_calib_mode_requires_blocktt(self):
        p = _make_parser()
        argv = ["--train-mode", "lora", "--calib-mode", "v2"]
        args = p.parse_args(argv)
        with self.assertRaisesRegex(ValueError, "only valid with --train-mode blocktt"):
            ci.validate_calibrated_btt_args(args, argv=argv)

    def test_calib_mode_traces_requires_path(self):
        p = _make_parser()
        argv = ["--train-mode", "blocktt", "--calib-mode", "v2", "--calib-source", "traces"]
        args = p.parse_args(argv)
        with self.assertRaisesRegex(ValueError, "calib-traces-path"):
            ci.validate_calibrated_btt_args(args, argv=argv)

    def test_int_blocktt_rank_rejected_on_calibrated(self):
        p = _make_parser()
        argv = ["--train-mode", "blocktt", "--calib-mode", "v2",
                "--calib-source", "c4", "--blocktt-rank", "4"]
        args = p.parse_args(argv)
        with self.assertRaisesRegex(ValueError, "integer"):
            ci.validate_calibrated_btt_args(args, argv=argv)

    def test_float_blocktt_rank_ok_on_calibrated(self):
        p = _make_parser()
        argv = ["--train-mode", "blocktt", "--calib-mode", "v2",
                "--calib-source", "c4", "--blocktt-rank", "0.7"]
        args = p.parse_args(argv)
        ci.validate_calibrated_btt_args(args, argv=argv)

    def test_calib_flag_passed_with_mode_none_accepted(self):
        # Non-default calib-* flags with --calib-mode=none are ignored, not errored.
        # Shell scripts pass these unconditionally and we don't want to force them
        # to branch on calib_mode.
        p = _make_parser()
        argv = ["--train-mode", "blocktt", "--calib-num-seqs", "64"]
        args = p.parse_args(argv)
        ci.validate_calibrated_btt_args(args, argv=argv)

    def test_underscore_style_also_works(self):
        p = _make_parser(hyphen_style=False)
        argv = ["--train_mode", "blocktt", "--calib_mode", "v2", "--calib_source", "c4"]
        args = p.parse_args(argv)
        ci.validate_calibrated_btt_args(args, argv=argv, hyphen_style=False)

    def test_lift_no_train_mode_attr_ok(self):
        # LIFT parser has no --train-mode; we accept args without that attribute.
        p = argparse.ArgumentParser()
        p.add_argument("--blocktt_rank", default="full")
        ci.add_calibrated_btt_args(p, hyphen_style=False)
        argv = ["--calib_mode", "v2", "--calib_source", "c4"]
        args = p.parse_args(argv)
        ci.validate_calibrated_btt_args(args, argv=argv, hyphen_style=False)

    def test_abbreviated_calib_flag_with_mode_none_accepted(self):
        # argparse allows --calib-n as abbreviation of --calib-num-seqs;
        # like the full-spelling case, this is accepted when calib-mode=none.
        p = _make_parser()
        argv = ["--train-mode", "blocktt", "--calib-n", "64"]
        args = p.parse_args(argv)
        ci.validate_calibrated_btt_args(args, argv=argv)

    def test_negative_float_rank_rejected(self):
        p = _make_parser()
        argv = ["--train-mode", "blocktt", "--calib-mode", "v2",
                "--calib-source", "c4", "--blocktt-rank", "-0.5"]
        args = p.parse_args(argv)
        with self.assertRaisesRegex(ValueError, r"\(0, 1\]"):
            ci.validate_calibrated_btt_args(args, argv=argv)

    def test_out_of_range_float_rank_rejected(self):
        p = _make_parser()
        argv = ["--train-mode", "blocktt", "--calib-mode", "v2",
                "--calib-source", "c4", "--blocktt-rank", "1.5"]
        args = p.parse_args(argv)
        with self.assertRaisesRegex(ValueError, r"\(0, 1\]"):
            ci.validate_calibrated_btt_args(args, argv=argv)


if __name__ == "__main__":
    unittest.main()
