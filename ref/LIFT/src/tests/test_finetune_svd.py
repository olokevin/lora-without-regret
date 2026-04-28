"""Tests for ref/LIFT/src/finetune_svd.py.

Mirrors the SVD pipeline behavior in run_rl.py:
  - full-rank SVD on pretrained weights
  - train U / S / V according to --train_position and --s_merged_to
  - materialize SVDLayer back to nn.Linear at save time
"""
import os
import sys
import unittest

import torch
import torch.nn as nn

# Make repo root importable so `import svd_layer` works from this test.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir, os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
# Make ref/LIFT/src importable so `import finetune_svd` works.
_LIFT_SRC = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _LIFT_SRC not in sys.path:
    sys.path.insert(0, _LIFT_SRC)

from svd_layer import (
    SVDLayer,
    configure_svd_trainability,
    convert_linear_to_svd,
    get_svd_target_module_names,
)


class _ToyDecoderBlock(nn.Module):
    """Mimics the leaf-name layout of a Qwen/Llama transformer block so that
    `include_names` filtering in convert_linear_to_svd selects the right modules."""

    def __init__(self, dim=16):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.o_proj = nn.Linear(dim, dim, bias=True)
        self.gate_proj = nn.Linear(dim, dim, bias=True)
        self.up_proj = nn.Linear(dim, dim, bias=True)
        self.down_proj = nn.Linear(dim, dim, bias=True)
        # not in the SVD target list -> must remain nn.Linear after conversion
        self.other = nn.Linear(dim, dim, bias=True)
        # Looks like an lm_head; convert_linear_to_svd skips by default.
        self.lm_head = nn.Linear(dim, dim, bias=False)


def _requires_cuda(t):
    return unittest.skipUnless(torch.cuda.is_available(), "CUDA required")(t)


class TestSVDLayerCore(unittest.TestCase):
    """CPU-only sanity tests on SVDLayer itself (no Linear->SVD conversion)."""

    def test_full_rank_init_reconstructs_weight_with_keep_frozen(self):
        torch.manual_seed(0)
        weight = torch.randn(18, 12)
        bias = torch.randn(18)
        if torch.cuda.is_available():
            weight = weight.cuda()
            bias = bias.cuda()

            layer = SVDLayer(in_features=12, out_features=18, bias=True).to(weight.device)
            layer.init_from_linear_weight(
                weight,
                bias=bias,
                s_merged_to="keep_frozen",
                train_position="output",
            )
            self.assertIsNotNone(layer.svd_s)
            recon = layer.materialize_dense_weight()
            self.assertTrue(torch.allclose(recon, weight, atol=1e-4, rtol=1e-4))

    def test_configure_trainability_output_position(self):
        layer = SVDLayer(in_features=12, out_features=18, bias=True)

        class _Wrap(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

        model = _Wrap(layer)
        stats = configure_svd_trainability(
            model, train_position="output", train_bias=True
        )
        self.assertEqual(stats["num_svd_layers"], 1)
        self.assertEqual(stats["tuned_output_cores"], 1)
        self.assertEqual(stats["tuned_input_cores"], 0)
        self.assertTrue(layer.svd_a.requires_grad)
        self.assertFalse(layer.svd_b.requires_grad)
        self.assertTrue(layer.bias.requires_grad)

    def test_configure_trainability_input_position(self):
        layer = SVDLayer(in_features=12, out_features=18, bias=True)

        class _Wrap(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

        model = _Wrap(layer)
        stats = configure_svd_trainability(
            model, train_position="input", train_bias=False
        )
        self.assertEqual(stats["tuned_output_cores"], 0)
        self.assertEqual(stats["tuned_input_cores"], 1)
        self.assertFalse(layer.svd_a.requires_grad)
        self.assertTrue(layer.svd_b.requires_grad)
        self.assertFalse(layer.bias.requires_grad)

    def test_configure_trainability_both_position(self):
        layer = SVDLayer(in_features=12, out_features=18, bias=False)

        class _Wrap(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

        model = _Wrap(layer)
        stats = configure_svd_trainability(
            model, train_position="both", train_bias=False
        )
        self.assertTrue(layer.svd_a.requires_grad)
        self.assertTrue(layer.svd_b.requires_grad)
        self.assertEqual(stats["tuned_output_cores"], 1)
        self.assertEqual(stats["tuned_input_cores"], 1)

    def test_singular_values_trainable_only_when_kept_trainable(self):
        # We build the layer manually with svd_s registered (mimicking keep_*).
        layer = SVDLayer(in_features=12, out_features=18, bias=False)
        # Manually attach a singular-value parameter as init_from_linear_weight would.
        layer.svd_s = nn.Parameter(torch.ones(layer.rank), requires_grad=True)

        class _Wrap(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

        model = _Wrap(layer)
        configure_svd_trainability(
            model, train_position="output", train_singular_values=True
        )
        self.assertTrue(layer.svd_s.requires_grad)
        configure_svd_trainability(
            model, train_position="output", train_singular_values=False
        )
        self.assertFalse(layer.svd_s.requires_grad)


class TestSvdTargetSelection(unittest.TestCase):
    def test_get_svd_target_module_names_all(self):
        names = get_svd_target_module_names("all")
        self.assertEqual(
            set(names),
            {"q_proj", "k_proj", "v_proj", "o_proj",
             "gate_proj", "up_proj", "down_proj"},
        )

    def test_get_svd_target_module_names_attn_excludes_mlp(self):
        names = get_svd_target_module_names("attn")
        self.assertIn("q_proj", names)
        self.assertNotIn("gate_proj", names)

    def test_get_svd_target_module_names_mlp_excludes_attn(self):
        names = get_svd_target_module_names("mlp")
        self.assertIn("gate_proj", names)
        self.assertNotIn("q_proj", names)


@_requires_cuda
class TestSvdEndToEndConversion(unittest.TestCase):
    """convert_linear_to_svd requires weights to live on CUDA."""

    def test_conversion_preserves_outputs_for_full_rank(self):
        torch.manual_seed(0)
        model = _ToyDecoderBlock(dim=16).cuda()
        x = torch.randn(4, 16, device="cuda")

        # Reference output on the original (untouched) lm_head/other path is
        # not exercised, but per-target outputs should match after full-rank
        # SVD with keep_frozen (lossless reconstruction up to fp precision).
        ref_q = model.q_proj(x).detach().clone()

        convert_linear_to_svd(
            model,
            include_names=get_svd_target_module_names("all"),
            s_merged_to="keep_frozen",
            train_position="output",
        )

        self.assertIsInstance(model.q_proj, SVDLayer)
        self.assertIsInstance(model.gate_proj, SVDLayer)
        # Untargeted modules untouched.
        self.assertIsInstance(model.other, nn.Linear)
        # lm_head is in skip_names default.
        self.assertIsInstance(model.lm_head, nn.Linear)

        out_q = model.q_proj(x)
        self.assertTrue(torch.allclose(out_q, ref_q, atol=1e-3, rtol=1e-3))

    def test_train_position_output_freezes_svd_b(self):
        torch.manual_seed(0)
        model = _ToyDecoderBlock(dim=16).cuda()
        convert_linear_to_svd(
            model,
            include_names=get_svd_target_module_names("all"),
            s_merged_to="frozen",
            train_position="output",
        )
        stats = configure_svd_trainability(
            model,
            train_position="output",
            train_bias=True,
            train_embed_lm_head=False,
            train_singular_values=False,
        )
        self.assertGreater(stats["tuned_output_cores"], 0)
        self.assertEqual(stats["tuned_input_cores"], 0)
        for mod in model.modules():
            if isinstance(mod, SVDLayer):
                self.assertTrue(mod.svd_a.requires_grad)
                self.assertFalse(mod.svd_b.requires_grad)

    def test_train_position_input_freezes_svd_a(self):
        torch.manual_seed(0)
        model = _ToyDecoderBlock(dim=16).cuda()
        convert_linear_to_svd(
            model,
            include_names=get_svd_target_module_names("all"),
            s_merged_to="frozen",
            train_position="input",
        )
        stats = configure_svd_trainability(
            model,
            train_position="input",
            train_bias=True,
            train_embed_lm_head=False,
            train_singular_values=False,
        )
        self.assertEqual(stats["tuned_output_cores"], 0)
        self.assertGreater(stats["tuned_input_cores"], 0)
        for mod in model.modules():
            if isinstance(mod, SVDLayer):
                self.assertFalse(mod.svd_a.requires_grad)
                self.assertTrue(mod.svd_b.requires_grad)

    def test_keep_trainable_makes_singular_values_trainable(self):
        torch.manual_seed(0)
        model = _ToyDecoderBlock(dim=16).cuda()
        convert_linear_to_svd(
            model,
            include_names=("q_proj",),
            s_merged_to="keep_trainable",
            train_position="output",
        )
        configure_svd_trainability(
            model,
            train_position="output",
            train_bias=True,
            train_singular_values=True,
        )
        self.assertIsNotNone(model.q_proj.svd_s)
        self.assertTrue(model.q_proj.svd_s.requires_grad)

    def test_one_optimizer_step_actually_updates_trainable_factor(self):
        torch.manual_seed(0)
        model = _ToyDecoderBlock(dim=16).cuda()
        convert_linear_to_svd(
            model,
            include_names=get_svd_target_module_names("all"),
            s_merged_to="frozen",
            train_position="output",
        )
        configure_svd_trainability(
            model,
            train_position="output",
            train_bias=True,
            train_singular_values=False,
        )

        trainable = [p for p in model.parameters() if p.requires_grad]
        self.assertGreater(len(trainable), 0)
        opt = torch.optim.AdamW(trainable, lr=1e-2)

        before_a = model.q_proj.svd_a.detach().clone()
        before_b = model.q_proj.svd_b.detach().clone()

        x = torch.randn(2, 16, device="cuda")
        # Sum the q_proj output as a stand-in loss.
        loss = model.q_proj(x).pow(2).mean()
        loss.backward()
        opt.step()

        # Trainable side moved.
        self.assertFalse(torch.equal(before_a, model.q_proj.svd_a))
        # Frozen side untouched.
        self.assertTrue(torch.equal(before_b, model.q_proj.svd_b))

    def test_materialize_svd_to_linear_replaces_layers_and_preserves_output(self):
        from finetune_svd import materialize_svd_to_linear

        torch.manual_seed(0)
        model = _ToyDecoderBlock(dim=16).cuda()
        convert_linear_to_svd(
            model,
            include_names=get_svd_target_module_names("all"),
            s_merged_to="keep_frozen",
            train_position="output",
        )

        x = torch.randn(2, 16, device="cuda")
        before_out = model.q_proj(x).detach().clone()

        materialize_svd_to_linear(model)

        for leaf in (model.q_proj, model.k_proj, model.gate_proj, model.down_proj):
            self.assertIsInstance(leaf, nn.Linear)

        after_out = model.q_proj(x)
        self.assertTrue(
            torch.allclose(before_out, after_out, atol=1e-4, rtol=1e-4)
        )


class TestFinetuneSvdCli(unittest.TestCase):
    """Argparse smoke tests for finetune_svd.py.

    These don't load a model; they just exercise parse_args() so we catch
    regressions in flag wiring (defaults, choices, mode-specific normalization).
    """

    def _parse(self, extra_argv):
        import finetune_svd
        argv_backup = sys.argv
        try:
            sys.argv = ["finetune_svd.py", "--model_name_or_path", "stub"] + extra_argv
            return finetune_svd.parse_args()
        finally:
            sys.argv = argv_backup

    def test_defaults(self):
        args = self._parse([])
        self.assertEqual(args.train_position, "output")
        self.assertEqual(args.s_merged_to, "frozen")
        self.assertEqual(args.trainable_type, "all")
        self.assertFalse(args.no_train_bias)

    def test_train_position_choices(self):
        for pos in ("output", "input", "both"):
            args = self._parse(["--train_position", pos,
                                "--s_merged_to", "split" if pos == "both" else "frozen"])
            self.assertEqual(args.train_position, pos)

    def test_train_position_both_promotes_frozen_alias_to_split(self):
        # With train_position=both, the frozen/trainable aliases have no
        # single side to merge into, so finetune_svd promotes to "split".
        args = self._parse(["--train_position", "both", "--s_merged_to", "frozen"])
        self.assertEqual(args.s_merged_to, "split")

    def test_keep_trainable_passes_through(self):
        args = self._parse(["--s_merged_to", "keep_trainable"])
        self.assertEqual(args.s_merged_to, "keep_trainable")

    def test_invalid_s_merged_to_rejected(self):
        with self.assertRaises(SystemExit):
            self._parse(["--s_merged_to", "not_a_real_option"])

    def test_invalid_train_position_rejected(self):
        with self.assertRaises(SystemExit):
            self._parse(["--train_position", "small"])  # blocktt-only value

    def test_invalid_trainable_type_rejected(self):
        with self.assertRaises(SystemExit):
            self._parse(["--trainable_type", "everything"])


if __name__ == "__main__":
    unittest.main()
