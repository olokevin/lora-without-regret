"""Tests for `convert_mode` flag in BTT init (svd default vs qr)."""
import unittest
import warnings

import torch
import torch.nn as nn

from btt_layer import (
    BTTLayer,
    convert_linear_to_btt,
    normalize_blocktt_convert_mode,
)


CUDA_AVAILABLE = torch.cuda.is_available()


def _device():
    return "cuda" if CUDA_AVAILABLE else "cpu"


def _maybe_skip_no_cuda(test):
    return unittest.skipUnless(CUDA_AVAILABLE, "convert_linear_to_btt requires CUDA weights")(test)


class TestNormalizeConvertMode(unittest.TestCase):
    def test_accepts_svd_and_qr(self):
        self.assertEqual(normalize_blocktt_convert_mode("svd"), "svd")
        self.assertEqual(normalize_blocktt_convert_mode("qr"), "qr")
        self.assertEqual(normalize_blocktt_convert_mode("  QR  "), "qr")

    def test_rejects_other(self):
        with self.assertRaises(ValueError):
            normalize_blocktt_convert_mode("cur")
        with self.assertRaises(ValueError):
            normalize_blocktt_convert_mode("")


def _build_layer_qr_init(in_features, out_features, decomp_mode, rank="full",
                         s_merged_to=None, train_position="small", dtype=torch.float32):
    layer = BTTLayer(
        in_features=in_features,
        out_features=out_features,
        rank=rank,
        bias=False,
        lr_act=False,
        decomp_mode=decomp_mode,
    ).to(device=_device(), dtype=dtype)
    weight = torch.randn(out_features, in_features, device=_device(), dtype=dtype)
    layer.init_from_linear_weight(
        weight,
        bias=None,
        s_merged_to=s_merged_to,
        train_position=train_position,
        convert_mode="qr",
    )
    return layer, weight


@_maybe_skip_no_cuda
class TestQRInitLossless(unittest.TestCase):
    def test_qr_full_rank_lossless_tall_block(self):
        # square layout: m=a=4, n=b=2 (out=16, in=8) -> per-block (a=4, b=2), tall
        layer, weight = _build_layer_qr_init(
            in_features=8, out_features=16, decomp_mode="square", rank="full"
        )
        recon = layer.materialize_dense_weight()
        err = (recon - weight).abs().max().item()
        self.assertLess(err, 1e-4)
        self.assertIsNone(layer.btt_s)

    def test_qr_full_rank_lossless_wide_block(self):
        # square layout: m=a=2, n=b=4 (out=8, in=16) -> per-block (a=2, b=4), wide
        layer, weight = _build_layer_qr_init(
            in_features=16, out_features=8, decomp_mode="square", rank="full"
        )
        recon = layer.materialize_dense_weight()
        err = (recon - weight).abs().max().item()
        self.assertLess(err, 1e-4)
        self.assertIsNone(layer.btt_s)

    def test_qr_full_rank_lossless_square_block(self):
        # a==b=4 (out=16, in=16, square layout)
        layer, weight = _build_layer_qr_init(
            in_features=16, out_features=16, decomp_mode="square", rank="full"
        )
        recon = layer.materialize_dense_weight()
        err = (recon - weight).abs().max().item()
        self.assertLess(err, 1e-4)
        self.assertIsNone(layer.btt_s)

    def test_qr_input_one_block_lossless(self):
        layer, weight = _build_layer_qr_init(
            in_features=8, out_features=16, decomp_mode="input_one_block", rank="full"
        )
        recon = layer.materialize_dense_weight()
        err = (recon - weight).abs().max().item()
        self.assertLess(err, 1e-4)

    def test_qr_output_one_block_lossless(self):
        layer, weight = _build_layer_qr_init(
            in_features=8, out_features=16, decomp_mode="output_one_block", rank="full"
        )
        recon = layer.materialize_dense_weight()
        err = (recon - weight).abs().max().item()
        self.assertLess(err, 1e-4)


@_maybe_skip_no_cuda
class TestQRInitOrthogonality(unittest.TestCase):
    def _check_orthogonal_small_factor(self, layer):
        """Whichever core has the (k x k) shape on the small side should be orthogonal."""
        m, n, a, b = layer.m, layer.n, layer.a, layer.b
        rank = layer.rank
        # Reshape btt_l back to per-block (m*n, a, rank) and btt_r to (m*n, rank, b)
        core_l = layer.btt_l.reshape(m, rank, n, a).permute(0, 2, 3, 1).reshape(m * n, a, rank)
        core_r = layer.btt_r.reshape(n, b, m, rank).permute(2, 0, 3, 1).reshape(m * n, rank, b)

        if a >= b:
            # LQ: small factor is core_r ((rank,b)=(b,b)), should have orthonormal *rows*
            # core_r @ core_r.T == I_b (per block)
            gram = core_r @ core_r.transpose(-1, -2)
            eye = torch.eye(rank, device=gram.device, dtype=gram.dtype).expand_as(gram)
            err = (gram - eye).abs().max().item()
        else:
            # QR: small factor is core_l ((a,rank)=(a,a)), should have orthonormal *columns*
            # core_l.T @ core_l == I_a (per block)
            gram = core_l.transpose(-1, -2) @ core_l
            eye = torch.eye(rank, device=gram.device, dtype=gram.dtype).expand_as(gram)
            err = (gram - eye).abs().max().item()
        return err

    def test_tall_block_core_r_orthogonal(self):
        layer, _ = _build_layer_qr_init(
            in_features=8, out_features=16, decomp_mode="square", rank="full"
        )
        # a=4, b=2 -> tall, core_r should be orthogonal
        err = self._check_orthogonal_small_factor(layer)
        self.assertLess(err, 1e-4)

    def test_wide_block_core_l_orthogonal(self):
        layer, _ = _build_layer_qr_init(
            in_features=16, out_features=8, decomp_mode="square", rank="full"
        )
        # a=2, b=4 -> wide, core_l should be orthogonal
        err = self._check_orthogonal_small_factor(layer)
        self.assertLess(err, 1e-4)


@_maybe_skip_no_cuda
class TestQRInitSemantics(unittest.TestCase):
    def test_qr_warns_on_non_keep_s_merged_to(self):
        layer = BTTLayer(
            in_features=8, out_features=16, rank="full", bias=False,
            lr_act=False, decomp_mode="square",
        ).to(device=_device(), dtype=torch.float32)
        weight = torch.randn(16, 8, device=_device())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            layer.init_from_linear_weight(
                weight, bias=None,
                s_merged_to="output",  # ignored under qr
                train_position="small",
                convert_mode="qr",
            )
        self.assertTrue(
            any("convert_mode='qr'" in str(w.message) for w in caught),
            f"Expected warning about ignored s_merged_to, got: {[str(w.message) for w in caught]}"
        )
        self.assertIsNone(layer.btt_s)

    def test_qr_rejects_keep_frozen(self):
        layer = BTTLayer(
            in_features=8, out_features=16, rank="full", bias=False,
            lr_act=False, decomp_mode="square",
        ).to(device=_device(), dtype=torch.float32)
        weight = torch.randn(16, 8, device=_device())
        with self.assertRaises(ValueError) as ctx:
            layer.init_from_linear_weight(
                weight, bias=None,
                s_merged_to="keep_frozen",
                train_position="small",
                convert_mode="qr",
            )
        self.assertIn("qr", str(ctx.exception).lower())

    def test_qr_rejects_keep_trainable(self):
        layer = BTTLayer(
            in_features=8, out_features=16, rank="full", bias=False,
            lr_act=False, decomp_mode="square",
        ).to(device=_device(), dtype=torch.float32)
        weight = torch.randn(16, 8, device=_device())
        with self.assertRaises(ValueError):
            layer.init_from_linear_weight(
                weight, bias=None,
                s_merged_to="keep_trainable",
                train_position="small",
                convert_mode="qr",
            )

    def test_qr_truncated_rank_runs(self):
        # rank=1 < min(a,b)=2 ; should still run, no losslessness check
        layer = BTTLayer(
            in_features=8, out_features=16, rank=1, bias=False,
            lr_act=False, decomp_mode="square",
        ).to(device=_device(), dtype=torch.float32)
        weight = torch.randn(16, 8, device=_device())
        layer.init_from_linear_weight(
            weight, bias=None,
            train_position="small",
            convert_mode="qr",
        )
        # forward should not crash
        x = torch.randn(2, 8, device=_device())
        out = layer(x)
        self.assertEqual(out.shape, (2, 16))

    def test_qr_with_s_merged_to_none_no_warning(self):
        layer = BTTLayer(
            in_features=8, out_features=16, rank="full", bias=False,
            lr_act=False, decomp_mode="square",
        ).to(device=_device(), dtype=torch.float32)
        weight = torch.randn(16, 8, device=_device())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            layer.init_from_linear_weight(
                weight, bias=None,
                s_merged_to=None,
                train_position="small",
                convert_mode="qr",
            )
        qr_warnings = [w for w in caught if "convert_mode='qr'" in str(w.message)]
        self.assertEqual(len(qr_warnings), 0)


@_maybe_skip_no_cuda
class TestConvertLinearToBTTAcceptsConvertMode(unittest.TestCase):
    def test_default_is_svd(self):
        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(8, 16, bias=False)

        torch.manual_seed(0)
        model = Toy().to(device=_device(), dtype=torch.float32)
        original_w = model.q_proj.weight.data.clone()
        convert_linear_to_btt(
            model, btt_rank="full", include_names=("q_proj",),
            decomp_mode="input_one_block",
        )
        # SVD path: full rank should reconstruct losslessly
        recon = model.q_proj.materialize_dense_weight()
        self.assertLess((recon - original_w).abs().max().item(), 1e-4)
        # btt_s presence depends on default s_merged_to; we don't assert here.

    def test_qr_mode_full_rank_lossless(self):
        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(8, 16, bias=False)

        torch.manual_seed(0)
        model = Toy().to(device=_device(), dtype=torch.float32)
        original_w = model.q_proj.weight.data.clone()
        convert_linear_to_btt(
            model, btt_rank="full", include_names=("q_proj",),
            decomp_mode="square",
            convert_mode="qr",
        )
        recon = model.q_proj.materialize_dense_weight()
        self.assertLess((recon - original_w).abs().max().item(), 1e-4)
        self.assertIsNone(model.q_proj.btt_s)


if __name__ == "__main__":
    unittest.main()
