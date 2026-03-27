import unittest
import json
import tempfile
import os
from pathlib import Path
import torch
from safetensors.torch import save_file
from analysis.analyze_weights import materialize_blocktt_weight
from analysis.analyze_weights import parse_args
from analysis.analyze_weights import (
    TARGET_MODULES,
    get_base_weight_key,
    get_checkpoint_keys,
    compute_update_row_col_norms,
    compute_singular_vector_angles,
    compute_spectrum_and_nss,
    compute_principal_angles,
    compute_principal_weight_overlap,
    compute_update_spectrum,
)
from analysis.plot_singular_relationship import (
    build_singular_maps,
    factor_keys_for_mode,
    find_max_step,
    flatten_heatmap_tensor,
    resolve_trained_sides,
)


class TestMaterializeBlockTT(unittest.TestCase):
    def test_roundtrip_identity(self):
        """A BTT decomposition of a known matrix should reconstruct it."""
        m, n, rank, a, b = 2, 2, 3, 4, 4
        torch.manual_seed(42)
        btt_l = torch.randn(m, rank * n, a)
        btt_r = torch.randn(n, b, m * rank)
        W = materialize_blocktt_weight(btt_l, btt_r, btt_s=None)
        self.assertEqual(W.shape, (m * a, n * b))

    def test_with_btt_s(self):
        """Materialization with separate singular values."""
        m, n, rank, a, b = 2, 2, 3, 4, 4
        torch.manual_seed(42)
        btt_l = torch.randn(m, rank * n, a)
        btt_r = torch.randn(n, b, m * rank)
        btt_s = torch.ones(m, n, rank)
        W_no_s = materialize_blocktt_weight(btt_l, btt_r, btt_s=None)
        W_with_s = materialize_blocktt_weight(btt_l, btt_r, btt_s=btt_s)
        torch.testing.assert_close(W_no_s, W_with_s)

    def test_matches_btt_layer(self):
        """Our standalone function matches BTTLayer.materialize_dense_weight()."""
        from btt_layer import BTTLayer

        torch.manual_seed(7)
        layer = BTTLayer(
            in_features=16, out_features=16,
            rank=4, decomp_mode="square",
            init_mode="default", bias=False, lr_act=False,
        )
        W_ref = layer.materialize_dense_weight()
        W_ours = materialize_blocktt_weight(
            layer.btt_l.data, layer.btt_r.data,
            btt_s=layer.btt_s.data if layer.btt_s is not None else None,
        )
        torch.testing.assert_close(W_ours, W_ref, atol=1e-5, rtol=1e-5)


class TestMaterializeSVD(unittest.TestCase):
    def test_basic(self):
        from analysis.analyze_weights import materialize_svd_weight
        torch.manual_seed(42)
        svd_a = torch.randn(8, 4)
        svd_b = torch.randn(4, 6)
        W = materialize_svd_weight(svd_a, svd_b, svd_s=None)
        torch.testing.assert_close(W, svd_a @ svd_b)

    def test_with_svd_s(self):
        from analysis.analyze_weights import materialize_svd_weight
        torch.manual_seed(42)
        svd_a = torch.randn(8, 4)
        svd_b = torch.randn(4, 6)
        svd_s = torch.tensor([2.0, 1.0, 0.5, 0.1])
        W = materialize_svd_weight(svd_a, svd_b, svd_s=svd_s)
        expected = (svd_a * svd_s.unsqueeze(0)) @ svd_b
        torch.testing.assert_close(W, expected)

    def test_matches_svd_layer(self):
        from analysis.analyze_weights import materialize_svd_weight
        from svd_layer import SVDLayer

        torch.manual_seed(7)
        linear = torch.nn.Linear(16, 8, bias=False)
        layer = SVDLayer(in_features=16, out_features=8, bias=False)
        layer.init_from_linear_weight(linear.weight, bias=None, s_merged_to="keep")
        W_ref = layer.materialize_dense_weight()
        W_ours = materialize_svd_weight(
            layer.svd_a.data, layer.svd_b.data,
            svd_s=layer.svd_s.data if layer.svd_s is not None else None,
        )
        torch.testing.assert_close(W_ours, W_ref, atol=1e-5, rtol=1e-5)


class TestReconstructLoRA(unittest.TestCase):
    def test_basic(self):
        from analysis.analyze_weights import reconstruct_lora_weight
        torch.manual_seed(42)
        W_base = torch.randn(8, 16)
        lora_A = torch.randn(4, 16)
        lora_B = torch.randn(8, 4)
        W = reconstruct_lora_weight(W_base, lora_A, lora_B, lora_alpha=16, r=4)
        expected = W_base + (16 / 4) * lora_B @ lora_A
        torch.testing.assert_close(W, expected)

    def test_alpha_equals_r(self):
        from analysis.analyze_weights import reconstruct_lora_weight
        torch.manual_seed(42)
        W_base = torch.randn(8, 16)
        lora_A = torch.randn(4, 16)
        lora_B = torch.randn(8, 4)
        W = reconstruct_lora_weight(W_base, lora_A, lora_B, lora_alpha=4, r=4)
        expected = W_base + lora_B @ lora_A
        torch.testing.assert_close(W, expected)


class TestKeyMapping(unittest.TestCase):
    def test_base_weight_key(self):
        key = get_base_weight_key(layer_idx=3, module_name="q_proj")
        self.assertEqual(key, "model.layers.3.self_attn.q_proj.weight")

    def test_base_weight_key_mlp(self):
        key = get_base_weight_key(layer_idx=0, module_name="gate_proj")
        self.assertEqual(key, "model.layers.0.mlp.gate_proj.weight")

    def test_checkpoint_keys_blocktt(self):
        keys = get_checkpoint_keys(layer_idx=1, module_name="q_proj", train_mode="blocktt")
        self.assertIn("model.layers.1.self_attn.q_proj.btt_l", keys.values())
        self.assertIn("model.layers.1.self_attn.q_proj.btt_r", keys.values())

    def test_checkpoint_keys_lora(self):
        keys = get_checkpoint_keys(layer_idx=0, module_name="q_proj", train_mode="lora")
        self.assertIn(
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight", keys.values()
        )
        self.assertIn(
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight", keys.values()
        )

    def test_target_modules(self):
        self.assertEqual(len(TARGET_MODULES), 7)
        self.assertIn("q_proj", TARGET_MODULES)
        self.assertIn("gate_proj", TARGET_MODULES)


class TestMetrics(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.W_before = torch.randn(32, 16)
        # Small perturbation
        self.W_after = self.W_before + 0.01 * torch.randn(32, 16)
        self.delta_W = self.W_after - self.W_before

    def test_update_row_col_norms_shapes(self):
        row_norms, col_norms = compute_update_row_col_norms(self.delta_W)
        self.assertEqual(row_norms.shape[0], 32)
        self.assertEqual(col_norms.shape[0], 16)

    def test_update_row_col_norms_nonneg(self):
        row_norms, col_norms = compute_update_row_col_norms(self.delta_W)
        self.assertTrue((row_norms >= 0).all())
        self.assertTrue((col_norms >= 0).all())

    def test_singular_vector_angles_small_perturbation(self):
        """Small perturbation should give small angles."""
        angles_u, angles_v = compute_singular_vector_angles(
            self.W_before, self.W_after, top_k=8
        )
        self.assertEqual(len(angles_u), 8)
        self.assertEqual(len(angles_v), 8)
        # Angles should be small (< 10 degrees) for tiny perturbation
        self.assertTrue(all(a < 10.0 for a in angles_u))

    def test_singular_vector_angles_identity(self):
        """No change should give near-zero angles (within floating-point tolerance)."""
        angles_u, angles_v = compute_singular_vector_angles(
            self.W_before, self.W_before, top_k=4
        )
        for a in angles_u:
            self.assertLess(a, 1.0)  # less than 1 degree

    def test_spectrum_and_nss(self):
        s_before, s_after, nss = compute_spectrum_and_nss(
            self.W_before, self.W_after
        )
        self.assertEqual(len(s_before), min(32, 16))
        self.assertEqual(len(s_after), min(32, 16))
        self.assertGreater(nss, 0)
        # Small perturbation -> small NSS
        self.assertLess(nss, 0.1)

    def test_spectrum_nss_zero_for_identical(self):
        _, _, nss = compute_spectrum_and_nss(self.W_before, self.W_before)
        self.assertAlmostEqual(nss, 0.0, places=6)

    def test_principal_angles_shape(self):
        angles = compute_principal_angles(self.W_before, self.W_after, top_k=8)
        self.assertEqual(len(angles), 8)

    def test_principal_angles_small_perturbation(self):
        angles = compute_principal_angles(self.W_before, self.W_after, top_k=8)
        # Small perturbation -> small angles
        self.assertTrue(all(a < 10.0 for a in angles))

    def test_overlap_ratio(self):
        overlap, baseline = compute_principal_weight_overlap(
            self.W_before, self.delta_W, top_k=8, alpha=0.1, threshold_frac=0.01
        )
        # overlap and baseline should be between 0 and 1
        self.assertGreaterEqual(overlap, 0.0)
        self.assertLessEqual(overlap, 1.0)
        self.assertAlmostEqual(baseline, 0.1, places=5)

    def test_update_spectrum_shape(self):
        s = compute_update_spectrum(self.delta_W)
        self.assertEqual(len(s), min(32, 16))
        # Should be non-negative and sorted descending
        for i in range(len(s) - 1):
            self.assertGreaterEqual(s[i], s[i + 1] - 1e-6)


class TestCLI(unittest.TestCase):
    def test_required_args(self):
        args = parse_args([
            "--base-model", "Qwen/Qwen3-1.7B",
            "--checkpoint", "runs/26/step=25",
            "--train-mode", "lora",
        ])
        self.assertEqual(args.base_model, "Qwen/Qwen3-1.7B")
        self.assertEqual(args.train_mode, "lora")
        self.assertEqual(args.top_k, 64)  # default

    def test_layers_parsing(self):
        args = parse_args([
            "--base-model", "x", "--checkpoint", "y", "--train-mode", "blocktt",
            "--layers", "0,5,10",
        ])
        self.assertEqual(args.layers, [0, 5, 10])

    def test_defaults(self):
        args = parse_args([
            "--base-model", "x", "--checkpoint", "y", "--train-mode", "svd",
        ])
        self.assertEqual(args.principal_alpha, 0.1)
        self.assertEqual(args.update_threshold, 0.01)
        self.assertEqual(args.output, "analysis_results/metrics.json")


class TestBuildReport(unittest.TestCase):
    def test_generates_html(self):
        from analysis.build_report import build_report

        # Create minimal valid JSON
        data = {
            "meta": {"base_model": "test", "checkpoint": "test", "train_mode": "lora",
                     "top_k": 4, "principal_alpha": 0.1, "timestamp": "2026-01-01"},
            "summary": {
                "layers": [0, 1],
                "modules": ["q_proj", "k_proj"],
                "nss": [[0.01, 0.02], [0.03, 0.04]],
                "max_principal_angle": [[1.0, 2.0], [3.0, 4.0]],
                "overlap_ratio": [[0.05, 0.06], [0.07, 0.08]],
                "relative_update_norm": [[0.001, 0.002], [0.003, 0.004]],
            },
            "detailed": {},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "test.json")
            html_path = os.path.join(tmpdir, "report.html")
            with open(json_path, "w") as f:
                json.dump(data, f)
            build_report([json_path], html_path)
            self.assertTrue(os.path.exists(html_path))
            with open(html_path) as f:
                content = f.read()
            self.assertIn("plotly", content.lower())
            self.assertIn("test", content)  # meta.base_model


class TestSingularRelationshipHelpers(unittest.TestCase):
    def test_factor_keys_for_svd(self):
        keys = factor_keys_for_mode("svd", "model.layers.0.self_attn.q_proj")
        self.assertEqual(keys["left"], "model.layers.0.self_attn.q_proj.svd_a")
        self.assertEqual(keys["right"], "model.layers.0.self_attn.q_proj.svd_b")
        self.assertEqual(keys["singular"], "model.layers.0.self_attn.q_proj.svd_s")

    def test_blocktt_flatten_shapes(self):
        l = torch.zeros(2, 6, 4)
        r = torch.zeros(3, 5, 4)
        self.assertEqual(tuple(flatten_heatmap_tensor("blocktt", "left", l).shape), (12, 4))
        self.assertEqual(tuple(flatten_heatmap_tensor("blocktt", "right", r).shape), (15, 4))

    def test_svd_singular_maps_shapes(self):
        s = torch.tensor([3.0, 2.0, 1.0])
        u = torch.zeros(5, 3)
        v = torch.zeros(3, 4)
        left_map, right_map = build_singular_maps("svd", s, u, v)
        self.assertEqual(tuple(left_map.shape), (5, 3))
        self.assertEqual(tuple(right_map.shape), (3, 4))

    def test_find_max_step(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            (run_dir / "step=1").mkdir()
            (run_dir / "step=10").mkdir()
            (run_dir / "step=50").mkdir()
            self.assertEqual(find_max_step(run_dir), 50)

    def test_resolve_trained_sides_svd_input(self):
        l = torch.zeros(8, 4)
        r = torch.zeros(4, 16)
        sides = resolve_trained_sides(
            "svd",
            {"train_position": "input"},
            l,
            r,
        )
        self.assertEqual(sides, ["right"])

    def test_resolve_trained_sides_blocktt_small(self):
        l = torch.zeros(1, 8, 16)  # 128 elems
        r = torch.zeros(2, 16, 16)  # 512 elems
        sides = resolve_trained_sides(
            "blocktt",
            {"train_position": "small"},
            l,
            r,
        )
        self.assertEqual(sides, ["left"])


class TestEndToEnd(unittest.TestCase):
    def test_lora_pipeline(self):
        """Full pipeline: create synthetic weights -> analyze -> build report."""
        from analysis.analyze_weights import parse_args, analyze
        from analysis.build_report import build_report

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = os.path.join(tmpdir, "base")
            ckpt_dir = os.path.join(tmpdir, "ckpt")
            os.makedirs(base_dir)
            os.makedirs(ckpt_dir)

            # Create synthetic base model weights (2 layers, all 7 target modules)
            torch.manual_seed(42)
            base_tensors = {}
            for l in range(2):
                base_tensors[f"model.layers.{l}.self_attn.q_proj.weight"] = torch.randn(16, 8)
                base_tensors[f"model.layers.{l}.self_attn.k_proj.weight"] = torch.randn(8, 8)
                base_tensors[f"model.layers.{l}.self_attn.v_proj.weight"] = torch.randn(8, 8)
                base_tensors[f"model.layers.{l}.self_attn.o_proj.weight"] = torch.randn(8, 16)
                base_tensors[f"model.layers.{l}.mlp.gate_proj.weight"] = torch.randn(32, 8)
                base_tensors[f"model.layers.{l}.mlp.up_proj.weight"] = torch.randn(32, 8)
                base_tensors[f"model.layers.{l}.mlp.down_proj.weight"] = torch.randn(8, 32)
            save_file(base_tensors, os.path.join(base_dir, "model.safetensors"))

            # Create synthetic LoRA adapter
            ckpt_tensors = {}
            for l in range(2):
                for mod, out_dim, in_dim in [
                    ("self_attn.q_proj", 16, 8), ("self_attn.k_proj", 8, 8),
                    ("self_attn.v_proj", 8, 8), ("self_attn.o_proj", 8, 16),
                    ("mlp.gate_proj", 32, 8), ("mlp.up_proj", 32, 8),
                    ("mlp.down_proj", 8, 32),
                ]:
                    r = 2
                    ckpt_tensors[f"base_model.model.model.layers.{l}.{mod}.lora_A.weight"] = torch.randn(r, in_dim)
                    ckpt_tensors[f"base_model.model.model.layers.{l}.{mod}.lora_B.weight"] = torch.randn(out_dim, r)
            save_file(ckpt_tensors, os.path.join(ckpt_dir, "adapter_model.safetensors"))

            # Write adapter config
            with open(os.path.join(ckpt_dir, "adapter_config.json"), "w") as f:
                json.dump({"lora_alpha": 4, "r": 2}, f)

            # Run analysis
            args = parse_args([
                "--base-model", base_dir,
                "--checkpoint", ckpt_dir,
                "--train-mode", "lora",
                "--top-k", "4",
                "--layers", "0,1",
                "--output", os.path.join(tmpdir, "results.json"),
            ])
            results = analyze(args)

            # Verify results structure
            self.assertEqual(len(results["summary"]["nss"]), 2)  # 2 layers
            self.assertEqual(len(results["summary"]["nss"][0]), 7)  # 7 modules
            self.assertIn("layer_0.q_proj", results["detailed"])

            # Write JSON and build report
            json_path = os.path.join(tmpdir, "results.json")
            with open(json_path, "w") as f:
                json.dump(results, f)
            html_path = os.path.join(tmpdir, "report.html")
            build_report([json_path], html_path)
            self.assertTrue(os.path.exists(html_path))


if __name__ == "__main__":
    unittest.main()
