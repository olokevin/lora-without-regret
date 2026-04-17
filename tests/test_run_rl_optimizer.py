import unittest

import torch

import run_rl
from optim.muon import Muon


class TestRunRLOptimizer(unittest.TestCase):
    def test_parse_args_optimizer_defaults(self):
        args = run_rl.parse_args(["--train-mode", "full"])
        self.assertEqual(args.optimizer, "adamw")
        self.assertEqual(args.weight_decay, 0.0)
        self.assertIsNone(args.lr_adam)
        self.assertIsNone(args.lr_embedding)
        self.assertIsNone(args.norm_method)

    def test_parse_args_muon_extra_args(self):
        args = run_rl.parse_args(
            [
                "--train-mode",
                "full",
                "--lr-adam",
                "2e-4",
                "--lr-embedding",
                "3e-4",
                "--norm-method",
                "shape",
            ]
        )
        self.assertEqual(args.lr_adam, 2e-4)
        self.assertEqual(args.lr_embedding, 3e-4)
        self.assertEqual(args.norm_method, "shape")

    def test_parse_args_lr_scheduler_defaults(self):
        args = run_rl.parse_args(["--train-mode", "full"])
        self.assertEqual(args.lr_scheduler, "none")
        self.assertEqual(args.warmup_ratio, 0.0)
        self.assertIsNone(args.cycle_length)
        self.assertEqual(args.min_lr_ratio, 0.1)

    def test_resolve_warmup_steps_uses_ceil(self):
        self.assertEqual(run_rl.resolve_warmup_steps(0.0, 10), 0)
        self.assertEqual(run_rl.resolve_warmup_steps(0.1, 10), 1)
        self.assertEqual(run_rl.resolve_warmup_steps(0.11, 10), 2)

    def test_build_optimizer_adamw(self):
        p = torch.nn.Parameter(torch.ones(4, 4))
        p.requires_grad = True
        named = [("proj.weight", p)]
        opt = run_rl.build_optimizer(
            argparse_namespace(optimizer="adamw", lr=1e-4, weight_decay=0.0),
            [p],
            named,
        )
        self.assertIsInstance(opt, torch.optim.AdamW)

    def test_build_optimizer_muon(self):
        p = torch.nn.Parameter(torch.ones(4, 4))
        p.requires_grad = True
        named = [("proj.weight", p)]
        opt = run_rl.build_optimizer(
            argparse_namespace(
                optimizer="muon",
                lr=1e-4,
                weight_decay=0.0,
                lr_adam=None,
                lr_embedding=None,
                norm_method=None,
            ),
            [p],
            named,
        )
        self.assertIsInstance(opt, Muon)

    def test_build_optimizer_muon_forwards_extra_args(self):
        p = torch.nn.Parameter(torch.ones(4, 4))
        bias = torch.nn.Parameter(torch.ones(4))
        emb = torch.nn.Parameter(torch.ones(16, 4))
        for t in (p, bias, emb):
            t.requires_grad = True
        named = [
            ("proj.weight", p),
            ("proj.bias", bias),
            ("embed_tokens.weight", emb),
        ]
        opt = run_rl.build_optimizer(
            argparse_namespace(
                optimizer="muon",
                lr=1e-4,
                weight_decay=0.0,
                lr_adam=2e-4,
                lr_embedding=3e-4,
                norm_method="shape",
            ),
            [p, bias, emb],
            named,
        )
        self.assertIsInstance(opt, Muon)
        group_lrs = sorted(group["lr"] for group in opt.param_groups)
        self.assertEqual(group_lrs, [1e-4, 2e-4, 3e-4])
        self.assertEqual(opt.param_groups[0]["norm_method"], "shape")

    def test_build_lr_scheduler_none_returns_none(self):
        p = torch.nn.Parameter(torch.ones(2, 2))
        opt = torch.optim.AdamW([p], lr=1e-4)
        args = argparse_namespace(
            lr_scheduler="none",
            warmup_ratio=0.0,
            cycle_length=None,
            min_lr_ratio=0.1,
        )
        scheduler = run_rl.build_lr_scheduler(args=args, optimizer=opt, num_training_steps=10)
        self.assertIsNone(scheduler)

    def test_build_lr_scheduler_linear_returns_lambda_lr(self):
        p = torch.nn.Parameter(torch.ones(2, 2))
        opt = torch.optim.AdamW([p], lr=1e-4)
        args = argparse_namespace(
            lr_scheduler="linear",
            warmup_ratio=0.2,
            cycle_length=None,
            min_lr_ratio=0.1,
        )
        scheduler = run_rl.build_lr_scheduler(args=args, optimizer=opt, num_training_steps=10)
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.LambdaLR)

    def test_build_lr_scheduler_cosine_validates_cycle_divisibility(self):
        p = torch.nn.Parameter(torch.ones(2, 2))
        opt = torch.optim.AdamW([p], lr=1e-4)
        args = argparse_namespace(
            lr_scheduler="cosine",
            warmup_ratio=0.0,
            cycle_length=6,
            min_lr_ratio=0.1,
        )
        with self.assertRaises(ValueError):
            run_rl.build_lr_scheduler(args=args, optimizer=opt, num_training_steps=10)

    def test_assert_muon_routing_svd(self):
        svd_a = torch.nn.Parameter(torch.randn(8, 4))
        svd_b = torch.nn.Parameter(torch.randn(4, 8))
        named = [
            ("model.layers.0.mlp.down_proj.svd_a", svd_a),
            ("model.layers.0.mlp.down_proj.svd_b", svd_b),
        ]
        opt = Muon(named, lr=1e-4, weight_decay=0.0)
        run_rl.assert_muon_routing("svd", named, opt)

    def test_assert_muon_routing_blocktt(self):
        btt_r = torch.nn.Parameter(torch.randn(1, 16, 16))
        btt_l = torch.nn.Parameter(torch.randn(1, 16, 16))
        btt_r.btt_rank = 4
        btt_r.btt_m = 4
        btt_l.btt_rank = 4
        btt_l.btt_n = 4
        named = [
            ("model.layers.0.mlp.down_proj.btt_r", btt_r),
            ("model.layers.0.mlp.down_proj.btt_l", btt_l),
        ]
        opt = Muon(named, lr=1e-4, weight_decay=0.0)
        run_rl.assert_muon_routing("blocktt", named, opt)

    def test_assert_muon_routing_lora(self):
        lora_a = torch.nn.Parameter(torch.randn(4, 8))
        lora_b = torch.nn.Parameter(torch.randn(8, 4))
        named = [
            ("model.layers.0.mlp.down_proj.lora_A.default.weight", lora_a),
            ("model.layers.0.mlp.down_proj.lora_B.default.weight", lora_b),
        ]
        opt = Muon(named, lr=1e-4, weight_decay=0.0)
        run_rl.assert_muon_routing("lora", named, opt)

    def test_assert_muon_routing_lora_full(self):
        lora_a = torch.nn.Parameter(torch.randn(4, 8))
        lora_b = torch.nn.Parameter(torch.randn(8, 4))
        named = [
            ("model.layers.0.mlp.down_proj.weight", torch.nn.Parameter(torch.randn(8, 8))),
            ("model.layers.0.mlp.down_proj.lora_A.default.weight", lora_a),
            ("model.layers.0.mlp.down_proj.lora_B.default.weight", lora_b),
        ]
        opt = Muon(named, lr=1e-4, weight_decay=0.0)
        run_rl.assert_muon_routing("lora_full", named, opt)



def argparse_namespace(**kwargs):
    class _Args:
        pass

    out = _Args()
    for key, value in kwargs.items():
        setattr(out, key, value)
    return out


if __name__ == "__main__":
    unittest.main()
