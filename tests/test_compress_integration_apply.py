import argparse
import os, sys
import unittest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import compress_integration as ci
from compress.btt.btt_linear import BTTLinear


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "q_proj": nn.Linear(16, 16),
                "k_proj": nn.Linear(16, 16),
                "v_proj": nn.Linear(16, 16),
                "o_proj": nn.Linear(16, 16),
                "gate_proj": nn.Linear(16, 16),
                "up_proj": nn.Linear(16, 16),
                "down_proj": nn.Linear(16, 16),
            })
        ])
        self.lm_head = nn.Linear(16, 32)

    def forward(self, input_ids, labels=None, **kwargs):
        x = torch.nn.functional.one_hot(input_ids, num_classes=16).float()
        for block in self.layers:
            x = block["q_proj"](x)
        logits = self.lm_head(x)
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100,
            )
            return type("Out", (), {"loss": loss, "logits": logits})()
        return type("Out", (), {"logits": logits})()


class CalibDS(Dataset):
    def __init__(self, n=16, seqlen=4):
        self.items = [
            {"input_ids": torch.randint(0, 16, (seqlen,)),
             "labels": torch.randint(0, 16, (seqlen,))}
            for _ in range(n)
        ]
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def _collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--train-mode", default="blocktt")
    p.add_argument("--blocktt-rank", default="full")
    p.add_argument("--trainable-type", default="all")
    p.add_argument("--decomp-mode", default="square")
    p.add_argument("--train-position", default="small")
    p.add_argument("--s-merged-to", default=None)
    p.add_argument("--blocktt-factorize-by-head", action="store_true", default=True)
    ci.add_calibrated_btt_args(p, hyphen_style=True)
    return p.parse_args([
        "--calib-mode", "v2", "--calib-source", "training_data",
        "--calib-num-seqs", "8", "--calib-batch-size", "4",
    ])


class TestApply(unittest.TestCase):
    def test_build_calib_loader_training_data(self):
        args = _parse()
        loader = ci.build_calib_loader(
            args, tokenizer=None, training_dataset=CalibDS(),
            training_collate_fn=_collate,
        )
        self.assertIsNotNone(loader)
        batch = next(iter(loader))
        self.assertIn("input_ids", batch)

    def test_build_calib_loader_returns_none_for_calib_none(self):
        p = argparse.ArgumentParser()
        p.add_argument("--train-mode", default="blocktt")
        p.add_argument("--blocktt-rank", default="full")
        ci.add_calibrated_btt_args(p, hyphen_style=True)
        args = p.parse_args(["--train-mode", "blocktt"])
        self.assertIsNone(
            ci.build_calib_loader(args, tokenizer=None)
        )

    @unittest.skipUnless(torch.cuda.is_available(), "calibrated BTT requires CUDA")
    def test_apply_calibrated_btt_installs_btt_linear(self):
        args = _parse()
        model = TinyModel().cuda()
        loader = ci.build_calib_loader(
            args, tokenizer=None, training_dataset=CalibDS(),
            training_collate_fn=_collate,
        )
        model, stats = ci.apply_calibrated_btt(model, args, calib_loader=loader)
        self.assertGreater(stats["num_btt_layers"], 0)
        btt_layers = [m for m in model.modules() if isinstance(m, BTTLinear)]
        self.assertGreater(len(btt_layers), 0)


if __name__ == "__main__":
    unittest.main()
