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


class TinyQwenLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "q_proj": nn.Linear(16, 16, bias=False),
                "k_proj": nn.Linear(16, 16, bias=False),
                "v_proj": nn.Linear(16, 16, bias=False),
                "o_proj": nn.Linear(16, 16, bias=False),
                "gate_proj": nn.Linear(16, 32, bias=False),
                "up_proj": nn.Linear(16, 32, bias=False),
                "down_proj": nn.Linear(32, 16, bias=False),
            })
            for _ in range(2)
        ])
        self.embed = nn.Embedding(16, 16)
        self.lm_head = nn.Linear(16, 16, bias=False)

    def forward(self, input_ids, labels=None, attention_mask=None, **kw):
        x = self.embed(input_ids)
        for block in self.layers:
            x = block["q_proj"](x)
            x = block["gate_proj"](x)
            x = block["down_proj"](x)
        logits = self.lm_head(x)
        out = type("Out", (), {})()
        out.logits = logits
        if labels is not None:
            out.loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1), ignore_index=-100,
            )
        return out


class TinyDS(Dataset):
    def __init__(self, n=16, L=4):
        self.items = [{
            "input_ids": torch.randint(0, 16, (L,)),
            "labels": torch.randint(0, 16, (L,)),
        } for _ in range(n)]
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def _collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def _parse(calib_mode, **kv):
    p = argparse.ArgumentParser()
    p.add_argument("--train-mode", default="blocktt")
    p.add_argument("--blocktt-rank", default="full")
    p.add_argument("--trainable-type", default="all")
    p.add_argument("--decomp-mode", default="square")
    p.add_argument("--train-position", default=kv.get("train_position", "small"))
    p.add_argument("--s-merged-to", default=None)
    p.add_argument("--blocktt-factorize-by-head", action="store_true", default=True)
    ci.add_calibrated_btt_args(p, hyphen_style=True)
    return p.parse_args([
        "--calib-mode", calib_mode, "--calib-source", "training_data",
        "--calib-num-seqs", "8", "--calib-batch-size", "4",
    ])


@unittest.skipUnless(torch.cuda.is_available(), "calibrated BTT requires CUDA")
class TestPipeline(unittest.TestCase):
    def test_v2_installs_btt_layers_and_respects_train_position(self):
        args = _parse("v2", train_position="small")
        model = TinyQwenLike().cuda()
        loader = ci.build_calib_loader(
            args, tokenizer=None,
            training_dataset=TinyDS(), training_collate_fn=_collate,
        )
        model, stats = ci.apply_calibrated_btt(model, args, calib_loader=loader)
        self.assertGreater(stats["num_btt_layers"], 0)
        # train_position=small: for each layer, exactly one of (btt_l, btt_r) trains.
        for module in model.modules():
            if isinstance(module, BTTLinear):
                self.assertNotEqual(module.btt_l.requires_grad, module.btt_r.requires_grad)
        # Forward runs without error
        x = torch.randint(0, 16, (2, 4), device="cuda")
        out = model(input_ids=x)
        self.assertIsNotNone(out.logits)


if __name__ == "__main__":
    unittest.main()
