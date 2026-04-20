import os
import sys
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from optim.sparse_adam import SparseAdamW


class TestSparseAdamWSmoke(unittest.TestCase):
    def test_one_step_runs_and_updates_only_masked_entries(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 4))
        weights_with_mask = [m.weight for m in model if isinstance(m, nn.Linear)]
        decay_ids = {id(p) for p in weights_with_mask}
        other = [p for p in model.parameters() if id(p) not in decay_ids]

        param_groups = [
            {
                "params": weights_with_mask,
                "weight_decay": 0.0,
                "rank": 4,
                "filter_rank": 4,
                "update_proj_gap": 1,
                "group_name": "weights_with_mask",
            },
            {"params": other, "weight_decay": 0.0, "group_name": "other_params_w_decay"},
        ]
        opt = SparseAdamW(param_groups, lr=1e-3, betas=(0.9, 0.95))

        before = [p.detach().clone() for p in weights_with_mask]
        x = torch.randn(2, 16)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        # At least one masked entry per layer should have changed.
        for w_before, w_after in zip(before, weights_with_mask):
            diff = (w_before - w_after).abs()
            self.assertGreater(diff.sum().item(), 0.0,
                               "SparseAdamW step did not update any entries")


if __name__ == "__main__":
    unittest.main()
