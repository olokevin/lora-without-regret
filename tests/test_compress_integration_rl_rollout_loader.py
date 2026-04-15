import os, sys
import unittest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import compress_integration as ci


class FakeTokenizer:
    """Character-level tokenizer for test determinism."""
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = [ord(c) % 256 for c in text]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids])}
        return {"input_ids": ids}

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 256 for c in text]


def fake_rollout(n):
    return [(f"prompt{i}-", f"completion{i}") for i in range(n)]


class TestRLRolloutCalibLoader(unittest.TestCase):
    def test_yields_masked_labels(self):
        tok = FakeTokenizer()
        loader = ci.build_rl_rollout_calib_loader(
            rl_rollout_fn=fake_rollout, tokenizer=tok,
            num_seqs=4, batch_size=2, max_length=32, seed=0,
        )
        batches = list(loader)
        self.assertEqual(sum(b["input_ids"].shape[0] for b in batches), 4)
        for batch in batches:
            self.assertIn("labels", batch)
            # Each row should have some -100s (prompt tokens masked) and some real labels.
            for row in batch["labels"]:
                n_masked = int((row == -100).sum())
                n_real = int((row != -100).sum() - (row == tok.pad_token_id).sum())
                self.assertGreater(n_masked, 0)
                self.assertGreater(n_real, 0)

    def test_respects_max_length(self):
        tok = FakeTokenizer()
        def long_rollout(n):
            return [("x" * 40, "y" * 40) for _ in range(n)]
        loader = ci.build_rl_rollout_calib_loader(
            rl_rollout_fn=long_rollout, tokenizer=tok,
            num_seqs=2, batch_size=2, max_length=16, seed=0,
        )
        batch = next(iter(loader))
        self.assertLessEqual(batch["input_ids"].shape[1], 16)


if __name__ == "__main__":
    unittest.main()
