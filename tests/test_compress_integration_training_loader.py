import os, sys
import unittest
import torch
from torch.utils.data import Dataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import compress_integration as ci


class ToyDataset(Dataset):
    def __init__(self, n=50, seqlen=6):
        self.items = [
            {"input_ids": torch.arange(seqlen) + i,
             "labels": torch.arange(seqlen) + i}
            for i in range(n)
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def toy_collate(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


class TestTrainingDataCalibLoader(unittest.TestCase):
    def test_yields_contract_dicts(self):
        loader = ci.build_training_data_calib_loader(
            ToyDataset(), toy_collate, num_seqs=16, batch_size=4, seed=0,
        )
        batches = list(loader)
        self.assertEqual(len(batches), 4)
        for batch in batches:
            self.assertIn("input_ids", batch)
            self.assertEqual(batch["input_ids"].shape, (4, 6))
            self.assertIn("labels", batch)

    def test_respects_num_seqs(self):
        loader = ci.build_training_data_calib_loader(
            ToyDataset(), toy_collate, num_seqs=10, batch_size=4, seed=0,
        )
        total = sum(b["input_ids"].shape[0] for b in loader)
        self.assertEqual(total, 10)

    def test_deterministic_under_seed(self):
        def first_ids(seed):
            loader = ci.build_training_data_calib_loader(
                ToyDataset(), toy_collate, num_seqs=8, batch_size=4, seed=seed,
            )
            return next(iter(loader))["input_ids"].tolist()
        self.assertEqual(first_ids(0), first_ids(0))

    def test_num_seqs_larger_than_dataset_clamped(self):
        loader = ci.build_training_data_calib_loader(
            ToyDataset(n=5), toy_collate, num_seqs=100, batch_size=2, seed=0,
        )
        total = sum(b["input_ids"].shape[0] for b in loader)
        self.assertEqual(total, 5)


if __name__ == "__main__":
    unittest.main()
