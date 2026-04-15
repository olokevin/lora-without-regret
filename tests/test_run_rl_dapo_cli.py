import unittest

import torch

import run_rl_dapo


class _DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        assert add_generation_prompt is True
        return " || ".join(f"{m['role']}:{m['content']}" for m in messages)

    def encode(self, _text):
        return [1, 2, 3]


class TestRunRLDapoCLI(unittest.TestCase):
    def test_parse_defaults(self):
        args = run_rl_dapo.parse_args(["--train-mode", "full"])
        run_rl_dapo.apply_mode_defaults(args)
        self.assertEqual(args.loss_type, "grpo")
        self.assertEqual(args.train_dataset_id, run_rl_dapo.DEFAULT_TRAIN_DATASET_ID)
        self.assertTrue(args.final_eval)
        self.assertEqual(args.eval_majority_k, 32)

    def test_parse_explicit_dapo(self):
        args = run_rl_dapo.parse_args(
            [
                "--train-mode",
                "lora",
                "--loss-type",
                "dapo",
                "--clip-ratio-low",
                "0.1",
                "--clip-ratio-high",
                "0.3",
            ]
        )
        self.assertEqual(args.loss_type, "dapo")
        self.assertAlmostEqual(args.clip_ratio_low, 0.1)
        self.assertAlmostEqual(args.clip_ratio_high, 0.3)


class TestLossMath(unittest.TestCase):
    def test_dapo_loss_uses_asymmetric_clip(self):
        ratio = torch.tensor([[1.5, 0.7]], dtype=torch.float32)
        adv = torch.tensor([1.0], dtype=torch.float32)
        mask = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
        loss, clip_frac = run_rl_dapo.compute_policy_loss(
            "dapo",
            ratio,
            adv,
            mask,
            clip_ratio_low=0.2,
            clip_ratio_high=0.1,
        )
        # clipped_ratio = [1.1, 0.8], objective=min([1.5,0.7],[1.1,0.8])=[1.1,0.7]
        self.assertAlmostEqual(loss.item(), -0.9, places=5)
        self.assertAlmostEqual(clip_frac, 1.0, places=5)

    def test_grpo_loss_matches_sequence_mean(self):
        ratio = torch.tensor([[1.2, 0.8], [1.0, 1.0]], dtype=torch.float32)
        adv = torch.tensor([1.0, -1.0], dtype=torch.float32)
        mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
        loss, clip_frac = run_rl_dapo.compute_policy_loss(
            "grpo",
            ratio,
            adv,
            mask,
            clip_ratio_low=0.2,
            clip_ratio_high=0.2,
        )
        # prompt1: -(1.2+0.8)/2=-1.0, prompt2: -(-1.0*1.0)/1=1.0 => mean=0
        self.assertAlmostEqual(loss.item(), 0.0, places=5)
        self.assertEqual(clip_frac, 0.0)


class TestDataHelpers(unittest.TestCase):
    def test_process_math_example_dapo_schema(self):
        tokenizer = _DummyTokenizer()
        example = {
            "prompt": [{"role": "user", "content": "2+2?"}],
            "reward_model": {"ground_truth": r"\boxed{4}"},
        }
        processed = run_rl_dapo.process_math_example(example, tokenizer, "{question}")
        self.assertIn("user:2+2?", processed["prompt"])
        self.assertEqual(processed["answer"], "4")

    def test_unique_by_problem(self):
        rows = [
            {"problem": "A", "answer": "1"},
            {"problem": "A", "answer": "1"},
            {"problem": "B", "answer": "2"},
        ]
        unique = run_rl_dapo.unique_by_problem(rows)
        self.assertEqual(len(unique), 2)


if __name__ == "__main__":
    unittest.main()
