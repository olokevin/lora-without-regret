import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


class _FakeTokenizer:
    """Stand-in tokenizer; chat-template is unused since we patch the dataset loader."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return messages[0]["content"]


def _fake_dataset(_name, _tokenizer, _template):
    # Two problems, gold = "42" each.
    return [
        {"prompt": "p1", "gold_answer": "42", "n_samples": 1, "temperature": 0.0, "top_p": 1.0},
        {"prompt": "p2", "gold_answer": "42", "n_samples": 1, "temperature": 0.0, "top_p": 1.0},
    ]


class TestMathVerifyEval(unittest.TestCase):
    def test_grades_two_problems_one_correct(self):
        import math_verify_eval

        # Fake LLM returns: problem 1 → correct boxed{42}, problem 2 → wrong boxed{0}.
        fake_llm = MagicMock()
        fake_outputs = [
            MagicMock(outputs=[MagicMock(text=r"reasoning... \boxed{42}")]),
            MagicMock(outputs=[MagicMock(text=r"reasoning... \boxed{0}")]),
        ]
        fake_llm.generate.return_value = fake_outputs

        results = math_verify_eval.math_verify_eval(
            model=None,
            tokenizer=_FakeTokenizer(),
            datasets=["MATH-500"],
            max_tokens=128,
            prompt_template_path=None,
            vllm_kwargs={"llm": fake_llm},
            _dataset_loader=_fake_dataset,
        )

        self.assertIn("MATH-500", results["datasets"])
        ds = results["datasets"]["MATH-500"]
        self.assertEqual(ds["n_total"], 2)
        self.assertEqual(ds["n_correct"], 1)
        self.assertAlmostEqual(ds["accuracy"], 0.5)

    def test_unparseable_counts_as_incorrect(self):
        import math_verify_eval

        fake_llm = MagicMock()
        # No \boxed{} in either output.
        fake_outputs = [
            MagicMock(outputs=[MagicMock(text="some prose")]),
            MagicMock(outputs=[MagicMock(text="more prose")]),
        ]
        fake_llm.generate.return_value = fake_outputs

        results = math_verify_eval.math_verify_eval(
            model=None,
            tokenizer=_FakeTokenizer(),
            datasets=["MATH-500"],
            max_tokens=128,
            prompt_template_path=None,
            vllm_kwargs={"llm": fake_llm},
            _dataset_loader=_fake_dataset,
        )

        ds = results["datasets"]["MATH-500"]
        self.assertEqual(ds["n_correct"], 0)
        self.assertEqual(ds["n_unparseable"], 2)


if __name__ == "__main__":
    unittest.main()
