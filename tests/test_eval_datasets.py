import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


class TestRegistry(unittest.TestCase):
    def test_registry_has_exactly_five_expected_datasets(self):
        from eval_datasets import REGISTRY

        self.assertEqual(
            set(REGISTRY.keys()),
            {"MATH-500", "AIME-24", "AIME-25", "AMC23", "Minerva"},
        )

    def test_aime_datasets_use_multi_sample_defaults(self):
        from eval_datasets import REGISTRY

        for name in ("AIME-24", "AIME-25"):
            spec = REGISTRY[name]
            self.assertEqual(spec.n_samples, 8, f"{name} should default to 8 samples")
            self.assertAlmostEqual(spec.temperature, 0.6)
            self.assertAlmostEqual(spec.top_p, 0.95)

    def test_greedy_datasets_use_temperature_zero(self):
        from eval_datasets import REGISTRY

        for name in ("MATH-500", "AMC23", "Minerva"):
            spec = REGISTRY[name]
            self.assertEqual(spec.n_samples, 1, f"{name} should default to 1 sample")
            self.assertEqual(spec.temperature, 0.0)
            self.assertEqual(spec.top_p, 1.0)

    def test_dataset_spec_is_frozen(self):
        from eval_datasets import REGISTRY
        import dataclasses

        spec = REGISTRY["MATH-500"]
        with self.assertRaises(dataclasses.FrozenInstanceError):
            spec.hf_id = "other"


@unittest.skipUnless(
    os.environ.get("RUN_HF_TESTS") == "1",
    "Set RUN_HF_TESTS=1 to enable HF Hub round-trip tests.",
)
class TestLoadEvalDatasetHFRoundTrip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        cls.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
        with open(
            os.path.join(os.path.dirname(__file__), os.path.pardir, "boxed.prompt"),
            "r",
            encoding="utf-8",
        ) as f:
            cls.template = f.read().strip()

    def test_load_each_dataset_returns_nonempty(self):
        from eval_datasets import load_eval_dataset, known_dataset_names

        for name in known_dataset_names():
            with self.subTest(dataset=name):
                items = load_eval_dataset(name, self.tokenizer, self.template)
                self.assertGreater(len(items), 0, f"{name} returned 0 items")
                first = items[0]
                self.assertIn("prompt", first)
                self.assertIn("gold_answer", first)
                self.assertIn("\\boxed", first["prompt"])


if __name__ == "__main__":
    unittest.main()
