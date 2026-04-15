import os, sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from compress.decomposition import DecompositionConfig


class TestDecompositionConfigExtraFields(unittest.TestCase):
    def test_defaults(self):
        cfg = DecompositionConfig(train_mode="btt_llm_v2")
        self.assertIsNone(cfg.s_merged_to)
        self.assertTrue(cfg.factorize_by_head)

    def test_custom_values(self):
        cfg = DecompositionConfig(
            train_mode="btt_llm_v2",
            s_merged_to="trainable",
            factorize_by_head=False,
        )
        self.assertEqual(cfg.s_merged_to, "trainable")
        self.assertFalse(cfg.factorize_by_head)


if __name__ == "__main__":
    unittest.main()
