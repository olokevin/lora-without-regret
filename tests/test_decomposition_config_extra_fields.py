import dataclasses
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

    def test_asdict_roundtrip(self):
        cfg = DecompositionConfig(train_mode="btt_llm_v2")
        d = dataclasses.asdict(cfg)
        self.assertIn("s_merged_to", d)
        self.assertIn("factorize_by_head", d)
        self.assertIsNone(d["s_merged_to"])
        self.assertTrue(d["factorize_by_head"])

    def test_help_metadata_present(self):
        cfg = DecompositionConfig(train_mode="btt_llm_v2")
        fields_by_name = {f.name: f for f in dataclasses.fields(cfg)}
        self.assertIn("help", fields_by_name["s_merged_to"].metadata)
        self.assertIn("help", fields_by_name["factorize_by_head"].metadata)
        self.assertTrue(fields_by_name["s_merged_to"].metadata["help"])
        self.assertTrue(fields_by_name["factorize_by_head"].metadata["help"])


if __name__ == "__main__":
    unittest.main()
