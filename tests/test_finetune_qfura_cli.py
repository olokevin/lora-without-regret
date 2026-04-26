import subprocess
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "ref" / "LIFT" / "src" / "finetune_qfura.py"


def _run(args_tail):
    """Run the script with the given args and capture stdout+stderr."""
    cmd = [sys.executable, str(_SCRIPT)] + args_tail
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=_REPO_ROOT)
    return result


class TestFinetuneQfuraCLI(unittest.TestCase):
    def test_rejects_train_position_not_small(self):
        result = _run([
            "--model_name_or_path", "bogus",
            "--quant_block_layout", "flat",
            "--train_position", "large",
            "--gradient_checkpointing",
        ])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "requires --train_position=small",
            result.stderr + result.stdout,
        )

    def test_rejects_missing_gradient_checkpointing(self):
        result = _run([
            "--model_name_or_path", "bogus",
            "--quant_block_layout", "flat",
            # no --gradient_checkpointing
        ])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "requires --gradient_checkpointing",
            result.stderr + result.stdout,
        )

    def test_rejects_unknown_quant_block_layout(self):
        result = _run([
            "--model_name_or_path", "bogus",
            "--quant_block_layout", "bogus_layout",
            "--gradient_checkpointing",
        ])
        self.assertNotEqual(result.returncode, 0)
        # argparse formats the error as: "argument --quant_block_layout: invalid choice"
        self.assertIn("quant_block_layout", result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
