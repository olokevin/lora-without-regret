import subprocess
import sys


def test_finetune_sft_accepts_max_steps_flag():
    result = subprocess.run(
        [sys.executable, "ref/LIFT/src/finetune_sft.py", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--max_steps" in result.stdout
