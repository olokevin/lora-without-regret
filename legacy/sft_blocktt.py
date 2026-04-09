"""Compatibility wrapper for run_sft.py (blocktt mode)."""

import sys

from run_sft import main


if __name__ == "__main__":
    if any(arg == "--train-mode" or arg.startswith("--train-mode=") for arg in sys.argv[1:]):
        raise SystemExit("Do not pass --train-mode to sft_blocktt.py. Use run_sft.py directly.")
    main(["--train-mode", "blocktt", *sys.argv[1:]])
