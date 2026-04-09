"""Compatibility wrapper for run_sft.py (full mode)."""

import sys

from run_sft import main


if __name__ == "__main__":
    if any(arg == "--train-mode" or arg.startswith("--train-mode=") for arg in sys.argv[1:]):
        raise SystemExit("Do not pass --train-mode to sft_full.py. Use run_sft.py directly.")
    main(["--train-mode", "full", *sys.argv[1:]])
