# tests/test_bench_merge_and_decode.py
import subprocess
import sys


def test_bench_merge_and_decode_help():
    result = subprocess.run(
        [sys.executable, "tools/bench_merge_and_decode.py", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--method" in result.stdout
    assert "--out" in result.stdout
