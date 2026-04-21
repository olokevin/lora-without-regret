# tests/test_bench_fused_kernel.py
import subprocess
import sys


def test_bench_fused_kernel_help():
    result = subprocess.run(
        [sys.executable, "tools/bench_fused_kernel.py", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--out" in result.stdout
