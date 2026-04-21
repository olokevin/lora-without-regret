# tests/test_run_sft_matrix_shell.py
import subprocess
import pathlib


def test_driver_script_parses():
    subprocess.run(["bash", "-n", "tools/run_sft_matrix.sh"], check=True)


def test_driver_script_enumerates_18_runs():
    src = pathlib.Path("tools/run_sft_matrix.sh").read_text()
    invocations = src.count('run_one "')
    # 2 singleton invocations + 4 looped method blocks = 6 textual invocations
    # expanding to 2 + 4*4 = 18 actual runs.
    assert invocations == 6, f"expected 6 textual run_one calls, got {invocations}"
