# tests/test_bench_fbopt.py
import json
import subprocess
import sys
from pathlib import Path


def test_bench_fbopt_emits_json_for_toy_shape(tmp_path):
    out = tmp_path / "fbopt.json"
    result = subprocess.run(
        [
            sys.executable,
            "tools/bench_fbopt.py",
            "--method", "toy",
            "--d_in", "64",
            "--d_out", "64",
            "--batch", "32",
            "--iters", "3",
            "--warmup", "1",
            "--out", str(out),
            "--device", "cpu",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(out.read_text())
    assert data["method"] == "toy"
    assert "fwd_ms" in data
    assert "fwd_bwd_ms" in data
    assert "fwd_bwd_opt_ms" in data
    assert data["fwd_ms"] >= 0.0
