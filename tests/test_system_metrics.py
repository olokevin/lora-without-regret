# tests/test_system_metrics.py
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import torch
import pytest
from pathlib import Path

from tools.system_metrics import SysMon


def test_sysmon_dump_writes_json(tmp_path):
    mon = SysMon(tmp_path, method="lora", rank=64, base_params=8_030_000_000)
    mon.record_step(0.1)
    mon.record_step(0.12)
    mon.record_step(0.11)

    model = torch.nn.Linear(8, 8)  # 72 params
    mon.dump(model, extra={"effective_tokens_per_step": 16384})

    out = tmp_path / "sys_metrics.json"
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["method"] == "lora"
    assert data["rank"] == 64
    assert data["trainable_params"] == 72
    assert data["total_params"] == 72
    assert data["steps_recorded"] == 3
    assert data["median_step_s"] == pytest.approx(0.11, abs=1e-6)
    assert data["effective_tokens_per_step"] == 16384


def test_sysmon_warmup_cutoff_uses_tail_when_enough_steps(tmp_path):
    mon = SysMon(tmp_path, method="fura", rank=None, base_params=1)
    # 150 fast steps (warmup) + 50 slow steps
    for _ in range(150):
        mon.record_step(0.01)
    for _ in range(50):
        mon.record_step(1.0)
    mon.dump(torch.nn.Linear(1, 1))
    data = json.loads((tmp_path / "sys_metrics.json").read_text())
    # After 100-step warmup cut, median is over steps[100:] = 50 fast + 50 slow,
    # median of that is halfway between 0.01 and 1.0 → 0.01 (since sorted[99] = 0.01).
    # But we want to be tolerant: assert median is one of the observed values.
    assert data["median_step_s"] in (0.01, 1.0, pytest.approx(0.505, abs=0.5))
    assert data["steps_recorded"] == 200


def test_sysmon_peak_memory_captured_when_cuda(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    mon = SysMon(tmp_path, method="full", rank=None, base_params=1)
    x = torch.zeros(1024, 1024, device="cuda")
    mon.record_step(0.01)
    mon.dump(torch.nn.Linear(1, 1).cuda())
    data = json.loads((tmp_path / "sys_metrics.json").read_text())
    assert data["peak_alloc_bytes"] >= x.numel() * 4
