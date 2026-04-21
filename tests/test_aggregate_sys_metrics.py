# tests/test_aggregate_sys_metrics.py
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from tools.aggregate_sys_metrics import (
    collect_runs,
    build_headline_table,
    build_sweep_table,
)


def _write_run(root: Path, method: str, rank, step_s: float, total: int, trainable: int):
    d = root / method / (f"r{rank}" if rank else "")
    d.mkdir(parents=True, exist_ok=True)
    (d / "sys_metrics.json").write_text(json.dumps({
        "method": method,
        "rank": rank,
        "base_params": 8_030_000_000,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": 100.0 * trainable / 8_030_000_000,
        "stored_extra_pct": 100.0 * max(0, total - 8_030_000_000) / 8_030_000_000,
        "steps_recorded": 300,
        "median_step_s": step_s,
        "total_wall_s": step_s * 300,
        "peak_alloc_bytes": 80 * 1024**3,
        "peak_reserved_bytes": 90 * 1024**3,
        "effective_tokens_per_step": 32768,
    }))


def test_collect_runs(tmp_path):
    _write_run(tmp_path, "lora", 64, 1.0, 8_060_000_000, 30_000_000)
    _write_run(tmp_path, "lora", 128, 1.1, 8_090_000_000, 60_000_000)
    _write_run(tmp_path, "fura", None, 0.9, 8_130_000_000, 100_000_000)

    runs = collect_runs(tmp_path)
    assert len(runs) == 3
    methods = {r["method"] for r in runs}
    assert methods == {"lora", "fura"}


def test_headline_table_picks_matched_ranks(tmp_path):
    _write_run(tmp_path, "lora", 16, 0.5, 8_050_000_000, 10_000_000)
    _write_run(tmp_path, "lora", 64, 1.0, 8_060_000_000, 30_000_000)
    _write_run(tmp_path, "lora", 128, 1.1, 8_090_000_000, 60_000_000)
    _write_run(tmp_path, "lift", 32, 1.2, 8_200_000_000, 200_000_000)
    _write_run(tmp_path, "lift", 64, 1.3, 8_300_000_000, 300_000_000)
    _write_run(tmp_path, "fura", None, 0.9, 8_130_000_000, 100_000_000)

    runs = collect_runs(tmp_path)
    md = build_headline_table(runs)
    assert "LoRA" in md and "64" in md
    assert "LIFT" in md and "32" in md
    assert "FuRA" in md
    assert "r=16" not in md


def test_sweep_table_has_all_runs(tmp_path):
    _write_run(tmp_path, "lora", 16, 0.5, 8_050_000_000, 10_000_000)
    _write_run(tmp_path, "lora", 64, 1.0, 8_060_000_000, 30_000_000)
    runs = collect_runs(tmp_path)
    md = build_sweep_table(runs)
    assert "| LoRA" in md or "| lora" in md
    assert md.count("LoRA") + md.count("lora") >= 2
