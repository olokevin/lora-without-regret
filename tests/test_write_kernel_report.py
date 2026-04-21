# tests/test_write_kernel_report.py
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import json
from pathlib import Path

from tools.write_kernel_report import render_report


def test_render_report_includes_key_sections():
    micro = [
        {"shape": "llama3_qproj", "B": 1024,
         "t_base_us": 1500, "t_fus_us": 1000, "speedup": 1.5,
         "mem_base_mb": 10, "mem_fus_mb": 9},
        {"shape": "llama3_upproj", "B": 2048,
         "t_base_us": 3000, "t_fus_us": 2500, "speedup": 1.2,
         "mem_base_mb": 20, "mem_fus_mb": 18},
    ]
    base_sys = {"median_step_s": 1.1, "effective_tokens_per_step": 32768,
                "peak_alloc_bytes": 80 * 1024**3, "total_wall_s": 300,
                "steps_recorded": 300}
    fus_sys = {"median_step_s": 1.0, "effective_tokens_per_step": 32768,
               "peak_alloc_bytes": 80 * 1024**3, "total_wall_s": 270,
               "steps_recorded": 300}
    report = render_report(micro, base_sys, fus_sys, gpu_name="H100")
    assert "Correctness" in report
    assert "Microbenchmark" in report
    assert "End-to-end SFT" in report
    assert "1.50" in report  # geometric mean or individual speedup
    assert "9.1" in report or "9." in report  # step time delta ~9.1%
