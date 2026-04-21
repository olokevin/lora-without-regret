"""Consume microbench + baseline/fused sys_metrics.json and emit
docs/26_nips_fura_paper/kernel_eval_report.md.
"""
import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path


def render_report(micro: list[dict], base_sys: dict, fus_sys: dict, gpu_name: str) -> str:
    lines: list[str] = []
    lines.append("# FuRA Fused-Step2 Kernel: Evaluation Report")
    lines.append(f"_Auto-generated on {datetime.now(timezone.utc).isoformat(timespec='seconds')} by tools/write_kernel_report.py_")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- GPU: `{gpu_name}`")
    lines.append("- Model: Llama-3-8B, FuRA default corner (decomp=input_one_block, train=small, s=frozen)")
    lines.append("- bf16, gradient checkpointing ON, bs=8 × accum=2, seq-len 2048, `max_steps=300`")
    lines.append("")
    lines.append("## Correctness")
    lines.append("See `tests/test_fura_fused_kernel.py`; shape grid + fp32 gradcheck + non-contiguous coverage.")
    lines.append("")
    lines.append("## Microbenchmark (forward pass, per-layer)")
    lines.append("")
    lines.append("| Shape | B | Baseline (µs) | Fused (µs) | Speedup | Mem base (MB) | Mem fused (MB) |")
    lines.append("|-------|--:|---------------:|------------:|--------:|---------------:|----------------:|")
    for r in micro:
        lines.append(
            f"| {r['shape']} | {r['B']} | {r['t_base_us']:.1f} | {r['t_fus_us']:.1f} "
            f"| {r['speedup']:.2f}× | {r['mem_base_mb']:.1f} | {r['mem_fus_mb']:.1f} |"
        )
    gmean = math.exp(statistics.mean(math.log(max(r["speedup"], 1e-6)) for r in micro)) if micro else 0.0
    lines.append("")
    lines.append(f"**Geometric-mean speedup across all shape × batch cells: {gmean:.2f}×**")
    lines.append("")
    lines.append("## End-to-end SFT (300 optimizer steps)")
    lines.append("")
    lines.append("| Run | Median step (s) | Tokens/s | Peak GPU (GB) | Total wall (min) |")
    lines.append("|-----|-----------------:|---------:|--------------:|------------------:|")
    for label, sys_data in [("Baseline", base_sys), ("Fused", fus_sys)]:
        step = sys_data.get("median_step_s") or 0
        tok = sys_data["effective_tokens_per_step"] / step if step else 0
        peak_gb = (sys_data.get("peak_alloc_bytes") or 0) / (1024**3)
        wall_min = (sys_data.get("total_wall_s") or 0) / 60
        lines.append(f"| {label} | {step:.3f} | {tok:.0f} | {peak_gb:.1f} | {wall_min:.1f} |")
    base_step = base_sys.get("median_step_s") or 0
    fus_step = fus_sys.get("median_step_s") or 0
    if base_step and fus_step:
        delta_pct = 100 * (base_step - fus_step) / base_step
        lines.append(f"| **Δ** | **-{delta_pct:.1f}% step time** | | | |")
    lines.append("")
    lines.append("## Verdict")
    if base_step and fus_step:
        delta_pct = 100 * (base_step - fus_step) / base_step
        verdict = f"Fused Step 2 delivers {delta_pct:.1f}% end-to-end step-time reduction."
    else:
        verdict = "Insufficient data to form a verdict."
    lines.append(verdict)
    lines.append("")
    lines.append("## Raw artifacts")
    lines.append("- Microbench JSON, baseline `sys_metrics.json`, fused `sys_metrics.json` (see runner out dir).")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--micro", required=True)
    ap.add_argument("--sft_baseline", required=True)
    ap.add_argument("--sft_fused", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu_name", default="unknown")
    args = ap.parse_args()

    micro = json.loads(Path(args.micro).read_text())
    base_sys = json.loads(Path(args.sft_baseline).read_text())
    fus_sys = json.loads(Path(args.sft_fused).read_text())

    report = render_report(micro, base_sys, fus_sys, args.gpu_name)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
