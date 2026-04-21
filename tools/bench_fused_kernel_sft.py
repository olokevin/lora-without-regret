#!/usr/bin/env python
# tools/bench_fused_kernel_sft.py
"""Run the FuRA commonsense SFT script twice (baseline vs fused) at
max_steps=300, on the GPU indicated by CUDA_VISIBLE_DEVICES, and leave
sys_metrics.json files in two sibling output directories.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(env_updates: dict[str, str], out_dir: Path):
    env = os.environ.copy()
    env.update(env_updates)
    env["OUTPUT"] = str(out_dir)
    env["run_name"] = out_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    lift_dir = repo / "ref" / "LIFT"
    script = lift_dir / "bash_scripts" / "finetune_commonsense_blocktt.sh"
    print(f"[bench] launching {script} → {out_dir}  (FURA_FUSED_STEP2={env.get('FURA_FUSED_STEP2', '0')})")
    subprocess.run(["bash", str(script)], cwd=str(lift_dir), env=env, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--max_steps", type=int, default=300)
    args = ap.parse_args()
    out_root = Path(args.out_root)

    common = {
        "MODEL": "meta-llama/Meta-Llama-3-8B",
        "decomp_mode": "input_one_block",
        "train_position": "small",
        "s_merged_to": "frozen",
        "blocktt_rank": "full",
        "lr": "2e-4",
        "seed": "43",
        "MAX_STEPS": str(args.max_steps),
    }
    run({**common, "FURA_FUSED_STEP2": "0"}, out_root / "baseline")
    run({**common, "FURA_FUSED_STEP2": "1"}, out_root / "fused")


if __name__ == "__main__":
    main()
