"""Collect all RL run summaries and print a best-per-method table.

Usage: python tools/collect_rl_results.py [--root /data/yequan/fura/rl_runs]
Emits a markdown-ready table on stdout. Reads each run's wandb-summary.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METHOD_DIRS = {
    "full": "full",
    "lora": "lora",
    "lora_full": "lora_full",
    "dora": "dora",
    "pissa": "pissa",
    "milora": "milora",
    "lift": "lift",
    "randlora": "randlora",
    "fura": "blocktt",  # project-internal name
    "svd": "svd",
}

EVAL_KEYS = [
    ("eval/accuracy", "eval_acc"),
    ("train/accuracy", "train_acc"),
    ("eval/MATH-500/accuracy", "math500"),
    ("eval/AMC23/accuracy", "amc23"),
    ("eval/AIME-24/accuracy", "aime24"),
    ("eval/AIME-25/accuracy", "aime25"),
    ("eval/Minerva/accuracy", "minerva"),
    ("_runtime", "runtime"),
    ("_step", "step"),
]


def pick_summary(run_dir: Path) -> Path | None:
    candidates = list(run_dir.glob("wandb/**/wandb-summary.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_size)


def collect(root: Path) -> list[dict]:
    rows: list[dict] = []
    for method_label, subdir in METHOD_DIRS.items():
        m = root / subdir
        if not m.exists():
            continue
        for run_dir in sorted(m.iterdir()):
            if not run_dir.is_dir():
                continue
            summary = pick_summary(run_dir)
            row = {"method": method_label, "run": run_dir.name}
            if summary is None:
                rows.append(row)
                continue
            try:
                data = json.loads(summary.read_text())
            except Exception:
                rows.append(row)
                continue
            for key, label in EVAL_KEYS:
                row[label] = data.get(key)
            rows.append(row)
    return rows


def fmt_pct(v):
    if v is None:
        return "-"
    try:
        return f"{float(v)*100:.1f}"
    except (TypeError, ValueError):
        return str(v)


def fmt_frac(v):
    if v is None:
        return "-"
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return str(v)


def main(root: Path) -> None:
    rows = collect(root)
    completed = [r for r in rows if r.get("eval_acc") is not None]
    completed_ext = [r for r in completed if r.get("math500") is not None]

    # Best-per-method by eval/accuracy (primary).
    by_method: dict[str, list[dict]] = {}
    for r in completed:
        by_method.setdefault(r["method"], []).append(r)

    print("## Best-per-method by primary eval/accuracy")
    print()
    print(
        "| Method | Runs | Best eval/acc | Mean eval/acc | Best run |"
    )
    print("|---|---:|---:|---:|---|")
    for method in METHOD_DIRS:
        runs = by_method.get(method, [])
        if not runs:
            print(f"| {method} | 0 | - | - | - |")
            continue
        runs_sorted = sorted(runs, key=lambda r: r["eval_acc"], reverse=True)
        best = runs_sorted[0]
        mean = sum(r["eval_acc"] for r in runs_sorted) / len(runs_sorted)
        print(
            f"| {method} | {len(runs)} | {fmt_pct(best['eval_acc'])}"
            f" | {fmt_pct(mean)} | `{best['run']}` |"
        )

    # Best extended-eval row per method.
    def ext_score(r):
        """Mean of available extended-eval accuracies.

        Missing entries are skipped rather than treated as zero, so an older
        run that never ran Minerva is not penalised relative to a newer run
        that has it. Ties by this mean are broken by eval/accuracy."""
        keys = ("math500", "amc23", "aime24", "aime25", "minerva")
        vals = [r.get(k) for k in keys]
        vals = [v for v in vals if v is not None]
        if not vals:
            return (0.0, r.get("eval_acc") or 0.0)
        return (sum(vals) / len(vals), r.get("eval_acc") or 0.0)

    print()
    print("## Best extended-eval row per method (by mean of available extended evals)")
    print()
    print(
        "| Method | Run | eval/acc | train/acc | MATH-500 | AMC23 | AIME-24 | AIME-25 | Minerva | mean(ext) |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    by_method_ext: dict[str, list[dict]] = {}
    for r in completed_ext:
        by_method_ext.setdefault(r["method"], []).append(r)
    for method in METHOD_DIRS:
        runs = by_method_ext.get(method, [])
        if not runs:
            continue
        runs_sorted = sorted(runs, key=ext_score, reverse=True)
        b = runs_sorted[0]
        score = ext_score(b)[0]
        print(
            f"| {method} | `{b['run']}` | "
            f"{fmt_pct(b['eval_acc'])} | {fmt_pct(b['train_acc'])} | "
            f"{fmt_pct(b['math500'])} | {fmt_pct(b['amc23'])} | "
            f"{fmt_pct(b['aime24'])} | {fmt_pct(b['aime25'])} | "
            f"{fmt_pct(b['minerva'])} | {score*100:.1f} |"
        )

    # Full list of extended-eval runs sorted.
    print()
    print("## All runs with extended eval (sorted by MATH-500)")
    print()
    print(
        "| Method | Run | eval/acc | train/acc | MATH-500 | AMC23 | AIME-24 | AIME-25 | Minerva |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(
        completed_ext,
        key=lambda r: r.get("math500") or 0,
        reverse=True,
    ):
        print(
            f"| {r['method']} | `{r['run']}` | "
            f"{fmt_pct(r['eval_acc'])} | {fmt_pct(r['train_acc'])} | "
            f"{fmt_pct(r['math500'])} | {fmt_pct(r['amc23'])} | "
            f"{fmt_pct(r['aime24'])} | {fmt_pct(r['aime25'])} | "
            f"{fmt_pct(r['minerva'])} |"
        )

    # All completed.
    print()
    print(f"Total: {len(rows)} runs, {len(completed)} completed, {len(completed_ext)} with extended eval.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("/data/yequan/fura/rl_runs"))
    args = p.parse_args()
    main(args.root)
