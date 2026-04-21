"""Walk a run-matrix output root, read every sys_metrics.json, emit:
  - tab_system_perf_commonsense_headline.{md,tex}
  - tab_system_perf_commonsense_sweep.{md,tex}
  - sys_perf_vs_rank.pdf   (if matplotlib available)
"""
import argparse
import json
from pathlib import Path
from typing import Any

HEADLINE_RANKS = {
    "lora": 64,
    "dora": 64,
    "randlora": 64,
    "lift": 32,
}
METHOD_DISPLAY = {
    "full": "Full FT",
    "lora": "LoRA",
    "dora": "DoRA",
    "randlora": "RandLoRA",
    "lift": "LIFT",
    "fura": "FuRA",
    "blocktt": "FuRA",
    "sparse": "LIFT",
}
METHOD_ORDER = ["full", "lora", "dora", "randlora", "lift", "fura"]


def collect_runs(root: Path) -> list[dict[str, Any]]:
    runs = []
    for p in Path(root).rglob("sys_metrics.json"):
        try:
            runs.append(json.loads(p.read_text()))
        except Exception as e:
            print(f"[warn] failed to read {p}: {e}")
    return runs


def _norm_method(m: str) -> str:
    if m == "blocktt":
        return "fura"
    if m == "sparse":
        return "lift"
    return m


def build_headline_table(runs: list[dict]) -> str:
    lines = [
        "| Method | Rank | Trainable % | Stored extra % | Step (s) | Tokens/s | Peak GPU (GB) |",
        "|--------|-----:|-------------|-----------------|----------|----------|---------------|",
    ]
    for method in METHOD_ORDER:
        want_rank = HEADLINE_RANKS.get(method)
        candidates = [r for r in runs if _norm_method(r["method"]) == method]
        if want_rank is not None:
            candidates = [r for r in candidates if r.get("rank") == want_rank]
        if not candidates:
            continue
        r = candidates[0]
        step_s = r.get("median_step_s") or 0
        tok_s = r["effective_tokens_per_step"] / step_s if step_s else 0
        peak_gb = (r.get("peak_alloc_bytes") or 0) / (1024**3)
        rank_str = str(r.get("rank")) if r.get("rank") is not None else "—"
        lines.append(
            f"| {METHOD_DISPLAY[method]} | {rank_str} "
            f"| {r['trainable_pct']:.2f} | {r['stored_extra_pct']:.2f} "
            f"| {step_s:.3f} | {tok_s:.0f} | {peak_gb:.1f} |"
        )
    return "\n".join(lines)


def build_sweep_table(runs: list[dict]) -> str:
    lines = [
        "| Method | Rank | Trainable % | Stored extra % | Step (s) | Tokens/s | Peak GPU (GB) |",
        "|--------|-----:|-------------|-----------------|----------|----------|---------------|",
    ]
    def _key(r):
        m = _norm_method(r["method"])
        return (METHOD_ORDER.index(m) if m in METHOD_ORDER else 999,
                r.get("rank") if r.get("rank") is not None else -1)
    for r in sorted(runs, key=_key):
        method = _norm_method(r["method"])
        step_s = r.get("median_step_s") or 0
        tok_s = r["effective_tokens_per_step"] / step_s if step_s else 0
        peak_gb = (r.get("peak_alloc_bytes") or 0) / (1024**3)
        rank_str = str(r.get("rank")) if r.get("rank") is not None else "—"
        lines.append(
            f"| {METHOD_DISPLAY.get(method, method)} | {rank_str} "
            f"| {r['trainable_pct']:.2f} | {r['stored_extra_pct']:.2f} "
            f"| {step_s:.3f} | {tok_s:.0f} | {peak_gb:.1f} |"
        )
    return "\n".join(lines)


def build_headline_tex(runs: list[dict]) -> str:
    body = []
    for method in METHOD_ORDER:
        want_rank = HEADLINE_RANKS.get(method)
        candidates = [r for r in runs if _norm_method(r["method"]) == method]
        if want_rank is not None:
            candidates = [r for r in candidates if r.get("rank") == want_rank]
        if not candidates:
            continue
        r = candidates[0]
        step_s = r.get("median_step_s") or 0
        tok_s = r["effective_tokens_per_step"] / step_s if step_s else 0
        peak_gb = (r.get("peak_alloc_bytes") or 0) / (1024**3)
        rank_str = str(r.get("rank")) if r.get("rank") is not None else "--"
        body.append(
            f"{METHOD_DISPLAY[method]} & {rank_str} & "
            f"{r['trainable_pct']:.2f} & {r['stored_extra_pct']:.2f} & "
            f"{step_s:.3f} & {tok_s:.0f} & {peak_gb:.1f} \\\\"
        )
    return (
        "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
        "Method & Rank & Trainable (\\%) & Stored extra (\\%) & Step (s) & Tokens/s & Peak GPU (GB) \\\\\n"
        "\\midrule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}"
    )


def plot_sweep(runs: list[dict], out_pdf: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not available — skipping sweep plot")
        return
    import collections
    by_method: dict[str, list] = collections.defaultdict(list)
    for r in runs:
        by_method[_norm_method(r["method"])].append(r)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for method, rs in by_method.items():
        rs = sorted(rs, key=lambda r: (r.get("rank") is None, r.get("rank") or 0))
        ranks = [r.get("rank") for r in rs]
        steps = [r.get("median_step_s") or 0 for r in rs]
        peaks = [(r.get("peak_alloc_bytes") or 0) / 1024**3 for r in rs]
        trains = [r["trainable_pct"] for r in rs]
        ranked = [(rk, s, p, t) for rk, s, p, t in zip(ranks, steps, peaks, trains) if rk is not None]
        if ranked:
            rk, s, p, t = zip(*ranked)
            axs[0].plot(rk, s, marker="o", label=METHOD_DISPLAY.get(method, method))
            axs[1].plot(rk, p, marker="o", label=METHOD_DISPLAY.get(method, method))
            axs[2].plot(rk, t, marker="o", label=METHOD_DISPLAY.get(method, method))
        else:
            for ax, y in zip(axs, [steps[0], peaks[0], trains[0]]):
                ax.axhline(y, linestyle="--", label=METHOD_DISPLAY.get(method, method))
    for ax, title, ylabel in zip(
        axs,
        ["Step time", "Peak GPU", "Trainable %"],
        ["seconds", "GB", "% of base"],
    ):
        ax.set_xlabel("rank")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_root", required=True)
    p.add_argument("--out_dir", default="docs/26_nips_fura_paper/tables")
    p.add_argument("--plot_dir", default="docs/26_nips_fura_paper/figs")
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = collect_runs(Path(args.runs_root))
    print(f"collected {len(runs)} runs")

    (out_dir / "tab_system_perf_commonsense_headline.md").write_text(build_headline_table(runs))
    (out_dir / "tab_system_perf_commonsense_sweep.md").write_text(build_sweep_table(runs))
    (out_dir / "tab_system_perf_commonsense_headline.tex").write_text(build_headline_tex(runs))
    plot_sweep(runs, Path(args.plot_dir) / "sys_perf_vs_rank.pdf")
    print(f"wrote tables to {out_dir}")


if __name__ == "__main__":
    main()
