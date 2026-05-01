"""Generate the 6-panel motivating-example figure (Figure 2) for the paper.

Panels (2x3, first row a/b/c, second row d/e/f):
  (a) Full FT — effective rank of (W'-W0) vs layer index, two curves: q_proj and up_proj
  (b) Full FT — heatmap of right singular vector update |V'-V0| for one layer
  (c) Learning curves (Full FT, SVD FT) — accuracy or loss
  (d) FuRA   — effective rank of (W'-W0) vs layer index, two curves: q_proj and up_proj
  (e) FuRA   — heatmap of core-R update |R'-R0|, plotted as flat 2D with bold block lines
  (f) Source-domain preservation — bar chart of averaged commonsense score

Usage (RL):
    CUDA_VISIBLE_DEVICES=0 python analysis/plot_motivation_figure.py \\
        --base-model Qwen/Qwen3-1.7B \\
        --full-ft-ckpt /data/yequan/fura/rl_runs/full/full-adamw-lr_2e-5-0325-215533/step=50 \\
        --fura-ckpt /data/yequan/fura/rl_runs/blocktt/blocktt-adamw-lr_8e-5-...-0423/step=50 \\
        --svd-run /data/yequan/fura/rl_runs/svd/svd-adamw-lr_1e-5-...-0421 \\
        --output docs/exp_results/figs/motivation.pdf

Usage (Math SFT):
    CUDA_VISIBLE_DEVICES=0 python analysis/plot_motivation_figure.py \\
        --base-model meta-llama/Meta-Llama-3-8B \\
        --full-ft-ckpt /data/yequan/fura/lift/math/.../full-lr_5e-5-seed_43 \\
        --fura-ckpt /data/yequan/fura/lift/math/.../blocktt-...-seed_43 \\
        --svd-ckpt /data/yequan/fura/lift/math/.../svd-...-seed_43 \\
        --output docs/26_nips_fura_paper/figs/motivation_math.pdf
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import numpy as np
import torch
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from btt_layer import BTTLayer, normalize_blocktt_decomp_mode

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
ATTN_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")

COLOR_FULL = "#2171B5"
COLOR_SVD = "#6A51A3"
COLOR_FURA = "#D94801"
COLOR_LORA = "#969696"
COLOR_Q = "#2171B5"    # blue for q_proj
COLOR_UP = "#D94801"   # orange for up_proj

# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def _module_prefix(layer_idx: int, module_name: str) -> str:
    if module_name in ATTN_MODULES:
        return f"model.layers.{layer_idx}.self_attn.{module_name}"
    return f"model.layers.{layer_idx}.mlp.{module_name}"


def load_safetensors_index(directory: str) -> dict[str, str]:
    directory = Path(directory)
    index = {}
    for f in sorted(directory.glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as sf:
            for key in sf.keys():
                index[key] = str(f)
    return index


def load_tensor_sf(index: dict[str, str], key: str) -> torch.Tensor | None:
    if key not in index:
        return None
    with safe_open(index[key], framework="pt") as sf:
        return sf.get_tensor(key)


def detect_num_layers(index: dict[str, str]) -> int:
    layers = set()
    for key in index:
        parts = key.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layers.add(int(parts[i + 1]))
    return max(layers) + 1 if layers else 0


def resolve_base_model(path_or_id: str) -> str:
    p = Path(path_or_id)
    if p.exists() and any(p.glob("*.safetensors")):
        return str(p)
    from huggingface_hub import snapshot_download
    return snapshot_download(path_or_id)


def load_checkpoint(ckpt_path: str) -> tuple[dict, str]:
    """Return (weights, fmt) where fmt is 'safetensors' or 'state_dict'."""
    p = Path(ckpt_path)
    if any(p.glob("*.safetensors")):
        return load_safetensors_index(ckpt_path), "safetensors"
    bin_path = p / "pytorch_model.bin"
    if bin_path.exists():
        return torch.load(str(bin_path), map_location="cpu", weights_only=True), "state_dict"
    raise FileNotFoundError(f"No checkpoint found in {ckpt_path}")


def get_weight(weights, key: str, fmt: str) -> torch.Tensor | None:
    if fmt == "safetensors":
        return load_tensor_sf(weights, key)
    return weights.get(key)

# ---------------------------------------------------------------------------
# Analysis: effective rank
# ---------------------------------------------------------------------------

def compute_effective_rank(delta_W: torch.Tensor, threshold: float = 0.90) -> int:
    s = torch.linalg.svdvals(delta_W.float())
    cumulative = torch.cumsum(s ** 2, dim=0)
    total = cumulative[-1]
    if total == 0:
        return 0
    k = int((cumulative / total < threshold).sum().item()) + 1
    return min(k, len(s))


def compute_rank_curves(
    base_index: dict, ckpt_weights, ckpt_fmt: str,
    num_layers: int, modules: list[str], device: torch.device,
) -> dict[str, np.ndarray]:
    """Return {module_name: array of effective_rank per layer}."""
    result = {mod: np.zeros(num_layers) for mod in modules}
    for layer_idx in range(num_layers):
        for mod in modules:
            key = f"{_module_prefix(layer_idx, mod)}.weight"
            W0 = load_tensor_sf(base_index, key)
            W1 = get_weight(ckpt_weights, key, ckpt_fmt)
            if W0 is None or W1 is None:
                continue
            delta = W1.float().to(device) - W0.float().to(device)
            result[mod][layer_idx] = compute_effective_rank(delta)
            del W0, W1, delta
        if layer_idx % 8 == 0:
            print(f"  rank: layer {layer_idx}/{num_layers - 1}")
    return result


# ---------------------------------------------------------------------------
# Analysis: right singular vector update heatmap (panel b)
# ---------------------------------------------------------------------------

def compute_V_update_heatmap(
    base_index: dict, ckpt_weights, ckpt_fmt: str,
    num_layers: int, module: str, device: torch.device,
) -> np.ndarray:
    """Compute |V' - V0| for every layer, return (num_layers, min_dim, min_dim)."""
    all_updates = []
    for layer_idx in range(num_layers):
        key = f"{_module_prefix(layer_idx, module)}.weight"
        W0 = load_tensor_sf(base_index, key)
        W1 = get_weight(ckpt_weights, key, ckpt_fmt)
        if W0 is None or W1 is None:
            all_updates.append(None)
            continue
        W0f = W0.float().to(device)
        W1f = W1.float().to(device)
        _, _, Vt0 = torch.linalg.svd(W0f, full_matrices=False)
        _, _, Vt1 = torch.linalg.svd(W1f, full_matrices=False)
        # Align signs: for each i, flip Vt1[i] if dot(Vt0[i], Vt1[i]) < 0
        signs = torch.sign((Vt0 * Vt1).sum(dim=-1))
        signs[signs == 0] = 1.0
        Vt1_aligned = Vt1 * signs.unsqueeze(-1)
        delta_Vt = (Vt1_aligned - Vt0).abs()  # (rank, d_in)
        all_updates.append(delta_Vt.cpu().numpy())
        del W0, W1, W0f, W1f, Vt0, Vt1, Vt1_aligned, delta_Vt
        if layer_idx % 8 == 0:
            print(f"  V-heatmap: layer {layer_idx}/{num_layers - 1}")
    return all_updates


# ---------------------------------------------------------------------------
# Analysis: core-R update heatmap (panel e)
# ---------------------------------------------------------------------------

def decompose_to_btt_R(
    W: torch.Tensor, decomp_mode: str = "output_one_block",
) -> tuple[torch.Tensor, int, int, int, int]:
    """Decompose weight W into BlockTT and return (R, n, b, m, rank).

    R has shape (n, b, m*rank).
    """
    out_features, in_features = W.shape
    layer = BTTLayer(in_features, out_features, rank="full",
                     decomp_mode=decomp_mode, bias=False)
    layer.init_from_linear_weight(W, bias=None, s_merged_to="trainable")
    return layer.btt_r.data.clone(), layer.n, layer.b, layer.m, layer.rank


def compute_R_update_heatmap(
    base_index: dict, ckpt_weights, ckpt_fmt: str,
    num_layers: int, module: str, device: torch.device,
    decomp_mode: str = "output_one_block",
) -> list[dict | None]:
    """Compute |R' - R0| for every layer. Returns list of dicts with 'delta', 'n', 'b', 'm', 'rank'."""
    results = []
    for layer_idx in range(num_layers):
        key = f"{_module_prefix(layer_idx, module)}.weight"
        W0 = load_tensor_sf(base_index, key)
        W1 = get_weight(ckpt_weights, key, ckpt_fmt)
        if W0 is None or W1 is None:
            results.append(None)
            continue
        W0f = W0.float().to(device)
        W1f = W1.float().to(device)
        R0, n, b, m, rank = decompose_to_btt_R(W0f, decomp_mode)
        R1, _, _, _, _ = decompose_to_btt_R(W1f, decomp_mode)
        delta_R = (R1 - R0).abs().cpu().numpy()  # (n, b, m*rank)
        results.append({"delta": delta_R, "n": n, "b": b, "m": m, "rank": rank})
        del W0, W1, W0f, W1f, R0, R1
        if layer_idx % 8 == 0:
            print(f"  R-heatmap: layer {layer_idx}/{num_layers - 1}")
    return results


# ---------------------------------------------------------------------------
# Learning curve extraction
# ---------------------------------------------------------------------------

def extract_rl_curve(run_dir: str) -> tuple[list[int], list[float]]:
    """Extract (steps, eval_accuracy%) from wandb output.log."""
    log_files = list(Path(run_dir).rglob("output.log"))
    if not log_files:
        raise FileNotFoundError(f"No output.log under {run_dir}")
    log_file = log_files[0]
    for lf in log_files:
        if "latest-run" in str(lf):
            log_file = lf
            break
    steps, accs = [], []
    pat = re.compile(r"step=(\d+),\s*correct:\s*(\d+)\s*/\s*(\d+)")
    with open(log_file) as f:
        for line in f:
            m = pat.search(line)
            if m:
                steps.append(int(m.group(1)))
                accs.append(int(m.group(2)) / int(m.group(3)) * 100)
    return steps, accs


def extract_sft_curve(run_dir: str) -> tuple[list[int], list[float]]:
    """Extract (steps, loss) from training.log."""
    log_path = Path(run_dir) / "training.log"
    if not log_path.exists():
        raise FileNotFoundError(f"No training.log in {run_dir}")
    steps, losses = [], []
    pat = re.compile(r"Step:\s*(\d+),.*Loss:\s*([\d.]+)")
    with open(log_path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
    return steps, losses


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
})


def _panel_label(ax, label: str):
    """Add bold panel label in upper-left corner."""
    ax.text(-0.12, 1.08, f"({label})", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top")


def plot_rank_curves(ax, rank_data: dict[str, np.ndarray], title: str,
                     ylim_max: float = None):
    """Panel (a)/(d): effective rank vs layer index, two module curves."""
    layers = np.arange(len(next(iter(rank_data.values()))))
    colors = {"q_proj": COLOR_Q, "up_proj": COLOR_UP}
    labels = {"q_proj": "Attn Q", "up_proj": "MLP Up"}

    for mod, vals in rank_data.items():
        ax.plot(layers, vals, label=labels.get(mod, mod), linewidth=1.5,
                color=colors.get(mod, "gray"), marker=".", markersize=3)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Effective rank of ΔW")
    ax.set_title(title, fontsize=9)
    ax.legend(loc="best", framealpha=0.9, edgecolor="gray")
    if ylim_max:
        ax.set_ylim(0, ylim_max)
    ax.grid(True, alpha=0.2, linewidth=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_V_heatmap(ax, v_updates: list, layer_idx: int, title: str):
    """Panel (b): heatmap of |V' - V0| for one layer."""
    data = v_updates[layer_idx]  # (rank, d_in)
    if data is None:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=9)
        return

    # Use log scale for visibility
    eps = 1e-10
    data_log = np.log10(data + eps)

    im = ax.imshow(data_log, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_xlabel("Column index (input dim)")
    ax.set_ylabel("Singular vector index (by σ₀)")
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log₁₀|V'−V₀|")


def plot_R_heatmap(ax, r_updates: list, layer_idx: int, title: str):
    """Panel (e): heatmap of |R' - R0| as (n*b) x (m*rank) rectangle.

    R has shape (n, b, m*rank). Each of the n slices is b × (m*rank).
    We stack vertically into (n*b, m*rank) and draw bold lines every b rows.
    aspect="equal" so each matrix element is square.
    """
    info = r_updates[layer_idx]
    if info is None:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=9)
        return

    delta = info["delta"]  # (n, b, m*rank)
    n, b, mr = delta.shape

    # Flatten to (n*b, m*rank)
    flat = delta.reshape(n * b, mr)

    eps = 1e-10
    flat_log = np.log10(flat + eps)

    im = ax.imshow(flat_log, aspect="equal", cmap="magma", interpolation="nearest")

    # Bold horizontal lines at block boundaries to separate n slices
    for i in range(1, n):
        ax.axhline(y=i * b - 0.5, color="white", linewidth=1.8, alpha=0.9)

    ax.set_xlabel("rank")
    ax.set_ylabel(f"block index (n={n})")
    ax.set_title(title, fontsize=9)

    # y-tick labels at block centers (show every few blocks)
    step = max(1, n // 8)
    ytick_positions = [i * b + b // 2 for i in range(0, n, step)]
    ytick_labels = [str(i) for i in range(0, n, step)]
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log₁₀|R'−R₀|")


def plot_learning_curves(ax, curves: dict[str, tuple], title: str,
                         mode: str = "accuracy"):
    """Panel (c): learning curves for Full FT and SVD FT."""
    style = {
        "Full FT": {"color": COLOR_FULL, "ls": "-", "marker": "o", "zorder": 3},
        "SVD FT": {"color": COLOR_SVD, "ls": "--", "marker": "s", "zorder": 4},
        "FuRA": {"color": COLOR_FURA, "ls": "-", "marker": "^", "zorder": 5},
    }
    for label, (steps, vals) in curves.items():
        kw = style.get(label, {"color": "gray", "ls": "-", "marker": "."})
        if mode == "loss":
            ax.plot(steps, vals, label=label, linewidth=1.3,
                    color=kw["color"], ls=kw["ls"], zorder=kw.get("zorder", 2))
        else:
            ax.plot(steps, vals, label=label, markersize=4, linewidth=1.5,
                    markeredgewidth=0.5, markeredgecolor="white", **kw)

    if mode == "loss":
        ax.set_ylabel("Training loss")
        ax.set_xlabel("Step")
        ax.legend(loc="upper right", framealpha=0.9, edgecolor="gray")
    else:
        ax.set_ylabel("Eval accuracy (%)")
        ax.set_xlabel("GRPO step")
        ax.legend(loc="lower right", framealpha=0.9, edgecolor="gray")
        ax.set_ylim(10, 100)

    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.2, linewidth=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_forgetting_bar(ax, data: dict[str, float], title: str):
    """Panel (f): bar chart of averaged commonsense score per method."""
    methods = list(data.keys())
    values = list(data.values())
    colors_map = {"Base": "#BDBDBD", "Full FT": COLOR_FULL, "FuRA": COLOR_FURA,
                  "LoRA": COLOR_LORA, "SVD FT": COLOR_SVD}
    colors = [colors_map.get(m, "#888888") for m in methods]

    bars = ax.bar(methods, values, color=colors, edgecolor="white", linewidth=0.5,
                  width=0.55)
    # Annotate values
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{v:.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_ylabel("Avg commonsense acc (%)")
    ax.set_title(title, fontsize=9)
    ax.set_ylim(0, max(values) * 1.15)
    ax.grid(True, axis="y", alpha=0.2, linewidth=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------

def build_figure(
    full_rank: dict, fura_rank: dict,
    v_updates: list, r_updates: list,
    heatmap_layer: int,
    curves: dict, curve_mode: str,
    forgetting: dict,
    output_path: str,
    curve_title: str = "Learning curves",
):
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           width_ratios=[1.2, 1.2, 1],
                           height_ratios=[1, 1.5],
                           hspace=0.40, wspace=0.40)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])

    # Shared y-axis for rank panels
    all_ranks = np.concatenate([v for v in full_rank.values()] +
                                [v for v in fura_rank.values()])
    rank_ylim = max(all_ranks.max() * 1.15, 100)

    # (a) Full FT rank curves
    plot_rank_curves(ax_a, full_rank, "Full FT: effective rank of ΔW", ylim_max=rank_ylim)
    _panel_label(ax_a, "a")

    # (b) Full FT V-heatmap
    plot_V_heatmap(ax_b, v_updates, heatmap_layer,
                   f"Full FT: |V'−V₀| (layer {heatmap_layer}, q_proj)")
    _panel_label(ax_b, "b")

    # (c) Learning curves
    plot_learning_curves(ax_c, curves, curve_title, mode=curve_mode)
    _panel_label(ax_c, "c")

    # (d) FuRA rank curves
    plot_rank_curves(ax_d, fura_rank, "FuRA: effective rank of ΔW", ylim_max=rank_ylim)
    _panel_label(ax_d, "d")

    # (e) FuRA R-heatmap
    plot_R_heatmap(ax_e, r_updates, heatmap_layer,
                   f"FuRA: |R'−R₀| (layer {heatmap_layer}, q_proj)")
    _panel_label(ax_e, "e")

    # (f) Source-domain preservation
    plot_forgetting_bar(ax_f, forgetting, "Source-domain preservation")
    _panel_label(ax_f, "f")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out))
    fig.savefig(str(out.with_suffix(".png")), dpi=300)
    print(f"Saved: {out}")
    print(f"Saved: {out.with_suffix('.png')}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate Figure 2 (motivating examples)")
    p.add_argument("--base-model", required=True)
    p.add_argument("--full-ft-ckpt", required=True)
    p.add_argument("--fura-ckpt", required=True)
    p.add_argument("--svd-ckpt", default=None, help="SVD FT ckpt dir (for learning curve)")
    p.add_argument("--svd-run", default=None, help="SVD FT run dir (for RL learning curve)")
    p.add_argument("--full-ft-run", default=None, help="Override run dir for Full FT curve")
    p.add_argument("--fura-run", default=None, help="Override run dir for FuRA curve")
    p.add_argument("--heatmap-layer", type=int, default=None,
                   help="Layer index for heatmap panels (b) and (e). Default: middle layer.")
    p.add_argument("--heatmap-module", default="q_proj",
                   help="Module for heatmap panels (default: q_proj)")
    p.add_argument("--decomp-mode", default="output_one_block",
                   help="BlockTT decomposition mode for panel (e)")
    p.add_argument("--curve-mode", default="auto", choices=["auto", "accuracy", "loss"])
    p.add_argument("--forgetting-json", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", default="docs/exp_results/figs/motivation.pdf")
    p.add_argument("--cache-dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Device: {device}")

    # Resolve base model
    base_path = resolve_base_model(args.base_model)
    base_index = load_safetensors_index(base_path)
    num_layers = detect_num_layers(base_index)
    print(f"Base model: {args.base_model} ({num_layers} layers)")

    if args.heatmap_layer is None:
        args.heatmap_layer = num_layers // 2
    print(f"Heatmap layer: {args.heatmap_layer}")

    # Load checkpoints
    full_weights, full_fmt = load_checkpoint(args.full_ft_ckpt)
    fura_weights, fura_fmt = load_checkpoint(args.fura_ckpt)

    # --- Panel (a): Full FT rank curves ---
    print("Computing Full FT rank curves...")
    full_rank = compute_rank_curves(
        base_index, full_weights, full_fmt,
        num_layers, ["q_proj", "up_proj"], device,
    )

    # --- Panel (d): FuRA rank curves ---
    print("Computing FuRA rank curves...")
    fura_rank = compute_rank_curves(
        base_index, fura_weights, fura_fmt,
        num_layers, ["q_proj", "up_proj"], device,
    )

    # --- Panel (b): Full FT V-update heatmap ---
    print("Computing V-update heatmap...")
    v_updates = compute_V_update_heatmap(
        base_index, full_weights, full_fmt,
        num_layers, args.heatmap_module, device,
    )

    # --- Panel (e): FuRA R-update heatmap ---
    print("Computing R-update heatmap...")
    r_updates = compute_R_update_heatmap(
        base_index, fura_weights, fura_fmt,
        num_layers, args.heatmap_module, device,
        decomp_mode=args.decomp_mode,
    )

    # --- Panel (c): Learning curves ---
    curves = {}
    curve_mode = args.curve_mode

    # Determine run dirs
    full_run = args.full_ft_run or args.full_ft_ckpt
    fura_run = args.fura_run or args.fura_ckpt
    svd_run = args.svd_run or args.svd_ckpt

    # Auto-detect mode
    if curve_mode == "auto":
        # Try RL from parent dir first
        for d in [full_run, str(Path(full_run).parent)]:
            try:
                extract_rl_curve(d)
                curve_mode = "accuracy"
                if d != full_run:
                    full_run = d
                    fura_run = args.fura_run or str(Path(args.fura_ckpt).parent)
                    svd_run = args.svd_run or (str(Path(args.svd_ckpt).parent) if args.svd_ckpt else None)
                break
            except FileNotFoundError:
                pass
        else:
            try:
                extract_sft_curve(full_run)
                curve_mode = "loss"
            except FileNotFoundError:
                curve_mode = "accuracy"

    print(f"Curve mode: {curve_mode}")
    extractor = extract_rl_curve if curve_mode == "accuracy" else extract_sft_curve

    for label, run_dir in [("Full FT", full_run), ("SVD FT", svd_run)]:
        if run_dir is None:
            continue
        try:
            steps, vals = extractor(run_dir)
            curves[label] = (steps, vals)
            last = f"{vals[-1]:.1f}%" if curve_mode == "accuracy" else f"{vals[-1]:.4f}"
            print(f"{label}: {len(steps)} pts, final={last}")
        except FileNotFoundError as e:
            print(f"Warning ({label}): {e}")

    curve_title = "Math SFT training loss" if curve_mode == "loss" else "Math RL eval accuracy"

    # --- Panel (f): Forgetting ---
    if args.forgetting_json:
        with open(args.forgetting_json) as f:
            forgetting = json.load(f)
    else:
        # Default: from docs/exp_results/forgetting.md (Llama-3-8B math SFT)
        forgetting = {
            "Base": 37.6,
            "FuRA": 45.5,
            "LoRA": 35.6,
        }

    # --- Build figure ---
    print("Building figure...")
    build_figure(
        full_rank=full_rank, fura_rank=fura_rank,
        v_updates=v_updates, r_updates=r_updates,
        heatmap_layer=args.heatmap_layer,
        curves=curves, curve_mode=curve_mode,
        forgetting=forgetting,
        output_path=args.output,
        curve_title=curve_title,
    )


if __name__ == "__main__":
    main()
