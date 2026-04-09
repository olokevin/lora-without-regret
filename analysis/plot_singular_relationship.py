"""Plot singular-value relationships for SVD and BlockTT runs.

For a selected layer prefix, this script produces heatmaps for:
1) first-step gradient magnitude, and
2) total update magnitude from initialization to final checkpoint.

If first-step gradients are not cached, it can replay one RL step via run_rl.py
and cache gradients for later reuse.

--full-run /data/yequan/fura/runs/full/full-adamw-lr_2e-5-0325-215533 \
--svd-run /data/yequan/fura/runs/svd/svd-adamw-lr_1e-5-s_to_keep-train_input-0317-141139 \
--blocktt-run /data/yequan/fura/runs/blocktt/blocktt-adamw-lr_1e-5-output_one_block-s_to_keep-train_both-0317-155422 \
--blocktt-run /data/yequan/fura/runs/blocktt/blocktt-adamw-lr_1e-4-output_one_block-s_to_frozen-train_small-0317-150342 \

CUDA_VISIBLE_DEVICES=0 python analysis/plot_singular_relationship.py \
    --full-run runs/full/full-adamw-lr_2e-5-0325-215533 \
    --layer-prefix model.layers.12.self_attn.q_proj \
    --color-scale log

### --sft-run applies to SFT RL KD runs with the same grad and ckpt saving

CUDA_VISIBLE_DEVICES=2 python analysis/plot_singular_relationship.py \
    --sft-run /data/yequan/fura/sft_runs/blocktt/blocktt-adamw-lr_2e-4-output_one_block-s_to_frozen-train_small-0331-164017 \
    --base-model-id Qwen/Qwen3-4B \
    --layer-prefix model.layers.12.self_attn.q_proj \
    --color-scale log

CUDA_VISIBLE_DEVICES=2 python analysis/plot_singular_relationship.py \
    --sft-run /data/yequan/fura/kd_runs/blocktt/blocktt-sft-adamw-lr_1e-4-output_one_block-s_to_input-train_both-0401-212120 \
    --base-model-id Qwen/Qwen2.5-0.5B \
    --layer-prefix model.layers.12 \
    --color-scale log
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

# Support running as `python analysis/plot_singular_relationship.py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from btt_layer import BTTLayer
from svd_layer import SVDLayer


LAYER_PREFIX_RE = re.compile(r"^model\.layers\.(\d+)\.(self_attn|mlp)\.([A-Za-z0-9_]+)$")
FACTOR_SUFFIXES = (".btt_l", ".btt_r", ".btt_s", ".svd_a", ".svd_b", ".svd_s")

# ---------------------------------------------------------------------------
# Directory-name config parsing (for runs without wandb)
# ---------------------------------------------------------------------------

_DIR_S_TO_RE = re.compile(r"-s_to_(\w+)-")
_DIR_TRAIN_RE = re.compile(r"-train_(\w+)-")
_DIR_LR_RE = re.compile(r"-lr_([\d.eE+-]+)-")


def parse_run_dir_config(run_dir: Path) -> dict[str, Any]:
    """Extract training config from the run directory name."""
    name = run_dir.name

    # train_mode is the first token
    train_mode = name.split("-", 1)[0]

    s_merged_to = None
    m = _DIR_S_TO_RE.search(name)
    if m:
        s_merged_to = m.group(1)

    train_position = None
    m = _DIR_TRAIN_RE.search(name)
    if m:
        train_position = m.group(1)

    lr = None
    m = _DIR_LR_RE.search(name)
    if m:
        lr = float(m.group(1))

    # decomp_mode: between lr_...- and -s_to_ (or end)
    decomp_mode: str | dict[str, str] | None = None
    lr_match = _DIR_LR_RE.search(name)
    s_to_match = _DIR_S_TO_RE.search(name)
    if lr_match and s_to_match:
        segment = name[lr_match.end() : s_to_match.start()]
        if segment:
            decomp_mode = segment
            if "{" in segment:
                # Parse dict-like: {qkv:input,o:output,...}
                inner = segment.strip("{}")
                decomp_mode = {}
                for part in inner.split(","):
                    k, v = part.split(":", 1)
                    decomp_mode[k.strip()] = v.strip()

    # Build decomp_mode_by_module if we have a dict
    decomp_mode_by_module = None
    if isinstance(decomp_mode, dict):
        from btt_layer import BLOCKTT_DECOMP_GROUP_TO_MODULES
        decomp_mode_by_module = {}
        for group_name, module_names in BLOCKTT_DECOMP_GROUP_TO_MODULES.items():
            mode_val = decomp_mode.get(group_name, "input_one_block")
            # Normalize short aliases
            mode_map = {"input": "input_one_block", "output": "output_one_block"}
            mode_val = mode_map.get(mode_val, mode_val)
            for mod_name in module_names:
                decomp_mode_by_module[mod_name] = mode_val

    return {
        "train_mode": train_mode,
        "s_merged_to": s_merged_to,
        "train_position": train_position,
        "learning_rate": lr,
        "decomp_mode": decomp_mode if not isinstance(decomp_mode, dict) else "mixed",
        "decomp_mode_by_module": decomp_mode_by_module,
        "blocktt_rank": "full",
    }


# ---------------------------------------------------------------------------
# Flexible layer prefix resolution
# ---------------------------------------------------------------------------

def _extract_layer_prefixes_from_keys(keys: list[str]) -> list[str]:
    """Strip factor suffixes from safetensors keys to get unique layer prefixes."""
    prefixes = set()
    for key in keys:
        for suffix in FACTOR_SUFFIXES:
            if key.endswith(suffix):
                prefixes.add(key[: -len(suffix)])
                break
    return sorted(prefixes)


def resolve_layer_prefixes(prefix: str, run_dir: Path) -> list[str]:
    """Expand a flexible layer prefix into a list of fully-qualified prefixes.

    Supported forms:
      - model.layers.12.self_attn.q_proj  -> exact match
      - model.layers.12                   -> all projection modules in layer 12
      - q_proj                            -> that module across all layers
    """
    # Find a safetensors file to scan keys from
    key_source = _find_key_source(run_dir)
    if key_source is None:
        raise FileNotFoundError(f"No safetensors files found in {run_dir} to resolve layer prefixes")

    with safe_open(str(key_source), framework="pt") as sf:
        all_keys = list(sf.keys())
    all_prefixes = _extract_layer_prefixes_from_keys(all_keys)

    if not all_prefixes:
        raise ValueError(f"No factor keys found in {key_source}")

    # Exact match
    if prefix in all_prefixes:
        return [prefix]

    # Layer-level match: model.layers.12
    if re.match(r"^model\.layers\.\d+$", prefix):
        matched = [p for p in all_prefixes if p.startswith(prefix + ".")]
        if not matched:
            raise ValueError(f"No modules found under '{prefix}'. Available: {all_prefixes[:5]}")
        return matched

    # Module-name match: q_proj, down_proj, etc.
    if "." not in prefix:
        matched = [p for p in all_prefixes if p.endswith("." + prefix)]
        if not matched:
            raise ValueError(f"No layers found with module '{prefix}'. Available: {all_prefixes[:5]}")
        return matched

    raise ValueError(
        f"Layer prefix '{prefix}' not found. "
        f"Expected one of: exact prefix, model.layers.N, or module_name. "
        f"Available prefixes (first 5): {all_prefixes[:5]}"
    )


def _find_key_source(run_dir: Path) -> Path | None:
    """Find a safetensors file in the run dir to scan for available keys."""
    # Prefer grads at step 0 (always has all trainable param keys)
    for pattern in ["step=0/grads.safetensors", "optim_step=0/grads.safetensors"]:
        p = run_dir / pattern
        if p.exists():
            return p
    # Fall back to any grads file
    for sub in sorted(run_dir.iterdir()):
        if sub.is_dir():
            g = sub / "grads.safetensors"
            if g.exists():
                return g
    # Fall back to model checkpoint
    for sub in sorted(run_dir.iterdir()):
        if sub.is_dir():
            m = sub / "model.safetensors"
            if m.exists():
                return m
    return None


# ---------------------------------------------------------------------------
# Sharded checkpoint loading
# ---------------------------------------------------------------------------

def load_tensor_from_checkpoint(step_dir: Path, key: str) -> torch.Tensor | None:
    """Load a tensor from a (possibly sharded) model checkpoint directory."""
    single = step_dir / "model.safetensors"
    if single.exists():
        return load_tensor_from_safetensor(single, key)

    index_path = step_dir / "model.safetensors.index.json"
    if not index_path.exists():
        return None

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    weight_map = index.get("weight_map", {})
    shard_name = weight_map.get(key)
    if shard_name is None:
        return None

    shard_path = step_dir / shard_name
    return load_tensor_from_safetensor(shard_path, key)


# ---------------------------------------------------------------------------
# Grad file discovery
# ---------------------------------------------------------------------------

def find_grad_paths(run_dir: Path) -> dict[int, Path]:
    """Find all pre-saved grad files. Returns {step: path_to_grads.safetensors}."""
    result: dict[int, Path] = {}
    for sub in run_dir.iterdir():
        if not sub.is_dir():
            continue
        grads_file = sub / "grads.safetensors"
        if not grads_file.exists():
            continue
        # Parse step from dir name: step=N or optim_step=N
        name = sub.name
        if "=" not in name:
            continue
        try:
            step = int(name.split("=", 1)[1])
        except ValueError:
            continue
        result[step] = grads_file
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Plot singular-value vs gradient/update heatmaps")
    p.add_argument(
        "--layer-prefix", required=True,
        help="Layer prefix: model.layers.12.self_attn.q_proj (exact), "
             "model.layers.12 (all modules in layer), or q_proj (module across all layers)",
    )
    p.add_argument("--svd-run", type=Path, default=None, help="Path to one SVD run directory")
    p.add_argument("--blocktt-run", type=Path, default=None, help="Path to one BlockTT run directory")
    p.add_argument("--full-run", type=Path, default=None, help="Path to one full-model run directory")
    p.add_argument("--sft-run", type=Path, default=None, help="Path to an SFT run directory (with pre-saved grads/ckpts)")
    p.add_argument("--base-model-id", default=None, help="Override base model ID")
    p.add_argument("--output-dir", type=Path, default=Path("analysis_results/singular_relationship"))
    p.add_argument("--color-scale", choices=["linear", "log"], default="linear")
    p.add_argument("--force-replay-grad", action="store_true", help="Force replay even if grad cache exists")
    p.add_argument("--no-replay-grad", action="store_true", help="Do not replay; require existing grad cache")
    p.add_argument("--device", default="cpu", help="Device for base model init extraction")
    p.add_argument("--final-step", type=int, default=None, help="Optional explicit final checkpoint step")
    p.add_argument(
        "--to-plot",
        nargs="+",
        choices=["grad_0", "grad_n", "update"],
        default=None,
        help="What to plot: grad_0 (grad at step 0), grad_n (grad at step n), update (final - pretrained). Default: all.",
    )
    p.add_argument("--grad-n-step", type=int, default=None, help="Which step for grad_n (default: max step with grads)")
    p.add_argument(
        "--full-step-start",
        type=int,
        default=1,
        help="Start checkpoint step for full-model SVD delta (default: 0).",
    )
    p.add_argument(
        "--blocktt-subplot-shape",
        default=None,
        help=(
            "Optional ROWSxCOLS layout for BlockTT slice figures. "
            "Must satisfy ROWS*COLS == m*n for the selected layer."
        ),
    )
    return p.parse_args(argv)


def parse_layer_prefix(prefix: str) -> tuple[int, str]:
    """Validate a fully-qualified layer prefix. Returns (layer_idx, module_name)."""
    m = LAYER_PREFIX_RE.match(prefix)
    if not m:
        raise ValueError(
            "Invalid --layer-prefix format. Expected e.g. model.layers.12.self_attn.q_proj"
        )
    return int(m.group(1)), m.group(3)


def is_exact_layer_prefix(prefix: str) -> bool:
    """Check if a prefix is fully-qualified (e.g. model.layers.12.self_attn.q_proj)."""
    return LAYER_PREFIX_RE.match(prefix) is not None


def parse_subplot_shape(shape: str | None) -> tuple[int, int] | None:
    if shape is None:
        return None
    m = re.match(r"^\s*(\d+)\s*[xX,]\s*(\d+)\s*$", shape)
    if not m:
        raise ValueError("--blocktt-subplot-shape must be in ROWSxCOLS format, e.g. 2x8")
    rows = int(m.group(1))
    cols = int(m.group(2))
    if rows <= 0 or cols <= 0:
        raise ValueError("--blocktt-subplot-shape rows and cols must be > 0")
    return rows, cols


def factor_keys_for_mode(mode: str, layer_prefix: str) -> dict[str, str]:
    if mode == "svd":
        return {
            "left": f"{layer_prefix}.svd_a",
            "right": f"{layer_prefix}.svd_b",
            "singular": f"{layer_prefix}.svd_s",
        }
    if mode == "blocktt":
        return {
            "left": f"{layer_prefix}.btt_l",
            "right": f"{layer_prefix}.btt_r",
            "singular": f"{layer_prefix}.btt_s",
        }
    raise ValueError(f"Unsupported mode: {mode}")


def resolve_trained_sides(
    mode: str,
    run_cfg: dict[str, Any],
    left_tensor: torch.Tensor,
    right_tensor: torch.Tensor,
) -> list[str]:
    if mode == "svd":
        train_position = str(run_cfg.get("train_position", "output"))
        if train_position == "output":
            return ["left"]
        if train_position == "input":
            return ["right"]
        return ["left", "right"]

    if mode == "blocktt":
        train_position = str(run_cfg.get("train_position", "small"))
        if train_position == "both":
            return ["left", "right"]
        left_size = left_tensor.numel()
        right_size = right_tensor.numel()
        if train_position == "small":
            train_left = left_size <= right_size
            return ["left"] if train_left else ["right"]
        if train_position == "large":
            train_left = left_size >= right_size
            return ["left"] if train_left else ["right"]
        return ["left", "right"]

    if mode == "full":
        return ["left", "right"]

    raise ValueError(f"Unsupported mode: {mode}")


def find_max_step(run_dir: Path) -> int:
    max_step = None
    for p in run_dir.iterdir():
        if not p.is_dir() or not p.name.startswith("step="):
            continue
        try:
            step = int(p.name.split("=", 1)[1])
        except ValueError:
            continue
        max_step = step if max_step is None else max(max_step, step)
    if max_step is None:
        raise ValueError(f"No step=* checkpoints found under {run_dir}")
    return max_step


def resolve_step_model_path(run_dir: Path, step: int) -> Path:
    model_path = run_dir / f"step={step}" / "model.safetensors"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {model_path}")
    return model_path


def load_tensor_from_safetensor(path: Path, key: str) -> torch.Tensor | None:
    with safe_open(str(path), framework="pt") as sf:
        if key not in sf.keys():
            return None
        return sf.get_tensor(key)


def flatten_heatmap_tensor(mode: str, side: str, tensor: torch.Tensor) -> torch.Tensor:
    if mode in {"svd", "full"}:
        if tensor.ndim != 2:
            raise ValueError(f"{mode.upper()} tensor must be 2D, got shape {tuple(tensor.shape)}")
        return tensor

    if mode != "blocktt":
        raise ValueError(f"Unsupported mode: {mode}")

    if side == "left":
        if tensor.ndim != 3:
            raise ValueError(f"BlockTT left tensor must be 3D, got shape {tuple(tensor.shape)}")
        m, nr, a = tensor.shape
        return tensor.reshape(m * nr, a)

    if side == "right":
        if tensor.ndim != 3:
            raise ValueError(f"BlockTT right tensor must be 3D, got shape {tuple(tensor.shape)}")
        n, b, mr = tensor.shape
        return tensor.reshape(n * b, mr)

    raise ValueError(f"Unsupported side: {side}")


def build_singular_maps(
    mode: str,
    singular: torch.Tensor | None,
    left_tensor: torch.Tensor,
    right_tensor: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if singular is None:
        return None, None

    if mode == "svd":
        if singular.ndim != 1:
            raise ValueError(f"SVD singular tensor must be 1D, got {tuple(singular.shape)}")
        if left_tensor.shape[1] != singular.shape[0] or right_tensor.shape[0] != singular.shape[0]:
            raise ValueError("Singular values rank does not match SVD factor shapes")
        left_map = singular.unsqueeze(0).expand(left_tensor.shape[0], -1)
        right_map = singular.unsqueeze(1).expand(-1, right_tensor.shape[1])
        return flatten_heatmap_tensor("svd", "left", left_map), flatten_heatmap_tensor("svd", "right", right_map)

    if mode == "blocktt":
        if singular.ndim != 3:
            raise ValueError(f"BlockTT singular tensor must be 3D, got {tuple(singular.shape)}")

        m, n, r = singular.shape
        if left_tensor.shape[0] != m or left_tensor.shape[1] != n * r:
            raise ValueError("BlockTT left shape incompatible with btt_s")
        if right_tensor.shape[0] != n or right_tensor.shape[2] != m * r:
            raise ValueError("BlockTT right shape incompatible with btt_s")

        a = left_tensor.shape[2]
        b = right_tensor.shape[1]

        left_map_3d = singular.unsqueeze(-1).expand(m, n, r, a).reshape(m, n * r, a)
        right_map_3d = singular.permute(1, 0, 2).unsqueeze(1).expand(n, b, m, r).reshape(n, b, m * r)

        return (
            flatten_heatmap_tensor("blocktt", "left", left_map_3d),
            flatten_heatmap_tensor("blocktt", "right", right_map_3d),
        )

    raise ValueError(f"Unsupported mode: {mode}")


def pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x_flat = x.flatten().float()
    y_flat = y.flatten().float()
    if x_flat.numel() != y_flat.numel() or x_flat.numel() == 0:
        return float("nan")

    x_mean = x_flat.mean()
    y_mean = y_flat.mean()
    x_std = x_flat.std(unbiased=False)
    y_std = y_flat.std(unbiased=False)
    if x_std == 0 or y_std == 0:
        return float("nan")

    cov = ((x_flat - x_mean) * (y_flat - y_mean)).mean()
    return float(cov / (x_std * y_std))


def build_cache_id(run_cfg: dict[str, Any], layer_prefix: str) -> str:
    payload = {
        "layer_prefix": layer_prefix,
        "train_mode": run_cfg.get("train_mode"),
        "model_id": run_cfg.get("model_id"),
        "learning_rate": run_cfg.get("learning_rate"),
        "seed": run_cfg.get("seed"),
        "train_position": run_cfg.get("train_position"),
        "s_merged_to": run_cfg.get("s_merged_to"),
        "decomp_mode": run_cfg.get("decomp_mode"),
        "blocktt_rank": run_cfg.get("blocktt_rank"),
        "target_modules": run_cfg.get("target_modules"),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


def resolve_grad_cache_path(run_dir: Path, cache_id: str) -> Path:
    cache_dir = run_dir / "analysis_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"first_step_grads_{cache_id}.safetensors"


def resolve_grad_cache_meta_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(".json")


def _parse_run_wandb_config(config_path: Path) -> tuple[dict[str, Any], list[str]]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to parse W&B config.yaml") from exc

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    flattened: dict[str, Any] = {}
    for k, v in raw.items():
        if k == "_wandb":
            continue
        if isinstance(v, dict) and "value" in v:
            flattened[k] = v["value"]
        else:
            flattened[k] = v

    cli_args: list[str] = []
    wandb_meta = raw.get("_wandb", {}).get("value", {})
    runs = wandb_meta.get("e", {})
    if isinstance(runs, dict) and runs:
        first = next(iter(runs.values()))
        cli_args = list(first.get("args", []))

    return flattened, cli_args


def load_run_context(run_dir: Path) -> tuple[dict[str, Any], list[str]]:
    configs = sorted(run_dir.glob("wandb/run-*/files/config.yaml"))
    if not configs:
        raise FileNotFoundError(f"No W&B config found under {run_dir}/wandb/run-*/files/config.yaml")
    return _parse_run_wandb_config(configs[-1])


def _drop_flags(argv: list[str], flags_with_values: set[str], bool_flags: set[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in bool_flags:
            i += 1
            continue
        matched_with_value = False
        for flag in flags_with_values:
            if tok == flag:
                i += 2
                matched_with_value = True
                break
            if tok.startswith(flag + "="):
                i += 1
                matched_with_value = True
                break
        if matched_with_value:
            continue
        out.append(tok)
        i += 1
    return out


def _build_replay_command(
    run_cli_args: list[str],
    grad_cache_path: Path,
    layer_prefix: str,
    temp_base_dir: Path,
) -> list[str]:
    cleaned = _drop_flags(
        run_cli_args,
        flags_with_values={"--wandb-project", "--wandb-run-name", "--base-dir", "--n-grpo-steps"},
        bool_flags={"--enable-save-ckpt", "--no-wandb"},
    )

    cmd = [
        sys.executable,
        "run_rl.py",
        *cleaned,
        "--no-wandb",
        "--base-dir",
        str(temp_base_dir),
        "--n-grpo-steps",
        "1",
        "--save-first-step-grads-path",
        str(grad_cache_path),
        "--save-first-step-grads-prefixes",
        layer_prefix,
        "--stop-after-first-step",
    ]
    return cmd


def replay_and_cache_first_step_gradients(
    run_dir: Path,
    run_cli_args: list[str],
    grad_cache_path: Path,
    layer_prefix: str,
) -> None:
    temp_base_dir = run_dir / "analysis_cache" / "replay_runs"
    temp_base_dir.mkdir(parents=True, exist_ok=True)

    cmd = _build_replay_command(run_cli_args, grad_cache_path, layer_prefix, temp_base_dir)
    proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            "Failed to replay first training step for gradients. "
            f"Command exited with {proc.returncode}: {' '.join(cmd)}"
        )


def load_or_compute_first_step_gradients(
    run_dir: Path,
    run_cfg: dict[str, Any],
    run_cli_args: list[str],
    mode: str,
    layer_prefix: str,
    trained_sides: list[str],
    force_replay: bool,
    allow_replay: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None, Path]:
    keys = factor_keys_for_mode(mode, layer_prefix)
    cache_id = build_cache_id(run_cfg, layer_prefix)
    grad_cache_path = resolve_grad_cache_path(run_dir, cache_id)
    meta_path = resolve_grad_cache_meta_path(grad_cache_path)

    if force_replay or not grad_cache_path.exists():
        if not allow_replay:
            raise FileNotFoundError(
                f"Gradient cache not found and replay disabled: {grad_cache_path}"
            )
        replay_and_cache_first_step_gradients(run_dir, run_cli_args, grad_cache_path, layer_prefix)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "cache_id": cache_id,
                    "mode": mode,
                    "layer_prefix": layer_prefix,
                    "source": "run_rl_one_step_replay",
                },
                f,
                indent=2,
            )

    grads = load_file(str(grad_cache_path))
    left_key = keys["left"]
    right_key = keys["right"]
    left_grad = grads[left_key].abs().float() if left_key in grads else None
    right_grad = grads[right_key].abs().float() if right_key in grads else None

    missing = []
    if "left" in trained_sides and left_grad is None:
        missing.append(left_key)
    if "right" in trained_sides and right_grad is None:
        missing.append(right_key)
    if missing:
        available = ", ".join(sorted(grads.keys())[:20])
        raise KeyError(
            "Gradient cache missing required keys: "
            + ", ".join(missing)
            + f". Available (first 20): {available}"
        )

    return left_grad, right_grad, grad_cache_path


def _get_module_by_prefix(model: torch.nn.Module, layer_prefix: str) -> torch.nn.Module:
    module = model
    for part in layer_prefix.split("."):
        if not hasattr(module, part):
            raise AttributeError(f"Model has no submodule '{part}' in prefix '{layer_prefix}'")
        module = getattr(module, part)
    return module


def _resolve_blocktt_mode_for_module(run_cfg: dict[str, Any], module_name: str) -> str:
    mode_by_module = run_cfg.get("decomp_mode_by_module")
    if isinstance(mode_by_module, dict) and module_name in mode_by_module:
        return mode_by_module[module_name]
    decomp_mode = run_cfg.get("decomp_mode", "input_one_block")
    if isinstance(decomp_mode, dict):
        if module_name in decomp_mode:
            return decomp_mode[module_name]
        return "input_one_block"
    return str(decomp_mode)


def _resolve_blocktt_rank(run_cfg: dict[str, Any]) -> int | str:
    rank = run_cfg.get("blocktt_rank", "full")
    if isinstance(rank, str) and rank != "full":
        return int(rank)
    return rank


def build_initialized_factors(
    base_model: torch.nn.Module,
    mode: str,
    layer_prefix: str,
    run_cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    module = _get_module_by_prefix(base_model, layer_prefix)
    if not hasattr(module, "weight"):
        raise ValueError(f"Layer '{layer_prefix}' has no weight tensor")

    weight = module.weight.detach()
    bias = module.bias.detach() if getattr(module, "bias", None) is not None else None

    if mode == "svd":
        layer = SVDLayer(
            in_features=weight.shape[1],
            out_features=weight.shape[0],
            bias=bias is not None,
        ).to(dtype=weight.dtype, device=weight.device)
        layer.init_from_linear_weight(
            weight,
            bias=bias,
            s_merged_to=run_cfg.get("s_merged_to"),
            train_position=run_cfg.get("train_position", "output"),
        )
        singular = layer.svd_s.detach().clone() if layer.svd_s is not None else None
        return layer.svd_a.detach().clone(), layer.svd_b.detach().clone(), singular

    if mode == "blocktt":
        module_name = layer_prefix.split(".")[-1]
        layer = BTTLayer(
            in_features=weight.shape[1],
            out_features=weight.shape[0],
            rank=_resolve_blocktt_rank(run_cfg),
            bias=bias is not None,
            lr_act=False,
            decomp_mode=_resolve_blocktt_mode_for_module(run_cfg, module_name),
            init_mode="default",
            output_factorization=run_cfg.get("_output_factorization"),
            input_factorization=run_cfg.get("_input_factorization"),
        ).to(dtype=weight.dtype, device=weight.device)
        layer.init_from_linear_weight(
            weight,
            bias=bias,
            s_merged_to=run_cfg.get("s_merged_to"),
            train_position=run_cfg.get("train_position", "small"),
        )
        singular = layer.btt_s.detach().clone() if layer.btt_s is not None else None
        return layer.btt_l.detach().clone(), layer.btt_r.detach().clone(), singular

    raise ValueError(f"Unsupported mode: {mode}")


def load_final_factors(
    run_dir: Path,
    mode: str,
    layer_prefix: str,
    final_step: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int, Path]:
    step = final_step if final_step is not None else find_max_step(run_dir)
    model_path = resolve_step_model_path(run_dir, step)
    keys = factor_keys_for_mode(mode, layer_prefix)

    left = load_tensor_from_safetensor(model_path, keys["left"])
    right = load_tensor_from_safetensor(model_path, keys["right"])
    singular = load_tensor_from_safetensor(model_path, keys["singular"])

    if left is None or right is None:
        raise KeyError(
            f"Checkpoint {model_path} missing required factor keys for {layer_prefix} in mode {mode}"
        )

    return left.float(), right.float(), (singular.float() if singular is not None else None), step, model_path


def load_full_weight_checkpoint(
    run_dir: Path,
    layer_prefix: str,
    step: int,
) -> tuple[torch.Tensor, Path]:
    model_path = resolve_step_model_path(run_dir, step)
    weight = load_tensor_from_safetensor(model_path, f"{layer_prefix}.weight")
    if weight is None:
        raise KeyError(f"Checkpoint {model_path} missing dense weight key '{layer_prefix}.weight'")
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D dense weight at '{layer_prefix}.weight', got {tuple(weight.shape)}")
    return weight.float(), model_path


def svd_factors_from_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    u, _, vh = torch.linalg.svd(weight, full_matrices=False)
    return u.float(), vh.float()


def align_svd_signs(
    u_ref: torch.Tensor,
    v_ref: torch.Tensor,
    u_target: torch.Tensor,
    v_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if u_ref.shape != u_target.shape:
        raise ValueError(f"SVD U shapes mismatch: {tuple(u_ref.shape)} vs {tuple(u_target.shape)}")
    if v_ref.shape != v_target.shape:
        raise ValueError(f"SVD V shapes mismatch: {tuple(v_ref.shape)} vs {tuple(v_target.shape)}")
    signs = torch.sign((u_ref * u_target).sum(dim=0))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs).to(dtype=u_target.dtype)
    return u_target * signs.unsqueeze(0), v_target * signs.unsqueeze(1)


def infer_blocktt_dims(left_tensor: torch.Tensor, right_tensor: torch.Tensor) -> dict[str, int]:
    if left_tensor.ndim != 3:
        raise ValueError(f"BlockTT left tensor must be 3D, got shape {tuple(left_tensor.shape)}")
    if right_tensor.ndim != 3:
        raise ValueError(f"BlockTT right tensor must be 3D, got shape {tuple(right_tensor.shape)}")

    m, nr, a = left_tensor.shape
    n, b, mr = right_tensor.shape
    if nr % n != 0:
        raise ValueError(
            f"BlockTT left shape {tuple(left_tensor.shape)} is incompatible with right n={n}"
        )
    if mr % m != 0:
        raise ValueError(
            f"BlockTT right shape {tuple(right_tensor.shape)} is incompatible with left m={m}"
        )
    r_from_left = nr // n
    r_from_right = mr // m
    if r_from_left != r_from_right:
        raise ValueError(
            "BlockTT rank mismatch inferred from left/right factors: "
            f"{r_from_left} vs {r_from_right}"
        )
    return {"m": m, "n": n, "r": r_from_left, "a": a, "b": b}


def apply_color_scale(arr: np.ndarray, color_scale: str, eps: float = 1e-12) -> np.ndarray:
    if color_scale == "linear":
        return arr
    if color_scale == "log":
        return np.log10(arr + eps)
    raise ValueError(f"Unsupported color scale: {color_scale}")


def _format_corr(v: float) -> str:
    if math.isnan(v):
        return "nan"
    return f"{v:.4f}"


def _resolve_blocktt_slice_layout(
    dims: dict[str, int],
    subplot_shape: tuple[int, int] | None,
) -> tuple[int, int]:
    num_slices = dims["m"] * dims["n"]
    if subplot_shape is None:
        cols = max(1, int(math.ceil(math.sqrt(num_slices))))
        rows = max(1, int(math.ceil(num_slices / cols)))
        return rows, cols
    rows, cols = subplot_shape
    if rows * cols != num_slices:
        raise ValueError(
            "--blocktt-subplot-shape must satisfy rows*cols == m*n "
            f"(got {rows}x{cols} for m*n={num_slices})"
        )
    return rows, cols


def _figure_size_for_square_cells(rows: int, cols: int, cell_h: int, cell_w: int) -> tuple[float, float]:
    raw_w = max(1, cols * cell_w)
    raw_h = max(1, rows * cell_h)
    scale = 0.35
    fig_w = raw_w * scale
    fig_h = raw_h * scale

    max_dim = 40.0
    if max(fig_w, fig_h) > max_dim:
        shrink = max_dim / max(fig_w, fig_h)
        fig_w *= shrink
        fig_h *= shrink

    min_dim = 6.0
    if min(fig_w, fig_h) < min_dim:
        grow = min_dim / max(1e-9, min(fig_w, fig_h))
        fig_w *= grow
        fig_h *= grow

    return fig_w, fig_h


def _blocktt_flat_to_slices(
    side: str,
    data: torch.Tensor,
    dims: dict[str, int],
) -> tuple[np.ndarray, int, int, str, str]:
    m = dims["m"]
    n = dims["n"]
    r = dims["r"]
    a = dims["a"]
    b = dims["b"]

    if side == "left":
        expected = (m * n * r, a)
        if tuple(data.shape) != expected:
            raise ValueError(f"Expected BlockTT left shape {expected}, got {tuple(data.shape)}")
        slices = data.reshape(m, n, r, a).permute(0, 1, 3, 2).cpu().numpy()  # (m, n, a, r)
        return slices, a, r, "Core L", "r"

    if side == "right":
        expected = (n * b, m * r)
        if tuple(data.shape) != expected:
            raise ValueError(f"Expected BlockTT right shape {expected}, got {tuple(data.shape)}")
        slices = data.reshape(n, b, m, r).permute(2, 0, 3, 1).cpu().numpy()  # (m, n, r, b)
        return slices, r, b, "Core R", "b"

    raise ValueError(f"Unsupported BlockTT side: {side}")


def save_blocktt_side_figure(
    layer_prefix: str,
    side: str,
    grad_data: torch.Tensor,
    update_data: torch.Tensor,
    grad_corr: float,
    update_corr: float,
    dims: dict[str, int],
    out_path: Path,
    color_scale: str,
    subplot_shape: tuple[int, int] | None,
) -> None:
    import matplotlib.pyplot as plt

    rows, cols = _resolve_blocktt_slice_layout(dims, subplot_shape)
    m = dims["m"]
    n = dims["n"]

    grad_slices, cell_h, cell_w, side_name, x_label = _blocktt_flat_to_slices(side, grad_data, dims)
    update_slices, up_h, up_w, _, _ = _blocktt_flat_to_slices(side, update_data, dims)
    if up_h != cell_h or up_w != cell_w:
        raise ValueError("Grad/update slice shapes must match for BlockTT plotting")

    scaled_grad = apply_color_scale(grad_slices, color_scale)
    scaled_update = apply_color_scale(update_slices, color_scale)
    grad_vmin = float(np.min(scaled_grad))
    grad_vmax = float(np.max(scaled_grad))
    if grad_vmin == grad_vmax:
        grad_vmax = grad_vmin + 1e-12
    update_vmin = float(np.min(scaled_update))
    update_vmax = float(np.max(scaled_update))
    if update_vmin == update_vmax:
        update_vmax = update_vmin + 1e-12

    fig_w, panel_h = _figure_size_for_square_cells(rows, cols, cell_h, cell_w)
    fig_h = panel_h * 2.0
    fig, axes = plt.subplots(
        rows * 2,
        cols,
        figsize=(fig_w, fig_h),
        constrained_layout=True,
        squeeze=False,
    )
    last_grad_im = None
    last_update_im = None

    sections = [
        ("grad", scaled_grad, grad_corr, 0, grad_vmin, grad_vmax),
        ("update", scaled_update, update_corr, rows, update_vmin, update_vmax),
    ]
    num_slices = m * n
    for section_name, section_data, corr_value, row_offset, vmin, vmax in sections:
        for idx in range(rows * cols):
            ax = axes[row_offset + idx // cols, idx % cols]
            if idx >= num_slices:
                ax.axis("off")
                continue

            mi = idx // n
            ni = idx % n
            arr = section_data[mi, ni]
            im = ax.imshow(arr, aspect="equal", interpolation="nearest", vmin=vmin, vmax=vmax)
            ax.set_title(f"{section_name} | m={mi}, n={ni}", fontsize=11)
            ax.set_xlabel(x_label, fontsize=8)
            ax.set_ylabel("a" if side == "left" else "r", fontsize=8)
            ax.tick_params(labelsize=7)
            if section_name == "grad":
                last_grad_im = im
            else:
                last_update_im = im

    # One colorbar per section, placed beside the last column of each section's rows.
    grad_axes = axes[:rows, :].ravel().tolist()
    update_axes = axes[rows:, :].ravel().tolist()
    if last_grad_im is not None:
        fig.colorbar(last_grad_im, ax=grad_axes, fraction=0.02, pad=0.01)
    if last_update_im is not None:
        fig.colorbar(last_update_im, ax=update_axes, fraction=0.02, pad=0.01)

    fig.suptitle(
        f"BLOCKTT {side_name} by (m,n) slice | top=grad (corr={_format_corr(grad_corr)}), "
        f"bottom=update (corr={_format_corr(update_corr)})\n"
        f"{layer_prefix} ({color_scale} scale)",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_heatmap_figure(
    mode: str,
    layer_prefix: str,
    trained_sides: list[str],
    left_grad: torch.Tensor | None,
    left_update: torch.Tensor | None,
    right_grad: torch.Tensor | None,
    right_update: torch.Tensor | None,
    left_corr_grad: float,
    left_corr_update: float,
    right_corr_grad: float,
    right_corr_update: float,
    out_path: Path,
    color_scale: str,
    blocktt_dims: dict[str, int] | None = None,
    blocktt_subplot_shape: tuple[int, int] | None = None,
) -> list[Path]:
    import matplotlib.pyplot as plt

    left_name = "U" if mode in {"svd", "full"} else "Core L"
    right_name = "V" if mode in {"svd", "full"} else "Core R"

    entries = []
    if mode == "full":
        if left_update is not None:
            entries.append(
                {
                    "side": "left",
                    "kind": "update",
                    "data": left_update,
                    "title": f"{left_name} |total update|",
                    "corr": float("nan"),
                }
            )
        if right_update is not None:
            entries.append(
                {
                    "side": "right",
                    "kind": "update",
                    "data": right_update,
                    "title": f"{right_name} |total update|",
                    "corr": float("nan"),
                }
            )
    elif "left" in trained_sides and left_grad is not None and left_update is not None:
        entries.append(
            {
                "side": "left",
                "kind": "grad",
                "data": left_grad,
                "title": f"{left_name} |grad| at first step (corr={_format_corr(left_corr_grad)})",
                "corr": left_corr_grad,
            }
        )
        entries.append(
            {
                "side": "left",
                "kind": "update",
                "data": left_update,
                "title": f"{left_name} |total update| (corr={_format_corr(left_corr_update)})",
                "corr": left_corr_update,
            }
        )
    if mode != "full" and "right" in trained_sides and right_grad is not None and right_update is not None:
        entries.append(
            {
                "side": "right",
                "kind": "grad",
                "data": right_grad,
                "title": f"{right_name} |grad| at first step (corr={_format_corr(right_corr_grad)})",
                "corr": right_corr_grad,
            }
        )
        entries.append(
            {
                "side": "right",
                "kind": "update",
                "data": right_update,
                "title": f"{right_name} |total update| (corr={_format_corr(right_corr_update)})",
                "corr": right_corr_update,
            }
        )

    if not entries:
        raise ValueError("No trained side with available gradients to plot.")

    if mode == "blocktt":
        if blocktt_dims is None:
            raise ValueError("blocktt_dims is required for BlockTT slice plotting")

        out_paths: list[Path] = []
        side_data: dict[str, dict[str, Any]] = {}
        for entry in entries:
            payload = side_data.setdefault(entry["side"], {})
            payload[entry["kind"]] = entry["data"]
            payload[f"{entry['kind']}_corr"] = float(entry["corr"])

        for side, payload in side_data.items():
            if "grad" not in payload or "update" not in payload:
                continue
            entry_path = out_path.with_name(f"{out_path.stem}_{side}_slices{out_path.suffix}")
            save_blocktt_side_figure(
                layer_prefix=layer_prefix,
                side=side,
                grad_data=payload["grad"],
                update_data=payload["update"],
                grad_corr=payload["grad_corr"],
                update_corr=payload["update_corr"],
                dims=blocktt_dims,
                out_path=entry_path,
                color_scale=color_scale,
                subplot_shape=blocktt_subplot_shape,
            )
            out_paths.append(entry_path)
        if not out_paths:
            raise ValueError("No complete BlockTT side data available for combined grad/update plotting")
        return out_paths

    n_plots = len(entries)
    n_rows = 1 if n_plots <= 2 else 2
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows), constrained_layout=True)
    axes = np.array(axes).reshape(n_rows, n_cols)

    # Compute per-kind (grad / update) shared vmin/vmax.
    kind_scaled: dict[str, list[np.ndarray]] = {}
    for entry in entries:
        scaled = apply_color_scale(entry["data"].cpu().numpy(), color_scale)
        kind_scaled.setdefault(entry["kind"], []).append(scaled)
    kind_vmin: dict[str, float] = {}
    kind_vmax: dict[str, float] = {}
    for kind, arrays in kind_scaled.items():
        vmin_k = float(min(a.min() for a in arrays))
        vmax_k = float(max(a.max() for a in arrays))
        if vmin_k == vmax_k:
            vmax_k = vmin_k + 1e-12
        kind_vmin[kind] = vmin_k
        kind_vmax[kind] = vmax_k

    last_im_by_kind: dict[str, Any] = {}
    kind_axes: dict[str, list] = {}
    for idx, entry in enumerate(entries):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r, c]
        kind = entry["kind"]
        data = apply_color_scale(entry["data"].cpu().numpy(), color_scale)
        im = ax.imshow(
            data,
            aspect="auto",
            interpolation="nearest",
            vmin=kind_vmin[kind],
            vmax=kind_vmax[kind],
        )
        ax.set_title(entry["title"])
        ax.set_xlabel("column")
        ax.set_ylabel("row")
        last_im_by_kind[kind] = im
        kind_axes.setdefault(kind, []).append(ax)

    # One colorbar per kind, shared across all axes of that kind.
    for kind, im in last_im_by_kind.items():
        fig.colorbar(im, ax=kind_axes[kind], fraction=0.046, pad=0.04)

    # Hide unused axes.
    for idx in range(len(entries), n_rows * n_cols):
        r = idx // n_cols
        c = idx % n_cols
        axes[r, c].axis("off")

    fig.suptitle(f"{mode.upper()} — {layer_prefix} ({color_scale} scale)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return [out_path]


def run_mode_analysis(
    mode: str,
    run_dir: Path,
    layer_prefix: str,
    base_model: torch.nn.Module | None,
    color_scale: str,
    final_step: int | None,
    full_step_start: int,
    force_replay_grad: bool,
    allow_replay_grad: bool,
) -> dict[str, Any]:
    run_cfg, run_cli_args = load_run_context(run_dir)

    if mode == "full":
        if full_step_start < 0:
            raise ValueError("--full-step-start must be >= 0")
        final_step_resolved = final_step if final_step is not None else find_max_step(run_dir)
        if final_step_resolved <= full_step_start:
            raise ValueError(
                "For full mode, final step must be greater than --full-step-start "
                f"(got start={full_step_start}, final={final_step_resolved})"
            )

        start_weight, start_path = load_full_weight_checkpoint(run_dir, layer_prefix, full_step_start)
        final_weight, final_path = load_full_weight_checkpoint(run_dir, layer_prefix, final_step_resolved)
        u_start, v_start = svd_factors_from_weight(start_weight)
        u_final, v_final = svd_factors_from_weight(final_weight)
        u_final_aligned, v_final_aligned = align_svd_signs(u_start, v_start, u_final, v_final)

        return {
            "run_dir": str(run_dir),
            "mode": mode,
            "layer_prefix": layer_prefix,
            "start_step": full_step_start,
            "start_checkpoint": str(start_path),
            "final_step": final_step_resolved,
            "final_checkpoint": str(final_path),
            "grad_cache": None,
            "trained_sides": ["left", "right"],
            "left_grad": None,
            "right_grad": None,
            "left_update": (u_final_aligned - u_start).abs(),
            "right_update": (v_final_aligned - v_start).abs(),
            "left_corr_grad": float("nan"),
            "left_corr_update": float("nan"),
            "right_corr_grad": float("nan"),
            "right_corr_update": float("nan"),
            "blocktt_dims": None,
            "update_only": True,
        }

    if base_model is None:
        raise ValueError(f"base_model is required for mode={mode}")

    init_left, init_right, init_s = build_initialized_factors(base_model, mode, layer_prefix, run_cfg)

    final_left, final_right, final_s, used_step, model_path = load_final_factors(
        run_dir=run_dir,
        mode=mode,
        layer_prefix=layer_prefix,
        final_step=final_step,
    )

    trained_sides = resolve_trained_sides(mode, run_cfg, init_left, init_right)

    grad_left_raw, grad_right_raw, grad_cache_path = load_or_compute_first_step_gradients(
        run_dir=run_dir,
        run_cfg=run_cfg,
        run_cli_args=run_cli_args,
        mode=mode,
        layer_prefix=layer_prefix,
        trained_sides=trained_sides,
        force_replay=force_replay_grad,
        allow_replay=allow_replay_grad,
    )

    total_update_left = (final_left - init_left).abs()
    total_update_right = (final_right - init_right).abs()

    grad_left = (
        flatten_heatmap_tensor(mode, "left", grad_left_raw)
        if grad_left_raw is not None
        else None
    )
    grad_right = (
        flatten_heatmap_tensor(mode, "right", grad_right_raw)
        if grad_right_raw is not None
        else None
    )
    update_left = (
        flatten_heatmap_tensor(mode, "left", total_update_left)
        if "left" in trained_sides
        else None
    )
    update_right = (
        flatten_heatmap_tensor(mode, "right", total_update_right)
        if "right" in trained_sides
        else None
    )
    blocktt_dims = (
        infer_blocktt_dims(init_left, init_right) if mode == "blocktt" else None
    )

    singular = init_s if init_s is not None else final_s
    left_s_map, right_s_map = build_singular_maps(mode, singular, init_left, init_right)

    if left_s_map is None or right_s_map is None:
        left_corr_grad = float("nan")
        left_corr_update = float("nan")
        right_corr_grad = float("nan")
        right_corr_update = float("nan")
    else:
        left_corr_grad = (
            pearson_corr(left_s_map, grad_left)
            if grad_left is not None and "left" in trained_sides
            else float("nan")
        )
        left_corr_update = (
            pearson_corr(left_s_map, update_left)
            if update_left is not None and "left" in trained_sides
            else float("nan")
        )
        right_corr_grad = (
            pearson_corr(right_s_map, grad_right)
            if grad_right is not None and "right" in trained_sides
            else float("nan")
        )
        right_corr_update = (
            pearson_corr(right_s_map, update_right)
            if update_right is not None and "right" in trained_sides
            else float("nan")
        )

    return {
        "run_dir": str(run_dir),
        "mode": mode,
        "layer_prefix": layer_prefix,
        "start_step": None,
        "start_checkpoint": None,
        "final_step": used_step,
        "final_checkpoint": str(model_path),
        "grad_cache": str(grad_cache_path),
        "trained_sides": trained_sides,
        "left_grad": grad_left,
        "right_grad": grad_right,
        "left_update": update_left,
        "right_update": update_right,
        "left_corr_grad": left_corr_grad,
        "left_corr_update": left_corr_update,
        "right_corr_grad": right_corr_grad,
        "right_corr_update": right_corr_update,
        "blocktt_dims": blocktt_dims,
        "update_only": False,
    }


# ---------------------------------------------------------------------------
# SFT / pre-saved-grads analysis
# ---------------------------------------------------------------------------

def _load_grads_for_prefix(
    grads_path: Path,
    mode: str,
    layer_prefix: str,
    trained_sides: list[str],
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Load grad tensors for a layer prefix from a grads.safetensors file."""
    keys = factor_keys_for_mode(mode, layer_prefix)
    grads = load_file(str(grads_path))
    left_grad = grads.get(keys["left"])
    right_grad = grads.get(keys["right"])
    left_grad = left_grad.abs().float() if left_grad is not None else None
    right_grad = right_grad.abs().float() if right_grad is not None else None
    return left_grad, right_grad


def _load_final_factors_sharded(
    run_dir: Path,
    mode: str,
    layer_prefix: str,
    final_step: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int, Path]:
    """Load final factors from a (possibly sharded) checkpoint."""
    step = final_step if final_step is not None else find_max_step(run_dir)
    step_dir = run_dir / f"step={step}"
    if not step_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {step_dir}")

    keys = factor_keys_for_mode(mode, layer_prefix)
    left = load_tensor_from_checkpoint(step_dir, keys["left"])
    right = load_tensor_from_checkpoint(step_dir, keys["right"])
    singular = load_tensor_from_checkpoint(step_dir, keys["singular"])

    if left is None or right is None:
        raise KeyError(
            f"Checkpoint {step_dir} missing required factor keys for {layer_prefix} in mode {mode}"
        )
    return left.float(), right.float(), (singular.float() if singular is not None else None), step, step_dir


def run_sft_analysis(
    run_dir: Path,
    layer_prefix: str,
    to_plot: list[str],
    run_cfg: dict[str, Any],
    base_model: torch.nn.Module | None,
    grad_n_step: int | None,
    final_step: int | None,
) -> dict[str, Any]:
    """Run analysis for an SFT/RL run with pre-saved grads and checkpoints."""
    mode = run_cfg["train_mode"]
    result: dict[str, Any] = {
        "run_dir": str(run_dir),
        "mode": mode,
        "layer_prefix": layer_prefix,
        "to_plot": to_plot,
    }

    # Find available grads
    grad_paths = find_grad_paths(run_dir)

    # Build init factors if we need update or trained_sides resolution
    init_left = init_right = init_s = None
    final_left = final_right = final_s = None
    used_step = None
    if "update" in to_plot:
        if base_model is None:
            raise ValueError("--base-model-id is required for update computation")

        # Load final factors first to infer BTT dimensions from shapes
        final_left, final_right, final_s, used_step, ckpt_path = _load_final_factors_sharded(
            run_dir, mode, layer_prefix, final_step,
        )
        if mode == "blocktt":
            dims = infer_blocktt_dims(final_left, final_right)
            # Override run_cfg with exact dims from checkpoint so init matches
            run_cfg = {
                **run_cfg,
                "blocktt_rank": dims["r"],
                "_output_factorization": (dims["m"], dims["a"]),
                "_input_factorization": (dims["n"], dims["b"]),
            }

        init_left, init_right, init_s = build_initialized_factors(
            base_model, mode, layer_prefix, run_cfg,
        )

    # Resolve trained sides from init factors or from grad keys
    if init_left is not None:
        trained_sides = resolve_trained_sides(mode, run_cfg, init_left, init_right)
    else:
        first_grad_path = grad_paths.get(0) or next(iter(grad_paths.values()), None)
        if first_grad_path is not None:
            keys = factor_keys_for_mode(mode, layer_prefix)
            grads = load_file(str(first_grad_path))
            trained_sides = []
            if keys["left"] in grads:
                trained_sides.append("left")
            if keys["right"] in grads:
                trained_sides.append("right")
        else:
            trained_sides = ["left", "right"]
    result["trained_sides"] = trained_sides

    # Grad at step 0
    if "grad_0" in to_plot:
        if 0 not in grad_paths:
            raise FileNotFoundError(f"No grads at step 0 in {run_dir}")
        left_g0, right_g0 = _load_grads_for_prefix(grad_paths[0], mode, layer_prefix, trained_sides)
        result["grad_0_left"] = flatten_heatmap_tensor(mode, "left", left_g0) if left_g0 is not None else None
        result["grad_0_right"] = flatten_heatmap_tensor(mode, "right", right_g0) if right_g0 is not None else None
        result["grad_0_step"] = 0

    # Grad at step n
    if "grad_n" in to_plot:
        if grad_n_step is not None:
            step_n = grad_n_step
        else:
            non_zero = [s for s in grad_paths if s > 0]
            if not non_zero:
                raise FileNotFoundError(f"No grads beyond step 0 in {run_dir}")
            step_n = max(non_zero)
        if step_n not in grad_paths:
            raise FileNotFoundError(f"No grads at step {step_n} in {run_dir}. Available: {sorted(grad_paths.keys())}")
        left_gn, right_gn = _load_grads_for_prefix(grad_paths[step_n], mode, layer_prefix, trained_sides)
        result["grad_n_left"] = flatten_heatmap_tensor(mode, "left", left_gn) if left_gn is not None else None
        result["grad_n_right"] = flatten_heatmap_tensor(mode, "right", right_gn) if right_gn is not None else None
        result["grad_n_step"] = step_n

    # Update (final - init) — final factors already loaded above
    if "update" in to_plot:
        update_left = (final_left - init_left).abs() if "left" in trained_sides else None
        update_right = (final_right - init_right).abs() if "right" in trained_sides else None
        result["update_left"] = flatten_heatmap_tensor(mode, "left", update_left) if update_left is not None else None
        result["update_right"] = flatten_heatmap_tensor(mode, "right", update_right) if update_right is not None else None
        result["final_step"] = used_step

    # BlockTT dims and singular maps for correlation
    blocktt_dims = None
    if mode == "blocktt" and init_left is not None:
        blocktt_dims = infer_blocktt_dims(init_left, init_right)
    result["blocktt_dims"] = blocktt_dims

    singular = init_s
    if singular is not None and init_left is not None:
        left_s_map, right_s_map = build_singular_maps(mode, singular, init_left, init_right)
    else:
        left_s_map = right_s_map = None
    result["left_s_map"] = left_s_map
    result["right_s_map"] = right_s_map

    return result


def save_sft_heatmap_figure(
    layer_prefix: str,
    to_plot: list[str],
    analysis: dict[str, Any],
    out_path: Path,
    color_scale: str,
    blocktt_subplot_shape: tuple[int, int] | None = None,
) -> list[Path]:
    """Save heatmap figure(s) for SFT analysis results."""
    import matplotlib.pyplot as plt

    mode = analysis["mode"]
    trained_sides = analysis["trained_sides"]
    blocktt_dims = analysis.get("blocktt_dims")

    # Collect panels: list of (label, side, data_tensor)
    panels: list[tuple[str, str, torch.Tensor]] = []
    for plot_type in to_plot:
        if plot_type == "grad_0":
            step = analysis.get("grad_0_step", 0)
            label = f"|grad| step={step}"
            if "left" in trained_sides and analysis.get("grad_0_left") is not None:
                panels.append((label, "left", analysis["grad_0_left"]))
            if "right" in trained_sides and analysis.get("grad_0_right") is not None:
                panels.append((label, "right", analysis["grad_0_right"]))
        elif plot_type == "grad_n":
            step = analysis.get("grad_n_step", "?")
            label = f"|grad| step={step}"
            if "left" in trained_sides and analysis.get("grad_n_left") is not None:
                panels.append((label, "left", analysis["grad_n_left"]))
            if "right" in trained_sides and analysis.get("grad_n_right") is not None:
                panels.append((label, "right", analysis["grad_n_right"]))
        elif plot_type == "update":
            label = f"|update| (step={analysis.get('final_step', '?')} - init)"
            if "left" in trained_sides and analysis.get("update_left") is not None:
                panels.append((label, "left", analysis["update_left"]))
            if "right" in trained_sides and analysis.get("update_right") is not None:
                panels.append((label, "right", analysis["update_right"]))

    if not panels:
        print(f"Warning: no data to plot for {layer_prefix}")
        return []

    # For blocktt with slice layout, produce per-side slice figures
    if mode == "blocktt" and blocktt_dims is not None:
        out_paths: list[Path] = []
        # Group panels by side
        by_side: dict[str, list[tuple[str, torch.Tensor]]] = {}
        for label, side, data in panels:
            by_side.setdefault(side, []).append((label, data))

        for side, side_panels in by_side.items():
            n_panels = len(side_panels)
            side_name = "Core L" if side == "left" else "Core R"
            m = blocktt_dims["m"]
            n = blocktt_dims["n"]
            rows, cols = _resolve_blocktt_slice_layout(blocktt_dims, blocktt_subplot_shape)
            num_slices = m * n

            _, cell_h, cell_w, _, x_label = _blocktt_flat_to_slices(side, side_panels[0][1], blocktt_dims)
            fig_w, panel_h = _figure_size_for_square_cells(rows, cols, cell_h, cell_w)
            fig_h = panel_h * n_panels
            fig, axes = plt.subplots(
                rows * n_panels, cols,
                figsize=(fig_w, fig_h),
                constrained_layout=True,
                squeeze=False,
            )

            # Compute per-panel vmin/vmax (each panel gets its own shared range across its slices)
            all_slices = []
            for label, data in side_panels:
                slices, _, _, _, _ = _blocktt_flat_to_slices(side, data, blocktt_dims)
                scaled = apply_color_scale(slices, color_scale)
                p_vmin = float(np.min(scaled))
                p_vmax = float(np.max(scaled))
                if p_vmin == p_vmax:
                    p_vmax = p_vmin + 1e-12
                all_slices.append((label, scaled, p_vmin, p_vmax))

            last_im_by_panel: list[Any] = [None] * n_panels
            panel_axes: list[list] = [[] for _ in range(n_panels)]
            for panel_idx, (label, scaled_slices, p_vmin, p_vmax) in enumerate(all_slices):
                row_offset = panel_idx * rows
                for idx in range(rows * cols):
                    ax = axes[row_offset + idx // cols, idx % cols]
                    if idx >= num_slices:
                        ax.axis("off")
                        continue
                    mi = idx // n
                    ni = idx % n
                    arr = scaled_slices[mi, ni]
                    im = ax.imshow(arr, aspect="equal", interpolation="nearest", vmin=p_vmin, vmax=p_vmax)
                    ax.set_title(f"{label} | m={mi}, n={ni}", fontsize=9)
                    ax.set_xlabel(x_label, fontsize=7)
                    ax.set_ylabel("a" if side == "left" else "r", fontsize=7)
                    ax.tick_params(labelsize=6)
                    last_im_by_panel[panel_idx] = im
                    panel_axes[panel_idx].append(ax)

            for panel_idx, im in enumerate(last_im_by_panel):
                if im is not None:
                    fig.colorbar(im, ax=panel_axes[panel_idx], fraction=0.02, pad=0.01)
            fig.suptitle(
                f"SFT BLOCKTT {side_name} by (m,n) slice\n{layer_prefix} ({color_scale} scale)",
                fontsize=11,
            )
            entry_path = out_path.with_name(f"{out_path.stem}_{side}_slices{out_path.suffix}")
            entry_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(entry_path, dpi=180)
            plt.close(fig)
            out_paths.append(entry_path)

        return out_paths

    # Non-blocktt: simple grid layout
    n_plots = len(panels)
    n_cols = min(n_plots, 2)
    n_rows = math.ceil(n_plots / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), constrained_layout=True)
    axes = np.array(axes).reshape(n_rows, n_cols) if n_rows * n_cols > 1 else np.array([[axes]])

    left_name = "U" if mode in {"svd", "full"} else "Core L"
    right_name = "V" if mode in {"svd", "full"} else "Core R"

    for idx, (label, side, data) in enumerate(panels):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r, c]
        side_name = left_name if side == "left" else right_name
        arr = apply_color_scale(data.cpu().numpy(), color_scale)
        im = ax.imshow(arr, aspect="auto", interpolation="nearest")
        ax.set_title(f"{side_name} {label}")
        ax.set_xlabel("column")
        ax.set_ylabel("row")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(len(panels), n_rows * n_cols):
        r = idx // n_cols
        c = idx % n_cols
        axes[r, c].axis("off")

    fig.suptitle(f"SFT {mode.upper()} — {layer_prefix} ({color_scale} scale)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return [out_path]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    blocktt_subplot_shape = parse_subplot_shape(args.blocktt_subplot_shape)

    # SFT run path
    if args.sft_run is not None:
        run_dir = args.sft_run
        run_cfg = parse_run_dir_config(run_dir)
        mode = run_cfg["train_mode"]
        to_plot = args.to_plot or ["grad_0", "grad_n", "update"]

        layer_prefixes = resolve_layer_prefixes(args.layer_prefix, run_dir)
        print(f"Resolved {len(layer_prefixes)} layer prefix(es) for '{args.layer_prefix}'")

        # Load base model if needed
        base_model = None
        if "update" in to_plot:
            model_id = args.base_model_id
            if model_id is None:
                raise ValueError("--base-model-id is required when plotting 'update' with --sft-run")
            from transformers import AutoModelForCausalLM
            device = torch.device(args.device)
            base_model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.float32, device_map=None,
            ).to(device)
            base_model.eval()

        args.output_dir.mkdir(parents=True, exist_ok=True)
        all_fig_paths: list[Path] = []
        for lp in layer_prefixes:
            print(f"Analyzing {lp} ...")
            analysis = run_sft_analysis(
                run_dir=run_dir,
                layer_prefix=lp,
                to_plot=to_plot,
                run_cfg=run_cfg,
                base_model=base_model,
                grad_n_step=args.grad_n_step,
                final_step=args.final_step,
            )
            fig_path = args.output_dir / f"sft_{mode}_{lp.replace('.', '_')}_heatmaps.png"
            fig_paths = save_sft_heatmap_figure(
                layer_prefix=lp,
                to_plot=to_plot,
                analysis=analysis,
                out_path=fig_path,
                color_scale=args.color_scale,
                blocktt_subplot_shape=blocktt_subplot_shape,
            )
            for p in fig_paths:
                print(f"  Saved: {p}")
            all_fig_paths.extend(fig_paths)

        print(f"Done. {len(all_fig_paths)} figure(s) saved to {args.output_dir}")
        return

    # Existing RL path
    if not is_exact_layer_prefix(args.layer_prefix):
        raise ValueError(
            "For --svd-run/--blocktt-run/--full-run, --layer-prefix must be fully qualified "
            "(e.g. model.layers.12.self_attn.q_proj). "
            "Use --sft-run for flexible prefix patterns."
        )
    parse_layer_prefix(args.layer_prefix)

    run_by_mode: dict[str, Path] = {}
    if args.svd_run is not None:
        run_by_mode["svd"] = args.svd_run
    if args.blocktt_run is not None:
        run_by_mode["blocktt"] = args.blocktt_run
    if args.full_run is not None:
        run_by_mode["full"] = args.full_run

    if not run_by_mode:
        raise ValueError("Provide at least one of --sft-run, --svd-run, --blocktt-run, or --full-run")

    model_id = args.base_model_id
    if model_id is None:
        first_cfg, _ = load_run_context(next(iter(run_by_mode.values())))
        model_id = first_cfg.get("model_id")
        if model_id is None:
            raise ValueError("Could not infer model_id from run config; pass --base-model-id")

    base_model = None
    needs_base_model = any(mode in {"svd", "blocktt"} for mode in run_by_mode)
    if needs_base_model:
        from transformers import AutoModelForCausalLM

        device = torch.device(args.device)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map=None,
        ).to(device)
        base_model.eval()

    results: dict[str, Any] = {
        "model_id": model_id,
        "layer_prefix": args.layer_prefix,
        "color_scale": args.color_scale,
        "modes": {},
    }

    for mode, run_dir in run_by_mode.items():
        analysis = run_mode_analysis(
            mode=mode,
            run_dir=run_dir,
            layer_prefix=args.layer_prefix,
            base_model=base_model,
            color_scale=args.color_scale,
            final_step=args.final_step,
            full_step_start=args.full_step_start,
            force_replay_grad=args.force_replay_grad,
            allow_replay_grad=not args.no_replay_grad,
        )

        fig_path = args.output_dir / f"{mode}_{args.layer_prefix.replace('.', '_')}_heatmaps.png"
        fig_paths = save_heatmap_figure(
            mode=mode,
            layer_prefix=args.layer_prefix,
            trained_sides=analysis["trained_sides"],
            left_grad=analysis["left_grad"],
            left_update=analysis["left_update"],
            right_grad=analysis["right_grad"],
            right_update=analysis["right_update"],
            left_corr_grad=analysis["left_corr_grad"],
            left_corr_update=analysis["left_corr_update"],
            right_corr_grad=analysis["right_corr_grad"],
            right_corr_update=analysis["right_corr_update"],
            out_path=fig_path,
            color_scale=args.color_scale,
            blocktt_dims=analysis["blocktt_dims"],
            blocktt_subplot_shape=blocktt_subplot_shape,
        )
        heatmap_field: str | list[str]
        if len(fig_paths) == 1:
            heatmap_field = str(fig_paths[0])
        else:
            heatmap_field = [str(p) for p in fig_paths]

        results["modes"][mode] = {
            "run_dir": analysis["run_dir"],
            "start_step": analysis["start_step"],
            "start_checkpoint": analysis["start_checkpoint"],
            "final_step": analysis["final_step"],
            "final_checkpoint": analysis["final_checkpoint"],
            "grad_cache": analysis["grad_cache"],
            "update_only": analysis["update_only"],
            "heatmap": heatmap_field,
            "corr": {
                "left_grad": analysis["left_corr_grad"],
                "left_update": analysis["left_corr_update"],
                "right_grad": analysis["right_corr_grad"],
                "right_update": analysis["right_corr_update"],
            },
            "trained_sides": analysis["trained_sides"],
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / f"{args.layer_prefix.replace('.', '_')}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved metrics: {metrics_path}")
    for mode in results["modes"]:
        heatmap = results["modes"][mode]["heatmap"]
        if isinstance(heatmap, list):
            for p in heatmap:
                print(f"Saved {mode} heatmap: {p}")
        else:
            print(f"Saved {mode} heatmap: {heatmap}")


if __name__ == "__main__":
    main()
