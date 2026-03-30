from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from btt_layer import (  # noqa: E402
    BTTLayer,
    configure_blocktt_trainability,
    convert_linear_to_btt,
    get_blocktt_target_module_names,
    resolve_blocktt_decomp_modes,
)


BLOCKTT_CONFIG_FILENAME = "blocktt_config.json"


def parse_blocktt_rank(raw_rank: Any) -> int | str:
    if isinstance(raw_rank, int):
        if raw_rank <= 0:
            raise ValueError("--blocktt-rank must be a positive integer or 'full'")
        return raw_rank

    rank_text = str(raw_rank).strip().lower()
    if rank_text == "full":
        return "full"

    try:
        rank_int = int(rank_text)
    except ValueError as exc:
        raise ValueError("--blocktt-rank must be a positive integer or 'full'") from exc
    if rank_int <= 0:
        raise ValueError("--blocktt-rank must be a positive integer or 'full'")
    return rank_int


def apply_blocktt_to_model(
    model,
    *,
    blocktt_rank: int | str,
    blocktt_type: str,
    decomp_mode: str,
    train_position: str,
    s_merged_to: str | None,
    train_bias: bool,
    set_trainability: bool = True,
):
    resolved_rank = parse_blocktt_rank(blocktt_rank)
    target_modules = get_blocktt_target_module_names(blocktt_type)
    decomp_display, module_decomp_modes = resolve_blocktt_decomp_modes(
        decomp_mode,
        include_names=target_modules,
    )

    converted_modules = convert_linear_to_btt(
        model,
        btt_rank=resolved_rank,
        decomp_mode=module_decomp_modes,
        include_names=target_modules,
        train_position=train_position,
        s_merged_to=s_merged_to,
    )

    trainability_stats = None
    if set_trainability:
        trainability_stats = configure_blocktt_trainability(
            model,
            train_bias=train_bias,
            train_position=train_position,
        )

    metadata = {
        "blocktt_rank": resolved_rank,
        "blocktt_type": blocktt_type,
        "decomp_mode_input": decomp_mode,
        "decomp_mode_resolved": decomp_display,
        "module_decomp_modes": module_decomp_modes,
        "target_modules": list(target_modules),
        "train_position": train_position,
        "s_merged_to": s_merged_to,
        "train_bias": train_bias,
        "converted_modules": converted_modules,
    }
    return metadata, trainability_stats


def apply_blocktt_from_metadata(model, metadata: dict[str, Any], *, set_trainability: bool = False):
    return apply_blocktt_to_model(
        model,
        blocktt_rank=metadata["blocktt_rank"],
        blocktt_type=metadata["blocktt_type"],
        decomp_mode=metadata.get("decomp_mode_input", "input_one_block"),
        train_position=metadata.get("train_position", "small"),
        s_merged_to=metadata.get("s_merged_to"),
        train_bias=metadata.get("train_bias", True),
        set_trainability=set_trainability,
    )


def write_blocktt_metadata(checkpoint_dir: str | Path, metadata: dict[str, Any]) -> Path:
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    metadata_path = checkpoint_path / BLOCKTT_CONFIG_FILENAME
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    return metadata_path


def load_blocktt_metadata(checkpoint_dir: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_dir)
    metadata_path = checkpoint_path / BLOCKTT_CONFIG_FILENAME
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing BlockTT metadata file: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@torch.no_grad()
def materialize_blocktt_to_linear_(model) -> int:
    converted = 0
    for full_name, module in list(model.named_modules()):
        if not isinstance(module, BTTLayer):
            continue
        if module.lr_act:
            raise ValueError(f"BTTLayer with lr_act=True cannot be materialized: {full_name}")

        parent = model
        path = full_name.split(".")
        for key in path[:-1]:
            parent = getattr(parent, key)

        replacement = nn.Linear(
            module.in_features,
            module.out_features,
            bias=(module.bias is not None),
        ).to(device=module.btt_l.device, dtype=module.btt_l.dtype)
        replacement.weight.data.copy_(module.materialize_dense_weight().to(dtype=replacement.weight.dtype))
        if module.bias is not None:
            replacement.bias.data.copy_(module.bias.data.to(dtype=replacement.bias.dtype))
        setattr(parent, path[-1], replacement)
        converted += 1

    return converted
