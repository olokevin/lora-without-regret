"""Shared helpers for integrating src/compress calibrated BTT with the
run_sft / run_rl / run_rl_dapo / LIFT training entrypoints.

Keeps legacy btt_layer.py / svd_layer.py paths untouched; activated only
when --calib-mode (or --calib_mode) is non-'none'.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from typing import Any, Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Ensure the sibling `compress` package under <repo_root>/src is importable.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if os.path.isdir(_SRC_DIR) and _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from compress.decomposition import (  # noqa: E402
    DecompositionConfig,
    decompose_with_loader,
)
from compress.btt.btt_linear import BTTLinear  # noqa: E402
from compress.loaders import (  # noqa: E402
    build_c4_calib_loader,
    build_traces_jsonl_calib_loader,
)
from compress.topology import export_btt_topology, rebuild_btt_from_topology  # noqa: E402


# Public mapping from CLI --calib-mode value to compress train_mode.
CALIB_MODE_TO_TRAIN_MODE = {
    "v2": "btt_llm_v2",
    "v2_bp": "btt_llm_v2_bp",
    "v2_combined": "btt_llm_v2_combined",
    "twosteps": "btt_twosteps",
}
VALID_CALIB_MODES = ("none",) + tuple(CALIB_MODE_TO_TRAIN_MODE.keys())
VALID_CALIB_SOURCES = ("c4", "traces", "training_data")


def _arg_name(name: str, *, hyphen_style: bool) -> str:
    return "--" + (name if hyphen_style else name.replace("-", "_"))


def _attr_name(name: str, *, hyphen_style: bool) -> str:
    return (name if hyphen_style else name).replace("-", "_")


def add_calibrated_btt_args(parser, *, hyphen_style: bool = True) -> None:
    """Register --calib-mode / --calib-source / ... flags on the given parser."""
    parser.add_argument(
        _arg_name("calib-mode", hyphen_style=hyphen_style),
        type=str, default="none", choices=list(VALID_CALIB_MODES),
        help="Calibrated BTT mode. 'none' keeps the legacy blocktt path.",
    )
    parser.add_argument(
        _arg_name("calib-source", hyphen_style=hyphen_style),
        type=str, default="c4", choices=list(VALID_CALIB_SOURCES),
        help="Calibration data source. Only used when --calib-mode != none.",
    )
    parser.add_argument(
        _arg_name("calib-traces-path", hyphen_style=hyphen_style),
        type=str, default=None,
        help="Path to trace JSONL; required when --calib-source=traces.",
    )
    parser.add_argument(
        _arg_name("calib-num-seqs", hyphen_style=hyphen_style),
        type=int, default=128,
    )
    parser.add_argument(
        _arg_name("calib-max-length", hyphen_style=hyphen_style),
        type=int, default=2048,
    )
    parser.add_argument(
        _arg_name("calib-seed", hyphen_style=hyphen_style),
        type=int, default=3,
    )
    parser.add_argument(
        _arg_name("calib-batch-size", hyphen_style=hyphen_style),
        type=int, default=8,
    )


# ---- stubs below; filled in by subsequent tasks ----

def validate_calibrated_btt_args(args, *, argv: Sequence[str], hyphen_style: bool = True) -> None:
    raise NotImplementedError("filled in Task 7")


def build_decomposition_config(args, *, hyphen_style: bool = True) -> DecompositionConfig:
    raise NotImplementedError("filled in Task 8")


def build_training_data_calib_loader(
    dataset, collate_fn, *, num_seqs: int, batch_size: int, seed: int,
) -> DataLoader:
    raise NotImplementedError("filled in Task 9")


def build_rl_rollout_calib_loader(
    *, rl_rollout_fn, tokenizer, num_seqs: int, batch_size: int,
    max_length: int, seed: int,
) -> DataLoader:
    raise NotImplementedError("filled in Task 13")


def build_calib_loader(
    args, *, tokenizer, training_dataset=None, training_collate_fn=None,
    rl_rollout_fn=None, hyphen_style: bool = True,
) -> Optional[DataLoader]:
    raise NotImplementedError("filled in Task 10")


def apply_calibrated_btt(
    model, args, *, calib_loader, device: Optional[str] = None,
    hyphen_style: bool = True,
) -> Tuple[nn.Module, dict]:
    raise NotImplementedError("filled in Task 10")


def materialize_calibrated_btt_weights(model) -> List[Tuple[str, torch.Tensor]]:
    raise NotImplementedError("filled in Task 14")


def restore_calibrated_btt_weights(model, saved_state) -> None:
    raise NotImplementedError("filled in Task 14")


def save_calibrated_btt_checkpoint(model, out_dir: str) -> None:
    raise NotImplementedError("filled in Task 17")


def load_calibrated_btt_for_eval(model, checkpoint_dir: str) -> nn.Module:
    raise NotImplementedError("filled in Task 17")
