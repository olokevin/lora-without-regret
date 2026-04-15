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


def _attr_name(name: str) -> str:
    return name.replace("-", "_")


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
        help="Number of calibration sequences to sample.",
    )
    parser.add_argument(
        _arg_name("calib-max-length", hyphen_style=hyphen_style),
        type=int, default=2048,
        help="Max token length per calibration sample.",
    )
    parser.add_argument(
        _arg_name("calib-seed", hyphen_style=hyphen_style),
        type=int, default=3,
        help="RNG seed for calibration sampling.",
    )
    parser.add_argument(
        _arg_name("calib-batch-size", hyphen_style=hyphen_style),
        type=int, default=8,
        help="Batch size used by the calibration DataLoader.",
    )


# ---- helpers below; most are stubs filled in by later tasks ----

_CALIB_FLAG_DEFAULTS = {
    "calib_source": "c4",
    "calib_traces_path": None,
    "calib_num_seqs": 128,
    "calib_max_length": 2048,
    "calib_seed": 3,
    "calib_batch_size": 8,
}


def validate_calibrated_btt_args(args, *, argv: Sequence[str], hyphen_style: bool = True) -> None:
    """Raise ValueError if the calib-* args are inconsistent with train-mode / blocktt-rank.

    Note: ``argv`` is unused by the current rules but kept in the signature for
    stability — Task 11 wires ``argv=argv`` from run_sft.py / run_rl.py, and
    future rules may need it.
    """
    del argv  # currently unused; see docstring
    calib_mode = getattr(args, "calib_mode", "none")
    calib_source = getattr(args, "calib_source", "c4")

    # 1. --calib-mode != none requires --train-mode blocktt (if parser has train_mode)
    if calib_mode != "none" and hasattr(args, "train_mode"):
        if args.train_mode != "blocktt":
            raise ValueError(
                "--calib-mode only valid with --train-mode blocktt "
                f"(got --train-mode={args.train_mode!r})"
            )

    # 2. --calib-source=traces requires --calib-traces-path
    if calib_mode != "none" and calib_source == "traces":
        path = getattr(args, "calib_traces_path", None)
        if not path:
            flag = "--calib-traces-path" if hyphen_style else "--calib_traces_path"
            raise ValueError(f"{flag} must be set when --calib-source=traces")

    # 3. --calib-mode=none forbids any --calib-* flag differing from its default.
    #    We compare parsed attr values against known defaults rather than
    #    scanning argv for literal flag tokens, because argparse's default
    #    allow_abbrev=True lets users pass e.g. --calib-n as an abbreviation
    #    of --calib-num-seqs, which an argv-prefix scan would miss.
    if calib_mode == "none":
        non_default_flags = []
        for attr, default in _CALIB_FLAG_DEFAULTS.items():
            if getattr(args, attr, default) != default:
                flag = "--" + (attr.replace("_", "-") if hyphen_style else attr)
                non_default_flags.append(flag)
        if non_default_flags:
            raise ValueError(
                f"{', '.join(non_default_flags)} requires --calib-mode!=none "
                "(got --calib-mode=none)"
            )

    # 4. Integer --blocktt-rank rejected on calibrated path; float must be in (0, 1]
    if calib_mode != "none":
        rank_raw = getattr(args, "blocktt_rank", "full")
        if isinstance(rank_raw, str):
            if rank_raw != "full":
                try:
                    int(rank_raw)
                    is_int = "." not in rank_raw
                except ValueError:
                    is_int = False
                if is_int:
                    raise ValueError(
                        "integer --blocktt-rank is only valid when --calib-mode=none; "
                        "for calibrated BTT pass 'full' or a float in (0, 1]"
                    )
                # Float rank must be in (0, 1]
                if "." in rank_raw or "e" in rank_raw.lower():
                    try:
                        val = float(rank_raw)
                    except ValueError:
                        val = None
                    if val is not None and not (0.0 < val <= 1.0):
                        raise ValueError(
                            "float --blocktt-rank must be in (0, 1] for calibrated BTT "
                            f"(got {rank_raw!r})"
                        )


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
