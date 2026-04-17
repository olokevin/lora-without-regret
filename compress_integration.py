"""Shared helpers for integrating src/compress calibrated BTT with the
run_sft / run_rl / run_rl_dapo / LIFT training entrypoints.

Keeps legacy btt_layer.py / svd_layer.py paths untouched; activated only
when --calib-mode (or --calib_mode) is non-'none'.
"""

from __future__ import annotations

import json
import math
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

    # 3. Integer --blocktt-rank rejected on calibrated path; float must be in (0, 1]
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


_BLOCKTT_TARGET_NAMES = {
    "all": ("gate_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj"),
    "mlp": ("gate_proj", "up_proj", "down_proj"),
    "attn": ("q_proj", "k_proj", "v_proj", "o_proj"),
}


def _resolve_ratio_from_rank(rank_raw) -> float:
    """--blocktt-rank on the calibrated path: 'full' -> 1.0; float in (0, 1] -> itself.

    Integer strings like '4' are rejected: on the calibrated path, rank is a
    compression ratio, not a fixed rank. `validate_calibrated_btt_args` also
    catches this, but we re-enforce here so direct callers (tests, LIFT) get
    a clear error.
    """
    if isinstance(rank_raw, bool):
        raise ValueError(f"--blocktt-rank must be 'full' or a float; got bool {rank_raw!r}")
    if isinstance(rank_raw, str):
        if rank_raw == "full":
            return 1.0
        # Reject integer-string form ('4', '+4', '-1') before the float parse.
        if "." not in rank_raw and "e" not in rank_raw.lower():
            stripped = rank_raw.lstrip("+-")
            if stripped.isdigit():
                raise ValueError(
                    "integer --blocktt-rank is only valid when --calib-mode=none; "
                    f"for calibrated BTT pass 'full' or a float in (0, 1] (got {rank_raw!r})"
                )
        try:
            val = float(rank_raw)
        except ValueError as exc:
            raise ValueError(f"--blocktt-rank must be 'full' or a float; got {rank_raw!r}") from exc
    elif isinstance(rank_raw, (int, float)):
        val = float(rank_raw)
    else:
        raise ValueError(f"--blocktt-rank has unsupported type {type(rank_raw).__name__}")
    if not math.isfinite(val):
        raise ValueError(f"--blocktt-rank must be finite; got {val}")
    if not (0.0 < val <= 1.0):
        raise ValueError(f"--blocktt-rank float must be in (0, 1]; got {val}")
    return val


def _build_skip_layers(model: nn.Module, trainable_type: str) -> str:
    """Return comma-separated leaf names to skip, i.e. every nn.Linear leaf whose
    name is NOT in the trainable_type target set. 'lm_head' is always included."""
    targets = set(_BLOCKTT_TARGET_NAMES.get(trainable_type, _BLOCKTT_TARGET_NAMES["all"]))
    skip = {"lm_head"}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in targets:
            skip.add(leaf)
    return ",".join(sorted(skip))


def build_decomposition_config(args, *, hyphen_style: bool = True, model=None) -> DecompositionConfig:
    """Translate CLI args to a DecompositionConfig. `model` is required so that
    --trainable-type can be inverted into a skip_layers list."""
    del hyphen_style  # kept for signature stability; routed via _arg_name upstream
    if model is None:
        raise ValueError("build_decomposition_config requires model= for skip_layers inversion")

    calib_mode = getattr(args, "calib_mode", None)
    if calib_mode is None:
        raise ValueError(
            "build_decomposition_config requires args.calib_mode; did you call "
            "add_calibrated_btt_args() on the parser?"
        )
    if calib_mode not in CALIB_MODE_TO_TRAIN_MODE:
        raise ValueError(
            f"calib_mode must be one of {sorted(CALIB_MODE_TO_TRAIN_MODE.keys())}; "
            f"got {calib_mode!r}"
        )
    train_mode = CALIB_MODE_TO_TRAIN_MODE[calib_mode]
    ratio = _resolve_ratio_from_rank(getattr(args, "blocktt_rank", "full"))
    trainable_type = getattr(args, "trainable_type", "all")
    skip_layers = _build_skip_layers(model, trainable_type)

    return DecompositionConfig(
        train_mode=train_mode,
        compression_ratio=ratio,
        calib_source=getattr(args, "calib_source", "c4"),
        calib_traces_path=getattr(args, "calib_traces_path", None),
        calib_num_seqs=getattr(args, "calib_num_seqs", 128),
        calib_max_length=getattr(args, "calib_max_length", 2048),
        calib_seed=getattr(args, "calib_seed", 3),
        skip_layers=skip_layers,
        decomp_mode=getattr(args, "decomp_mode", "square"),
        train_position=getattr(args, "train_position", "both"),
        s_merged_to=getattr(args, "s_merged_to", None),
        factorize_by_head=bool(getattr(args, "blocktt_factorize_by_head", True)),
    )


def build_training_data_calib_loader(
    dataset, collate_fn, *, num_seqs: int, batch_size: int, seed: int,
) -> DataLoader:
    """Take a deterministic subset of the training dataset and wrap it in a
    DataLoader using the training collate function. Yields the same batch shape
    the training loop sees."""
    import random
    n_available = len(dataset)
    n_take = min(int(num_seqs), n_available)
    rng = random.Random(int(seed))
    indices = list(range(n_available))
    rng.shuffle(indices)
    subset = Subset(dataset, indices[:n_take])
    return DataLoader(
        subset,
        batch_size=int(batch_size),
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )


def build_rl_rollout_calib_loader(
    *, rl_rollout_fn: Callable[[int], List[Tuple[str, str]]],
    tokenizer, num_seqs: int, batch_size: int, max_length: int, seed: int,
) -> DataLoader:
    """Run a caller-supplied rollout function to produce (prompt, completion)
    text pairs; tokenize each pair; build a DataLoader of
    {input_ids, attention_mask, labels} where prompt tokens are masked to -100
    in labels (mirroring the RL loss mask)."""
    pairs = rl_rollout_fn(int(num_seqs))
    if len(pairs) == 0:
        raise ValueError("rl_rollout_fn returned no (prompt, completion) pairs")

    examples = []
    for prompt, completion in pairs[:int(num_seqs)]:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        completion_ids = tokenizer.encode(completion, add_special_tokens=False)
        full = (prompt_ids + completion_ids)[: int(max_length)]
        prompt_len = min(len(prompt_ids), len(full))
        labels = [-100] * prompt_len + full[prompt_len:]
        # If prompt was truncated away entirely, fall back to masking half.
        if all(l == -100 for l in labels):
            labels = full.copy()
        examples.append({
            "input_ids": torch.tensor(full, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        })

    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0

    def _collate(batch):
        max_len = max(e["input_ids"].shape[0] for e in batch)
        input_ids = torch.full((len(batch), max_len), fill_value=pad_id, dtype=torch.long)
        attn = torch.zeros((len(batch), max_len), dtype=torch.long)
        labels = torch.full((len(batch), max_len), fill_value=-100, dtype=torch.long)
        for i, e in enumerate(batch):
            L = e["input_ids"].shape[0]
            input_ids[i, :L] = e["input_ids"]
            attn[i, :L] = 1
            labels[i, :L] = e["labels"]
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

    return DataLoader(
        examples, batch_size=int(batch_size), shuffle=False, collate_fn=_collate,
    )


def build_calib_loader(
    args, *, tokenizer, training_dataset=None, training_collate_fn=None,
    rl_rollout_fn=None, hyphen_style: bool = True,
) -> Optional[DataLoader]:
    calib_mode = getattr(args, "calib_mode", "none")
    if calib_mode == "none":
        return None

    source = getattr(args, "calib_source", "c4")
    num_seqs = int(getattr(args, "calib_num_seqs", 128))
    max_length = int(getattr(args, "calib_max_length", 2048))
    seed = int(getattr(args, "calib_seed", 3))
    batch_size = int(getattr(args, "calib_batch_size", 8))

    if source == "c4":
        return build_c4_calib_loader(
            tokenizer, num_seqs=num_seqs, max_length=max_length,
            batch_size=batch_size, seed=seed,
        )
    if source == "traces":
        return build_traces_jsonl_calib_loader(
            tokenizer,
            jsonl_path=getattr(args, "calib_traces_path"),
            num_seqs=num_seqs, max_length=max_length, batch_size=batch_size,
        )
    if source == "training_data":
        if training_dataset is not None and training_collate_fn is not None:
            return build_training_data_calib_loader(
                training_dataset, training_collate_fn,
                num_seqs=num_seqs, batch_size=batch_size, seed=seed,
            )
        if rl_rollout_fn is not None:
            return build_rl_rollout_calib_loader(
                rl_rollout_fn=rl_rollout_fn, tokenizer=tokenizer,
                num_seqs=num_seqs, batch_size=batch_size,
                max_length=max_length, seed=seed,
            )
        raise ValueError(
            "--calib-source=training_data requires either "
            "(training_dataset, training_collate_fn) or rl_rollout_fn"
        )
    raise ValueError(f"Unknown --calib-source {source!r}")


def apply_calibrated_btt(
    model, args, *, calib_loader, device: Optional[str] = None,
    hyphen_style: bool = True,
) -> Tuple[nn.Module, dict]:
    cfg = build_decomposition_config(args, hyphen_style=hyphen_style, model=model)
    model, stats = decompose_with_loader(
        model, cfg, calib_loader=calib_loader, device=device,
        return_trainability_stats=True,
    )
    if stats is None or stats.get("num_btt_layers", 0) == 0:
        raise ValueError("No BTT layers were installed; check --trainable-type selection.")
    return model, stats


@torch.no_grad()
def materialize_calibrated_btt_weights(model) -> List[Tuple[str, torch.Tensor]]:
    """Return [(param_name, dense_tensor)] for every BTTLinear in the model.

    For each BTTLinear `M` at path `p`, yields (`p.weight`, dense) and
    optionally (`p.bias`, bias). This list is appended to any other
    (name, tensor) pairs the caller assembles for vLLM weight sync.
    """
    out: List[Tuple[str, torch.Tensor]] = []
    for name, module in model.named_modules():
        if not isinstance(module, BTTLinear):
            continue
        out.append((f"{name}.weight", module.materialize_dense_weight()))
        if module.bias is not None:
            out.append((f"{name}.bias", module.bias.detach()))
    return out


def restore_calibrated_btt_weights(model, saved_state) -> None:
    """No-op: BTTLinear weights are never overwritten by vLLM weight export,
    only materialized-and-copied. The real factored cores remain in place
    on the training model, so no restore is needed. Provided for API
    symmetry with the legacy SVDLayer/BTTLayer restore flow."""
    return None


@torch.no_grad()
def materialize_calibrated_btt_to_linear(model: nn.Module) -> nn.Module:
    """In-place: replace every BTTLinear in `model` with an nn.Linear whose
    weight is the materialized dense equivalent. Mirrors the legacy
    `materialize_btt_to_linear` in ref/LIFT/src/finetune_blocktt.py but for
    the `compress` package's BTTLinear."""
    replacements = [(n, m) for n, m in model.named_modules() if isinstance(m, BTTLinear)]
    for name, btt in replacements:
        dense = btt.materialize_dense_weight()
        linear = nn.Linear(
            btt.in_features,
            btt.out_features,
            bias=btt.bias is not None,
            device=dense.device,
            dtype=dense.dtype,
        )
        linear.weight.data.copy_(dense)
        if btt.bias is not None:
            linear.bias.data.copy_(btt.bias.data)
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], linear)
    return model


def save_calibrated_btt_checkpoint(model, out_dir: str, tokenizer=None) -> None:
    """LIFT legacy HF format: materialize BTTLinear -> nn.Linear, then write
    exactly `pytorch_model.bin` + `config.json`. Byte-for-byte identical
    to `ref/LIFT/src/utils/model_utils.py:save_hf_format`, so a converted
    calib checkpoint is indistinguishable from a non-calib blocktt one.

    `tokenizer` is accepted for signature symmetry with non-calib callers
    but intentionally not written — `save_hf_format` does not save the
    tokenizer either, and LIFT eval loads it from `base_model`.
    """
    del tokenizer  # unused; signature kept for symmetry
    os.makedirs(out_dir, exist_ok=True)
    materialize_calibrated_btt_to_linear(model)
    torch.save(model.state_dict(), os.path.join(out_dir, "pytorch_model.bin"))
    model.config.to_json_file(os.path.join(out_dir, "config.json"))


def save_calibrated_btt_hf_pretrained(model, out_dir: str) -> None:
    """run_sft / run_rl / run_rl_dapo format: materialize BTTLinear ->
    nn.Linear, then `model.save_pretrained(out_dir)`. Byte-for-byte
    identical to the non-calib branch of those scripts, which also just
    call `model.save_pretrained(ckpt_dir)`.

    Does NOT save the tokenizer; callers in run_* already invoke
    `tokenizer.save_pretrained(ckpt_dir)` once, outside the calib/non-calib
    branch, so both branches produce the same on-disk file set.
    """
    os.makedirs(out_dir, exist_ok=True)
    materialize_calibrated_btt_to_linear(model)
    model.save_pretrained(out_dir)


def load_calibrated_btt_for_eval(model, checkpoint_dir: str) -> nn.Module:
    """Read btt_topology.json, rebuild BTTLinear modules in `model`, load
    model.safetensors. Returns the mutated model. No calibration pass."""
    from safetensors.torch import load_file

    topology_path = os.path.join(checkpoint_dir, "btt_topology.json")
    with open(topology_path) as f:
        topology = json.load(f)

    rebuild_btt_from_topology(model, topology)

    state = load_file(os.path.join(checkpoint_dir, "model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected keys in checkpoint: {unexpected[:5]}")
    # 'missing' is OK: topology only rebuilt BTT paths; other params may have
    # been loaded from state directly.
    return model
