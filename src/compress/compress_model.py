"""End-to-end model compression pipeline."""

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch.nn as nn
from loguru import logger
import torch
from compress.aa_btt import aa_btt_decompose_layer
from compress.baselines.svd_als import svd_als_compress_model
from compress.baselines.svd_twosteps import svd_twosteps_compress_model
from compress.baselines.svd_llm_v2 import svd_compress_layer, svd_compress_layer_combined, svd_llm_v2_compress_model
from compress.btt_linear import BTTLinear
from compress.calibration import (
    collect_backward_covariances_from_loader,
    collect_both_covariances_from_loader,
    collect_calibration_covariances,
    collect_covariances_from_loader,
    collect_mixed_statistics,
)
from compress.nystrom import nystrom_compress_model
from compress.whitening import compute_whitening
from compress.utils import _closest_factor_pair

# Method name registry:
#   svd           — plain SVD on raw weight matrix, no calibration data needed
#   svd_llm       — activation-whitened SVD (SVD-LLM V1), uniform rank across layers
#   svd_llm_v2    — activation-whitened SVD (SVD-LLM V2), heterogeneous rank allocation
#   svd_llm_v2_bp — SVD-LLM V2 with backward reconstruction objective (uses Cov(dy))
#   btt           — BTT decomposition on raw weight matrix, no calibration data needed
#   aa_btt        — activation-aware BTT decomposition with whitening
SUPPORTED_METHODS = (
    "svd",
    "svd_llm",
    "svd_llm_v2",
    "svd_llm_v2_bp",
    "svd_llm_v2_combined",
    "svd_als",
    "svd_twosteps",
    "btt",
    "aa_btt",
)
SUPPORTED_METHOD_SET = set(SUPPORTED_METHODS)
# Methods that require calibration data to collect input activation covariances
_CALIB_FREE_METHODS = {"svd", "btt"}
# Methods that collect output gradient covariances instead of input activation covariances
_BACKWARD_CALIB_METHODS = {"svd_llm_v2_bp"}
# Methods that collect both forward and backward covariances
_BOTH_CALIB_METHODS = {"svd_llm_v2_combined", "svd_als", "svd_twosteps"}


@dataclass(frozen=True)
class MethodSpec:
    """Routing spec for mixed attention/MLP compression."""

    attn: Optional[str]
    mlp: Optional[str]


def _is_mlp(name: str) -> bool:
    """Heuristic for identifying MLP submodules by dotted path."""
    keywords = (
        "mlp",
        "ff",
        "feed_forward",
        "ffn",
        "gate_proj",
        "down_proj",
        "up_proj",
    )
    lowered = name.lower()
    return any(keyword in lowered for keyword in keywords)


def _normalize_method_name(method: Optional[str], *, key: str) -> Optional[str]:
    if method is None:
        return None
    if not isinstance(method, str):
        raise ValueError(f"{key} method must be a string or null, got {type(method).__name__}")
    normalized = method.strip().lower()
    if normalized == "none":
        return None
    if normalized not in SUPPORTED_METHOD_SET and normalized != "nystrom":
        raise ValueError(
            f"Unknown {key} method {method!r}, choose from {SUPPORTED_METHODS} or 'nystrom'"
        )
    return normalized


def _parse_method_spec_dict(payload: Dict[str, Any]) -> MethodSpec:
    missing = {key for key in ("attn", "mlp") if key not in payload}
    if missing:
        raise ValueError(
            "Method spec JSON must include both 'attn' and 'mlp' keys"
        )

    attn = _normalize_method_name(payload["attn"], key="attn")
    mlp = _normalize_method_name(payload["mlp"], key="mlp")

    if attn == "nystrom":
        raise ValueError("attn='nystrom' is not supported; nystrom is only valid for mlp")

    for key, val in (("attn", attn), ("mlp", mlp)):
        if val in _BACKWARD_CALIB_METHODS or val in _BOTH_CALIB_METHODS:
            raise ValueError(
                f"{key}={val!r} requires backward calibration and cannot be used in a mixed MethodSpec; "
                "use the plain string method path instead."
            )

    return MethodSpec(attn=attn, mlp=mlp)


def parse_method_spec(method: Any) -> Optional[MethodSpec]:
    """Parse a mixed compression method specification.

    Plain string methods keep the existing single-method path and return None.
    JSON dict strings are converted to MethodSpec.
    """
    if method is None:
        return None
    if isinstance(method, MethodSpec):
        return method
    if isinstance(method, dict):
        return _parse_method_spec_dict(method)
    if not isinstance(method, str):
        raise ValueError(f"Unsupported method type {type(method).__name__}")

    stripped = method.strip()
    if stripped == "nystrom":
        raise ValueError("Bare string 'nystrom' is not allowed; use a JSON dict with attn/mlp keys")
    if not stripped.startswith("{"):
        return None

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON method spec: {method!r}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Method spec JSON must decode to an object with attn/mlp keys")
    return _parse_method_spec_dict(payload)


def _compress_with_covariances(
    model: nn.Module,
    covariances: Dict[str, "torch.Tensor"],
    compression_ratio: float,
    method: str,
    device: str,
    skip_layers: Tuple[str, ...],
) -> nn.Module:
    """Core compression logic shared by both entry points."""
    # Delegate to activation-whitened SVD baselines
    if method in ("svd_llm", "svd_llm_v2"):
        return svd_llm_v2_compress_model(
            model, covariances,
            compression_ratio=compression_ratio,
            skip_layers=skip_layers,
            heterogeneous=(method == "svd_llm_v2"),
            device=device,
            objective="forward",
        )
    if method == "svd_llm_v2_bp":
        return svd_llm_v2_compress_model(
            model, covariances,
            compression_ratio=compression_ratio,
            skip_layers=skip_layers,
            heterogeneous=True,
            device=device,
            objective="backward",
        )

    # Collect layers to compress
    layers_to_compress = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.split(".")[-1]
        if leaf in skip_layers:
            continue
        if method not in _CALIB_FREE_METHODS and name not in covariances:
            logger.warning(f"No covariance for {name}, skipping")
            continue
        layers_to_compress.append((name, module))

    total_orig_params = sum(m.weight.numel() for _, m in layers_to_compress)
    logger.info(f"Compressing {len(layers_to_compress)} layers: "
                f"{total_orig_params} original params")
    if total_orig_params == 0:
        logger.info("No eligible layers found for compression; returning model unchanged.")
        return model

    # Plain SVD: no calibration, no whitening
    if method == "svd":
        for name, module in layers_to_compress:
            d_out, d_in = module.weight.shape
            if compression_ratio >= 1.0:
                r = min(d_out, d_in)
            else:
                r = max(1, int(compression_ratio * d_out * d_in / (d_out + d_in)))
            compressed = svd_compress_layer(
                module.weight.data,
                rank=r,
                bias=module.bias.data if module.bias is not None else None,
                precomputed_whitening=None,
                device=device,
            ).to(device=module.weight.device, dtype=module.weight.dtype)
            path = name.split(".")
            parent = model
            for key in path[:-1]:
                parent = getattr(parent, key)
            setattr(parent, path[-1], compressed)
            logger.info(f"  {name}: ({d_out}, {d_in}) → rank {r}")
        return model

    use_whitening = method == "aa_btt"

    logger.info("Computing per-layer info...")
    layer_info = {}
    for name, module in layers_to_compress:
        d_out, d_in = module.weight.shape
        m, a = _closest_factor_pair(d_out)
        layer_info[name] = {"m": m, "a": a, "d_in": d_in, "d_out": d_out}

    # Budget allocation — uniform rank per layer
    logger.info("Allocating rank budget (uniform per layer)...")
    per_block_ranks = {}
    for name, info in layer_info.items():
        m, a, d_in = info["m"], info["a"], info["d_in"]
        if compression_ratio >= 1.0:
            r = min(a, d_in)  # full-rank: apply transform only, no truncation
        else:
            r = int(compression_ratio * a * d_in / (d_in + a))
            r = max(1, min(r, min(a, d_in)))
        per_block_ranks[name] = [r] * m

    # Decompose and replace each layer: compute whitening lazily and discard immediately
    logger.info("Decomposing layers...")
    total_compressed_params = 0
    for name, module in layers_to_compress:
        info = layer_info[name]
        ranks = per_block_ranks[name]

        whitening = compute_whitening(covariances[name], device=device) if use_whitening else None
        result = aa_btt_decompose_layer(
            module.weight.data,
            rank=max(ranks),
            bias=module.bias.data if module.bias is not None else None,
            per_block_ranks=ranks,
            precomputed_whitening=whitening,
            device=device,
        )
        del whitening

        # Create inference module
        compressed = BTTLinear(
            result["btt_l"],
            result["btt_r"],
            bias=result["bias"],
            inv_permutation=result["inv_permutation"],
            m=result["m"],
            a=result["a"],
        ).to(device=module.weight.device, dtype=module.weight.dtype)

        # Replace in model
        path = name.split(".")
        parent = model
        for key in path[:-1]:
            parent = getattr(parent, key)
        setattr(parent, path[-1], compressed)

        layer_params = sum(r * (info["d_in"] + info["a"]) for r in ranks)
        total_compressed_params += layer_params

    logger.info(
        f"Compression complete: {total_compressed_params}/{total_orig_params} params "
        f"({total_compressed_params/total_orig_params:.1%})"
    )
    return model


def _compress_with_covariances_combined(
    model: nn.Module,
    fwd_covariances: Dict[str, "torch.Tensor"],
    bwd_covariances: Dict[str, "torch.Tensor"],
    compression_ratio: float,
    method: str,
    device: str,
    skip_layers: Tuple[str, ...],
    als_n_iter: int = 10,
    als_tol: float = 1e-6,
    als_weighting: str = "equal",
    als_reg_eps: float = 1e-4,
    twosteps_n_refine: int = 1,
    twosteps_reg_eps: float = 1e-4,
) -> nn.Module:
    """Core compression logic for methods needing both forward and backward covariances."""
    if method == "svd_als":
        return svd_als_compress_model(
            model, fwd_covariances, bwd_covariances,
            compression_ratio=compression_ratio,
            skip_layers=skip_layers,
            device=device,
            als_n_iter=als_n_iter,
            als_tol=als_tol,
            als_weighting=als_weighting,
            als_reg_eps=als_reg_eps,
        )
    if method == "svd_twosteps":
        return svd_twosteps_compress_model(
            model, fwd_covariances, bwd_covariances,
            compression_ratio=compression_ratio,
            skip_layers=skip_layers,
            device=device,
            n_refine=twosteps_n_refine,
            reg_eps=twosteps_reg_eps,
        )
    return svd_llm_v2_compress_model(
        model, fwd_covariances,
        compression_ratio=compression_ratio,
        skip_layers=skip_layers,
        device=device,
        objective="combined",
        backward_covariances=bwd_covariances,
    )


def _compress_with_method_spec(
    model: nn.Module,
    statistics: Dict[str, "torch.Tensor"],
    compression_ratio: float,
    method_spec: MethodSpec,
    device: str,
    skip_layers: Tuple[str, ...],
) -> nn.Module:
    """Route mixed attention/MLP compression according to a MethodSpec."""
    mlp_statistics = {
        name: stat
        for name, stat in statistics.items()
        if _is_mlp(name) and (method_spec.mlp != "nystrom" or name.endswith("down_proj"))
    }
    attn_statistics = {
        name: stat
        for name, stat in statistics.items()
        if not _is_mlp(name)
    }

    if method_spec.mlp == "nystrom":
        logger.info(
            f"Compressing MLP with nystrom, ratio={compression_ratio:.0%}"
        )
        model = nystrom_compress_model(
            model,
            mlp_statistics,
            sparsity=1.0 - compression_ratio,
            skip_layers=skip_layers,
            device=device,
        )
    elif method_spec.mlp in SUPPORTED_METHOD_SET:
        logger.info(
            f"Compressing MLP with method={method_spec.mlp}, ratio={compression_ratio:.0%}"
        )
        model = _compress_with_covariances(
            model,
            mlp_statistics,
            compression_ratio,
            method_spec.mlp,
            device,
            skip_layers,
        )

    if method_spec.attn in SUPPORTED_METHOD_SET:
        logger.info(
            f"Compressing attention with method={method_spec.attn}, ratio={compression_ratio:.0%}"
        )
        model = _compress_with_covariances(
            model,
            attn_statistics,
            compression_ratio,
            method_spec.attn,
            device,
            skip_layers,
        )

    return model


def compress_model(
    model: nn.Module,
    tokenizer,
    compression_ratio: float = 0.7,
    method: str = "aa_btt",
    n_calib_samples: int = 128,
    seq_len: int = 2048,
    batch_size: int = 4,
    device: str = "cuda",
    skip_layers: Tuple[str, ...] = ("lm_head",),
) -> nn.Module:
    """Compress a pretrained model using C4 calibration data.

    Args:
        model: pretrained HuggingFace model
        tokenizer: model tokenizer
        compression_ratio: fraction of params to retain (0.5 = 50%)
        method: "svd", "svd_llm", "svd_llm_v2", "btt", "aa_btt"
            (not "svd_llm_v2_bp"/"svd_llm_v2_combined"/"svd_als" —
             those require backward calibration; use compress_model_with_loader())
        n_calib_samples: calibration samples
        seq_len: calibration sequence length
        batch_size: calibration batch size
        device: compute device
        skip_layers: layers to skip

    Returns:
        Compressed model (in-place)
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown method {method!r}, choose from {SUPPORTED_METHODS}")

    if method in _BACKWARD_CALIB_METHODS or method in _BOTH_CALIB_METHODS:
        raise ValueError(
            f"Method {method!r} requires backward calibration; use compress_model_with_loader() instead."
        )

    logger.info(f"Compressing with method={method}, ratio={compression_ratio:.0%}")

    logger.info("Collecting calibration covariances from C4...")
    covariances = collect_calibration_covariances(
        model, tokenizer,
        n_samples=n_calib_samples,
        seq_len=seq_len,
        batch_size=batch_size,
        device=device,
    )

    return _compress_with_covariances(
        model, covariances, compression_ratio, method, device, skip_layers,
    )


def compress_model_with_loader(
    model: nn.Module,
    calib_loader,
    compression_ratio: float = 0.7,
    method: str = "aa_btt",
    device: str = "cuda",
    skip_layers: Tuple[str, ...] = ("lm_head",),
    als_n_iter: int = 10,
    als_tol: float = 1e-6,
    als_weighting: str = "equal",
    als_reg_eps: float = 1e-4,
    twosteps_n_refine: int = 1,
    twosteps_reg_eps: float = 1e-4,
) -> nn.Module:
    """Compress a model using activations collected from an external DataLoader.

    Same decomposition logic as compress_model(), but uses a DataLoader
    (e.g. from make_calib_loader with trace data) instead of C4.

    Args:
        model: pretrained HuggingFace model
        calib_loader: DataLoader yielding {input_ids, attention_mask, ...}
        compression_ratio: fraction of params to retain (0.5 = 50%)
        method: "svd", "svd_llm", "svd_llm_v2", "svd_llm_v2_bp",
            "svd_llm_v2_combined", "svd_als", "btt", "aa_btt"
        device: compute device
        skip_layers: layers to skip

    Returns:
        Compressed model (in-place)
    """
    method_spec = parse_method_spec(method)
    if method_spec is None:
        if not isinstance(method, str):
            raise ValueError(
                "method must be a string when using the plain-method path; "
                f"got {type(method).__name__}"
            )
        normalized_method = method.strip().lower()
        if normalized_method not in SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown method {method!r}, choose from {SUPPORTED_METHODS}"
            )

        logger.info(
            f"Compressing with method={normalized_method}, ratio={compression_ratio:.0%}"
        )
        if normalized_method in _CALIB_FREE_METHODS:
            logger.info("Calibration-free method — skipping DataLoader pass.")
            covariances = {}
        elif normalized_method in _BACKWARD_CALIB_METHODS:
            logger.info("Collecting backward (gradient) covariances from DataLoader...")
            covariances = collect_backward_covariances_from_loader(
                model, calib_loader, device=device, skip_layers=skip_layers,
            )
        elif normalized_method in _BOTH_CALIB_METHODS:
            logger.info("Collecting forward and backward covariances from DataLoader...")
            fwd_covariances, bwd_covariances = collect_both_covariances_from_loader(
                model, calib_loader, device=device, skip_layers=skip_layers,
            )
            return _compress_with_covariances_combined(
                model, fwd_covariances, bwd_covariances,
                compression_ratio, normalized_method, device, skip_layers,
                als_n_iter=als_n_iter,
                als_tol=als_tol,
                als_weighting=als_weighting,
                als_reg_eps=als_reg_eps,
                twosteps_n_refine=twosteps_n_refine,
                twosteps_reg_eps=twosteps_reg_eps,
            )
        else:
            logger.info("Collecting calibration covariances from DataLoader...")
            covariances = collect_covariances_from_loader(
                model, calib_loader, device=device, skip_layers=skip_layers,
            )
        return _compress_with_covariances(
            model, covariances, compression_ratio, normalized_method, device, skip_layers,
        )

    logger.info(
        "Compressing with mixed method spec "
        f"(attn={method_spec.attn}, mlp={method_spec.mlp}), "
        f"ratio={compression_ratio:.0%}"
    )
    logger.info("Collecting mixed calibration statistics from DataLoader...")
    statistics = collect_mixed_statistics(
        model,
        calib_loader,
        method_spec,
        device=device,
        skip_layers=skip_layers,
        valid_methods=SUPPORTED_METHOD_SET,
    )
    return _compress_with_method_spec(
        model, statistics, compression_ratio, method_spec, device, skip_layers,
    )
