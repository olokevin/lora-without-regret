"""Collect calibration covariance matrices from linear layer activations."""

import torch
import torch.nn as nn
from datasets import load_dataset
from loguru import logger
from typing import Any, Dict, Optional, Tuple


def _collect_c4_batches(tokenizer, n_samples=128, seq_len=2048, seed=42):
    """Stream C4 dataset and return tokenized batches."""
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    dataset = dataset.shuffle(seed=seed)

    all_input_ids = []
    for sample in dataset:
        tokens = tokenizer(
            sample["text"],
            return_tensors="pt",
            max_length=seq_len,
            truncation=True,
            padding=False,
        )
        if tokens["input_ids"].shape[1] >= seq_len:
            all_input_ids.append(tokens["input_ids"][:, :seq_len])
        if len(all_input_ids) >= n_samples:
            break

    logger.info(f"Collected {len(all_input_ids)} calibration samples of length {seq_len}")
    return torch.cat(all_input_ids, dim=0)  # (n_samples, seq_len)


def _run_forward_accumulate_cov(
    model: nn.Module,
    linear_layers: Dict[str, nn.Module],
    batches,  # iterable of (input_ids, attention_mask_or_None)
    device: str,
    log_interval: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    """Run forward passes and accumulate per-layer covariance matrices on CPU.

    For each layer accumulates C += x^T x (shape d_in x d_in, float64 on CPU)
    and a token count.  Never stores the full (d_in, N) activation matrix.
    """
    cov_store: Dict[str, torch.Tensor] = {}
    token_counts: Dict[str, int] = {}
    for name, module in linear_layers.items():
        d_in = module.weight.shape[1]
        cov_store[name] = torch.zeros(d_in, d_in, dtype=torch.float32, device="cpu")
        token_counts[name] = 0

    def _make_hook(name):
        def _hook(module, inputs, output):
            x = inputs[0].detach()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])  # (B*T, d_in)
            x = x.to(dtype=torch.float32)
            cov_store[name] += (x.t() @ x).to(device="cpu")
            token_counts[name] += x.shape[0]
            del x
        return _hook

    handles = [m.register_forward_hook(_make_hook(n)) for n, m in linear_layers.items()]
    model.eval()
    try:
        with torch.no_grad():
            for idx, (input_ids, attention_mask) in enumerate(batches):
                input_ids = input_ids.to(device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                if (idx + 1) % log_interval == 0:
                    logger.info(f"  Processed {idx + 1} batches")
    finally:
        for h in handles:
            h.remove()

    return cov_store, token_counts


def collect_calibration_covariances(
    model: nn.Module,
    tokenizer,
    n_samples: int = 128,
    seq_len: int = 2048,
    batch_size: int = 4,
    device: str = "cuda",
    seed: int = 42,
    skip_layers: tuple = ("lm_head",),
) -> Dict[str, torch.Tensor]:
    """Collect input covariance matrices for all nn.Linear layers using C4 data.

    Returns:
        Dict mapping "layer_name" -> (d_in, d_in) float32 covariance matrix (on device).
        Covariance is (1/N) * X^T X where X is (N, d_in) activation matrix.
    """
    input_ids_all = _collect_c4_batches(tokenizer, n_samples, seq_len, seed)
    logger.info(f"Calibration data shape: {input_ids_all.shape}")

    linear_layers = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name.split(".")[-1] not in skip_layers
    }
    logger.info(f"Hooking {len(linear_layers)} linear layers for covariance accumulation")

    def _batches():
        for i in range(0, input_ids_all.shape[0], batch_size):
            yield input_ids_all[i: i + batch_size], None

    cov_store, token_counts = _run_forward_accumulate_cov(
        model, linear_layers, _batches(), device, log_interval=10
    )

    covariances = {}
    for name, C in cov_store.items():
        N = token_counts[name]
        if N == 0:
            continue
        covariances[name] = (C / N).to(device=device)
        logger.info(f"  {name}: cov shape {covariances[name].shape}, N={N}")

    return covariances


def collect_covariances_from_loader(
    model: nn.Module,
    calib_loader,
    device: str = "cuda",
    skip_layers: tuple = ("lm_head",),
) -> Dict[str, torch.Tensor]:
    """Collect input covariance matrices for all nn.Linear layers from a DataLoader.

    Args:
        model: pretrained HuggingFace model
        calib_loader: DataLoader yielding dicts with 'input_ids' and 'attention_mask'
        device: compute device
        skip_layers: leaf layer names to skip

    Returns:
        Dict mapping "layer_name" -> (d_in, d_in) float32 covariance matrix (on device).
        Covariance is (1/N) * X^T X where X is (N, d_in) activation matrix.
    """
    linear_layers = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name.split(".")[-1] not in skip_layers
    }
    logger.info(f"Hooking {len(linear_layers)} linear layers for covariance accumulation")

    def _batches():
        for batch in calib_loader:
            yield batch["input_ids"], batch.get("attention_mask")

    cov_store, token_counts = _run_forward_accumulate_cov(
        model, linear_layers, _batches(), device, log_interval=10
    )

    covariances = {}
    for name, C in cov_store.items():
        N = token_counts[name]
        if N == 0:
            continue
        covariances[name] = (C / N).to(device=device)
        logger.info(f"  {name}: cov shape {covariances[name].shape}, N={N}")

    return covariances


def collect_mixed_statistics(
    model: nn.Module,
    calib_loader,
    method_spec: Any,
    device: str = "cuda",
    skip_layers: tuple = ("lm_head",),
    valid_methods: Optional[set] = None,
) -> Dict[str, torch.Tensor]:
    """Collect layerwise statistics for mixed attention/MLP compression.

    The hooked layers depend on the routing spec:
    - mlp=nystrom: only down_proj inputs are collected
    - mlp in existing methods: all MLP linear inputs are collected
    - attn in existing methods: all non-MLP linear inputs are collected
    - None routes are skipped
    """
    valid_methods = (
        set(valid_methods)
        if valid_methods is not None
        else {"svd", "svd_llm_v2", "vanilla_btt", "aa_btt"}
    )
    mlp_method = getattr(method_spec, "mlp", None)
    attn_method = getattr(method_spec, "attn", None)

    def _is_mlp(name: str) -> bool:
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

    linear_layers = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name.split(".")[-1] in skip_layers:
            continue

        is_mlp = _is_mlp(name)
        if is_mlp:
            if mlp_method == "nystrom":
                if not name.endswith("down_proj"):
                    continue
            elif mlp_method in valid_methods:
                pass
            else:
                continue
        else:
            if attn_method not in valid_methods:
                continue

        linear_layers[name] = module

    logger.info(
        f"Hooking {len(linear_layers)} linear layers for mixed covariance accumulation"
    )

    cov_store: Dict[str, torch.Tensor] = {}
    token_counts: Dict[str, int] = {}
    for name, module in linear_layers.items():
        d_in = module.weight.shape[1]
        cov_store[name] = torch.zeros(d_in, d_in, dtype=torch.float32, device="cpu")
        token_counts[name] = 0

    def _make_hook(name):
        def _hook(module, inputs, output):
            x = inputs[0].detach()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            x = x.to(dtype=torch.float32)
            cov_store[name] += (x.t() @ x).to(device="cpu")
            token_counts[name] += x.shape[0]

        return _hook

    handles = [m.register_forward_hook(_make_hook(n)) for n, m in linear_layers.items()]
    model.eval()
    try:
        with torch.no_grad():
            for idx, batch in enumerate(calib_loader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                if (idx + 1) % 10 == 0:
                    logger.info(f"  Processed {idx + 1} batches")
    finally:
        for handle in handles:
            handle.remove()

    statistics = {}
    for name, C in cov_store.items():
        N = token_counts[name]
        if N == 0:
            continue
        statistics[name] = (C / N).to(device=device)
        logger.info(f"  {name}: mixed stat shape {statistics[name].shape}, N={N}")

    return statistics
