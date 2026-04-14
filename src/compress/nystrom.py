"""Nystrom approximation for MLP compression (MoDeGPT Algorithm 1)."""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from loguru import logger


def find_mlp_triplets(
    model: nn.Module,
    skip_layers: Tuple[str, ...],
) -> List[Tuple[str, str, str, str]]:
    """Find gated MLP triplets with gate_proj, up_proj, and down_proj."""
    triplets: List[Tuple[str, str, str, str]] = []

    for parent_path, module in model.named_modules():
        gate = getattr(module, "gate_proj", None)
        up = getattr(module, "up_proj", None)
        down = getattr(module, "down_proj", None)
        if not all(isinstance(layer, nn.Linear) for layer in (gate, up, down)):
            continue

        gate_name = f"{parent_path}.gate_proj" if parent_path else "gate_proj"
        up_name = f"{parent_path}.up_proj" if parent_path else "up_proj"
        down_name = f"{parent_path}.down_proj" if parent_path else "down_proj"
        if any(name.split(".")[-1] in skip_layers for name in (gate_name, up_name, down_name)):
            continue
        triplets.append((parent_path, gate_name, up_name, down_name))

    if not triplets:
        raise NotImplementedError(
            "No gated MLP triplets (gate_proj/up_proj/down_proj) found in model."
        )
    return triplets


def _make_linear(weight: torch.Tensor, bias: Optional[torch.Tensor]) -> nn.Linear:
    out_features, in_features = weight.shape
    layer = nn.Linear(in_features, out_features, bias=bias is not None)
    layer.weight = nn.Parameter(weight.detach().clone(), requires_grad=False)
    if bias is not None:
        layer.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)
    return layer


def nystrom_compress_mlp(
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
    C_sigma: torch.Tensor,
    sparsity: float,
    lambda_ridge: float = 1.0,
    device: str = "cuda",
    bias_gate: Optional[torch.Tensor] = None,
    bias_up: Optional[torch.Tensor] = None,
    bias_down: Optional[torch.Tensor] = None,
) -> Tuple[nn.Linear, nn.Linear, nn.Linear]:
    """Compress one gated MLP block via Nystrom approximation."""
    if not 0.0 <= sparsity < 1.0:
        raise ValueError("sparsity must be in [0, 1)")
    if lambda_ridge <= 0.0:
        raise ValueError("lambda_ridge must be positive")

    dint, dh = W_gate.shape
    if W_up.shape != (dint, dh) or W_down.shape != (dh, dint):
        raise ValueError("W_gate, W_up, and W_down shapes do not match a gated MLP")
    if C_sigma.ndim != 2 or C_sigma.shape != (dint, dint):
        raise ValueError(
            f"C_sigma must have shape (dint, dint) == ({dint}, {dint}), "
            f"got {tuple(C_sigma.shape)}"
        )

    k = math.ceil((1.0 - sparsity) * dint)
    k = max(1, min(k, dint))

    compute_device = torch.device(device)
    target_device = W_gate.device
    orig_dtype = W_gate.dtype

    # Symmetrize covariance in float32 on compute device
    C = C_sigma.to(device=compute_device, dtype=torch.float32)
    C = 0.5 * (C + C.T)

    # Ridge matrix via in-place diagonal update — avoids allocating a (dint,dint) identity
    ridge_mat = C.clone()
    ridge_mat.diagonal().add_(lambda_ridge)

    # Score each neuron: diag of ridge_mat^{-1} @ C; solve column-wise is equivalent
    # torch.linalg.solve(A, B) where B==C gives (dint,dint) but we only need the diagonal.
    # Solve A x = c_i for each column c_i simultaneously, then extract diagonal.
    scores = torch.linalg.solve(ridge_mat, C).diagonal()
    del ridge_mat
    idx = torch.topk(scores, k=k, largest=True).indices.sort().values
    del scores

    C_sub = C.index_select(0, idx).index_select(1, idx)
    rhs = C.index_select(0, idx) @ W_down.to(device=compute_device, dtype=torch.float32).T
    del C

    sub_scale = C_sub.diagonal().abs().mean().clamp_min(1.0)
    sub_ridge = (
        max(lambda_ridge, 1.0)
        * torch.finfo(C_sub.dtype).eps
        * sub_scale
    )
    # In-place diagonal ridge — avoids allocating a (k,k) identity matrix
    W_down_new = None
    for _ in range(6):
        C_sub_reg = C_sub.clone()
        C_sub_reg.diagonal().add_(sub_ridge)
        chol, info = torch.linalg.cholesky_ex(C_sub_reg)
        del C_sub_reg
        if int(info.item()) == 0:
            W_down_new = torch.cholesky_solve(rhs, chol).T
            break
        sub_ridge *= 10.0
    del chol
    if W_down_new is None:
        C_sub_reg = C_sub.clone()
        C_sub_reg.diagonal().add_(sub_ridge)
        W_down_new = torch.linalg.solve(C_sub_reg, rhs).T
        del C_sub_reg
    del C_sub, rhs

    W_gate_new = W_gate.to(device=compute_device, dtype=torch.float32).index_select(0, idx)
    W_up_new = W_up.to(device=compute_device, dtype=torch.float32).index_select(0, idx)

    W_gate_new = W_gate_new.to(device=target_device, dtype=orig_dtype)
    W_up_new = W_up_new.to(device=target_device, dtype=orig_dtype)
    W_down_new = W_down_new.to(device=target_device, dtype=orig_dtype)

    gate_bias = (
        bias_gate.to(device=compute_device, dtype=torch.float32).index_select(0, idx).to(
            device=target_device, dtype=orig_dtype
        )
        if bias_gate is not None
        else None
    )
    up_bias = (
        bias_up.to(device=compute_device, dtype=torch.float32).index_select(0, idx).to(
            device=target_device, dtype=orig_dtype
        )
        if bias_up is not None
        else None
    )
    down_bias = (
        bias_down.to(device=target_device, dtype=orig_dtype)
        if bias_down is not None
        else None
    )
    del idx
    gate_linear = _make_linear(W_gate_new, gate_bias)
    up_linear = _make_linear(W_up_new, up_bias)
    down_linear = _make_linear(W_down_new, down_bias)
    return gate_linear, up_linear, down_linear


def nystrom_compress_model(
    model: nn.Module,
    statistics: Dict[str, torch.Tensor],
    sparsity: float,
    skip_layers: Tuple[str, ...] = ("lm_head",),
    device: str = "cuda",
    lambda_ridge: float = 1.0,
) -> nn.Module:
    """Compress all gated MLP triplets in-place using Nystrom approximation."""
    triplets = find_mlp_triplets(model, skip_layers)
    modules = dict(model.named_modules())

    total_orig = 0
    total_comp = 0
    for parent_path, _, _, down_name in triplets:
        if down_name not in statistics:
            logger.warning(f"No C_sigma statistics for {down_name}, skipping triplet")
            continue

        parent = modules[parent_path]
        gate_mod = parent.gate_proj
        up_mod = parent.up_proj
        down_mod = parent.down_proj

        # Load covariance from CPU to GPU just-in-time, then free both copies
        c_sigma_gpu = statistics[down_name].to(device=device)
        del statistics[down_name]

        gate_new, up_new, down_new = nystrom_compress_mlp(
            gate_mod.weight.data,
            up_mod.weight.data,
            down_mod.weight.data,
            c_sigma_gpu,
            sparsity=sparsity,
            lambda_ridge=lambda_ridge,
            device=device,
            bias_gate=gate_mod.bias.data if gate_mod.bias is not None else None,
            bias_up=up_mod.bias.data if up_mod.bias is not None else None,
            bias_down=down_mod.bias.data if down_mod.bias is not None else None,
        )
        del c_sigma_gpu

        orig_device = gate_mod.weight.device
        orig_dtype = gate_mod.weight.dtype
        parent.gate_proj = gate_new.to(device=orig_device, dtype=orig_dtype)
        parent.up_proj = up_new.to(device=orig_device, dtype=orig_dtype)
        parent.down_proj = down_new.to(device=orig_device, dtype=orig_dtype)

        k = parent.gate_proj.weight.shape[0]
        dh = gate_mod.weight.shape[1]
        dint = gate_mod.weight.shape[0]
        total_orig += 3 * dint * dh
        total_comp += 3 * k * dh
        logger.info(f"  {parent_path}: dint {dint} -> {k}")

    if total_orig:
        logger.info(f"Nystrom compression complete: {total_comp}/{total_orig} params")
    return model
