"""SVD-LLM-V2 baseline: activation-aware truncated SVD with heterogeneous rank."""

import torch
import torch.nn as nn
from loguru import logger
from typing import Dict, Tuple, Optional, List
from compress.whitening import compute_whitening
import heapq


class SVDCompressedLinear(nn.Module):
    """A linear layer compressed via truncated SVD: y = x @ V_r @ U_r + bias."""

    def __init__(self, U_r: torch.Tensor, V_r: torch.Tensor, bias: Optional[torch.Tensor] = None):
        super().__init__()
        # U_r: (rank, d_out), V_r: (d_in, rank)
        self.V_r = nn.Parameter(V_r, requires_grad=False)
        self.U_r = nn.Parameter(U_r, requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.bias = None

    def forward(self, x):
        out = x @ self.V_r  # (B, rank)
        out = out @ self.U_r  # (B, d_out)
        if self.bias is not None:
            out = out + self.bias
        return out

    @property
    def in_features(self):
        return self.V_r.shape[0]

    @property
    def out_features(self):
        return self.U_r.shape[1]


def svd_compress_layer(
    weight: torch.Tensor,
    rank: int,
    bias: Optional[torch.Tensor] = None,
    precomputed_whitening: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    device: str = "cuda",
) -> SVDCompressedLinear:
    """Compress a single linear layer using activation-aware SVD (SVD-LLM-V2 style).

    Args:
        weight: (d_out, d_in) weight matrix
        rank: target rank
        bias: optional bias vector
        precomputed_whitening: (Phi, Phi_inv) whitening matrices (required)
        device: compute device for SVD/matmul

    Returns:
        SVDCompressedLinear module
    """
    d_out, d_in = weight.shape

    if precomputed_whitening is None:
        raise ValueError("precomputed_whitening is required; call compute_whitening() first")
    Phi, Phi_inv = precomputed_whitening
    Phi = Phi.to(device=device)
    Phi_inv = Phi_inv.to(device=device)

    # Whitened weight: D = W @ Phi (on GPU)
    D = weight.to(device=device, dtype=torch.float32) @ Phi

    # SVD of whitened weight (on GPU)
    U, S, Vh = torch.linalg.svd(D.to(torch.float64), full_matrices=False)

    # Truncate
    use_rank = min(rank, min(d_out, d_in))
    U_r = U[:, :use_rank].float()  # (d_out, use_rank)
    S_r = S[:use_rank].float()  # (use_rank,)
    Vh_r = Vh[:use_rank, :].float()  # (use_rank, d_in)

    # Reconstruct: W' = U_r @ diag(S_r) @ Vh_r @ Phi_inv
    sqrt_S = torch.sqrt(S_r)
    V_r = (Phi_inv.t() @ Vh_r.t() * sqrt_S.unsqueeze(0))  # (d_in, rank)
    U_r = (U_r * sqrt_S.unsqueeze(0)).t()  # (rank, d_out)

    return SVDCompressedLinear(
        U_r.cpu().to(weight.dtype),
        V_r.cpu().to(weight.dtype),
        bias=bias,
    )


def compute_layer_sensitivity(
    weight: torch.Tensor,
    precomputed_whitening: Tuple[torch.Tensor, torch.Tensor],
    device: str = "cuda",
) -> torch.Tensor:
    """Compute per-rank truncation loss for heterogeneous rank allocation.

    Returns:
        Tensor of shape (min(d_out, d_in),) containing squared singular values
        of the whitened weight. The truncation loss at rank r is sum(sv[r:]).
    """
    Phi, _ = precomputed_whitening
    Phi = Phi.to(device=device)
    D = weight.to(device=device, dtype=torch.float32) @ Phi
    _, S, _ = torch.linalg.svd(D.to(torch.float64), full_matrices=False)
    return (S ** 2).float().cpu()


def svd_llm_v2_compress_model(
    model: nn.Module,
    covariances: Dict[str, torch.Tensor],
    compression_ratio: float = 0.7,
    skip_layers: Tuple[str, ...] = ("lm_head",),
    heterogeneous: bool = True,  # kept for API compat, currently ignored
    device: str = "cuda",
) -> nn.Module:
    """Compress entire model using SVD-LLM-V2.

    Args:
        model: pretrained model
        covariances: {layer_name: (d_in, d_in)} covariance matrices from calibration
        compression_ratio: fraction of original params to retain (0.5 = 50%)
        skip_layers: layer names to skip
        heterogeneous: currently ignored (uniform rank for all layers)
        device: compute device

    Returns:
        Compressed model (in-place modification)
    """
    # Collect all linear layers
    layers_to_compress = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.split(".")[-1]
        if leaf in skip_layers:
            continue
        if name not in covariances:
            logger.warning(f"No covariance for {name}, skipping")
            continue
        layers_to_compress.append((name, module))

    total_orig_params = sum(
        m.weight.numel() for _, m in layers_to_compress
    )
    target_params = int(compression_ratio * total_orig_params)
    logger.info(
        f"SVD-LLM-V2: compressing {len(layers_to_compress)} layers, "
        f"target {target_params}/{total_orig_params} params ({compression_ratio:.0%})"
    )

    # Compress each layer: compute whitening lazily and discard immediately
    for name, module in layers_to_compress:
        d_out, d_in = module.weight.shape
        r = int(compression_ratio * d_out * d_in / (d_out + d_in))
        r = max(1, r)

        whitening = compute_whitening(covariances[name], device=device)
        compressed = svd_compress_layer(
            module.weight.data,
            rank=r,
            bias=module.bias.data if module.bias is not None else None,
            precomputed_whitening=whitening,
            device=device,
        )
        del whitening
        compressed = compressed.to(device=module.weight.device, dtype=module.weight.dtype)

        # Replace in model
        path = name.split(".")
        parent = model
        for key in path[:-1]:
            parent = getattr(parent, key)
        setattr(parent, path[-1], compressed)

        d_out, d_in = module.weight.shape
        logger.info(f"  {name}: ({d_out},{d_in}) -> rank {r}, "
                     f"params {r*(d_out+d_in)}/{d_out*d_in}")

    return model


def _allocate_ranks_across_layers(
    layers: list,
    sensitivities: Dict[str, torch.Tensor],
    target_params: int,
) -> Dict[str, int]:
    """Greedy rank allocation across layers based on sensitivity."""
    # Initialize all ranks to 0
    ranks = {name: 0 for name, _ in layers}
    dims = {name: (m.weight.shape[0], m.weight.shape[1]) for name, m in layers}
    current_params = 0

    # Build priority queue: (marginal_gain, layer_name)
    heap = []
    for name, _ in layers:
        sv_sq = sensitivities[name]
        d_out, d_in = dims[name]
        if len(sv_sq) > 0:
            heapq.heappush(heap, (-sv_sq[0].item(), name))

    while current_params < target_params and heap:
        neg_gain, name = heapq.heappop(heap)
        d_out, d_in = dims[name]
        cost = d_out + d_in
        if current_params + cost > target_params:
            continue
        ranks[name] += 1
        current_params += cost

        # Push next rank's gain
        r = ranks[name]
        sv_sq = sensitivities[name]
        max_rank = min(d_out, d_in)
        if r < max_rank and r < len(sv_sq):
            heapq.heappush(heap, (-sv_sq[r].item(), name))

    # Ensure all ranks >= 1
    for name in ranks:
        if ranks[name] == 0:
            ranks[name] = 1

    logger.info(f"Rank allocation: total params = {current_params}/{target_params}")
    return ranks
