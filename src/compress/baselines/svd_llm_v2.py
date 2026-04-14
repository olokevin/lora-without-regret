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
    """Compress a single linear layer via SVD.

    When precomputed_whitening is provided, applies activation-aware whitening
    (SVD-LLM style): SVD is performed on W @ Phi instead of W directly.
    When None, performs plain SVD on the raw weight matrix (no calibration needed).

    Args:
        weight: (d_out, d_in) weight matrix
        rank: target rank
        bias: optional bias vector
        precomputed_whitening: (Phi, Phi_inv) from compute_whitening(), or None for plain SVD
        device: compute device for SVD/matmul

    Returns:
        SVDCompressedLinear module
    """
    d_out, d_in = weight.shape
    use_rank = min(rank, min(d_out, d_in))
    W = weight.to(device=device, dtype=torch.float32)

    if precomputed_whitening is not None:
        Phi, Phi_inv = precomputed_whitening
        Phi = Phi.to(device=device)
        Phi_inv = Phi_inv.to(device=device)
        # SVD of whitened weight: D = W @ Phi
        U, S, Vh = torch.linalg.svd(W @ Phi, full_matrices=False)
        U_r, S_r, Vh_r = U[:, :use_rank], S[:use_rank], Vh[:use_rank, :]
        # W' = U_r @ diag(S_r) @ Vh_r @ Phi_inv
        sqrt_S = torch.sqrt(S_r)
        V_r = (Phi_inv.t() @ Vh_r.t() * sqrt_S.unsqueeze(0))  # (d_in, rank)
        U_r = (U_r * sqrt_S.unsqueeze(0)).t()                  # (rank, d_out)
    else:
        # Plain SVD: W = U @ diag(S) @ Vh
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        U_r, S_r, Vh_r = U[:, :use_rank], S[:use_rank], Vh[:use_rank, :]
        sqrt_S = torch.sqrt(S_r)
        V_r = (Vh_r.t() * sqrt_S.unsqueeze(0))   # (d_in, rank)
        U_r = (U_r * sqrt_S.unsqueeze(0)).t()     # (rank, d_out)

    return SVDCompressedLinear(
        U_r.cpu().to(weight.dtype),
        V_r.cpu().to(weight.dtype),
        bias=bias,
    )


def svd_compress_layer_backward(
    weight: torch.Tensor,
    rank: int,
    bias: Optional[torch.Tensor] = None,
    precomputed_whitening: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    device: str = "cuda",
) -> SVDCompressedLinear:
    """Compress a single linear layer via backward-objective SVD.

    Minimises ||(W - W_hat).T @ Phi||_F^2 where Phi comes from the output
    gradient covariance C_dy (i.e. Phi satisfies Phi @ Phi.T ~ C_dy).

    When precomputed_whitening is provided, performs SVD on Phi.T @ W (left-
    whitened). When None, falls back to plain SVD on W.

    Args:
        weight: (d_out, d_in) weight matrix
        rank: target rank
        bias: optional bias vector
        precomputed_whitening: (Phi, Phi_inv) from compute_whitening(C_dy), or None
        device: compute device for SVD/matmul

    Returns:
        SVDCompressedLinear module
    """
    d_out, d_in = weight.shape
    use_rank = min(rank, min(d_out, d_in))
    W = weight.to(device=device, dtype=torch.float32)

    if precomputed_whitening is not None:
        Phi, Phi_inv = precomputed_whitening
        Phi = Phi.to(device=device)
        Phi_inv = Phi_inv.to(device=device)
        # Left-whiten: D_bp = Phi.T @ W  (d_out × d_in)
        D_bp = Phi.t() @ W
        U, S, Vh = torch.linalg.svd(D_bp, full_matrices=False)
        U_r, S_r, Vh_r = U[:, :use_rank], S[:use_rank], Vh[:use_rank, :]
        sqrt_S = torch.sqrt(S_r)
        # W ≈ Phi_inv.T @ U_r diag(sqrt_S)^2 Vh_r  =>  split as (Phi_inv.T @ U_r diag(sqrt_S)) @ (diag(sqrt_S) Vh_r)
        # Equivalently: V_r @ U_r_out = W.T where
        #   V_r     = Vh_r.T * sqrt_S          (d_in, rank)
        #   U_r_out = (U_r * sqrt_S).T @ Phi_inv  (rank, d_out)
        V_r = Vh_r.t() * sqrt_S.unsqueeze(0)                    # (d_in, rank)
        U_r_out = (U_r * sqrt_S.unsqueeze(0)).t() @ Phi_inv      # (rank, d_out)
    else:
        # Plain SVD fallback: W = U @ diag(S) @ Vh
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        U_r, S_r, Vh_r = U[:, :use_rank], S[:use_rank], Vh[:use_rank, :]
        sqrt_S = torch.sqrt(S_r)
        V_r = Vh_r.t() * sqrt_S.unsqueeze(0)    # (d_in, rank)
        U_r_out = (U_r * sqrt_S.unsqueeze(0)).t()  # (rank, d_out)

    return SVDCompressedLinear(
        U_r_out.cpu().to(weight.dtype),
        V_r.cpu().to(weight.dtype),
        bias=bias,
    )


def svd_compress_layer_combined(
    weight: torch.Tensor,
    rank: int,
    bias: Optional[torch.Tensor] = None,
    whitening_input: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    whitening_gradient: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    device: str = "cuda",
) -> SVDCompressedLinear:
    """Compress a single linear layer via doubly-whitened SVD.

    Minimises ||C_g^{1/2} (W - W_hat) C_x^{1/2}||_F^2 by performing truncated
    SVD on M = Phi_g^T @ W @ Phi_x, then reconstructing via inverse whitening.

    When only one whitening is provided, falls back to the corresponding
    single-sided method. When neither is provided, falls back to plain SVD.

    Args:
        weight: (d_out, d_in) weight matrix
        rank: target rank
        bias: optional bias vector
        whitening_input: (Phi_x, Phi_x_inv) from compute_whitening(C_x)
        whitening_gradient: (Phi_g, Phi_g_inv) from compute_whitening(C_g)
        device: compute device

    Returns:
        SVDCompressedLinear module
    """
    # Fallback: if only one side provided, delegate to single-sided method
    if whitening_input is not None and whitening_gradient is None:
        return svd_compress_layer(weight, rank, bias=bias,
                                  precomputed_whitening=whitening_input, device=device)
    if whitening_gradient is not None and whitening_input is None:
        return svd_compress_layer_backward(weight, rank, bias=bias,
                                           precomputed_whitening=whitening_gradient, device=device)
    if whitening_input is None and whitening_gradient is None:
        return svd_compress_layer(weight, rank, bias=bias,
                                  precomputed_whitening=None, device=device)

    d_out, d_in = weight.shape
    use_rank = min(rank, min(d_out, d_in))
    W = weight.to(device=device, dtype=torch.float32)

    Phi_x, Phi_x_inv = whitening_input
    Phi_g, Phi_g_inv = whitening_gradient
    Phi_x = Phi_x.to(device); Phi_x_inv = Phi_x_inv.to(device)
    Phi_g = Phi_g.to(device); Phi_g_inv = Phi_g_inv.to(device)

    # Doubly-whitened matrix: M = Phi_g^T @ W @ Phi_x
    M = Phi_g.t() @ W @ Phi_x
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    U_r, S_r, Vh_r = U[:, :use_rank], S[:use_rank], Vh[:use_rank, :]

    # Reconstruct: W_hat = Phi_g_inv^T @ U_r @ diag(S_r) @ Vh_r @ Phi_x_inv
    # Split into sqrt for balanced factored storage
    sqrt_S = torch.sqrt(S_r)
    V_r = (Phi_x_inv.t() @ Vh_r.t()) * sqrt_S.unsqueeze(0)   # (d_in, rank)
    U_r_out = (U_r * sqrt_S.unsqueeze(0)).t() @ Phi_g_inv     # (rank, d_out)

    return SVDCompressedLinear(
        U_r_out.cpu().to(weight.dtype),
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
    _, S, _ = torch.linalg.svd(D, full_matrices=False)
    return (S ** 2).cpu()


def _resolve_parent_and_module(model: nn.Module, module_name: str) -> Tuple[nn.Module, str, nn.Module]:
    """Resolve dotted module path into (parent, leaf_name, module)."""
    path = module_name.split(".")
    parent = model
    for key in path[:-1]:
        parent = getattr(parent, key)
    leaf = path[-1]
    module = getattr(parent, leaf)
    return parent, leaf, module


def svd_llm_v2_compress_model(
    model: nn.Module,
    covariances: Dict[str, torch.Tensor],
    compression_ratio: float = 0.7,
    skip_layers: Tuple[str, ...] = ("lm_head",),
    heterogeneous: bool = True,  # kept for API compat, currently ignored
    device: str = "cuda",
    objective: str = "forward",
    backward_covariances: Optional[Dict[str, torch.Tensor]] = None,
) -> nn.Module:
    """Compress entire model using SVD-LLM-V2.

    Args:
        model: pretrained model
        covariances: {layer_name: covariance matrix} from calibration.
            For objective="forward": input activation covariances (d_in, d_in).
            For objective="backward": output gradient covariances (d_out, d_out).
        compression_ratio: fraction of original params to retain (0.5 = 50%)
        skip_layers: layer names to skip
        heterogeneous: currently ignored (uniform rank for all layers)
        device: compute device
        objective: "forward" (default, input whitening) or "backward" (output gradient whitening)
        backward_covariances: {layer_name: (d_out, d_out) gradient covariance} required
            when objective="combined". Ignored for other objectives.

    Returns:
        Compressed model (in-place modification)
    """
    # Collect target layer names only (not module refs), so replaced dense modules
    # can be released immediately instead of being held by a local list.
    layer_names: List[str] = []
    total_orig_params = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.split(".")[-1]
        if leaf in skip_layers:
            continue
        if name not in covariances:
            logger.warning(f"No covariance for {name}, skipping")
            continue
        layer_names.append(name)
        total_orig_params += module.weight.numel()

    target_params = int(compression_ratio * total_orig_params)
    logger.info(
        f"SVD-LLM-V2: compressing {len(layer_names)} layers, "
        f"target {target_params}/{total_orig_params} params ({compression_ratio:.0%})"
    )

    # Compress each layer: load covariance to GPU one at a time, free immediately
    for name in layer_names:
        parent, leaf, module = _resolve_parent_and_module(model, name)
        if not isinstance(module, nn.Linear):
            logger.warning(f"Layer {name} is no longer nn.Linear, skipping")
            continue

        d_out, d_in = module.weight.shape
        if compression_ratio >= 1.0:
            r = min(d_out, d_in)  # full-rank: apply whitening transform only, no truncation
        else:
            r = int(compression_ratio * d_out * d_in / (d_out + d_in))
            r = max(1, r)

        cov_cpu = covariances.pop(name, None)
        if cov_cpu is None:
            logger.warning(f"Missing covariance for {name} at compression time, skipping")
            continue
        cov_gpu = cov_cpu.to(device=device)
        del cov_cpu
        whitening = compute_whitening(cov_gpu, device=device)
        del cov_gpu
        if objective == "combined":
            bwd_cov_cpu = backward_covariances.pop(name, None)
            if bwd_cov_cpu is None:
                logger.warning(f"Missing backward covariance for {name} at compression time, skipping")
                del whitening
                continue
            bwd_cov_gpu = bwd_cov_cpu.to(device=device)
            del bwd_cov_cpu
            bwd_whitening = compute_whitening(bwd_cov_gpu, device=device)
            del bwd_cov_gpu
            compressed = svd_compress_layer_combined(
                module.weight.data,
                rank=r,
                bias=module.bias.data if module.bias is not None else None,
                whitening_input=whitening,
                whitening_gradient=bwd_whitening,
                device=device,
            )
            del bwd_whitening
        elif objective == "backward":
            compressed = svd_compress_layer_backward(
                module.weight.data,
                rank=r,
                bias=module.bias.data if module.bias is not None else None,
                precomputed_whitening=whitening,
                device=device,
            )
        else:
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
        setattr(parent, leaf, compressed)

        logger.info(f"  {name}: ({d_out},{d_in}) -> rank {r}, "
                     f"params {r*(d_out+d_in)}/{d_out*d_in}")
        del module
        del compressed
        del parent

    # Ensure unconsumed covariance tensors do not linger in memory.
    if covariances:
        logger.info(f"Releasing {len(covariances)} unused covariance entries.")
        covariances.clear()

    if backward_covariances:
        logger.info(f"Releasing {len(backward_covariances)} unused backward covariance entries.")
        backward_covariances.clear()

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
