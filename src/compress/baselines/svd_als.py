"""SVD-ALS: Alternating Least Squares low-rank compression.

Directly minimizes the additive combined loss:
    alpha * ||E @ C_x^{1/2}||_F^2 + beta * ||C_g^{1/2} @ E||_F^2
where E = W - L @ R^T, via alternating least squares.
"""

import torch
import torch.nn as nn
from loguru import logger
from typing import Dict, List, Optional, Tuple

from compress.baselines.svd_llm_v2 import (
    SVDCompressedLinear,
    svd_compress_layer_combined,
    _resolve_parent_and_module,
)
from compress.whitening import compute_whitening


def _warm_start_LR(
    weight: torch.Tensor,
    rank: int,
    C_x: torch.Tensor,
    C_g: torch.Tensor,
    reg_eps: float,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Initialize L, R from doubly-whitened SVD (svd_compress_layer_combined).

    Returns:
        L: (d_out, rank) float32 on device
        R: (d_in, rank) float32 on device
    """
    whitening_x = compute_whitening(C_x, regularize_eps=reg_eps, device=device)
    whitening_g = compute_whitening(C_g, regularize_eps=reg_eps, device=device)
    compressed = svd_compress_layer_combined(
        weight, rank,
        whitening_input=whitening_x,
        whitening_gradient=whitening_g,
        device=device,
    )
    # SVDCompressedLinear stores U_r: (rank, d_out), V_r: (d_in, rank)
    # We need L: (d_out, rank), R: (d_in, rank)
    L = compressed.U_r.data.t().to(device=device, dtype=torch.float32)  # (d_out, rank)
    R = compressed.V_r.data.to(device=device, dtype=torch.float32)      # (d_in, rank)
    del compressed, whitening_x, whitening_g
    return L, R


def _compute_loss(
    W: torch.Tensor,
    L: torch.Tensor,
    R: torch.Tensor,
    C_x: torch.Tensor,
    C_g: torch.Tensor,
    alpha: float,
    beta: float,
) -> float:
    """Compute additive loss: alpha * tr(E C_x E^T) + beta * tr(C_g E E^T)."""
    E = W - L @ R.t()
    # Forward term: alpha * ||E @ C_x^{1/2}||^2 = alpha * tr(E C_x E^T)
    fwd = torch.sum(E * (E @ C_x))  # tr(E C_x E^T) = sum(E * (E @ C_x))
    # Backward term: beta * ||C_g^{1/2} @ E||^2 = beta * tr(C_g E E^T)
    bwd = torch.sum(E * (C_g @ E))  # tr(C_g E E^T) = sum(E * (C_g @ E))
    return (alpha * fwd + beta * bwd).item()


def svd_als_compress_layer(
    weight: torch.Tensor,
    rank: int,
    C_x: torch.Tensor,
    C_g: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    n_iter: int = 10,
    tol: float = 1e-6,
    reg_eps: float = 1e-4,
    weighting: str = "equal",
    device: str = "cuda",
) -> SVDCompressedLinear:
    """Compress a single linear layer via ALS on the additive combined loss.

    Minimises alpha * ||E C_x^{1/2}||_F^2 + beta * ||C_g^{1/2} E||_F^2
    where E = W - L R^T, using alternating least squares with warm start
    from doubly-whitened SVD.

    Args:
        weight: (d_out, d_in) weight matrix
        rank: target rank
        C_x: (d_in, d_in) input activation covariance
        C_g: (d_out, d_out) output gradient covariance
        bias: optional bias vector (passed through unchanged)
        alpha: forward loss weight (overridden if weighting != "equal")
        beta: backward loss weight (overridden if weighting != "equal")
        n_iter: max ALS iterations
        tol: relative loss decrease threshold for early stopping
        reg_eps: regularization for eigendecompositions
        weighting: "equal" (alpha=beta=1) or "trace" (normalize by trace)
        device: compute device

    Returns:
        SVDCompressedLinear module
    """
    d_out, d_in = weight.shape
    use_rank = min(rank, min(d_out, d_in))

    W = weight.to(device=device, dtype=torch.float32)
    C_x = C_x.to(device=device, dtype=torch.float32)
    C_g = C_g.to(device=device, dtype=torch.float32)

    # Determine alpha, beta
    if weighting == "trace":
        tr_x = torch.trace(C_x).item()
        tr_g = torch.trace(C_g).item()
        alpha = 1.0 / max(tr_x, 1e-12)
        beta = 1.0 / max(tr_g, 1e-12)

    # Warm start from doubly-whitened SVD
    L, R = _warm_start_LR(weight, use_rank, C_x, C_g, reg_eps, device)

    if n_iter == 0:
        return SVDCompressedLinear(
            L.t().cpu().to(weight.dtype),
            R.cpu().to(weight.dtype),
            bias=bias,
        )

    # Eigendecompose covariances (done once, reused across iterations)
    # C_g = Q_g @ diag(eigvals_g) @ Q_g^T
    eigvals_g, Q_g = torch.linalg.eigh(C_g + reg_eps * torch.eye(d_out, device=device))
    # C_x = Q_x @ diag(eigvals_x) @ Q_x^T
    eigvals_x, Q_x = torch.linalg.eigh(C_x + reg_eps * torch.eye(d_in, device=device))

    prev_loss = _compute_loss(W, L, R, C_x, C_g, alpha, beta)

    for it in range(n_iter):
        # === L-update (R fixed) ===
        # Normal eq: alpha * L (R^T C_x R) + beta * C_g L (R^T R) = alpha * W C_x R + beta * C_g W R
        # Decouple via Q_g: let L_tilde = Q_g^T L, RHS_tilde = Q_g^T RHS
        # Row i: (alpha * R^T C_x R + beta * eigvals_g[i] * R^T R) @ l_tilde_i^T = rhs_tilde_i^T
        RtCxR = R.t() @ C_x @ R                            # (k, k)
        RtR = R.t() @ R                                     # (k, k)
        RHS_L = alpha * W @ C_x @ R + beta * C_g @ W @ R   # (d_out, k)
        RHS_L_tilde = Q_g.t() @ RHS_L                       # (d_out, k)

        # Build batch of (d_out, k, k) coefficient matrices
        # M_i = alpha * RtCxR + beta * eigvals_g[i] * RtR
        M_L = alpha * RtCxR.unsqueeze(0) + beta * eigvals_g.unsqueeze(1).unsqueeze(2) * RtR.unsqueeze(0)
        # Solve: M_L[i] @ L_tilde[i] = RHS_L_tilde[i] for each row i
        L_tilde = torch.linalg.solve(M_L, RHS_L_tilde.unsqueeze(-1)).squeeze(-1)  # (d_out, k)
        L = Q_g @ L_tilde                                   # (d_out, k)

        # === R-update (L fixed) ===
        # Normal eq: alpha * C_x R (L^T L) + beta * R (L^T C_g L) = alpha * C_x W^T L + beta * W^T C_g^T L
        # Decouple via Q_x: let R_tilde = Q_x^T R
        # Row j: (alpha * eigvals_x[j] * L^T L + beta * L^T C_g L) @ r_tilde_j^T = rhs_tilde_j^T
        LtL = L.t() @ L                                     # (k, k)
        LtCgL = L.t() @ C_g @ L                             # (k, k)
        RHS_R = alpha * C_x @ W.t() @ L + beta * W.t() @ C_g.t() @ L  # (d_in, k)
        RHS_R_tilde = Q_x.t() @ RHS_R                       # (d_in, k)

        # Build batch of (d_in, k, k) coefficient matrices
        # M_j = alpha * eigvals_x[j] * LtL + beta * LtCgL
        M_R = alpha * eigvals_x.unsqueeze(1).unsqueeze(2) * LtL.unsqueeze(0) + beta * LtCgL.unsqueeze(0)
        R_tilde = torch.linalg.solve(M_R, RHS_R_tilde.unsqueeze(-1)).squeeze(-1)  # (d_in, k)
        R = Q_x @ R_tilde                                   # (d_in, k)

        # Convergence check
        loss = _compute_loss(W, L, R, C_x, C_g, alpha, beta)
        rel_decrease = (prev_loss - loss) / max(abs(prev_loss), 1e-12)
        if rel_decrease < tol and it > 0:
            logger.debug(f"ALS converged at iteration {it + 1}, loss={loss:.6e}")
            break
        prev_loss = loss

    return SVDCompressedLinear(
        L.t().cpu().to(weight.dtype),  # U_r: (rank, d_out)
        R.cpu().to(weight.dtype),      # V_r: (d_in, rank)
        bias=bias,
    )


def svd_als_compress_model(
    model: nn.Module,
    fwd_covariances: Dict[str, torch.Tensor],
    bwd_covariances: Dict[str, torch.Tensor],
    compression_ratio: float = 0.7,
    skip_layers: Tuple[str, ...] = ("lm_head",),
    device: str = "cuda",
    als_n_iter: int = 10,
    als_tol: float = 1e-6,
    als_weighting: str = "equal",
    als_reg_eps: float = 1e-4,
) -> nn.Module:
    """Compress entire model using SVD-ALS.

    Args:
        model: pretrained model
        fwd_covariances: {layer_name: (d_in, d_in) input activation covariance}
        bwd_covariances: {layer_name: (d_out, d_out) gradient covariance}
        compression_ratio: fraction of original params to retain
        skip_layers: layer names to skip
        device: compute device
        als_n_iter: max ALS iterations per layer
        als_tol: early stopping tolerance
        als_weighting: "equal" or "trace"
        als_reg_eps: regularization epsilon

    Returns:
        Compressed model (in-place modification)
    """
    layer_names: List[str] = []
    total_orig_params = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.split(".")[-1]
        if leaf in skip_layers:
            continue
        if name not in fwd_covariances:
            logger.warning(f"No forward covariance for {name}, skipping")
            continue
        if name not in bwd_covariances:
            logger.warning(f"No backward covariance for {name}, skipping")
            continue
        layer_names.append(name)
        total_orig_params += module.weight.numel()

    logger.info(
        f"SVD-ALS: compressing {len(layer_names)} layers, "
        f"target {compression_ratio:.0%} of {total_orig_params} params, "
        f"n_iter={als_n_iter}, weighting={als_weighting}"
    )

    for name in layer_names:
        parent, leaf, module = _resolve_parent_and_module(model, name)
        if not isinstance(module, nn.Linear):
            logger.warning(f"Layer {name} is no longer nn.Linear, skipping")
            continue

        d_out, d_in = module.weight.shape
        if compression_ratio >= 1.0:
            r = min(d_out, d_in)
        else:
            r = int(compression_ratio * d_out * d_in / (d_out + d_in))
            r = max(1, r)

        cov_x = fwd_covariances.pop(name, None)
        cov_g = bwd_covariances.pop(name, None)
        if cov_x is None or cov_g is None:
            logger.warning(f"Missing covariance for {name}, skipping")
            continue

        compressed = svd_als_compress_layer(
            module.weight.data,
            rank=r,
            C_x=cov_x,
            C_g=cov_g,
            bias=module.bias.data if module.bias is not None else None,
            n_iter=als_n_iter,
            tol=als_tol,
            reg_eps=als_reg_eps,
            weighting=als_weighting,
            device=device,
        )
        del cov_x, cov_g
        compressed = compressed.to(device=module.weight.device, dtype=module.weight.dtype)

        setattr(parent, leaf, compressed)
        logger.info(f"  {name}: ({d_out},{d_in}) -> rank {r}, "
                     f"params {r*(d_out+d_in)}/{d_out*d_in}")
        del module, compressed, parent

    if fwd_covariances:
        logger.info(f"Releasing {len(fwd_covariances)} unused forward covariance entries.")
        fwd_covariances.clear()
    if bwd_covariances:
        logger.info(f"Releasing {len(bwd_covariances)} unused backward covariance entries.")
        bwd_covariances.clear()

    return model
