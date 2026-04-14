"""SVD-TwoSteps: Trainability-aware two-step L/R decomposition.

Computes balanced low-rank initialization, updates L via doubly-whitened
surrogate, then refits R with forward-covariance weighting.
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
    """Initialize L, R from doubly-whitened SVD (same as svd_als warm start).

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
    L = compressed.U_r.data.t().to(device=device, dtype=torch.float32)
    R = compressed.V_r.data.to(device=device, dtype=torch.float32)
    del compressed, whitening_x, whitening_g
    return L, R


def _update_L_closed_form(
    L0: torch.Tensor,
    C_z: torch.Tensor,
    C_g: torch.Tensor,
    rank: int,
    reg_eps: float,
    device: str,
) -> torch.Tensor:
    """Update L via doubly-whitened surrogate.

    Args:
        L0: (d_out, r) current L estimate
        C_z: (r, r) intermediate covariance R^T C_x R
        C_g: (d_out, d_out) backward covariance
        rank: target rank for truncation
        reg_eps: regularization epsilon
        device: compute device

    Returns:
        L_hat: (d_out, r) updated L
    """
    phi_z, phi_z_inv = compute_whitening(C_z, regularize_eps=reg_eps, device=device)
    phi_g, phi_g_inv = compute_whitening(C_g, regularize_eps=reg_eps, device=device)

    # Move to device
    phi_z = phi_z.to(device=device)
    phi_z_inv = phi_z_inv.to(device=device)
    phi_g = phi_g.to(device=device)
    phi_g_inv = phi_g_inv.to(device=device)

    M = phi_g.t() @ L0 @ phi_z
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    r = min(rank, U.shape[1], Vh.shape[0])
    M_r = (U[:, :r] * S[:r].unsqueeze(0)) @ Vh[:r, :]
    L_hat = phi_g_inv.t() @ M_r @ phi_z_inv
    return L_hat


def _refit_R_forward_weighted(
    W: torch.Tensor,
    L_hat: torch.Tensor,
    C_x: torch.Tensor,
    reg_eps: float,
) -> torch.Tensor:
    """Refit R via regularized least squares: R = W^T L (L^T L + eps*I)^{-1}.

    For the forward-weighted objective min_R tr((W-LR^T) C_x (W-LR^T)^T),
    the normal equation is L^T L R^T C_x = L^T W C_x, where C_x cancels
    from both sides, leaving R = W^T L (L^T L)^{-1}.

    Args:
        W: (d_out, d_in) weight matrix
        L_hat: (d_out, r) updated L
        C_x: (d_in, d_in) forward covariance (unused — kept for API stability)
        reg_eps: regularization epsilon

    Returns:
        R: (d_in, r) refitted R
    """
    k = L_hat.shape[1]
    eye_k = torch.eye(k, device=L_hat.device, dtype=L_hat.dtype)
    G = L_hat.t() @ L_hat + reg_eps * eye_k  # (k, k)
    WtL = W.t() @ L_hat  # (d_in, k)
    R = torch.linalg.solve(G.t(), WtL.t()).t()  # (d_in, k)
    return R


def svd_twosteps_compress_layer(
    weight: torch.Tensor,
    rank: int,
    C_x: torch.Tensor,
    C_g: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    n_refine: int = 1,
    reg_eps: float = 1e-4,
    device: str = "cuda",
) -> SVDCompressedLinear:
    """Compress a single linear layer via two-step L/R decomposition.

    Args:
        weight: (d_out, d_in) weight matrix
        rank: target rank
        C_x: (d_in, d_in) input activation covariance
        C_g: (d_out, d_out) output gradient covariance
        bias: optional bias vector (passed through unchanged)
        n_refine: number of refinement iterations (0 = init only)
        reg_eps: regularization epsilon for solves/whitening
        device: compute device

    Returns:
        SVDCompressedLinear module
    """
    if n_refine < 0:
        raise ValueError(f"n_refine must be >= 0, got {n_refine}")
    if reg_eps <= 0:
        raise ValueError(f"reg_eps must be > 0, got {reg_eps}")

    d_out, d_in = weight.shape
    use_rank = min(rank, min(d_out, d_in))

    W = weight.to(device=device, dtype=torch.float32)
    C_x = C_x.to(device=device, dtype=torch.float32)
    C_g = C_g.to(device=device, dtype=torch.float32)

    # Step 1: Doubly-whitened SVD warm start (same as svd_als / svd_llm_v2_combined)
    L, R = _warm_start_LR(weight, use_rank, C_x, C_g, reg_eps, device)

    # Steps 2-4, repeated n_refine times (0 = warm start only)
    for _ in range(n_refine):
        # Step 2: Intermediate covariance
        C_z = R.t() @ C_x @ R  # (r, r)

        # Step 3: Update L
        L = _update_L_closed_form(L, C_z, C_g, use_rank, reg_eps, device)

        # Step 4: Refit R
        R = _refit_R_forward_weighted(W, L, C_x, reg_eps)

    return SVDCompressedLinear(
        L.t().cpu().to(weight.dtype),  # U_r: (rank, d_out)
        R.cpu().to(weight.dtype),      # V_r: (d_in, rank)
        bias=bias,
    )


def svd_twosteps_compress_model(
    model: nn.Module,
    fwd_covariances: Dict[str, torch.Tensor],
    bwd_covariances: Dict[str, torch.Tensor],
    compression_ratio: float = 0.7,
    skip_layers: Tuple[str, ...] = ("lm_head",),
    device: str = "cuda",
    n_refine: int = 1,
    reg_eps: float = 1e-4,
) -> nn.Module:
    """Compress entire model using SVD-TwoSteps.

    Args:
        model: pretrained model
        fwd_covariances: {layer_name: (d_in, d_in) input activation covariance}
        bwd_covariances: {layer_name: (d_out, d_out) gradient covariance}
        compression_ratio: fraction of original params to retain
        skip_layers: layer names to skip
        device: compute device
        n_refine: refinement iterations per layer
        reg_eps: regularization epsilon

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
        f"SVD-TwoSteps: compressing {len(layer_names)} layers, "
        f"target {compression_ratio:.0%} of {total_orig_params} params, "
        f"n_refine={n_refine}"
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

        compressed = svd_twosteps_compress_layer(
            module.weight.data,
            rank=r,
            C_x=cov_x,
            C_g=cov_g,
            bias=module.bias.data if module.bias is not None else None,
            n_refine=n_refine,
            reg_eps=reg_eps,
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
