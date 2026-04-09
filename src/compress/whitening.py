"""Compute activation-aware whitening transforms."""

import torch
from typing import Tuple


def compute_whitening(
    covariance: torch.Tensor,
    regularize_eps: float = 1e-4,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute whitening transform from a pre-computed covariance matrix.

    Given covariance C = (1/N) X^T X of shape (d_in, d_in), computes the
    whitening factor Phi = U_s @ diag(sqrt(S_s)) such that Phi_inv @ X has
    identity covariance.

    Uses SVD of the covariance matrix (not Cholesky) for numerical stability,
    following SVD-LLM-V2.

    Args:
        covariance: (d_in, d_in) covariance matrix (e.g. from collect_covariances_from_loader)
        regularize_eps: regularization added to singular values
        device: compute device for SVD

    Returns:
        Phi: (d_in, d_in) whitening factor (float32, on CPU)
        Phi_inv: (d_in, d_in) inverse whitening factor (float32, on CPU)
    """
    C = covariance.to(dtype=torch.float64, device=device)

    # SVD of symmetric PSD matrix C = U_s S_s U_s^T
    U_s, S_s, _ = torch.linalg.svd(C, full_matrices=False)

    S_s_reg = S_s + regularize_eps
    sqrt_S = torch.sqrt(S_s_reg)

    Phi = U_s * sqrt_S.unsqueeze(0)                         # (d_in, d_in)
    Phi_inv = (U_s * (1.0 / sqrt_S).unsqueeze(0)).t()       # (d_in, d_in)

    return Phi.to(torch.float32), Phi_inv.to(torch.float32)
