"""Weight update analysis: compute metrics comparing base model vs trained checkpoint."""

import torch
import numpy as np


def materialize_blocktt_weight(
    btt_l: torch.Tensor, btt_r: torch.Tensor, btt_s: torch.Tensor | None = None
) -> torch.Tensor:
    """Reconstruct dense weight from BlockTT cores.

    Args:
        btt_l: (m, rank*n, a) — left/output core
        btt_r: (n, b, m*rank) — right/input core
        btt_s: (m, n, rank) — optional singular values

    Returns:
        Dense weight tensor of shape (m*a, n*b).
    """
    m, _, a = btt_l.shape
    n, b, _ = btt_r.shape
    rank = btt_r.shape[2] // m

    # Reshape to (m, n, rank, a) and (m, n, rank, b)
    l = btt_l.reshape(m, n, rank, a)
    r = btt_r.reshape(n, b, m, rank).permute(2, 0, 3, 1)  # (m, n, rank, b)

    if btt_s is not None:
        l = l * btt_s.unsqueeze(-1)  # (m, n, rank, a) * (m, n, rank, 1)

    # Contract over rank: (m, n, a, b)
    w_blocks = torch.einsum("mnra,mnrb->mnab", l, r)

    # Assemble into (m*a, n*b)
    return w_blocks.permute(0, 2, 1, 3).reshape(m * a, n * b)


def materialize_svd_weight(
    svd_a: torch.Tensor, svd_b: torch.Tensor, svd_s: torch.Tensor | None = None
) -> torch.Tensor:
    """Reconstruct dense weight from SVD factors.

    Args:
        svd_a: (out_features, rank)
        svd_b: (rank, in_features)
        svd_s: (rank,) — optional singular values

    Returns:
        Dense weight of shape (out_features, in_features).
    """
    if svd_s is not None:
        return (svd_a * svd_s.unsqueeze(0)) @ svd_b
    return svd_a @ svd_b


def reconstruct_lora_weight(
    W_base: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    lora_alpha: float,
    r: int,
) -> torch.Tensor:
    """Reconstruct dense weight from base weight + LoRA adapter.

    Args:
        W_base: (out_features, in_features) — original linear weight
        lora_A: (rank, in_features)
        lora_B: (out_features, rank)
        lora_alpha: LoRA scaling alpha
        r: LoRA rank

    Returns:
        W_base + (lora_alpha / r) * lora_B @ lora_A
    """
    scaling = lora_alpha / r
    return W_base + scaling * (lora_B @ lora_A)
