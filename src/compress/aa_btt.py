"""Activation-Aware Block Tensor Train decomposition (Layer 1 + 2 + 3)."""

import torch
import torch.nn as nn
from loguru import logger
from typing import Dict, List, Tuple, Optional, Union
from compress.whitening import compute_whitening
from compress.utils import _closest_factor_pair


def compute_inverse_permutation(perm: torch.Tensor) -> torch.Tensor:
    """Compute inverse permutation."""
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(len(perm), device=perm.device)
    return inv


def aa_btt_decompose_layer(
    weight: torch.Tensor,
    rank: Union[int, float],
    bias: Optional[torch.Tensor] = None,
    permutation: Optional[torch.Tensor] = None,
    per_block_ranks: Optional[List[int]] = None,
    precomputed_whitening: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    device: str = "cuda",
) -> dict:
    """Activation-aware BTT decomposition of a single linear layer.

    Uses input_one_block mode: n=1, b=d_in.

    Args:
        weight: (d_out, d_in) weight matrix
        rank: target rank (int) or compression ratio (float in (0,1))
        bias: optional bias vector
        permutation: optional (d_out,) row permutation
        per_block_ranks: optional list of per-block ranks (overrides rank)
        precomputed_whitening: optional (Phi, Phi_inv) whitening matrices
        device: compute device for SVD/matmul

    Returns:
        Dict with keys: btt_l, btt_r, bias, permutation, inv_permutation,
        m, a, ranks
    """
    d_out, d_in = weight.shape
    m, a = _closest_factor_pair(d_out)

    # Resolve rank
    if per_block_ranks is not None:
        ranks = per_block_ranks
        max_rank = max(ranks)
    elif isinstance(rank, float) and 0 < rank < 1:
        target_params = rank * d_out * d_in
        uniform_r = int(target_params / (m * (d_in + a)))
        uniform_r = max(1, min(uniform_r, min(a, d_in)))
        ranks = [uniform_r] * m
        max_rank = uniform_r
    else:
        uniform_r = int(rank)
        ranks = [uniform_r] * m
        max_rank = uniform_r

    # Whitening (on GPU) — None means naive SVD (vanilla_btt)
    use_whitening = False
    if precomputed_whitening is not None:
        Phi, Phi_inv = precomputed_whitening
        Phi = Phi.to(device=device)
        Phi_inv = Phi_inv.to(device=device)
        use_whitening = True

    # Move weight to GPU for decomposition
    W = weight.to(device=device, dtype=torch.float32)
    if permutation is not None:
        W = W[permutation.to(device=device)]

    # Reshape into blocks: (m, a, d_in) for input_one_block
    blocks = W.reshape(m, a, d_in)

    svd_dtype = torch.float64

    # Allocate padded cores
    core_l = torch.zeros(m, a, max_rank, dtype=torch.float32, device=device)
    core_r = torch.zeros(m, max_rank, d_in, dtype=torch.float32, device=device)

    for i in range(m):
        block = blocks[i]  # (a, d_in)
        r_i = ranks[i]

        # Optionally whiten: D_i = W_i @ Phi (or just D_i = W_i for vanilla)
        if use_whitening:
            D_i = block.to(svd_dtype) @ Phi.to(svd_dtype)  # (a, d_in)
        else:
            D_i = block.to(svd_dtype)

        U, S, Vh = torch.linalg.svd(D_i, full_matrices=False)

        use_r = min(r_i, len(S))

        sqrt_S = torch.sqrt(S[:use_r])
        L_i = (U[:, :use_r] * sqrt_S.unsqueeze(0)).float()  # (a, use_r)
        R_i = (sqrt_S.unsqueeze(1) * Vh[:use_r, :]).float()  # (use_r, d_in)

        # Undo whitening on R_i: R_i_final = R_i @ Phi_inv (skip for vanilla)
        if use_whitening:
            R_i_final = (R_i.to(svd_dtype) @ Phi_inv.to(svd_dtype)).float()
        else:
            R_i_final = R_i

        core_l[i, :, :use_r] = L_i
        core_r[i, :use_r, :] = R_i_final

    # Pack into BTT format for input_one_block (n=1)
    btt_r = core_r.permute(0, 2, 1).permute(1, 0, 2).reshape(d_in, m * max_rank).unsqueeze(0).contiguous()
    btt_l = core_l.permute(0, 2, 1).contiguous()  # (m, max_rank, a)

    inv_perm = compute_inverse_permutation(permutation) if permutation is not None else None

    total_params = sum(r_i * (d_in + a) for r_i in ranks)
    orig_params = d_out * d_in

    logger.info(
        f"  AA-BTT: ({d_out},{d_in}) -> m={m},a={a}, ranks={ranks if len(set(ranks))>1 else ranks[0]}, "
        f"params={total_params}/{orig_params} ({total_params/orig_params:.1%})"
    )

    return {
        "btt_l": btt_l.cpu(),
        "btt_r": btt_r.cpu(),
        "bias": bias,
        "permutation": permutation,
        "inv_permutation": inv_perm,
        "m": m,
        "a": a,
        "d_in": d_in,
        "d_out": d_out,
        "ranks": ranks,
        "max_rank": max_rank,
    }
