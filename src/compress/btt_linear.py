"""BTTLinear: inference module for compressed BTT layers with optional permutation."""

import torch
import torch.nn as nn
from typing import Optional, List


class BTTLinear(nn.Module):
    """Linear layer compressed as 2-core BTT with input_one_block layout.

    Supports:
    - Activation-aware initialization
    - Row permutation (stored as index tensor)
    - Variable per-block ranks (padded to max_rank)

    Parameter shapes (input_one_block, n=1):
        btt_r: (1, d_in, m * max_rank)
        btt_l: (m, max_rank, a)
    """

    def __init__(
        self,
        btt_l: torch.Tensor,
        btt_r: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        inv_permutation: Optional[torch.Tensor] = None,
        m: int = 0,
        a: int = 0,
    ):
        super().__init__()
        self.btt_l = nn.Parameter(btt_l, requires_grad=False)
        self.btt_r = nn.Parameter(btt_r, requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.bias = None

        if inv_permutation is not None:
            self.register_buffer("inv_permutation", inv_permutation)
        else:
            self.inv_permutation = None

        self.m = m if m > 0 else btt_l.shape[0]
        self.a = a if a > 0 else btt_l.shape[2]
        self.d_in = btt_r.shape[1]
        self.d_out = self.m * self.a
        self.max_rank = btt_l.shape[1]

    @property
    def in_features(self):
        return self.d_in

    @property
    def out_features(self):
        return self.d_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape  # (..., d_in)
        x_flat = x.reshape(-1, self.d_in)  # (B, d_in)
        B = x_flat.shape[0]

        # Step 1: x @ btt_r[0] -> (B, m*max_rank)
        inner = x_flat @ self.btt_r[0]  # (B, m * max_rank)
        inner = inner.reshape(B, self.m, self.max_rank)  # (B, m, max_rank)

        # Step 2: batched matmul: (m, B, max_rank) @ (m, max_rank, a) -> (m, B, a)
        inner_t = inner.permute(1, 0, 2)  # (m, B, max_rank)
        out = torch.bmm(inner_t, self.btt_l)  # (m, B, a)
        out = out.permute(1, 0, 2).reshape(B, self.d_out)  # (B, m*a = d_out)

        # Step 3: inverse permutation to restore original output order
        if self.inv_permutation is not None:
            out = out[:, self.inv_permutation]

        # Step 4: bias
        if self.bias is not None:
            out = out + self.bias

        return out.reshape(*orig_shape[:-1], self.d_out)

    def extra_repr(self):
        return (
            f"d_in={self.d_in}, d_out={self.d_out}, m={self.m}, a={self.a}, "
            f"max_rank={self.max_rank}, permuted={self.inv_permutation is not None}"
        )
