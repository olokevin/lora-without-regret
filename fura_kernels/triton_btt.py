"""FuRA Step-2 fused kernel.

Step 2 of FuRA's forward is: out = bmm(inner, L_eff) where
  inner:  (m, B, r*n)
  L:      (m, r*n, a)
  S:      (m, n, r)  (optional, diagonal per-block singular scale)
  L_eff:  L with S folded in, shape (m, r*n, a).

The eager path in btt_layer.py materialises L_eff as a fresh tensor every
forward (m * n * r * a floats). This kernel folds S into the GEMM epilogue
so that materialisation is avoided.

This file starts with a pure-PyTorch reference wrapper (for autograd + API
stability) and will be replaced by a Triton kernel in Task 11.
"""
from __future__ import annotations
import torch


class _Step2SScaledBMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inner: torch.Tensor, L: torch.Tensor, S: torch.Tensor | None):
        inner_c = inner.contiguous()
        L_c = L.contiguous()
        if S is not None:
            S_c = S.contiguous()
            m, rn, a = L.shape
            n, r = S.shape[1], S.shape[2]
            L_eff = (L_c.view(m, n, r, a) * S_c.unsqueeze(-1)).view(m, rn, a)
        else:
            S_c = None
            L_eff = L_c
        out = torch.bmm(inner_c, L_eff)
        ctx.save_for_backward(inner_c, L_c, S_c if S is not None else torch.empty(0, device=inner.device))
        ctx.has_s = S is not None
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        inner, L, S_or_empty = ctx.saved_tensors
        has_s = ctx.has_s
        m, rn, a = L.shape
        if has_s:
            S = S_or_empty
            n, r = S.shape[1], S.shape[2]
            L_eff = (L.view(m, n, r, a) * S.unsqueeze(-1)).view(m, rn, a)
        else:
            S = None
            L_eff = L

        # d inner = grad_out @ L_eff^T
        grad_inner = torch.bmm(grad_out, L_eff.transpose(1, 2))
        # d L_eff = inner^T @ grad_out
        grad_L_eff = torch.bmm(inner.transpose(1, 2), grad_out)

        if has_s:
            # d L = grad_L_eff, reshaped and scaled by S broadcast
            grad_L = (grad_L_eff.view(m, n, r, a) * S.unsqueeze(-1)).view(m, rn, a)
            # d S = reduce grad_L_eff * L over the 'a' axis
            grad_S = (grad_L_eff.view(m, n, r, a) * L.view(m, n, r, a)).sum(dim=-1)
        else:
            grad_L = grad_L_eff
            grad_S = None
        return grad_inner, grad_L, grad_S


def step2_s_scaled_bmm(inner: torch.Tensor, L: torch.Tensor, S: torch.Tensor | None) -> torch.Tensor:
    """Public entry. Shapes:
      inner: (m, B, r*n)
      L:     (m, r*n, a)
      S:     (m, n, r) or None
    Returns: (m, B, a)
    """
    return _Step2SScaledBMM.apply(inner, L, S)
