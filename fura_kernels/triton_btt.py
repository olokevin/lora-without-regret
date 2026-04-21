"""FuRA Step-2 fused kernel.

Step 2 of FuRA's forward is: out = bmm(inner, L_eff) where
  inner:  (m, B, r*n)
  L:      (m, r*n, a)
  S:      (m, n, r)  (optional, diagonal per-block singular scale)
  L_eff:  L with S folded in, shape (m, r*n, a).

The eager path in btt_layer.py materialises L_eff as a fresh tensor every
forward (m * n * r * a floats). This kernel folds S into the GEMM epilogue
so that materialisation is avoided.

Forward: Triton kernel when on CUDA with large enough shapes, otherwise
pure-PyTorch fallback.  Backward: always torch.bmm (unchanged).
"""
from __future__ import annotations
import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------
if _HAS_TRITON:
    @triton.jit
    def _step2_s_scaled_kernel(
        INNER_ptr, L_ptr, S_ptr, OUT_ptr,
        M, B, RN, A, N, R,
        s_im, s_ib, s_irn,         # inner strides
        s_lm, s_lrn, s_la,         # L strides
        s_sm, s_sn, s_sr,          # S strides
        s_om, s_ob, s_oa,          # OUT strides
        BLOCK_B: tl.constexpr,
        BLOCK_A: tl.constexpr,
        BLOCK_K: tl.constexpr,     # chunk over RN (= R*N)
        HAS_S: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_b = tl.program_id(1)
        pid_a = tl.program_id(2)

        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        offs_a = pid_a * BLOCK_A + tl.arange(0, BLOCK_A)

        acc = tl.zeros((BLOCK_B, BLOCK_A), dtype=tl.float32)

        for k0 in range(0, RN, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < RN

            # inner tile: (BLOCK_B, BLOCK_K)
            inner_ptrs = (
                INNER_ptr
                + pid_m * s_im
                + offs_b[:, None] * s_ib
                + offs_k[None, :] * s_irn
            )
            inner_tile = tl.load(
                inner_ptrs,
                mask=(offs_b[:, None] < B) & mask_k[None, :],
                other=0.0,
            )

            # L tile: (BLOCK_K, BLOCK_A)
            L_ptrs = (
                L_ptr
                + pid_m * s_lm
                + offs_k[:, None] * s_lrn
                + offs_a[None, :] * s_la
            )
            L_tile = tl.load(
                L_ptrs,
                mask=(mask_k[:, None]) & (offs_a[None, :] < A),
                other=0.0,
            )

            if HAS_S:
                # L.view(m, n, r, a) means rn index k maps to:
                #   n_idx = k // r,  r_idx = k % r
                n_idx = offs_k // R
                r_idx = offs_k % R
                S_ptrs = (
                    S_ptr
                    + pid_m * s_sm
                    + n_idx * s_sn
                    + r_idx * s_sr
                )
                s_vec = tl.load(S_ptrs, mask=mask_k, other=0.0)  # (BLOCK_K,)
                L_tile = L_tile * s_vec[:, None]

            acc += tl.dot(inner_tile.to(tl.float32), L_tile.to(tl.float32))

        # Store output
        out_ptrs = (
            OUT_ptr
            + pid_m * s_om
            + offs_b[:, None] * s_ob
            + offs_a[None, :] * s_oa
        )
        tl.store(
            out_ptrs,
            acc.to(OUT_ptr.dtype.element_ty),
            mask=(offs_b[:, None] < B) & (offs_a[None, :] < A),
        )


# ---------------------------------------------------------------------------
# Python launcher for the Triton kernel
# ---------------------------------------------------------------------------
def _triton_step2(inner: torch.Tensor, L: torch.Tensor, S: torch.Tensor | None) -> torch.Tensor:
    assert inner.is_cuda and L.is_cuda
    m, B, rn = inner.shape
    _, _, a = L.shape
    if S is not None:
        n, r = S.shape[1], S.shape[2]
        assert r * n == rn, f"S shape ({n},{r}) inconsistent with rn={rn}"
    else:
        n = r = 1

    inner = inner.contiguous()
    L = L.contiguous()
    if S is not None:
        S = S.contiguous()

    out = torch.empty((m, B, a), device=inner.device, dtype=inner.dtype)

    BLOCK_B = 64 if B >= 64 else max(16, triton.next_power_of_2(B))
    BLOCK_A = 64 if a >= 64 else max(16, triton.next_power_of_2(a))
    BLOCK_K = 64 if rn >= 64 else max(16, triton.next_power_of_2(rn))

    grid = (m, triton.cdiv(B, BLOCK_B), triton.cdiv(a, BLOCK_A))
    _step2_s_scaled_kernel[grid](
        inner, L, (S if S is not None else inner), out,
        m, B, rn, a, n, r,
        inner.stride(0), inner.stride(1), inner.stride(2),
        L.stride(0), L.stride(1), L.stride(2),
        (S.stride(0) if S is not None else 0),
        (S.stride(1) if S is not None else 0),
        (S.stride(2) if S is not None else 0),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_B=BLOCK_B, BLOCK_A=BLOCK_A, BLOCK_K=BLOCK_K,
        HAS_S=(S is not None),
    )
    return out


# ---------------------------------------------------------------------------
# Autograd wrapper
# ---------------------------------------------------------------------------
class _Step2SScaledBMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inner: torch.Tensor, L: torch.Tensor, S: torch.Tensor | None):
        inner_c = inner.contiguous()
        L_c = L.contiguous()
        S_c = S.contiguous() if S is not None else None

        use_triton = (
            _HAS_TRITON
            and inner.is_cuda
            and inner.shape[-1] >= 16
            and L.shape[-1] >= 16
            and inner.shape[1] >= 16
        )
        if use_triton:
            out = _triton_step2(inner_c, L_c, S_c)
        else:
            m, rn, a = L.shape
            if S is not None:
                n, r = S.shape[1], S.shape[2]
                L_eff = (L_c.view(m, n, r, a) * S_c.unsqueeze(-1)).view(m, rn, a)
            else:
                L_eff = L_c
            out = torch.bmm(inner_c, L_eff)

        ctx.save_for_backward(
            inner_c, L_c,
            S_c if S is not None else torch.empty(0, device=inner.device),
        )
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
