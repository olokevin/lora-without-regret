# tests/test_fura_fused_kernel.py
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import itertools
import pytest
import torch

# Skip if triton not importable — but we don't actually use it in Task 10
# Keep this for forward-compatibility when Task 11 adds the real kernel


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    return "cuda"


def _ref_step2(inner: torch.Tensor, L: torch.Tensor, S: torch.Tensor | None) -> torch.Tensor:
    """Reference implementation."""
    m, _, rn = inner.shape
    if S is not None:
        n, r = S.shape[1], S.shape[2]
        a = L.shape[-1]
        L_eff = (L.reshape(m, n, r, a) * S.unsqueeze(-1)).reshape(m, r * n, a)
    else:
        L_eff = L
    return torch.bmm(inner, L_eff)


@pytest.mark.parametrize(
    "m,n,r,a,B,has_s",
    list(itertools.product([4, 64], [4, 64], [4, 64], [8, 64], [1, 64], [True, False])),
)
def test_fused_matches_reference(m, n, r, a, B, has_s, device):
    from fura_kernels.triton_btt import step2_s_scaled_bmm
    rn = r * n
    inner = torch.randn(m, B, rn, device=device, dtype=torch.bfloat16)
    L = torch.randn(m, rn, a, device=device, dtype=torch.bfloat16)
    S = torch.randn(m, n, r, device=device, dtype=torch.bfloat16) if has_s else None
    ref = _ref_step2(inner, L, S)
    fused = step2_s_scaled_bmm(inner, L, S)
    assert ref.shape == fused.shape
    assert torch.allclose(ref.float(), fused.float(), atol=1e-2, rtol=1e-2), \
        f"max diff = {(ref.float() - fused.float()).abs().max().item()}"


def test_fused_backward_matches_reference(device):
    from fura_kernels.triton_btt import step2_s_scaled_bmm
    torch.manual_seed(0)
    m, n, r, a, B = 2, 4, 4, 4, 2
    rn = r * n
    inner_a = torch.randn(m, B, rn, device=device, dtype=torch.float32, requires_grad=True)
    inner_b = inner_a.detach().clone().requires_grad_(True)
    L_a = torch.randn(m, rn, a, device=device, dtype=torch.float32, requires_grad=True)
    L_b = L_a.detach().clone().requires_grad_(True)
    S_a = torch.randn(m, n, r, device=device, dtype=torch.float32, requires_grad=True)
    S_b = S_a.detach().clone().requires_grad_(True)

    _ref_step2(inner_a, L_a, S_a).sum().backward()
    step2_s_scaled_bmm(inner_b, L_b, S_b).sum().backward()

    for ga, gb, name in [(inner_a.grad, inner_b.grad, "inner"),
                         (L_a.grad, L_b.grad, "L"),
                         (S_a.grad, S_b.grad, "S")]:
        assert torch.allclose(ga, gb, atol=1e-4, rtol=1e-4), \
            f"gradient mismatch on {name}"


def test_non_contiguous_inputs_are_handled(device):
    from fura_kernels.triton_btt import step2_s_scaled_bmm
    m, n, r, a, B = 2, 4, 4, 4, 4
    rn = r * n
    inner_full = torch.randn(m, 2 * B, rn, device=device, dtype=torch.bfloat16)
    inner = inner_full[:, ::2, :]  # non-contiguous stride
    L = torch.randn(m, rn, a, device=device, dtype=torch.bfloat16)
    S = torch.randn(m, n, r, device=device, dtype=torch.bfloat16)
    ref = _ref_step2(inner.contiguous(), L, S)
    fused = step2_s_scaled_bmm(inner, L, S)
    assert torch.allclose(ref.float(), fused.float(), atol=1e-2, rtol=1e-2)
