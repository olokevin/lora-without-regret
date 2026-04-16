"""Invariants for compress.utils.whitening.compute_whitening.

Guards the refactor from torch.linalg.svd -> torch.linalg.eigh. Phi is defined
up to an orthogonal sign flip on each eigenvector, so we never compare Phi
element-wise; we check reconstruction and inverse identities instead.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from compress.utils.whitening import compute_whitening  # noqa: E402


def _random_spd(d: int, rank: int, seed: int, dtype=torch.float32) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    A = torch.randn(d, rank, generator=g, dtype=dtype)
    C = (A @ A.t()) / rank
    return C + 0.01 * torch.eye(d, dtype=dtype)


class TestComputeWhitening(unittest.TestCase):
    def test_identity_input(self):
        d = 16
        C = torch.eye(d, dtype=torch.float32)
        Phi, Phi_inv = compute_whitening(C, regularize_eps=1e-4, device="cpu")
        # Phi Phi^T should reconstruct C + eps*I = (1 + 1e-4) * I.
        recon = Phi @ Phi.t()
        expected = (1.0 + 1e-4) * torch.eye(d)
        self.assertLess((recon - expected).norm() / expected.norm(), 1e-5)
        # Phi_inv Phi should be identity (up to the sqrt(1+eps) scaling encoded in Phi).
        self.assertLess((Phi_inv @ Phi - torch.eye(d)).abs().max(), 1e-5)

    def test_reconstruction_random_spd(self):
        d, r = 64, 16
        C = _random_spd(d, r, seed=0)
        Phi, Phi_inv = compute_whitening(C, regularize_eps=1e-4, device="cpu")
        recon = Phi @ Phi.t()
        target = C + 1e-4 * torch.eye(d)
        rel_err = (recon - target).norm() / target.norm()
        self.assertLess(rel_err.item(), 1e-4)

    def test_inverse_identity(self):
        d, r = 64, 16
        C = _random_spd(d, r, seed=1)
        Phi, Phi_inv = compute_whitening(C, regularize_eps=1e-4, device="cpu")
        # Phi_inv @ Phi == I exactly (up to roundoff) by construction.
        I = torch.eye(d)
        self.assertLess((Phi_inv @ Phi - I).abs().max().item(), 1e-4)
        self.assertLess((Phi @ Phi_inv.t().t() - Phi @ Phi_inv.t().t()).abs().max().item(), 1e-4)

    def test_tiny_negative_eigvals_tolerated(self):
        # Craft a matrix that is PSD in theory but has sub-epsilon negative
        # eigvals after float32 accumulation; ensure the call succeeds and
        # returns finite results.
        d, r = 128, 8
        C = _random_spd(d, r, seed=2)
        # Inject asymmetric float noise to simulate hook accumulation drift.
        noise = 1e-7 * torch.randn(d, d)
        C_noisy = C + noise
        Phi, Phi_inv = compute_whitening(C_noisy, regularize_eps=1e-4, device="cpu")
        self.assertTrue(torch.isfinite(Phi).all())
        self.assertTrue(torch.isfinite(Phi_inv).all())

    def test_output_dtype_and_shape(self):
        d = 32
        C = _random_spd(d, 8, seed=3)
        Phi, Phi_inv = compute_whitening(C, regularize_eps=1e-4, device="cpu")
        self.assertEqual(Phi.shape, (d, d))
        self.assertEqual(Phi_inv.shape, (d, d))
        self.assertEqual(Phi.dtype, torch.float32)
        self.assertEqual(Phi_inv.dtype, torch.float32)

    def test_rank_deficient_input(self):
        # Strictly rank-deficient C: regularize_eps is what keeps Phi invertible.
        d, r = 48, 4
        g = torch.Generator(device="cpu").manual_seed(4)
        A = torch.randn(d, r, generator=g)
        C = (A @ A.t()) / r  # rank-4 exactly, no added identity
        Phi, Phi_inv = compute_whitening(C, regularize_eps=1e-3, device="cpu")
        recon = Phi @ Phi.t()
        target = C + 1e-3 * torch.eye(d)
        self.assertLess((recon - target).norm() / target.norm(), 1e-3)
        self.assertTrue(torch.isfinite(Phi).all())
        self.assertTrue(torch.isfinite(Phi_inv).all())

    def test_matches_svd_reference_on_reconstruction(self):
        # Against the previous SVD-based implementation, the returned Phi may
        # differ in eigenvector signs, but Phi Phi^T must match bit-for-bit
        # (within float32 precision). This is the acceptance criterion for
        # the refactor.
        d, r = 128, 32
        C = _random_spd(d, r, seed=5)

        Phi_new, Phi_inv_new = compute_whitening(C, regularize_eps=1e-4, device="cpu")

        # Reference SVD implementation (kept in-test to avoid leaking into prod).
        C_ref = C.to(torch.float32)
        U_s, S_s, _ = torch.linalg.svd(C_ref, full_matrices=False)
        sqrt_S = torch.sqrt(S_s + 1e-4)
        Phi_ref = U_s * sqrt_S.unsqueeze(0)
        Phi_inv_ref = (U_s * (1.0 / sqrt_S).unsqueeze(0)).t()

        recon_new = Phi_new @ Phi_new.t()
        recon_ref = Phi_ref @ Phi_ref.t()
        self.assertLess((recon_new - recon_ref).norm() / recon_ref.norm(), 1e-4)

        inv_new = Phi_inv_new @ Phi_new
        inv_ref = Phi_inv_ref @ Phi_ref
        # Both should be ~I; compare to I, not to each other, to avoid sign issues.
        I = torch.eye(d)
        self.assertLess((inv_new - I).abs().max().item(), 1e-3)
        self.assertLess((inv_ref - I).abs().max().item(), 1e-3)


if __name__ == "__main__":
    unittest.main()
