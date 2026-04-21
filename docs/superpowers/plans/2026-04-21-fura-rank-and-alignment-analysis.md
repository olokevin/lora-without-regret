# FuRA Rank and Alignment Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the paper-motivation 3×3 figure plus a long-form data-notes markdown from three existing Qwen3-1.7B RL checkpoints (Full FT, FuRA-blocktt, SVD), with per-layer stable rank, U/V-subspace alignment trajectory, and per-pretrained-direction update-energy heatmaps.

**Architecture:** Two entry points share a metric library and a checkpoint loader. `analyze_rank_alignment.py` does the expensive SVD sweep and writes per-method JSON caches. `plot_rank_alignment_3x3.py` consumes those caches to render the PDF figure and the auto-generated notes markdown. Analysis-only — no training code is touched.

**Tech Stack:** Python 3.13 / uv, PyTorch (with CUDA for SVD), NumPy, matplotlib, safetensors, pytest. No new third-party deps.

**Spec:** `docs/superpowers/specs/2026-04-21-fura-rank-and-alignment-analysis.md`

**Reference paths (read-only):**
- Base model: HuggingFace `Qwen/Qwen3-1.7B` (resolved via `huggingface_hub.snapshot_download`)
- Full FT ckpt: `/data/yequan/fura/rl_runs/full/full-adamw-lr_2e-5-0325-215533/` (step=1, 10, 50)
- FuRA blocktt ckpt: `/data/yequan/fura/rl_runs/blocktt/blocktt-adamw-lr_1e-4-output_one_block-s_to_frozen-train_small-0317-150342/` (step=1, 10, 50)
- SVD ckpt: `/data/yequan/fura/rl_runs/svd/svd-adamw-lr_1e-5-s_to_keep-train_input-0317-141139/` (step=1, 10, 50)
- Existing reusable helpers: `analysis/analyze_weights.py:materialize_blocktt_weight` (lines 12–40), `materialize_svd_weight` (lines 43–58), `get_base_weight_key`, `get_checkpoint_keys`, `load_safetensors_index`, `load_tensor`, `detect_num_layers`.
- SVD parameterisation confirmed from `svd_layer.py:SVDLayer.materialize_dense_weight` (line 132): **replacement** — `W = svd_a @ svd_b` or `(svd_a * svd_s.unsqueeze(0)) @ svd_b`. No base addition.

---

## File Structure

**New files:**
- `analysis/rank_alignment_metrics.py` — pure metric functions (no I/O)
- `analysis/rank_alignment_loader.py` — base-SVD cache + per-method ΔW materialisation
- `analysis/analyze_rank_alignment.py` — entry point: metric sweep → JSON caches
- `analysis/plot_rank_alignment_3x3.py` — entry point: caches → PDF + notes markdown
- `tests/test_rank_alignment_metrics.py` — pytest unit tests for metric functions
- `tests/test_rank_alignment_loader.py` — pytest tests for the loader (synthetic tensors)

**Modified files:** none.

**Output paths (not committed):**
- Per-method caches: `analysis_cache/rank_alignment/{full,blocktt,svd}.json`
- Base-SVD cache: `analysis_cache/rank_alignment/base_svd.safetensors`
- Per-ckpt ΔW caches: `analysis_cache/rank_alignment/<method>_step<N>_deltas.safetensors`

**Output paths (committed):**
- Figure: `docs/26_nips_fura_paper/figs/rank_and_alignment_3x3.pdf` + `.png`
- Data notes: `docs/26_nips_fura_paper/rank_and_alignment_notes.md`
- Figure-data JSON: `docs/26_nips_fura_paper/figs/rank_and_alignment_3x3.data.json`

---

## Task 1: Metric library — stable rank + friends

**Files:**
- Create: `analysis/rank_alignment_metrics.py`
- Test: `tests/test_rank_alignment_metrics.py`

- [ ] **Step 1: Write the failing test for `stable_rank`**

Create `tests/test_rank_alignment_metrics.py`:

```python
"""Unit tests for rank_alignment_metrics — all synthetic tensors, no real ckpts."""

import math
import numpy as np
import pytest
import torch

from analysis.rank_alignment_metrics import (
    align_energy_topk,
    align_energy_u,
    align_energy_v,
    entropy_rank,
    per_direction_energy,
    principal_mass_fractions,
    random_baseline_align,
    stable_rank,
    threshold_rank,
)


def test_stable_rank_identity_matrix_is_n():
    I = torch.eye(32, dtype=torch.float64)
    assert stable_rank(I) == pytest.approx(32.0)


def test_stable_rank_rank1_matrix_is_one():
    u = torch.randn(32, dtype=torch.float64)
    v = torch.randn(48, dtype=torch.float64)
    W = torch.outer(u, v)
    assert stable_rank(W) == pytest.approx(1.0, rel=1e-6)


def test_stable_rank_zero_matrix_returns_nan():
    Z = torch.zeros(16, 16)
    result = stable_rank(Z)
    assert math.isnan(result)
```

- [ ] **Step 2: Run tests — verify they fail with ImportError**

Run: `uv run python -m pytest tests/test_rank_alignment_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'analysis.rank_alignment_metrics'`

- [ ] **Step 3: Implement `stable_rank` + file skeleton**

Create `analysis/rank_alignment_metrics.py`:

```python
"""Pure metric functions for rank + alignment analysis.

All functions take torch tensors and return scalars or numpy arrays.
No file I/O, no globals, no side effects — easy to unit-test.
"""

from typing import Literal

import numpy as np
import torch


_ZERO_THRESHOLD = 1e-8


def _svdvals_fp32(W: torch.Tensor) -> torch.Tensor:
    """Singular values of W in fp32 on the tensor's own device, descending."""
    return torch.linalg.svdvals(W.float())


def stable_rank(delta_W: torch.Tensor) -> float:
    """Stable rank: ‖ΔW‖_F² / ‖ΔW‖_2². NaN for zero matrices."""
    if delta_W.norm() < _ZERO_THRESHOLD:
        return float("nan")
    sigma = _svdvals_fp32(delta_W)
    if sigma[0] < _ZERO_THRESHOLD:
        return float("nan")
    return float((sigma ** 2).sum() / (sigma[0] ** 2))
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `uv run python -m pytest tests/test_rank_alignment_metrics.py -v`
Expected: 3 passed.

- [ ] **Step 5: Write tests for `entropy_rank` and `threshold_rank`**

Append to `tests/test_rank_alignment_metrics.py`:

```python
def test_entropy_rank_identity_matrix_is_n():
    I = torch.eye(16, dtype=torch.float64)
    assert entropy_rank(I) == pytest.approx(16.0, rel=1e-6)


def test_entropy_rank_rank1_matrix_is_one():
    u = torch.randn(16, dtype=torch.float64)
    v = torch.randn(16, dtype=torch.float64)
    W = torch.outer(u, v)
    assert entropy_rank(W) == pytest.approx(1.0, abs=1e-6)


def test_threshold_rank_identity_is_n():
    I = torch.eye(16, dtype=torch.float64)
    assert threshold_rank(I, frac=0.01) == 16


def test_threshold_rank_decaying_spectrum():
    # Spectrum 1, 0.5, 0.1, 0.005 — threshold 0.01 of max (=0.01) keeps first 3
    U = torch.eye(4, dtype=torch.float64)
    s = torch.tensor([1.0, 0.5, 0.1, 0.005], dtype=torch.float64)
    W = U * s.unsqueeze(0)
    assert threshold_rank(W, frac=0.01) == 3


def test_threshold_rank_zero_matrix_is_zero():
    Z = torch.zeros(8, 8)
    assert threshold_rank(Z, frac=0.01) == 0
```

- [ ] **Step 6: Run tests — verify failure**

Run: `uv run python -m pytest tests/test_rank_alignment_metrics.py -v`
Expected: 3 pass, 4 fail with `ImportError: cannot import name 'entropy_rank' ...`.

- [ ] **Step 7: Implement `entropy_rank` and `threshold_rank`**

Append to `analysis/rank_alignment_metrics.py`:

```python
def entropy_rank(delta_W: torch.Tensor) -> float:
    """Exponential of spectral entropy: exp(-Σ p_i log p_i) where p_i = σ_i²/Σσ_j²."""
    if delta_W.norm() < _ZERO_THRESHOLD:
        return float("nan")
    sigma = _svdvals_fp32(delta_W)
    energy = sigma ** 2
    total = energy.sum()
    if total < _ZERO_THRESHOLD:
        return float("nan")
    p = energy / total
    # Guard log(0): drop zero-probability entries
    p_nz = p[p > 0]
    H = -(p_nz * p_nz.log()).sum()
    return float(H.exp())


def threshold_rank(delta_W: torch.Tensor, frac: float = 0.01) -> int:
    """Number of singular values σ_i ≥ frac · σ_1. Returns 0 for zero matrices."""
    if delta_W.norm() < _ZERO_THRESHOLD:
        return 0
    sigma = _svdvals_fp32(delta_W)
    if sigma[0] < _ZERO_THRESHOLD:
        return 0
    return int((sigma >= frac * sigma[0]).sum().item())
```

- [ ] **Step 8: Run tests — verify pass**

Run: `uv run python -m pytest tests/test_rank_alignment_metrics.py -v`
Expected: 7 passed.

- [ ] **Step 9: Commit**

```bash
git add analysis/rank_alignment_metrics.py tests/test_rank_alignment_metrics.py
git commit -m "analysis: add stable/entropy/threshold rank metrics"
```

---

## Task 2: Metric library — U/V alignment energies

**Files:**
- Modify: `analysis/rank_alignment_metrics.py`
- Modify: `tests/test_rank_alignment_metrics.py`

- [ ] **Step 1: Write alignment tests (in-subspace, orthogonal, mixture, baseline)**

Append to `tests/test_rank_alignment_metrics.py`:

```python
def _thin_svd(W):
    U, _, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    return U, Vh.T  # V, not V^T


def test_align_energy_u_in_subspace_is_one():
    # ΔW constructed to live entirely in col(U0)
    W_base = torch.randn(48, 32, dtype=torch.float64)
    U0, V0 = _thin_svd(W_base)
    # delta = U0 @ X — X: (r0, d_in)
    X = torch.randn(U0.shape[1], 32, dtype=torch.float64)
    delta = U0 @ X
    assert align_energy_u(delta, U0) == pytest.approx(1.0, abs=1e-6)


def test_align_energy_u_orthogonal_to_subspace_is_zero():
    # For tall W_base (48×32), col(U0) has dim 32; the 16-dim orthogonal
    # complement of col(U0) in R^48 gives a ΔW with zero U-side alignment.
    W_base = torch.randn(48, 32, dtype=torch.float64)
    U0, _ = _thin_svd(W_base)               # U0: (48, 32)
    # Orthonormal basis Q of R^48 = [U0 | U_perp]; use tail 16 cols of Q as "outside"
    Q, _ = torch.linalg.qr(torch.cat([U0, torch.randn(48, 16, dtype=torch.float64)], dim=1))
    U_perp = Q[:, 32:]                       # (48, 16), orthogonal to span(U0)
    X = torch.randn(U_perp.shape[1], 32, dtype=torch.float64)
    delta = U_perp @ X
    assert align_energy_u(delta, U0) == pytest.approx(0.0, abs=1e-6)


def test_align_energy_u_mixture_matches_formula():
    W_base = torch.randn(48, 32, dtype=torch.float64)
    U0, _ = _thin_svd(W_base)
    Q, _ = torch.linalg.qr(torch.cat([U0, torch.randn(48, 16, dtype=torch.float64)], dim=1))
    U_perp = Q[:, 32:]
    A = U0 @ torch.randn(U0.shape[1], 32, dtype=torch.float64)
    B = U_perp @ torch.randn(U_perp.shape[1], 32, dtype=torch.float64)
    delta = 2.0 * A + 3.0 * B
    expected = (2.0 * A).norm() ** 2 / delta.norm() ** 2
    assert align_energy_u(delta, U0) == pytest.approx(float(expected), abs=1e-6)


def test_align_energy_v_symmetric_to_u():
    # W shape (32, 48) so V0 is (48, 32) and the 16-dim outside lives on the V side
    W_base = torch.randn(32, 48, dtype=torch.float64)
    U0, V0 = _thin_svd(W_base)
    # delta in row(V0): delta = X @ V0^T
    X = torch.randn(32, V0.shape[1], dtype=torch.float64)
    delta = X @ V0.T
    assert align_energy_v(delta, V0) == pytest.approx(1.0, abs=1e-6)


def test_align_energy_zero_delta_returns_nan():
    U0 = torch.eye(32, dtype=torch.float64)[:, :16]
    Z = torch.zeros(32, 16)
    assert math.isnan(align_energy_u(Z, U0))


def test_random_baseline_u_side_formula():
    # For a (6144, 2048) gate/up layer, U-side baseline is 2048/6144 = 1/3
    baseline = random_baseline_align(d_out=6144, d_in=2048, side="U")
    assert baseline == pytest.approx(2048 / 6144)


def test_random_baseline_v_side_formula():
    baseline = random_baseline_align(d_out=2048, d_in=6144, side="V")
    assert baseline == pytest.approx(2048 / 6144)


def test_align_energy_topk_matches_full_when_k_equals_r0():
    W_base = torch.randn(48, 32, dtype=torch.float64)
    U0, V0 = _thin_svd(W_base)
    delta = torch.randn(48, 32, dtype=torch.float64)
    full_u = align_energy_u(delta, U0)
    topk_u = align_energy_topk(delta, U0, k=U0.shape[1], side="U")
    assert topk_u == pytest.approx(full_u, abs=1e-6)


def test_align_energy_topk_k1_picks_principal_direction():
    # Build delta aligned entirely with the first principal direction of U0
    W_base = torch.randn(48, 32, dtype=torch.float64)
    U0, _ = _thin_svd(W_base)
    x = torch.randn(32, dtype=torch.float64)
    delta = torch.outer(U0[:, 0], x)
    assert align_energy_topk(delta, U0, k=1, side="U") == pytest.approx(1.0, abs=1e-6)


def test_random_baseline_monte_carlo_matches_analytic():
    # Random i.i.d. ΔW should empirically hit r0 / d_out for U-side projection.
    torch.manual_seed(0)
    d_out, d_in, trials = 128, 64, 50
    W_base = torch.randn(d_out, d_in, dtype=torch.float64)
    U0, _ = _thin_svd(W_base)
    ratios = []
    for _ in range(trials):
        delta = torch.randn(d_out, d_in, dtype=torch.float64)
        ratios.append(align_energy_u(delta, U0))
    empirical = sum(ratios) / len(ratios)
    analytic = random_baseline_align(d_out=d_out, d_in=d_in, side="U")
    assert empirical == pytest.approx(analytic, abs=0.02)
```

- [ ] **Step 2: Run tests — verify failure**

Run: `uv run python -m pytest tests/test_rank_alignment_metrics.py -v`
Expected: first 7 pass, 10 new alignment/baseline/topk tests fail with ImportError.

- [ ] **Step 3: Implement alignment functions**

Append to `analysis/rank_alignment_metrics.py`:

```python
def align_energy_u(delta_W: torch.Tensor, U0: torch.Tensor) -> float:
    """‖U0 U0ᵀ ΔW‖_F² / ‖ΔW‖_F². Returns NaN for zero ΔW."""
    if delta_W.norm() < _ZERO_THRESHOLD:
        return float("nan")
    projected = U0 @ (U0.T @ delta_W.float())
    denom = (delta_W.float() ** 2).sum()
    if denom < _ZERO_THRESHOLD:
        return float("nan")
    return float((projected ** 2).sum() / denom)


def align_energy_v(delta_W: torch.Tensor, V0: torch.Tensor) -> float:
    """‖ΔW V0 V0ᵀ‖_F² / ‖ΔW‖_F². Returns NaN for zero ΔW."""
    if delta_W.norm() < _ZERO_THRESHOLD:
        return float("nan")
    projected = (delta_W.float() @ V0) @ V0.T
    denom = (delta_W.float() ** 2).sum()
    if denom < _ZERO_THRESHOLD:
        return float("nan")
    return float((projected ** 2).sum() / denom)


def align_energy_topk(
    delta_W: torch.Tensor,
    U0_or_V0: torch.Tensor,
    k: int,
    side: Literal["U", "V"],
) -> float:
    """Alignment using truncated basis (first `k` columns)."""
    basis_k = U0_or_V0[:, :k]
    if side == "U":
        return align_energy_u(delta_W, basis_k)
    if side == "V":
        return align_energy_v(delta_W, basis_k)
    raise ValueError(f"side must be 'U' or 'V', got {side!r}")


def random_baseline_align(d_out: int, d_in: int, side: Literal["U", "V"]) -> float:
    """Expected alignment ratio for a random i.i.d. ΔW.

    For a matrix with no preference for col(W_base) (U-side) or row(W_base)
    (V-side), the expected projection-fraction is r0 / (d_out or d_in)
    depending on which side we project.
    """
    r0 = min(d_out, d_in)
    if side == "U":
        return r0 / d_out
    if side == "V":
        return r0 / d_in
    raise ValueError(f"side must be 'U' or 'V', got {side!r}")
```

- [ ] **Step 4: Run tests — verify pass**

Run: `uv run python -m pytest tests/test_rank_alignment_metrics.py -v`
Expected: 17 passed.

- [ ] **Step 5: Commit**

```bash
git add analysis/rank_alignment_metrics.py tests/test_rank_alignment_metrics.py
git commit -m "analysis: add U/V alignment energies + topk + random baseline"
```

---

## Task 3: Metric library — per-direction energy and mass fractions

**Files:**
- Modify: `analysis/rank_alignment_metrics.py`
- Modify: `tests/test_rank_alignment_metrics.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_rank_alignment_metrics.py`:

```python
def test_per_direction_energy_diagonal_delta_is_normalised_squared_c():
    W_base = torch.randn(48, 32, dtype=torch.float64)
    U0, V0 = _thin_svd(W_base)
    c = torch.tensor([1.0, 0.5, 2.0, 0.1] + [0.0] * (U0.shape[1] - 4), dtype=torch.float64)
    delta = U0 @ torch.diag(c) @ V0.T
    energy = per_direction_energy(delta, U0, V0)
    expected = (c ** 2) / (c ** 2).sum()
    np.testing.assert_allclose(energy, expected.numpy(), atol=1e-6)


def test_per_direction_energy_sum_is_one_when_ortho_basis_spans():
    # For square non-degenerate W_base, U0 and V0 are full orthogonal bases;
    # then A = U0^T ΔW V0 is square and Σ |A_ii|² / Σ_j |A_jj|² sums to 1.
    W_base = torch.randn(16, 16, dtype=torch.float64)
    U0, V0 = _thin_svd(W_base)
    delta = torch.randn(16, 16, dtype=torch.float64)
    energy = per_direction_energy(delta, U0, V0)
    assert energy.sum() == pytest.approx(1.0, abs=1e-6)


def test_per_direction_energy_zero_delta_returns_zeros():
    U0 = torch.eye(32, dtype=torch.float64)[:, :16]
    V0 = torch.eye(32, dtype=torch.float64)[:, :16]
    Z = torch.zeros(32, 32)
    energy = per_direction_energy(Z, U0, V0)
    assert energy.shape == (16,)
    np.testing.assert_allclose(energy, np.zeros(16))


def test_principal_mass_fractions_uniform_energy_is_thirds():
    # Uniform energy across r0=12 directions: top_quarter=3, mid_half=6, tail_quarter=3
    # Fractions: 3/12, 6/12, 3/12 = 0.25, 0.5, 0.25
    energy = np.ones(12) / 12.0
    top, mid, tail = principal_mass_fractions(energy, r0=12)
    assert top == pytest.approx(0.25)
    assert mid == pytest.approx(0.5)
    assert tail == pytest.approx(0.25)


def test_principal_mass_fractions_top_heavy():
    # All mass in first direction — top_quarter = 1.0, mid = 0, tail = 0
    energy = np.zeros(12)
    energy[0] = 1.0
    top, mid, tail = principal_mass_fractions(energy, r0=12)
    assert top == pytest.approx(1.0)
    assert mid == pytest.approx(0.0)
    assert tail == pytest.approx(0.0)
```

- [ ] **Step 2: Run tests — verify failure**

Run: `uv run python -m pytest tests/test_rank_alignment_metrics.py -v`
Expected: 17 pass, 5 new tests fail.

- [ ] **Step 3: Implement `per_direction_energy` and `principal_mass_fractions`**

Append to `analysis/rank_alignment_metrics.py`:

```python
def per_direction_energy(
    delta_W: torch.Tensor,
    U0: torch.Tensor,
    V0: torch.Tensor,
) -> np.ndarray:
    """Normalised squared diagonal of U0^T ΔW V0.

    Returns a numpy array of shape (r0,) with entries e_i = d_i² / Σ_j d_j²
    where d_i = (U0^T ΔW V0)_{ii}. Zeros for zero ΔW.
    """
    r0 = min(U0.shape[1], V0.shape[1])
    if delta_W.norm() < _ZERO_THRESHOLD:
        return np.zeros(r0)
    A = U0[:, :r0].T @ delta_W.float() @ V0[:, :r0]
    d = torch.diagonal(A).pow(2)
    denom = d.sum()
    if denom < _ZERO_THRESHOLD:
        return np.zeros(r0)
    return (d / denom).cpu().numpy()


def principal_mass_fractions(
    energy: np.ndarray,
    r0: int,
) -> tuple[float, float, float]:
    """Partition normalised per-direction energy into (top 25%, mid 50%, tail 25%)."""
    q = r0 // 4
    top = float(energy[:q].sum())
    mid = float(energy[q : r0 - q].sum())
    tail = float(energy[r0 - q :].sum())
    return top, mid, tail
```

- [ ] **Step 4: Run tests — verify pass**

Run: `uv run python -m pytest tests/test_rank_alignment_metrics.py -v`
Expected: 22 passed.

- [ ] **Step 5: Commit**

```bash
git add analysis/rank_alignment_metrics.py tests/test_rank_alignment_metrics.py
git commit -m "analysis: add per-direction energy and principal-mass fractions"
```

---

## Task 4: Loader — base-model SVD cache

**Files:**
- Create: `analysis/rank_alignment_loader.py`
- Test: `tests/test_rank_alignment_loader.py`

Target modules and their layer path prefixes (reused across the project; see `analysis/analyze_weights.py:84-94`):
- Attention: `q_proj, k_proj, v_proj, o_proj` under `model.layers.{i}.self_attn`
- MLP: `gate_proj, up_proj, down_proj` under `model.layers.{i}.mlp`

- [ ] **Step 1: Write the failing loader test**

Create `tests/test_rank_alignment_loader.py`:

```python
"""Unit tests for the base-SVD loader (synthetic ckpts, no real model)."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from analysis.rank_alignment_loader import (
    TARGET_MODULES,
    load_base_svd,
    load_delta_w,
)


def _make_fake_base_model(tmp_path: Path, num_layers: int = 2):
    """Create a directory with a minimal fake Qwen3-style safetensors file."""
    tensors = {}
    for i in range(num_layers):
        # Attention: (64, 64)
        for m in ("q_proj", "k_proj", "v_proj", "o_proj"):
            tensors[f"model.layers.{i}.self_attn.{m}.weight"] = torch.randn(64, 64)
        # MLP shapes echo Qwen3-1.7B ratio 3:1 — up/gate 192x64, down 64x192
        tensors[f"model.layers.{i}.mlp.gate_proj.weight"] = torch.randn(192, 64)
        tensors[f"model.layers.{i}.mlp.up_proj.weight"] = torch.randn(192, 64)
        tensors[f"model.layers.{i}.mlp.down_proj.weight"] = torch.randn(64, 192)
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["Qwen3ForCausalLM"]}))
    save_file(tensors, str(tmp_path / "model.safetensors"))
    return tmp_path


def test_target_modules_list():
    expected = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    assert TARGET_MODULES == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="SVD runs on CUDA")
def test_load_base_svd_returns_u_v_per_module(tmp_path):
    base_dir = _make_fake_base_model(tmp_path / "base", num_layers=2)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    result = load_base_svd(str(base_dir), cache_dir=cache_dir, device=torch.device("cuda"))
    # 2 layers × 7 modules = 14 entries
    assert len(result) == 14
    for (layer, module), (U0, V0) in result.items():
        assert isinstance(layer, int) and module in TARGET_MODULES
        assert U0.device.type == "cuda"
        assert V0.device.type == "cuda"
        # SVD shape invariants
        assert U0.shape[1] == V0.shape[1]  # both columns = r0 = min(d_out, d_in)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="SVD runs on CUDA")
def test_load_base_svd_uses_cache(tmp_path):
    base_dir = _make_fake_base_model(tmp_path / "base", num_layers=1)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    _ = load_base_svd(str(base_dir), cache_dir=cache_dir, device=torch.device("cuda"))
    assert (cache_dir / "base_svd.safetensors").exists()
    # Second call should load from cache — tested by deleting the safetensors in base_dir
    (tmp_path / "base" / "model.safetensors").unlink()
    result2 = load_base_svd(str(base_dir), cache_dir=cache_dir, device=torch.device("cuda"))
    assert len(result2) == 7
```

- [ ] **Step 2: Run tests — verify failure**

Run: `uv run python -m pytest tests/test_rank_alignment_loader.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the loader module skeleton + `load_base_svd`**

Create `analysis/rank_alignment_loader.py`:

```python
"""Load Qwen3 base-model SVDs and per-method checkpoint ΔW tensors."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from analysis.analyze_weights import (
    detect_num_layers,
    get_base_weight_key,
    get_checkpoint_keys,
    load_safetensors_index,
    load_tensor,
    materialize_blocktt_weight,
    materialize_svd_weight,
)


TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def _base_svd_cache_path(cache_dir: Path) -> Path:
    return cache_dir / "base_svd.safetensors"


def _resolve_base_dir(base_model: str) -> str:
    """If `base_model` is a directory, return it; else snapshot_download from HF."""
    p = Path(base_model)
    if p.is_dir():
        return str(p)
    from huggingface_hub import snapshot_download

    return snapshot_download(base_model)


def load_base_svd(
    base_model: str,
    cache_dir: Path,
    device: torch.device,
) -> dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]]:
    """SVD every target-module W_base; cache the (U0, V0) tensors on disk.

    Returns a dict mapping (layer_idx, module_name) -> (U0, V0) where U0 is
    (d_out, r0) and V0 is (d_in, r0) with r0 = min(d_out, d_in).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _base_svd_cache_path(cache_dir)

    # Determine num_layers from the base model directory (or cache if present)
    base_dir = _resolve_base_dir(base_model)
    base_index = load_safetensors_index(base_dir)
    num_layers = detect_num_layers(base_index)

    # Fast path: cache hit
    if cache_path.exists():
        cache = load_file(str(cache_path), device=str(device))
        result: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
        for layer in range(num_layers):
            for module in TARGET_MODULES:
                u_key = f"L{layer}.{module}.U0"
                v_key = f"L{layer}.{module}.V0"
                if u_key in cache and v_key in cache:
                    result[(layer, module)] = (cache[u_key], cache[v_key])
        if len(result) == num_layers * len(TARGET_MODULES):
            return result
        # Partial cache is treated as stale — fall through to recompute
        print(f"Base-SVD cache incomplete ({len(result)} entries); recomputing.")

    # Slow path: compute SVDs
    print(f"Computing base-SVDs for {num_layers} layers × {len(TARGET_MODULES)} modules on {device}...")
    result = {}
    tensors_to_save: dict[str, torch.Tensor] = {}
    for layer in range(num_layers):
        for module in TARGET_MODULES:
            key = get_base_weight_key(layer, module)
            W = load_tensor(base_index, key)
            if W is None:
                print(f"  warning: missing {key}, skipping")
                continue
            W = W.to(device=device).float()
            U, _, Vh = torch.linalg.svd(W, full_matrices=False)
            V = Vh.T.contiguous()
            result[(layer, module)] = (U, V)
            tensors_to_save[f"L{layer}.{module}.U0"] = U.cpu()
            tensors_to_save[f"L{layer}.{module}.V0"] = V.cpu()
        print(f"  layer {layer}/{num_layers - 1} done")

    save_file(tensors_to_save, str(cache_path))
    # Move back to device for return value
    return {k: (u.to(device), v.to(device)) for k, (u, v) in result.items()}
```

- [ ] **Step 4: Run tests — verify pass**

Run: `uv run python -m pytest tests/test_rank_alignment_loader.py -v`
Expected: 1 passes (`test_target_modules_list`), 2 SVD tests skip if no CUDA or pass if CUDA is present.

If on a machine without CUDA, instead run:
```bash
uv run python -m pytest tests/test_rank_alignment_loader.py::test_target_modules_list -v
```
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add analysis/rank_alignment_loader.py tests/test_rank_alignment_loader.py
git commit -m "analysis: add base-SVD loader with safetensors cache"
```

---

## Task 5: Loader — `load_delta_w` for all three methods

**Files:**
- Modify: `analysis/rank_alignment_loader.py`
- Modify: `tests/test_rank_alignment_loader.py`

- [ ] **Step 1: Write failing test for `load_delta_w` on a Full FT synthetic ckpt**

Append to `tests/test_rank_alignment_loader.py`:

```python
def _make_fake_full_ckpt(ckpt_dir: Path, base_tensors: dict[str, torch.Tensor]):
    """Create a step ckpt by perturbing the base tensors; mimics dense `nn.Linear`."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    perturbed = {k: v + 0.01 * torch.randn_like(v) for k, v in base_tensors.items()}
    save_file(perturbed, str(ckpt_dir / "model.safetensors"))
    return perturbed


def test_load_delta_w_full_method(tmp_path):
    base_dir = _make_fake_base_model(tmp_path / "base", num_layers=2)
    # Read the base tensors back to pass to the fake ckpt maker
    base_idx = load_safetensors_index(str(base_dir))
    base_tensors = {k: load_tensor(base_idx, k) for k in base_idx}
    ckpt_dir = tmp_path / "full_step50"
    perturbed = _make_fake_full_ckpt(ckpt_dir, base_tensors)

    base_W = base_tensors["model.layers.0.mlp.up_proj.weight"]
    delta = load_delta_w(
        ckpt_dir=ckpt_dir,
        method="full",
        layer=0,
        module="up_proj",
        base_W=base_W,
        device=torch.device("cpu"),
    )
    expected = perturbed["model.layers.0.mlp.up_proj.weight"] - base_W
    torch.testing.assert_close(delta, expected, atol=1e-5, rtol=1e-5)


def test_load_delta_w_svd_method(tmp_path):
    # Build a fake SVD ckpt: store svd_a/svd_b/svd_s such that
    # materialize_svd_weight(svd_a, svd_b, svd_s) = some tensor W_ckpt,
    # then verify ΔW = W_ckpt - W_base.
    base_dir = _make_fake_base_model(tmp_path / "base", num_layers=1)
    base_idx = load_safetensors_index(str(base_dir))
    base_W = load_tensor(base_idx, "model.layers.0.mlp.up_proj.weight")  # (192, 64)

    # Synthetic factors: W_ckpt = A @ diag(s) @ B
    r = 64
    A = torch.randn(192, r)
    B = torch.randn(r, 64)
    s = torch.rand(r) + 0.5
    W_ckpt_expected = (A * s.unsqueeze(0)) @ B
    ckpt_dir = tmp_path / "svd_step50"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "model.layers.0.mlp.up_proj.svd_a": A,
            "model.layers.0.mlp.up_proj.svd_b": B,
            "model.layers.0.mlp.up_proj.svd_s": s,
        },
        str(ckpt_dir / "model.safetensors"),
    )
    delta = load_delta_w(
        ckpt_dir=ckpt_dir,
        method="svd",
        layer=0,
        module="up_proj",
        base_W=base_W,
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(delta, W_ckpt_expected - base_W, atol=1e-4, rtol=1e-4)


def test_load_delta_w_blocktt_method(tmp_path):
    # For up_proj (192, 64), need factors whose materialize matches the expected shape.
    # materialize_blocktt_weight returns (m*a, n*b). Choose m=8, a=24, n=8, b=8 → (192, 64).
    # r = 2 (arbitrary). Shapes: btt_l (m, r*n, a) = (8, 16, 24), btt_r (n, b, m*r) = (8, 8, 16),
    # btt_s (m, n, r) = (8, 8, 2).
    base_dir = _make_fake_base_model(tmp_path / "base", num_layers=1)
    base_idx = load_safetensors_index(str(base_dir))
    base_W = load_tensor(base_idx, "model.layers.0.mlp.up_proj.weight")

    btt_l = torch.randn(8, 16, 24)
    btt_r = torch.randn(8, 8, 16)
    btt_s = torch.randn(8, 8, 2)
    W_ckpt = materialize_blocktt_weight(btt_l, btt_r, btt_s)
    assert W_ckpt.shape == base_W.shape

    ckpt_dir = tmp_path / "blocktt_step50"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "model.layers.0.mlp.up_proj.btt_l": btt_l,
            "model.layers.0.mlp.up_proj.btt_r": btt_r,
            "model.layers.0.mlp.up_proj.btt_s": btt_s,
        },
        str(ckpt_dir / "model.safetensors"),
    )
    delta = load_delta_w(
        ckpt_dir=ckpt_dir,
        method="blocktt",
        layer=0,
        module="up_proj",
        base_W=base_W,
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(delta, W_ckpt - base_W, atol=1e-4, rtol=1e-4)


def test_load_delta_w_rejects_unknown_method(tmp_path):
    base_dir = _make_fake_base_model(tmp_path / "base", num_layers=1)
    base_idx = load_safetensors_index(str(base_dir))
    base_W = load_tensor(base_idx, "model.layers.0.mlp.up_proj.weight")
    with pytest.raises(ValueError, match="method"):
        load_delta_w(
            ckpt_dir=tmp_path,
            method="bogus",
            layer=0,
            module="up_proj",
            base_W=base_W,
            device=torch.device("cpu"),
        )
```

- [ ] **Step 2: Run tests — verify failure**

Run: `uv run python -m pytest tests/test_rank_alignment_loader.py -v`
Expected: CPU tests fail with ImportError (`load_delta_w` not defined).

- [ ] **Step 3: Implement `load_delta_w`**

Append to `analysis/rank_alignment_loader.py`:

```python
def load_delta_w(
    ckpt_dir: Path,
    method: Literal["full", "blocktt", "svd"],
    layer: int,
    module: str,
    base_W: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Return ΔW = W_ckpt − W_base as a dense tensor on `device`.

    - full:    load the ckpt's `.weight` key; ΔW = W_ckpt − W_base.
    - blocktt: materialize btt cores into a dense W_ckpt; ΔW = W_ckpt − W_base.
    - svd:     materialize svd factors into a dense W_ckpt (replacement
               parameterisation, per svd_layer.py:SVDLayer.materialize_dense_weight);
               ΔW = W_ckpt − W_base.
    """
    if method not in {"full", "blocktt", "svd"}:
        raise ValueError(f"method must be one of {{full, blocktt, svd}}, got {method!r}")

    ckpt_path = Path(ckpt_dir)
    ckpt_index = load_safetensors_index(str(ckpt_path))
    base_W_dev = base_W.to(device=device).float()

    if method == "full":
        key = get_base_weight_key(layer, module)
        W_ckpt = load_tensor(ckpt_index, key)
        if W_ckpt is None:
            raise KeyError(f"Missing key {key} in {ckpt_path}")
        return W_ckpt.to(device=device).float() - base_W_dev

    if method == "blocktt":
        keys = get_checkpoint_keys(layer, module, "blocktt")
        btt_l = load_tensor(ckpt_index, keys["btt_l"])
        btt_r = load_tensor(ckpt_index, keys["btt_r"])
        btt_s = load_tensor(ckpt_index, keys["btt_s"])  # may be None if not stored
        if btt_l is None or btt_r is None:
            raise KeyError(f"Missing btt_l/btt_r for layer {layer} module {module}")
        btt_l = btt_l.to(device=device).float()
        btt_r = btt_r.to(device=device).float()
        btt_s_f = btt_s.to(device=device).float() if btt_s is not None else None
        W_ckpt = materialize_blocktt_weight(btt_l, btt_r, btt_s_f)
        return W_ckpt - base_W_dev

    # method == "svd"
    keys = get_checkpoint_keys(layer, module, "svd")
    svd_a = load_tensor(ckpt_index, keys["svd_a"])
    svd_b = load_tensor(ckpt_index, keys["svd_b"])
    svd_s = load_tensor(ckpt_index, keys["svd_s"])
    if svd_a is None or svd_b is None:
        raise KeyError(f"Missing svd_a/svd_b for layer {layer} module {module}")
    svd_a = svd_a.to(device=device).float()
    svd_b = svd_b.to(device=device).float()
    svd_s_f = svd_s.to(device=device).float() if svd_s is not None else None
    W_ckpt = materialize_svd_weight(svd_a, svd_b, svd_s_f)
    return W_ckpt - base_W_dev
```

- [ ] **Step 4: Run tests — verify pass**

Run: `uv run python -m pytest tests/test_rank_alignment_loader.py -v`
Expected: target-modules test + 4 new tests pass (CUDA tests skip or pass).

- [ ] **Step 5: Commit**

```bash
git add analysis/rank_alignment_loader.py tests/test_rank_alignment_loader.py
git commit -m "analysis: add load_delta_w for full/blocktt/svd checkpoints"
```

---

## Task 6: Entry point 1 — `analyze_rank_alignment.py`

**Files:**
- Create: `analysis/analyze_rank_alignment.py`

No unit tests for this entry point — it orchestrates loader + metric library (both already tested) and writes JSON. An end-to-end smoke test is run manually in Task 8.

- [ ] **Step 1: Create the entry point with argparse + hardcoded paths**

Create `analysis/analyze_rank_alignment.py`:

```python
"""Entry point: sweep all three methods' checkpoints and write metric JSON caches."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Iterable

import torch

from analysis.rank_alignment_loader import (
    TARGET_MODULES,
    load_base_svd,
    load_delta_w,
)
from analysis.rank_alignment_metrics import (
    align_energy_topk,
    align_energy_u,
    align_energy_v,
    entropy_rank,
    per_direction_energy,
    principal_mass_fractions,
    random_baseline_align,
    stable_rank,
    threshold_rank,
)


# Hardcoded ckpt paths for --method all (§2 of the spec)
METHOD_CKPT_ROOTS = {
    "full":    Path("/data/yequan/fura/rl_runs/full/full-adamw-lr_2e-5-0325-215533"),
    "blocktt": Path("/data/yequan/fura/rl_runs/blocktt/blocktt-adamw-lr_1e-4-output_one_block-s_to_frozen-train_small-0317-150342"),
    "svd":     Path("/data/yequan/fura/rl_runs/svd/svd-adamw-lr_1e-5-s_to_keep-train_input-0317-141139"),
}

# U-side projection on gate/up (d_out > d_in); V-side on down (d_in > d_out);
# q/k/v/o_proj are excluded from the main alignment figure (square → trivial 1.0)
ALIGN_SIDE_BY_MODULE = {
    "gate_proj": "U",
    "up_proj":   "U",
    "down_proj": "V",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--method", choices=["full", "blocktt", "svd", "all"], default="all")
    p.add_argument("--ckpt-dir", type=Path, default=None,
                   help="Required when --method != all; ignored otherwise.")
    p.add_argument("--steps", type=lambda s: [int(x) for x in s.split(",")],
                   default=[1, 10, 50])
    p.add_argument("--out-dir", type=Path, default=Path("analysis_cache/rank_alignment"))
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--topk-list", type=lambda s: [int(x) for x in s.split(",")],
                   default=[32, 64, 128, 256, 512])
    args = p.parse_args(argv)
    if args.method != "all" and args.ckpt_dir is None:
        p.error(f"--ckpt-dir is required when --method={args.method}")
    return args


def _method_ckpt_dirs(method: str, ckpt_dir_override: Path | None) -> dict[str, Path]:
    """Return {method: root_dir} for the methods we should sweep."""
    if method == "all":
        return dict(METHOD_CKPT_ROOTS)
    if ckpt_dir_override is None:
        raise ValueError("ckpt_dir_override required when method != all")
    return {method: ckpt_dir_override}


def _step_dir(root: Path, step: int) -> Path:
    return root / f"step={step}"


def _assert_ckpts_exist(method_roots: dict[str, Path], steps: list[int]) -> None:
    for method, root in method_roots.items():
        for step in steps:
            sd = _step_dir(root, step)
            if not sd.exists():
                raise FileNotFoundError(f"{method} step={step}: {sd} not found")


def _tensor_bases(base_svd: dict, base_W_lookup, layer: int, module: str):
    """Return (U0, V0, base_W). base_W_lookup is a callable so we can keep it lazy."""
    U0, V0 = base_svd[(layer, module)]
    base_W = base_W_lookup(layer, module)
    return U0, V0, base_W


def compute_metrics_one_module(
    delta_W: torch.Tensor,
    U0: torch.Tensor,
    V0: torch.Tensor,
    d_out: int,
    d_in: int,
    module: str,
    topk_list: Iterable[int],
    base_frob: float | None = None,
) -> dict:
    """Full metric suite for one (delta_W, U0, V0) triple."""
    r0 = min(d_out, d_in)
    frob_delta = float(delta_W.float().norm())
    out = {
        "shape": [d_out, d_in],
        "r0": r0,
        "frob_delta": frob_delta,
        "frob_ratio": (frob_delta / base_frob) if base_frob else None,
        "stable_rank": stable_rank(delta_W),
        "entropy_rank": entropy_rank(delta_W),
        "threshold_rank_0p01": threshold_rank(delta_W, frac=0.01),
        "align_u_full": align_energy_u(delta_W, U0),
        "align_v_full": align_energy_v(delta_W, V0),
        "align_topk_u": {k: align_energy_topk(delta_W, U0, k=k, side="U")
                         for k in topk_list if k <= U0.shape[1]},
        "align_topk_v": {k: align_energy_topk(delta_W, V0, k=k, side="V")
                         for k in topk_list if k <= V0.shape[1]},
    }
    # Main-figure alignment: whichever side is informative
    side = ALIGN_SIDE_BY_MODULE.get(module)
    if side == "U":
        out["align_main"] = out["align_u_full"]
    elif side == "V":
        out["align_main"] = out["align_v_full"]
    else:
        out["align_main"] = None

    # Per-direction energy + mass fractions
    energy = per_direction_energy(delta_W, U0, V0)
    out["per_direction_energy"] = energy.tolist()
    out["mass_top_quarter"], out["mass_mid_half"], out["mass_tail_quarter"] = (
        principal_mass_fractions(energy, r0=r0)
    )

    # Diagnostics (spec §3.4): singular spectrum of ΔW + principal angles between U0 and U_ΔW
    if delta_W.norm() >= 1e-8:
        sigma_delta = torch.linalg.svdvals(delta_W.float()).cpu().numpy()
        out["sigma_delta_top32"] = sigma_delta[:32].tolist()
        # Principal angles: cos(θ_i) = σ_i(U0^T U_ΔW), on top-min(r0, 10)
        U_d, _, _ = torch.linalg.svd(delta_W.float(), full_matrices=False)
        k_pa = min(10, U0.shape[1], U_d.shape[1])
        cos_angles = torch.linalg.svdvals(U0[:, :k_pa].T @ U_d[:, :k_pa])
        cos_angles = torch.clamp(cos_angles, 0.0, 1.0)
        out["principal_angles_deg_top10"] = (
            torch.acos(cos_angles).rad2deg().cpu().tolist()
        )
    else:
        out["sigma_delta_top32"] = []
        out["principal_angles_deg_top10"] = []

    return out


def _base_w_lookup_factory(base_model: str, device: torch.device):
    """Cache-resolve the base model once and return a callable for per-module base_W."""
    from analysis.analyze_weights import (
        get_base_weight_key,
        load_safetensors_index,
        load_tensor,
    )
    from analysis.rank_alignment_loader import _resolve_base_dir

    base_dir = _resolve_base_dir(base_model)
    base_index = load_safetensors_index(base_dir)

    def _fetch(layer: int, module: str) -> torch.Tensor:
        key = get_base_weight_key(layer, module)
        W = load_tensor(base_index, key)
        if W is None:
            raise KeyError(f"Missing base weight {key}")
        return W.to(device=device).float()

    return _fetch


def run_sweep(args) -> dict[str, Path]:
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    method_roots = _method_ckpt_dirs(args.method, args.ckpt_dir)
    _assert_ckpts_exist(method_roots, args.steps)

    print(f"[{time.strftime('%H:%M:%S')}] Loading base SVDs...")
    base_svd = load_base_svd(args.base_model, cache_dir=args.out_dir, device=device)
    base_W_lookup = _base_w_lookup_factory(args.base_model, device=device)
    num_layers = max(layer for (layer, _) in base_svd.keys()) + 1

    # Preflight: one triple, print result
    print("[preflight] full / step=50 / layer=0 / up_proj:")
    pf_root = method_roots.get("full") or next(iter(method_roots.values()))
    pf_method = "full" if "full" in method_roots else next(iter(method_roots))
    pf_step = args.steps[-1]
    U0_pf, V0_pf = base_svd[(0, "up_proj")]
    base_W_pf = base_W_lookup(0, "up_proj")
    delta_pf = load_delta_w(
        ckpt_dir=_step_dir(pf_root, pf_step),
        method=pf_method,
        layer=0,
        module="up_proj",
        base_W=base_W_pf,
        device=device,
    )
    preflight = compute_metrics_one_module(
        delta_pf, U0_pf, V0_pf,
        d_out=base_W_pf.shape[0], d_in=base_W_pf.shape[1],
        module="up_proj", topk_list=args.topk_list,
    )
    print(json.dumps({k: v for k, v in preflight.items() if not isinstance(v, list)}, indent=2))

    cache_paths: dict[str, Path] = {}
    for method, root in method_roots.items():
        print(f"\n[{time.strftime('%H:%M:%S')}] === method={method} ===")
        results: dict = {"meta": {"method": method, "ckpt_root": str(root),
                                  "base_model": args.base_model,
                                  "steps": args.steps, "num_layers": num_layers,
                                  "topk_list": args.topk_list,
                                  "target_modules": list(TARGET_MODULES)},
                         "results": {}}
        for step in args.steps:
            step_key = str(step)
            results["results"][step_key] = {}
            ckpt_step = _step_dir(root, step)
            for layer in range(num_layers):
                results["results"][step_key][str(layer)] = {}
                for module in TARGET_MODULES:
                    try:
                        base_W = base_W_lookup(layer, module)
                        delta = load_delta_w(
                            ckpt_dir=ckpt_step,
                            method=method,
                            layer=layer,
                            module=module,
                            base_W=base_W,
                            device=device,
                        )
                        U0, V0 = base_svd[(layer, module)]
                        metrics = compute_metrics_one_module(
                            delta, U0, V0,
                            d_out=base_W.shape[0], d_in=base_W.shape[1],
                            module=module, topk_list=args.topk_list,
                            base_frob=float(base_W.norm()),
                        )
                        assert -1e-6 <= metrics["align_u_full"] <= 1.0 + 1e-6, (
                            f"align_u_full out of range: {metrics['align_u_full']}"
                        )
                        assert metrics["stable_rank"] != metrics["stable_rank"] or (
                            metrics["stable_rank"] < min(base_W.shape) + 1
                        )
                        results["results"][step_key][str(layer)][module] = metrics
                    except Exception as exc:
                        print(f"  [warn] {method}/step={step}/layer={layer}/{module}: {exc}")
                        results["results"][step_key][str(layer)][module] = {"error": str(exc)}
                    finally:
                        del delta
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                gc.collect()
                if layer % 4 == 0:
                    print(f"  step={step} layer {layer}/{num_layers - 1} done")

        out_path = args.out_dir / f"{method}.json"
        out_path.write_text(json.dumps(results, indent=2))
        cache_paths[method] = out_path
        print(f"  wrote {out_path}")

    _print_quick_read(cache_paths, base_svd)
    return cache_paths


def _print_quick_read(cache_paths: dict[str, Path], base_svd: dict) -> None:
    print("\n=== Quick-read summary ===")
    any_pair = next(iter(base_svd))
    U0, V0 = base_svd[any_pair]
    for method, path in cache_paths.items():
        data = json.loads(path.read_text())
        last_step = str(data["meta"]["steps"][-1])
        sr_vals = []
        align_by_mod = {m: [] for m in ("gate_proj", "up_proj", "down_proj")}
        for layer_str, mods in data["results"][last_step].items():
            for m, metrics in mods.items():
                if "error" in metrics:
                    continue
                if metrics["stable_rank"] == metrics["stable_rank"]:  # not NaN
                    sr_vals.append(metrics["stable_rank"])
                if m in align_by_mod and metrics.get("align_main") is not None:
                    align_by_mod[m].append(metrics["align_main"])
        if sr_vals:
            print(f"{method:>8s}: stable rank range [{min(sr_vals):.1f}, {max(sr_vals):.1f}]")
        for m, vals in align_by_mod.items():
            if vals:
                mean = sum(vals) / len(vals)
                var = sum((v - mean) ** 2 for v in vals) / len(vals)
                std = var ** 0.5
                print(f"          align_main {m}: {mean:.3f} ± {std:.3f} (n={len(vals)})")
    # Baseline reminder (0.333 for Qwen3-1.7B 6144×2048 shape)
    print(f"Random baseline (r0 / max(d_out, d_in)) for 6144×2048 layers: "
          f"{random_baseline_align(6144, 2048, 'U'):.3f}")


def main(argv=None):
    args = parse_args(argv)
    run_sweep(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax-check the entry point**

Run: `uv run python -c "import analysis.analyze_rank_alignment"`
Expected: exits with code 0, no ImportError.

- [ ] **Step 3: Commit**

```bash
git add analysis/analyze_rank_alignment.py
git commit -m "analysis: add analyze_rank_alignment entry point"
```

---

## Task 7: Entry point 2 — `plot_rank_alignment_3x3.py`

**Files:**
- Create: `analysis/plot_rank_alignment_3x3.py`

No unit tests; rendered artifact is inspected manually.

- [ ] **Step 1: Create the plotting entry point**

Create `analysis/plot_rank_alignment_3x3.py`:

```python
"""Entry point: consume the three per-method JSONs and write PDF + notes markdown."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
)
ALIGN_MODULES = ("gate_proj", "up_proj", "down_proj")
HEATMAP_MODULES = ("up_proj", "down_proj")
METHOD_ORDER = ("full", "blocktt", "svd")
METHOD_LABELS = {"full": "Full FT", "blocktt": "FuRA (blocktt)", "svd": "SVD"}

MODULE_COLORS = {
    "q_proj":    "#b22222",  # warm: attention
    "k_proj":    "#d2691e",
    "v_proj":    "#daa520",
    "o_proj":    "#cd853f",
    "gate_proj": "#2ca02c",  # cool: MLP
    "up_proj":   "#1f77b4",
    "down_proj": "#6a3d9a",
}
ALIGN_COLORS = {"gate_proj": "#2ca02c", "up_proj": "#1f77b4", "down_proj": "#6a3d9a"}
HEATMAP_SIGMA_CAP = 512


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, default=Path("analysis_cache/rank_alignment"))
    p.add_argument("--out-pdf", type=Path,
                   default=Path("docs/26_nips_fura_paper/figs/rank_and_alignment_3x3.pdf"))
    p.add_argument("--out-md", type=Path,
                   default=Path("docs/26_nips_fura_paper/rank_and_alignment_notes.md"))
    return p.parse_args(argv)


def _load_caches(cache_dir: Path) -> dict[str, dict]:
    out = {}
    for method in METHOD_ORDER:
        p = cache_dir / f"{method}.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing cache for {method}: {p}")
        out[method] = json.loads(p.read_text())
    return out


def _num_layers(cache: dict) -> int:
    return cache["meta"]["num_layers"]


def _last_step(cache: dict) -> str:
    return str(cache["meta"]["steps"][-1])


def _gather_stable_rank(cache: dict) -> dict[str, list[float]]:
    """Return {module: [stable_rank per layer]} at the last step."""
    step = _last_step(cache)
    per_module: dict[str, list[float]] = {m: [] for m in TARGET_MODULES}
    for layer_str, mods in cache["results"][step].items():
        for m in TARGET_MODULES:
            metrics = mods.get(m, {})
            v = metrics.get("stable_rank", float("nan"))
            per_module[m].append(v if v is not None else float("nan"))
    return per_module


def _gather_align_trajectory(cache: dict) -> dict[str, dict[str, tuple[float, float]]]:
    """Return {module: {step: (mean, std)}} for the 3 alignment modules, across layers."""
    out: dict[str, dict[str, tuple[float, float]]] = {m: {} for m in ALIGN_MODULES}
    for step in cache["meta"]["steps"]:
        step_key = str(step)
        for m in ALIGN_MODULES:
            vals = []
            for layer_str, mods in cache["results"][step_key].items():
                v = mods.get(m, {}).get("align_main")
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    vals.append(v)
            if vals:
                mean = float(np.mean(vals))
                std = float(np.std(vals))
                out[m][step_key] = (mean, std)
    return out


def _gather_heatmap(cache: dict, module: str) -> np.ndarray:
    """Return (num_layers, HEATMAP_SIGMA_CAP) per-direction energy matrix at last step."""
    step = _last_step(cache)
    num_layers = _num_layers(cache)
    out = np.zeros((num_layers, HEATMAP_SIGMA_CAP))
    for layer_str, mods in cache["results"][step].items():
        layer = int(layer_str)
        metrics = mods.get(module, {})
        energy = metrics.get("per_direction_energy", [])
        k = min(len(energy), HEATMAP_SIGMA_CAP)
        out[layer, :k] = np.array(energy[:k])
    return out


def _random_baseline_align() -> float:
    # Qwen3-1.7B 6144×2048: 2048/6144 = 1/3
    return 2048 / 6144


def build_figure(caches: dict[str, dict], out_pdf: Path) -> dict:
    fig, axes = plt.subplots(
        nrows=3, ncols=3, figsize=(14, 10),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.25]},
    )
    fig_data: dict = {"per_panel": {}}

    for row, method in enumerate(METHOD_ORDER):
        cache = caches[method]
        label = METHOD_LABELS[method]

        # Col 1: stable rank per layer × module
        ax1 = axes[row, 0]
        sr = _gather_stable_rank(cache)
        layers = np.arange(_num_layers(cache))
        for m in TARGET_MODULES:
            ax1.plot(layers, sr[m], marker="o", markersize=3, linewidth=1.0,
                     color=MODULE_COLORS[m], label=m)
        ax1.set_yscale("log")
        ax1.set_xlabel("Layer idx")
        ax1.set_ylabel(f"{label}\nstable rank(ΔW)")
        ax1.grid(True, which="both", alpha=0.3)
        if row == 0:
            ax1.legend(fontsize=7, ncol=2, loc="best")
        fig_data["per_panel"][f"row{row}_col1"] = sr

        # Col 2: alignment trajectory
        ax2 = axes[row, 1]
        traj = _gather_align_trajectory(cache)
        steps_int = cache["meta"]["steps"]
        for m in ALIGN_MODULES:
            means = [traj[m].get(str(s), (float("nan"), 0.0))[0] for s in steps_int]
            stds = [traj[m].get(str(s), (float("nan"), 0.0))[1] for s in steps_int]
            ax2.plot(steps_int, means, marker="o", linewidth=1.5,
                     color=ALIGN_COLORS[m], label=m)
            ax2.fill_between(
                steps_int,
                [mu - sd for mu, sd in zip(means, stds)],
                [mu + sd for mu, sd in zip(means, stds)],
                alpha=0.15, color=ALIGN_COLORS[m],
            )
        ax2.axhline(_random_baseline_align(), linestyle="--", color="black",
                    linewidth=0.8, label="random baseline" if row == 0 else None)
        ax2.set_xscale("log")
        ax2.set_xlabel("RL step")
        ax2.set_ylabel(f"{label}\nalign_ΔW (main-side)")
        ax2.set_ylim(0, 1.05)
        ax2.grid(True, alpha=0.3)
        if row == 0:
            ax2.legend(fontsize=7, loc="lower right")
        fig_data["per_panel"][f"row{row}_col2"] = traj

        # Col 3: heatmap (2 subpanels side by side: up_proj, down_proj)
        # Use a nested gridspec inside axes[row, 2]
        gs = axes[row, 2].get_subplotspec().subgridspec(1, 2, wspace=0.15)
        axes[row, 2].axis("off")
        hm_data = {}
        for i, module in enumerate(HEATMAP_MODULES):
            ax_h = fig.add_subplot(gs[0, i])
            H = _gather_heatmap(cache, module)
            hm_data[module] = H
            # Replace zeros with small value for log plotting
            H_log = np.where(H > 0, H, np.nan)
            im = ax_h.imshow(
                H_log,
                aspect="auto",
                origin="lower",
                cmap="viridis",
                norm=LogNorm(vmin=max(1e-6, np.nanmin(H_log)) if np.any(~np.isnan(H_log)) else 1e-6,
                             vmax=np.nanmax(H_log) if np.any(~np.isnan(H_log)) else 1.0),
                interpolation="nearest",
            )
            ax_h.set_xlabel(f"σ-index ({module})")
            if i == 0:
                ax_h.set_ylabel(f"{label}\nlayer idx")
            if row == 0 and i == 1:
                fig.colorbar(im, ax=ax_h, fraction=0.04, pad=0.03, label="normalised energy")
        fig_data["per_panel"][f"row{row}_col3"] = {
            m: hm.tolist() for m, hm in hm_data.items()
        }

    fig.suptitle("Rank and alignment of ΔW on Qwen3-1.7B RL (step=50)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    return fig_data


def _fmt_float(x, digits=3):
    if x is None:
        return "—"
    if isinstance(x, float) and math.isnan(x):
        return "NaN"
    return f"{x:.{digits}f}"


def _fmt_int(x):
    if x is None:
        return "—"
    if isinstance(x, float) and math.isnan(x):
        return "NaN"
    return f"{x:.1f}"


def write_notes_md(caches: dict[str, dict], out_md: Path) -> None:
    lines = [
        "# Rank and Alignment: Full Metric Tables",
        f"_Auto-generated on {datetime.now().isoformat(timespec='seconds')} "
        f"by analysis/plot_rank_alignment_3x3.py_",
        "",
        "## 1. Setup",
    ]
    for method in METHOD_ORDER:
        meta = caches[method]["meta"]
        lines += [
            f"- **{METHOD_LABELS[method]}** — ckpt `{meta['ckpt_root']}`, "
            f"steps {meta['steps']}, layers {meta['num_layers']}",
        ]
    lines += [
        f"- Base model: `{caches['full']['meta']['base_model']}`",
        f"- Modules: {list(TARGET_MODULES)}",
        "- SVD precision: fp32 on CUDA, full matrices off.",
        "",
        "## 2. Column-1 metrics — per-layer stable rank of ΔW (last step)",
    ]

    for method in METHOD_ORDER:
        cache = caches[method]
        lines += [f"", f"### 2.{METHOD_ORDER.index(method) + 1} {METHOD_LABELS[method]}", ""]
        lines += [
            "| layer | " + " | ".join(TARGET_MODULES) + " |",
            "|-------|" + "|".join(["--------"] * len(TARGET_MODULES)) + "|",
        ]
        sr = _gather_stable_rank(cache)
        for layer in range(_num_layers(cache)):
            row = [f"{layer:>5d}"]
            for m in TARGET_MODULES:
                row.append(_fmt_int(sr[m][layer]))
            lines.append("| " + " | ".join(row) + " |")

    lines += ["", "### 2.4 Summary stats (across layers)", ""]
    lines += [
        "| method | module | median | IQR | min | max |",
        "|--------|--------|--------|-----|-----|-----|",
    ]
    for method in METHOD_ORDER:
        sr = _gather_stable_rank(caches[method])
        for m in TARGET_MODULES:
            vals = [v for v in sr[m] if not (isinstance(v, float) and math.isnan(v))]
            if not vals:
                continue
            arr = np.array(vals)
            lines.append(
                f"| {method} | {m} | {np.median(arr):.1f} | "
                f"{np.percentile(arr, 75) - np.percentile(arr, 25):.1f} | "
                f"{arr.min():.1f} | {arr.max():.1f} |"
            )

    lines += ["", "## 3. Column-2 metrics — U/V alignment trajectory", ""]
    lines += [
        "| method | module | step=1 | step=10 | step=50 |",
        "|--------|--------|--------|---------|---------|",
    ]
    for method in METHOD_ORDER:
        traj = _gather_align_trajectory(caches[method])
        for m in ALIGN_MODULES:
            row = [method, m]
            for s in caches[method]["meta"]["steps"]:
                mu = traj[m].get(str(s), (float("nan"), 0.0))[0]
                row.append(_fmt_float(mu))
            lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        f"Random-baseline alignment: {_random_baseline_align():.3f} "
        f"(for 6144×2048 Qwen3-1.7B MLP layers).",
    ]

    lines += ["", "## 4. Column-3 metrics — per-direction mass fractions", ""]
    lines += [
        "| method | module | mass top-¼ | mass mid-½ | mass tail-¼ |",
        "|--------|--------|------------|------------|-------------|",
    ]
    for method in METHOD_ORDER:
        cache = caches[method]
        step = _last_step(cache)
        for m in HEATMAP_MODULES:
            tops, mids, tails = [], [], []
            for layer_str, mods in cache["results"][step].items():
                metrics = mods.get(m, {})
                t = metrics.get("mass_top_quarter")
                md = metrics.get("mass_mid_half")
                tl = metrics.get("mass_tail_quarter")
                if t is not None:
                    tops.append(t); mids.append(md); tails.append(tl)
            if tops:
                lines.append(
                    f"| {method} | {m} | "
                    f"{np.mean(tops):.3f} ± {np.std(tops):.3f} | "
                    f"{np.mean(mids):.3f} ± {np.std(mids):.3f} | "
                    f"{np.mean(tails):.3f} ± {np.std(tails):.3f} |"
                )

    lines += [
        "",
        "## 5. Caveats",
        "- No raw-gradient measurement in this first peek. See spec §9.",
        "- Only step=1/10/50 available; no dense trajectory.",
        "- Square attention modules (q/k/v/o_proj) are trivial under full-U alignment ",
        "  (U0 spans R^{d_out}) and are excluded from the main-figure alignment panel.",
        "  They are included in §3.3 of this notes file via the top-k variant.",
        "",
        "## 6. Raw artifacts",
        "",
    ]
    for method in METHOD_ORDER:
        lines.append(f"- `analysis_cache/rank_alignment/{method}.json`")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n")


def main(argv=None):
    args = parse_args(argv)
    caches = _load_caches(args.cache_dir)
    fig_data = build_figure(caches, args.out_pdf)
    (args.out_pdf.with_suffix(".data.json")).write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "methods": list(METHOD_ORDER),
                "summary": {
                    method: {
                        "num_layers": _num_layers(caches[method]),
                        "last_step": _last_step(caches[method]),
                    }
                    for method in METHOD_ORDER
                },
            },
            indent=2,
        )
    )
    write_notes_md(caches, args.out_md)
    print(f"Wrote {args.out_pdf} + .png + .data.json and {args.out_md}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax-check the plotting entry point**

Run: `uv run python -c "import analysis.plot_rank_alignment_3x3"`
Expected: exits 0, no ImportError.

- [ ] **Step 3: Commit**

```bash
git add analysis/plot_rank_alignment_3x3.py
git commit -m "analysis: add plot_rank_alignment_3x3 entry point"
```

---

## Task 8: End-to-end run + manual inspection

**No new files.** This task runs the pipeline end-to-end against the real ckpts. The commit at the end only adds the output artifacts.

- [ ] **Step 1: Run the metric sweep**

Run on a free H100-class GPU:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python analysis/analyze_rank_alignment.py \
    --base-model Qwen/Qwen3-1.7B \
    --method all \
    --out-dir analysis_cache/rank_alignment 2>&1 | tee analysis_cache/rank_alignment/run.log
```

Expected:
- Prints "Computing base-SVDs for 28 layers × 7 modules" on first run (subsequent runs hit the cache).
- Prints a preflight metric dict for `full/step=50/layer=0/up_proj`.
- Prints `method=full`, `method=blocktt`, `method=svd` section headers, with progress every 4 layers.
- Writes `analysis_cache/rank_alignment/{full,blocktt,svd}.json` (one per method).
- Writes `analysis_cache/rank_alignment/base_svd.safetensors`.
- Prints the quick-read summary at the end.
- Total runtime: 5–15 min on H100.

- [ ] **Step 2: Sanity-check the quick-read summary**

Expected shape of the printout (actual numbers will vary):
```
=== Quick-read summary ===
    full: stable rank range [X.X, Y.Y]
          align_main gate_proj: 0.XXX ± 0.YYY (n=28)
          align_main up_proj:   0.XXX ± 0.YYY (n=28)
          align_main down_proj: 0.XXX ± 0.YYY (n=28)
 blocktt: ...
     svd: ...
Random baseline ...: 0.333
```

Check:
- All three methods report stable-rank ranges (no NaNs dominating).
- `align_main` for gate/up/down_proj is well above 0.333 for at least Full FT (Claim 2 — update aligns with pretrained col space).
- Stable-rank max/min ratio within a method is ≥ 5 for at least one module (Claim 1 — different layers need different ranks).

If any of the above fails, stop and open a follow-up spec; do not proceed to plotting with weak data. Record the fail in `analysis_cache/rank_alignment/run.log`.

- [ ] **Step 3: Run the plotting entry point**

```bash
uv run python analysis/plot_rank_alignment_3x3.py \
    --cache-dir analysis_cache/rank_alignment \
    --out-pdf docs/26_nips_fura_paper/figs/rank_and_alignment_3x3.pdf \
    --out-md  docs/26_nips_fura_paper/rank_and_alignment_notes.md
```

Expected:
- Writes `docs/26_nips_fura_paper/figs/rank_and_alignment_3x3.pdf`, `.png`, and `.data.json`.
- Writes `docs/26_nips_fura_paper/rank_and_alignment_notes.md`.
- Prints one "Wrote ..." line. Exits 0 in < 30 s.

- [ ] **Step 4: Visually inspect the figure**

Open `docs/26_nips_fura_paper/figs/rank_and_alignment_3x3.png`. Check:

- 3 rows × 3 columns rendered; no NaN gaps dominating any panel.
- Col 1 (stable rank): 7 coloured lines per panel; y-axis on log; legend readable on panel (a).
- Col 2 (alignment): 3 coloured lines + dashed baseline at 0.333; ±1σ shaded bands visible; y-axis in [0, 1].
- Col 3 (heatmap): two sub-heatmaps per panel (`up_proj` on the left, `down_proj` on the right); viridis log colormap; colorbar on row 1.
- Row-to-row y-axes match within each column.

If the figure is visually off, edit `analysis/plot_rank_alignment_3x3.py` and re-run step 3 (no need to re-run the sweep).

- [ ] **Step 5: Inspect the notes markdown**

Open `docs/26_nips_fura_paper/rank_and_alignment_notes.md`. Check sections 1–6 are present and populated with actual numbers (not "—" everywhere).

- [ ] **Step 6: Commit the output artifacts**

```bash
git add docs/26_nips_fura_paper/figs/rank_and_alignment_3x3.pdf \
        docs/26_nips_fura_paper/figs/rank_and_alignment_3x3.png \
        docs/26_nips_fura_paper/figs/rank_and_alignment_3x3.data.json \
        docs/26_nips_fura_paper/rank_and_alignment_notes.md
git commit -m "paper: add rank-and-alignment 3x3 figure and notes"
```

Do NOT stage `analysis_cache/`, `run.log`, or any `.safetensors` caches — these are re-derivable and stay out of git.

---

## Task 9: Final verification

- [ ] **Step 1: Run the full test suite one more time**

Run: `uv run python -m pytest tests/test_rank_alignment_metrics.py tests/test_rank_alignment_loader.py -v`
Expected: all tests green.

- [ ] **Step 2: Run syntax-check on every file**

Run: `uv run python -m py_compile analysis/rank_alignment_metrics.py analysis/rank_alignment_loader.py analysis/analyze_rank_alignment.py analysis/plot_rank_alignment_3x3.py`
Expected: exits 0.

- [ ] **Step 3: Verify git status is clean**

Run: `git status`
Expected: working tree clean or shows only untracked `analysis_cache/` and any unrelated preexisting changes — no modified or unstaged files from this plan.

- [ ] **Step 4: Verify the branch log**

Run: `git log --oneline -10`
Expected: 5 analysis commits + 1 paper commit near the top of the log, in the order from the commit plan in spec §8.4.

---

## Success criteria (from spec §8.3)

**Minimum (first-peek done, no scientific judgement yet):**
- Tests pass.
- 3×3 PDF renders with no NaN/zero panels dominating.
- Notes markdown written with §2–§6 populated.

**Paper-ready (scientific criteria, checked in Task 8 step 2):**
- Col 1: stable rank spans ≥5× across layers within at least one module, for at least one method.
- Col 2: `align_main` ≫ 0.333 for Full FT at step=50 on gate/up/down_proj.
- Col 3: heatmaps show non-uniform, non-monotone-in-σ patterns.

If paper-ready criteria hold → write the caption and ship. Otherwise → open a follow-up spec for the deferred second-peek (base-model forward-backward pass to capture `align_g(0)`).
