# Weight Update Analysis Pipeline Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a modular pipeline that computes weight-space metrics (spectral drift, principal angles, overlap with principal weights, update heatmaps) comparing base model weights against trained checkpoints, and renders results as an interactive HTML report.

**Architecture:** Three scripts — `analysis/analyze_weights.py` computes metrics and writes JSON, `analysis/build_report.py` reads JSON and generates a self-contained HTML page with Plotly.js, and `analysis/templates/report_template.html` provides the HTML skeleton. Metrics are computed on CPU one layer at a time.

**Tech Stack:** Python (torch, safetensors, numpy, json, argparse), Plotly.js (CDN), vanilla HTML/CSS/JS.

**Spec:** `docs/superpowers/specs/2026-03-16-weight-analysis-design.md`

---

## File Structure

```
analysis/
  analyze_weights.py      # CLI: loads checkpoints, computes metrics, writes JSON
  build_report.py         # CLI: reads JSON, generates HTML report
  templates/
    report_template.html  # HTML/JS template with Plotly scaffolding
tests/
  test_analyze_weights.py # Unit tests for metric computation and weight loading
analysis_results/         # gitignored output directory
```

Key responsibilities:
- `analyze_weights.py`: weight loading (3 modes: blocktt/svd/lora), metric computation (6 dense-weight metrics), JSON serialization
- `build_report.py`: JSON deserialization, HTML generation with embedded data, multi-run comparison
- `report_template.html`: 3-tab interactive layout (summary, detail, trends)

**Descoped from initial implementation:**
- Spec metrics 7-8 (per-core change heatmaps, per-block Frobenius norms) — these require comparing two factored checkpoints (e.g., BlockTT step=0 vs step=25), but the current use case compares base `nn.Linear` weights against a factored checkpoint. The base model has no BTT/SVD cores to diff against. Tab 3 (Factored Parameters) is deferred until two-checkpoint comparison is needed. The `core_changes` field is omitted from JSON output.
- Tab 3 (Factored Parameters) in the HTML report is deferred.

---

## Chunk 1: Core Metric Computation and Weight Loading

### Task 1: Project scaffolding and gitignore

**Files:**
- Create: `analysis/__init__.py` (empty)
- Create: `analysis/templates/` (directory)
- Modify: `.gitignore`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p analysis/templates
touch analysis/__init__.py
```

- [ ] **Step 2: Add analysis_results to gitignore**

Append to `.gitignore`:
```
analysis_results/
```

- [ ] **Step 3: Commit**

```bash
git add analysis/__init__.py .gitignore
git commit -m "scaffold: create analysis/ directory structure"
```

---

### Task 2: Weight materialization functions

**Files:**
- Create: `analysis/analyze_weights.py`
- Create: `tests/test_analyze_weights.py`

This task implements the three weight reconstruction functions (BlockTT, SVD, LoRA) that convert checkpoint tensors back into dense weight matrices. These are the foundation — every metric depends on getting `W_before` and `W_after` right.

- [ ] **Step 1: Write failing tests for BlockTT materialization**

Create `tests/test_analyze_weights.py`:

```python
import unittest
import torch
from analysis.analyze_weights import materialize_blocktt_weight


class TestMaterializeBlockTT(unittest.TestCase):
    def test_roundtrip_identity(self):
        """A BTT decomposition of a known matrix should reconstruct it."""
        # Create a simple block-diagonal-ish weight via known cores
        # btt_l: (m, rank*n, a), btt_r: (n, b, m*rank)
        m, n, rank, a, b = 2, 2, 3, 4, 4
        torch.manual_seed(42)
        btt_l = torch.randn(m, rank * n, a)
        btt_r = torch.randn(n, b, m * rank)
        W = materialize_blocktt_weight(btt_l, btt_r, btt_s=None)
        self.assertEqual(W.shape, (m * a, n * b))

    def test_with_btt_s(self):
        """Materialization with separate singular values."""
        m, n, rank, a, b = 2, 2, 3, 4, 4
        torch.manual_seed(42)
        btt_l = torch.randn(m, rank * n, a)
        btt_r = torch.randn(n, b, m * rank)
        btt_s = torch.ones(m, n, rank)
        W_no_s = materialize_blocktt_weight(btt_l, btt_r, btt_s=None)
        W_with_s = materialize_blocktt_weight(btt_l, btt_r, btt_s=btt_s)
        # With all-ones S, should be identical
        torch.testing.assert_close(W_no_s, W_with_s)

    def test_matches_btt_layer(self):
        """Our standalone function matches BTTLayer.materialize_dense_weight()."""
        from btt_layer import BTTLayer

        torch.manual_seed(7)
        layer = BTTLayer(
            in_features=16, out_features=16,
            btt_rank=4, decomp_mode="square",
            init_mode="default", bias=False,
        )
        W_ref = layer.materialize_dense_weight()
        W_ours = materialize_blocktt_weight(
            layer.btt_l.data, layer.btt_r.data,
            btt_s=layer.btt_s.data if layer.btt_s is not None else None,
        )
        torch.testing.assert_close(W_ours, W_ref, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_analyze_weights.py::TestMaterializeBlockTT -v
```

Expected: `ImportError: cannot import name 'materialize_blocktt_weight'`

- [ ] **Step 3: Implement BlockTT materialization**

Create `analysis/analyze_weights.py` with the initial function. The logic replicates `BTTLayer.materialize_dense_weight()` from `btt_layer.py:672-684` but works on raw tensors without a model instance:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_analyze_weights.py::TestMaterializeBlockTT -v
```

Expected: all 3 tests PASS

- [ ] **Step 5: Write failing tests for SVD materialization**

Add to `tests/test_analyze_weights.py`:

```python
from analysis.analyze_weights import materialize_svd_weight


class TestMaterializeSVD(unittest.TestCase):
    def test_basic(self):
        torch.manual_seed(42)
        svd_a = torch.randn(8, 4)  # (out, rank)
        svd_b = torch.randn(4, 6)  # (rank, in)
        W = materialize_svd_weight(svd_a, svd_b, svd_s=None)
        torch.testing.assert_close(W, svd_a @ svd_b)

    def test_with_svd_s(self):
        torch.manual_seed(42)
        svd_a = torch.randn(8, 4)
        svd_b = torch.randn(4, 6)
        svd_s = torch.tensor([2.0, 1.0, 0.5, 0.1])
        W = materialize_svd_weight(svd_a, svd_b, svd_s=svd_s)
        expected = (svd_a * svd_s.unsqueeze(0)) @ svd_b
        torch.testing.assert_close(W, expected)

    def test_matches_svd_layer(self):
        from svd_layer import SVDLayer

        torch.manual_seed(7)
        linear = torch.nn.Linear(16, 8, bias=False)
        layer = SVDLayer.from_linear(linear, s_merged_to="keep")
        W_ref = layer.materialize_dense_weight()
        W_ours = materialize_svd_weight(
            layer.svd_a.data, layer.svd_b.data,
            svd_s=layer.svd_s.data if layer.svd_s is not None else None,
        )
        torch.testing.assert_close(W_ours, W_ref, atol=1e-5, rtol=1e-5)
```

- [ ] **Step 6: Implement SVD materialization**

Add to `analysis/analyze_weights.py`:

```python
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
```

- [ ] **Step 7: Run SVD tests**

```bash
python -m pytest tests/test_analyze_weights.py::TestMaterializeSVD -v
```

Expected: all 3 tests PASS

- [ ] **Step 8: Write failing tests for LoRA reconstruction**

Add to `tests/test_analyze_weights.py`:

```python
from analysis.analyze_weights import reconstruct_lora_weight


class TestReconstructLoRA(unittest.TestCase):
    def test_basic(self):
        torch.manual_seed(42)
        W_base = torch.randn(8, 16)
        lora_A = torch.randn(4, 16)  # (rank, in)
        lora_B = torch.randn(8, 4)   # (out, rank)
        W = reconstruct_lora_weight(W_base, lora_A, lora_B, lora_alpha=16, r=4)
        expected = W_base + (16 / 4) * lora_B @ lora_A
        torch.testing.assert_close(W, expected)

    def test_alpha_equals_r(self):
        """When alpha==r, scaling is 1.0."""
        torch.manual_seed(42)
        W_base = torch.randn(8, 16)
        lora_A = torch.randn(4, 16)
        lora_B = torch.randn(8, 4)
        W = reconstruct_lora_weight(W_base, lora_A, lora_B, lora_alpha=4, r=4)
        expected = W_base + lora_B @ lora_A
        torch.testing.assert_close(W, expected)
```

- [ ] **Step 9: Implement LoRA reconstruction**

Add to `analysis/analyze_weights.py`:

```python
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
```

- [ ] **Step 10: Run all materialization tests**

```bash
python -m pytest tests/test_analyze_weights.py -v
```

Expected: all 8 tests PASS

- [ ] **Step 11: Commit**

```bash
git add analysis/analyze_weights.py tests/test_analyze_weights.py
git commit -m "feat: weight materialization functions for BlockTT, SVD, LoRA"
```

---

### Task 3: Core metric computation functions

**Files:**
- Modify: `analysis/analyze_weights.py`
- Modify: `tests/test_analyze_weights.py`

Implements the 6 dense-weight-level metrics from the spec. Each is a pure function taking `W_before`, `W_after` (and config params) and returning metric values.

- [ ] **Step 1: Write failing tests for metric functions**

Add to `tests/test_analyze_weights.py`:

```python
from analysis.analyze_weights import (
    compute_update_row_col_norms,
    compute_singular_vector_angles,
    compute_spectrum_and_nss,
    compute_principal_angles,
    compute_principal_weight_overlap,
    compute_update_spectrum,
)


class TestMetrics(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.W_before = torch.randn(32, 16)
        # Small perturbation
        self.W_after = self.W_before + 0.01 * torch.randn(32, 16)
        self.delta_W = self.W_after - self.W_before

    def test_update_row_col_norms_shapes(self):
        row_norms, col_norms = compute_update_row_col_norms(self.delta_W)
        self.assertEqual(row_norms.shape[0], 32)
        self.assertEqual(col_norms.shape[0], 16)

    def test_update_row_col_norms_nonneg(self):
        row_norms, col_norms = compute_update_row_col_norms(self.delta_W)
        self.assertTrue((row_norms >= 0).all())
        self.assertTrue((col_norms >= 0).all())

    def test_singular_vector_angles_small_perturbation(self):
        """Small perturbation should give small angles."""
        angles_u, angles_v = compute_singular_vector_angles(
            self.W_before, self.W_after, top_k=8
        )
        self.assertEqual(len(angles_u), 8)
        self.assertEqual(len(angles_v), 8)
        # Angles should be small (< 10 degrees) for tiny perturbation
        self.assertTrue(all(a < 10.0 for a in angles_u))

    def test_singular_vector_angles_identity(self):
        """No change should give zero angles."""
        angles_u, angles_v = compute_singular_vector_angles(
            self.W_before, self.W_before, top_k=4
        )
        for a in angles_u:
            self.assertAlmostEqual(a, 0.0, places=3)

    def test_spectrum_and_nss(self):
        s_before, s_after, nss = compute_spectrum_and_nss(
            self.W_before, self.W_after
        )
        self.assertEqual(len(s_before), min(32, 16))
        self.assertEqual(len(s_after), min(32, 16))
        self.assertGreater(nss, 0)
        # Small perturbation -> small NSS
        self.assertLess(nss, 0.1)

    def test_spectrum_nss_zero_for_identical(self):
        _, _, nss = compute_spectrum_and_nss(self.W_before, self.W_before)
        self.assertAlmostEqual(nss, 0.0, places=6)

    def test_principal_angles_shape(self):
        angles = compute_principal_angles(self.W_before, self.W_after, top_k=8)
        self.assertEqual(len(angles), 8)

    def test_principal_angles_small_perturbation(self):
        angles = compute_principal_angles(self.W_before, self.W_after, top_k=8)
        # Small perturbation -> small angles
        self.assertTrue(all(a < 10.0 for a in angles))

    def test_overlap_ratio(self):
        overlap, baseline = compute_principal_weight_overlap(
            self.W_before, self.delta_W, top_k=8, alpha=0.1, threshold_frac=0.01
        )
        # overlap and baseline should be between 0 and 1
        self.assertGreaterEqual(overlap, 0.0)
        self.assertLessEqual(overlap, 1.0)
        self.assertAlmostEqual(baseline, 0.1, places=5)

    def test_update_spectrum_shape(self):
        s = compute_update_spectrum(self.delta_W)
        self.assertEqual(len(s), min(32, 16))
        # Should be non-negative and sorted descending
        for i in range(len(s) - 1):
            self.assertGreaterEqual(s[i], s[i + 1] - 1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_analyze_weights.py::TestMetrics -v
```

Expected: `ImportError` for the metric functions

- [ ] **Step 3: Implement metric functions**

Add to `analysis/analyze_weights.py`:

```python
def compute_update_row_col_norms(
    delta_W: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise and column-wise L2 norms of the update matrix.

    Returns:
        (row_norms, col_norms) as numpy arrays.
    """
    row_norms = torch.norm(delta_W, dim=1).numpy()
    col_norms = torch.norm(delta_W, dim=0).numpy()
    return row_norms, col_norms


def compute_singular_vector_angles(
    W_before: torch.Tensor, W_after: torch.Tensor, top_k: int
) -> tuple[list[float], list[float]]:
    """Angle (degrees) between each top-k singular vector before/after.

    Uses absolute inner product to handle SVD sign ambiguity.
    """
    U0, _, Vt0 = torch.linalg.svd(W_before.float(), full_matrices=False)
    U1, _, Vt1 = torch.linalg.svd(W_after.float(), full_matrices=False)
    k = min(top_k, U0.shape[1], U1.shape[1])

    angles_u = []
    angles_v = []
    for i in range(k):
        cos_u = torch.clamp(torch.abs(U0[:, i] @ U1[:, i]), 0.0, 1.0)
        angles_u.append(float(torch.acos(cos_u).rad2deg()))
        cos_v = torch.clamp(torch.abs(Vt0[i] @ Vt1[i]), 0.0, 1.0)
        angles_v.append(float(torch.acos(cos_v).rad2deg()))
    return angles_u, angles_v


def compute_spectrum_and_nss(
    W_before: torch.Tensor, W_after: torch.Tensor
) -> tuple[list[float], list[float], float]:
    """Singular value spectra and normalized spectral shift.

    NSS = ||sigma(W_after) - sigma(W_before)||_2 / ||sigma(W_before)||_2
    """
    s_before = torch.linalg.svdvals(W_before.float())
    s_after = torch.linalg.svdvals(W_after.float())
    norm_before = torch.norm(s_before)
    nss = float(torch.norm(s_after - s_before) / norm_before) if norm_before > 0 else 0.0
    return s_before.tolist(), s_after.tolist(), nss


def compute_principal_angles(
    W_before: torch.Tensor, W_after: torch.Tensor, top_k: int
) -> list[float]:
    """Top-k subspace principal angles (degrees).

    cos theta_i = sigma_i(U_before_k^T @ U_after_k)
    """
    U0, _, _ = torch.linalg.svd(W_before.float(), full_matrices=False)
    U1, _, _ = torch.linalg.svd(W_after.float(), full_matrices=False)
    k = min(top_k, U0.shape[1], U1.shape[1])
    cos_angles = torch.linalg.svdvals(U0[:, :k].T @ U1[:, :k])
    cos_angles = torch.clamp(cos_angles, 0.0, 1.0)
    angles_deg = torch.acos(cos_angles).rad2deg()
    return angles_deg.tolist()


def compute_principal_weight_overlap(
    W_before: torch.Tensor,
    delta_W: torch.Tensor,
    top_k: int,
    alpha: float,
    threshold_frac: float,
) -> tuple[float, float]:
    """Overlap between update mask and principal weight mask.

    Principal mask: top-alpha fraction of |rank-k SVD reconstruction| by magnitude.
    Update mask: |delta_W_ij| > threshold_frac * max(|delta_W|).

    Returns:
        (overlap_ratio, random_baseline=alpha)
    """
    # Principal weight mask from rank-k approximation
    U, S, Vt = torch.linalg.svd(W_before.float(), full_matrices=False)
    k = min(top_k, len(S))
    W_lowrank = (U[:, :k] * S[:k].unsqueeze(0)) @ Vt[:k]
    n_principal = max(1, int(alpha * W_lowrank.numel()))
    threshold_princ = torch.topk(W_lowrank.abs().flatten(), n_principal).values[-1]
    M_princ = W_lowrank.abs() >= threshold_princ

    # Update mask
    max_delta = delta_W.abs().max()
    if max_delta == 0:
        return 0.0, alpha
    M_update = delta_W.abs() > threshold_frac * max_delta

    n_update = M_update.sum().item()
    if n_update == 0:
        return 0.0, alpha

    overlap = float((M_princ & M_update).sum().item() / n_update)
    return overlap, alpha


def compute_update_spectrum(delta_W: torch.Tensor) -> list[float]:
    """Singular values of the update matrix itself."""
    s = torch.linalg.svdvals(delta_W.float())
    return s.tolist()
```

- [ ] **Step 4: Run metric tests**

```bash
python -m pytest tests/test_analyze_weights.py::TestMetrics -v
```

Expected: all 12 tests PASS

- [ ] **Step 5: Commit**

```bash
git add analysis/analyze_weights.py tests/test_analyze_weights.py
git commit -m "feat: core metric computation functions (NSS, principal angles, overlap, spectra)"
```

---

### Task 4: Weight loading from safetensors

**Files:**
- Modify: `analysis/analyze_weights.py`
- Modify: `tests/test_analyze_weights.py`

Implements the logic that maps checkpoint keys to base model keys and loads the right tensors. This is the glue between raw safetensors files and the metric functions.

- [ ] **Step 1: Write failing test for key mapping**

Add to `tests/test_analyze_weights.py`:

```python
from analysis.analyze_weights import (
    get_base_weight_key,
    get_checkpoint_keys,
    TARGET_MODULES,
)


class TestKeyMapping(unittest.TestCase):
    def test_base_weight_key(self):
        key = get_base_weight_key(layer_idx=3, module_name="q_proj")
        self.assertEqual(key, "model.layers.3.self_attn.q_proj.weight")

    def test_base_weight_key_mlp(self):
        key = get_base_weight_key(layer_idx=0, module_name="gate_proj")
        self.assertEqual(key, "model.layers.0.mlp.gate_proj.weight")

    def test_checkpoint_keys_blocktt(self):
        keys = get_checkpoint_keys(layer_idx=1, module_name="q_proj", train_mode="blocktt")
        self.assertIn("model.layers.1.self_attn.q_proj.btt_l", keys)
        self.assertIn("model.layers.1.self_attn.q_proj.btt_r", keys)

    def test_checkpoint_keys_lora(self):
        keys = get_checkpoint_keys(layer_idx=0, module_name="q_proj", train_mode="lora")
        self.assertIn(
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight", keys
        )
        self.assertIn(
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight", keys
        )

    def test_target_modules(self):
        self.assertEqual(len(TARGET_MODULES), 7)
        self.assertIn("q_proj", TARGET_MODULES)
        self.assertIn("gate_proj", TARGET_MODULES)
```

- [ ] **Step 2: Implement key mapping**

Add to `analysis/analyze_weights.py`:

```python
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

ATTN_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_MODULES = ("gate_proj", "up_proj", "down_proj")


def _module_prefix(layer_idx: int, module_name: str) -> str:
    if module_name in ATTN_MODULES:
        return f"model.layers.{layer_idx}.self_attn.{module_name}"
    return f"model.layers.{layer_idx}.mlp.{module_name}"


def get_base_weight_key(layer_idx: int, module_name: str) -> str:
    return f"{_module_prefix(layer_idx, module_name)}.weight"


def get_checkpoint_keys(
    layer_idx: int, module_name: str, train_mode: str
) -> dict[str, str]:
    """Return dict mapping role -> safetensors key for a given module.

    Roles depend on train_mode:
        blocktt: btt_l, btt_r, (btt_s)
        svd: svd_a, svd_b, (svd_s)
        lora: lora_A, lora_B
    """
    prefix = _module_prefix(layer_idx, module_name)
    if train_mode == "blocktt":
        return {
            "btt_l": f"{prefix}.btt_l",
            "btt_r": f"{prefix}.btt_r",
            "btt_s": f"{prefix}.btt_s",
        }
    elif train_mode == "svd":
        return {
            "svd_a": f"{prefix}.svd_a",
            "svd_b": f"{prefix}.svd_b",
            "svd_s": f"{prefix}.svd_s",
        }
    elif train_mode == "lora":
        lora_prefix = f"base_model.model.{prefix}"
        return {
            "lora_A": f"{lora_prefix}.lora_A.weight",
            "lora_B": f"{lora_prefix}.lora_B.weight",
        }
    else:
        raise ValueError(f"Unknown train_mode: {train_mode}")
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/test_analyze_weights.py::TestKeyMapping -v
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add analysis/analyze_weights.py tests/test_analyze_weights.py
git commit -m "feat: checkpoint key mapping for blocktt/svd/lora modes"
```

---

### Task 5: CLI and main analysis loop

**Files:**
- Modify: `analysis/analyze_weights.py`

This wires everything together: argparse CLI, safetensors loading, iterating layers × modules, calling metric functions, and writing JSON output.

- [ ] **Step 1: Write failing test for CLI arg parsing**

Add to `tests/test_analyze_weights.py`:

```python
from analysis.analyze_weights import parse_args


class TestCLI(unittest.TestCase):
    def test_required_args(self):
        args = parse_args([
            "--base-model", "Qwen/Qwen3-1.7B",
            "--checkpoint", "runs/26/step=25",
            "--train-mode", "lora",
        ])
        self.assertEqual(args.base_model, "Qwen/Qwen3-1.7B")
        self.assertEqual(args.train_mode, "lora")
        self.assertEqual(args.top_k, 64)  # default

    def test_layers_parsing(self):
        args = parse_args([
            "--base-model", "x", "--checkpoint", "y", "--train-mode", "blocktt",
            "--layers", "0,5,10",
        ])
        self.assertEqual(args.layers, [0, 5, 10])

    def test_defaults(self):
        args = parse_args([
            "--base-model", "x", "--checkpoint", "y", "--train-mode", "svd",
        ])
        self.assertEqual(args.principal_alpha, 0.1)
        self.assertEqual(args.update_threshold, 0.01)
        self.assertEqual(args.output, "analysis_results/metrics.json")
```

- [ ] **Step 2: Implement CLI parsing**

Add to `analysis/analyze_weights.py`:

```python
import argparse
import json
from pathlib import Path
from datetime import datetime


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Analyze weight updates in trained checkpoints")
    p.add_argument("--base-model", required=True, help="Path or HF ID of base model")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint directory")
    p.add_argument("--train-mode", required=True, choices=["blocktt", "svd", "lora"])
    p.add_argument("--output", default="analysis_results/metrics.json")
    p.add_argument("--top-k", type=int, default=64)
    p.add_argument("--principal-alpha", type=float, default=0.1)
    p.add_argument("--layers", type=str, default=None,
                    help="Comma-separated layer indices for detailed analysis (default: 0,mid,last)")
    p.add_argument("--update-threshold", type=float, default=0.01)
    args = p.parse_args(argv)
    if args.layers is not None:
        args.layers = [int(x) for x in args.layers.split(",")]
    return args
```

- [ ] **Step 3: Run CLI tests**

```bash
python -m pytest tests/test_analyze_weights.py::TestCLI -v
```

Expected: PASS

- [ ] **Step 4: Implement main analysis loop**

Add to `analysis/analyze_weights.py`. This is the core orchestration — loads safetensors, iterates layers, calls metrics, writes JSON:

```python
from safetensors import safe_open


def load_safetensors_index(directory: str) -> dict[str, str]:
    """Map tensor keys to their safetensors file paths."""
    directory = Path(directory)
    index = {}
    for f in sorted(directory.glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as sf:
            for key in sf.keys():
                index[key] = str(f)
    return index


def load_tensor(index: dict[str, str], key: str) -> torch.Tensor | None:
    """Load a single tensor by key, or return None if key not found."""
    if key not in index:
        return None
    with safe_open(index[key], framework="pt") as sf:
        return sf.get_tensor(key)


def detect_num_layers(index: dict[str, str]) -> int:
    """Detect number of transformer layers from key names."""
    layers = set()
    for key in index:
        parts = key.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layers.add(int(parts[i + 1]))
    return max(layers) + 1 if layers else 0


def analyze(args) -> dict:
    """Main analysis: load weights, compute metrics, return results dict."""
    base_index = load_safetensors_index(args.base_model)

    # If base_model is an HF model ID, resolve to cache path
    if not base_index:
        from huggingface_hub import snapshot_download
        local_path = snapshot_download(args.base_model)
        base_index = load_safetensors_index(local_path)

    ckpt_index = load_safetensors_index(args.checkpoint)

    num_layers = detect_num_layers(base_index)
    if args.layers is None:
        args.layers = [0, num_layers // 2, num_layers - 1]
    for l in args.layers:
        if l >= num_layers:
            raise ValueError(f"Layer {l} out of range (model has {num_layers} layers)")

    # Read LoRA config if needed
    lora_alpha, lora_r = 1.0, 1
    if args.train_mode == "lora":
        config_path = Path(args.checkpoint) / "adapter_config.json"
        with open(config_path) as f:
            cfg = json.load(f)
        lora_alpha = cfg["lora_alpha"]
        lora_r = cfg["r"]

    detailed_layers = set(args.layers)

    results = {
        "meta": {
            "base_model": args.base_model,
            "checkpoint": args.checkpoint,
            "train_mode": args.train_mode,
            "top_k": args.top_k,
            "principal_alpha": args.principal_alpha,
            "timestamp": datetime.now().isoformat(),
        },
        "summary": {
            "layers": list(range(num_layers)),
            "modules": list(TARGET_MODULES),
            "nss": [],
            "max_principal_angle": [],
            "overlap_ratio": [],
            "relative_update_norm": [],
        },
        "detailed": {},
    }

    for layer_idx in range(num_layers):
        print(f"Processing layer {layer_idx}/{num_layers - 1}...")
        layer_nss = []
        layer_angles = []
        layer_overlap = []
        layer_norms = []

        for mod in TARGET_MODULES:
            # Load W_before
            base_key = get_base_weight_key(layer_idx, mod)
            W_before = load_tensor(base_index, base_key)
            if W_before is None:
                layer_nss.append(0.0)
                layer_angles.append(0.0)
                layer_overlap.append(0.0)
                layer_norms.append(0.0)
                continue

            W_before = W_before.float()

            # Reconstruct W_after
            ckpt_keys = get_checkpoint_keys(layer_idx, mod, args.train_mode)
            if args.train_mode == "blocktt":
                btt_l = load_tensor(ckpt_index, ckpt_keys["btt_l"])
                btt_r = load_tensor(ckpt_index, ckpt_keys["btt_r"])
                btt_s = load_tensor(ckpt_index, ckpt_keys["btt_s"])
                if btt_l is None or btt_r is None:
                    layer_nss.append(0.0)
                    layer_angles.append(0.0)
                    layer_overlap.append(0.0)
                    layer_norms.append(0.0)
                    continue
                W_after = materialize_blocktt_weight(btt_l.float(), btt_r.float(), btt_s=btt_s.float() if btt_s is not None else None)
            elif args.train_mode == "svd":
                svd_a = load_tensor(ckpt_index, ckpt_keys["svd_a"])
                svd_b = load_tensor(ckpt_index, ckpt_keys["svd_b"])
                svd_s = load_tensor(ckpt_index, ckpt_keys["svd_s"])
                if svd_a is None or svd_b is None:
                    layer_nss.append(0.0)
                    layer_angles.append(0.0)
                    layer_overlap.append(0.0)
                    layer_norms.append(0.0)
                    continue
                W_after = materialize_svd_weight(svd_a.float(), svd_b.float(), svd_s=svd_s.float() if svd_s is not None else None)
            elif args.train_mode == "lora":
                lora_A = load_tensor(ckpt_index, ckpt_keys["lora_A"])
                lora_B = load_tensor(ckpt_index, ckpt_keys["lora_B"])
                if lora_A is None or lora_B is None:
                    layer_nss.append(0.0)
                    layer_angles.append(0.0)
                    layer_overlap.append(0.0)
                    layer_norms.append(0.0)
                    continue
                W_after = reconstruct_lora_weight(W_before, lora_A.float(), lora_B.float(), lora_alpha, lora_r)

            delta_W = W_after - W_before

            # Compute spectra once (reuse for summary + detail)
            s_before, s_after, nss = compute_spectrum_and_nss(W_before, W_after)
            pa = compute_principal_angles(W_before, W_after, args.top_k)
            overlap, baseline = compute_principal_weight_overlap(
                W_before, delta_W, args.top_k, args.principal_alpha, args.update_threshold
            )
            rel_norm = float(torch.norm(delta_W) / torch.norm(W_before)) if torch.norm(W_before) > 0 else 0.0

            layer_nss.append(nss)
            layer_angles.append(max(pa) if pa else 0.0)
            layer_overlap.append(overlap)
            layer_norms.append(rel_norm)

            # Detailed metrics for selected layers
            if layer_idx in detailed_layers:
                angles_u, angles_v = compute_singular_vector_angles(W_before, W_after, args.top_k)
                row_norms, col_norms = compute_update_row_col_norms(delta_W)
                update_s = compute_update_spectrum(delta_W)

                detail_key = f"layer_{layer_idx}.{mod}"
                results["detailed"][detail_key] = {
                    "spectrum_before": s_before,
                    "spectrum_after": s_after,
                    "singular_vector_angles_u": angles_u,
                    "singular_vector_angles_v": angles_v,
                    "principal_angles": pa,
                    "row_norms": row_norms.tolist(),
                    "col_norms": col_norms.tolist(),
                    "update_spectrum": update_s,
                    "overlap_ratio": overlap,
                    "overlap_random_baseline": baseline,
                }

            # Free memory
            del W_before, W_after, delta_W

        results["summary"]["nss"].append(layer_nss)
        results["summary"]["max_principal_angle"].append(layer_angles)
        results["summary"]["overlap_ratio"].append(layer_overlap)
        results["summary"]["relative_update_norm"].append(layer_norms)

    return results


def main():
    args = parse_args()
    results = analyze(args)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run all tests**

```bash
python -m pytest tests/test_analyze_weights.py -v
```

Expected: all tests PASS

- [ ] **Step 6: Smoke test with real data (LoRA checkpoint)**

```bash
python analysis/analyze_weights.py \
  --base-model Qwen/Qwen3-1.7B \
  --checkpoint runs/26/step=25 \
  --train-mode lora \
  --layers 0,13 \
  --output analysis_results/lora_rl_test.json
```

Verify JSON is written with expected structure. Check that `nss`, `max_principal_angle` etc. contain reasonable non-zero values.

- [ ] **Step 7: Smoke test with BlockTT SFT checkpoint**

```bash
python analysis/analyze_weights.py \
  --base-model Qwen/Qwen3-4B \
  --checkpoint sft/blocktt_model \
  --train-mode blocktt \
  --layers 0,13 \
  --output analysis_results/blocktt_sft_test.json
```

Note: for BlockTT, the base model is `Qwen3-4B` (SFT uses 4B). Verify output.

- [ ] **Step 8: Commit**

```bash
git add analysis/analyze_weights.py tests/test_analyze_weights.py
git commit -m "feat: CLI and main analysis loop for analyze_weights.py"
```

---

## Chunk 2: HTML Report Generation

### Task 6: Report template HTML

**Files:**
- Create: `analysis/templates/report_template.html`

The HTML template with Plotly.js scaffolding. Uses a `DATA_PLACEHOLDER` token that `build_report.py` replaces with the actual JSON data.

- [ ] **Step 1: Create the HTML template**

Create `analysis/templates/report_template.html`. This is a single self-contained HTML file with 3 tabs (Summary, Detail, Trends). Data is injected as a `<script id="analysis-data" type="application/json">` block. Plotly.js loaded from CDN.

The file structure and Plotly trace configurations:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Weight Update Analysis</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    /* Reset and base */
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: system-ui, sans-serif; background: #f5f5f5; color: #333; }

    /* Header */
    #header { background: #1a1a2e; color: white; padding: 16px 24px; }
    #header h1 { font-size: 1.4rem; margin-bottom: 4px; }
    #header .meta { font-size: 0.85rem; opacity: 0.7; }

    /* Run selector (multi-run mode) */
    #run-selector { padding: 8px 24px; background: #e8e8e8; display: none; }

    /* Tabs */
    .tab-bar { display: flex; background: #2d2d44; }
    .tab-btn { padding: 10px 24px; color: #aaa; cursor: pointer; border: none;
               background: none; font-size: 0.95rem; }
    .tab-btn.active { color: white; border-bottom: 2px solid #5c6bc0; }
    .tab-content { display: none; padding: 24px; }
    .tab-content.active { display: block; }

    /* Summary grid: 2x2 heatmaps */
    .summary-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .summary-grid .plot-card { background: white; border-radius: 8px;
                               padding: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }

    /* Detail: two-column layout */
    .detail-controls { margin-bottom: 16px; }
    .detail-controls select { padding: 6px 12px; margin-right: 12px; }
    .detail-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .detail-grid .plot-card { background: white; border-radius: 8px;
                              padding: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }

    /* Trends: single column */
    .trends-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  </style>
</head>
<body>
  <div id="header">
    <h1 id="title">Weight Update Analysis</h1>
    <div class="meta" id="meta-info"></div>
  </div>
  <div id="run-selector">
    Run: <select id="run-select"></select>
  </div>
  <div class="tab-bar">
    <button class="tab-btn active" data-tab="summary">Summary</button>
    <button class="tab-btn" data-tab="detail">Detail</button>
    <button class="tab-btn" data-tab="trends">Trends</button>
  </div>

  <!-- Tab 1: Summary -->
  <div id="tab-summary" class="tab-content active">
    <div class="summary-grid">
      <div class="plot-card"><div id="heatmap-nss"></div></div>
      <div class="plot-card"><div id="heatmap-angle"></div></div>
      <div class="plot-card"><div id="heatmap-overlap"></div></div>
      <div class="plot-card"><div id="heatmap-norm"></div></div>
    </div>
  </div>

  <!-- Tab 2: Detail -->
  <div id="tab-detail" class="tab-content">
    <div class="detail-controls">
      Layer: <select id="detail-layer"></select>
      Module: <select id="detail-module"></select>
    </div>
    <div class="detail-grid">
      <div class="plot-card"><div id="plot-spectrum"></div></div>
      <div class="plot-card"><div id="plot-sv-angles"></div></div>
      <div class="plot-card"><div id="plot-principal-angles"></div></div>
      <div class="plot-card"><div id="plot-row-col-norms"></div></div>
      <div class="plot-card"><div id="plot-update-spectrum"></div></div>
      <div class="plot-card"><div id="plot-overlap-bar"></div></div>
    </div>
  </div>

  <!-- Tab 3: Trends -->
  <div id="tab-trends" class="tab-content">
    <div class="trends-grid">
      <div class="plot-card"><div id="trend-nss"></div></div>
      <div class="plot-card"><div id="trend-angle"></div></div>
      <div class="plot-card"><div id="trend-overlap"></div></div>
      <div class="plot-card"><div id="trend-norm"></div></div>
    </div>
  </div>

  <script id="analysis-data" type="application/json">DATA_PLACEHOLDER</script>
  <script>
    // === Data Loading ===
    const allData = JSON.parse(document.getElementById('analysis-data').textContent);
    let currentRunIdx = 0;
    const data = () => allData[currentRunIdx];

    // === Header ===
    function renderHeader() {
      const m = data().meta;
      document.getElementById('meta-info').textContent =
        `Model: ${m.base_model} | Mode: ${m.train_mode} | Checkpoint: ${m.checkpoint} | ${m.timestamp}`;
    }

    // === Multi-run selector ===
    if (allData.length > 1) {
      document.getElementById('run-selector').style.display = 'block';
      const sel = document.getElementById('run-select');
      allData.forEach((d, i) => {
        const opt = document.createElement('option');
        opt.value = i;
        opt.text = `${d.meta.train_mode} - ${d.meta.checkpoint}`;
        sel.add(opt);
      });
      sel.onchange = () => { currentRunIdx = parseInt(sel.value); renderAll(); };
    }

    // === Tab Switching ===
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.onclick = () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
      };
    });

    // === Tab 1: Summary Heatmaps ===
    function renderSummary() {
      const s = data().summary;
      const configs = [
        { id: 'heatmap-nss', data: s.nss, title: 'Normalized Spectral Shift', colorscale: 'YlOrRd' },
        { id: 'heatmap-angle', data: s.max_principal_angle, title: 'Max Principal Angle (deg)', colorscale: 'YlOrRd' },
        { id: 'heatmap-overlap', data: s.overlap_ratio, title: 'Overlap with Principal Weights', colorscale: 'RdYlGn' },
        { id: 'heatmap-norm', data: s.relative_update_norm, title: 'Relative Update Norm', colorscale: 'YlOrRd' },
      ];
      configs.forEach(cfg => {
        const trace = {
          z: cfg.data, x: s.modules, y: s.layers.map(String),
          type: 'heatmap', colorscale: cfg.colorscale,
          hovertemplate: 'Layer %{y}, %{x}<br>Value: %{z:.4f}<extra></extra>',
        };
        const layout = {
          title: cfg.title, height: 400,
          xaxis: { title: 'Module' }, yaxis: { title: 'Layer', autorange: 'reversed' },
          margin: { l: 60, r: 20, t: 40, b: 60 },
        };
        Plotly.newPlot(cfg.id, [trace], layout);
        // Click handler: navigate to detail tab
        document.getElementById(cfg.id).on('plotly_click', function(d) {
          const layerIdx = d.points[0].y;
          const modIdx = d.points[0].x;
          document.getElementById('detail-layer').value = layerIdx;
          document.getElementById('detail-module').value = modIdx;
          document.querySelector('[data-tab="detail"]').click();
          renderDetail();
        });
      });
    }

    // === Tab 2: Detail Plots ===
    function populateDetailDropdowns() {
      const s = data().summary;
      const layerSel = document.getElementById('detail-layer');
      const modSel = document.getElementById('detail-module');
      layerSel.innerHTML = '';
      modSel.innerHTML = '';
      // Only add layers that have detailed data
      const detailedKeys = Object.keys(data().detailed);
      const detailedLayers = [...new Set(detailedKeys.map(k => k.split('.')[0].replace('layer_','')))];
      detailedLayers.forEach(l => {
        const opt = document.createElement('option');
        opt.value = l; opt.text = 'Layer ' + l;
        layerSel.add(opt);
      });
      s.modules.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m; opt.text = m;
        modSel.add(opt);
      });
      layerSel.onchange = renderDetail;
      modSel.onchange = renderDetail;
    }

    function renderDetail() {
      const layer = document.getElementById('detail-layer').value;
      const mod = document.getElementById('detail-module').value;
      const key = `layer_${layer}.${mod}`;
      const d = data().detailed[key];
      if (!d) {
        ['plot-spectrum','plot-sv-angles','plot-principal-angles',
         'plot-row-col-norms','plot-update-spectrum','plot-overlap-bar']
          .forEach(id => Plotly.purge(id));
        return;
      }
      const compact = { l: 50, r: 20, t: 40, b: 50 };

      // Spectrum: before vs after (log Y)
      Plotly.newPlot('plot-spectrum', [
        { y: d.spectrum_before, name: 'Before', mode: 'lines' },
        { y: d.spectrum_after, name: 'After', mode: 'lines' },
      ], { title: 'Singular Value Spectrum', yaxis: { type: 'log', title: 'σ' },
           xaxis: { title: 'Index' }, height: 350, margin: compact });

      // Singular vector angles (U and V)
      Plotly.newPlot('plot-sv-angles', [
        { y: d.singular_vector_angles_u, name: 'Left (U)', mode: 'markers',
          marker: { size: d.spectrum_before.slice(0, d.singular_vector_angles_u.length)
                     .map(v => Math.max(3, Math.min(15, v / d.spectrum_before[0] * 15))) }},
        { y: d.singular_vector_angles_v, name: 'Right (V)', mode: 'markers' },
      ], { title: 'Singular Vector Rotation', yaxis: { title: 'Angle (deg)' },
           xaxis: { title: 'SV Index' }, height: 350, margin: compact });

      // Principal angles
      Plotly.newPlot('plot-principal-angles', [
        { y: d.principal_angles, type: 'bar' },
      ], { title: 'Principal Subspace Angles', yaxis: { title: 'Angle (deg)' },
           xaxis: { title: 'Index' }, height: 350, margin: compact });

      // Row/col norms as subplots (horizontal bars)
      Plotly.newPlot('plot-row-col-norms', [
        { y: d.row_norms, name: 'Row norms', type: 'bar', orientation: 'v' },
      ], { title: 'Row-wise Update Norms', height: 350, margin: compact,
           xaxis: { title: 'Row index' } });

      // Update spectrum
      Plotly.newPlot('plot-update-spectrum', [
        { y: d.update_spectrum, mode: 'lines+markers' },
      ], { title: 'SVD of ΔW', yaxis: { type: 'log', title: 'σ(ΔW)' },
           xaxis: { title: 'Index' }, height: 350, margin: compact });

      // Overlap bar
      Plotly.newPlot('plot-overlap-bar', [
        { x: ['Overlap', 'Random baseline'], y: [d.overlap_ratio, d.overlap_random_baseline],
          type: 'bar', marker: { color: ['#5c6bc0', '#bbb'] } },
      ], { title: 'Update ∩ Principal Weights', yaxis: { title: 'Ratio' },
           height: 350, margin: compact });
    }

    // === Tab 3: Cross-Layer Trends ===
    function renderTrends() {
      const s = data().summary;
      const metrics = [
        { id: 'trend-nss', data: s.nss, title: 'NSS Across Layers' },
        { id: 'trend-angle', data: s.max_principal_angle, title: 'Max Principal Angle Across Layers' },
        { id: 'trend-overlap', data: s.overlap_ratio, title: 'Overlap Ratio Across Layers' },
        { id: 'trend-norm', data: s.relative_update_norm, title: 'Relative Update Norm Across Layers' },
      ];
      metrics.forEach(cfg => {
        const traces = s.modules.map((mod, mi) => ({
          x: s.layers, y: cfg.data.map(row => row[mi]),
          name: mod, mode: 'lines+markers', marker: { size: 4 },
        }));
        Plotly.newPlot(cfg.id, traces, {
          title: cfg.title, height: 400,
          xaxis: { title: 'Layer' }, yaxis: { title: cfg.title.split(' ')[0] },
          margin: { l: 60, r: 20, t: 40, b: 50 },
          legend: { orientation: 'h', y: -0.2 },
        });
      });
    }

    // === Render All ===
    function renderAll() { renderHeader(); renderSummary(); populateDetailDropdowns(); renderDetail(); renderTrends(); }
    renderAll();
  </script>
</body>
</html>
```

This is the complete template. The implementing agent should write it as-is to `analysis/templates/report_template.html`.

- [ ] **Step 2: Commit template**

```bash
git add -f analysis/templates/report_template.html
git commit -m "feat: HTML report template with 3-tab Plotly.js layout"
```

---

### Task 7: Report builder script

**Files:**
- Create: `analysis/build_report.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_analyze_weights.py`:

```python
import tempfile
import os


class TestBuildReport(unittest.TestCase):
    def test_generates_html(self):
        from analysis.build_report import build_report

        # Create minimal valid JSON
        data = {
            "meta": {"base_model": "test", "checkpoint": "test", "train_mode": "lora",
                     "top_k": 4, "principal_alpha": 0.1, "timestamp": "2026-01-01"},
            "summary": {
                "layers": [0, 1],
                "modules": ["q_proj", "k_proj"],
                "nss": [[0.01, 0.02], [0.03, 0.04]],
                "max_principal_angle": [[1.0, 2.0], [3.0, 4.0]],
                "overlap_ratio": [[0.05, 0.06], [0.07, 0.08]],
                "relative_update_norm": [[0.001, 0.002], [0.003, 0.004]],
            },
            "detailed": {},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "test.json")
            html_path = os.path.join(tmpdir, "report.html")
            with open(json_path, "w") as f:
                json.dump(data, f)
            build_report([json_path], html_path)
            self.assertTrue(os.path.exists(html_path))
            with open(html_path) as f:
                content = f.read()
            self.assertIn("plotly", content.lower())
            self.assertIn("test", content)  # meta.base_model
```

- [ ] **Step 2: Implement build_report.py**

Create `analysis/build_report.py`:

```python
"""Generate an interactive HTML report from weight analysis JSON."""

import argparse
import json
from pathlib import Path


def build_report(data_paths: list[str], output_path: str):
    """Read analysis JSON files and generate HTML report.

    Args:
        data_paths: List of paths to JSON files from analyze_weights.py.
        output_path: Path to write the HTML report.
    """
    datasets = []
    for p in data_paths:
        with open(p) as f:
            datasets.append(json.load(f))

    # Load template
    template_path = Path(__file__).parent / "templates" / "report_template.html"
    template = template_path.read_text()

    # Embed data as JSON
    data_json = json.dumps(datasets)
    html = template.replace("DATA_PLACEHOLDER", data_json)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    print(f"Report written to {output_path}")


def main():
    p = argparse.ArgumentParser(description="Generate HTML report from analysis JSON")
    p.add_argument("--data", nargs="+", required=True, help="Path(s) to analysis JSON files")
    p.add_argument("--output", default="analysis_results/report.html")
    args = p.parse_args()
    build_report(args.data, args.output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/test_analyze_weights.py::TestBuildReport -v
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add analysis/build_report.py tests/test_analyze_weights.py
git commit -m "feat: build_report.py generates HTML from analysis JSON"
```

---

### Task 8: End-to-end integration test

**Files:**
- Modify: `tests/test_analyze_weights.py`

- [ ] **Step 1: Write an end-to-end test with synthetic data**

This test creates small synthetic "base model" and "checkpoint" safetensors files, runs the full analysis pipeline, generates HTML, and verifies the output is valid.

Add to `tests/test_analyze_weights.py`:

```python
from safetensors.torch import save_file


class TestEndToEnd(unittest.TestCase):
    def test_lora_pipeline(self):
        """Full pipeline: create synthetic weights -> analyze -> build report."""
        from analysis.analyze_weights import parse_args, analyze
        from analysis.build_report import build_report

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = os.path.join(tmpdir, "base")
            ckpt_dir = os.path.join(tmpdir, "ckpt")
            os.makedirs(base_dir)
            os.makedirs(ckpt_dir)

            # Create synthetic base model weights (2 layers, q_proj + gate_proj)
            torch.manual_seed(42)
            base_tensors = {}
            for l in range(2):
                base_tensors[f"model.layers.{l}.self_attn.q_proj.weight"] = torch.randn(16, 8)
                base_tensors[f"model.layers.{l}.self_attn.k_proj.weight"] = torch.randn(8, 8)
                base_tensors[f"model.layers.{l}.self_attn.v_proj.weight"] = torch.randn(8, 8)
                base_tensors[f"model.layers.{l}.self_attn.o_proj.weight"] = torch.randn(8, 16)
                base_tensors[f"model.layers.{l}.mlp.gate_proj.weight"] = torch.randn(32, 8)
                base_tensors[f"model.layers.{l}.mlp.up_proj.weight"] = torch.randn(32, 8)
                base_tensors[f"model.layers.{l}.mlp.down_proj.weight"] = torch.randn(8, 32)
            save_file(base_tensors, os.path.join(base_dir, "model.safetensors"))

            # Create synthetic LoRA adapter
            ckpt_tensors = {}
            for l in range(2):
                for mod, out_dim, in_dim in [
                    ("self_attn.q_proj", 16, 8), ("self_attn.k_proj", 8, 8),
                    ("self_attn.v_proj", 8, 8), ("self_attn.o_proj", 8, 16),
                    ("mlp.gate_proj", 32, 8), ("mlp.up_proj", 32, 8),
                    ("mlp.down_proj", 8, 32),
                ]:
                    r = 2
                    ckpt_tensors[f"base_model.model.model.layers.{l}.{mod}.lora_A.weight"] = torch.randn(r, in_dim)
                    ckpt_tensors[f"base_model.model.model.layers.{l}.{mod}.lora_B.weight"] = torch.randn(out_dim, r)
            save_file(ckpt_tensors, os.path.join(ckpt_dir, "adapter_model.safetensors"))

            # Write adapter config
            with open(os.path.join(ckpt_dir, "adapter_config.json"), "w") as f:
                json.dump({"lora_alpha": 4, "r": 2}, f)

            # Run analysis
            args = parse_args([
                "--base-model", base_dir,
                "--checkpoint", ckpt_dir,
                "--train-mode", "lora",
                "--top-k", "4",
                "--layers", "0,1",
                "--output", os.path.join(tmpdir, "results.json"),
            ])
            results = analyze(args)

            # Verify results structure
            self.assertEqual(len(results["summary"]["nss"]), 2)  # 2 layers
            self.assertEqual(len(results["summary"]["nss"][0]), 7)  # 7 modules
            self.assertIn("layer_0.q_proj", results["detailed"])

            # Write JSON and build report
            json_path = os.path.join(tmpdir, "results.json")
            with open(json_path, "w") as f:
                json.dump(results, f)
            html_path = os.path.join(tmpdir, "report.html")
            build_report([json_path], html_path)
            self.assertTrue(os.path.exists(html_path))
```

- [ ] **Step 2: Run end-to-end test**

```bash
python -m pytest tests/test_analyze_weights.py::TestEndToEnd -v
```

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_analyze_weights.py
git commit -m "test: end-to-end integration test for analysis pipeline"
```

---

### Task 9: Run on real data and verify report

**Files:** None (manual verification)

- [ ] **Step 1: Run analysis on LoRA RL checkpoint**

```bash
python analysis/analyze_weights.py \
  --base-model Qwen/Qwen3-1.7B \
  --checkpoint runs/26/step=25 \
  --train-mode lora \
  --output analysis_results/lora_rl_r64.json
```

- [ ] **Step 2: Generate HTML report**

```bash
python analysis/build_report.py \
  --data analysis_results/lora_rl_r64.json \
  --output analysis_results/lora_rl_r64.html
```

- [ ] **Step 3: Open report and verify**

```bash
xdg-open analysis_results/lora_rl_r64.html
```

Check:
- Summary heatmaps render with data
- Clicking a cell switches to detail tab
- Spectrum plots show two curves
- Singular vector angle plots show expected pattern
- Cross-layer trends have 7 lines (one per module)

- [ ] **Step 4: Run on BlockTT SFT checkpoint**

```bash
python analysis/analyze_weights.py \
  --base-model Qwen/Qwen3-4B \
  --checkpoint sft/blocktt_model \
  --train-mode blocktt \
  --output analysis_results/blocktt_sft.json

python analysis/build_report.py \
  --data analysis_results/blocktt_sft.json \
  --output analysis_results/blocktt_sft.html
```

- [ ] **Step 5: Test multi-run comparison report**

```bash
python analysis/build_report.py \
  --data analysis_results/lora_rl_r64.json analysis_results/blocktt_sft.json \
  --output analysis_results/comparison.html
```

Verify side-by-side comparison works.

- [ ] **Step 6: Final commit with any fixes**

```bash
git add analysis/ tests/
git commit -m "feat: complete weight analysis pipeline with report generation"
```
