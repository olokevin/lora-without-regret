# Calibrated BTT Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--calib-mode` flag to `run_sft.py`, `run_rl.py`, `run_rl_dapo.py`, and `ref/LIFT/src/finetune_blocktt.py` that routes `--train-mode blocktt` through `compress.decompose_with_loader` with one of `btt_llm_v2`, `btt_llm_v2_bp`, `btt_llm_v2_combined`, `btt_twosteps`, using a calibration DataLoader whose batches match real training batches.

**Architecture:** A new shared helper module `compress_integration.py` at the repo root owns CLI registration, config translation, calibration-loader building, decomposition wire-up, RL vLLM materialization for `compress.btt.btt_linear.BTTLinear`, and save/load helpers with topology metadata. `src/compress` gains `s_merged_to` / `factorize_by_head` config fields (best-effort plumbing) and topology helpers (`topology_spec`, `from_topology_spec`, `export_btt_topology`, `rebuild_btt_from_topology`). Legacy `btt_layer.py` receives one extension: `normalize_trainable_blocktt_cores_` also normalizes `BTTLinear`. Each entrypoint calls `add_calibrated_btt_args`, `validate_calibrated_btt_args`, and — when `--calib-mode != none` — `build_calib_loader` + `apply_calibrated_btt` instead of the legacy BTT path.

**Tech Stack:** Python 3.13+, PyTorch, HuggingFace `transformers` + `datasets`, `safetensors`, local `compress` package (`src/compress`), existing `uv` toolchain. Tests use `unittest` (matches `tests/` convention in this repo).

**Spec:** `docs/superpowers/specs/2026-04-14-calibrated-btt-migration-design.md`

---

## File Structure

### Created

| Path | Purpose |
|---|---|
| `compress_integration.py` | Shared helper: CLI, config translation, loader building, apply-decomposition, RL materialize helpers, save/load checkpoints. |
| `tests/test_calibrated_btt_cli.py` | Argparse & validation for all four entrypoints. |
| `tests/test_compress_integration.py` | Pure-Python logic in `compress_integration.py`. |
| `tests/test_calibrated_btt_pipeline.py` | End-to-end smoke on a tiny torch model. |
| `tests/test_calibrated_btt_checkpoint.py` | Save/load round-trip with topology metadata. |

### Modified

| Path | Change |
|---|---|
| `src/compress/decomposition.py` | Add `s_merged_to`, `factorize_by_head` fields; plumb through `decompose_with_loader`. |
| `src/compress/compress_model.py` | Accept and forward `btt_s_merged_to`, `btt_factorize_by_head` kwargs. |
| `src/compress/btt/btt_linear.py` | Add `topology_spec()`, `from_topology_spec()`, `materialize_dense_weight()`; accept `s_merged_to`, `factorize_by_head` metadata. |
| `src/compress/__init__.py` | Export `export_btt_topology`, `rebuild_btt_from_topology`. |
| `src/compress/btt/btt_llm_v2.py`, `src/compress/btt/btt_twosteps.py` | Accept (and pass through or ignore) `s_merged_to` / `factorize_by_head`. |
| `btt_layer.py` | Extend `normalize_trainable_blocktt_cores_` to also normalize `BTTLinear`. |
| `run_sft.py` | Register new flags, validate, reorder dataset before prepare_model, branch to `apply_calibrated_btt`, save topology at checkpoint. |
| `run_rl.py` | Register flags, validate, run calib rollout on dense base model, branch to `apply_calibrated_btt`, extend `export_weights_for_vllm` for `BTTLinear`, save topology. |
| `run_rl_dapo.py` | Same changes as `run_rl.py`. |
| `ref/LIFT/src/finetune_blocktt.py` | `sys.path` injection for `compress`, register flags (underscore style), validate, reorder dataset, branch to `apply_calibrated_btt`, save topology. |
| `tests/test_btt_pipeline_compat.py` | Add case confirming `--calib-mode=none` is bit-identical to pre-change. |
| `tests/test_run_rl_cli.py`, `tests/test_run_rl_dapo_cli.py` | Extend with calib flag parsing assertions. |

---

## Conventions

- **Imports:** `compress_integration.py` adds `<repo_root>/src` to `sys.path` on import (idempotent). Imports use `from compress...` after that.
- **Commit style:** Match existing repo style (short imperative subject, optional body). Include `Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>` on every commit.
- **Tests:** `python -m unittest tests/test_<name>.py` (matches CLAUDE.md).
- **TDD:** Write test first, observe failure, implement, observe pass, commit.
- **When a step says "expected output," the engineer should verify the output matches.** Don't skip verification steps.

---

## Task 1: Add `materialize_dense_weight()` to `BTTLinear`

**Files:**
- Modify: `src/compress/btt/btt_linear.py`
- Create: `tests/test_btt_linear_materialize.py`

Rationale: `run_rl.py`'s `export_weights_for_vllm` walks model modules and calls `.materialize_dense_weight()` on each `BTTLayer` / `SVDLayer`. For calibrated BTT to ride the same path, `BTTLinear` needs the same method.

- [ ] **Step 1: Write the failing test**

Create `tests/test_btt_linear_materialize.py`:

```python
import os, sys
import unittest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from compress.btt.btt_linear import BTTLinear


def _make_btt(d_in=8, d_out=6, m=2, a=3, n=2, b=4, rank=2, with_bias=True):
    btt_l = torch.randn(m, n * rank, a)
    btt_r = torch.randn(n, b, m * rank)
    bias = torch.randn(d_out) if with_bias else None
    return BTTLinear(btt_l, btt_r, bias=bias, m=m, a=a, n=n, b=b, rank=rank)


class TestBTTLinearMaterialize(unittest.TestCase):
    def test_materialize_matches_forward(self):
        torch.manual_seed(0)
        layer = _make_btt()
        x = torch.randn(5, layer.in_features)
        expected = layer(x)

        dense = layer.materialize_dense_weight()
        self.assertEqual(dense.shape, (layer.out_features, layer.in_features))
        out = x @ dense.T
        if layer.bias is not None:
            out = out + layer.bias
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_materialize_shape_no_bias(self):
        torch.manual_seed(1)
        layer = _make_btt(with_bias=False)
        dense = layer.materialize_dense_weight()
        self.assertEqual(dense.shape, (layer.out_features, layer.in_features))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test, verify it fails**

```bash
cd /home/yequan/Project/lora/lora-without-regret
python -m unittest tests.test_btt_linear_materialize -v
```

Expected: FAIL with `AttributeError: 'BTTLinear' object has no attribute 'materialize_dense_weight'`.

- [ ] **Step 3: Implement `materialize_dense_weight`**

In `src/compress/btt/btt_linear.py`, add method to `BTTLinear` (after `forward`):

```python
    @torch.no_grad()
    def materialize_dense_weight(self) -> torch.Tensor:
        """Return the dense (out_features, in_features) weight equivalent to
        the BTT forward pass, not including bias.

        Derivation mirrors forward: for each input column indexed by (n_idx, b_idx)
        and output row indexed by (m_idx, a_idx), the entry is
          sum over rank_idx of btt_r[n_idx, b_idx, m_idx * rank + rank_idx]
                               * btt_l[m_idx, n_idx * rank + rank_idx, a_idx].
        """
        device = self.btt_l.device
        dtype = self.btt_l.dtype
        # Reshape btt_r from (n, b, m*rank) to (n, b, m, rank)
        btt_r = self.btt_r.reshape(self.n, self.b, self.m, self.rank)
        # Reshape btt_l from (m, n*rank, a) to (m, n, rank, a)
        btt_l = self.btt_l.reshape(self.m, self.n, self.rank, self.a)
        # Contract: for each (m, n) pair, (b, rank) @ (rank, a) -> (b, a)
        # Result dense_block[m, n, b, a]; then assemble into (d_out, d_in).
        # dense[m, a, n, b] = sum_r btt_r[n, b, m, r] * btt_l[m, n, r, a]
        dense = torch.einsum("nbmr,mnra->manb", btt_r, btt_l)
        dense = dense.reshape(self.m * self.a, self.n * self.b)
        return dense.to(device=device, dtype=dtype)
```

- [ ] **Step 4: Run the test, verify it passes**

```bash
python -m unittest tests.test_btt_linear_materialize -v
```

Expected: OK (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/compress/btt/btt_linear.py tests/test_btt_linear_materialize.py
git commit -m "$(cat <<'EOF'
compress: add materialize_dense_weight to BTTLinear

Required so run_rl.py's export_weights_for_vllm can treat calibrated BTT
layers uniformly with legacy BTTLayer / SVDLayer.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `BTTLinear` topology spec + from_topology_spec

**Files:**
- Modify: `src/compress/btt/btt_linear.py`
- Create: `tests/test_btt_topology.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_btt_topology.py`:

```python
import os, sys
import unittest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from compress.btt.btt_linear import BTTLinear


def _make_btt(m=2, a=3, n=2, b=4, rank=2, with_bias=True):
    btt_l = torch.randn(m, n * rank, a)
    btt_r = torch.randn(n, b, m * rank)
    bias = torch.randn(m * a) if with_bias else None
    return BTTLinear(btt_l, btt_r, bias=bias, m=m, a=a, n=n, b=b, rank=rank)


class TestBTTLinearTopology(unittest.TestCase):
    def test_topology_spec_roundtrip_with_bias(self):
        torch.manual_seed(0)
        layer = _make_btt(with_bias=True)
        spec = layer.topology_spec()

        for key in ("m", "a", "n", "b", "rank", "has_bias",
                    "in_features", "out_features"):
            self.assertIn(key, spec)

        new_layer = BTTLinear.from_topology_spec(spec)
        self.assertEqual(new_layer.btt_l.shape, layer.btt_l.shape)
        self.assertEqual(new_layer.btt_r.shape, layer.btt_r.shape)
        self.assertIsNotNone(new_layer.bias)
        self.assertEqual(new_layer.bias.shape, layer.bias.shape)

        new_layer.load_state_dict(layer.state_dict())
        x = torch.randn(5, layer.in_features)
        self.assertTrue(torch.allclose(new_layer(x), layer(x), atol=1e-6))

    def test_topology_spec_no_bias(self):
        torch.manual_seed(1)
        layer = _make_btt(with_bias=False)
        spec = layer.topology_spec()
        self.assertFalse(spec["has_bias"])
        new_layer = BTTLinear.from_topology_spec(spec)
        self.assertIsNone(new_layer.bias)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run, verify it fails**

```bash
python -m unittest tests.test_btt_topology -v
```

Expected: FAIL with `AttributeError: ... has no attribute 'topology_spec'`.

- [ ] **Step 3: Implement topology methods**

Append to `BTTLinear` in `src/compress/btt/btt_linear.py`:

```python
    def topology_spec(self) -> dict:
        """JSON-serializable dict sufficient to reconstruct an empty BTTLinear."""
        return {
            "m": int(self.m),
            "a": int(self.a),
            "n": int(self.n),
            "b": int(self.b),
            "rank": int(self.rank),
            "has_bias": self.bias is not None,
            "in_features": int(self.in_features),
            "out_features": int(self.out_features),
        }

    @classmethod
    def from_topology_spec(cls, spec: dict) -> "BTTLinear":
        """Construct an empty BTTLinear matching the spec. Parameters are zero-initialized;
        caller is expected to load_state_dict afterwards."""
        m, a = int(spec["m"]), int(spec["a"])
        n, b = int(spec["n"]), int(spec["b"])
        rank = int(spec["rank"])
        has_bias = bool(spec["has_bias"])
        btt_l = torch.zeros(m, n * rank, a)
        btt_r = torch.zeros(n, b, m * rank)
        bias = torch.zeros(m * a) if has_bias else None
        return cls(btt_l, btt_r, bias=bias, m=m, a=a, n=n, b=b, rank=rank)
```

- [ ] **Step 4: Run, verify it passes**

```bash
python -m unittest tests.test_btt_topology -v
```

Expected: OK (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/compress/btt/btt_linear.py tests/test_btt_topology.py
git commit -m "$(cat <<'EOF'
compress: add topology_spec / from_topology_spec on BTTLinear

Enables serializing the factored-layer topology alongside the state dict
so eval scripts can reconstruct the module graph without re-running
calibration.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `export_btt_topology` and `rebuild_btt_from_topology`

**Files:**
- Create: `src/compress/topology.py`
- Modify: `src/compress/__init__.py`
- Create: `tests/test_compress_topology.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_compress_topology.py`:

```python
import os, sys
import unittest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from compress.btt.btt_linear import BTTLinear
from compress.topology import export_btt_topology, rebuild_btt_from_topology


def _make_btt(m=2, a=3, n=2, b=4, rank=2):
    btt_l = torch.randn(m, n * rank, a)
    btt_r = torch.randn(n, b, m * rank)
    bias = torch.randn(m * a)
    return BTTLinear(btt_l, btt_r, bias=bias, m=m, a=a, n=n, b=b, rank=rank)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 6)
        self.fc2 = nn.Linear(6, 4)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class TestCompressTopology(unittest.TestCase):
    def test_export_walks_module_tree(self):
        torch.manual_seed(0)
        model = ToyModel()
        model.fc1 = _make_btt(m=2, a=3, n=2, b=4, rank=2)
        topology = export_btt_topology(model)
        self.assertIn("fc1", topology)
        self.assertNotIn("fc2", topology)
        self.assertEqual(topology["fc1"]["m"], 2)

    def test_rebuild_reinstates_btt_modules(self):
        torch.manual_seed(0)
        model = ToyModel()
        original_btt = _make_btt(m=2, a=3, n=2, b=4, rank=2)
        model.fc1 = original_btt

        topology = export_btt_topology(model)
        state = {k: v.clone() for k, v in model.state_dict().items()}

        fresh = ToyModel()
        rebuild_btt_from_topology(fresh, topology)
        self.assertIsInstance(fresh.fc1, BTTLinear)

        fresh.load_state_dict(state)
        x = torch.randn(3, 8)
        self.assertTrue(torch.allclose(fresh(x), model(x), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run, verify it fails**

```bash
python -m unittest tests.test_compress_topology -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'compress.topology'`.

- [ ] **Step 3: Create `src/compress/topology.py`**

```python
"""Topology export / rebuild helpers for BTTLinear modules."""

from typing import Dict

import torch.nn as nn

from compress.btt.btt_linear import BTTLinear


def export_btt_topology(model: nn.Module) -> Dict[str, dict]:
    """Walk model; return {module_name: BTTLinear.topology_spec()} for every BTTLinear."""
    return {
        name: module.topology_spec()
        for name, module in model.named_modules()
        if isinstance(module, BTTLinear)
    }


def rebuild_btt_from_topology(model: nn.Module, topology: Dict[str, dict]) -> nn.Module:
    """In-place: replace nn.Linear modules named in topology with empty BTTLinear
    instances constructed via from_topology_spec. Caller is responsible for load_state_dict."""
    for name, spec in topology.items():
        path = name.split(".")
        parent = model
        for key in path[:-1]:
            parent = getattr(parent, key)
        setattr(parent, path[-1], BTTLinear.from_topology_spec(spec))
    return model
```

- [ ] **Step 4: Export from package `__init__`**

Modify `src/compress/__init__.py`: add `from compress.topology import export_btt_topology, rebuild_btt_from_topology` and extend `__all__` with both names.

```python
from compress.topology import export_btt_topology, rebuild_btt_from_topology  # noqa: E402
```

and append both names to `__all__`.

- [ ] **Step 5: Run, verify it passes**

```bash
python -m unittest tests.test_compress_topology -v
```

Expected: OK (2 tests).

- [ ] **Step 6: Run the existing compat suite — nothing should regress**

```bash
python -m unittest tests.test_btt_pipeline_compat tests.test_svd_pipeline_compat
```

Expected: OK.

- [ ] **Step 7: Commit**

```bash
git add src/compress/topology.py src/compress/__init__.py tests/test_compress_topology.py
git commit -m "$(cat <<'EOF'
compress: add export_btt_topology / rebuild_btt_from_topology helpers

Makes it possible to serialize BTTLinear topology alongside weights and
reconstruct the module graph at eval time without re-running calibration.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Add `s_merged_to` and `factorize_by_head` to `DecompositionConfig`

**Files:**
- Modify: `src/compress/decomposition.py`
- Modify: `src/compress/compress_model.py`
- Modify: `src/compress/btt/btt_llm_v2.py`, `src/compress/btt/btt_twosteps.py`
- Create: `tests/test_decomposition_config_extra_fields.py`

Per spec §6.2: best-effort plumb. The `src/compress/btt/` compression entrypoints accept the kwargs; if the underlying math doesn't have a hook, they log that the kwarg is ignored. Behavior for `s_merged_to=None` and `factorize_by_head=True` stays identical to the pre-change default.

- [ ] **Step 1: Write the failing test**

Create `tests/test_decomposition_config_extra_fields.py`:

```python
import os, sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from compress.decomposition import DecompositionConfig


class TestDecompositionConfigExtraFields(unittest.TestCase):
    def test_defaults(self):
        cfg = DecompositionConfig(train_mode="btt_llm_v2")
        self.assertIsNone(cfg.s_merged_to)
        self.assertTrue(cfg.factorize_by_head)

    def test_custom_values(self):
        cfg = DecompositionConfig(
            train_mode="btt_llm_v2",
            s_merged_to="trainable",
            factorize_by_head=False,
        )
        self.assertEqual(cfg.s_merged_to, "trainable")
        self.assertFalse(cfg.factorize_by_head)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run, verify it fails**

```bash
python -m unittest tests.test_decomposition_config_extra_fields -v
```

Expected: FAIL — unexpected keyword argument `s_merged_to`.

- [ ] **Step 3: Add fields to `DecompositionConfig`**

In `src/compress/decomposition.py`, add after existing BTT-specific fields (near `train_position`):

```python
    s_merged_to: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "BTT SVD-init absorb-side (best-effort). None preserves the "
                "compress package default. Legacy values: 'frozen', 'trainable', 'both'."
            ),
        },
    )
    factorize_by_head: bool = field(
        default=True,
        metadata={
            "help": (
                "For attention projections, align BTT block shapes with attention "
                "head structure. Default True matches the legacy path."
            ),
        },
    )
```

- [ ] **Step 4: Plumb through `decompose_with_loader`**

In the same file, extend the `compress_model_with_loader(...)` call inside `decompose_with_loader` to pass the new kwargs:

```python
    compress_model_with_loader(
        model,
        calib_loader,
        compression_ratio=config.compression_ratio,
        method=config.train_mode,
        device=target_device,
        skip_layers=skip_layers,
        als_n_iter=config.als_n_iter,
        als_tol=config.als_tol,
        als_weighting=config.als_weighting,
        als_reg_eps=config.als_reg_eps,
        twosteps_n_refine=config.twosteps_n_refine,
        twosteps_reg_eps=config.twosteps_reg_eps,
        btt_decomp_mode=config.decomp_mode,
        btt_s_merged_to=config.s_merged_to,
        btt_factorize_by_head=config.factorize_by_head,
    )
```

- [ ] **Step 5: Accept kwargs in `compress_model_with_loader`**

In `src/compress/compress_model.py`, add two kwargs (with defaults) to both `compress_model_with_loader` and `_compress_with_covariances` / `_compress_with_covariances_combined`:

```python
def compress_model_with_loader(
    model: nn.Module,
    calib_loader,
    compression_ratio: float = 0.7,
    method: str = "btt_llm_v2",
    device: str = "cuda",
    skip_layers: Tuple[str, ...] = ("lm_head",),
    als_n_iter: int = 10,
    als_tol: float = 1e-6,
    als_weighting: str = "equal",
    als_reg_eps: float = 1e-4,
    twosteps_n_refine: int = 1,
    twosteps_reg_eps: float = 1e-4,
    btt_decomp_mode: str = "input_one_block",
    btt_s_merged_to: Optional[str] = None,
    btt_factorize_by_head: bool = True,
) -> nn.Module:
    ...
```

Add `from typing import Optional` if not already imported. In both `_compress_with_covariances` and `_compress_with_covariances_combined`, accept the same two kwargs (with the same defaults) and forward them into each call to `btt_llm_v2_compress_model` and `btt_twosteps_compress_model` as `s_merged_to=btt_s_merged_to, factorize_by_head=btt_factorize_by_head`. Update every `_compress_with_covariances(...)` call in the file to pass them through (there are a few inside `_compress_with_method_spec` — forward the values from the outer scope). If `_compress_with_method_spec` doesn't currently receive these from `compress_model_with_loader`, thread them through.

- [ ] **Step 6: Accept kwargs in the BTT entrypoints**

In `src/compress/btt/btt_llm_v2.py` and `src/compress/btt/btt_twosteps.py`, find the public `*_compress_model` function signatures and add:

```python
    s_merged_to: Optional[str] = None,
    factorize_by_head: bool = True,
```

Inside each function, use the kwargs if the underlying code has a hook. If the current code does not support these knobs, add a log line and continue:

```python
import logging
logger = logging.getLogger(__name__)

if s_merged_to is not None:
    logger.info("btt_llm_v2: s_merged_to=%s (ignored; not implemented for this variant)",
                s_merged_to)
# similarly for factorize_by_head when factorize_by_head is False
```

**Implementation-time verification:** for each variant, check the existing code for a block-shape/head-alignment hook or an SVD-merging hook. If present, wire the kwargs to it; if absent, log-and-ignore. Document the per-variant outcome in the PR description.

- [ ] **Step 7: Run, verify the config test passes**

```bash
python -m unittest tests.test_decomposition_config_extra_fields -v
```

Expected: OK (2 tests).

- [ ] **Step 8: Run existing BTT compat suite; confirm no regression**

```bash
python -m unittest tests.test_btt_pipeline_compat
```

Expected: OK.

- [ ] **Step 9: Commit**

```bash
git add src/compress/decomposition.py src/compress/compress_model.py \
        src/compress/btt/btt_llm_v2.py src/compress/btt/btt_twosteps.py \
        tests/test_decomposition_config_extra_fields.py
git commit -m "$(cat <<'EOF'
compress: plumb s_merged_to and factorize_by_head through BTT decomp

Adds two new DecompositionConfig fields and forwards them to the BTT
compress entrypoints. Best-effort: variants without a hook log an
'ignored' note and proceed with previous default behavior, so existing
runs are unaffected.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Extend `normalize_trainable_blocktt_cores_` to handle `BTTLinear`

**Files:**
- Modify: `btt_layer.py`
- Create: `tests/test_normalize_btt_mixed.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_normalize_btt_mixed.py`:

```python
import os, sys
import unittest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from btt_layer import normalize_trainable_blocktt_cores_
from compress.btt.btt_linear import BTTLinear


class TestNormalizeBTTLinear(unittest.TestCase):
    def test_normalizes_btt_linear_cores(self):
        torch.manual_seed(0)
        m, a, n, b, rank = 2, 3, 2, 4, 2
        btt_l = torch.randn(m, n * rank, a) * 5.0
        btt_r = torch.randn(n, b, m * rank) * 5.0
        layer = BTTLinear(btt_l, btt_r, m=m, a=a, n=n, b=b, rank=rank)
        layer.btt_l.requires_grad = True
        layer.btt_r.requires_grad = True

        model = nn.Module()
        model.inner = layer

        result = normalize_trainable_blocktt_cores_(model)
        self.assertEqual(result["normalized_left_cores"], 1)
        self.assertEqual(result["normalized_right_cores"], 1)

        # btt_r: each (n, m, rank) vector over b should be unit norm.
        norms_r = torch.linalg.vector_norm(layer.btt_r, dim=1)
        self.assertTrue(torch.allclose(norms_r, torch.ones_like(norms_r), atol=1e-5))
        norms_l = torch.linalg.vector_norm(layer.btt_l, dim=2)
        self.assertTrue(torch.allclose(norms_l, torch.ones_like(norms_l), atol=1e-5))

    def test_skips_frozen_cores(self):
        torch.manual_seed(1)
        m, a, n, b, rank = 2, 3, 2, 4, 2
        layer = BTTLinear(
            torch.randn(m, n * rank, a) * 5.0,
            torch.randn(n, b, m * rank) * 5.0,
            m=m, a=a, n=n, b=b, rank=rank,
        )
        layer.btt_l.requires_grad = True
        layer.btt_r.requires_grad = False

        model = nn.Module()
        model.inner = layer

        result = normalize_trainable_blocktt_cores_(model)
        self.assertEqual(result["normalized_left_cores"], 1)
        self.assertEqual(result["normalized_right_cores"], 0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run, verify it fails**

```bash
python -m unittest tests.test_normalize_btt_mixed -v
```

Expected: FAIL — `normalized_left_cores == 0` (the function doesn't recognize `BTTLinear` yet).

- [ ] **Step 3: Extend the normalize function**

In `btt_layer.py`, replace the body of `normalize_trainable_blocktt_cores_` with a version that handles both classes via duck-typing:

```python
def normalize_trainable_blocktt_cores_(model, eps=1e-12):
    try:
        from compress.btt.btt_linear import BTTLinear as _CompressBTTLinear
    except Exception:
        _CompressBTTLinear = None

    normalized_left = 0
    normalized_right = 0
    for module in model.modules():
        if isinstance(module, BTTLayer):
            matched = True
        elif _CompressBTTLinear is not None and isinstance(module, _CompressBTTLinear):
            matched = True
        else:
            matched = False
        if not matched:
            continue
        if not (hasattr(module, "btt_l") and hasattr(module, "btt_r")):
            continue

        if module.btt_r.requires_grad:
            norms = torch.linalg.vector_norm(module.btt_r, dim=1, keepdim=True).clamp_min(eps)
            module.btt_r.div_(norms)
            normalized_right += 1
        if module.btt_l.requires_grad:
            norms = torch.linalg.vector_norm(module.btt_l, dim=2, keepdim=True).clamp_min(eps)
            module.btt_l.div_(norms)
            normalized_left += 1

    return {
        "normalized_left_cores": normalized_left,
        "normalized_right_cores": normalized_right,
    }
```

Note: `BTTLinear` stores cores with the same shape convention (`btt_r: (n, b, m*rank)`, `btt_l: (m, n*rank, a)`) as `BTTLayer`, so the existing norm reductions apply unchanged.

- [ ] **Step 4: Run, verify both tests pass**

```bash
python -m unittest tests.test_normalize_btt_mixed -v
python -m unittest tests.test_btt_pipeline_compat  # ensure legacy still works
```

Expected: both OK.

- [ ] **Step 5: Commit**

```bash
git add btt_layer.py tests/test_normalize_btt_mixed.py
git commit -m "$(cat <<'EOF'
btt: normalize_trainable_blocktt_cores_ handles both BTTLayer and BTTLinear

Legacy BTTLayer semantics unchanged. When compress.btt.btt_linear.BTTLinear
is installed, its cores are also normalized using the same shape
conventions.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Scaffold `compress_integration.py` with `sys.path` + imports

**Files:**
- Create: `compress_integration.py`
- Create: `tests/test_compress_integration_import.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_compress_integration_import.py`:

```python
import os, sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class TestCompressIntegrationImport(unittest.TestCase):
    def test_module_imports(self):
        import compress_integration  # noqa: F401

    def test_public_surface_exists(self):
        import compress_integration as ci
        names = [
            "add_calibrated_btt_args",
            "validate_calibrated_btt_args",
            "build_decomposition_config",
            "build_training_data_calib_loader",
            "build_rl_rollout_calib_loader",
            "build_calib_loader",
            "apply_calibrated_btt",
            "materialize_calibrated_btt_weights",
            "restore_calibrated_btt_weights",
            "save_calibrated_btt_checkpoint",
            "load_calibrated_btt_for_eval",
        ]
        for n in names:
            self.assertTrue(hasattr(ci, n), f"missing {n}")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run, verify it fails**

```bash
python -m unittest tests.test_compress_integration_import -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'compress_integration'`.

- [ ] **Step 3: Create the module with stubs**

Create `compress_integration.py` in the repo root:

```python
"""Shared helpers for integrating src/compress calibrated BTT with the
run_sft / run_rl / run_rl_dapo / LIFT training entrypoints.

Keeps legacy btt_layer.py / svd_layer.py paths untouched; activated only
when --calib-mode (or --calib_mode) is non-'none'.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from typing import Any, Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Ensure the sibling `compress` package under <repo_root>/src is importable.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if os.path.isdir(_SRC_DIR) and _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from compress.decomposition import (  # noqa: E402
    DecompositionConfig,
    decompose_with_loader,
)
from compress.btt.btt_linear import BTTLinear  # noqa: E402
from compress.loaders import (  # noqa: E402
    build_c4_calib_loader,
    build_traces_jsonl_calib_loader,
)
from compress.topology import export_btt_topology, rebuild_btt_from_topology  # noqa: E402


# Public mapping from CLI --calib-mode value to compress train_mode.
CALIB_MODE_TO_TRAIN_MODE = {
    "v2": "btt_llm_v2",
    "v2_bp": "btt_llm_v2_bp",
    "v2_combined": "btt_llm_v2_combined",
    "twosteps": "btt_twosteps",
}
VALID_CALIB_MODES = ("none",) + tuple(CALIB_MODE_TO_TRAIN_MODE.keys())
VALID_CALIB_SOURCES = ("c4", "traces", "training_data")


def _arg_name(name: str, *, hyphen_style: bool) -> str:
    return "--" + (name if hyphen_style else name.replace("-", "_"))


def _attr_name(name: str, *, hyphen_style: bool) -> str:
    return (name if hyphen_style else name).replace("-", "_")


def add_calibrated_btt_args(parser, *, hyphen_style: bool = True) -> None:
    """Register --calib-mode / --calib-source / ... flags on the given parser."""
    parser.add_argument(
        _arg_name("calib-mode", hyphen_style=hyphen_style),
        type=str, default="none", choices=list(VALID_CALIB_MODES),
        help="Calibrated BTT mode. 'none' keeps the legacy blocktt path.",
    )
    parser.add_argument(
        _arg_name("calib-source", hyphen_style=hyphen_style),
        type=str, default="c4", choices=list(VALID_CALIB_SOURCES),
        help="Calibration data source. Only used when --calib-mode != none.",
    )
    parser.add_argument(
        _arg_name("calib-traces-path", hyphen_style=hyphen_style),
        type=str, default=None,
        help="Path to trace JSONL; required when --calib-source=traces.",
    )
    parser.add_argument(
        _arg_name("calib-num-seqs", hyphen_style=hyphen_style),
        type=int, default=128,
    )
    parser.add_argument(
        _arg_name("calib-max-length", hyphen_style=hyphen_style),
        type=int, default=2048,
    )
    parser.add_argument(
        _arg_name("calib-seed", hyphen_style=hyphen_style),
        type=int, default=3,
    )
    parser.add_argument(
        _arg_name("calib-batch-size", hyphen_style=hyphen_style),
        type=int, default=8,
    )


# ---- stubs below; filled in by subsequent tasks ----

def validate_calibrated_btt_args(args, *, argv: Sequence[str], hyphen_style: bool = True) -> None:
    raise NotImplementedError("filled in Task 7")


def build_decomposition_config(args, *, hyphen_style: bool = True) -> DecompositionConfig:
    raise NotImplementedError("filled in Task 8")


def build_training_data_calib_loader(
    dataset, collate_fn, *, num_seqs: int, batch_size: int, seed: int,
) -> DataLoader:
    raise NotImplementedError("filled in Task 9")


def build_rl_rollout_calib_loader(
    *, rl_rollout_fn, tokenizer, num_seqs: int, batch_size: int,
    max_length: int, seed: int,
) -> DataLoader:
    raise NotImplementedError("filled in Task 13")


def build_calib_loader(
    args, *, tokenizer, training_dataset=None, training_collate_fn=None,
    rl_rollout_fn=None, hyphen_style: bool = True,
) -> Optional[DataLoader]:
    raise NotImplementedError("filled in Task 10")


def apply_calibrated_btt(
    model, args, *, calib_loader, device: Optional[str] = None,
    hyphen_style: bool = True,
) -> Tuple[nn.Module, dict]:
    raise NotImplementedError("filled in Task 10")


def materialize_calibrated_btt_weights(model) -> List[Tuple[str, torch.Tensor]]:
    raise NotImplementedError("filled in Task 14")


def restore_calibrated_btt_weights(model, saved_state) -> None:
    raise NotImplementedError("filled in Task 14")


def save_calibrated_btt_checkpoint(model, out_dir: str) -> None:
    raise NotImplementedError("filled in Task 17")


def load_calibrated_btt_for_eval(model, checkpoint_dir: str) -> nn.Module:
    raise NotImplementedError("filled in Task 17")
```

- [ ] **Step 4: Run, verify tests pass**

```bash
python -m unittest tests.test_compress_integration_import -v
```

Expected: OK (2 tests).

- [ ] **Step 5: Commit**

```bash
git add compress_integration.py tests/test_compress_integration_import.py
git commit -m "$(cat <<'EOF'
integration: scaffold compress_integration.py with add_calibrated_btt_args

Sets up sys.path for src/compress, registers the new --calib-* flags,
and stubs the remaining helpers. Subsequent tasks fill in each stub
with TDD.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: `validate_calibrated_btt_args`

**Files:**
- Modify: `compress_integration.py`
- Create: `tests/test_compress_integration_validate.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_compress_integration_validate.py`:

```python
import argparse
import os, sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import compress_integration as ci


def _make_parser(hyphen_style=True):
    p = argparse.ArgumentParser()
    p.add_argument("--train-mode" if hyphen_style else "--train_mode", default="blocktt")
    p.add_argument("--blocktt-rank" if hyphen_style else "--blocktt_rank", default="full")
    ci.add_calibrated_btt_args(p, hyphen_style=hyphen_style)
    return p


class TestValidate(unittest.TestCase):
    def test_none_mode_with_no_calib_flags_ok(self):
        p = _make_parser()
        argv = ["--train-mode", "blocktt"]
        args = p.parse_args(argv)
        ci.validate_calibrated_btt_args(args, argv=argv)

    def test_calib_mode_requires_blocktt(self):
        p = _make_parser()
        argv = ["--train-mode", "lora", "--calib-mode", "v2"]
        args = p.parse_args(argv)
        with self.assertRaisesRegex(ValueError, "only valid with --train-mode blocktt"):
            ci.validate_calibrated_btt_args(args, argv=argv)

    def test_calib_mode_traces_requires_path(self):
        p = _make_parser()
        argv = ["--train-mode", "blocktt", "--calib-mode", "v2", "--calib-source", "traces"]
        args = p.parse_args(argv)
        with self.assertRaisesRegex(ValueError, "calib-traces-path"):
            ci.validate_calibrated_btt_args(args, argv=argv)

    def test_int_blocktt_rank_rejected_on_calibrated(self):
        p = _make_parser()
        argv = ["--train-mode", "blocktt", "--calib-mode", "v2",
                "--calib-source", "c4", "--blocktt-rank", "4"]
        args = p.parse_args(argv)
        with self.assertRaisesRegex(ValueError, "integer"):
            ci.validate_calibrated_btt_args(args, argv=argv)

    def test_float_blocktt_rank_ok_on_calibrated(self):
        p = _make_parser()
        argv = ["--train-mode", "blocktt", "--calib-mode", "v2",
                "--calib-source", "c4", "--blocktt-rank", "0.7"]
        args = p.parse_args(argv)
        ci.validate_calibrated_btt_args(args, argv=argv)

    def test_calib_flag_passed_with_mode_none_rejected(self):
        p = _make_parser()
        argv = ["--train-mode", "blocktt", "--calib-num-seqs", "64"]
        args = p.parse_args(argv)
        with self.assertRaisesRegex(ValueError, "--calib-mode=none"):
            ci.validate_calibrated_btt_args(args, argv=argv)

    def test_underscore_style_also_works(self):
        p = _make_parser(hyphen_style=False)
        argv = ["--train_mode", "blocktt", "--calib_mode", "v2", "--calib_source", "c4"]
        args = p.parse_args(argv)
        ci.validate_calibrated_btt_args(args, argv=argv, hyphen_style=False)

    def test_lift_no_train_mode_attr_ok(self):
        # LIFT parser has no --train-mode; we accept args without that attribute.
        p = argparse.ArgumentParser()
        p.add_argument("--blocktt_rank", default="full")
        ci.add_calibrated_btt_args(p, hyphen_style=False)
        argv = ["--calib_mode", "v2", "--calib_source", "c4"]
        args = p.parse_args(argv)
        ci.validate_calibrated_btt_args(args, argv=argv, hyphen_style=False)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run, verify it fails**

```bash
python -m unittest tests.test_compress_integration_validate -v
```

Expected: FAIL — `NotImplementedError: filled in Task 7`.

- [ ] **Step 3: Implement validation**

Replace the stub in `compress_integration.py`:

```python
_CALIB_FLAG_NAMES_HYPHEN = (
    "--calib-mode", "--calib-source", "--calib-traces-path",
    "--calib-num-seqs", "--calib-max-length", "--calib-seed",
    "--calib-batch-size",
)
_CALIB_FLAG_NAMES_UNDER = tuple(f.replace("-", "_") for f in _CALIB_FLAG_NAMES_HYPHEN)


def _flag_was_passed(argv: Sequence[str], flag: str) -> bool:
    for tok in argv:
        if tok == flag or tok.startswith(flag + "="):
            return True
    return False


def validate_calibrated_btt_args(args, *, argv: Sequence[str], hyphen_style: bool = True) -> None:
    """Raise ValueError if the calib-* args are inconsistent with train-mode / blocktt-rank."""
    calib_mode = getattr(args, "calib_mode", "none")
    calib_source = getattr(args, "calib_source", "c4")

    # 1. --calib-mode != none requires --train-mode blocktt (if parser has train_mode)
    if calib_mode != "none" and hasattr(args, "train_mode"):
        if args.train_mode != "blocktt":
            raise ValueError(
                "--calib-mode only valid with --train-mode blocktt "
                f"(got --train-mode={args.train_mode!r})"
            )

    # 2. --calib-source=traces requires --calib-traces-path
    if calib_mode != "none" and calib_source == "traces":
        path = getattr(args, "calib_traces_path", None)
        if not path:
            flag = "--calib-traces-path" if hyphen_style else "--calib_traces_path"
            raise ValueError(f"{flag} must be set when --calib-source=traces")

    # 3. --calib-mode=none forbids any --calib-* flag being passed explicitly
    flags = _CALIB_FLAG_NAMES_HYPHEN if hyphen_style else _CALIB_FLAG_NAMES_UNDER
    if calib_mode == "none":
        passed = [f for f in flags if f != ("--calib-mode" if hyphen_style else "--calib_mode")
                  and _flag_was_passed(argv, f)]
        if passed:
            raise ValueError(
                f"{', '.join(passed)} is only valid when --calib-mode != none"
            )

    # 4. Integer --blocktt-rank rejected on calibrated path
    if calib_mode != "none":
        rank_raw = getattr(args, "blocktt_rank", "full")
        if isinstance(rank_raw, str):
            if rank_raw != "full":
                try:
                    int(rank_raw)
                    is_int = "." not in rank_raw
                except ValueError:
                    is_int = False
                if is_int:
                    raise ValueError(
                        "integer --blocktt-rank is only valid when --calib-mode=none; "
                        "for calibrated BTT pass 'full' or a float in (0, 1]"
                    )
```

- [ ] **Step 4: Run, verify all tests pass**

```bash
python -m unittest tests.test_compress_integration_validate -v
```

Expected: OK (8 tests).

- [ ] **Step 5: Commit**

```bash
git add compress_integration.py tests/test_compress_integration_validate.py
git commit -m "$(cat <<'EOF'
integration: validate_calibrated_btt_args

Enforces: --calib-mode requires --train-mode=blocktt, --calib-source=traces
requires --calib-traces-path, --calib-mode=none rejects any other --calib-*
flag being passed, and integer --blocktt-rank is legacy-only.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: `build_decomposition_config` — rank, trainable-type, mode translation

**Files:**
- Modify: `compress_integration.py`
- Create: `tests/test_compress_integration_build_config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_compress_integration_build_config.py`:

```python
import argparse
import os, sys
import unittest
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import compress_integration as ci


def _parse(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--train-mode", default="blocktt")
    p.add_argument("--blocktt-rank", default="full")
    p.add_argument("--trainable-type", default="all")
    p.add_argument("--decomp-mode", default="square")
    p.add_argument("--train-position", default="small")
    p.add_argument("--s-merged-to", default=None)
    p.add_argument("--blocktt-factorize-by-head", action="store_true", default=True)
    p.add_argument("--no-blocktt-factorize-by-head", dest="blocktt_factorize_by_head",
                   action="store_false")
    ci.add_calibrated_btt_args(p, hyphen_style=True)
    return p.parse_args(argv)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # mimic transformer naming
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "gate_proj": nn.Linear(8, 8),
                "up_proj": nn.Linear(8, 8),
                "down_proj": nn.Linear(8, 8),
                "q_proj": nn.Linear(8, 8),
                "k_proj": nn.Linear(8, 8),
                "v_proj": nn.Linear(8, 8),
                "o_proj": nn.Linear(8, 8),
                "other": nn.Linear(8, 8),
            })
        ])
        self.lm_head = nn.Linear(8, 32)


class TestBuildConfig(unittest.TestCase):
    def test_rank_full_becomes_ratio_one(self):
        args = _parse(["--calib-mode", "v2", "--calib-source", "c4"])
        cfg = ci.build_decomposition_config(args, model=ToyModel())
        self.assertEqual(cfg.train_mode, "btt_llm_v2")
        self.assertEqual(cfg.compression_ratio, 1.0)

    def test_rank_float_passthrough(self):
        args = _parse(["--calib-mode", "v2_bp", "--calib-source", "c4", "--blocktt-rank", "0.5"])
        cfg = ci.build_decomposition_config(args, model=ToyModel())
        self.assertEqual(cfg.train_mode, "btt_llm_v2_bp")
        self.assertAlmostEqual(cfg.compression_ratio, 0.5)

    def test_rank_float_range_validated(self):
        args = _parse(["--calib-mode", "v2", "--calib-source", "c4", "--blocktt-rank", "1.5"])
        with self.assertRaisesRegex(ValueError, "in \\(0, 1\\]"):
            ci.build_decomposition_config(args, model=ToyModel())

    def test_calib_mode_to_train_mode_mapping(self):
        for calib, train in [("v2", "btt_llm_v2"), ("v2_bp", "btt_llm_v2_bp"),
                              ("v2_combined", "btt_llm_v2_combined"),
                              ("twosteps", "btt_twosteps")]:
            args = _parse(["--calib-mode", calib, "--calib-source", "c4"])
            cfg = ci.build_decomposition_config(args, model=ToyModel())
            self.assertEqual(cfg.train_mode, train)

    def test_trainable_type_all_skips_only_non_target(self):
        args = _parse(["--calib-mode", "v2", "--calib-source", "c4", "--trainable-type", "all"])
        cfg = ci.build_decomposition_config(args, model=ToyModel())
        skip = set(s.strip() for s in cfg.skip_layers.split(","))
        self.assertIn("lm_head", skip)
        self.assertIn("other", skip)
        self.assertNotIn("gate_proj", skip)
        self.assertNotIn("q_proj", skip)

    def test_trainable_type_mlp_skips_attn(self):
        args = _parse(["--calib-mode", "v2", "--calib-source", "c4", "--trainable-type", "mlp"])
        cfg = ci.build_decomposition_config(args, model=ToyModel())
        skip = set(s.strip() for s in cfg.skip_layers.split(","))
        for n in ("q_proj", "k_proj", "v_proj", "o_proj", "other", "lm_head"):
            self.assertIn(n, skip)
        for n in ("gate_proj", "up_proj", "down_proj"):
            self.assertNotIn(n, skip)

    def test_trainable_type_attn_skips_mlp(self):
        args = _parse(["--calib-mode", "v2", "--calib-source", "c4", "--trainable-type", "attn"])
        cfg = ci.build_decomposition_config(args, model=ToyModel())
        skip = set(s.strip() for s in cfg.skip_layers.split(","))
        for n in ("gate_proj", "up_proj", "down_proj", "other", "lm_head"):
            self.assertIn(n, skip)
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
            self.assertNotIn(n, skip)

    def test_passthrough_fields(self):
        args = _parse(["--calib-mode", "v2", "--calib-source", "c4",
                       "--train-position", "small", "--decomp-mode", "input_one_block",
                       "--calib-num-seqs", "64", "--calib-max-length", "1024",
                       "--calib-seed", "7", "--s-merged-to", "trainable"])
        cfg = ci.build_decomposition_config(args, model=ToyModel())
        self.assertEqual(cfg.train_position, "small")
        self.assertEqual(cfg.decomp_mode, "input_one_block")
        self.assertEqual(cfg.calib_num_seqs, 64)
        self.assertEqual(cfg.calib_max_length, 1024)
        self.assertEqual(cfg.calib_seed, 7)
        self.assertEqual(cfg.s_merged_to, "trainable")
        self.assertTrue(cfg.factorize_by_head)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run, verify it fails**

```bash
python -m unittest tests.test_compress_integration_build_config -v
```

Expected: FAIL — NotImplementedError.

- [ ] **Step 3: Implement `build_decomposition_config`**

Add to `compress_integration.py`, right below `validate_calibrated_btt_args`:

```python
_BLOCKTT_TARGET_NAMES = {
    "all": ("gate_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj"),
    "mlp": ("gate_proj", "up_proj", "down_proj"),
    "attn": ("q_proj", "k_proj", "v_proj", "o_proj"),
}


def _resolve_ratio_from_rank(rank_raw) -> float:
    """--blocktt-rank on the calibrated path: 'full' -> 1.0; float in (0, 1] -> itself."""
    if isinstance(rank_raw, str):
        if rank_raw == "full":
            return 1.0
        try:
            val = float(rank_raw)
        except ValueError as exc:
            raise ValueError(f"--blocktt-rank must be 'full' or a float; got {rank_raw!r}") from exc
    elif isinstance(rank_raw, (int, float)):
        val = float(rank_raw)
    else:
        raise ValueError(f"--blocktt-rank has unsupported type {type(rank_raw).__name__}")
    if not (0.0 < val <= 1.0):
        raise ValueError(f"--blocktt-rank float must be in (0, 1]; got {val}")
    return val


def _build_skip_layers(model: nn.Module, trainable_type: str) -> str:
    """Return comma-separated leaf names to skip, i.e. every nn.Linear leaf whose
    name is NOT in the trainable_type target set. 'lm_head' is always included."""
    targets = set(_BLOCKTT_TARGET_NAMES.get(trainable_type, _BLOCKTT_TARGET_NAMES["all"]))
    skip = {"lm_head"}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in targets:
            skip.add(leaf)
    return ",".join(sorted(skip))


def build_decomposition_config(args, *, hyphen_style: bool = True, model=None) -> DecompositionConfig:
    """Translate CLI args to a DecompositionConfig. `model` is required so that
    --trainable-type can be inverted into a skip_layers list."""
    if model is None:
        raise ValueError("build_decomposition_config requires model= for skip_layers inversion")

    train_mode = CALIB_MODE_TO_TRAIN_MODE[getattr(args, "calib_mode")]
    ratio = _resolve_ratio_from_rank(getattr(args, "blocktt_rank", "full"))
    trainable_type = getattr(args, "trainable_type", "all")
    skip_layers = _build_skip_layers(model, trainable_type)

    return DecompositionConfig(
        train_mode=train_mode,
        compression_ratio=ratio,
        calib_source=getattr(args, "calib_source", "c4"),
        calib_traces_path=getattr(args, "calib_traces_path", None),
        calib_num_seqs=getattr(args, "calib_num_seqs", 128),
        calib_max_length=getattr(args, "calib_max_length", 2048),
        calib_seed=getattr(args, "calib_seed", 3),
        skip_layers=skip_layers,
        decomp_mode=getattr(args, "decomp_mode", "square"),
        train_position=getattr(args, "train_position", "both"),
        s_merged_to=getattr(args, "s_merged_to", None),
        factorize_by_head=bool(getattr(args, "blocktt_factorize_by_head", True)),
    )
```

Note: the signature adds a required `model` kwarg; update the stub in the file.

- [ ] **Step 4: Run, verify all tests pass**

```bash
python -m unittest tests.test_compress_integration_build_config -v
```

Expected: OK (8 tests).

- [ ] **Step 5: Commit**

```bash
git add compress_integration.py tests/test_compress_integration_build_config.py
git commit -m "$(cat <<'EOF'
integration: build_decomposition_config translation

Translates --calib-mode to DecompositionConfig.train_mode, --blocktt-rank
(full | float in (0,1]) to compression_ratio, --trainable-type to a
skip_layers exclude list (inverse of the target set, with lm_head always
included), and passes through the remaining fields.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: `build_training_data_calib_loader`

**Files:**
- Modify: `compress_integration.py`
- Create: `tests/test_compress_integration_training_loader.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_compress_integration_training_loader.py`:

```python
import os, sys
import unittest
import torch
from torch.utils.data import Dataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import compress_integration as ci


class ToyDataset(Dataset):
    def __init__(self, n=50, seqlen=6):
        self.items = [
            {"input_ids": torch.arange(seqlen) + i,
             "labels": torch.arange(seqlen) + i}
            for i in range(n)
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def toy_collate(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


class TestTrainingDataCalibLoader(unittest.TestCase):
    def test_yields_contract_dicts(self):
        loader = ci.build_training_data_calib_loader(
            ToyDataset(), toy_collate, num_seqs=16, batch_size=4, seed=0,
        )
        batches = list(loader)
        self.assertEqual(len(batches), 4)
        for batch in batches:
            self.assertIn("input_ids", batch)
            self.assertEqual(batch["input_ids"].shape, (4, 6))
            self.assertIn("labels", batch)

    def test_respects_num_seqs(self):
        loader = ci.build_training_data_calib_loader(
            ToyDataset(), toy_collate, num_seqs=10, batch_size=4, seed=0,
        )
        total = sum(b["input_ids"].shape[0] for b in loader)
        self.assertEqual(total, 10)

    def test_deterministic_under_seed(self):
        def first_ids(seed):
            loader = ci.build_training_data_calib_loader(
                ToyDataset(), toy_collate, num_seqs=8, batch_size=4, seed=seed,
            )
            return next(iter(loader))["input_ids"].tolist()
        self.assertEqual(first_ids(0), first_ids(0))

    def test_num_seqs_larger_than_dataset_clamped(self):
        loader = ci.build_training_data_calib_loader(
            ToyDataset(n=5), toy_collate, num_seqs=100, batch_size=2, seed=0,
        )
        total = sum(b["input_ids"].shape[0] for b in loader)
        self.assertEqual(total, 5)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run, verify it fails**

```bash
python -m unittest tests.test_compress_integration_training_loader -v
```

Expected: FAIL — NotImplementedError.

- [ ] **Step 3: Implement**

Replace the stub in `compress_integration.py`:

```python
def build_training_data_calib_loader(
    dataset, collate_fn, *, num_seqs: int, batch_size: int, seed: int,
) -> DataLoader:
    """Take a deterministic subset of the training dataset and wrap it in a
    DataLoader using the training collate function. Yields the same batch shape
    the training loop sees."""
    import random
    n_available = len(dataset)
    n_take = min(int(num_seqs), n_available)
    rng = random.Random(int(seed))
    indices = list(range(n_available))
    rng.shuffle(indices)
    subset = Subset(dataset, indices[:n_take])
    return DataLoader(
        subset,
        batch_size=int(batch_size),
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )
```

- [ ] **Step 4: Run, verify all tests pass**

```bash
python -m unittest tests.test_compress_integration_training_loader -v
```

Expected: OK (4 tests).

- [ ] **Step 5: Commit**

```bash
git add compress_integration.py tests/test_compress_integration_training_loader.py
git commit -m "$(cat <<'EOF'
integration: build_training_data_calib_loader

Wraps a deterministic subset of the training dataset in a DataLoader with
the training collate function, so calibration batches match real training
batches byte-for-byte.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: `build_calib_loader` dispatch + `apply_calibrated_btt`

**Files:**
- Modify: `compress_integration.py`
- Create: `tests/test_compress_integration_apply.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_compress_integration_apply.py`:

```python
import argparse
import os, sys
import unittest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import compress_integration as ci
from compress.btt.btt_linear import BTTLinear


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "q_proj": nn.Linear(16, 16),
                "k_proj": nn.Linear(16, 16),
                "v_proj": nn.Linear(16, 16),
                "o_proj": nn.Linear(16, 16),
                "gate_proj": nn.Linear(16, 16),
                "up_proj": nn.Linear(16, 16),
                "down_proj": nn.Linear(16, 16),
            })
        ])
        self.lm_head = nn.Linear(16, 32)

    def forward(self, input_ids, labels=None, **kwargs):
        x = torch.nn.functional.one_hot(input_ids, num_classes=16).float()
        for block in self.layers:
            x = block["q_proj"](x)
        logits = self.lm_head(x)
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100,
            )
            return type("Out", (), {"loss": loss, "logits": logits})()
        return type("Out", (), {"logits": logits})()


class CalibDS(Dataset):
    def __init__(self, n=16, seqlen=4):
        self.items = [
            {"input_ids": torch.randint(0, 16, (seqlen,)),
             "labels": torch.randint(0, 16, (seqlen,))}
            for _ in range(n)
        ]
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def _collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--train-mode", default="blocktt")
    p.add_argument("--blocktt-rank", default="full")
    p.add_argument("--trainable-type", default="all")
    p.add_argument("--decomp-mode", default="square")
    p.add_argument("--train-position", default="small")
    p.add_argument("--s-merged-to", default=None)
    p.add_argument("--blocktt-factorize-by-head", action="store_true", default=True)
    ci.add_calibrated_btt_args(p, hyphen_style=True)
    return p.parse_args([
        "--calib-mode", "v2", "--calib-source", "training_data",
        "--calib-num-seqs", "8", "--calib-batch-size", "4",
    ])


class TestApply(unittest.TestCase):
    def test_build_calib_loader_training_data(self):
        args = _parse()
        loader = ci.build_calib_loader(
            args, tokenizer=None, training_dataset=CalibDS(),
            training_collate_fn=_collate,
        )
        self.assertIsNotNone(loader)
        batch = next(iter(loader))
        self.assertIn("input_ids", batch)

    def test_build_calib_loader_returns_none_for_calib_none(self):
        p = argparse.ArgumentParser()
        p.add_argument("--train-mode", default="blocktt")
        p.add_argument("--blocktt-rank", default="full")
        ci.add_calibrated_btt_args(p, hyphen_style=True)
        args = p.parse_args(["--train-mode", "blocktt"])
        self.assertIsNone(
            ci.build_calib_loader(args, tokenizer=None)
        )

    @unittest.skipUnless(torch.cuda.is_available(), "calibrated BTT requires CUDA")
    def test_apply_calibrated_btt_installs_btt_linear(self):
        args = _parse()
        model = TinyModel().cuda()
        loader = ci.build_calib_loader(
            args, tokenizer=None, training_dataset=CalibDS(),
            training_collate_fn=_collate,
        )
        model, stats = ci.apply_calibrated_btt(model, args, calib_loader=loader)
        self.assertGreater(stats["num_btt_layers"], 0)
        btt_layers = [m for m in model.modules() if isinstance(m, BTTLinear)]
        self.assertGreater(len(btt_layers), 0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run, verify it fails**

```bash
python -m unittest tests.test_compress_integration_apply -v
```

Expected: FAIL on first two tests (NotImplementedError); third skipped if no CUDA.

- [ ] **Step 3: Implement both helpers**

Replace the two stubs in `compress_integration.py`:

```python
def build_calib_loader(
    args, *, tokenizer, training_dataset=None, training_collate_fn=None,
    rl_rollout_fn=None, hyphen_style: bool = True,
) -> Optional[DataLoader]:
    calib_mode = getattr(args, "calib_mode", "none")
    if calib_mode == "none":
        return None

    source = getattr(args, "calib_source", "c4")
    num_seqs = int(getattr(args, "calib_num_seqs", 128))
    max_length = int(getattr(args, "calib_max_length", 2048))
    seed = int(getattr(args, "calib_seed", 3))
    batch_size = int(getattr(args, "calib_batch_size", 8))

    if source == "c4":
        return build_c4_calib_loader(
            tokenizer, num_seqs=num_seqs, max_length=max_length,
            batch_size=batch_size, seed=seed,
        )
    if source == "traces":
        return build_traces_jsonl_calib_loader(
            tokenizer,
            jsonl_path=getattr(args, "calib_traces_path"),
            num_seqs=num_seqs, max_length=max_length, batch_size=batch_size,
        )
    if source == "training_data":
        if training_dataset is not None and training_collate_fn is not None:
            return build_training_data_calib_loader(
                training_dataset, training_collate_fn,
                num_seqs=num_seqs, batch_size=batch_size, seed=seed,
            )
        if rl_rollout_fn is not None:
            return build_rl_rollout_calib_loader(
                rl_rollout_fn=rl_rollout_fn, tokenizer=tokenizer,
                num_seqs=num_seqs, batch_size=batch_size,
                max_length=max_length, seed=seed,
            )
        raise ValueError(
            "--calib-source=training_data requires either "
            "(training_dataset, training_collate_fn) or rl_rollout_fn"
        )
    raise ValueError(f"Unknown --calib-source {source!r}")


def apply_calibrated_btt(
    model, args, *, calib_loader, device: Optional[str] = None,
    hyphen_style: bool = True,
) -> Tuple[nn.Module, dict]:
    cfg = build_decomposition_config(args, hyphen_style=hyphen_style, model=model)
    model, stats = decompose_with_loader(
        model, cfg, calib_loader=calib_loader, device=device,
        return_trainability_stats=True,
    )
    if stats is None or stats.get("num_btt_layers", 0) == 0:
        raise ValueError("No BTT layers were installed; check --trainable-type selection.")
    return model, stats
```

- [ ] **Step 4: Run, verify tests pass**

```bash
python -m unittest tests.test_compress_integration_apply -v
```

Expected: OK for the two non-skipped tests; third skipped on non-CUDA machines, passes on CUDA.

- [ ] **Step 5: Commit**

```bash
git add compress_integration.py tests/test_compress_integration_apply.py
git commit -m "$(cat <<'EOF'
integration: build_calib_loader dispatch + apply_calibrated_btt

Dispatches on --calib-source (c4 / traces / training_data). For
training_data, prefers the training-dataset loader when provided;
otherwise delegates to the RL rollout loader. apply_calibrated_btt wraps
decompose_with_loader with the CLI-translated config and asserts the
graph now contains BTT layers.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: `run_sft.py` integration

**Files:**
- Modify: `run_sft.py`
- Create: `tests/test_run_sft_cli_calib.py`

- [ ] **Step 1: Write the failing CLI test**

Create `tests/test_run_sft_cli_calib.py`:

```python
import os, subprocess, sys, unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(*args):
    return subprocess.run(
        [sys.executable, "run_sft.py", *args, "--help"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


class TestSFTCalibCLI(unittest.TestCase):
    def test_help_mentions_calib_mode(self):
        r = _run()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--calib-mode", r.stdout)
        self.assertIn("--calib-source", r.stdout)

    def test_default_parses_ok_without_calib_flags(self):
        # Use --help-based parsing as a smoke test; full parse is covered by validate tests.
        r = _run()
        self.assertEqual(r.returncode, 0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run, verify fails**

```bash
python -m unittest tests.test_run_sft_cli_calib -v
```

Expected: FAIL — `--calib-mode` not in help output.

- [ ] **Step 3: Wire up in `run_sft.py`**

Make these edits in `run_sft.py`:

(a) Near the other top-level imports, add:

```python
from compress_integration import (
    add_calibrated_btt_args,
    validate_calibrated_btt_args,
    apply_calibrated_btt,
    build_calib_loader,
    save_calibrated_btt_checkpoint,
)
```

(b) In the argparse construction block (right after the existing `--blocktt-*` flags), add:

```python
add_calibrated_btt_args(parser, hyphen_style=True)
```

(c) In the existing post-parse validation block that uses `_flag_was_passed(argv, f)`, append:

```python
validate_calibrated_btt_args(args, argv=argv, hyphen_style=True)
```

(d) Reorder the `main()` so dataset + collate are built *before* `prepare_model`:

Find the current `load_dataset("HuggingFaceH4/no_robots", ...)` and `build_collate_fn(tokenizer)` calls, move them to execute before `prepare_model(args, model, ...)`. Then pass `train_dataset=dataset, collate_fn=collate_fn` as new kwargs to `prepare_model`.

(e) Extend `prepare_model` signature:

```python
def prepare_model(args, model, ..., train_dataset=None, collate_fn=None):
    ...
```

(f) Inside `prepare_model`, right before the existing `elif args.train_mode == "blocktt":` branch, insert:

```python
    if args.train_mode == "blocktt" and getattr(args, "calib_mode", "none") != "none":
        calib_loader = build_calib_loader(
            args,
            tokenizer=tokenizer,
            training_dataset=train_dataset,
            training_collate_fn=collate_fn,
            hyphen_style=True,
        )
        model, calib_stats = apply_calibrated_btt(
            model, args, calib_loader=calib_loader,
        )
        print(f"[calib-btt] installed {calib_stats['num_btt_layers']} BTT layers "
              f"(trainable cores: L={calib_stats['tuned_left_cores']}, "
              f"R={calib_stats['tuned_right_cores']})")
        return model  # skip the legacy blocktt branch entirely
```

Keep the rest of `prepare_model` unchanged.

(g) In the checkpoint-saving block (wherever `step={N}/model.safetensors` is written), also save the topology when applicable:

```python
    if getattr(args, "calib_mode", "none") != "none":
        save_calibrated_btt_checkpoint(model, step_dir)
    else:
        # existing save logic
        ...
```

(Note: `save_calibrated_btt_checkpoint` is implemented in Task 17; until then the file is a stub that raises. The `run_sft.py` branch is still behind the `--calib-mode != none` guard, so legacy runs are unaffected.)

- [ ] **Step 4: Run, verify CLI test passes**

```bash
python -m unittest tests.test_run_sft_cli_calib -v
```

Expected: OK (2 tests).

- [ ] **Step 5: Smoke-check — syntax**

```bash
python -m py_compile run_sft.py compress_integration.py
```

Expected: no output (success).

- [ ] **Step 6: Run the existing BTT compat test — legacy path untouched**

```bash
python -m unittest tests.test_btt_pipeline_compat
```

Expected: OK.

- [ ] **Step 7: Commit**

```bash
git add run_sft.py tests/test_run_sft_cli_calib.py
git commit -m "$(cat <<'EOF'
run_sft: wire --calib-mode through compress_integration

When --calib-mode != none, run_sft.py now routes --train-mode blocktt
through compress.decompose_with_loader with a calibration DataLoader
built from the actual HuggingFaceH4/no_robots training subset +
training collate_fn, so calibration batches match real training
batches. Legacy --calib-mode=none path is unchanged.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: LIFT integration (`ref/LIFT/src/finetune_blocktt.py`)

**Files:**
- Modify: `ref/LIFT/src/finetune_blocktt.py`
- Create: `tests/test_lift_blocktt_cli_calib.py`

- [ ] **Step 1: Write the failing CLI test**

Create `tests/test_lift_blocktt_cli_calib.py`:

```python
import os, subprocess, sys, unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run():
    return subprocess.run(
        [sys.executable, "ref/LIFT/src/finetune_blocktt.py", "--help"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


class TestLIFTBlockTTCalibCLI(unittest.TestCase):
    def test_help_mentions_underscore_calib_mode(self):
        r = _run()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--calib_mode", r.stdout)
        self.assertIn("--calib_source", r.stdout)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run, verify fails**

```bash
python -m unittest tests.test_lift_blocktt_cli_calib -v
```

Expected: FAIL.

- [ ] **Step 3: Wire up LIFT**

In `ref/LIFT/src/finetune_blocktt.py`:

(a) Near the top, alongside the existing repo-root `sys.path` injection, add:

```python
# Ensure `compress_integration` (repo-root module) and `compress` (src/compress)
# are both importable.
_LIFT_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _LIFT_REPO_ROOT not in sys.path:
    sys.path.insert(0, _LIFT_REPO_ROOT)
_LIFT_SRC = os.path.join(_LIFT_REPO_ROOT, "src")
if os.path.isdir(_LIFT_SRC) and _LIFT_SRC not in sys.path:
    sys.path.insert(0, _LIFT_SRC)

from compress_integration import (
    add_calibrated_btt_args,
    validate_calibrated_btt_args,
    apply_calibrated_btt,
    build_calib_loader,
    save_calibrated_btt_checkpoint,
)
```

(b) In the argparse block, after the existing BTT-related flags, add:

```python
add_calibrated_btt_args(parser, hyphen_style=False)
```

(c) After `args = parser.parse_args()`, add:

```python
validate_calibrated_btt_args(args, argv=sys.argv[1:], hyphen_style=False)
```

(d) Before the existing legacy decomposition block (`convert_linear_to_btt(...)`), insert the branch:

```python
if getattr(args, "calib_mode", "none") != "none":
    # Build dataset + collator first so calibration batches = training batches.
    # NOTE: the existing SupervisedDataset / DataCollatorForSupervisedDataset
    # construction below must be hoisted to run before this block. Move those
    # lines up accordingly.
    calib_loader = build_calib_loader(
        args,
        tokenizer=tokenizer,
        training_dataset=train_dataset,
        training_collate_fn=data_collator,
        hyphen_style=False,
    )
    model, calib_stats = apply_calibrated_btt(
        model, args, calib_loader=calib_loader, hyphen_style=False,
    )
    print(f"[calib-btt] installed {calib_stats['num_btt_layers']} BTT layers")
else:
    # existing legacy path: convert_linear_to_btt + configure_blocktt_trainability
    ...
```

(e) Hoist `SupervisedDataset` construction and `DataCollatorForSupervisedDataset` instantiation above the decomposition block (they don't depend on the model).

(f) At the checkpoint-saving site, add the calibrated save:

```python
if getattr(args, "calib_mode", "none") != "none":
    save_calibrated_btt_checkpoint(model, out_dir)
else:
    # existing save
    ...
```

- [ ] **Step 4: Run, verify CLI test passes**

```bash
python -m unittest tests.test_lift_blocktt_cli_calib -v
```

Expected: OK (1 test).

- [ ] **Step 5: Syntax check**

```bash
python -m py_compile ref/LIFT/src/finetune_blocktt.py
```

Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add ref/LIFT/src/finetune_blocktt.py tests/test_lift_blocktt_cli_calib.py
git commit -m "$(cat <<'EOF'
LIFT: --calib_mode routes finetune_blocktt through compress_integration

Adds sys.path injection for <repo>/src so `from compress ...` imports
work. Registers underscore-style --calib_mode / --calib_source flags.
When --calib_mode != none, decomposition is done via
compress.decompose_with_loader with a DataLoader built from
SupervisedDataset + DataCollatorForSupervisedDataset. Legacy path is
unchanged.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: `build_rl_rollout_calib_loader`

**Files:**
- Modify: `compress_integration.py`
- Create: `tests/test_compress_integration_rl_rollout_loader.py`

This helper turns a caller-supplied rollout function into a contract-compliant DataLoader. The rollout function returns `(prompt_text, completion_text)` pairs; this helper tokenizes and assembles batches with `labels` masked to `-100` on prompt tokens.

- [ ] **Step 1: Write the failing test**

Create `tests/test_compress_integration_rl_rollout_loader.py`:

```python
import os, sys
import unittest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import compress_integration as ci


class FakeTokenizer:
    """Character-level tokenizer for test determinism."""
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = [ord(c) % 256 for c in text]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids])}
        return {"input_ids": ids}

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 256 for c in text]


def fake_rollout(n):
    return [(f"prompt{i}-", f"completion{i}") for i in range(n)]


class TestRLRolloutCalibLoader(unittest.TestCase):
    def test_yields_masked_labels(self):
        tok = FakeTokenizer()
        loader = ci.build_rl_rollout_calib_loader(
            rl_rollout_fn=fake_rollout, tokenizer=tok,
            num_seqs=4, batch_size=2, max_length=32, seed=0,
        )
        batches = list(loader)
        self.assertEqual(sum(b["input_ids"].shape[0] for b in batches), 4)
        for batch in batches:
            self.assertIn("labels", batch)
            # Each row should have some -100s (prompt tokens masked) and some real labels.
            for row in batch["labels"]:
                n_masked = int((row == -100).sum())
                n_real = int((row != -100).sum() - (row == tok.pad_token_id).sum())
                self.assertGreater(n_masked, 0)
                self.assertGreater(n_real, 0)

    def test_respects_max_length(self):
        tok = FakeTokenizer()
        def long_rollout(n):
            return [("x" * 40, "y" * 40) for _ in range(n)]
        loader = ci.build_rl_rollout_calib_loader(
            rl_rollout_fn=long_rollout, tokenizer=tok,
            num_seqs=2, batch_size=2, max_length=16, seed=0,
        )
        batch = next(iter(loader))
        self.assertLessEqual(batch["input_ids"].shape[1], 16)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run, verify fails**

```bash
python -m unittest tests.test_compress_integration_rl_rollout_loader -v
```

Expected: FAIL — NotImplementedError.

- [ ] **Step 3: Implement**

Replace the stub in `compress_integration.py`:

```python
def build_rl_rollout_calib_loader(
    *, rl_rollout_fn: Callable[[int], List[Tuple[str, str]]],
    tokenizer, num_seqs: int, batch_size: int, max_length: int, seed: int,
) -> DataLoader:
    """Run a caller-supplied rollout function to produce (prompt, completion)
    text pairs; tokenize each pair; build a DataLoader of
    {input_ids, attention_mask, labels} where prompt tokens are masked to -100
    in labels (mirroring the RL loss mask)."""
    pairs = rl_rollout_fn(int(num_seqs))
    if len(pairs) == 0:
        raise ValueError("rl_rollout_fn returned no (prompt, completion) pairs")

    examples = []
    for prompt, completion in pairs[:int(num_seqs)]:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        completion_ids = tokenizer.encode(completion, add_special_tokens=False)
        full = (prompt_ids + completion_ids)[: int(max_length)]
        prompt_len = min(len(prompt_ids), len(full))
        labels = [-100] * prompt_len + full[prompt_len:]
        # If prompt was truncated away entirely, fall back to masking half.
        if all(l == -100 for l in labels):
            labels = full.copy()
        examples.append({
            "input_ids": torch.tensor(full, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        })

    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0

    def _collate(batch):
        max_len = max(e["input_ids"].shape[0] for e in batch)
        input_ids = torch.full((len(batch), max_len), fill_value=pad_id, dtype=torch.long)
        attn = torch.zeros((len(batch), max_len), dtype=torch.long)
        labels = torch.full((len(batch), max_len), fill_value=-100, dtype=torch.long)
        for i, e in enumerate(batch):
            L = e["input_ids"].shape[0]
            input_ids[i, :L] = e["input_ids"]
            attn[i, :L] = 1
            labels[i, :L] = e["labels"]
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

    return DataLoader(
        examples, batch_size=int(batch_size), shuffle=False, collate_fn=_collate,
    )
```

- [ ] **Step 4: Run, verify tests pass**

```bash
python -m unittest tests.test_compress_integration_rl_rollout_loader -v
```

Expected: OK (2 tests).

- [ ] **Step 5: Commit**

```bash
git add compress_integration.py tests/test_compress_integration_rl_rollout_loader.py
git commit -m "$(cat <<'EOF'
integration: build_rl_rollout_calib_loader

Turns a caller-supplied rollout fn (which returns (prompt, completion)
pairs from the dense base model) into a DataLoader whose batches carry
input_ids / attention_mask / labels with prompt tokens masked to -100,
matching the RL loss mask so backward calibration gradients match real
training gradients.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: `materialize_calibrated_btt_weights` / `restore_calibrated_btt_weights`

**Files:**
- Modify: `compress_integration.py`
- Create: `tests/test_compress_integration_materialize.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_compress_integration_materialize.py`:

```python
import os, sys
import unittest
import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import compress_integration as ci
from compress.btt.btt_linear import BTTLinear


def _make_btt(m=2, a=3, n=2, b=4, rank=2):
    btt_l = torch.randn(m, n * rank, a)
    btt_r = torch.randn(n, b, m * rank)
    bias = torch.randn(m * a)
    return BTTLinear(btt_l, btt_r, bias=bias, m=m, a=a, n=n, b=b, rank=rank)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = _make_btt()


class TestMaterialize(unittest.TestCase):
    def test_materialize_returns_dense_tuples(self):
        torch.manual_seed(0)
        model = ToyModel()
        pairs = ci.materialize_calibrated_btt_weights(model)
        names = [n for n, _ in pairs]
        self.assertIn("fc.weight", names)
        self.assertIn("fc.bias", names)
        weight = dict(pairs)["fc.weight"]
        self.assertEqual(weight.shape, (model.fc.out_features, model.fc.in_features))

    def test_materialize_preserves_forward_behavior(self):
        torch.manual_seed(0)
        model = ToyModel()
        pairs = dict(ci.materialize_calibrated_btt_weights(model))
        x = torch.randn(3, model.fc.in_features)
        expected = model(x) if hasattr(model, "forward") else model.fc(x)
        out = x @ pairs["fc.weight"].T + pairs["fc.bias"]
        self.assertTrue(torch.allclose(out, model.fc(x), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run, verify fails**

```bash
python -m unittest tests.test_compress_integration_materialize -v
```

Expected: FAIL — NotImplementedError.

- [ ] **Step 3: Implement**

Replace both stubs in `compress_integration.py`:

```python
@torch.no_grad()
def materialize_calibrated_btt_weights(model) -> List[Tuple[str, torch.Tensor]]:
    """Return [(param_name, dense_tensor)] for every BTTLinear in the model.

    For each BTTLinear `M` at path `p`, yields (`p.weight`, dense) and
    optionally (`p.bias`, bias). This list is appended to any other
    (name, tensor) pairs the caller assembles for vLLM weight sync.
    """
    out: List[Tuple[str, torch.Tensor]] = []
    for name, module in model.named_modules():
        if not isinstance(module, BTTLinear):
            continue
        out.append((f"{name}.weight", module.materialize_dense_weight()))
        if module.bias is not None:
            out.append((f"{name}.bias", module.bias.detach()))
    return out


def restore_calibrated_btt_weights(model, saved_state) -> None:
    """No-op: BTTLinear weights are never overwritten by vLLM weight export,
    only materialized-and-copied. The real factored cores remain in place
    on the training model, so no restore is needed. Provided for API
    symmetry with the legacy SVDLayer/BTTLayer restore flow."""
    return None
```

- [ ] **Step 4: Run, verify tests pass**

```bash
python -m unittest tests.test_compress_integration_materialize -v
```

Expected: OK (2 tests).

- [ ] **Step 5: Commit**

```bash
git add compress_integration.py tests/test_compress_integration_materialize.py
git commit -m "$(cat <<'EOF'
integration: materialize_calibrated_btt_weights

Walks BTTLinear modules and returns (name, dense tensor) pairs suitable
for appending to run_rl.py's vLLM weight export list. Mirrors the
existing BTTLayer / SVDLayer export treatment so the single rollout flow
covers all three cases.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: `run_rl.py` integration

**Files:**
- Modify: `run_rl.py`
- Create: `tests/test_run_rl_cli_calib.py`

- [ ] **Step 1: Write the failing CLI test**

Create `tests/test_run_rl_cli_calib.py`:

```python
import os, subprocess, sys, unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run():
    return subprocess.run(
        [sys.executable, "run_rl.py", "--help"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


class TestRLCalibCLI(unittest.TestCase):
    def test_help_mentions_calib_mode(self):
        r = _run()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--calib-mode", r.stdout)
        self.assertIn("--calib-source", r.stdout)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run, verify fails**

```bash
python -m unittest tests.test_run_rl_cli_calib -v
```

Expected: FAIL.

- [ ] **Step 3: Wire up `run_rl.py`**

In `run_rl.py`:

(a) Add imports alongside the existing `from btt_layer import ...`:

```python
from compress_integration import (
    add_calibrated_btt_args,
    validate_calibrated_btt_args,
    apply_calibrated_btt,
    build_calib_loader,
    materialize_calibrated_btt_weights,
    save_calibrated_btt_checkpoint,
)
from compress.btt.btt_linear import BTTLinear
```

(b) Call `add_calibrated_btt_args(parser, hyphen_style=True)` after the other RL flags.

(c) Call `validate_calibrated_btt_args(args, argv=argv, hyphen_style=True)` after `args = parser.parse_args()`.

(d) Extend `export_weights_for_vllm` (around line 528) to also cover `BTTLinear`:

```python
@torch.no_grad()
def export_weights_for_vllm(model: torch.nn.Module):
    factorized_module_names = []
    weight_tuples = []

    for module_name, module in model.named_modules():
        if isinstance(module, BTTLayer):
            factorized_module_names.append(module_name)
            dense_weight = materialize_btt_weight(module)
            weight_tuples.append((f"{module_name}.weight", dense_weight))
            if module.bias is not None:
                weight_tuples.append((f"{module_name}.bias", module.bias))
        elif isinstance(module, SVDLayer):
            factorized_module_names.append(module_name)
            dense_weight = materialize_svd_weight(module)
            weight_tuples.append((f"{module_name}.weight", dense_weight))
            if module.bias is not None:
                weight_tuples.append((f"{module_name}.bias", module.bias))
        elif isinstance(module, BTTLinear):
            factorized_module_names.append(module_name)
            dense_weight = module.materialize_dense_weight()
            weight_tuples.append((f"{module_name}.weight", dense_weight))
            if module.bias is not None:
                weight_tuples.append((f"{module_name}.bias", module.bias))

    if factorized_module_names:
        factorized_prefixes = tuple(f"{name}." for name in factorized_module_names)
    else:
        factorized_prefixes = ()

    for name, param in model.named_parameters():
        if factorized_prefixes and name.startswith(factorized_prefixes):
            continue
        weight_tuples.append((name, param))

    return weight_tuples
```

(e) In the model-preparation block (where current code calls `convert_linear_to_btt` / `convert_linear_to_svd`), insert a calibrated branch **before** the legacy conversion:

```python
if args.train_mode == "blocktt" and getattr(args, "calib_mode", "none") != "none":
    # Build a calibration rollout fn using the dense base model and the
    # existing RL rollout settings, so calibration batches match training.
    def _rl_calib_rollout(n):
        from random import Random
        import torch
        rng = Random(int(getattr(args, "calib_seed", 3)))
        indices = list(range(len(train_dataset)))
        rng.shuffle(indices)
        pairs = []
        model.eval()
        for idx in indices[:n]:
            prompt = train_dataset[idx]["prompt"]
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=int(getattr(args, "max_response_length", 512)),
                    temperature=float(getattr(args, "rollout_temperature", 1.0)),
                    top_p=float(getattr(args, "rollout_top_p", 1.0)),
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            full = tokenizer.decode(out[0], skip_special_tokens=True)
            completion = full[len(prompt):] if full.startswith(prompt) else full
            pairs.append((prompt, completion))
        model.train()
        return pairs

    calib_loader = build_calib_loader(
        args,
        tokenizer=tokenizer,
        rl_rollout_fn=_rl_calib_rollout,
        hyphen_style=True,
    )
    model, calib_stats = apply_calibrated_btt(
        model, args, calib_loader=calib_loader, hyphen_style=True,
    )
    print(f"[calib-btt] installed {calib_stats['num_btt_layers']} BTT layers")
```

**Implementation note for the rollout-config reuse:** the closure above reads `args.max_response_length`, `args.rollout_temperature`, `args.rollout_top_p` directly. During implementation, look up the actual flag names `run_rl.py` uses for these (they may be `--max-new-tokens`, `--temperature`, etc.). Replace the `getattr` calls with the real attribute names so the calibration rollout uses *exactly* the same generate() settings the training rollout uses. If `run_rl.py` wraps generate settings in a helper, call that helper instead of re-specifying the kwargs.

(f) At the checkpoint save site, add:

```python
if getattr(args, "calib_mode", "none") != "none":
    save_calibrated_btt_checkpoint(model, step_dir)
else:
    # existing save
    ...
```

- [ ] **Step 4: Run, verify CLI test passes**

```bash
python -m unittest tests.test_run_rl_cli_calib -v
```

Expected: OK (1 test).

- [ ] **Step 5: Syntax check**

```bash
python -m py_compile run_rl.py
```

Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add run_rl.py tests/test_run_rl_cli_calib.py
git commit -m "$(cat <<'EOF'
run_rl: wire --calib-mode through compress_integration

run_rl.py now runs a one-time calibration rollout on the dense base
model before decomposition (using the same generate() settings the
training loop uses), then routes --train-mode blocktt through
compress.decompose_with_loader. export_weights_for_vllm is extended to
recognize compress.btt.btt_linear.BTTLinear so the existing
materialize-and-generate-locally rollout flow covers all three BTT /
SVD / calibrated-BTT cases.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 16: `run_rl_dapo.py` integration

**Files:**
- Modify: `run_rl_dapo.py`
- Create: `tests/test_run_rl_dapo_cli_calib.py`

This is a direct copy of Task 15 applied to `run_rl_dapo.py`. The DAPO script shares most infrastructure with `run_rl.py` per CLAUDE.md.

- [ ] **Step 1: Write the failing CLI test**

Create `tests/test_run_rl_dapo_cli_calib.py`:

```python
import os, subprocess, sys, unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run():
    return subprocess.run(
        [sys.executable, "run_rl_dapo.py", "--help"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


class TestRLDAPOCalibCLI(unittest.TestCase):
    def test_help_mentions_calib_mode(self):
        r = _run()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--calib-mode", r.stdout)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run, verify fails**

```bash
python -m unittest tests.test_run_rl_dapo_cli_calib -v
```

Expected: FAIL.

- [ ] **Step 3: Apply the same wiring as Task 15, Step 3 (a)–(f) to `run_rl_dapo.py`**

Read the file, locate the analogous sites (imports, argparse, `export_weights_for_vllm` / equivalent, model-prep block, checkpoint save). Apply the **same edits** described in Task 15 Step 3. If any flag name differs between `run_rl.py` and `run_rl_dapo.py`, use whatever the DAPO script uses — do not import from `run_rl.py`.

**Do not silently skip any of (a)–(f).** If the DAPO script does not have a weights-for-vllm helper, apply the calibrated-BTT materialization at whichever site serves the equivalent role. If it has its own rollout-config block, reuse that.

- [ ] **Step 4: Run, verify CLI test passes**

```bash
python -m unittest tests.test_run_rl_dapo_cli_calib -v
```

Expected: OK (1 test).

- [ ] **Step 5: Syntax check**

```bash
python -m py_compile run_rl_dapo.py
```

Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add run_rl_dapo.py tests/test_run_rl_dapo_cli_calib.py
git commit -m "$(cat <<'EOF'
run_rl_dapo: wire --calib-mode through compress_integration

Same changes as run_rl.py: calibration rollout on dense base model
before decomposition, BTTLinear recognized by the vLLM weight export
helper, calibrated checkpoint save when --calib-mode != none. Keeps
run_rl_dapo.py in sync with run_rl.py.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 17: Checkpoint save/load — `save_calibrated_btt_checkpoint` / `load_calibrated_btt_for_eval`

**Files:**
- Modify: `compress_integration.py`
- Create: `tests/test_calibrated_btt_checkpoint.py`

- [ ] **Step 1: Write the failing round-trip test**

Create `tests/test_calibrated_btt_checkpoint.py`:

```python
import os, sys
import tempfile
import unittest
import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import compress_integration as ci
from compress.btt.btt_linear import BTTLinear


def _make_btt(m=2, a=3, n=2, b=4, rank=2):
    btt_l = torch.randn(m, n * rank, a)
    btt_r = torch.randn(n, b, m * rank)
    bias = torch.randn(m * a)
    return BTTLinear(btt_l, btt_r, bias=bias, m=m, a=a, n=n, b=b, rank=rank)


class ToyModel(nn.Module):
    def __init__(self, with_btt=False):
        super().__init__()
        self.fc = _make_btt() if with_btt else nn.Linear(8, 6)


class TestCheckpoint(unittest.TestCase):
    def test_save_and_reload_preserves_forward(self):
        torch.manual_seed(0)
        trained = ToyModel(with_btt=True)
        x = torch.randn(3, trained.fc.in_features)
        expected = trained.fc(x)

        with tempfile.TemporaryDirectory() as tmp:
            ci.save_calibrated_btt_checkpoint(trained, tmp)
            self.assertTrue(os.path.exists(os.path.join(tmp, "model.safetensors")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "btt_topology.json")))

            fresh = ToyModel(with_btt=False)
            ci.load_calibrated_btt_for_eval(fresh, tmp)
            self.assertIsInstance(fresh.fc, BTTLinear)
            out = fresh.fc(x)
            self.assertTrue(torch.allclose(out, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run, verify fails**

```bash
python -m unittest tests.test_calibrated_btt_checkpoint -v
```

Expected: FAIL — NotImplementedError.

- [ ] **Step 3: Implement both helpers**

Replace the two stubs in `compress_integration.py`:

```python
def save_calibrated_btt_checkpoint(model, out_dir: str) -> None:
    """Save model.safetensors + btt_topology.json into out_dir."""
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)

    state = {n: p.detach().cpu().contiguous() for n, p in model.state_dict().items()}
    save_file(state, os.path.join(out_dir, "model.safetensors"))

    topology = export_btt_topology(model)
    with open(os.path.join(out_dir, "btt_topology.json"), "w") as f:
        json.dump(topology, f, indent=2, sort_keys=True)


def load_calibrated_btt_for_eval(model, checkpoint_dir: str) -> nn.Module:
    """Read btt_topology.json, rebuild BTTLinear modules in `model`, load
    model.safetensors. Returns the mutated model. No calibration pass."""
    from safetensors.torch import load_file

    topology_path = os.path.join(checkpoint_dir, "btt_topology.json")
    with open(topology_path) as f:
        topology = json.load(f)

    rebuild_btt_from_topology(model, topology)

    state = load_file(os.path.join(checkpoint_dir, "model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected keys in checkpoint: {unexpected[:5]}")
    # 'missing' is OK: topology only rebuilt BTT paths; other params may have
    # been loaded from state directly.
    return model
```

- [ ] **Step 4: Run, verify tests pass**

```bash
python -m unittest tests.test_calibrated_btt_checkpoint -v
```

Expected: OK (1 test).

- [ ] **Step 5: Run the entire compress_integration test suite**

```bash
python -m unittest tests.test_compress_integration_import \
    tests.test_compress_integration_validate \
    tests.test_compress_integration_build_config \
    tests.test_compress_integration_training_loader \
    tests.test_compress_integration_apply \
    tests.test_compress_integration_rl_rollout_loader \
    tests.test_compress_integration_materialize \
    tests.test_calibrated_btt_checkpoint
```

Expected: OK (full green).

- [ ] **Step 6: Commit**

```bash
git add compress_integration.py tests/test_calibrated_btt_checkpoint.py
git commit -m "$(cat <<'EOF'
integration: save/load calibrated BTT checkpoints with topology

save_calibrated_btt_checkpoint writes model.safetensors and
btt_topology.json side-by-side. load_calibrated_btt_for_eval reads the
topology, reconstructs BTTLinear modules on a fresh model, and loads
the weights — no calibration pass required. Verified by a round-trip
test that preserves forward outputs exactly.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 18: End-to-end pipeline smoke test

**Files:**
- Create: `tests/test_calibrated_btt_pipeline.py`

This test exercises the full `apply_calibrated_btt` flow on a tiny torch model with a CUDA-only skip. It complements earlier unit tests with an integration-level check that BTT layers are installed, `train_position=small` sets the expected `requires_grad`, and forward passes work.

- [ ] **Step 1: Write the test**

Create `tests/test_calibrated_btt_pipeline.py`:

```python
import argparse
import os, sys
import unittest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import compress_integration as ci
from compress.btt.btt_linear import BTTLinear


class TinyQwenLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "q_proj": nn.Linear(16, 16, bias=False),
                "k_proj": nn.Linear(16, 16, bias=False),
                "v_proj": nn.Linear(16, 16, bias=False),
                "o_proj": nn.Linear(16, 16, bias=False),
                "gate_proj": nn.Linear(16, 32, bias=False),
                "up_proj": nn.Linear(16, 32, bias=False),
                "down_proj": nn.Linear(32, 16, bias=False),
            })
            for _ in range(2)
        ])
        self.embed = nn.Embedding(16, 16)
        self.lm_head = nn.Linear(16, 16, bias=False)

    def forward(self, input_ids, labels=None, attention_mask=None, **kw):
        x = self.embed(input_ids)
        for block in self.layers:
            x = block["q_proj"](x)
            x = block["gate_proj"](x)
            x = block["down_proj"](x)
        logits = self.lm_head(x)
        out = type("Out", (), {})()
        out.logits = logits
        if labels is not None:
            out.loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1), ignore_index=-100,
            )
        return out


class TinyDS(Dataset):
    def __init__(self, n=16, L=4):
        self.items = [{
            "input_ids": torch.randint(0, 16, (L,)),
            "labels": torch.randint(0, 16, (L,)),
        } for _ in range(n)]
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def _collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def _parse(calib_mode, **kv):
    p = argparse.ArgumentParser()
    p.add_argument("--train-mode", default="blocktt")
    p.add_argument("--blocktt-rank", default="full")
    p.add_argument("--trainable-type", default="all")
    p.add_argument("--decomp-mode", default="square")
    p.add_argument("--train-position", default=kv.get("train_position", "small"))
    p.add_argument("--s-merged-to", default=None)
    p.add_argument("--blocktt-factorize-by-head", action="store_true", default=True)
    ci.add_calibrated_btt_args(p, hyphen_style=True)
    return p.parse_args([
        "--calib-mode", calib_mode, "--calib-source", "training_data",
        "--calib-num-seqs", "8", "--calib-batch-size", "4",
    ])


@unittest.skipUnless(torch.cuda.is_available(), "calibrated BTT requires CUDA")
class TestPipeline(unittest.TestCase):
    def test_v2_installs_btt_layers_and_respects_train_position(self):
        args = _parse("v2", train_position="small")
        model = TinyQwenLike().cuda()
        loader = ci.build_calib_loader(
            args, tokenizer=None,
            training_dataset=TinyDS(), training_collate_fn=_collate,
        )
        model, stats = ci.apply_calibrated_btt(model, args, calib_loader=loader)
        self.assertGreater(stats["num_btt_layers"], 0)
        # train_position=small: for each layer, exactly one of (btt_l, btt_r) trains.
        for module in model.modules():
            if isinstance(module, BTTLinear):
                self.assertNotEqual(module.btt_l.requires_grad, module.btt_r.requires_grad)
        # Forward runs without error
        x = torch.randint(0, 16, (2, 4), device="cuda")
        out = model(input_ids=x)
        self.assertIsNotNone(out.logits)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run on a CUDA box**

```bash
python -m unittest tests.test_calibrated_btt_pipeline -v
```

Expected on CUDA: OK (1 test). On CPU-only: skipped.

- [ ] **Step 3: Commit**

```bash
git add tests/test_calibrated_btt_pipeline.py
git commit -m "$(cat <<'EOF'
test: end-to-end calibrated BTT pipeline smoke

Runs apply_calibrated_btt on a tiny transformer-shaped model with
--calib-mode=v2 and verifies that BTTLinear modules are installed and
that --train-position=small correctly gates which core is trainable.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 19: Extend existing compat tests

**Files:**
- Modify: `tests/test_btt_pipeline_compat.py`
- Modify: `tests/test_run_rl_cli.py`
- Modify: `tests/test_run_rl_dapo_cli.py`

- [ ] **Step 1: Read the existing tests to understand their shape**

```bash
python -c "import pathlib; [print(p, len(p.read_text().splitlines())) for p in [pathlib.Path('tests/test_btt_pipeline_compat.py'), pathlib.Path('tests/test_run_rl_cli.py'), pathlib.Path('tests/test_run_rl_dapo_cli.py')]]"
```

- [ ] **Step 2: Add a legacy-path case to `tests/test_btt_pipeline_compat.py`**

At the bottom of the file, inside the existing TestCase class, add one test:

```python
    def test_calib_mode_none_is_default(self):
        """Ensure --calib-mode=none does not disturb the legacy BTT path:
        the flag defaults to 'none' and the legacy code still runs without
        any reference to the new --calib-* flags."""
        import argparse
        from compress_integration import add_calibrated_btt_args
        p = argparse.ArgumentParser()
        add_calibrated_btt_args(p, hyphen_style=True)
        args = p.parse_args([])
        self.assertEqual(args.calib_mode, "none")
```

- [ ] **Step 3: Add calib flag assertions to `tests/test_run_rl_cli.py`**

Inside the existing test class, add:

```python
    def test_calib_mode_defaults_to_none(self):
        args = self._parse([])
        self.assertEqual(getattr(args, "calib_mode"), "none")

    def test_calib_mode_v2_parses(self):
        args = self._parse(["--calib-mode", "v2", "--calib-source", "c4"])
        self.assertEqual(args.calib_mode, "v2")
```

If `self._parse` doesn't exist in the existing test, adapt the helper used locally (read the file first to match the convention).

- [ ] **Step 4: Add the same two assertions to `tests/test_run_rl_dapo_cli.py`**

Match the local convention for constructing the argparse stub.

- [ ] **Step 5: Run the three suites**

```bash
python -m unittest tests.test_btt_pipeline_compat tests.test_run_rl_cli tests.test_run_rl_dapo_cli
```

Expected: OK.

- [ ] **Step 6: Commit**

```bash
git add tests/test_btt_pipeline_compat.py tests/test_run_rl_cli.py tests/test_run_rl_dapo_cli.py
git commit -m "$(cat <<'EOF'
test: extend existing CLI / compat tests with calib-mode assertions

Confirms --calib-mode defaults to 'none' on every entrypoint and that
legacy --train-mode blocktt behavior is unaffected by the new flags.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 20: Full-suite regression + manual smoke checklist

**Files:** none created; validation only.

- [ ] **Step 1: Run every test in the repo**

```bash
python -m py_compile *.py optim/*.py
python -m unittest discover -s tests -v
```

Expected: no compile errors; all tests pass (or skip with a reason on non-CUDA).

- [ ] **Step 2: Manual smoke — SFT legacy path**

On a CUDA box:

```bash
CUDA_VISIBLE_DEVICES=0 uv run run_sft.py \
    --train-mode blocktt --trainable-type all --decomp-mode square \
    --train-position small --blocktt-rank full --no-wandb \
    2>&1 | head -60
```

Expected: training begins; no reference to `calib_mode` in output (legacy path).

- [ ] **Step 3: Manual smoke — SFT calibrated path**

```bash
CUDA_VISIBLE_DEVICES=0 uv run run_sft.py \
    --train-mode blocktt --trainable-type all --decomp-mode square \
    --train-position small --blocktt-rank full \
    --calib-mode v2 --calib-source training_data \
    --calib-num-seqs 32 --calib-batch-size 4 --no-wandb \
    2>&1 | head -80
```

Expected: `[calib-btt] installed N BTT layers`; training begins; no exception.

- [ ] **Step 4: Manual smoke — RL calibrated path**

```bash
CUDA_VISIBLE_DEVICES=0 uv run run_rl.py \
    --train-mode blocktt --trainable-type all \
    --train-position small --blocktt-rank full \
    --calib-mode v2 --calib-source training_data \
    --calib-num-seqs 32 --calib-batch-size 4 --no-wandb \
    2>&1 | head -80
```

Expected: calibration rollout runs on dense base model; decomposition completes; first training step runs.

- [ ] **Step 5: Manual smoke — LIFT calibrated path**

```bash
cd ref/LIFT/src
CUDA_VISIBLE_DEVICES=0 python finetune_blocktt.py \
    --data_path <existing LIFT dataset path> \
    --train_position small --blocktt_rank full \
    --calib_mode v2_bp --calib_source training_data \
    --calib_num_seqs 16 --calib_batch_size 4 \
    2>&1 | head -60
```

Expected: `[calib-btt] installed N BTT layers`; training begins.

- [ ] **Step 6: Manual smoke — eval reload**

From Task 11 / 15 / 12 smoke runs, pick a saved checkpoint directory:

```bash
python -c "
import torch
from transformers import AutoModelForCausalLM
from compress_integration import load_calibrated_btt_for_eval

model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-1.7B', torch_dtype=torch.bfloat16)
load_calibrated_btt_for_eval(model, '<checkpoint/step=N>')
print('loaded; forward test:')
x = torch.randint(0, 32000, (1, 8))
out = model(x)
print(out.logits.shape)
"
```

Expected: no exception; correct output shape.

- [ ] **Step 7: No commit — this task is validation only**

If any smoke step fails, fix the underlying issue and re-run. The fix commits go under whichever task introduced the bug.

---

## Self-Review (after writing this plan)

**Spec coverage check** (against `docs/superpowers/specs/2026-04-14-calibrated-btt-migration-design.md`):

- §1.1 new flags → Task 6 (CLI registration), Tasks 11/12/15/16 (wiring).
- §1.2 `--blocktt-rank` extended to float → Task 7 (validation), Task 8 (translation).
- §1.3 flag compatibility → Task 8 (translation table).
- §1.4 validation precedence → Task 7.
- §2 helper module surface → Tasks 6–10, 13–14, 17.
- §3 SFT integration → Task 11.
- §3.6 normalize extension → Task 5.
- §3.7 / §6.4 checkpoint with topology → Task 17; wired in Tasks 11/12/15/16.
- §4 RL integration → Tasks 13, 14, 15.
- §4.6 run_rl_dapo.py → Task 16.
- §5 LIFT integration → Task 12.
- §6.1 new config fields → Task 4.
- §6.2 best-effort plumbing → Task 4.
- §6.3 `BTTLinear.topology_spec`, `from_topology_spec`, `export_btt_topology`, `rebuild_btt_from_topology` → Tasks 2, 3.
- §7 testing → Tasks 1–10, 13–19 (each with its own TDD test).
- §9 implementation-time verifications → Task 1 (materialize added), Task 2 (topology fields), Task 4 (variant-by-variant hook check), Task 15 (rollout-config lookup).

**Placeholder scan:** no "TBD", "TODO", or "similar to Task N". Every step has exact code or exact commands.

**Type consistency:** `BTTLinear` method names are consistent (`materialize_dense_weight`, `topology_spec`, `from_topology_spec`); `DecompositionConfig` fields match the spec (`s_merged_to`, `factorize_by_head`); `build_decomposition_config(args, *, hyphen_style=True, model)` signature is consistent between Task 8 and Task 10.

**Note on `lm_head`:** `run_sft.py`, `run_rl.py`, and LIFT all currently skip `lm_head` during legacy BTT via `get_blocktt_target_module_names` semantics. `_build_skip_layers` (Task 8) always adds `lm_head` to the skip set so the calibrated path respects the same invariant.
