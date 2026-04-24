# qfura Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement qfura — QLoRA-style NF4 quantization of the frozen BTT core — to fine-tune Llama-3-70B on commonsense reasoning on a single H100, with a forward/backward quantization-error benchmark selecting between `flat` and `per_core_block` layouts.

**Architecture:** New `QBTTLayer(BTTLayer)` subclass in `btt_layer.py` stores the frozen BTT core as `bitsandbytes` `Params4bit` (NF4 + double-quant, bf16 compute dtype); conversion helpers (`quantize_frozen_core_`, `convert_btt_to_qbtt_`) run after the existing `configure_blocktt_trainability(..., train_position="small")` call. A sibling LIFT script `ref/LIFT/src/finetune_qfura.py` reuses the commonsense data loader, swaps the optimizer to `bnb.optim.PagedAdamW8bit`, and dequants-on-save via an overridden `materialize_dense_weight`. A benchmark script produces `docs/reports/qfura-quant-error.md` that is used to pick the default layout.

**Tech Stack:** Python 3.13, PyTorch 2.8, `bitsandbytes>=0.43`, `transformers`, `accelerate`, existing `btt_layer.py` machinery.

**Spec:** `docs/superpowers/specs/2026-04-24-qfura-design.md`

---

## Task 1: Add `bitsandbytes` dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add bitsandbytes to dependencies**

Open `pyproject.toml`. The `[project].dependencies` list currently has:

```toml
dependencies = [
    "datasets>=4.2.0",
    "kernels>=0.10.4",
    "loguru>=0.7.3",
    "math-verify>=0.5",
    "numpy>=2.2.6",
    "peft>=0.17.1",
    "torch>=2.8.0",
    "tqdm>=4.67.1",
    "transformers>=4.57.1",
    "vllm==0.10.2",
    "wandb>=0.22.2",
]
```

Add `"bitsandbytes>=0.43"` keeping alphabetical order:

```toml
dependencies = [
    "bitsandbytes>=0.43",
    "datasets>=4.2.0",
    "kernels>=0.10.4",
    "loguru>=0.7.3",
    "math-verify>=0.5",
    "numpy>=2.2.6",
    "peft>=0.17.1",
    "torch>=2.8.0",
    "tqdm>=4.67.1",
    "transformers>=4.57.1",
    "vllm==0.10.2",
    "wandb>=0.22.2",
]
```

- [ ] **Step 2: Sync the environment**

Run: `uv sync`
Expected: `uv.lock` updates with `bitsandbytes==0.43.x` and its dependencies; command exits 0.

- [ ] **Step 3: Verify import works**

Run: `uv run python -c "import bitsandbytes as bnb; print(bnb.__version__)"`
Expected: version string like `0.43.x` prints to stdout. If it fails with a CUDA runtime error, the environment's CUDA does not match — escalate to the user before continuing.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add bitsandbytes for qfura NF4 quantization"
```

---

## Task 2: Add `QBTTLayer` skeleton (no forward yet) and flat-layout quantization helper

**Files:**
- Modify: `btt_layer.py` (append new class and helper functions at end of file, before the trailing `if _os.environ.get("FURA_FUSED_STEP2") == "1":` block)
- Test: `tests/test_qbtt_conversion.py` (new)

The skeleton holds quantized state + shape metadata. `forward` lands in Task 4. Conversion (Task 3) depends on this being in place.

- [ ] **Step 1: Write the failing test**

Create `tests/test_qbtt_conversion.py`:

```python
import unittest

import torch
import torch.nn as nn

from btt_layer import (
    BTTLayer,
    QBTTLayer,
    configure_blocktt_trainability,
    convert_linear_to_btt,
    quantize_frozen_core_,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for bnb NF4")
class TestQBTTConversionFlat(unittest.TestCase):
    def _make_btt(self):
        torch.manual_seed(0)
        linear = nn.Linear(64, 64, bias=True).cuda().to(torch.bfloat16)
        model = nn.Module()
        model.add_module("q_proj", linear)
        convert_linear_to_btt(
            model,
            btt_rank="full",
            decomp_mode="square",
            include_names=("q_proj",),
            train_position="small",
            s_merged_to="frozen",
        )
        configure_blocktt_trainability(model, train_position="small")
        return model.q_proj

    def test_flat_layout_replaces_btt_with_qbtt(self):
        btt = self._make_btt()
        self.assertIsInstance(btt, BTTLayer)
        qbtt = quantize_frozen_core_(btt, layout="flat")
        self.assertIsInstance(qbtt, QBTTLayer)
        # Frozen side must have been removed from the module's parameter list.
        # Trainable side retains its btt_* metadata for the Muon optimizer.
        trainable = qbtt.btt_r if qbtt._qfura_frozen_side == "btt_l" else qbtt.btt_l
        self.assertTrue(trainable.requires_grad)
        self.assertTrue(hasattr(trainable, "btt_layout"))
        self.assertEqual(trainable.btt_layout, "new")

    def test_flat_layout_dequant_roundtrip_within_tolerance(self):
        btt = self._make_btt()
        frozen_side = "btt_l" if btt.btt_l.numel() >= btt.btt_r.numel() else "btt_r"
        frozen_before = getattr(btt, frozen_side).detach().clone()
        qbtt = quantize_frozen_core_(btt, layout="flat")
        frozen_after = qbtt._dequantize_frozen_core()
        self.assertEqual(frozen_after.shape, frozen_before.shape)
        rel_err = (
            (frozen_after.float() - frozen_before.float()).norm()
            / frozen_before.float().norm().clamp_min(1e-12)
        )
        self.assertLess(rel_err.item(), 0.15)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_qbtt_conversion -v`
Expected: `ImportError: cannot import name 'QBTTLayer' from 'btt_layer'`.

- [ ] **Step 3: Implement `QBTTLayer` skeleton + flat `quantize_frozen_core_`**

Open `btt_layer.py`. Find the line `import os as _os` near the end. Insert the following code **before** that line:

```python
try:
    import bitsandbytes as _bnb
    _HAS_BNB = True
except ImportError:
    _bnb = None
    _HAS_BNB = False


class QBTTLayer(BTTLayer):
    """BTTLayer with the frozen core stored as NF4 via bitsandbytes.

    The frozen side is determined by which BTTLayer core has requires_grad=False
    after configure_blocktt_trainability. The trainable core and btt_s remain as
    regular bf16 nn.Parameters.

    Attributes:
      _qfura_layout: "flat" or "per_core_block".
      _qfura_frozen_side: "btt_l" or "btt_r".
      _qfura_frozen_shape: tuple, original 3D shape of the frozen core.
      _qfura_frozen_dtype: torch.dtype, dtype pre-quantization.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._qfura_layout = None
        self._qfura_frozen_side = None
        self._qfura_frozen_shape = None
        self._qfura_frozen_dtype = None

    def _dequantize_frozen_core(self):
        """Dequant the frozen NF4 core back to its original 3D bf16 layout."""
        if self._qfura_layout == "flat":
            # _qfura_frozen_flat is a Params4bit of shape (numel, 1).
            dequanted = _bnb.functional.dequantize_4bit(
                self._qfura_frozen_flat.data,
                quant_state=self._qfura_frozen_flat.quant_state,
            )
            return dequanted.reshape(self._qfura_frozen_shape).to(
                self._qfura_frozen_dtype
            )
        elif self._qfura_layout == "per_core_block":
            blocks = []
            for params_4bit in self._qfura_frozen_blocks:
                deq = _bnb.functional.dequantize_4bit(
                    params_4bit.data, quant_state=params_4bit.quant_state
                )
                blocks.append(deq)
            stacked = torch.stack(blocks, dim=0)
            return stacked.reshape(self._qfura_frozen_shape).to(
                self._qfura_frozen_dtype
            )
        else:
            raise RuntimeError(
                f"QBTTLayer has invalid _qfura_layout: {self._qfura_layout}"
            )


def _pick_frozen_side(btt_layer):
    """Return 'btt_l' or 'btt_r' depending on which side is frozen after
    configure_blocktt_trainability."""
    l_trainable = btt_layer.btt_l.requires_grad
    r_trainable = btt_layer.btt_r.requires_grad
    if l_trainable and r_trainable:
        raise ValueError(
            "quantize_frozen_core_ requires exactly one frozen BTT core. "
            "Both btt_l and btt_r have requires_grad=True; this is train_position='both'."
        )
    if not l_trainable and not r_trainable:
        raise ValueError(
            "quantize_frozen_core_ requires exactly one frozen BTT core. "
            "Neither btt_l nor btt_r has requires_grad=True; trainability was not configured."
        )
    return "btt_r" if l_trainable else "btt_l"


def quantize_frozen_core_(
    btt_layer,
    layout,
    compute_dtype=torch.bfloat16,
    double_quant=True,
    quant_type="nf4",
):
    """Mutate `btt_layer` in place: replace its frozen BTT core with an NF4 blob.

    Returns a QBTTLayer (same Python object, reclassed via __class__ assignment).
    """
    if not _HAS_BNB:
        raise ImportError("bitsandbytes is not installed; required for qfura")
    if layout not in {"flat", "per_core_block"}:
        raise ValueError("layout must be 'flat' or 'per_core_block'")
    if not isinstance(btt_layer, BTTLayer):
        raise TypeError(f"expected BTTLayer, got {type(btt_layer).__name__}")

    frozen_side = _pick_frozen_side(btt_layer)
    frozen_param = getattr(btt_layer, frozen_side)
    frozen_shape = tuple(frozen_param.shape)
    frozen_dtype = frozen_param.dtype

    # Reclass to QBTTLayer without re-running __init__.
    btt_layer.__class__ = QBTTLayer
    btt_layer._qfura_layout = layout
    btt_layer._qfura_frozen_side = frozen_side
    btt_layer._qfura_frozen_shape = frozen_shape
    btt_layer._qfura_frozen_dtype = frozen_dtype

    if layout == "flat":
        flat = frozen_param.detach().reshape(-1, 1).contiguous()
        p4 = _bnb.nn.Params4bit(
            flat,
            requires_grad=False,
            quant_type=quant_type,
            compress_statistics=double_quant,
            quant_storage=torch.uint8,
        )
        # Params4bit triggers NF4 quantization on .to(device). Must land on CUDA.
        p4 = p4.to(device=frozen_param.device)
        btt_layer._qfura_frozen_flat = p4
    else:  # per_core_block
        # btt_l shape (m, rank*n, a): one block per m axis.
        # btt_r shape (n, b, m*rank): one block per n axis.
        block_list = []
        outer = frozen_shape[0]
        for i in range(outer):
            block = frozen_param[i].detach().contiguous()
            p4 = _bnb.nn.Params4bit(
                block,
                requires_grad=False,
                quant_type=quant_type,
                compress_statistics=double_quant,
                quant_storage=torch.uint8,
            )
            p4 = p4.to(device=frozen_param.device)
            block_list.append(p4)
        # Store as a ParameterList substitute via regular list (not registered as
        # nn.Parameter; Params4bit handles its own state).
        btt_layer._qfura_frozen_blocks = block_list

    # Remove the frozen core from the module's parameter list.
    delattr(btt_layer, frozen_side)

    return btt_layer


def convert_btt_to_qbtt_(model, layout):
    """Walk `model` and replace every BTTLayer (with trainability configured)
    with a QBTTLayer. Returns stats dict."""
    num_converted = 0
    bytes_saved = 0
    names = []
    for name, module in model.named_modules():
        if not isinstance(module, BTTLayer):
            continue
        if isinstance(module, QBTTLayer):
            continue  # already converted
        # Only convert if exactly one core is frozen.
        l_train = module.btt_l.requires_grad
        r_train = module.btt_r.requires_grad
        if l_train == r_train:
            continue
        frozen_param = module.btt_r if l_train else module.btt_l
        bf16_bytes = frozen_param.numel() * 2  # bf16 is 2 bytes/elem
        nf4_bytes = frozen_param.numel() // 2  # nf4 is 0.5 bytes/elem
        bytes_saved += bf16_bytes - nf4_bytes
        quantize_frozen_core_(module, layout=layout)
        num_converted += 1
        names.append(name)
    return {
        "num_converted": num_converted,
        "bytes_saved": bytes_saved,
        "layout": layout,
        "names": names,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_qbtt_conversion -v`
Expected: two tests pass (if CUDA available) or skip cleanly (if not).

- [ ] **Step 5: Commit**

```bash
git add btt_layer.py tests/test_qbtt_conversion.py
git commit -m "qfura: add QBTTLayer skeleton and quantize_frozen_core_ (flat layout)"
```

---

## Task 3: Add per-core-block layout tests and confirm Task 2's per-block branch works

**Files:**
- Modify: `tests/test_qbtt_conversion.py`

Task 2 already wrote the per-block branch in `quantize_frozen_core_`. This task verifies it with a dedicated test and catches edge cases.

- [ ] **Step 1: Append per-block test cases**

Append to `tests/test_qbtt_conversion.py` (inside the existing file, add a new test class at the bottom before `if __name__ == "__main__":`):

```python
@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for bnb NF4")
class TestQBTTConversionPerCoreBlock(unittest.TestCase):
    def _make_btt(self):
        torch.manual_seed(0)
        linear = nn.Linear(64, 64, bias=True).cuda().to(torch.bfloat16)
        model = nn.Module()
        model.add_module("q_proj", linear)
        convert_linear_to_btt(
            model,
            btt_rank="full",
            decomp_mode="square",
            include_names=("q_proj",),
            train_position="small",
            s_merged_to="frozen",
        )
        configure_blocktt_trainability(model, train_position="small")
        return model.q_proj

    def test_per_core_block_produces_one_params4bit_per_outer_block(self):
        btt = self._make_btt()
        frozen_side = "btt_l" if btt.btt_l.numel() >= btt.btt_r.numel() else "btt_r"
        frozen_outer = getattr(btt, frozen_side).shape[0]
        qbtt = quantize_frozen_core_(btt, layout="per_core_block")
        self.assertEqual(len(qbtt._qfura_frozen_blocks), frozen_outer)

    def test_per_core_block_dequant_roundtrip_within_tolerance(self):
        btt = self._make_btt()
        frozen_side = "btt_l" if btt.btt_l.numel() >= btt.btt_r.numel() else "btt_r"
        frozen_before = getattr(btt, frozen_side).detach().clone()
        qbtt = quantize_frozen_core_(btt, layout="per_core_block")
        frozen_after = qbtt._dequantize_frozen_core()
        self.assertEqual(frozen_after.shape, frozen_before.shape)
        rel_err = (
            (frozen_after.float() - frozen_before.float()).norm()
            / frozen_before.float().norm().clamp_min(1e-12)
        )
        self.assertLess(rel_err.item(), 0.15)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for bnb NF4")
class TestQBTTConversionModelWide(unittest.TestCase):
    def test_convert_btt_to_qbtt_counts_all_layers(self):
        torch.manual_seed(0)

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64, bias=False)
                self.k_proj = nn.Linear(64, 64, bias=False)

        model = Block().cuda().to(torch.bfloat16)
        convert_linear_to_btt(
            model,
            btt_rank="full",
            decomp_mode="square",
            include_names=("q_proj", "k_proj"),
            train_position="small",
            s_merged_to="frozen",
        )
        configure_blocktt_trainability(model, train_position="small")
        stats = convert_btt_to_qbtt_(model, layout="flat")
        self.assertEqual(stats["num_converted"], 2)
        self.assertGreater(stats["bytes_saved"], 0)
        self.assertIsInstance(model.q_proj, QBTTLayer)
        self.assertIsInstance(model.k_proj, QBTTLayer)
```

- [ ] **Step 2: Run the new tests**

Run: `uv run python -m unittest tests.test_qbtt_conversion -v`
Expected: all five tests pass on a CUDA host; skip cleanly otherwise.

- [ ] **Step 3: Commit**

```bash
git add tests/test_qbtt_conversion.py
git commit -m "qfura: test per-core-block layout + model-wide conversion"
```

---

## Task 4: Implement `QBTTLayer.forward`

**Files:**
- Modify: `btt_layer.py` (add a `forward` override inside the existing `QBTTLayer` class body)
- Test: `tests/test_qbtt_forward_shape.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_qbtt_forward_shape.py`:

```python
import copy
import unittest

import torch
import torch.nn as nn

from btt_layer import (
    BTTLayer,
    QBTTLayer,
    configure_blocktt_trainability,
    convert_linear_to_btt,
    quantize_frozen_core_,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for bnb NF4")
class TestQBTTForward(unittest.TestCase):
    def _make_pair(self, layout):
        torch.manual_seed(0)
        linear = nn.Linear(64, 64, bias=True).cuda().to(torch.bfloat16)
        model = nn.Module()
        model.add_module("q_proj", linear)
        convert_linear_to_btt(
            model,
            btt_rank="full",
            decomp_mode="square",
            include_names=("q_proj",),
            train_position="small",
            s_merged_to="frozen",
        )
        configure_blocktt_trainability(model, train_position="small")
        btt_ref = copy.deepcopy(model.q_proj)
        qbtt = quantize_frozen_core_(model.q_proj, layout=layout)
        return btt_ref, qbtt

    def _check_forward(self, layout):
        btt, qbtt = self._make_pair(layout)
        torch.manual_seed(1)
        x = torch.randn(4, 8, 64, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            y_ref = btt(x)
            y_q = qbtt(x)
        self.assertEqual(y_ref.shape, y_q.shape)
        rel_err = (
            (y_ref.float() - y_q.float()).norm()
            / y_ref.float().norm().clamp_min(1e-12)
        )
        self.assertLess(rel_err.item(), 0.15)

    def test_forward_flat(self):
        self._check_forward("flat")

    def test_forward_per_core_block(self):
        self._check_forward("per_core_block")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_qbtt_forward_shape -v`
Expected: `AttributeError` on `QBTTLayer` forward or wrong shape — `QBTTLayer` currently inherits `BTTLayer.forward`, which reads `self.btt_l` / `self.btt_r` directly, but the frozen side was `delattr`'d. Should raise `AttributeError: 'QBTTLayer' object has no attribute 'btt_l'` (or btt_r).

- [ ] **Step 3: Implement `QBTTLayer.forward`**

Edit `btt_layer.py`. Inside the `QBTTLayer` class body (after `_dequantize_frozen_core`), add this method:

```python
    def forward(self, x):
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"QBTTLayer expected last dim {self.in_features}, got {x.shape[-1]}"
            )

        # Dequant the frozen core to its original 3D bf16 layout.
        frozen_dequanted = self._dequantize_frozen_core()
        if self._qfura_frozen_side == "btt_l":
            btt_l = frozen_dequanted
            btt_r = self.btt_r
        else:
            btt_l = self.btt_l
            btt_r = frozen_dequanted

        orig_shape = x.shape
        x = x.reshape(-1, self.n, self.b)
        batch_n = x.shape[0]
        x_t = x.transpose(0, 1).contiguous()

        # Step 1: (n, B, b) @ (n, b, m*r) -> (n, B, m*r)
        inner_up = torch.bmm(x_t, btt_r)
        inner_up = inner_up.reshape(self.n, batch_n, self.m, self.rank)
        inner_up = inner_up.permute(2, 1, 0, 3).contiguous()

        if self.use_gate_proj:
            inner_gate = torch.bmm(x_t, self.btt_g)
            inner_gate = inner_gate.reshape(self.n, batch_n, self.m, self.rank)
            inner_gate = inner_gate.permute(2, 1, 0, 3).contiguous()
            inner = torch.nn.functional.silu(inner_gate) * inner_up
        else:
            inner = inner_up
            if hasattr(self, "act_fn"):
                inner = self.act_fn(inner)

        # Step 2: (m, B, n*r) @ (m, n*r, a) -> (m, B, a)
        if BTTLayer.use_fused_step2 and inner.is_cuda:
            from fura_kernels import step2_s_scaled_bmm
            out = step2_s_scaled_bmm(
                inner.reshape(self.m, batch_n, self.rank * self.n),
                btt_l,
                self.btt_s,
            )
        else:
            if self.btt_s is not None:
                btt_l = (
                    btt_l.reshape(self.m, self.n, self.rank, self.a)
                    * self.btt_s.unsqueeze(-1)
                ).reshape(self.m, self.rank * self.n, self.a)
            out = torch.bmm(
                inner.reshape(self.m, batch_n, self.rank * self.n),
                btt_l,
            )
        out = out.permute(1, 0, 2).contiguous().reshape(
            *orig_shape[:-1], self.out_features
        )

        if self.bias is not None:
            out += self.bias

        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_qbtt_forward_shape -v`
Expected: two tests pass.

- [ ] **Step 5: Commit**

```bash
git add btt_layer.py tests/test_qbtt_forward_shape.py
git commit -m "qfura: implement QBTTLayer.forward with JIT dequant of frozen core"
```

---

## Task 5: Gradient-flow test — only trainable core + btt_s get gradients

**Files:**
- Test: `tests/test_qbtt_gradient_flow.py` (new)

No implementation changes — this verifies Task 2's `delattr` and `Params4bit(requires_grad=False)` behavior.

- [ ] **Step 1: Write the test**

Create `tests/test_qbtt_gradient_flow.py`:

```python
import unittest

import torch
import torch.nn as nn

from btt_layer import (
    QBTTLayer,
    configure_blocktt_trainability,
    convert_linear_to_btt,
    quantize_frozen_core_,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for bnb NF4")
class TestQBTTGradientFlow(unittest.TestCase):
    def _make_qbtt(self, s_merged_to, layout):
        torch.manual_seed(0)
        linear = nn.Linear(64, 64, bias=True).cuda().to(torch.bfloat16)
        model = nn.Module()
        model.add_module("q_proj", linear)
        convert_linear_to_btt(
            model,
            btt_rank="full",
            decomp_mode="square",
            include_names=("q_proj",),
            train_position="small",
            s_merged_to=s_merged_to,
        )
        configure_blocktt_trainability(
            model,
            train_position="small",
            train_singular_values=(s_merged_to == "keep_trainable"),
        )
        return quantize_frozen_core_(model.q_proj, layout=layout)

    def test_only_trainable_core_and_btt_s_receive_gradients(self):
        qbtt = self._make_qbtt(s_merged_to="keep_trainable", layout="flat")
        x = torch.randn(2, 8, 64, device="cuda", dtype=torch.bfloat16, requires_grad=False)
        y = qbtt(x)
        loss = y.pow(2).mean()
        loss.backward()

        # Trainable core should have a grad.
        trainable = qbtt.btt_r if qbtt._qfura_frozen_side == "btt_l" else qbtt.btt_l
        self.assertIsNotNone(trainable.grad)
        self.assertEqual(trainable.grad.shape, trainable.shape)

        # btt_s is trainable under keep_trainable; should also have grad.
        self.assertIsNotNone(qbtt.btt_s)
        self.assertIsNotNone(qbtt.btt_s.grad)

        # Bias should have a grad (default train_bias=True).
        self.assertIsNotNone(qbtt.bias.grad)

    def test_frozen_params4bit_has_no_grad(self):
        qbtt = self._make_qbtt(s_merged_to="frozen", layout="flat")
        # The frozen Params4bit is stored as _qfura_frozen_flat; requires_grad=False.
        self.assertFalse(qbtt._qfura_frozen_flat.requires_grad)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test**

Run: `uv run python -m unittest tests.test_qbtt_gradient_flow -v`
Expected: both tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_qbtt_gradient_flow.py
git commit -m "qfura: test gradient flow excludes frozen NF4 core"
```

---

## Task 6: Quantization-error unit test (guardrail thresholds)

**Files:**
- Test: `tests/test_qbtt_quant_error.py` (new)

- [ ] **Step 1: Write the test**

Create `tests/test_qbtt_quant_error.py`:

```python
import copy
import unittest

import torch
import torch.nn as nn

from btt_layer import (
    BTTLayer,
    configure_blocktt_trainability,
    convert_linear_to_btt,
    quantize_frozen_core_,
)


FORWARD_THRESHOLDS = {"flat": 0.05, "per_core_block": 0.03}
BACKWARD_THRESHOLD = 0.10


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for bnb NF4")
class TestQBTTQuantError(unittest.TestCase):
    def _build_reference(self):
        torch.manual_seed(0)
        linear = nn.Linear(4096, 4096, bias=False).cuda().to(torch.bfloat16)
        with torch.no_grad():
            linear.weight.normal_(mean=0.0, std=0.02)
        model = nn.Module()
        model.add_module("q_proj", linear)
        convert_linear_to_btt(
            model,
            btt_rank="full",
            decomp_mode="square",
            include_names=("q_proj",),
            train_position="small",
            s_merged_to="frozen",
        )
        configure_blocktt_trainability(model, train_position="small")
        return model.q_proj

    def _check_layout(self, layout):
        ref = self._build_reference()
        cloned = copy.deepcopy(ref)
        q = quantize_frozen_core_(cloned, layout=layout)

        # Deterministic inputs + targets.
        torch.manual_seed(1)
        x = torch.randn(4, 128, 4096, device="cuda", dtype=torch.bfloat16)
        target = torch.randn(4, 128, 4096, device="cuda", dtype=torch.bfloat16)

        # Forward.
        y_ref = ref(x)
        y_q = q(x)
        fwd_err = (
            (y_ref.float() - y_q.float()).norm()
            / y_ref.float().norm().clamp_min(1e-12)
        ).item()

        # Backward: MSE loss, grad w.r.t. the trainable core.
        def grad_of(module):
            for p in module.parameters():
                if p.grad is not None:
                    p.grad = None
            loss = (module(x) - target).pow(2).mean()
            loss.backward()
            # Collect the first parameter with grad (the trainable core).
            trainable_params = [p for p in module.parameters() if p.requires_grad]
            return [p.grad.detach().clone() for p in trainable_params]

        g_ref = grad_of(ref)
        g_q = grad_of(q)

        bwd_errs = []
        for a, b in zip(g_ref, g_q):
            denom = a.float().norm().clamp_min(1e-12)
            bwd_errs.append(((a.float() - b.float()).norm() / denom).item())
        bwd_err = max(bwd_errs)

        self.assertLess(
            fwd_err,
            FORWARD_THRESHOLDS[layout],
            f"{layout}: forward error {fwd_err:.4f} exceeds {FORWARD_THRESHOLDS[layout]}",
        )
        self.assertLess(
            bwd_err,
            BACKWARD_THRESHOLD,
            f"{layout}: backward error {bwd_err:.4f} exceeds {BACKWARD_THRESHOLD}",
        )

    def test_quant_error_flat(self):
        self._check_layout("flat")

    def test_quant_error_per_core_block(self):
        self._check_layout("per_core_block")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test**

Run: `uv run python -m unittest tests.test_qbtt_quant_error -v`

Expected: both tests pass. If forward or backward error exceeds thresholds, the thresholds are too tight — open `tests/test_qbtt_quant_error.py`, loosen only the failing threshold to the observed value rounded up to the next 0.01, and add a comment `# Threshold calibrated from initial benchmark run 2026-04-24`. Do not loosen speculatively.

- [ ] **Step 3: Commit**

```bash
git add tests/test_qbtt_quant_error.py
git commit -m "qfura: quantization error thresholds (forward + backward)"
```

---

## Task 7: Fused Step-2 compatibility test

**Files:**
- Test: `tests/test_qbtt_fused_step2_compat.py` (new)

Verifies that `QBTTLayer.forward` produces the same result under `FURA_FUSED_STEP2=1` as without it.

- [ ] **Step 1: Write the test**

Create `tests/test_qbtt_fused_step2_compat.py`:

```python
import copy
import unittest

import torch
import torch.nn as nn

from btt_layer import (
    BTTLayer,
    configure_blocktt_trainability,
    convert_linear_to_btt,
    quantize_frozen_core_,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for bnb NF4")
class TestQBTTFusedStep2Compat(unittest.TestCase):
    def test_fused_and_nonfused_match(self):
        torch.manual_seed(0)
        linear = nn.Linear(64, 64, bias=False).cuda().to(torch.bfloat16)
        model = nn.Module()
        model.add_module("q_proj", linear)
        convert_linear_to_btt(
            model,
            btt_rank="full",
            decomp_mode="square",
            include_names=("q_proj",),
            train_position="small",
            s_merged_to="keep_frozen",
        )
        configure_blocktt_trainability(model, train_position="small")
        qbtt = quantize_frozen_core_(model.q_proj, layout="flat")

        x = torch.randn(2, 8, 64, device="cuda", dtype=torch.bfloat16)

        # Non-fused.
        BTTLayer.use_fused_step2 = False
        y_nonfused = qbtt(x)

        # Fused.
        try:
            BTTLayer.use_fused_step2 = True
            y_fused = qbtt(x)
        except (ImportError, RuntimeError) as e:
            self.skipTest(f"Fused kernel unavailable: {e}")
        finally:
            BTTLayer.use_fused_step2 = False

        rel_err = (
            (y_nonfused.float() - y_fused.float()).norm()
            / y_nonfused.float().norm().clamp_min(1e-12)
        )
        self.assertLess(rel_err.item(), 0.02)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test**

Run: `uv run python -m unittest tests.test_qbtt_fused_step2_compat -v`
Expected: test passes or skips cleanly if the fused kernel is unavailable on this host.

- [ ] **Step 3: Commit**

```bash
git add tests/test_qbtt_fused_step2_compat.py
git commit -m "qfura: verify fused step2 kernel parity"
```

---

## Task 8: Override `QBTTLayer.materialize_dense_weight` for checkpoint save

**Files:**
- Modify: `btt_layer.py` (add a method inside `QBTTLayer`)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_qbtt_conversion.py` (before `if __name__ == "__main__":`):

```python
@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for bnb NF4")
class TestQBTTMaterialize(unittest.TestCase):
    def test_materialize_dense_weight_close_to_btt(self):
        import copy as _copy

        torch.manual_seed(0)
        linear = nn.Linear(64, 64, bias=True).cuda().to(torch.bfloat16)
        model = nn.Module()
        model.add_module("q_proj", linear)
        convert_linear_to_btt(
            model,
            btt_rank="full",
            decomp_mode="square",
            include_names=("q_proj",),
            train_position="small",
            s_merged_to="frozen",
        )
        configure_blocktt_trainability(model, train_position="small")
        btt_ref = _copy.deepcopy(model.q_proj)
        qbtt = quantize_frozen_core_(model.q_proj, layout="flat")

        w_ref = btt_ref.materialize_dense_weight()
        w_q = qbtt.materialize_dense_weight()

        self.assertEqual(w_q.shape, w_ref.shape)
        rel_err = (
            (w_ref.float() - w_q.float()).norm()
            / w_ref.float().norm().clamp_min(1e-12)
        )
        self.assertLess(rel_err.item(), 0.15)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_qbtt_conversion.TestQBTTMaterialize -v`
Expected: `AttributeError: 'QBTTLayer' object has no attribute 'btt_l'` or `btt_r` (the inherited `materialize_dense_weight` reads the deleted attribute).

- [ ] **Step 3: Implement the override**

Edit `btt_layer.py`. Inside the `QBTTLayer` class body (after `forward`), add:

```python
    @torch.no_grad()
    def materialize_dense_weight(self):
        """Dequant the frozen core and materialize the dense bf16 weight.

        Used at checkpoint save time: the saved HF-format weight is the dequanted
        BTT factorization, which introduces a one-shot quantization round-trip
        error (documented in the design spec).
        """
        if self.lr_act:
            raise ValueError("Dense materialization only supports lr_act=False")
        frozen_dequanted = self._dequantize_frozen_core()
        if self._qfura_frozen_side == "btt_l":
            btt_l = frozen_dequanted
            btt_r = self.btt_r
        else:
            btt_l = self.btt_l
            btt_r = frozen_dequanted
        r = btt_r.reshape(self.n, self.b, self.m, self.rank).permute(2, 0, 3, 1)
        l = btt_l.reshape(self.m, self.n, self.rank, self.a)
        if self.btt_s is not None:
            l = l * self.btt_s.unsqueeze(-1)
        w_blocks = torch.einsum("mnra,mnrb->mnab", l, r)
        return w_blocks.permute(0, 2, 1, 3).reshape(
            self.out_features, self.in_features
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_qbtt_conversion.TestQBTTMaterialize -v`
Expected: passes.

- [ ] **Step 5: Run the full conversion test suite**

Run: `uv run python -m unittest tests.test_qbtt_conversion -v`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add btt_layer.py tests/test_qbtt_conversion.py
git commit -m "qfura: QBTTLayer.materialize_dense_weight for HF checkpoint save"
```

---

## Task 9: Quantization-error benchmark script (layer-level + model-level)

**Files:**
- Create: `analysis/bench_qbtt_quant_error.py`

This script produces `docs/reports/qfura-quant-error.md` when run. It is not CI-wired; it is a manual artifact producer.

- [ ] **Step 1: Create the benchmark script**

Create `analysis/bench_qbtt_quant_error.py`:

```python
"""Quantization-error benchmark for qfura.

Measures forward + backward error for both 'flat' and 'per_core_block' layouts
across every target linear in a model, plus a model-level logit error.

Usage:
  uv run python analysis/bench_qbtt_quant_error.py \
      --model meta-llama/Meta-Llama-3-8B \
      --data-path ref/LIFT/LLM-Adapters/ft-training_set/commonsense_170k.json \
      --num-prompts 32 \
      --output docs/reports/qfura-quant-error.md

For Llama-3-70B, add --model meta-llama/Meta-Llama-3-70B. Requires an H100.
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from btt_layer import (
    BTTLayer,
    configure_blocktt_trainability,
    convert_linear_to_btt,
    quantize_frozen_core_,
)


TARGET_NAMES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Meta-Llama-3-8B")
    p.add_argument(
        "--data-path",
        default="ref/LIFT/LLM-Adapters/ft-training_set/commonsense_170k.json",
    )
    p.add_argument("--num-prompts", type=int, default=32)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument(
        "--output", default="docs/reports/qfura-quant-error.md"
    )
    return p.parse_args()


def load_prompts(data_path, num_prompts):
    with open(data_path) as f:
        data = json.load(f)
    prompts = []
    for entry in data[:num_prompts]:
        text = entry.get("instruction", "") + "\n" + entry.get("input", "")
        prompts.append(text.strip())
    return prompts


def tokenize_batch(tokenizer, prompts, max_seq_len):
    return tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_seq_len,
    )


@torch.no_grad()
def capture_layer_inputs(model, prompt_batch, target_names):
    """Run one forward pass; record inputs to every targeted linear in every layer."""
    captures = {}

    def make_hook(name):
        def hook(module, inputs, output):
            # Store on CPU to save GPU memory.
            captures.setdefault(name, []).append(inputs[0].detach().cpu())
        return hook

    handles = []
    for name, module in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in target_names and isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(make_hook(name)))
    model(**prompt_batch)
    for h in handles:
        h.remove()
    return captures


def per_layer_error(linear, layer_inputs, layout):
    """Convert a single nn.Linear -> BTT -> QBTT; measure forward + backward rel err."""
    device = linear.weight.device
    dtype = linear.weight.dtype
    # Wrap in a throwaway parent so convert_linear_to_btt can mutate it.
    parent = nn.Module()
    parent.add_module("target", copy.deepcopy(linear))
    convert_linear_to_btt(
        parent,
        btt_rank="full",
        decomp_mode="square",
        include_names=("target",),
        train_position="small",
        s_merged_to="frozen",
    )
    configure_blocktt_trainability(parent, train_position="small")
    btt_ref = copy.deepcopy(parent.target)
    qbtt = quantize_frozen_core_(parent.target, layout=layout)

    fwd_errs = []
    for xin in layer_inputs:
        xin = xin.to(device=device, dtype=dtype)
        y_ref = btt_ref(xin)
        y_q = qbtt(xin)
        denom = y_ref.float().norm().clamp_min(1e-12)
        fwd_errs.append(
            ((y_ref.float() - y_q.float()).norm() / denom).item()
        )

    # Backward on the first captured input.
    xin = layer_inputs[0].to(device=device, dtype=dtype)
    target = torch.randn_like(btt_ref(xin))

    def grad_of(module):
        for p in module.parameters():
            if p.grad is not None:
                p.grad = None
        loss = (module(xin) - target).pow(2).mean()
        loss.backward()
        trainable = [p for p in module.parameters() if p.requires_grad]
        return [p.grad.detach().clone() for p in trainable]

    g_ref = grad_of(btt_ref)
    g_q = grad_of(qbtt)
    bwd_errs = []
    for a, b in zip(g_ref, g_q):
        denom = a.float().norm().clamp_min(1e-12)
        bwd_errs.append(((a.float() - b.float()).norm() / denom).item())

    return {
        "fwd_err_mean": sum(fwd_errs) / len(fwd_errs),
        "fwd_err_p50": sorted(fwd_errs)[len(fwd_errs) // 2],
        "fwd_err_p95": sorted(fwd_errs)[int(len(fwd_errs) * 0.95)],
        "bwd_err_max": max(bwd_errs),
    }


@torch.no_grad()
def model_level_error(model_bf16, model_qfura, prompt_batch):
    out_bf16 = model_bf16(**prompt_batch).logits
    out_q = model_qfura(**prompt_batch).logits
    top1_match = (out_bf16.argmax(-1) == out_q.argmax(-1)).float().mean().item()
    # KL(bf16 || qfura), averaged over tokens.
    log_p = torch.log_softmax(out_bf16.float(), dim=-1)
    log_q = torch.log_softmax(out_q.float(), dim=-1)
    kl = (log_p.exp() * (log_p - log_q)).sum(-1).mean().item()
    logit_rel = (
        (out_bf16.float() - out_q.float()).norm()
        / out_bf16.float().norm().clamp_min(1e-12)
    ).item()
    return {"top1_match": top1_match, "kl": kl, "logit_rel_err": logit_rel}


def aggregate_by_leaf(per_layer_results):
    """Group per-layer results by leaf name (q_proj, k_proj, ...)."""
    by_leaf = {}
    for full_name, metrics in per_layer_results.items():
        leaf = full_name.split(".")[-1]
        by_leaf.setdefault(leaf, []).append(metrics)
    agg = {}
    for leaf, rows in by_leaf.items():
        fwd = [r["fwd_err_mean"] for r in rows]
        bwd = [r["bwd_err_max"] for r in rows]
        agg[leaf] = {
            "fwd_mean": sum(fwd) / len(fwd),
            "fwd_p95": sorted(fwd)[int(len(fwd) * 0.95)] if len(fwd) > 1 else fwd[0],
            "bwd_mean": sum(bwd) / len(bwd),
            "n_layers": len(rows),
        }
    return agg


def format_report(args, model_level, per_leaf_flat, per_leaf_pcb):
    lines = [
        "# qfura Quantization-Error Report",
        "",
        f"**Model:** `{args.model}`",
        f"**Num prompts:** {args.num_prompts}",
        f"**Max seq len:** {args.max_seq_len}",
        "",
        "## Reproduce",
        "",
        "```bash",
        "uv run python analysis/bench_qbtt_quant_error.py \\",
        f"    --model {args.model} \\",
        f"    --data-path {args.data_path} \\",
        f"    --num-prompts {args.num_prompts}",
        "```",
        "",
        "## Model-level error",
        "",
        "| Layout | top1 match | KL(bf16 ‖ qfura) | logit rel err |",
        "|---|---|---|---|",
    ]
    for layout, m in model_level.items():
        lines.append(
            f"| `{layout}` | {m['top1_match']:.4f} | {m['kl']:.4f} | {m['logit_rel_err']:.4f} |"
        )
    lines += [
        "",
        "## Per-linear-type error (averaged over all transformer layers)",
        "",
        "| Layer | Layout | fwd mean | fwd p95 | bwd mean | n layers |",
        "|---|---|---|---|---|---|",
    ]
    for leaf in TARGET_NAMES:
        f = per_leaf_flat.get(leaf)
        p = per_leaf_pcb.get(leaf)
        if f:
            lines.append(
                f"| `{leaf}` | flat | {f['fwd_mean']:.4f} | {f['fwd_p95']:.4f} | "
                f"{f['bwd_mean']:.4f} | {f['n_layers']} |"
            )
        if p:
            lines.append(
                f"| `{leaf}` | per_core_block | {p['fwd_mean']:.4f} | "
                f"{p['fwd_p95']:.4f} | {p['bwd_mean']:.4f} | {p['n_layers']} |"
            )
    lines += [
        "",
        "## Default layout recommendation",
        "",
        f"Model-level KL for `flat`: {model_level['flat']['kl']:.4f}; "
        f"for `per_core_block`: {model_level['per_core_block']['kl']:.4f}.",
        "",
        "Lower KL wins. Ties broken in favor of `flat` for simplicity.",
        "",
    ]
    return "\n".join(lines)


def main():
    args = parse_args()
    prompts = load_prompts(args.data_path, args.num_prompts)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    batch = tokenize_batch(tokenizer, prompts, args.max_seq_len)
    batch = {k: v.cuda() for k, v in batch.items()}

    print(f"Loading {args.model} in bf16 on cuda...")
    model_bf16 = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).cuda()
    model_bf16.eval()

    print("Capturing layer inputs...")
    layer_captures = capture_layer_inputs(model_bf16, batch, TARGET_NAMES)
    print(f"Captured inputs for {len(layer_captures)} linear layers")

    per_layer_flat = {}
    per_layer_pcb = {}
    model_level = {}

    for layout, bucket in (("flat", per_layer_flat), ("per_core_block", per_layer_pcb)):
        print(f"\n=== Layout: {layout} ===")
        for name, xins in layer_captures.items():
            mod = dict(model_bf16.named_modules())[name]
            metrics = per_layer_error(mod, xins[:4], layout)  # 4 captured batches
            bucket[name] = metrics
            print(
                f"  {name}: fwd_mean={metrics['fwd_err_mean']:.4f} "
                f"bwd_max={metrics['bwd_err_max']:.4f}"
            )

        print(f"Building full qfura model for {layout}...")
        model_q = copy.deepcopy(model_bf16)
        convert_linear_to_btt(
            model_q,
            btt_rank="full",
            decomp_mode="square",
            include_names=TARGET_NAMES,
            train_position="small",
            s_merged_to="frozen",
        )
        configure_blocktt_trainability(model_q, train_position="small")
        from btt_layer import convert_btt_to_qbtt_
        convert_btt_to_qbtt_(model_q, layout=layout)
        model_q.eval()
        model_level[layout] = model_level_error(model_bf16, model_q, batch)
        print(f"  model-level: {model_level[layout]}")
        del model_q
        torch.cuda.empty_cache()

    per_leaf_flat = aggregate_by_leaf(per_layer_flat)
    per_leaf_pcb = aggregate_by_leaf(per_layer_pcb)

    report = format_report(args, model_level, per_leaf_flat, per_leaf_pcb)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(report)
    print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Dry-run the script on a tiny model locally**

Run: `uv run python -c "from analysis.bench_qbtt_quant_error import parse_args, TARGET_NAMES; print(TARGET_NAMES)"`
Expected: prints the tuple without import errors.

- [ ] **Step 3: Run the actual benchmark on Llama-3-8B**

Run:
```bash
uv run python analysis/bench_qbtt_quant_error.py \
    --model meta-llama/Meta-Llama-3-8B \
    --num-prompts 32 \
    --output docs/reports/qfura-quant-error.md
```
Expected: script completes in ~10-30 minutes on H100, writes `docs/reports/qfura-quant-error.md` with per-layer and model-level tables. If the script crashes, read the stack trace and report the failure with the first 40 lines of output — do not silently "fix" by wrapping in try/except.

- [ ] **Step 4: Read the report and decide on default layout**

Open `docs/reports/qfura-quant-error.md`. Identify the layout with lower model-level KL. That is the default for `--quant_block_layout`.

- [ ] **Step 5: Update the spec's default notation**

Edit `docs/superpowers/specs/2026-04-24-qfura-design.md`. Find the line "Default placeholder `flat` until the benchmark report is committed." In Section 4.6 and Section 4.8, replace the placeholder with the chosen default, e.g. "Default: `flat` (per the benchmark: KL 0.023 flat vs 0.021 per_core_block, tie broken by simplicity)." If `per_core_block` wins by a meaningful margin, use that value instead.

- [ ] **Step 6: Commit**

```bash
git add analysis/bench_qbtt_quant_error.py docs/reports/qfura-quant-error.md docs/superpowers/specs/2026-04-24-qfura-design.md
git commit -m "qfura: quant-error benchmark + report, pick default layout"
```

---

## Task 10: Create `finetune_qfura.py` — data loader, BTT conversion, quantization, optimizer

**Files:**
- Create: `ref/LIFT/src/finetune_qfura.py`

This file is derived from `ref/LIFT/src/finetune_blocktt.py`. The delta is small and mechanical.

- [ ] **Step 1: Create the file**

Create `ref/LIFT/src/finetune_qfura.py` by first copying `ref/LIFT/src/finetune_blocktt.py`, then applying the modifications below. Do this as a fresh write, not a shell `cp` + `sed` — we want the end state explicit.

Write the following exact content. The first ~80 lines (imports and sys.path setup) are identical to `finetune_blocktt.py`; the substantive changes are flagged with `# qfura:` comments:

```python
import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir))
)
_LIFT_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir))
_LIFT_SRC = os.path.join(_LIFT_REPO_ROOT, "src")
if os.path.isdir(_LIFT_SRC) and _LIFT_SRC not in sys.path:
    sys.path.insert(0, _LIFT_SRC)

import copy
import time
import torch
import json
import random
import math
import argparse
from tqdm.auto import tqdm

from torch.utils.data import DataLoader
import torch.nn as nn

# qfura: bitsandbytes for NF4 and paged optimizer.
import bitsandbytes as bnb

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    SchedulerType,
    get_scheduler,
)

from utils.utils import (
    print_rank_0,
    get_all_reduce_mean,
    int_or_float,
)

from accelerate import Accelerator
from accelerate.utils import set_seed

from utils.model_utils import (
    load_hf_tokenizer,
    save_hf_format,
    make_model_gradient_checkpointing_compatible,
)

from utils.data_utils import SupervisedDataset, DataCollatorForSupervisedDataset

from btt_layer import (
    BTTLayer,
    QBTTLayer,                 # qfura
    convert_linear_to_btt,
    configure_blocktt_trainability,
    get_blocktt_target_module_names,
    normalize_trainable_blocktt_cores_,
    resolve_blocktt_decomp_modes,
    convert_btt_to_qbtt_,      # qfura
)

from tools.system_metrics import SysMon

from compress_integration import (
    add_calibrated_btt_args,
    validate_calibrated_btt_args,
    apply_calibrated_btt,
    build_calib_loader,
    save_calibrated_btt_checkpoint,
)


def resolve_blocktt_rank(rank_arg):
    if rank_arg == "full":
        return "full"
    try:
        rank = int(rank_arg)
    except ValueError as exc:
        raise ValueError("--blocktt_rank must be 'full' or a positive integer") from exc
    if rank <= 0:
        raise ValueError("--blocktt_rank must be > 0")
    return rank


def materialize_btt_to_linear(model):
    """Replace BTTLayer / QBTTLayer modules with nn.Linear containing materialized
    dense weights. QBTTLayer.materialize_dense_weight() dequants internally."""
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, BTTLayer):  # QBTTLayer is also BTTLayer
            replacements.append((name, module))

    for name, btt_module in replacements:
        dense_weight = btt_module.materialize_dense_weight()
        linear = nn.Linear(
            btt_module.in_features,
            btt_module.out_features,
            bias=btt_module.bias is not None,
            device=dense_weight.device,
            dtype=dense_weight.dtype,
        )
        linear.weight.data.copy_(dense_weight)
        if btt_module.bias is not None:
            linear.bias.data.copy_(btt_module.bias.data)

        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], linear)

    print(f"Materialized {len(replacements)} BTT/QBTT modules to nn.Linear")
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="qfura: Quantized BlockTT Fine-Tuning")
    parser.add_argument("--data_path", nargs="*",
        default=["./LLM-Adapters/ft-training_set/commonsense_170k.json"])
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)  # qfura: 70B-friendly default
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--val_set_size", type=int, default=100)
    parser.add_argument("--load_last_model", action="store_true")
    parser.add_argument("--eval_step", type=int, default=80)
    parser.add_argument("--eval_delay", type=int_or_float, default=0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)  # qfura: 70B-friendly default
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial",
                 "constant", "constant_with_warmup"])
    parser.add_argument("--num_warmup_steps", type=float, default=0.03)
    parser.add_argument("--mixed_precision", type=str, default="bf16",
        choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--instruction_type", type=str,
        choices=["single", "multi"], default="single")
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--use_flash_attn", type=str, default="False")

    # BlockTT-specific args (inherited).
    parser.add_argument("--trainable_type", type=str, default="all",
        choices=["all", "mlp", "attn"])
    parser.add_argument("--decomp_mode", type=str, default="input_one_block")
    parser.add_argument("--blocktt_rank", type=str, default="full")
    parser.add_argument("--train_position", type=str, default="small",
        choices=["small", "large", "both"])
    parser.add_argument("--s_merged_to", type=str, default="frozen",
        choices=["frozen", "trainable", "output", "input", "split",
                 "keep_frozen", "keep_trainable"])
    parser.add_argument("--blocktt_normalize_after_update", action="store_true")
    parser.add_argument("--blocktt_factorize_by_head", action="store_true", default=True)
    parser.add_argument("--no_blocktt_factorize_by_head", action="store_false",
        dest="blocktt_factorize_by_head")
    parser.add_argument("--no_train_bias", action="store_true")

    # qfura-specific args.
    parser.add_argument("--quant_block_layout", type=str, required=True,
        choices=["flat", "per_core_block"],
        help="NF4 block layout for the frozen BTT core.")

    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")

    add_calibrated_btt_args(parser, hyphen_style=False)

    args = parser.parse_args()
    validate_calibrated_btt_args(args, argv=sys.argv[1:], hyphen_style=False)

    # qfura: hard constraints.
    if args.train_position != "small":
        raise ValueError(
            f"qfura requires --train_position=small; got {args.train_position}"
        )
    if not args.gradient_checkpointing:
        raise ValueError("qfura requires --gradient_checkpointing")
    return args


def main():
    args = parse_args()

    use_wandb = not args.no_wandb
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb" if use_wandb else None,
    )
    if not torch.cuda.is_available() or accelerator.device.type != "cuda":
        raise RuntimeError(
            "finetune_qfura.py requires CUDA. "
            f"Current accelerator device: {accelerator.device}."
        )

    set_seed(args.seed)
    args.global_rank = 1

    if use_wandb:
        if args.wandb_project is None:
            args.wandb_project = "qfura"
        tracker_config = vars(args).copy()
        wandb_init_kwargs = {}
        if args.wandb_run_name:
            wandb_init_kwargs["name"] = args.wandb_run_name
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config=tracker_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    tokenizer.model_max_length = args.max_seq_len

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model_kwargs = {"torch_dtype": torch.bfloat16}
    if args.use_flash_attn == "True":
        model_kwargs["use_flash_attention_2"] = True
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        **model_kwargs,
    )
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))
    model = model.to(accelerator.device)

    if len(args.data_path) == 1 and ".json" in args.data_path[0]:
        train_dataset = SupervisedDataset(
            data_path=args.data_path[0],
            tokenizer=tokenizer,
            instruction_type=args.instruction_type,
            args=args,
        )
        if args.val_set_size > 0:
            train_dataset, eval_dataset = torch.utils.data.random_split(
                train_dataset,
                [len(train_dataset) - args.val_set_size, args.val_set_size],
            )
    else:
        raise ValueError("Only json format is supported for now.")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    if args.val_set_size > 0:
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=data_collator,
        )

    # --- BlockTT conversion ---
    if getattr(args, "calib_mode", "none") != "none":
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
        blocktt_rank = resolve_blocktt_rank(args.blocktt_rank)
        target_modules = get_blocktt_target_module_names(args.trainable_type)
        train_bias = not args.no_train_bias

        decomp_mode, module_decomp_modes = resolve_blocktt_decomp_modes(
            args.decomp_mode,
            include_names=target_modules,
            default_mode="input_one_block",
        )

        converted_modules = convert_linear_to_btt(
            model,
            btt_rank=blocktt_rank,
            decomp_mode=module_decomp_modes if module_decomp_modes is not None else decomp_mode,
            init_mode="default",
            include_names=target_modules,
            skip_names=("lm_head",),
            lr_act=False,
            s_merged_to=args.s_merged_to,
            train_position=args.train_position,
            factorize_by_head=args.blocktt_factorize_by_head,
            model_config=model.config,
        )
        stats = configure_blocktt_trainability(
            model,
            train_bias=train_bias,
            train_position=args.train_position,
            train_singular_values=(args.s_merged_to == "keep_trainable"),
        )
        if stats["num_btt_layers"] == 0:
            raise ValueError("No layers were converted to BTT; check --trainable_type.")
        print(f"Converted modules: {len(converted_modules)}")
        print(
            f"Trainable params: {stats['trainable_param_count']:,} / "
            f"{stats['total_param_count']:,} "
            f"({100 * stats['trainable_param_count'] / stats['total_param_count']:.4f}%)"
        )

    # qfura: quantize the frozen core of every BTT layer.
    qstats = convert_btt_to_qbtt_(model, layout=args.quant_block_layout)
    print(
        f"[qfura] quantized {qstats['num_converted']} BTT layers "
        f"(layout={qstats['layout']}, bytes_saved={qstats['bytes_saved']:,})"
    )
    if use_wandb:
        accelerator.log(
            {
                "qfura/num_converted": qstats["num_converted"],
                "qfura/bytes_saved": qstats["bytes_saved"],
            },
            step=0,
        )

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"param {name} is trainable")

    if args.gradient_checkpointing:
        model = make_model_gradient_checkpointing_compatible(model)
        model.gradient_checkpointing_enable()

    # qfura: paged AdamW 8-bit on trainable params.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = bnb.optim.PagedAdamW8bit(
        trainable_params,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    if args.max_steps > 0:
        max_train_steps = min(max_train_steps, args.max_steps)

    if args.num_warmup_steps < 1:
        args.num_warmup_steps = int(args.num_warmup_steps * max_train_steps)
    else:
        args.num_warmup_steps = int(args.num_warmup_steps)

    print(f"max trainable steps: {max_train_steps}, warmup steps: {args.num_warmup_steps}")
    total_batch_size = (
        args.per_device_train_batch_size * args.gradient_accumulation_steps
    )
    print("***** Running qfura training *****")
    print(f"  Num examples = {len(train_dataloader)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    print(f"  Total train batch size (w. accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {max_train_steps}")

    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    args.completed_steps = 0

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    if args.val_set_size > 0:
        eval_dataloader = accelerator.prepare(eval_dataloader)

    best_model = None

    sysmon = SysMon(
        out_dir=args.output_dir or ".",
        method="qfura",
        rank=(None if args.blocktt_rank == "full" else int(args.blocktt_rank)),
        base_params=sum(p.numel() for p in model.parameters()),
    )
    _base = sysmon.base_params
    for name, p in model.named_parameters():
        if "btt_" in name:
            _base -= p.numel()
    sysmon.base_params = _base

    def train_epoch(epoch):
        nonlocal best_model, best_eval_loss
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                total_loss += loss.detach().float()

            if accelerator.sync_gradients:
                _t0 = time.time()
                optimizer.step()
                if args.blocktt_normalize_after_update:
                    unwrapped = accelerator.unwrap_model(model)
                    normalize_trainable_blocktt_cores_(unwrapped)
                lr_scheduler.step()
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                sysmon.record_step(time.time() - _t0)
                progress_bar.update(1)
                args.completed_steps += 1
                if args.max_steps > 0 and args.completed_steps >= args.max_steps:
                    return

                if args.logging_steps and args.completed_steps % args.logging_steps == 0:
                    divisor = args.gradient_accumulation_steps * args.logging_steps
                    avg_loss = accelerator.gather(total_loss).mean().item() / divisor
                    print(
                        f"  Step: {args.completed_steps}, "
                        f"LR: {lr_scheduler.get_last_lr()[0]:.8f}, Loss: {avg_loss:.6f}"
                    )
                    accelerator.log(
                        {
                            "learning_rate": lr_scheduler.get_last_lr()[0],
                            "train_loss": avg_loss,
                        },
                        step=args.completed_steps,
                    )
                    total_loss = 0

                if (
                    args.completed_steps % args.eval_step == 0
                    and args.val_set_size > 0
                    and not args.load_last_model
                ):
                    perplexity, eval_loss = evaluate(model)
                    accelerator.print(
                        f"Epoch {epoch+1} Step {args.completed_steps}: "
                        f"Eval ppl = {perplexity:.4f}, Eval loss = {eval_loss:.4f}"
                    )
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        if accelerator.is_main_process and args.output_dir:
                            accelerator.wait_for_everyone()
                            unwrapped_model = accelerator.unwrap_model(model)
                            best_model = copy.deepcopy(unwrapped_model).to("cpu")
                            print("New best model")

        return total_loss / len(train_dataloader)

    def evaluate(model):
        model.eval()
        losses = 0
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            losses += loss.float()
        losses = losses / (step + 1)
        try:
            losses = get_all_reduce_mean(losses)
        except Exception:
            pass
        try:
            perplexity = torch.exp(losses).item()
        except OverflowError:
            perplexity = float("inf")
        model.train()
        return perplexity, losses.item()

    best_eval_loss = float("inf")
    for epoch in range(args.num_train_epochs):
        train_loss = train_epoch(epoch)
        if train_loss is not None:
            accelerator.print(f"Epoch {epoch+1}: Average loss = {train_loss:.4f}")
        if args.max_steps > 0 and args.completed_steps >= args.max_steps:
            break

    effective_tokens = (
        args.per_device_train_batch_size
        * args.gradient_accumulation_steps
        * args.max_seq_len
    )
    sysmon.dump(
        model,
        extra={
            "effective_tokens_per_step": effective_tokens,
            "learning_rate": args.learning_rate,
            "train_position": args.train_position,
            "decomp_mode": args.decomp_mode,
            "s_merged_to": args.s_merged_to,
            "quant_block_layout": args.quant_block_layout,
        },
    )

    if args.val_set_size == 0 and accelerator.is_main_process and args.output_dir:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        if getattr(args, "calib_mode", "none") != "none":
            save_calibrated_btt_checkpoint(unwrapped_model, args.output_dir, tokenizer)
        else:
            materialize_btt_to_linear(unwrapped_model)
            save_hf_format(unwrapped_model, tokenizer, args)

    if args.output_dir is not None:
        if args.val_set_size > 0 and not args.load_last_model:
            ppl, val_loss = evaluate(model)
            print_rank_0(
                f"Validation perplexity: {ppl}, Validation loss: {val_loss}",
                args.global_rank,
            )
            if val_loss < best_eval_loss:
                best_eval_loss = val_loss
                if args.global_rank == 0:
                    best_model = copy.deepcopy(model.module).to("cpu")

        model = best_model if best_model is not None else model
        if getattr(args, "calib_mode", "none") != "none":
            save_calibrated_btt_checkpoint(model, args.output_dir, tokenizer)
        else:
            materialize_btt_to_linear(model)
            save_hf_format(model, tokenizer, args)

    if use_wandb:
        accelerator.end_training()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Byte-compile check**

Run: `uv run python -m py_compile ref/LIFT/src/finetune_qfura.py`
Expected: exit code 0.

- [ ] **Step 3: Commit**

```bash
git add ref/LIFT/src/finetune_qfura.py
git commit -m "qfura: training entrypoint with PagedAdamW8bit + NF4 conversion"
```

---

## Task 11: CLI validation test

**Files:**
- Test: `tests/test_finetune_qfura_cli.py` (new)

- [ ] **Step 1: Write the test**

Create `tests/test_finetune_qfura_cli.py`:

```python
import subprocess
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "ref" / "LIFT" / "src" / "finetune_qfura.py"


def _run(args_tail):
    """Run the script in --help-style argparse validation mode."""
    cmd = [sys.executable, str(_SCRIPT)] + args_tail
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=_REPO_ROOT)
    return result


class TestFinetuneQfuraCLI(unittest.TestCase):
    def test_rejects_train_position_not_small(self):
        result = _run([
            "--model_name_or_path", "bogus",
            "--quant_block_layout", "flat",
            "--train_position", "large",
            "--gradient_checkpointing",
        ])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("qfura requires --train_position=small", result.stderr + result.stdout)

    def test_rejects_missing_gradient_checkpointing(self):
        result = _run([
            "--model_name_or_path", "bogus",
            "--quant_block_layout", "flat",
            # gradient_checkpointing flag omitted
        ])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("qfura requires --gradient_checkpointing", result.stderr + result.stdout)

    def test_rejects_unknown_quant_block_layout(self):
        result = _run([
            "--model_name_or_path", "bogus",
            "--quant_block_layout", "bogus_layout",
            "--gradient_checkpointing",
        ])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("quant_block_layout", result.stderr + result.stdout)

    def test_rejects_missing_quant_block_layout(self):
        result = _run([
            "--model_name_or_path", "bogus",
            "--gradient_checkpointing",
        ])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("quant_block_layout", result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test**

Run: `uv run python -m unittest tests.test_finetune_qfura_cli -v`
Expected: all four tests pass. Each runs the script with bad args and verifies it exits non-zero with a message referencing the misconfiguration. The tests do not require CUDA or a real model — the `argparse` and early validation reject before model loading.

- [ ] **Step 3: Commit**

```bash
git add tests/test_finetune_qfura_cli.py
git commit -m "qfura: CLI validation tests"
```

---

## Task 12: Shell runner for commonsense qfura

**Files:**
- Create: `ref/LIFT/bash_scripts/finetune_commonsense_qfura.sh`

- [ ] **Step 1: Create the shell script**

Create `ref/LIFT/bash_scripts/finetune_commonsense_qfura.sh`:

```bash
#!/bin/bash

pwd
hostname
date
echo starting job...
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export LIBRARY_PATH="/usr/local/cuda/lib64:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export HF_HOME="${HF_HOME:-/data/yequan/huggingface}"

SRC_DIR=/home/yequan/Project/lora/lora-without-regret/ref/LIFT
DATA_DIR=LLM-Adapters
OUTPUT_SRC_DIR=/data/yequan/fura/lift

MODEL="${MODEL:-meta-llama/Meta-Llama-3-70B}"
decomp_mode="${decomp_mode:-input_one_block}"
quant_block_layout="${quant_block_layout:-flat}"
s_merged_to="${s_merged_to:-frozen}"
blocktt_rank="${blocktt_rank:-full}"
trainable_type="${trainable_type:-all}"
lr="${lr:-2e-4}"
seed="${seed:-43}"
MAX_STEPS="${MAX_STEPS:-0}"
per_device_train_batch_size="${per_device_train_batch_size:-1}"
gradient_accumulation_steps="${gradient_accumulation_steps:-16}"
model_tag="${MODEL##*/}"

calib_mode="${calib_mode:-none}"
calib_source="${calib_source:-training_data}"
calib_num_seqs="${calib_num_seqs:-128}"
calib_batch_size="${calib_batch_size:-4}"

wandb_project="${wandb_project:-qfura-${model_tag}}"
wandb_run_id="${wandb_run_id:-$(python -c 'import wandb; print(wandb.util.generate_id())')}"
export WANDB_RUN_ID="${wandb_run_id}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"

echo $MODEL

OUTPUT="${OUTPUT:-${OUTPUT_SRC_DIR}/commonsense/${MODEL}/qfura-layout_${quant_block_layout}-lr_${lr}-decomp_${decomp_mode}-seed_${seed}}"
run_name="${run_name:-$(basename "$OUTPUT")}"

mkdir -p $OUTPUT

cd ${SRC_DIR}

accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    --mixed_precision="bf16" \
    src/finetune_qfura.py \
    --model_name_or_path ${MODEL} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --max_seq_len 2048 \
    --learning_rate ${lr} \
    --weight_decay 0. \
    --num_train_epochs 3 \
    --mixed_precision bf16 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --lr_scheduler_type linear \
    --num_warmup_steps 0.03 \
    --seed ${seed} \
    --gradient_checkpointing \
    --instruction_type single \
    --decomp_mode ${decomp_mode} \
    --train_position small \
    --blocktt_rank ${blocktt_rank} \
    --s_merged_to ${s_merged_to} \
    --trainable_type ${trainable_type} \
    --quant_block_layout ${quant_block_layout} \
    --calib_mode ${calib_mode} \
    --calib_source ${calib_source} \
    --calib_num_seqs ${calib_num_seqs} \
    --calib_batch_size ${calib_batch_size} \
    --save_interval 100000 \
    --val_set_size 120 \
    --eval_step 400 \
    --data_path ${DATA_DIR}/ft-training_set/commonsense_170k.json \
    --wandb_project "${wandb_project}" \
    --wandb_run_name "${run_name}" \
    --max_steps ${MAX_STEPS} \
    --output_dir $OUTPUT 2> >(tee $OUTPUT/err.log >&2) | tee $OUTPUT/training.log

if [ "${MAX_STEPS}" = "0" ]; then
    bash ./bash_scripts/eval_commonsense.sh \
        CKPT="$OUTPUT" \
        base_model="${MODEL}" \
        wandb_project="${wandb_project}" \
        wandb_run_name="${run_name}" \
        wandb_run_id="${wandb_run_id}"
fi
```

- [ ] **Step 2: Make it executable**

Run: `chmod +x ref/LIFT/bash_scripts/finetune_commonsense_qfura.sh`

- [ ] **Step 3: Smoke-test on Llama-3-8B for 50 steps**

Run:
```bash
MODEL=meta-llama/Meta-Llama-3-8B MAX_STEPS=50 \
    bash ref/LIFT/bash_scripts/finetune_commonsense_qfura.sh
```
Expected: script runs for 50 optimizer steps, loss decreases by the end, `training.log` shows normal step output, and peak GPU memory stays under 80 GB. If it OOMs or hits `ImportError`, report the failure — do not silently reduce batch size without an explanation of why the expected budget was wrong.

- [ ] **Step 4: Commit**

```bash
git add ref/LIFT/bash_scripts/finetune_commonsense_qfura.sh
git commit -m "qfura: shell runner for commonsense-reasoning SFT"
```

---

## Task 13: Run the headline Llama-3-70B experiments and write results doc

**Files:**
- Create: `docs/exp_results/qfura.md`

This is the paper-facing artifact. Runs take ~20-24h on H100 per layout.

- [ ] **Step 1: Launch the `flat` layout run**

Run:
```bash
MODEL=meta-llama/Meta-Llama-3-70B \
quant_block_layout=flat \
    bash ref/LIFT/bash_scripts/finetune_commonsense_qfura.sh
```
Expected: 3 epochs × ~10k-step epochs on commonsense_170k; final eval from `eval_commonsense.sh` produces 8 accuracy numbers (BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, OBQA).

- [ ] **Step 2: Launch the `per_core_block` layout run**

Run:
```bash
MODEL=meta-llama/Meta-Llama-3-70B \
quant_block_layout=per_core_block \
    bash ref/LIFT/bash_scripts/finetune_commonsense_qfura.sh
```
Expected: same shape of output.

- [ ] **Step 3: Collect results and write the report**

Create `docs/exp_results/qfura.md`. Populate from the two run directories:

```markdown
# qfura: Llama-3-70B Commonsense Results

**Model:** meta-llama/Meta-Llama-3-70B
**Dataset:** LIFT commonsense_170k (3 epochs)
**Hardware:** 1× H100 94 GB
**Training script:** `ref/LIFT/src/finetune_qfura.py`

## Configuration

| Setting | Value |
|---|---|
| Base dtype | bf16 |
| Frozen core | NF4 (double-quant, bf16 compute) |
| Trainable | small BTT core + btt_s + biases |
| Optimizer | PagedAdamW8bit |
| Learning rate | 2e-4 |
| Per-device batch | 1 |
| Grad accumulation | 16 |
| Seed | 43 |

## Commonsense-reasoning accuracy

| Layout | BoolQ | PIQA | SIQA | HellaSwag | WinoGrande | ARC-E | ARC-C | OBQA | Avg |
|---|---|---|---|---|---|---|---|---|---|
| flat            | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| per_core_block  | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

(Replace TBD with the numbers from each run's `eval_commonsense.sh` log.)

## System metrics

| Layout | Wall-clock (h) | Peak GPU mem (GB) | Tokens/sec | W&B run ID |
|---|---|---|---|---|
| flat | TBD | TBD | TBD | TBD |
| per_core_block | TBD | TBD | TBD | TBD |

## Notes

- Checkpoints saved via dequant-then-materialize; eval runs on the materialized bf16 HF checkpoint (round-trip caveat documented in the design spec).
- Quantization-error benchmark comparing these two layouts at model load time: `docs/reports/qfura-quant-error.md`.
```

Fill in every `TBD` from the actual run logs. Do not commit the file with TBDs in place.

- [ ] **Step 4: Commit**

```bash
git add docs/exp_results/qfura.md
git commit -m "qfura: Llama-3-70B commonsense-reasoning results"
```

---

## Task 14: Final verification — run full unit-test suite

- [ ] **Step 1: Run all qfura tests**

Run: `uv run python -m unittest tests.test_qbtt_conversion tests.test_qbtt_forward_shape tests.test_qbtt_gradient_flow tests.test_qbtt_quant_error tests.test_qbtt_fused_step2_compat tests.test_finetune_qfura_cli -v`
Expected: all tests pass (or skip cleanly if the fused kernel is unavailable).

- [ ] **Step 2: Confirm existing tests still pass**

Run: `uv run python -m unittest tests.test_btt_pipeline_compat tests.test_btt_layer_fused_flag tests.test_btt_linear_materialize -v`
Expected: all existing BTT tests still pass — no regression from adding `QBTTLayer` to `btt_layer.py`.

- [ ] **Step 3: If a regression exists, stop**

If a pre-existing test fails on a file that `btt_layer.py` changes, the refactor introduced a bug. Report the failing test with the first 20 lines of stack trace; do not mutate test files to make failures disappear.

No commit for this task (verification only).
