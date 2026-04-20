# LoRA Variants (DoRA, PiSSA, MiLoRA, RandLoRA, LIFT) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add five new `--train-mode` values (`dora`, `pissa`, `milora`, `randlora`, `lift`) to `run_rl.py` and `run_sft.py`; add `milora` and `randlora` adapter branches plus four bash scripts to `ref/LIFT/src/finetune_lora.py`; vendor `SparseAdamW` from LIFT into `optim/sparse_adam.py`.

**Architecture:** Two independent surfaces. Main repo (`run_rl.py`/`run_sft.py`) gets new train modes that route through the existing `{lora, lora_full}` checkpoint and math-verify branches (since `merge_adapter()` materializes a vanilla HF model for all four PEFT-based variants). LIFT script gets two new `adapter_name` branches with a locally-duplicated MiLoRA SVD init helper. PiSSA and MiLoRA are forced to local in-process vLLM rollout because they overwrite the base weight at init. LIFT is dense + custom `SparseAdamW`.

**Tech Stack:** Python 3.13, PyTorch, PEFT 0.17.1 (ships DoRA / PiSSA / RandLoRA), HuggingFace Transformers, vLLM, `unittest`. Spec: `docs/superpowers/specs/2026-04-19-lora-variants-design.md`.

---

## Background reference (read before starting)

Key files and current line numbers (snapshot 2026-04-19):

- `run_rl.py` (1967 lines)
  - L58: `MODE_DEFAULTS = {...}`
  - L96: `--train-mode choices=[...]`
  - L431: `apply_mode_defaults(args)`
  - L466: `_flag_was_passed`
  - L470: `validate_mode_specific_flags(args, argv)`
  - L559: `--enable-math-verify` warning
  - L739: `compute_run_name`
  - L799: `save_merged_checkpoint(model, tokenizer, ckpt_dir, train_mode, args)` ← extend
  - L913: `resolve_lora_rollout_backend(train_mode, vllm_url)` ← extend
  - L921: `normalize_lora_merged_weight_name(name)` ← extend skip-list
  - L935: `build_lora_http_generators(args, model, run_dir)`
  - L994: `build_lora_local_generators(args, model)`
  - L1060: `build_local_vllm_generators(args, model)` (used by `full`)
  - L1185: `build_optimizer(args, trainable_params, trainable_named_params)` ← extend with LIFT branch
  - L1320–1410: model construction + mode_info population in `main()` ← extend with five new branches
  - L1391: PEFT model wrapping for `{lora, lora_full}` ← extend
  - L1597: `build_local_vllm_generators` call for `full` ← extend to include `lift`
  - L1857: final merged-ckpt save block
  - L1867: post-training math-verify eval (in-memory hot-swap branches at L1874/L1876/L1894)

- `run_sft.py` (1144 lines)
  - L52: `MODE_DEFAULTS = {...}`
  - L82: `parse_args`
  - L295: `apply_mode_defaults`
  - L526: `prepare_model(args, ...)` ← extend with five new branches
  - L587: `elif args.train_mode == "lora":` block (model where new branches are inserted)
  - L766: `compute_run_name`
  - L801: `build_optimizer`

- `ref/LIFT/src/finetune_lora.py` (687 lines)
  - L48–55: PEFT imports
  - L292–340 area: `argparse` (`--lora_r`, `--lora_alpha`, `--target_modules`)
  - L390: `if args.adapter_name in ["lora", "dora", "pissa"]:` ← extend list, add branches
  - L425–431: dead `hira` branch ← delete
  - L651: same `if args.adapter_name in [...]:` cascade ← extend list

- `tests/`
  - `test_run_rl_cli.py` — patterns for argparse-validation tests
  - `test_run_rl_merged_ckpt.py` — patterns for `save_merged_checkpoint` tests using `MagicMock`
  - `test_run_sft_cli_calib.py` — pattern for `run_sft.py` CLI tests

PEFT 0.17.1 supports: `LoraConfig(use_dora=True)`, `LoraConfig(init_lora_weights="pissa_niter_4")`, `RandLoraConfig`. **Not** in PEFT: `init_lora_weights="milora"` (we add a custom post-init pass).

`SparseAdamW` source: `ref/LIFT/src/sparseAdam.py` (430 lines). It is self-contained PyTorch (no LIFT-specific imports). We copy it; no edits.

---

## File Structure

**Create:**
- `optim/sparse_adam.py` — vendored `SparseAdamW` (~430 lines) with provenance header.
- `tests/test_sparse_adam_smoke.py` — 1-step smoke test.
- `tests/test_milora_init.py` — reconstruction-error test for `apply_milora_init_`.
- `tests/test_lora_variants_cli.py` — argparse-validation tests for the five new modes (run_rl + run_sft).
- `tests/test_lora_variants_merged_ckpt.py` — `save_merged_checkpoint` routing tests for the five new modes.
- `ref/LIFT/bash_scripts/finetune_math_milora.sh`
- `ref/LIFT/bash_scripts/finetune_math_randlora.sh`
- `ref/LIFT/bash_scripts/finetune_commonsense_milora.sh`
- `ref/LIFT/bash_scripts/finetune_commonsense_randlora.sh`

**Modify:**
- `run_rl.py` — `MODE_DEFAULTS`, `--train-mode choices`, three new CLI flags, `apply_mode_defaults`, `validate_mode_specific_flags`, `compute_run_name`, `save_merged_checkpoint`, `resolve_lora_rollout_backend`, `normalize_lora_merged_weight_name`, `build_optimizer`, model-construction block, math-verify block, `--enable-math-verify` warning.
- `run_sft.py` — same axes (no rollout/math-verify); `prepare_model` is the central dispatch.
- `ref/LIFT/src/finetune_lora.py` — extend `adapter_name` cascade list, add `milora`+`randlora` branches, add `apply_milora_init_` local helper, add `--randlora_projection_prng_key` arg, delete dead `hira` branch.

**Test invocation:** `python -m unittest tests/test_<name>.py -v`. Per CLAUDE.md, `python -m py_compile *.py optim/*.py` is the lightweight syntax check.

---

## Task 1: Vendor `SparseAdamW`

**Files:**
- Create: `optim/sparse_adam.py`
- Test: `tests/test_sparse_adam_smoke.py`

- [ ] **Step 1: Capture LIFT submodule SHA for provenance**

```bash
cd /home/yequan/Project/lora/lora-without-regret
git -C ref/LIFT rev-parse HEAD
```

Record the SHA (call it `<LIFT_SHA>`); used in the file header below. If `ref/LIFT` isn't a git repo (gitlink was removed per commit `1c3139a`), use `unknown (gitlink removed)` as the SHA placeholder.

- [ ] **Step 2: Copy `SparseAdamW` into `optim/sparse_adam.py`**

```bash
cp ref/LIFT/src/sparseAdam.py optim/sparse_adam.py
```

- [ ] **Step 3: Prepend provenance header**

Open `optim/sparse_adam.py` and insert at the very top (before existing imports):

```python
"""SparseAdamW optimizer for LIFT-style sparse fine-tuning.

Vendored from ref/LIFT/src/sparseAdam.py @ <LIFT_SHA>
(replace <LIFT_SHA> with the value captured in Task 1 Step 1).
Do not edit; re-vendor from upstream LIFT to update.
"""

```

- [ ] **Step 4: Verify import works**

Run: `python -c "from optim.sparse_adam import SparseAdamW; print(SparseAdamW)"`
Expected: prints `<class 'optim.sparse_adam.SparseAdamW'>` with no errors.
If it fails on a missing import (LIFT used different package layout), inspect the imports at the top of `sparseAdam.py` and confirm they're stdlib/torch only — they should be.

- [ ] **Step 5: Write smoke test**

Create `tests/test_sparse_adam_smoke.py`:

```python
import os
import sys
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from optim.sparse_adam import SparseAdamW


class TestSparseAdamWSmoke(unittest.TestCase):
    def test_one_step_runs_and_updates_only_masked_entries(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 4))
        weights_with_mask = [m.weight for m in model if isinstance(m, nn.Linear)]
        decay_ids = {id(p) for p in weights_with_mask}
        other = [p for p in model.parameters() if id(p) not in decay_ids]

        param_groups = [
            {
                "params": weights_with_mask,
                "weight_decay": 0.0,
                "rank": 4,
                "filter_rank": 4,
                "update_proj_gap": 1,
                "group_name": "weights_with_mask",
            },
            {"params": other, "weight_decay": 0.0, "group_name": "other_params_w_decay"},
        ]
        opt = SparseAdamW(param_groups, lr=1e-3, betas=(0.9, 0.95))

        before = [p.detach().clone() for p in weights_with_mask]
        x = torch.randn(2, 16)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        # At least one masked entry per layer should have changed.
        for w_before, w_after in zip(before, weights_with_mask):
            diff = (w_before - w_after).abs()
            self.assertGreater(diff.sum().item(), 0.0,
                               "SparseAdamW step did not update any entries")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 6: Run the smoke test**

Run: `python -m unittest tests/test_sparse_adam_smoke.py -v`
Expected: 1 test passes. If it fails because `SparseAdamW`'s param-group key names differ from what we used, read the constructor in `optim/sparse_adam.py` to find the actual names and fix the test (do NOT edit the optimizer).

- [ ] **Step 7: Commit**

```bash
git add optim/sparse_adam.py tests/test_sparse_adam_smoke.py
git commit -m "$(cat <<'EOF'
optim: vendor SparseAdamW from ref/LIFT for --train-mode lift

Self-contained copy with provenance header. Smoke test covers a
single-step forward+backward+step on a 2-layer MLP. The optimizer
itself is unchanged from the LIFT source.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Define `apply_milora_init_` helper

**Files:**
- Modify: `run_sft.py` (insert helper near `prepare_model`, before line 526)
- Test: `tests/test_milora_init.py`

We define it once in `run_sft.py` (the existing shared-utilities home per CLAUDE.md). `run_rl.py` will import it from `run_sft` in Task 4. The LIFT-side duplicate lands in Task 9.

- [ ] **Step 1: Write the failing test**

Create `tests/test_milora_init.py`:

```python
import os
import sys
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


class TestMiloraInit(unittest.TestCase):
    """Verify (alpha/r) * B @ A + W_residual reconstructs W exactly."""

    def _make_peft_linear(self, in_features=64, out_features=48, rank=8, alpha=16):
        """Build a single nn.Linear wrapped in a PEFT LoRA layer."""
        from peft import LoraConfig, get_peft_model
        torch.manual_seed(0)
        base = nn.Sequential(nn.Linear(in_features, out_features, bias=False))
        config = LoraConfig(
            r=rank, lora_alpha=alpha,
            target_modules=["0"],   # the nn.Linear inside the Sequential
            lora_dropout=0,
            bias="none",
        )
        peft_model = get_peft_model(base, config)
        return peft_model

    def test_reconstruction_matches_original_weight(self):
        from run_sft import apply_milora_init_
        rank, alpha = 8, 16
        peft_model = self._make_peft_linear(rank=rank, alpha=alpha)

        # Capture original weights before MiLoRA mutates them.
        from peft.tuners.lora import LoraLayer
        lora_layer = next(m for m in peft_model.modules() if isinstance(m, LoraLayer))
        W_original = lora_layer.get_base_layer().weight.data.detach().clone().float()

        apply_milora_init_(peft_model, rank=rank)

        W_residual = lora_layer.get_base_layer().weight.data.float()
        adapter_name = list(lora_layer.lora_A.keys())[0]
        A = lora_layer.lora_A[adapter_name].weight.data.float()  # (r, in)
        B = lora_layer.lora_B[adapter_name].weight.data.float()  # (out, r)

        reconstructed = (alpha / rank) * (B @ A) + W_residual
        rel_err = (reconstructed - W_original).norm() / W_original.norm()
        self.assertLess(rel_err.item(), 1e-4,
                        f"Reconstruction error {rel_err.item():.2e} exceeds 1e-4")

    def test_residual_drops_bottom_r_components(self):
        """MiLoRA replaces W with the top-(n-r) components — its smallest singular
        value should be strictly larger than W's smallest."""
        from run_sft import apply_milora_init_
        peft_model = self._make_peft_linear(rank=4, alpha=8)
        from peft.tuners.lora import LoraLayer
        lora_layer = next(m for m in peft_model.modules() if isinstance(m, LoraLayer))
        W_original = lora_layer.get_base_layer().weight.data.detach().clone().float()
        s_orig = torch.linalg.svdvals(W_original)

        apply_milora_init_(peft_model, rank=4)
        W_residual = lora_layer.get_base_layer().weight.data.float()
        s_resid = torch.linalg.svdvals(W_residual)

        # W_residual has rank n-4; the (n-4)th singular value of residual is
        # the (n-4)th of original; its smallest non-zero is the 5th-smallest
        # of original (i.e. larger than s_orig[-1]).
        self.assertGreater(s_resid[-5].item(), s_orig[-1].item() * 0.99)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_milora_init.py -v`
Expected: FAIL with `ImportError: cannot import name 'apply_milora_init_' from 'run_sft'`.

- [ ] **Step 3: Add `apply_milora_init_` to `run_sft.py`**

In `run_sft.py`, insert this helper immediately before `def prepare_model(...)` (currently L526):

```python
@torch.no_grad()
def apply_milora_init_(peft_model, *, rank: int) -> None:
    """In-place MiLoRA initialization.

    For each LoRA-targeted nn.Linear, performs SVD on the base weight,
    uses the BOTTOM-r singular components to populate lora_A and lora_B,
    and replaces the base weight with the top-(n-r) residual.
    """
    from peft.tuners.lora import LoraLayer
    first_check = True
    for _name, module in peft_model.named_modules():
        if not isinstance(module, LoraLayer):
            continue
        base = module.get_base_layer()
        W = base.weight.data
        dtype, device = W.dtype, W.device
        U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
        r = rank
        U_r, S_r, Vh_r = U[:, -r:], S[-r:], Vh[-r:, :]
        sqrt_S = S_r.sqrt()
        adapter_name = list(module.lora_A.keys())[0]
        alpha = module.lora_alpha[adapter_name]
        scale_correction = (r / alpha) ** 0.5
        lora_A = (sqrt_S.unsqueeze(1) * Vh_r) * scale_correction      # (r, in)
        lora_B = (U_r * sqrt_S.unsqueeze(0)) * scale_correction       # (out, r)
        residual = W.float() - U_r @ torch.diag(S_r) @ Vh_r
        module.lora_A[adapter_name].weight.data.copy_(lora_A.to(dtype=dtype, device=device))
        module.lora_B[adapter_name].weight.data.copy_(lora_B.to(dtype=dtype, device=device))
        base.weight.data.copy_(residual.to(dtype=dtype, device=device))
        if first_check:
            reconstructed = (alpha / r) * lora_B.float() @ lora_A.float() + residual
            rel_err = (
                torch.linalg.norm(reconstructed - W.float())
                / torch.linalg.norm(W.float())
            )
            assert rel_err < 1e-3, (
                f"MiLoRA init reconstruction error too high: {rel_err:.2e}"
            )
            first_check = False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_milora_init.py -v`
Expected: 2 tests PASS. If `test_residual_drops_bottom_r_components` fails on the slack constant `0.99`, inspect the singular value ordering and adjust — but the inequality `s_resid[-5] > s_orig[-1]` should hold by construction.

- [ ] **Step 5: Commit**

```bash
git add run_sft.py tests/test_milora_init.py
git commit -m "$(cat <<'EOF'
sft: add apply_milora_init_ helper for MiLoRA initialization

In-place SVD pass that takes the bottom-r singular components of each
LoRA-targeted base weight as the trainable adapter, leaving the
top-(n-r) residual as the new base. Built-in self-check on the first
visited layer asserts (alpha/r)*B@A + W_residual == W within 1e-3.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Extend `MODE_DEFAULTS`, `--train-mode choices`, and CLI flags in `run_sft.py`

**Files:**
- Modify: `run_sft.py:52-78` (MODE_DEFAULTS), `run_sft.py:82+` (parse_args), `run_sft.py:295+` (apply_mode_defaults)

- [ ] **Step 1: Extend `MODE_DEFAULTS` (run_sft.py around L52)**

Open `run_sft.py`. The existing dict has entries for `full`, `lora`, `blocktt`, `svd`. Add the five new entries with the *same key names already used by existing entries* (`lr`, `wandb_project`, `micro_batch_size`, `gradient_accumulation_steps`). After the existing entries, insert:

```python
    "dora": {
        "lr": 9e-5,
        "wandb_project": "math-sft-dora",
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 128,
    },
    "pissa": {
        "lr": 9e-5,
        "wandb_project": "math-sft-pissa",
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 128,
    },
    "milora": {
        "lr": 9e-5,
        "wandb_project": "math-sft-milora",
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 128,
    },
    "randlora": {
        "lr": 9e-5,
        "wandb_project": "math-sft-randlora",
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 128,
    },
    "lift": {
        "lr": 1e-4,
        "wandb_project": "math-sft-lift",
        "micro_batch_size": 4,
        "gradient_accumulation_steps": 64,
    },
```

(The spec uses `math-grpo-*` for `run_rl.py` and `math-sft-*` here because we're in the SFT entrypoint. Keep consistent with existing `wandb_project` values in this file.)

If the existing `lora` entry's project name uses a different prefix, mirror that prefix instead. Inspect the file before editing.

- [ ] **Step 2: Extend `--train-mode` choices (run_sft.py argparse)**

Find the `--train-mode` `add_argument` (search for `"--train-mode"` in `parse_args`). Replace its `choices=[...]` with:

```python
        choices=["full", "lora", "blocktt", "svd",
                 "dora", "pissa", "milora", "randlora", "lift"],
```

(Note: `run_sft.py` does NOT have `lora_full`; that's only in `run_rl.py`. Do not add it here.)

Update the `help=` string accordingly.

- [ ] **Step 3: Add the new CLI flags**

In `run_sft.py`'s `parse_args`, find the existing `--lora-rank` argument. After the LoRA-family flags block, insert:

```python
    parser.add_argument(
        "--randlora-projection-prng-key",
        type=int,
        default=0,
        help="Seed for RandLoRA's shared random bases (default: 0). "
             "Only valid when --train-mode randlora.",
    )
    parser.add_argument(
        "--lift-lora-rank",
        type=int,
        default=128,
        help="LIFT: rank used for the low-rank approximation that drives "
             "mask selection (default: 128). Only valid when --train-mode lift.",
    )
    parser.add_argument(
        "--lift-filter-rank",
        type=int,
        default=128,
        help="LIFT: filter rank for mask selection (default: 128). "
             "Only valid when --train-mode lift.",
    )
    parser.add_argument(
        "--lift-update-interval",
        type=int,
        default=400,
        help="LIFT: optimizer steps between mask recomputations (default: 400). "
             "Only valid when --train-mode lift.",
    )
```

- [ ] **Step 4: Verify syntax**

Run: `python -m py_compile run_sft.py`
Expected: no output (success).

- [ ] **Step 5: Verify the flag plumbing parses**

Run:
```bash
uv run run_sft.py --train-mode dora --no-wandb --help 2>&1 | grep -E "randlora|lift-lora|lift-filter|lift-update|train-mode"
```
Expected: lines for `--train-mode` (showing all 9 choices), `--randlora-projection-prng-key`, `--lift-lora-rank`, `--lift-filter-rank`, `--lift-update-interval` appear.

- [ ] **Step 6: Commit**

```bash
git add run_sft.py
git commit -m "$(cat <<'EOF'
sft: register dora/pissa/milora/randlora/lift train modes and CLI flags

Extends MODE_DEFAULTS, --train-mode choices, and adds
--randlora-projection-prng-key / --lift-{lora,filter}-rank /
--lift-update-interval. No behavior change yet; prepare_model and
build_optimizer dispatch comes in subsequent commits.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Wire the five new modes through `prepare_model` and `compute_run_name` in `run_sft.py`

**Files:**
- Modify: `run_sft.py:526` (`prepare_model`), `run_sft.py:766` (`compute_run_name`)

- [ ] **Step 1: Insert new branches in `prepare_model`**

Open `run_sft.py` and find `elif args.train_mode == "lora":` (around L587). Immediately after the closing of that block (i.e. before the next `elif args.train_mode == "blocktt":`), insert:

```python
    elif args.train_mode == "dora":
        target_modules = get_lora_target_modules(args.trainable_type)
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=32,
            target_modules=target_modules,
            use_dora=True,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        mode_info["wandb_extra"] = {
            "lora_rank": args.lora_rank, "lora_alpha": 32,
            "trainable_type": args.trainable_type, "target_modules": target_modules,
            "use_dora": True,
        }
        mode_info["print_lines"] = [
            f"  DoRA rank: {args.lora_rank}",
            f"  Trainable type: {args.trainable_type}",
            f"  Target modules: {target_modules}",
        ]

    elif args.train_mode == "pissa":
        target_modules = get_lora_target_modules(args.trainable_type)
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=32,
            target_modules=target_modules,
            init_lora_weights="pissa_niter_4",
            lora_dropout=0,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        mode_info["wandb_extra"] = {
            "lora_rank": args.lora_rank, "lora_alpha": 32,
            "trainable_type": args.trainable_type, "target_modules": target_modules,
            "init_lora_weights": "pissa_niter_4",
        }
        mode_info["print_lines"] = [
            f"  PiSSA rank: {args.lora_rank}",
            f"  Trainable type: {args.trainable_type}",
            f"  Target modules: {target_modules}",
        ]

    elif args.train_mode == "milora":
        target_modules = get_lora_target_modules(args.trainable_type)
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=32,
            target_modules=target_modules,
            lora_dropout=0,
        )
        model = get_peft_model(model, peft_config)
        apply_milora_init_(model, rank=args.lora_rank)
        model.print_trainable_parameters()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        mode_info["wandb_extra"] = {
            "lora_rank": args.lora_rank, "lora_alpha": 32,
            "trainable_type": args.trainable_type, "target_modules": target_modules,
            "init_lora_weights": "milora",
        }
        mode_info["print_lines"] = [
            f"  MiLoRA rank: {args.lora_rank}",
            f"  Trainable type: {args.trainable_type}",
            f"  Target modules: {target_modules}",
        ]

    elif args.train_mode == "randlora":
        from peft import RandLoraConfig
        target_modules = get_lora_target_modules(args.trainable_type)
        peft_config = RandLoraConfig(
            r=args.lora_rank,
            randlora_alpha=32,
            target_modules=target_modules,
            projection_prng_key=args.randlora_projection_prng_key,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        mode_info["wandb_extra"] = {
            "randlora_rank": args.lora_rank, "randlora_alpha": 32,
            "trainable_type": args.trainable_type, "target_modules": target_modules,
            "projection_prng_key": args.randlora_projection_prng_key,
        }
        mode_info["print_lines"] = [
            f"  RandLoRA rank: {args.lora_rank}",
            f"  RandLoRA prng key: {args.randlora_projection_prng_key}",
            f"  Trainable type: {args.trainable_type}",
            f"  Target modules: {target_modules}",
        ]

    elif args.train_mode == "lift":
        # Dense model, no PEFT wrapping. All params trainable.
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        trainable_count = sum(p.numel() for p in trainable_params)
        total_count = sum(p.numel() for p in model.parameters())
        print(
            f"Trainable params: {trainable_count:,} || All params: {total_count:,} || "
            f"Trainable%: {100 * trainable_count / total_count:.2f}"
        )
        mode_info["wandb_extra"] = {
            "lift_lora_rank": args.lift_lora_rank,
            "lift_filter_rank": args.lift_filter_rank,
            "lift_update_interval": args.lift_update_interval,
        }
        mode_info["print_lines"] = [
            f"  LIFT lora_rank: {args.lift_lora_rank}",
            f"  LIFT filter_rank: {args.lift_filter_rank}",
            f"  LIFT update_interval: {args.lift_update_interval}",
        ]
```

Verify: the existing `lora` branch at L587 sets `trainable_named_params` too — check the actual code and mirror whatever assignments it makes. For DoRA/PiSSA/MiLoRA/RandLoRA which all use PEFT, the assignment pattern matches `lora` exactly. For `lift`, mirror the `full` branch (L576-585).

- [ ] **Step 2: Extend `compute_run_name` (run_sft.py L766)**

Replace the existing function body's `if/return` chain so it becomes:

```python
def compute_run_name(args, mode_info: dict) -> str:
    """Compute a human-readable run name (used for both directory and W&B)."""
    if args.wandb_run_name is not None:
        return args.wandb_run_name
    if args.train_mode == "full":
        return f"{args.model_id}_{args.lr:.1e}_full"
    if args.train_mode == "lora":
        return f"{args.model_id}_{args.lr:.1e}_r{args.lora_rank}"
    if args.train_mode == "dora":
        return f"{args.model_id}_{args.lr:.1e}_r{args.lora_rank}_dora"
    if args.train_mode == "pissa":
        return f"{args.model_id}_{args.lr:.1e}_r{args.lora_rank}_pissa"
    if args.train_mode == "milora":
        return f"{args.model_id}_{args.lr:.1e}_r{args.lora_rank}_milora"
    if args.train_mode == "randlora":
        return f"{args.model_id}_{args.lr:.1e}_r{args.lora_rank}_randlora"
    if args.train_mode == "lift":
        return (
            f"{args.model_id}_{args.lr:.1e}_r{args.lift_lora_rank}"
            f"_int{args.lift_update_interval}_lift"
        )
    if args.train_mode == "blocktt":
        decomp_mode_name = mode_info.get("decomp_mode_display", args.decomp_mode)
        return f"{args.model_id}_{args.lr:.1e}_{decomp_mode_name}_{args.train_position}_{args.trainable_type}"
    return f"{args.model_id}_{args.lr:.1e}_{args.train_position}_{args.trainable_type}"
```

- [ ] **Step 3: Verify syntax**

Run: `python -m py_compile run_sft.py`
Expected: no output.

- [ ] **Step 4: Verify all five modes resolve to a model construction without error**

Run a no-train smoke check (this will fail at the data-loading or training step but should pass argument parsing and entry into `prepare_model`):

```bash
for m in dora pissa milora randlora lift; do
  echo "=== $m ==="
  uv run run_sft.py --train-mode $m --lora-rank 4 --no-wandb --help > /dev/null && echo "$m: parse OK"
done
```
Expected: each prints `<m>: parse OK`. (The actual `prepare_model` only runs once we wire it into `main`, which already happens via `apply_mode_defaults`.)

- [ ] **Step 5: Commit**

```bash
git add run_sft.py
git commit -m "$(cat <<'EOF'
sft: dispatch dora/pissa/milora/randlora/lift in prepare_model

Adds branches for the five new train modes and extends compute_run_name
with matching templates. PEFT-based variants (dora/pissa/milora/randlora)
share the LoRA-family code path and emit standard PEFT adapters; lift
loads the dense model with all params trainable.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Add LIFT branch to `build_optimizer` in `run_sft.py`

**Files:**
- Modify: `run_sft.py:801` (`build_optimizer`)

- [ ] **Step 1: Add LIFT branch at top of `build_optimizer`**

Open `run_sft.py` and replace `def build_optimizer(...)`'s body (currently L801-L818) with:

```python
def build_optimizer(args, trainable_params, trainable_named_params):
    if args.train_mode == "lift":
        if args.optimizer == "muon":
            raise ValueError(
                "--train-mode lift is incompatible with --optimizer muon. "
                "LIFT supplies its own SparseAdamW optimizer."
            )
        from optim.sparse_adam import SparseAdamW
        import torch.nn as nn
        # Build LIFT param groups: nn.Linear weights (except lm_head) get masking;
        # everything else uses standard AdamW behavior.
        # Discover the model from the param-id graph.
        # NOTE: trainable_params/named are flat lists; we re-traverse the model
        # via the named params' .data identity to find which belong to nn.Linear.
        # To keep this simple, we accept a tuple (model, named_params) by convention:
        # callers pass model in args.__dict__["_lift_model"] (set in prepare_model).
        model = getattr(args, "_lift_model", None)
        if model is None:
            raise RuntimeError(
                "LIFT optimizer requires args._lift_model to be set by prepare_model."
            )

        weights_with_mask, decay_ids = [], []
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and "lm_head" not in name and mod.weight.requires_grad:
                weights_with_mask.append(mod.weight)
                decay_ids.append(id(mod.weight))
        decay_id_set = set(decay_ids)
        other_decay, other_nodecay = [], []
        no_decay_names: tuple[str, ...] = ()  # match LIFT default
        for name, p in model.named_parameters():
            if not p.requires_grad or id(p) in decay_id_set:
                continue
            if any(nd in name for nd in no_decay_names):
                other_nodecay.append(p)
            else:
                other_decay.append(p)

        param_groups = [
            {
                "params": weights_with_mask, "weight_decay": 0.0,
                "rank": args.lift_lora_rank, "filter_rank": args.lift_filter_rank,
                "update_proj_gap": args.lift_update_interval,
                "group_name": "weights_with_mask",
            },
            {"params": other_decay,    "weight_decay": 0.0, "group_name": "other_params_w_decay"},
            {"params": other_nodecay,  "weight_decay": 0.0, "group_name": "other_params"},
        ]
        return SparseAdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    if args.optimizer == "adamw":
        return torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    if args.optimizer == "muon":
        return Muon(
            trainable_named_params,
            lr=args.lr,
            lr_adam=args.lr_adam,
            lr_embedding=args.lr_embedding,
            weight_decay=args.weight_decay,
            norm_method=args.norm_method,
        )
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")
```

- [ ] **Step 2: Wire `args._lift_model` from `prepare_model`**

In `run_sft.py`'s `prepare_model`, find the LIFT branch you added in Task 4. At its end (just before the function returns), add:

```python
        args._lift_model = model
```

(This avoids changing the `build_optimizer` signature and keeps the wiring local. The attribute is consumed only by the LIFT branch.)

- [ ] **Step 3: Verify syntax**

Run: `python -m py_compile run_sft.py`
Expected: no output.

- [ ] **Step 4: Commit**

```bash
git add run_sft.py
git commit -m "$(cat <<'EOF'
sft: route --train-mode lift through SparseAdamW in build_optimizer

LIFT trains the dense model with a sparse AdamW that masks all
nn.Linear weights (except lm_head) and recomputes the mask every
--lift-update-interval steps. Rejects --optimizer muon as incompatible.
prepare_model now stashes the model on args._lift_model so the
optimizer can rediscover its nn.Linear layers without changing the
build_optimizer signature.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Mirror Tasks 3-5 in `run_rl.py`

**Files:**
- Modify: `run_rl.py:58` (MODE_DEFAULTS), `run_rl.py:96` (choices), `run_rl.py:439` (apply_mode_defaults — already loops via `MODE_DEFAULTS[args.train_mode]`, no change needed beyond MODE_DEFAULTS extension), `run_rl.py:739` (compute_run_name), `run_rl.py:1185` (build_optimizer), `run_rl.py:1320-1410` (model construction in main).

`run_rl.py` does not import `prepare_model` from `run_sft.py`; it has parallel construction code. We mirror the same five branches there.

- [ ] **Step 1: Extend `MODE_DEFAULTS` (run_rl.py L58)**

Insert after the existing entries, with `wandb_project` prefix `math-grpo-` (matching `lora`'s `math-grpo`):

```python
    "dora":     {"lr": 9e-5, "wandb_project": "math-grpo-dora",
                 "micro_batch_size": 2, "gradient_accumulation_steps": 128},
    "pissa":    {"lr": 9e-5, "wandb_project": "math-grpo-pissa",
                 "micro_batch_size": 2, "gradient_accumulation_steps": 128},
    "milora":   {"lr": 9e-5, "wandb_project": "math-grpo-milora",
                 "micro_batch_size": 2, "gradient_accumulation_steps": 128},
    "randlora": {"lr": 9e-5, "wandb_project": "math-grpo-randlora",
                 "micro_batch_size": 2, "gradient_accumulation_steps": 128},
    "lift":     {"lr": 1e-4, "wandb_project": "math-grpo-lift",
                 "micro_batch_size": 4, "gradient_accumulation_steps": 64},
```

- [ ] **Step 2: Extend `--train-mode choices` (run_rl.py L96-99)**

Replace `choices=["full", "lora", "lora_full", "blocktt", "svd"]` with:

```python
        choices=["full", "lora", "lora_full", "dora", "pissa", "milora", "randlora",
                 "lift", "blocktt", "svd"],
        help=("Training mode: full, lora, lora_full, dora, pissa, milora, "
              "randlora, lift, blocktt, or svd"),
```

- [ ] **Step 3: Add the four new CLI flags (same as Task 3 Step 3)**

In `run_rl.py`'s `parse_args`, after the existing LoRA-family args block, insert the same four flags as in Task 3 Step 3 (`--randlora-projection-prng-key`, `--lift-lora-rank`, `--lift-filter-rank`, `--lift-update-interval`).

- [ ] **Step 4: Extend `compute_run_name` (run_rl.py L739)**

Replace the function body to mirror Task 4 Step 2 (same templates).

- [ ] **Step 5: Add the five model-construction branches in `main()` (run_rl.py around L1391)**

Find the existing block `if args.train_mode in {"lora", "lora_full"}:` (currently L1391). Convert it to a chain that handles each variant. Replace L1391-L1404 (the `if/elif` for lora/lora_full → `model = get_peft_model(...)`) with:

```python
    if args.train_mode in {"lora", "lora_full"}:
        from peft import LoraConfig, get_peft_model
        peft_config = LoraConfig(
            r=args.lora_rank, lora_alpha=32,
            target_modules=mode_info["target_modules"],
        )
        model = get_peft_model(model, peft_config)
        if args.train_mode == "lora_full":
            for p in model.parameters():
                p.requires_grad = True
        model.print_trainable_parameters()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    elif args.train_mode == "dora":
        from peft import LoraConfig, get_peft_model
        peft_config = LoraConfig(
            r=args.lora_rank, lora_alpha=32,
            target_modules=mode_info["target_modules"],
            use_dora=True,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    elif args.train_mode == "pissa":
        from peft import LoraConfig, get_peft_model
        peft_config = LoraConfig(
            r=args.lora_rank, lora_alpha=32,
            target_modules=mode_info["target_modules"],
            init_lora_weights="pissa_niter_4", lora_dropout=0,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    elif args.train_mode == "milora":
        from peft import LoraConfig, get_peft_model
        from run_sft import apply_milora_init_
        peft_config = LoraConfig(
            r=args.lora_rank, lora_alpha=32,
            target_modules=mode_info["target_modules"],
            lora_dropout=0,
        )
        model = get_peft_model(model, peft_config)
        apply_milora_init_(model, rank=args.lora_rank)
        model.print_trainable_parameters()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    elif args.train_mode == "randlora":
        from peft import RandLoraConfig, get_peft_model
        peft_config = RandLoraConfig(
            r=args.lora_rank, randlora_alpha=32,
            target_modules=mode_info["target_modules"],
            projection_prng_key=args.randlora_projection_prng_key,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    elif args.train_mode == "lift":
        # Dense model; no PEFT wrapping. trainable = everything.
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        args._lift_model = model
```

Then immediately update the existing `mode_info` block (L1322-L1334) so the LoRA-family modes also populate `mode_info["target_modules"]`. Replace the `if args.train_mode in {"lora", "lora_full"}:` block at L1322 with:

```python
    if args.train_mode in {"lora", "lora_full", "dora", "pissa", "milora", "randlora"}:
        lora_target_modules = get_lora_target_modules(args.trainable_type)
        mode_info.update({
            "lora_rank": args.lora_rank,
            "lora_full_backbone_trainable": args.train_mode == "lora_full",
            "trainable_type": args.trainable_type,
            "target_modules": lora_target_modules,
            "vllm_url": args.vllm_url,
            "rollout_backend": lora_rollout_backend,
        })
        if args.train_mode == "randlora":
            mode_info["randlora_projection_prng_key"] = args.randlora_projection_prng_key
    elif args.train_mode == "lift":
        mode_info.update({
            "lift_lora_rank": args.lift_lora_rank,
            "lift_filter_rank": args.lift_filter_rank,
            "lift_update_interval": args.lift_update_interval,
        })
    elif args.train_mode == "blocktt":
        # ... existing block unchanged
```

- [ ] **Step 6: Update `build_optimizer` in `run_rl.py` (L1185)**

Apply the same edit as Task 5 Step 1 (LIFT branch at top, rejecting Muon, reading `args._lift_model`). The LIFT branch is identical between the two files.

- [ ] **Step 7: Extend the dense-vLLM rollout branch (run_rl.py L1597)**

Find:
```python
    else:
        generate_for_train, generate_for_eval, in_process_llm = build_local_vllm_generators(
            args, model
        )
```
This branch is reached when `train_mode in {"full", "blocktt", "svd"}`. The current code uses `else`, so the only thing we need is to ensure `lift` reaches it. Find the surrounding `if/elif` chain (L1581-L1597) and confirm `lift` falls through to the `else`. If there's an explicit `if args.train_mode in {...}` filter that would skip `lift`, modify so `lift` falls through to `build_local_vllm_generators`. Read the full block to confirm.

If the chain is `if lora_rollout_backend == "http": ... elif lora_rollout_backend == "local_inproc": ... else: build_local_vllm_generators`, then since `resolve_lora_rollout_backend("lift", ...)` returns `None`, lift correctly hits the `else`. No change needed.

- [ ] **Step 8: Verify syntax**

Run: `python -m py_compile run_rl.py run_sft.py`
Expected: no output.

- [ ] **Step 9: Commit**

```bash
git add run_rl.py
git commit -m "$(cat <<'EOF'
rl: register and dispatch dora/pissa/milora/randlora/lift train modes

Mirrors the run_sft.py changes: extends MODE_DEFAULTS, --train-mode
choices, CLI flags, compute_run_name, and the model-construction
block. PEFT-based variants share the LoRA-family target_modules
plumbing; lift falls through to build_local_vllm_generators (which
currently serves --train-mode full).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Extend rollout-backend, normalize-name, save_merged_checkpoint, and math-verify in `run_rl.py`

**Files:**
- Modify: `run_rl.py:559` (warning), `run_rl.py:799` (`save_merged_checkpoint`), `run_rl.py:913` (`resolve_lora_rollout_backend`), `run_rl.py:921` (`normalize_lora_merged_weight_name`), `run_rl.py:1867` (math-verify in-memory dispatch).

- [ ] **Step 1: Extend `resolve_lora_rollout_backend` (L913)**

Replace the function body with:

```python
def resolve_lora_rollout_backend(train_mode: str, vllm_url: str) -> str | None:
    if train_mode == "lora_full":
        return "local_inproc"
    if train_mode in {"pissa", "milora"}:
        # PiSSA/MiLoRA modify the base weight at init; the running vLLM HTTP
        # server is out of sync. Force local in-process rollout so we
        # merge_adapter() and push the materialized dense weights into vLLM.
        return "local_inproc"
    if train_mode in {"lora", "dora", "randlora"}:
        return "http" if is_vllm_http_available(vllm_url) else "local_inproc"
    return None
```

- [ ] **Step 2: Extend `normalize_lora_merged_weight_name` skip-list (L921)**

Replace `lora_markers = (...)` with:

```python
    lora_markers = (
        ".lora_A.",
        ".lora_B.",
        ".lora_embedding_A.",
        ".lora_embedding_B.",
        ".lora_magnitude_vector",
        ".randlora_lambda",
        ".randlora_gamma",
        ".randlora_m",
    )
```

- [ ] **Step 3: Extend `save_merged_checkpoint` (L799)**

Open the function. Replace the `train_mode in {"lora", "lora_full"}` set with the full PEFT-family set, and add `lift` to the dense `full` set:

```python
def save_merged_checkpoint(model, tokenizer, ckpt_dir: str, train_mode: str, args):
    """Save model in plain HuggingFace format. The on-disk result contains
    only nn.Linear layers (no LoRA adapters, no BTT/SVD factored cores).
    The in-memory model object is never mutated; training can resume.
    """
    os.makedirs(ckpt_dir, exist_ok=True)

    if train_mode in {"full", "lift"}:
        model.save_pretrained(ckpt_dir)
    elif train_mode in {"lora", "lora_full", "dora", "pissa", "milora", "randlora"}:
        model.merge_adapter()
        try:
            base = model.get_base_model()
            base.save_pretrained(ckpt_dir)
        finally:
            model.unmerge_adapter()
    elif train_mode in {"blocktt", "svd"}:
        if train_mode == "blocktt" and getattr(args, "calib_mode", "none") != "none":
            save_calibrated_btt_hf_pretrained(model, ckpt_dir)
        else:
            state_dict = _build_factored_dense_state_dict(model)
            model.save_pretrained(ckpt_dir, state_dict=state_dict)
    else:
        raise ValueError(f"Unknown train_mode for save_merged_checkpoint: {train_mode}")

    tokenizer.save_pretrained(ckpt_dir)
```

- [ ] **Step 4: Update `--enable-math-verify` warning (L559)**

Find:
```python
    if args.enable_math_verify and not args.enable_merged_ckpt and args.train_mode != "full":
```
Replace with:
```python
    if (args.enable_math_verify and not args.enable_merged_ckpt
            and args.train_mode not in {"full", "lift"}):
```

- [ ] **Step 5: Extend math-verify in-memory hot-swap (L1867)**

Find the math-verify block. The dispatch on `args.train_mode` lives around L1874-L1895. Replace it with:

```python
                if args.train_mode in {"blocktt", "svd"}:
                    weight_tuples = export_weights_for_vllm(model)
                elif args.train_mode in {"lora", "lora_full", "dora", "pissa", "milora", "randlora"}:
                    model.merge_adapter()
                    try:
                        base = model.get_base_model()
                        weight_tuples = []
                        seen = set()
                        for name, p in base.named_parameters():
                            normalized = normalize_lora_merged_weight_name(name)
                            if normalized is None or normalized in seen:
                                continue
                            seen.add(normalized)
                            weight_tuples.append((normalized, p))
                        in_process_llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(
                            weight_tuples
                        )
                    finally:
                        model.unmerge_adapter()
                    weight_tuples = None  # already loaded
                else:
                    # full, lift -> raw named_parameters
                    weight_tuples = [(n, p) for n, p in model.named_parameters()]
```

- [ ] **Step 6: Verify syntax**

Run: `python -m py_compile run_rl.py`
Expected: no output.

- [ ] **Step 7: Commit**

```bash
git add run_rl.py
git commit -m "$(cat <<'EOF'
rl: route new train modes through rollout, save_merged_ckpt, math-verify

- resolve_lora_rollout_backend forces pissa/milora to local_inproc
  (their modified base weight desyncs from a running vLLM server).
  dora/randlora keep both backends.
- normalize_lora_merged_weight_name adds randlora_{lambda,gamma,m}
  to the skip-list for safety.
- save_merged_checkpoint extends the {lora, lora_full} branch to
  cover {dora, pissa, milora, randlora} (merge_adapter materializes
  the right effective weight for all four) and adds lift to the
  dense {full} branch.
- The --enable-math-verify warning silences for lift (it saves a
  loadable HF checkpoint like full).
- The math-verify in-memory hot-swap dispatch mirrors the same
  groupings.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Extend flag-validation table in `run_rl.py`

**Files:**
- Modify: `run_rl.py:470` (`validate_mode_specific_flags`)

- [ ] **Step 1: Replace the validation function**

Open `run_rl.py:470-540`. Rewrite the function to handle the new modes:

```python
def validate_mode_specific_flags(args, argv):
    LORA_FAMILY = {"lora", "lora_full", "dora", "pissa", "milora", "randlora"}
    LORA_HTTP_OK = {"lora", "dora", "randlora"}  # vllm-url valid for these only

    mode_to_flag_sets = {
        "lora_family_shared":   ["--lora-rank"],
        "lora_http":            ["--vllm-url"],
        "randlora":             ["--randlora-projection-prng-key"],
        "lift":                 ["--lift-lora-rank", "--lift-filter-rank",
                                 "--lift-update-interval"],
        "blocktt": [
            "--decomp-mode", "--blocktt-rank", "--no-train-bias",
            "--blocktt-normalize-after-update",
            "--blocktt-factorize-by-head", "--no-blocktt-factorize-by-head",
        ],
        "svd": [],
    }

    # --lora-rank is only valid for the lora family
    if args.train_mode not in LORA_FAMILY:
        passed = [f for f in mode_to_flag_sets["lora_family_shared"] if _flag_was_passed(argv, f)]
        if passed:
            raise ValueError(
                f"{', '.join(passed)} is only valid when --train-mode is one of: "
                f"{sorted(LORA_FAMILY)}"
            )

    # --vllm-url is only valid for lora/dora/randlora
    if args.train_mode not in LORA_HTTP_OK:
        passed = [f for f in mode_to_flag_sets["lora_http"] if _flag_was_passed(argv, f)]
        if passed:
            raise ValueError(
                f"{', '.join(passed)} is only valid when --train-mode is one of: "
                f"{sorted(LORA_HTTP_OK)}"
            )

    # --randlora-projection-prng-key is randlora-only
    if args.train_mode != "randlora":
        passed = [f for f in mode_to_flag_sets["randlora"] if _flag_was_passed(argv, f)]
        if passed:
            raise ValueError(f"{', '.join(passed)} is only valid when --train-mode randlora")

    # --lift-* is lift-only
    if args.train_mode != "lift":
        passed = [f for f in mode_to_flag_sets["lift"] if _flag_was_passed(argv, f)]
        if passed:
            raise ValueError(f"{', '.join(passed)} is only valid when --train-mode lift")

    # --optimizer muon is incompatible with lift
    if args.train_mode == "lift" and args.optimizer == "muon":
        raise ValueError(
            "--optimizer muon is incompatible with --train-mode lift. "
            "LIFT supplies its own SparseAdamW optimizer."
        )

    # --- existing blocktt/svd/full guards below, unchanged ---
    if args.train_mode != "blocktt":
        passed = [f for f in mode_to_flag_sets["blocktt"] if _flag_was_passed(argv, f)]
        if passed:
            raise ValueError(
                f"{', '.join(passed)} is only valid when --train-mode blocktt"
            )
    else:
        blocktt_targets = get_blocktt_target_module_names(args.trainable_type)
        decomp_mode, module_decomp_modes = resolve_blocktt_decomp_modes(
            args.decomp_mode,
            include_names=blocktt_targets,
            default_mode="input_one_block",
        )
        args.decomp_mode = decomp_mode
        args.blocktt_module_decomp_modes = module_decomp_modes
        args.decomp_mode_display = format_blocktt_decomp_mode(decomp_mode)

    if args.train_mode == "full" and _flag_was_passed(argv, "--trainable-type"):
        raise ValueError(
            "--trainable-type is only valid when --train-mode "
            "lora, lora_full, dora, pissa, milora, randlora, blocktt, or svd"
        )

    train_position_passed = _flag_was_passed(argv, "--train-position")
    if args.train_mode in ({"full", "lift"} | LORA_FAMILY) and train_position_passed:
        raise ValueError("--train-position is only valid when --train-mode blocktt or svd")
    if args.train_mode == "blocktt" and train_position_passed:
        if args.train_position not in {"small", "large", "both"}:
            raise ValueError("--train-position for blocktt must be one of: small, large, both")
    if args.train_mode == "svd" and train_position_passed:
        if args.train_position not in {"output", "input", "both"}:
            raise ValueError("--train-position for svd must be one of: output, input, both")

    s_merged_to_passed = _flag_was_passed(argv, "--s-merged-to")
    if args.train_mode in ({"full", "lift"} | LORA_FAMILY) and s_merged_to_passed:
        raise ValueError("--s-merged-to is only valid when --train-mode blocktt or svd")
    if (
        args.train_mode == "blocktt"
        and s_merged_to_passed
        and args.train_position == "both"
        and args.s_merged_to in {"frozen", "trainable"}
    ):
        raise ValueError(
            "--s-merged-to frozen/trainable is invalid when blocktt --train-position is both; "
            "use 'split' or 'keep_trainable'"
        )
```

(Preserve any further code in the function that follows the section above. Read the existing function fully and re-emit identical tail content.)

- [ ] **Step 2: Verify syntax**

Run: `python -m py_compile run_rl.py`
Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add run_rl.py
git commit -m "$(cat <<'EOF'
rl: extend flag-validation for new train modes

- --lora-rank valid for lora|lora_full|dora|pissa|milora|randlora.
- --vllm-url valid only for lora|dora|randlora (pissa/milora are
  forced to local rollout).
- --randlora-projection-prng-key is randlora-only.
- --lift-* flags are lift-only.
- --optimizer muon is rejected for lift.
- --train-position and --s-merged-to remain blocktt/svd only.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Add `milora` and `randlora` branches to `ref/LIFT/src/finetune_lora.py`

**Files:**
- Modify: `ref/LIFT/src/finetune_lora.py:390` (extend list), `:425-431` (delete dead `hira` branch), `:651` (extend list); add helper function and new CLI flag.

- [ ] **Step 1: Add `--randlora_projection_prng_key` CLI flag**

Open `ref/LIFT/src/finetune_lora.py`, find the argparse block (around L290-340). After the existing `--lora_alpha` / `--lora_dropout` arguments, add:

```python
    parser.add_argument(
        "--randlora_projection_prng_key",
        type=int,
        default=0,
        help="Seed for RandLoRA's shared random bases (default: 0).",
    )
```

- [ ] **Step 2: Add `apply_milora_init_` local helper**

Insert this near the top of `finetune_lora.py`, after the existing imports (after L78):

```python
@torch.no_grad()
def apply_milora_init_(peft_model, *, rank: int) -> None:
    """In-place MiLoRA initialization (vendored locally to keep LIFT
    independent from the main repo's run_sft.py helper)."""
    from peft.tuners.lora import LoraLayer
    first_check = True
    for _name, module in peft_model.named_modules():
        if not isinstance(module, LoraLayer):
            continue
        base = module.get_base_layer()
        W = base.weight.data
        dtype, device = W.dtype, W.device
        U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
        r = rank
        U_r, S_r, Vh_r = U[:, -r:], S[-r:], Vh[-r:, :]
        sqrt_S = S_r.sqrt()
        adapter_name = list(module.lora_A.keys())[0]
        alpha = module.lora_alpha[adapter_name]
        scale_correction = (r / alpha) ** 0.5
        lora_A = (sqrt_S.unsqueeze(1) * Vh_r) * scale_correction
        lora_B = (U_r * sqrt_S.unsqueeze(0)) * scale_correction
        residual = W.float() - U_r @ torch.diag(S_r) @ Vh_r
        module.lora_A[adapter_name].weight.data.copy_(lora_A.to(dtype=dtype, device=device))
        module.lora_B[adapter_name].weight.data.copy_(lora_B.to(dtype=dtype, device=device))
        base.weight.data.copy_(residual.to(dtype=dtype, device=device))
        if first_check:
            reconstructed = (alpha / r) * lora_B.float() @ lora_A.float() + residual
            rel_err = (
                torch.linalg.norm(reconstructed - W.float())
                / torch.linalg.norm(W.float())
            )
            assert rel_err < 1e-3, (
                f"MiLoRA init reconstruction error too high: {rel_err:.2e}"
            )
            first_check = False
```

- [ ] **Step 3: Extend the cascade list and add `milora` + `randlora` branches (L390 area)**

Find:
```python
    if args.adapter_name in ["lora", "dora", "pissa"]:
```
Replace with:
```python
    if args.adapter_name in ["lora", "dora", "pissa", "milora", "randlora"]:
```

Then find the existing `elif args.adapter_name == "pissa":` block (ends around L422 with `target_modules=args.target_modules, task_type="CAUSAL_LM",)`). Immediately after that closing `)`, insert:

```python
        elif args.adapter_name == "milora":
            print("MiLoRA Init")
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=0,
                target_modules=args.target_modules,
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)
            apply_milora_init_(model, rank=args.lora_r)
        elif args.adapter_name == "randlora":
            print("RandLoRA Init")
            from peft import RandLoraConfig
            config = RandLoraConfig(
                r=args.lora_r,
                randlora_alpha=args.lora_alpha,
                target_modules=args.target_modules,
                projection_prng_key=args.randlora_projection_prng_key,
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)
```

- [ ] **Step 4: Delete the dead `hira` branch (L425-431)**

The existing `elif args.adapter_name == "hira":` block calls `convert_layer_to_hira(...)`, which is undefined and unimported. Locate and delete the entire `elif` block (the four lines under it, plus the for-loop that freezes non-hira params if it's still part of that branch). Confirm by reading the source carefully — only the `hira` `elif` and its body should go.

- [ ] **Step 5: Extend the second cascade list (L651)**

Find the second occurrence of `if args.adapter_name in ["lora", "dora", "pissa"]:` (around L651 — handles save/eval routing) and replace identically:
```python
    if args.adapter_name in ["lora", "dora", "pissa", "milora", "randlora"]:
```

- [ ] **Step 6: Verify syntax**

Run: `python -m py_compile ref/LIFT/src/finetune_lora.py`
Expected: no output. If it fails on `torch.no_grad()`, ensure `import torch` is already at the top (it is, at L9).

- [ ] **Step 7: Commit**

```bash
git add ref/LIFT/src/finetune_lora.py
git commit -m "$(cat <<'EOF'
LIFT: add milora and randlora adapter branches; remove dead hira branch

- milora: standard PEFT LoraConfig + post-init SVD pass that swaps in
  the bottom-r singular components (apply_milora_init_ duplicated
  locally to keep LIFT independent from the main-repo helper).
- randlora: PEFT 0.17.1 RandLoraConfig with --randlora_projection_prng_key.
- Existing lora/dora/pissa branches are untouched.
- The hira elif branch called convert_layer_to_hira (undefined and
  unimported) and would have crashed at runtime; deleted per spec.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Add four LIFT bash scripts

**Files:**
- Create: `ref/LIFT/bash_scripts/finetune_math_milora.sh`
- Create: `ref/LIFT/bash_scripts/finetune_math_randlora.sh`
- Create: `ref/LIFT/bash_scripts/finetune_commonsense_milora.sh`
- Create: `ref/LIFT/bash_scripts/finetune_commonsense_randlora.sh`

- [ ] **Step 1: Create `finetune_math_milora.sh`**

```bash
cp ref/LIFT/bash_scripts/finetune_math_lora.sh \
   ref/LIFT/bash_scripts/finetune_math_milora.sh
```

Open `ref/LIFT/bash_scripts/finetune_math_milora.sh` and change:
- `adapter_name="${adapter_name:-lora}"` → `adapter_name="${adapter_name:-milora}"`
- `lora_r="${lora_r:-64}"` → `lora_r="${lora_r:-128}"`
- `lora_alpha="${lora_alpha:-128}"` → `lora_alpha="${lora_alpha:-128}"` (unchanged)
- `wandb_project="${wandb_project:-math-${model_tag}}"` → keep as-is (the `adapter_name` is already in the run_name)

- [ ] **Step 2: Create `finetune_math_randlora.sh`**

```bash
cp ref/LIFT/bash_scripts/finetune_math_lora.sh \
   ref/LIFT/bash_scripts/finetune_math_randlora.sh
```

Edit:
- `adapter_name="${adapter_name:-lora}"` → `adapter_name="${adapter_name:-randlora}"`
- `lora_r="${lora_r:-64}"` → `lora_r="${lora_r:-32}"`
- `lora_alpha="${lora_alpha:-128}"` → `lora_alpha="${lora_alpha:-640}"`
- In the `accelerate launch ... src/finetune_lora.py` block, add a new line after `--lora_alpha ${lora_alpha} \`:
  ```
      --randlora_projection_prng_key ${seed} \
  ```

- [ ] **Step 3: Create `finetune_commonsense_milora.sh`**

```bash
cp ref/LIFT/bash_scripts/finetune_commonsense_lora.sh \
   ref/LIFT/bash_scripts/finetune_commonsense_milora.sh
```

Apply the same `adapter_name`/`lora_r`/`lora_alpha` edits as Step 1. (Confirm `finetune_commonsense_lora.sh` exists; if not, copy from `finetune_math_lora.sh` and rename data path arguments.)

- [ ] **Step 4: Create `finetune_commonsense_randlora.sh`**

```bash
cp ref/LIFT/bash_scripts/finetune_commonsense_lora.sh \
   ref/LIFT/bash_scripts/finetune_commonsense_randlora.sh
```

Apply the same edits as Step 2.

- [ ] **Step 5: Make all four scripts executable**

```bash
chmod +x ref/LIFT/bash_scripts/finetune_math_milora.sh \
         ref/LIFT/bash_scripts/finetune_math_randlora.sh \
         ref/LIFT/bash_scripts/finetune_commonsense_milora.sh \
         ref/LIFT/bash_scripts/finetune_commonsense_randlora.sh
```

- [ ] **Step 6: Verify scripts are syntactically valid bash**

```bash
for f in ref/LIFT/bash_scripts/finetune_{math,commonsense}_{milora,randlora}.sh; do
  bash -n "$f" && echo "$f: OK"
done
```
Expected: each prints `<path>: OK`.

- [ ] **Step 7: Commit**

```bash
git add ref/LIFT/bash_scripts/finetune_math_milora.sh \
        ref/LIFT/bash_scripts/finetune_math_randlora.sh \
        ref/LIFT/bash_scripts/finetune_commonsense_milora.sh \
        ref/LIFT/bash_scripts/finetune_commonsense_randlora.sh
git commit -m "$(cat <<'EOF'
LIFT: add bash scripts for milora and randlora (math + commonsense)

Mirrors the existing finetune_{math,commonsense}_lora.sh scripts.
Defaults: milora uses PiSSA-like r=128, alpha=128; randlora uses
PEFT-published r=32, randlora_alpha=640 with the seed forwarded as
the projection PRNG key.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: CLI-validation tests for the five new modes

**Files:**
- Create: `tests/test_lora_variants_cli.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_lora_variants_cli.py`:

```python
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


class TestRunRlCliFlagValidation(unittest.TestCase):
    """--train-mode-specific flag validation in run_rl.py."""

    def _parse(self, argv):
        import run_rl
        args = run_rl.parse_args(argv)
        run_rl.apply_mode_defaults(args)
        run_rl.validate_mode_specific_flags(args, argv)
        return args

    # Each new mode parses with --no-wandb (smoke).
    def test_dora_parses(self):
        args = self._parse(["--train-mode", "dora", "--lora-rank", "8", "--no-wandb"])
        self.assertEqual(args.train_mode, "dora")
        self.assertEqual(args.lora_rank, 8)

    def test_pissa_parses(self):
        args = self._parse(["--train-mode", "pissa", "--lora-rank", "8", "--no-wandb"])
        self.assertEqual(args.train_mode, "pissa")

    def test_milora_parses(self):
        args = self._parse(["--train-mode", "milora", "--lora-rank", "8", "--no-wandb"])
        self.assertEqual(args.train_mode, "milora")

    def test_randlora_parses_with_prng_key(self):
        args = self._parse([
            "--train-mode", "randlora", "--lora-rank", "8",
            "--randlora-projection-prng-key", "42", "--no-wandb",
        ])
        self.assertEqual(args.randlora_projection_prng_key, 42)

    def test_lift_parses_with_lift_flags(self):
        args = self._parse([
            "--train-mode", "lift",
            "--lift-lora-rank", "64",
            "--lift-filter-rank", "64",
            "--lift-update-interval", "200",
            "--no-wandb",
        ])
        self.assertEqual(args.lift_lora_rank, 64)
        self.assertEqual(args.lift_filter_rank, 64)
        self.assertEqual(args.lift_update_interval, 200)

    # Mode-incompatible flags raise.
    def test_pissa_rejects_vllm_url(self):
        with self.assertRaises(ValueError) as cm:
            self._parse([
                "--train-mode", "pissa", "--lora-rank", "8",
                "--vllm-url", "http://localhost:8000", "--no-wandb",
            ])
        self.assertIn("vllm-url", str(cm.exception))

    def test_milora_rejects_vllm_url(self):
        with self.assertRaises(ValueError) as cm:
            self._parse([
                "--train-mode", "milora", "--lora-rank", "8",
                "--vllm-url", "http://localhost:8000", "--no-wandb",
            ])
        self.assertIn("vllm-url", str(cm.exception))

    def test_lora_rejects_lift_flags(self):
        with self.assertRaises(ValueError) as cm:
            self._parse([
                "--train-mode", "lora", "--lora-rank", "8",
                "--lift-lora-rank", "64", "--no-wandb",
            ])
        self.assertIn("lift", str(cm.exception))

    def test_lora_rejects_randlora_prng_key(self):
        with self.assertRaises(ValueError) as cm:
            self._parse([
                "--train-mode", "lora", "--lora-rank", "8",
                "--randlora-projection-prng-key", "7", "--no-wandb",
            ])
        self.assertIn("randlora", str(cm.exception))

    def test_lift_rejects_muon(self):
        with self.assertRaises(ValueError) as cm:
            self._parse([
                "--train-mode", "lift", "--optimizer", "muon", "--no-wandb",
            ])
        self.assertIn("muon", str(cm.exception))

    def test_blocktt_still_works(self):
        # Regression: existing modes unaffected.
        args = self._parse([
            "--train-mode", "blocktt", "--decomp-mode", "input_one_block",
            "--train-position", "small", "--no-wandb",
        ])
        self.assertEqual(args.train_mode, "blocktt")


class TestRunSftCliFlagValidation(unittest.TestCase):
    """run_sft.py mirrors but does not have --vllm-url."""

    def _parse(self, argv):
        import run_sft
        args = run_sft.parse_args(argv)
        run_sft.apply_mode_defaults(args)
        if hasattr(run_sft, "validate_mode_specific_flags"):
            run_sft.validate_mode_specific_flags(args, argv)
        return args

    def test_lift_parses(self):
        args = self._parse([
            "--train-mode", "lift",
            "--lift-lora-rank", "64",
            "--no-wandb",
        ])
        self.assertEqual(args.train_mode, "lift")

    def test_milora_parses(self):
        args = self._parse([
            "--train-mode", "milora", "--lora-rank", "8", "--no-wandb",
        ])
        self.assertEqual(args.train_mode, "milora")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `python -m unittest tests/test_lora_variants_cli.py -v`
Expected: all tests PASS. If `TestRunSftCliFlagValidation.test_lift_parses` fails because `run_sft.py` doesn't have a top-level `validate_mode_specific_flags`, the test gracefully skips that check via `hasattr`. The parsing alone should succeed.

If individual tests fail with mismatched error messages, adjust the `assertIn(...)` to match the actual exception text emitted by Task 8's validation code.

- [ ] **Step 3: Commit**

```bash
git add tests/test_lora_variants_cli.py
git commit -m "$(cat <<'EOF'
tests: argparse + flag-validation for new lora variants and lift

Covers acceptance of the five new --train-mode values, the new
mode-specific flags, and the rejections wired in Task 8 (pissa/milora
forbid --vllm-url; lora forbids --lift-* and --randlora-*; lift
forbids --optimizer muon). Includes a blocktt regression check.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: `save_merged_checkpoint` routing tests for the new modes

**Files:**
- Create: `tests/test_lora_variants_merged_ckpt.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_lora_variants_merged_ckpt.py`:

```python
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


class _Tokenizer:
    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "tokenizer.json"), "w", encoding="utf-8") as f:
            f.write("{}")


class TestSaveMergedCheckpointPeftFamily(unittest.TestCase):
    """dora/pissa/milora/randlora must reuse the lora merge_adapter path."""

    def _run(self, train_mode):
        import run_rl
        base = MagicMock()
        model = MagicMock()
        model.get_base_model.return_value = base
        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "step=1")
            args = MagicMock()
            args.calib_mode = "none"
            args.enable_merged_ckpt = True
            run_rl.save_merged_checkpoint(model, _Tokenizer(), ckpt, train_mode, args)
            model.merge_adapter.assert_called_once()
            base.save_pretrained.assert_called_once_with(ckpt)
            model.unmerge_adapter.assert_called_once()
            model.save_pretrained.assert_not_called()
            self.assertTrue(os.path.exists(os.path.join(ckpt, "tokenizer.json")))

    def test_dora(self):
        self._run("dora")

    def test_pissa(self):
        self._run("pissa")

    def test_milora(self):
        self._run("milora")

    def test_randlora(self):
        self._run("randlora")

    def test_unmerges_even_if_save_raises(self):
        import run_rl
        base = MagicMock()
        base.save_pretrained.side_effect = RuntimeError("disk full")
        model = MagicMock()
        model.get_base_model.return_value = base
        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "step=1")
            args = MagicMock()
            args.calib_mode = "none"
            args.enable_merged_ckpt = True
            with self.assertRaises(RuntimeError):
                run_rl.save_merged_checkpoint(model, _Tokenizer(), ckpt, "dora", args)
            model.merge_adapter.assert_called_once()
            model.unmerge_adapter.assert_called_once()


class TestSaveMergedCheckpointLift(unittest.TestCase):
    """lift uses the dense full-style save path."""

    def test_lift_calls_save_pretrained(self):
        import run_rl
        model = MagicMock()
        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "step=1")
            args = MagicMock()
            args.calib_mode = "none"
            args.enable_merged_ckpt = True
            run_rl.save_merged_checkpoint(model, _Tokenizer(), ckpt, "lift", args)
            model.save_pretrained.assert_called_once_with(ckpt)
            model.merge_adapter.assert_not_called()
            self.assertTrue(os.path.exists(os.path.join(ckpt, "tokenizer.json")))


class TestResolveLoraRolloutBackend(unittest.TestCase):
    def test_pissa_forced_local(self):
        import run_rl
        self.assertEqual(
            run_rl.resolve_lora_rollout_backend("pissa", "http://localhost:8000"),
            "local_inproc",
        )

    def test_milora_forced_local(self):
        import run_rl
        self.assertEqual(
            run_rl.resolve_lora_rollout_backend("milora", "http://localhost:8000"),
            "local_inproc",
        )

    def test_lift_returns_none(self):
        import run_rl
        self.assertIsNone(
            run_rl.resolve_lora_rollout_backend("lift", "http://localhost:8000"),
        )


class TestNormalizeLoraMergedWeightName(unittest.TestCase):
    def test_skips_randlora_lambda(self):
        import run_rl
        self.assertIsNone(
            run_rl.normalize_lora_merged_weight_name("foo.randlora_lambda")
        )

    def test_skips_randlora_gamma(self):
        import run_rl
        self.assertIsNone(
            run_rl.normalize_lora_merged_weight_name("foo.randlora_gamma")
        )

    def test_skips_randlora_m(self):
        import run_rl
        self.assertIsNone(
            run_rl.normalize_lora_merged_weight_name("foo.randlora_m")
        )

    def test_passes_normal_param_through(self):
        import run_rl
        self.assertEqual(
            run_rl.normalize_lora_merged_weight_name("model.layers.0.weight"),
            "model.layers.0.weight",
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `python -m unittest tests/test_lora_variants_merged_ckpt.py -v`
Expected: all PASS. If `is_vllm_http_available` makes a real HTTP request inside `resolve_lora_rollout_backend("pissa", ...)`, the test still passes because the `pissa` branch returns `"local_inproc"` *before* checking the URL — confirm by re-reading Task 7 Step 1.

- [ ] **Step 3: Commit**

```bash
git add tests/test_lora_variants_merged_ckpt.py
git commit -m "$(cat <<'EOF'
tests: save_merged_checkpoint, rollout backend, name normalization

- Each PEFT-family mode (dora/pissa/milora/randlora) routes through
  merge_adapter() + base.save_pretrained() + unmerge_adapter().
- lift routes through model.save_pretrained() (dense path).
- pissa/milora are forced to local_inproc rollout.
- randlora_{lambda,gamma,m} are skipped by normalize_lora_merged_weight_name.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Final whole-suite verification

**Files:** none modified.

- [ ] **Step 1: Run all tests touched or relevant to this PR**

```bash
python -m unittest \
  tests/test_sparse_adam_smoke.py \
  tests/test_milora_init.py \
  tests/test_lora_variants_cli.py \
  tests/test_lora_variants_merged_ckpt.py \
  tests/test_run_rl_cli.py \
  tests/test_run_rl_merged_ckpt.py \
  tests/test_run_rl_math_verify_cli.py \
  tests/test_run_sft_cli_calib.py \
  -v
```
Expected: all pass. If any pre-existing test fails (e.g. a regression from the validation rewrite in Task 8), inspect the failure and fix the validation logic — do NOT loosen the existing test assertions unless they encode an obsolete behavior.

- [ ] **Step 2: Lightweight syntax check (per CLAUDE.md)**

```bash
python -m py_compile *.py optim/*.py
```
Expected: no output.

- [ ] **Step 3: Verify every new train mode reaches `prepare_model` end-to-end (CPU-only, exits before model load)**

This is a discovery check, not a full run. Skip if it would download model weights.

```bash
for m in dora pissa milora randlora lift; do
  echo "=== $m ==="
  uv run run_sft.py --train-mode $m --lora-rank 4 --no-wandb \
      --model-id Qwen/Qwen2.5-0.5B 2>&1 | head -40
done
```

The runs will likely fail at GPU/model loading, but each should print the mode-specific log lines from Task 4 (e.g. `MiLoRA rank: 4`) before failing. If a mode dies in argparse instead, fix the validation block.

- [ ] **Step 4: Manual smoke checklist (record what was actually exercised)**

Per the spec's "Smoke runs" section. The implementer should run as many of the following on real hardware as time permits and record outcomes in the commit message of the final PR:

```bash
# 1. SFT smoke (5 steps each)
for m in dora pissa milora randlora lift; do
  CUDA_VISIBLE_DEVICES=0 uv run run_sft.py --train-mode $m --lora-rank 8 \
      --no-wandb --num-train-epochs 1 --max-steps 5
done

# 2. RL smoke with merged ckpt + math-verify (2 GRPO steps)
for m in dora pissa milora randlora lift; do
  CUDA_VISIBLE_DEVICES=0 uv run run_rl.py --train-mode $m --lora-rank 8 \
      --no-wandb --enable-save-ckpt --enable-merged-ckpt --enable-math-verify \
      --n-grpo-steps 2
done

# 3. eval_rl.py round-trip
uv run eval_rl.py --checkpoint <run-dir-from-step-2>/step=2

# 4. LIFT-side end-to-end
bash ref/LIFT/bash_scripts/finetune_math_milora.sh
bash ref/LIFT/bash_scripts/finetune_math_randlora.sh
```

If any smoke run fails, file the failure as a follow-up issue or fix in this branch — do NOT squash failures into the implementation commits.

- [ ] **Step 5: Final commit (housekeeping if anything was tweaked during smoke testing)**

```bash
git status
# If clean, no commit needed. If there are changes from smoke testing, commit them with a clear message.
```

---

## Self-review checklist

Run after writing the plan, before handing off.

**Spec coverage:**

- ✅ DoRA branch in `run_sft.py` and `run_rl.py` — Task 4, Task 6.
- ✅ PiSSA branch — Task 4, Task 6.
- ✅ MiLoRA branch + `apply_milora_init_` helper — Task 2, Task 4, Task 6, Task 9.
- ✅ RandLoRA branch — Task 4, Task 6.
- ✅ LIFT mode + `SparseAdamW` vendoring — Task 1, Task 5, Task 6.
- ✅ `MODE_DEFAULTS` extension — Task 3, Task 6.
- ✅ New CLI flags — Task 3, Task 6.
- ✅ `compute_run_name` extension — Task 4, Task 6.
- ✅ `resolve_lora_rollout_backend` extension — Task 7.
- ✅ `normalize_lora_merged_weight_name` skip-list — Task 7.
- ✅ `save_merged_checkpoint` extension — Task 7.
- ✅ `--enable-math-verify` warning extension — Task 7.
- ✅ Math-verify in-memory hot-swap dispatch — Task 7.
- ✅ Flag-validation table — Task 8.
- ✅ LIFT script `milora`/`randlora` branches — Task 9.
- ✅ Dead `hira` branch removal in LIFT — Task 9 Step 4.
- ✅ Four LIFT bash scripts — Task 10.
- ✅ Unit tests (`test_milora_init`, `test_sparse_adam_smoke`, `test_lora_variants_cli`, `test_lora_variants_merged_ckpt`) — Tasks 1, 2, 11, 12.
- ✅ Smoke-run checklist — Task 13.
- ✅ `eval_rl.py` is mode-agnostic; no code change needed — verified in Task 13 Step 4.

**Type / signature consistency:**

- `apply_milora_init_(peft_model, *, rank: int) -> None` — same signature in both `run_sft.py` (Task 2) and `ref/LIFT/src/finetune_lora.py` (Task 9). ✅
- `save_merged_checkpoint(model, tokenizer, ckpt_dir, train_mode, args)` — signature unchanged from current code. ✅
- `args._lift_model` attribute — set in Task 4 (run_sft) and Task 6 (run_rl), consumed by `build_optimizer` in Task 5 (run_sft) and Task 6 (run_rl). ✅
- `MODE_DEFAULTS` keys (`lr`, `wandb_project`, `micro_batch_size`, `gradient_accumulation_steps`) — match existing entries in both files. ✅

**Placeholder scan:** no TBDs, all code blocks are complete. The `<LIFT_SHA>` token in Task 1 is intentional and the step explains how to fill it in.

---

## Open notes for the implementer

- **PEFT version:** the spec assumes PEFT 0.17.1. `pyproject.toml` declares `peft>=0.17.1` so this holds. `RandLoraConfig` is importable as `from peft import RandLoraConfig`.
- **vLLM RandLoRA HTTP support:** unverified upstream. If Task 13 Step 4 #3 (RandLoRA HTTP rollout) fails with a vLLM error like "unknown adapter type", revert to forcing `randlora` to `local_inproc` in `resolve_lora_rollout_backend` (one-line change in Task 7 Step 1) and add a note to the open-risks section of the spec.
- **MiLoRA with `lora_alpha=32` (the default in run_sft.py / run_rl.py LoRA branches):** the scale correction `(r/α)^{1/2}` with `α=32`, `r=8` gives `0.5`. The reconstruction self-check in `apply_milora_init_` validates this on the first layer. If it fires, debug there before trusting downstream training.
- **LIFT memory ceiling:** the spec says LIFT fits H100 for both Qwen3-1.7B (RL) and Qwen3-4B (SFT). Confirm during the Task 13 Step 4 smoke runs. If OOM on 4B, add gradient checkpointing rather than reducing the dense param count.
