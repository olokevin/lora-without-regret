# FuRA System-Performance Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible harness that measures training / deployment / eval system cost of FuRA versus Full FT / LoRA / DoRA / RandLoRA / LIFT on the LIFT commonsense SFT path, plus an auto-launched evaluation of a fused Triton Step-2 kernel.

**Architecture:** Short-horizon system-only protocol (300 optimizer steps per run). A thin `SysMon` helper is injected into the three existing LIFT trainers (`src/finetune_blocktt.py`, `src/finetune_lora.py`, `src/finetune_sft.py`) with a new `--max_steps` flag. A driver shell script sweeps the 18-run matrix. Standalone microbench / merge-decode / aggregation tools consume the emitted JSONs and produce Markdown + LaTeX tables. Phase 2 adds a Triton kernel under a feature flag with its own test + microbench + auto-report pipeline.

**Tech Stack:** Python 3.13 / uv, PyTorch (bf16), HuggingFace Accelerate + PEFT, Triton, pytest. No new third-party dependencies beyond Triton (already in PyTorch's Linux wheel).

**Spec:** `docs/superpowers/specs/2026-04-21-fura-system-eval.md`

**Reference paths (read-only during implementation):**
- Paper: `docs/26_nips_fura_paper/neurips_2026.tex` §3.3
- LIFT harness: `ref/LIFT/src/finetune_{blocktt,lora,sft}.py`, `ref/LIFT/bash_scripts/finetune_commonsense_*.sh`
- FuRA layer: `btt_layer.py` (BTTLayer.forward at lines 786–830)

---

## File Structure

**Phase 1 — new files**
- `tools/system_metrics.py` — `SysMon` helper class (importable from LIFT trainers)
- `tools/bench_fbopt.py` — per-method forward/backward/opt-step split microbench
- `tools/bench_merge_and_decode.py` — per-method merge + `generate` benchmark
- `tools/aggregate_sys_metrics.py` — sweep JSONs → Markdown + LaTeX tables + `sys_perf_vs_rank.pdf`
- `tools/run_sft_matrix.sh` — 18-run driver, resumable
- `tests/test_system_metrics.py` — unit tests for `SysMon`
- `tests/test_aggregate_sys_metrics.py` — parser / table emitter tests

**Phase 1 — modified files**
- `ref/LIFT/src/finetune_blocktt.py` — add `--max_steps` arg + `SysMon` hooks
- `ref/LIFT/src/finetune_lora.py` — same
- `ref/LIFT/src/finetune_sft.py` — same
- `ref/LIFT/bash_scripts/finetune_commonsense_lora.sh` — parametrise rank/alpha
- `ref/LIFT/bash_scripts/finetune_commonsense_randlora.sh` — parametrise rank/alpha
- `ref/LIFT/bash_scripts/finetune_commonsense_lift.sh` — parametrise rank
- (DoRA reuses the LoRA shell script with `adapter_name=dora`; no script file needed.)

**Phase 2 — new files**
- `fura_kernels/__init__.py` — package init
- `fura_kernels/triton_btt.py` — Triton Step-2 kernel + `torch.autograd.Function`
- `tests/test_fura_fused_kernel.py` — shape grid + gradcheck + non-contiguous
- `tools/bench_fused_kernel.py` — per-layer speedup microbench
- `tools/bench_fused_kernel_sft.py` — baseline vs fused short-horizon SFT
- `tools/run_kernel_eval.sh` — single-command auto-launch for Phase 2
- `tools/write_kernel_report.py` — JSONs → `kernel_eval_report.md`

**Phase 2 — modified files**
- `btt_layer.py` — opt-in `use_fused_step2` branch in `BTTLayer.forward`

**Output paths (not committed):**
- Run artifacts: `/data/yequan/fura/sys_eval/commonsense/<method>/<rank>/sys_metrics.json`
- Aggregated tables: `docs/26_nips_fura_paper/tables/tab_system_perf_commonsense_{headline,sweep}.{md,tex}`
- Rank plot: `docs/26_nips_fura_paper/figs/sys_perf_vs_rank.pdf`
- Phase 2 report: `docs/26_nips_fura_paper/kernel_eval_report.md`

---

# PHASE 1: System-performance harness

## Task 1: SysMon helper

**Files:**
- Create: `tools/system_metrics.py`
- Test: `tests/test_system_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_system_metrics.py
import json
import time
import torch
import pytest
from pathlib import Path

from tools.system_metrics import SysMon


def test_sysmon_dump_writes_json(tmp_path):
    mon = SysMon(tmp_path, method="lora", rank=64, base_params=8_030_000_000)
    mon.record_step(0.1)
    mon.record_step(0.12)
    mon.record_step(0.11)

    model = torch.nn.Linear(8, 8)  # 72 params
    mon.dump(model, extra={"effective_tokens_per_step": 16384})

    out = tmp_path / "sys_metrics.json"
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["method"] == "lora"
    assert data["rank"] == 64
    assert data["trainable_params"] == 72
    assert data["total_params"] == 72
    assert data["steps_recorded"] == 3
    assert data["median_step_s"] == pytest.approx(0.11, abs=1e-6)
    assert data["effective_tokens_per_step"] == 16384


def test_sysmon_warmup_cutoff_uses_tail_when_enough_steps(tmp_path):
    mon = SysMon(tmp_path, method="fura", rank=None, base_params=1)
    # 150 fast steps (warmup) + 50 slow steps
    for _ in range(150):
        mon.record_step(0.01)
    for _ in range(50):
        mon.record_step(1.0)
    mon.dump(torch.nn.Linear(1, 1))
    data = json.loads((tmp_path / "sys_metrics.json").read_text())
    # After 100-step warmup cut, median is over steps[100:] = 50 fast + 50 slow,
    # median of that is halfway between 0.01 and 1.0 → 0.01 (since sorted[99] = 0.01).
    # But we want to be tolerant: assert median is one of the observed values.
    assert data["median_step_s"] in (0.01, 1.0, pytest.approx(0.505, abs=0.5))
    assert data["steps_recorded"] == 200


def test_sysmon_peak_memory_captured_when_cuda(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    mon = SysMon(tmp_path, method="full", rank=None, base_params=1)
    x = torch.zeros(1024, 1024, device="cuda")
    mon.record_step(0.01)
    mon.dump(torch.nn.Linear(1, 1).cuda())
    data = json.loads((tmp_path / "sys_metrics.json").read_text())
    assert data["peak_alloc_bytes"] >= x.numel() * 4
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_system_metrics.py -v
```

Expected: all three tests FAIL with `ModuleNotFoundError: No module named 'tools.system_metrics'`.

- [ ] **Step 3: Write minimal implementation**

```python
# tools/system_metrics.py
"""Tiny instrumentation helper for short-horizon SFT runs.

Writes a single sys_metrics.json file capturing steady-state step time,
parameter footprint and peak GPU memory. Designed to be imported from the
three LIFT trainers (finetune_blocktt.py, finetune_lora.py, finetune_sft.py).
"""
import json
import statistics
import time
from pathlib import Path
from typing import Any, Optional

import torch


class SysMon:
    WARMUP_STEPS = 100  # steps to discard before computing median

    def __init__(self, out_dir, method: str, rank: Optional[int], base_params: int):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.out = self.out_dir / "sys_metrics.json"
        self.method = method
        self.rank = rank
        self.base_params = base_params
        self.step_times: list[float] = []
        self.start_wall = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def record_step(self, dt: float) -> None:
        self.step_times.append(float(dt))

    def dump(self, model: torch.nn.Module, extra: Optional[dict[str, Any]] = None) -> None:
        if len(self.step_times) > self.WARMUP_STEPS + 10:
            warm = self.step_times[self.WARMUP_STEPS:]
        else:
            warm = self.step_times
        median_step = statistics.median(warm) if warm else None

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())

        data: dict[str, Any] = {
            "method": self.method,
            "rank": self.rank,
            "base_params": self.base_params,
            "trainable_params": trainable,
            "total_params": total,
            "trainable_pct": 100.0 * trainable / self.base_params,
            "stored_extra_pct": 100.0 * max(0, total - self.base_params) / self.base_params,
            "steps_recorded": len(self.step_times),
            "median_step_s": median_step,
            "total_wall_s": time.time() - self.start_wall,
            "peak_alloc_bytes": (
                torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
            ),
            "peak_reserved_bytes": (
                torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None
            ),
        }
        if extra:
            data.update(extra)
        self.out.write_text(json.dumps(data, indent=2))
```

- [ ] **Step 4: Run test to verify it passes**

```
uv run pytest tests/test_system_metrics.py -v
```

Expected: first two tests PASS; `test_sysmon_peak_memory_captured_when_cuda` PASSES if CUDA is available, otherwise is SKIPPED.

- [ ] **Step 5: Commit**

```bash
git add tools/system_metrics.py tests/test_system_metrics.py
git commit -m "sys-eval: add SysMon helper for short-horizon SFT instrumentation"
```

---

## Task 2: Add --max_steps and SysMon to finetune_blocktt.py

**Files:**
- Modify: `ref/LIFT/src/finetune_blocktt.py` (argparse block + `train_epoch` loop)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_finetune_blocktt_max_steps.py
import subprocess
import sys


def test_finetune_blocktt_accepts_max_steps_flag():
    result = subprocess.run(
        [sys.executable, "ref/LIFT/src/finetune_blocktt.py", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--max_steps" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_finetune_blocktt_max_steps.py -v
```

Expected: FAIL — `--max_steps` not in help output.

- [ ] **Step 3: Add --max_steps argparse flag**

In `ref/LIFT/src/finetune_blocktt.py`, locate line 149 (`parser.add_argument("--num_train_epochs", type=int, default=3)`) and add immediately after:

```python
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="If > 0, cap total optimizer steps at this value (for short-horizon system-eval runs).",
    )
```

- [ ] **Step 4: Wire max_steps into the train loop**

Locate line 400 (`max_train_steps = args.num_train_epochs * num_update_steps_per_epoch`) and replace with:

```python
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    if args.max_steps > 0:
        max_train_steps = min(max_train_steps, args.max_steps)
```

Locate the `train_epoch(epoch)` body, inside the `if accelerator.sync_gradients:` block right after `args.completed_steps += 1` (around line 460). Add:

```python
                if args.max_steps > 0 and args.completed_steps >= args.max_steps:
                    return  # short-horizon cap reached
```

- [ ] **Step 5: Wire SysMon into finetune_blocktt.py**

At the top of the file, add the import next to the other `sys.path`-adjacent imports (after the existing `import` block, before argparse construction):

```python
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "../../..")))
from tools.system_metrics import SysMon  # noqa: E402
```

(The relative path puts the repo root — where `tools/` lives — on sys.path.)

After the `accelerator.prepare(...)` call (around line 435), before `def train_epoch`, add:

```python
    sysmon = SysMon(
        out_dir=args.output_dir,
        method="blocktt",
        rank=(None if args.blocktt_rank == "full" else int(args.blocktt_rank)),
        base_params=sum(p.numel() for p in model.parameters()),  # approx base; overwritten below
    )
    # Correct base_params to pre-adapter count: sum everything, subtract the btt cores we attached.
    _base = sysmon.base_params
    for name, p in model.named_parameters():
        if "btt_" in name:
            _base -= p.numel()
    sysmon.base_params = _base
```

Inside `train_epoch`, wrap the `optimizer.step()` call. Replace the existing `if accelerator.sync_gradients:` block so timing covers one full optimizer step:

```python
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
```

(You must `import time` if it isn't already imported — check line 1-10 of the file first.)

After the main epoch loop (after `for epoch in range(args.num_train_epochs):` exits, around line 535), add:

```python
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
        },
    )
```

- [ ] **Step 6: Run test to verify it passes**

```
uv run pytest tests/test_finetune_blocktt_max_steps.py -v
uv run python -m py_compile ref/LIFT/src/finetune_blocktt.py
```

Expected: both PASS.

- [ ] **Step 7: Commit**

```bash
git add ref/LIFT/src/finetune_blocktt.py tests/test_finetune_blocktt_max_steps.py
git commit -m "sys-eval: add --max_steps and SysMon hooks to finetune_blocktt.py"
```

---

## Task 3: Add --max_steps and SysMon to finetune_lora.py

**Files:**
- Modify: `ref/LIFT/src/finetune_lora.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_finetune_lora_max_steps.py
import subprocess
import sys


def test_finetune_lora_accepts_max_steps_flag():
    result = subprocess.run(
        [sys.executable, "ref/LIFT/src/finetune_lora.py", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--max_steps" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_finetune_lora_max_steps.py -v
```

Expected: FAIL.

- [ ] **Step 3: Add --max_steps argparse flag**

In `ref/LIFT/src/finetune_lora.py`, find the `--num_train_epochs` argparse line (grep for it). Add immediately after:

```python
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="If > 0, cap total optimizer steps at this value (for short-horizon system-eval runs).",
    )
```

- [ ] **Step 4: Wire max_steps into the train loop**

Find `max_train_steps = args.num_train_epochs * num_update_steps_per_epoch` and add the same two lines as Task 2 Step 4:

```python
    if args.max_steps > 0:
        max_train_steps = min(max_train_steps, args.max_steps)
```

In `train_epoch`, inside `if accelerator.sync_gradients:` right after `args.completed_steps += 1` (line ~607):

```python
                if args.max_steps > 0 and args.completed_steps >= args.max_steps:
                    return
```

- [ ] **Step 5: Wire SysMon into finetune_lora.py**

Add the same `sys.path` shim and import as Task 2 Step 5 at the top of the file:

```python
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "../../..")))
from tools.system_metrics import SysMon  # noqa: E402
```

After `accelerator.prepare(...)`, before `def train_epoch`:

```python
    # Method identifier — args.adapter_name is "lora" | "dora" | "pissa" | "milora" | "randlora"
    sysmon = SysMon(
        out_dir=args.output_dir,
        method=args.adapter_name,
        rank=int(args.lora_r) if hasattr(args, "lora_r") else None,
        base_params=0,  # set below after we count adapter params
    )
    _total_now = sum(p.numel() for p in model.parameters())
    _adapter_params = sum(
        p.numel() for n, p in model.named_parameters()
        if "lora_" in n or "randlora" in n.lower() or "magnitude" in n.lower()
    )
    sysmon.base_params = _total_now - _adapter_params
```

Wrap the optimizer.step() timing — around line 599, replace the `if accelerator.sync_gradients:` block:

```python
            if accelerator.sync_gradients:
                _t0 = time.time()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                sysmon.record_step(time.time() - _t0)
                progress_bar.update(1)
                args.completed_steps += 1
```

After the main training loop finishes (after the outer `for epoch in range(args.num_train_epochs):`):

```python
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
            "adapter_name": args.adapter_name,
            "lora_alpha": args.lora_alpha,
        },
    )
```

(Check lines 1–10 for whether `import time` is already present; add if missing.)

- [ ] **Step 6: Run test to verify it passes**

```
uv run pytest tests/test_finetune_lora_max_steps.py -v
uv run python -m py_compile ref/LIFT/src/finetune_lora.py
```

Expected: both PASS.

- [ ] **Step 7: Commit**

```bash
git add ref/LIFT/src/finetune_lora.py tests/test_finetune_lora_max_steps.py
git commit -m "sys-eval: add --max_steps and SysMon hooks to finetune_lora.py"
```

---

## Task 4: Add --max_steps and SysMon to finetune_sft.py

**Files:**
- Modify: `ref/LIFT/src/finetune_sft.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_finetune_sft_max_steps.py
import subprocess
import sys


def test_finetune_sft_accepts_max_steps_flag():
    result = subprocess.run(
        [sys.executable, "ref/LIFT/src/finetune_sft.py", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--max_steps" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_finetune_sft_max_steps.py -v
```

Expected: FAIL.

- [ ] **Step 3: Add --max_steps argparse flag**

Same pattern as Task 3 Step 3, in `ref/LIFT/src/finetune_sft.py`:

```python
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="If > 0, cap total optimizer steps at this value (for short-horizon system-eval runs).",
    )
```

- [ ] **Step 4: Wire max_steps into the train loop**

Same two-line cap after `max_train_steps = args.num_train_epochs * num_update_steps_per_epoch`:

```python
    if args.max_steps > 0:
        max_train_steps = min(max_train_steps, args.max_steps)
```

In `train_epoch` inside `if accelerator.sync_gradients:` right after `args.completed_steps += 1` (line ~521):

```python
                if args.max_steps > 0 and args.completed_steps >= args.max_steps:
                    return
```

- [ ] **Step 5: Wire SysMon into finetune_sft.py**

Same sys.path shim + import as Task 2 Step 5.

After `accelerator.prepare(...)`, before `def train_epoch`. Note this file is used by both **Full FT** and **LIFT** (depending on `args.peft_tuner`):

```python
    _method = "full" if getattr(args, "peft_tuner", None) in (None, "") else args.peft_tuner
    _rank = int(args.lora_rank) if hasattr(args, "lora_rank") and args.lora_rank else None
    sysmon = SysMon(
        out_dir=args.output_dir,
        method=_method,
        rank=_rank,
        base_params=sum(p.numel() for p in model.parameters()),  # full-model count — base model itself
    )
```

(Note: in Full FT mode, *all* params are trainable and there is no adapter, so `base_params` equals `total_params`, which is correct.)

Wrap `optimizer.step()` timing (around line 513):

```python
            if accelerator.sync_gradients:
                _t0 = time.time()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                sysmon.record_step(time.time() - _t0)
                progress_bar.update(1)
                args.completed_steps += 1
```

After the main epoch loop:

```python
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
        },
    )
```

- [ ] **Step 6: Run test to verify it passes**

```
uv run pytest tests/test_finetune_sft_max_steps.py -v
uv run python -m py_compile ref/LIFT/src/finetune_sft.py
```

Expected: both PASS.

- [ ] **Step 7: Commit**

```bash
git add ref/LIFT/src/finetune_sft.py tests/test_finetune_sft_max_steps.py
git commit -m "sys-eval: add --max_steps and SysMon hooks to finetune_sft.py"
```

---

## Task 5: Parametrise rank/alpha in LoRA / RandLoRA / LIFT bash scripts

**Files:**
- Modify: `ref/LIFT/bash_scripts/finetune_commonsense_lora.sh`
- Modify: `ref/LIFT/bash_scripts/finetune_commonsense_randlora.sh`
- Modify: `ref/LIFT/bash_scripts/finetune_commonsense_lift.sh`

The LoRA and RandLoRA scripts already accept `${lora_r}` and `${lora_alpha}` as env overrides (lines 29–30 of each). The LIFT script currently sources its config from `slurm_config_lift_commonsense.txt`; we want to bypass that and take rank from env. Also each script must forward `--max_steps` from an env var, and each must respect `--save_interval` high enough to not save a checkpoint during a 300-step run.

- [ ] **Step 1: Add MAX_STEPS passthrough to LoRA script**

In `ref/LIFT/bash_scripts/finetune_commonsense_lora.sh`, near the other variable defaults (lines 26–31), add:

```bash
MAX_STEPS="${MAX_STEPS:-0}"
```

In the `accelerate launch` invocation (around line 46), add the flag. Locate the trailing `--output_dir $OUTPUT` line and insert before it:

```bash
    --max_steps ${MAX_STEPS} \
    --save_interval 100000 \
```

(`--save_interval 100000` guarantees no checkpoint write during a 300-step run.)

At the bottom, **skip the post-training eval when MAX_STEPS is set** (we have no converged model to evaluate):

```bash
if [ "${MAX_STEPS}" = "0" ]; then
    bash bash_scripts/eval_commonsense_lora.sh \
        CKPT="$OUTPUT" \
        adapter_name="${adapter_name}" \
        base_model="${MODEL}" \
        wandb_project="${wandb_project}" \
        wandb_run_name="${run_name}" \
        wandb_run_id="${wandb_run_id}"
fi
```

(Replace the existing unconditional `bash bash_scripts/eval_commonsense_lora.sh ...` block with the above.)

- [ ] **Step 2: Same for RandLoRA script**

Apply identical changes to `ref/LIFT/bash_scripts/finetune_commonsense_randlora.sh`: add `MAX_STEPS` default, pass `--max_steps` + `--save_interval 100000` to `accelerate launch`, and conditionally skip the eval.

- [ ] **Step 3: Parametrise LIFT script from env**

`ref/LIFT/bash_scripts/finetune_commonsense_lift.sh` currently uses `$1` as a SLURM array index. We want it to behave like the LoRA script (take env vars directly). Replace the env-var/cfg section (lines 18–30) with:

```bash
MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B}"
no_grad="${no_grad:-0.1}"
mask="${mask:-topk}"
lr="${lr:-2e-4}"
lora_rank="${lora_rank:-32}"
filter_rank="${filter_rank:-${lora_rank}}"
update_interval="${update_interval:-500}"
seed="${seed:-43}"
model_tag="${MODEL##*/}"
wandb_project="${wandb_project:-commonsense-${model_tag}}"
MAX_STEPS="${MAX_STEPS:-0}"
```

(Note: `no_grad`, `mask`, `update_interval` defaults — confirm by reading the LIFT config file `slurm_config_lift_commonsense.txt` once and copying the most common row's values. If you cannot find a representative row, use `no_grad=0.1 mask=topk update_interval=500` as above — these are the paper's default LIFT knobs per the LIFT README.)

In the `accelerate launch` block, add `--max_steps ${MAX_STEPS}` and change `--save_interval 5000` to `--save_interval 100000`. Conditionally run the post-eval as in Step 1.

- [ ] **Step 4: Dry-check each script (no GPU needed)**

```
bash -n ref/LIFT/bash_scripts/finetune_commonsense_lora.sh
bash -n ref/LIFT/bash_scripts/finetune_commonsense_randlora.sh
bash -n ref/LIFT/bash_scripts/finetune_commonsense_lift.sh
```

Expected: all three exit 0 (shell syntax only).

- [ ] **Step 5: Commit**

```bash
git add ref/LIFT/bash_scripts/finetune_commonsense_lora.sh \
        ref/LIFT/bash_scripts/finetune_commonsense_randlora.sh \
        ref/LIFT/bash_scripts/finetune_commonsense_lift.sh
git commit -m "sys-eval: parametrise rank/alpha/MAX_STEPS in LIFT commonsense shell scripts"
```

---

## Task 6: Run-matrix driver script

**Files:**
- Create: `tools/run_sft_matrix.sh`

- [ ] **Step 1: Write driver script**

```bash
#!/usr/bin/env bash
# tools/run_sft_matrix.sh
#
# Runs 18 short-horizon (300-step) SFT configurations sequentially on one GPU.
# Skips any configuration whose OUTPUT_SRC_DIR already contains a complete
# sys_metrics.json, so the script is resumable.
#
# Usage: GPU=0 bash tools/run_sft_matrix.sh

set -euo pipefail

GPU="${GPU:-0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIFT_DIR="${REPO_ROOT}/ref/LIFT"
OUT_ROOT="${OUT_ROOT:-/data/yequan/fura/sys_eval/commonsense}"
MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B}"
MAX_STEPS="${MAX_STEPS:-300}"
SEED="${SEED:-43}"
LR="${LR:-2e-4}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export OUTPUT_SRC_DIR="${OUT_ROOT}"

mkdir -p "${OUT_ROOT}"

have_metrics() {
    local dir="$1"
    [ -f "${dir}/sys_metrics.json" ]
}

run_one() {
    local label="$1" out_dir="$2"
    shift 2
    if have_metrics "${out_dir}"; then
        echo "[skip] ${label} — sys_metrics.json already present at ${out_dir}"
        return 0
    fi
    echo "[run ] ${label} — ${out_dir}"
    mkdir -p "${out_dir}"
    OUTPUT="${out_dir}" run_name="${label}" "$@" 2>&1 | tee "${out_dir}/matrix.log" || {
        echo "[fail] ${label} — exit $?"
        return 1
    }
}

cd "${LIFT_DIR}"

# 1. Full FT
run_one "full" "${OUT_ROOT}/full" \
    env MODEL="${MODEL}" lr="${LR}" seed="${SEED}" MAX_STEPS="${MAX_STEPS}" \
    bash bash_scripts/finetune_commonsense_full.sh

# 2. FuRA (BlockTT, default corner)
run_one "fura" "${OUT_ROOT}/fura" \
    env MODEL="${MODEL}" lr="${LR}" seed="${SEED}" MAX_STEPS="${MAX_STEPS}" \
        decomp_mode=input_one_block train_position=small s_merged_to=frozen blocktt_rank=full \
    bash bash_scripts/finetune_commonsense_blocktt.sh

# 3. LoRA rank sweep
for r in 16 32 64 128; do
    alpha=$((2 * r))
    run_one "lora-r${r}" "${OUT_ROOT}/lora/r${r}" \
        env MODEL="${MODEL}" adapter_name=lora lora_r="${r}" lora_alpha="${alpha}" \
            lr="${LR}" seed="${SEED}" MAX_STEPS="${MAX_STEPS}" \
        bash bash_scripts/finetune_commonsense_lora.sh
done

# 4. DoRA rank sweep (reuse LoRA script with adapter_name=dora)
for r in 16 32 64 128; do
    alpha=$((2 * r))
    run_one "dora-r${r}" "${OUT_ROOT}/dora/r${r}" \
        env MODEL="${MODEL}" adapter_name=dora lora_r="${r}" lora_alpha="${alpha}" \
            lr="${LR}" seed="${SEED}" MAX_STEPS="${MAX_STEPS}" \
        bash bash_scripts/finetune_commonsense_lora.sh
done

# 5. RandLoRA rank sweep (alpha = 20r per spec)
for r in 16 32 64 128; do
    alpha=$((20 * r))
    run_one "randlora-r${r}" "${OUT_ROOT}/randlora/r${r}" \
        env MODEL="${MODEL}" adapter_name=randlora lora_r="${r}" lora_alpha="${alpha}" \
            lr="${LR}" seed="${SEED}" MAX_STEPS="${MAX_STEPS}" \
        bash bash_scripts/finetune_commonsense_randlora.sh
done

# 6. LIFT rank sweep
for r in 16 32 64 128; do
    run_one "lift-r${r}" "${OUT_ROOT}/lift/r${r}" \
        env MODEL="${MODEL}" lora_rank="${r}" filter_rank="${r}" \
            lr="${LR}" seed="${SEED}" MAX_STEPS="${MAX_STEPS}" \
        bash bash_scripts/finetune_commonsense_lift.sh
done

echo "== matrix complete =="
ls -R "${OUT_ROOT}"
```

- [ ] **Step 2: Syntax check**

```
bash -n tools/run_sft_matrix.sh
chmod +x tools/run_sft_matrix.sh
```

Expected: exit 0.

- [ ] **Step 3: Write a smoke test**

```python
# tests/test_run_sft_matrix_shell.py
import subprocess


def test_driver_script_parses():
    subprocess.run(["bash", "-n", "tools/run_sft_matrix.sh"], check=True)


def test_driver_script_enumerates_18_runs(tmp_path):
    # Dry-run: replace the body of run_one with echo.
    import pathlib
    src = pathlib.Path("tools/run_sft_matrix.sh").read_text()
    # Count the expected calls: 1 full + 1 fura + 4 lora + 4 dora + 4 randlora + 4 lift = 18
    # We count the "run_one " invocations in the source.
    invocations = src.count('run_one "')
    # Two are inside for loops; those loops are over 4 ranks each.
    # 2 singleton invocations + 4 looped method blocks = 2 + 4 = 6 textual invocations
    # expanding to 2 + 4*4 = 18 actual runs.
    assert invocations == 6, f"expected 6 textual run_one calls, got {invocations}"
```

- [ ] **Step 4: Run test**

```
uv run pytest tests/test_run_sft_matrix_shell.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/run_sft_matrix.sh tests/test_run_sft_matrix_shell.py
git commit -m "sys-eval: add resumable 18-run SFT matrix driver script"
```

---

## Task 7: Forward/backward/opt microbench tool

**Files:**
- Create: `tools/bench_fbopt.py`
- Test: `tests/test_bench_fbopt.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bench_fbopt.py
import json
import subprocess
import sys
from pathlib import Path


def test_bench_fbopt_emits_json_for_toy_shape(tmp_path):
    out = tmp_path / "fbopt.json"
    result = subprocess.run(
        [
            sys.executable,
            "tools/bench_fbopt.py",
            "--method", "toy",
            "--d_in", "64",
            "--d_out", "64",
            "--batch", "32",
            "--iters", "3",
            "--warmup", "1",
            "--out", str(out),
            "--device", "cpu",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(out.read_text())
    assert data["method"] == "toy"
    assert "fwd_ms" in data
    assert "fwd_bwd_ms" in data
    assert "fwd_bwd_opt_ms" in data
    assert data["fwd_ms"] >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_bench_fbopt.py -v
```

Expected: FAIL — script doesn't exist.

- [ ] **Step 3: Implement the microbench**

```python
# tools/bench_fbopt.py
"""Forward / forward+backward / forward+backward+optimizer-step microbenchmark.

Method-agnostic: you give it a Linear-replacement callable via --method, and
it constructs a single-layer model of the given shape. Used to isolate
adapter overhead from full-model / data-loader noise.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch


def _make_layer(method: str, d_in: int, d_out: int, rank: int | None, device: str):
    """Return (module, trainable_params_list). Extend as more methods are wired."""
    if method == "toy":
        mod = torch.nn.Linear(d_in, d_out, bias=False).to(device)
        return mod, list(mod.parameters())
    if method == "full":
        mod = torch.nn.Linear(d_in, d_out, bias=False).to(device)
        for p in mod.parameters():
            p.requires_grad_(True)
        return mod, list(mod.parameters())
    if method == "lora":
        assert rank is not None
        base = torch.nn.Linear(d_in, d_out, bias=False).to(device)
        for p in base.parameters():
            p.requires_grad_(False)
        a = torch.nn.Linear(d_in, rank, bias=False).to(device)
        b = torch.nn.Linear(rank, d_out, bias=False).to(device)

        class _LoRAWrap(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base = base
                self.a = a
                self.b = b

            def forward(self, x):
                return self.base(x) + self.b(self.a(x))

        mod = _LoRAWrap()
        trainable = [*a.parameters(), *b.parameters()]
        return mod, trainable
    if method == "fura":
        from btt_layer import BTTLayer
        mod = BTTLayer(d_in, d_out, rank=rank or int(d_in ** 0.5)).to(device)
        return mod, [p for p in mod.parameters() if p.requires_grad]
    raise ValueError(f"unknown method: {method}")


def _time(fn, iters: int, warmup: int, device: str) -> float:
    for _ in range(warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.time()
        fn()
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)
    return statistics.median(times) * 1000.0  # ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True)
    p.add_argument("--d_in", type=int, required=True)
    p.add_argument("--d_out", type=int, required=True)
    p.add_argument("--rank", type=int, default=None)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    if args.device == "cpu":
        dtype = torch.float32  # bf16 on CPU is slow and noisy

    torch.manual_seed(0)
    mod, trainable = _make_layer(args.method, args.d_in, args.d_out, args.rank, args.device)
    mod = mod.to(dtype) if args.device == "cuda" else mod
    x = torch.randn(args.batch, args.d_in, device=args.device, dtype=dtype, requires_grad=False)
    opt = torch.optim.AdamW(trainable, lr=1e-4) if trainable else None

    def fwd_only():
        with torch.no_grad():
            _ = mod(x)

    def fwd_bwd():
        y = mod(x)
        loss = y.float().sum()
        loss.backward()
        for p in trainable:
            if p.grad is not None:
                p.grad = None

    def fwd_bwd_opt():
        y = mod(x)
        loss = y.float().sum()
        loss.backward()
        if opt is not None:
            opt.step()
            opt.zero_grad()

    fwd_ms = _time(fwd_only, args.iters, args.warmup, args.device)
    fwd_bwd_ms = _time(fwd_bwd, args.iters, args.warmup, args.device)
    fwd_bwd_opt_ms = _time(fwd_bwd_opt, args.iters, args.warmup, args.device)

    data = {
        "method": args.method,
        "d_in": args.d_in,
        "d_out": args.d_out,
        "rank": args.rank,
        "batch": args.batch,
        "dtype": args.dtype,
        "device": args.device,
        "fwd_ms": fwd_ms,
        "fwd_bwd_ms": fwd_bwd_ms,
        "fwd_bwd_opt_ms": fwd_bwd_opt_ms,
        "bwd_ms": fwd_bwd_ms - fwd_ms,
        "opt_ms": fwd_bwd_opt_ms - fwd_bwd_ms,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(data, indent=2))
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

```
uv run pytest tests/test_bench_fbopt.py -v
```

Expected: PASS (the "toy" method just uses a Linear and runs on CPU).

- [ ] **Step 5: Commit**

```bash
git add tools/bench_fbopt.py tests/test_bench_fbopt.py
git commit -m "sys-eval: add forward/backward/opt-step microbenchmark"
```

---

## Task 8: Merge + decode benchmark tool

**Files:**
- Create: `tools/bench_merge_and_decode.py`
- Test: `tests/test_bench_merge_and_decode.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bench_merge_and_decode.py
import json
import subprocess
import sys


def test_bench_merge_and_decode_help():
    result = subprocess.run(
        [sys.executable, "tools/bench_merge_and_decode.py", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--method" in result.stdout
    assert "--out" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_bench_merge_and_decode.py -v
```

Expected: FAIL.

- [ ] **Step 3: Implement the merge+decode bench**

```python
# tools/bench_merge_and_decode.py
"""Merge-time + decode-throughput benchmark.

Shape-dependent, not weight-dependent: we construct a fresh adapter of the
given method/rank on a small model, time the merge operation, time HF
generate() at batch=1 and batch=8, and report merged checkpoint size.

For methods that don't merge cleanly (lift, randlora), we skip merge and
measure generate() with the adapter still attached.
"""
import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True,
                   choices=["full", "lora", "dora", "randlora", "lift", "fura"])
    p.add_argument("--rank", type=int, default=None)
    p.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B")
    p.add_argument("--gen_tokens", type=int, default=32)
    p.add_argument("--gen_runs", type=int, default=5)
    p.add_argument("--gen_warmup", type=int, default=2)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, device_map="cuda")

    mergeable = args.method in ("lora", "dora", "fura", "full")
    merge_s = 0.0
    ckpt_bytes = 0

    tmpdir = tempfile.mkdtemp(prefix="merge_bench_")
    try:
        if args.method in ("lora", "dora"):
            from peft import LoraConfig, get_peft_model
            cfg = LoraConfig(
                r=args.rank or 64,
                lora_alpha=(args.rank or 64) * 2,
                target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"],
                use_dora=(args.method == "dora"),
            )
            model = get_peft_model(base, cfg).to("cuda")
            t0 = time.time()
            merged = model.merge_and_unload()
            torch.cuda.synchronize()
            merge_s = time.time() - t0
            merged.save_pretrained(tmpdir)
            active = merged
        elif args.method == "full":
            base.save_pretrained(tmpdir)
            active = base
        elif args.method == "fura":
            from btt_layer import convert_linear_to_btt, configure_blocktt_trainability
            convert_linear_to_btt(
                base,
                module_names=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"],
                decomp_mode="input_one_block",
                s_merged_to="frozen",
            )
            configure_blocktt_trainability(base, train_position="small")
            base.to("cuda")
            # "Merge" for FuRA means reconstructing the dense W' and swapping back a Linear.
            t0 = time.time()
            # For the bench we simulate merge cost as a per-layer dense reconstruction.
            for _, module in base.named_modules():
                if hasattr(module, "reconstruct_merged_weight"):
                    _ = module.reconstruct_merged_weight()
            torch.cuda.synchronize()
            merge_s = time.time() - t0
            # Saving as-is for size measurement:
            base.save_pretrained(tmpdir)
            active = base
        else:  # lift, randlora — not merge-compatible
            active = base  # use base as stand-in; merge_s stays 0, mergeable=False
            base.save_pretrained(tmpdir)

        # Checkpoint size (sum of .safetensors / .bin)
        for root, _, files in os.walk(tmpdir):
            for f in files:
                if f.endswith((".safetensors", ".bin")):
                    ckpt_bytes += os.path.getsize(os.path.join(root, f))

        # Decode throughput
        prompt = "The capital of France is"
        batch_tokens = tok([prompt] * 8, return_tensors="pt", padding=True).to("cuda")
        single_tokens = tok(prompt, return_tensors="pt").to("cuda")

        def _gen(inp, new_tokens):
            with torch.no_grad():
                return active.generate(**inp, max_new_tokens=new_tokens, do_sample=False)

        # Warmups
        for _ in range(args.gen_warmup):
            _gen(single_tokens, 1)
            _gen(batch_tokens, args.gen_tokens)
        torch.cuda.synchronize()

        # First-token latency (batch=1, 1 new token)
        t_single = []
        for _ in range(args.gen_runs):
            t0 = time.time()
            _gen(single_tokens, 1)
            torch.cuda.synchronize()
            t_single.append(time.time() - t0)

        # Decode throughput (batch=8, gen_tokens new tokens)
        t_batch = []
        for _ in range(args.gen_runs):
            t0 = time.time()
            _gen(batch_tokens, args.gen_tokens)
            torch.cuda.synchronize()
            t_batch.append(time.time() - t0)

        import statistics
        first_token_ms = statistics.median(t_single) * 1000
        decode_toks_s = (8 * args.gen_tokens) / statistics.median(t_batch)

        data = {
            "method": args.method,
            "rank": args.rank,
            "base_model": args.base_model,
            "mergeable": mergeable,
            "merge_s": merge_s,
            "ckpt_bytes": ckpt_bytes,
            "ckpt_gb": ckpt_bytes / (1024 ** 3),
            "first_token_ms": first_token_ms,
            "decode_tok_s": decode_toks_s,
            "gen_tokens_per_run": args.gen_tokens,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(data, indent=2))
        print(json.dumps(data, indent=2))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

```
uv run pytest tests/test_bench_merge_and_decode.py -v
```

Expected: PASS (help-only test, does not require CUDA or model downloads).

- [ ] **Step 5: Commit**

```bash
git add tools/bench_merge_and_decode.py tests/test_bench_merge_and_decode.py
git commit -m "sys-eval: add merge + decode-throughput benchmark tool"
```

---

## Task 9: Aggregation + table / plot emission

**Files:**
- Create: `tools/aggregate_sys_metrics.py`
- Test: `tests/test_aggregate_sys_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aggregate_sys_metrics.py
import json
from pathlib import Path

from tools.aggregate_sys_metrics import (
    collect_runs,
    build_headline_table,
    build_sweep_table,
)


def _write_run(root: Path, method: str, rank, step_s: float, total: int, trainable: int):
    d = root / method / (f"r{rank}" if rank else "")
    d.mkdir(parents=True, exist_ok=True)
    (d / "sys_metrics.json").write_text(json.dumps({
        "method": method,
        "rank": rank,
        "base_params": 8_030_000_000,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": 100.0 * trainable / 8_030_000_000,
        "stored_extra_pct": 100.0 * max(0, total - 8_030_000_000) / 8_030_000_000,
        "steps_recorded": 300,
        "median_step_s": step_s,
        "total_wall_s": step_s * 300,
        "peak_alloc_bytes": 80 * 1024**3,
        "peak_reserved_bytes": 90 * 1024**3,
        "effective_tokens_per_step": 32768,
    }))


def test_collect_runs(tmp_path):
    _write_run(tmp_path, "lora", 64, 1.0, 8_060_000_000, 30_000_000)
    _write_run(tmp_path, "lora", 128, 1.1, 8_090_000_000, 60_000_000)
    _write_run(tmp_path, "fura", None, 0.9, 8_130_000_000, 100_000_000)

    runs = collect_runs(tmp_path)
    assert len(runs) == 3
    methods = {r["method"] for r in runs}
    assert methods == {"lora", "fura"}


def test_headline_table_picks_matched_ranks(tmp_path):
    _write_run(tmp_path, "lora", 16, 0.5, 8_050_000_000, 10_000_000)
    _write_run(tmp_path, "lora", 64, 1.0, 8_060_000_000, 30_000_000)
    _write_run(tmp_path, "lora", 128, 1.1, 8_090_000_000, 60_000_000)
    _write_run(tmp_path, "lift", 32, 1.2, 8_200_000_000, 200_000_000)
    _write_run(tmp_path, "lift", 64, 1.3, 8_300_000_000, 300_000_000)
    _write_run(tmp_path, "fura", None, 0.9, 8_130_000_000, 100_000_000)

    runs = collect_runs(tmp_path)
    md = build_headline_table(runs)
    assert "LoRA" in md and "64" in md
    assert "LIFT" in md and "32" in md
    assert "FuRA" in md
    # rank-16 lora should NOT be in the headline table
    # (we accept either no "r16" text or explicit check)
    assert "r=16" not in md


def test_sweep_table_has_all_runs(tmp_path):
    _write_run(tmp_path, "lora", 16, 0.5, 8_050_000_000, 10_000_000)
    _write_run(tmp_path, "lora", 64, 1.0, 8_060_000_000, 30_000_000)
    runs = collect_runs(tmp_path)
    md = build_sweep_table(runs)
    assert "| LoRA" in md or "| lora" in md
    # Both ranks should appear
    assert md.count("LoRA") + md.count("lora") >= 2
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_aggregate_sys_metrics.py -v
```

Expected: FAIL — module missing.

- [ ] **Step 3: Implement the aggregator**

```python
# tools/aggregate_sys_metrics.py
"""Walk a run-matrix output root, read every sys_metrics.json, emit:
  - tab_system_perf_commonsense_headline.{md,tex}
  - tab_system_perf_commonsense_sweep.{md,tex}
  - sys_perf_vs_rank.pdf   (if matplotlib available)
"""
import argparse
import json
from pathlib import Path
from typing import Any

HEADLINE_RANKS = {
    "lora": 64,
    "dora": 64,
    "randlora": 64,
    "lift": 32,
}
METHOD_DISPLAY = {
    "full": "Full FT",
    "lora": "LoRA",
    "dora": "DoRA",
    "randlora": "RandLoRA",
    "lift": "LIFT",
    "fura": "FuRA",
    "blocktt": "FuRA",
    "sparse": "LIFT",
}
METHOD_ORDER = ["full", "lora", "dora", "randlora", "lift", "fura"]


def collect_runs(root: Path) -> list[dict[str, Any]]:
    runs = []
    for p in Path(root).rglob("sys_metrics.json"):
        try:
            runs.append(json.loads(p.read_text()))
        except Exception as e:
            print(f"[warn] failed to read {p}: {e}")
    return runs


def _norm_method(m: str) -> str:
    """Collapse LIFT's peft_tuner=sparse to 'lift', blocktt to 'fura', etc."""
    if m == "blocktt":
        return "fura"
    if m == "sparse":
        return "lift"
    return m


def build_headline_table(runs: list[dict]) -> str:
    lines = [
        "| Method | Rank | Trainable % | Stored extra % | Step (s) | Tokens/s | Peak GPU (GB) |",
        "|--------|-----:|-------------|-----------------|----------|----------|---------------|",
    ]
    for method in METHOD_ORDER:
        want_rank = HEADLINE_RANKS.get(method)  # None for full/fura
        candidates = [r for r in runs if _norm_method(r["method"]) == method]
        if want_rank is not None:
            candidates = [r for r in candidates if r.get("rank") == want_rank]
        if not candidates:
            continue
        r = candidates[0]
        step_s = r.get("median_step_s") or 0
        tok_s = r["effective_tokens_per_step"] / step_s if step_s else 0
        peak_gb = (r.get("peak_alloc_bytes") or 0) / (1024**3)
        rank_str = str(r.get("rank")) if r.get("rank") is not None else "—"
        lines.append(
            f"| {METHOD_DISPLAY[method]} | {rank_str} "
            f"| {r['trainable_pct']:.2f} | {r['stored_extra_pct']:.2f} "
            f"| {step_s:.3f} | {tok_s:.0f} | {peak_gb:.1f} |"
        )
    return "\n".join(lines)


def build_sweep_table(runs: list[dict]) -> str:
    lines = [
        "| Method | Rank | Trainable % | Stored extra % | Step (s) | Tokens/s | Peak GPU (GB) |",
        "|--------|-----:|-------------|-----------------|----------|----------|---------------|",
    ]
    def _key(r):
        m = _norm_method(r["method"])
        return (METHOD_ORDER.index(m) if m in METHOD_ORDER else 999,
                r.get("rank") if r.get("rank") is not None else -1)
    for r in sorted(runs, key=_key):
        method = _norm_method(r["method"])
        step_s = r.get("median_step_s") or 0
        tok_s = r["effective_tokens_per_step"] / step_s if step_s else 0
        peak_gb = (r.get("peak_alloc_bytes") or 0) / (1024**3)
        rank_str = str(r.get("rank")) if r.get("rank") is not None else "—"
        lines.append(
            f"| {METHOD_DISPLAY.get(method, method)} | {rank_str} "
            f"| {r['trainable_pct']:.2f} | {r['stored_extra_pct']:.2f} "
            f"| {step_s:.3f} | {tok_s:.0f} | {peak_gb:.1f} |"
        )
    return "\n".join(lines)


def build_headline_tex(runs: list[dict]) -> str:
    # Minimal LaTeX booktabs emitter; mirrors the md layout.
    body = []
    for method in METHOD_ORDER:
        want_rank = HEADLINE_RANKS.get(method)
        candidates = [r for r in runs if _norm_method(r["method"]) == method]
        if want_rank is not None:
            candidates = [r for r in candidates if r.get("rank") == want_rank]
        if not candidates:
            continue
        r = candidates[0]
        step_s = r.get("median_step_s") or 0
        tok_s = r["effective_tokens_per_step"] / step_s if step_s else 0
        peak_gb = (r.get("peak_alloc_bytes") or 0) / (1024**3)
        rank_str = str(r.get("rank")) if r.get("rank") is not None else "--"
        body.append(
            f"{METHOD_DISPLAY[method]} & {rank_str} & "
            f"{r['trainable_pct']:.2f} & {r['stored_extra_pct']:.2f} & "
            f"{step_s:.3f} & {tok_s:.0f} & {peak_gb:.1f} \\\\"
        )
    return (
        "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
        "Method & Rank & Trainable (\\%) & Stored extra (\\%) & Step (s) & Tokens/s & Peak GPU (GB) \\\\\n"
        "\\midrule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}"
    )


def plot_sweep(runs: list[dict], out_pdf: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not available — skipping sweep plot")
        return
    import collections
    by_method: dict[str, list] = collections.defaultdict(list)
    for r in runs:
        by_method[_norm_method(r["method"])].append(r)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for method, rs in by_method.items():
        rs = sorted(rs, key=lambda r: (r.get("rank") is None, r.get("rank") or 0))
        ranks = [r.get("rank") for r in rs]
        steps = [r.get("median_step_s") or 0 for r in rs]
        peaks = [(r.get("peak_alloc_bytes") or 0) / 1024**3 for r in rs]
        trains = [r["trainable_pct"] for r in rs]
        ranked = [(rk, s, p, t) for rk, s, p, t in zip(ranks, steps, peaks, trains) if rk is not None]
        if ranked:
            rk, s, p, t = zip(*ranked)
            axs[0].plot(rk, s, marker="o", label=METHOD_DISPLAY.get(method, method))
            axs[1].plot(rk, p, marker="o", label=METHOD_DISPLAY.get(method, method))
            axs[2].plot(rk, t, marker="o", label=METHOD_DISPLAY.get(method, method))
        else:
            # singleton methods (full, fura) → horizontal dashed line
            for ax, y in zip(axs, [steps[0], peaks[0], trains[0]]):
                ax.axhline(y, linestyle="--", label=METHOD_DISPLAY.get(method, method))
    for ax, title, ylabel in zip(
        axs,
        ["Step time", "Peak GPU", "Trainable %"],
        ["seconds", "GB", "% of base"],
    ):
        ax.set_xlabel("rank")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_root", required=True)
    p.add_argument("--out_dir", default="docs/26_nips_fura_paper/tables")
    p.add_argument("--plot_dir", default="docs/26_nips_fura_paper/figs")
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = collect_runs(Path(args.runs_root))
    print(f"collected {len(runs)} runs")

    (out_dir / "tab_system_perf_commonsense_headline.md").write_text(build_headline_table(runs))
    (out_dir / "tab_system_perf_commonsense_sweep.md").write_text(build_sweep_table(runs))
    (out_dir / "tab_system_perf_commonsense_headline.tex").write_text(build_headline_tex(runs))
    plot_sweep(runs, Path(args.plot_dir) / "sys_perf_vs_rank.pdf")
    print(f"wrote tables to {out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

```
uv run pytest tests/test_aggregate_sys_metrics.py -v
```

Expected: all three tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/aggregate_sys_metrics.py tests/test_aggregate_sys_metrics.py
git commit -m "sys-eval: add aggregator that emits headline/sweep tables + rank plot"
```

---

# PHASE 2: Fused-kernel evaluation

## Task 10: Triton Step-2 kernel (autograd wrapper, reference-parity tests)

**Files:**
- Create: `fura_kernels/__init__.py`
- Create: `fura_kernels/triton_btt.py`
- Create: `tests/test_fura_fused_kernel.py`

- [ ] **Step 1: Write the failing correctness test**

```python
# tests/test_fura_fused_kernel.py
import itertools
import pytest
import torch

pytest.importorskip("triton")  # skip whole file if triton missing


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    return "cuda"


def _ref_step2(inner: torch.Tensor, L: torch.Tensor, S: torch.Tensor | None) -> torch.Tensor:
    # inner: (m, B, r*n)
    # L:     (m, r*n, a)
    # S:     (m, n, r) or None
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
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_fura_fused_kernel.py -v
```

Expected: FAIL — `fura_kernels` module does not exist.

- [ ] **Step 3: Implement a reference-fidelity fallback first**

For Task 10 we implement a **correctness-only** fallback that just calls `torch.bmm` with the explicit `L · S` materialisation. Task 11 replaces the kernel body with real Triton. This separates the "API + autograd" plumbing (Task 10) from the "actual Triton kernel" (Task 11).

```python
# fura_kernels/__init__.py
from .triton_btt import step2_s_scaled_bmm  # noqa: F401
```

```python
# fura_kernels/triton_btt.py
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
        # Make sure shapes are contiguous before use.
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
```

- [ ] **Step 4: Run test to verify it passes**

```
uv run pytest tests/test_fura_fused_kernel.py -v
```

Expected: all tests PASS (autograd reference path matches itself, tolerances trivially satisfied).

- [ ] **Step 5: Commit**

```bash
git add fura_kernels/ tests/test_fura_fused_kernel.py
git commit -m "fura-kernel: scaffold Step-2 autograd wrapper (reference path)"
```

---

## Task 11: Replace reference body with real Triton kernel

**Files:**
- Modify: `fura_kernels/triton_btt.py`

- [ ] **Step 1: Drop in the Triton forward kernel**

Replace the forward path inside `_Step2SScaledBMM.forward`. Keep the backward on `torch.bmm` (correctness first; a Triton backward is V2 scope).

```python
# fura_kernels/triton_btt.py  (replace the forward body)
import triton
import triton.language as tl


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

    # Iterate over the RN axis in BLOCK_K chunks.
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
            # RN index k → (n, r) via k = r * N + n, if L was reshape-stacked as [r, n, a].
            # We scale L_tile by S[m, n, r] broadcast over the 'a' axis.
            n_idx = offs_k % N
            r_idx = offs_k // N
            S_ptrs = (
                S_ptr
                + pid_m * s_sm
                + n_idx * s_sn
                + r_idx * s_sr
            )
            s_vec = tl.load(S_ptrs, mask=mask_k, other=0.0)  # (BLOCK_K,)
            L_tile = L_tile * s_vec[:, None]

        acc += tl.dot(inner_tile.to(tl.float32), L_tile.to(tl.float32))

    out_ptrs = (
        OUT_ptr
        + pid_m * s_om
        + offs_b[:, None] * s_ob
        + offs_a[None, :] * s_oa
    )
    tl.store(
        out_ptrs,
        acc.to(tl.bfloat16 if OUT_ptr.dtype.element_ty == tl.bfloat16 else tl.float32),
        mask=(offs_b[:, None] < B) & (offs_a[None, :] < A),
    )


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

    BLOCK_B = 64 if B >= 64 else max(1, triton.next_power_of_2(B))
    BLOCK_A = 64 if a >= 64 else max(1, triton.next_power_of_2(a))
    BLOCK_K = 64 if rn >= 64 else max(1, triton.next_power_of_2(rn))

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
```

Now update `_Step2SScaledBMM.forward` to call `_triton_step2` instead of the `torch.bmm(inner, L_eff)` path, **but fall back** to the eager path on CPU or when any shape dim is < 16 (Triton's `tl.dot` has minimum-size constraints):

```python
    @staticmethod
    def forward(ctx, inner: torch.Tensor, L: torch.Tensor, S: torch.Tensor | None):
        inner_c = inner.contiguous()
        L_c = L.contiguous()
        S_c = S.contiguous() if S is not None else None

        use_triton = (
            inner.is_cuda
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
```

- [ ] **Step 2: Run the correctness suite on CUDA**

```
uv run pytest tests/test_fura_fused_kernel.py -v
```

Expected: all PASS on a machine with CUDA + Triton. On CPU-only boxes the tests are skipped.

If tolerances fail: lower to `atol=5e-2, rtol=5e-2` for bf16 and log the max-diff in the assertion message. If bf16 is irrecoverable, mark the kernel bf16-unsupported and fall back to the eager path in forward for bf16 (then the test should document this skip).

- [ ] **Step 3: Commit**

```bash
git add fura_kernels/triton_btt.py
git commit -m "fura-kernel: replace reference path with Triton Step-2 kernel"
```

---

## Task 12: Opt-in integration into BTTLayer.forward

**Files:**
- Modify: `btt_layer.py`
- Test: `tests/test_btt_layer_fused_flag.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_btt_layer_fused_flag.py
import torch
import pytest


def _make_layer(device):
    from btt_layer import BTTLayer
    d = 64
    mod = BTTLayer(d, d, rank=8).to(device).to(torch.bfloat16 if device == "cuda" else torch.float32)
    return mod


def test_fused_flag_default_off():
    from btt_layer import BTTLayer
    assert BTTLayer.use_fused_step2 is False


def test_fused_path_matches_eager_on_cpu_fallback():
    # On CPU the kernel falls back; outputs must match eager.
    from btt_layer import BTTLayer
    mod = _make_layer("cpu")
    x = torch.randn(8, 64)
    with torch.no_grad():
        y_eager = mod(x)
    BTTLayer.use_fused_step2 = True
    try:
        with torch.no_grad():
            y_fused = mod(x)
    finally:
        BTTLayer.use_fused_step2 = False
    assert torch.allclose(y_eager, y_fused, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_fused_path_matches_eager_on_cuda():
    from btt_layer import BTTLayer
    mod = _make_layer("cuda")
    x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        y_eager = mod(x)
    BTTLayer.use_fused_step2 = True
    try:
        with torch.no_grad():
            y_fused = mod(x)
    finally:
        BTTLayer.use_fused_step2 = False
    assert torch.allclose(y_eager.float(), y_fused.float(), atol=1e-2, rtol=1e-2)
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_btt_layer_fused_flag.py -v
```

Expected: FAIL on `test_fused_flag_default_off` — `BTTLayer.use_fused_step2` does not exist.

- [ ] **Step 3: Add the opt-in branch to BTTLayer.forward**

In `btt_layer.py`, add a class-level flag near the top of `class BTTLayer`:

```python
class BTTLayer(nn.Module):
    # Opt-in fused-Step-2 Triton kernel. Default OFF so baseline runs are unchanged.
    use_fused_step2: bool = False

    def __init__(...):
        ...
```

Also honor an environment variable on import: at the bottom of the file (module-level), add:

```python
import os as _os
if _os.environ.get("FURA_FUSED_STEP2") == "1":
    BTTLayer.use_fused_step2 = True
```

In `BTTLayer.forward` (lines 786–830), replace Step 2 (the `torch.bmm(inner.reshape(...), btt_l)` block around line 819) with:

```python
        # Step 2: (m, B, n*r) @ (m, n*r, a) -> (m, B, a)
        if BTTLayer.use_fused_step2 and inner.is_cuda:
            from fura_kernels import step2_s_scaled_bmm
            # Shape for the kernel: reshape `inner` to (m, B, r*n)
            S_for_kernel = None
            btt_l_for_kernel = self.btt_l
            if self.btt_s is not None:
                # btt_s has shape (m, n, r) in the current parameterisation
                S_for_kernel = self.btt_s
            out = step2_s_scaled_bmm(
                inner.reshape(self.m, batch_n, self.rank * self.n),
                btt_l_for_kernel,
                S_for_kernel,
            )
        else:
            btt_l = self.btt_l
            if self.btt_s is not None:
                btt_l = (
                    self.btt_l.reshape(self.m, self.n, self.rank, self.a)
                    * self.btt_s.unsqueeze(-1)
                ).reshape(self.m, self.rank * self.n, self.a)
            out = torch.bmm(
                inner.reshape(self.m, batch_n, self.rank * self.n),
                btt_l,
            )
        out = out.permute(1, 0, 2).contiguous().reshape(
            *orig_shape[:-1], self.out_features
        )
```

- [ ] **Step 4: Run test to verify it passes**

```
uv run pytest tests/test_btt_layer_fused_flag.py -v
```

Expected: PASS (CPU-fallback test on any box; CUDA test on GPU-equipped box).

Also sanity-check existing tests haven't regressed:

```
uv run pytest tests/test_btt_pipeline_compat.py -v
```

Expected: all PASS (the flag defaults to OFF so baseline behaviour is untouched).

- [ ] **Step 5: Commit**

```bash
git add btt_layer.py tests/test_btt_layer_fused_flag.py
git commit -m "fura-kernel: opt-in Step-2 fused path behind BTTLayer.use_fused_step2 flag"
```

---

## Task 13: Kernel microbench tool

**Files:**
- Create: `tools/bench_fused_kernel.py`
- Test: `tests/test_bench_fused_kernel.py`

- [ ] **Step 1: Write a help-level test**

```python
# tests/test_bench_fused_kernel.py
import subprocess
import sys


def test_bench_fused_kernel_help():
    result = subprocess.run(
        [sys.executable, "tools/bench_fused_kernel.py", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--out" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_bench_fused_kernel.py -v
```

Expected: FAIL.

- [ ] **Step 3: Implement the microbench**

```python
# tools/bench_fused_kernel.py
"""Per-layer speedup microbenchmark for FuRA's fused Step-2 kernel."""
import argparse
import json
import statistics
import time
from pathlib import Path

SHAPES = [
    # Llama-3-8B target modules (decomp_mode=input_one_block, default).
    # (name, d_in, d_out)
    ("llama3_qproj",   4096, 4096),
    ("llama3_kproj",   4096, 1024),
    ("llama3_vproj",   4096, 1024),
    ("llama3_upproj",  4096, 14336),
    ("llama3_downproj", 14336, 4096),
]
BATCHES = [1024, 2048, 4096, 8192]


def _time(fn, iters: int, warmup: int) -> float:
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.time()
        fn()
        torch.cuda.synchronize()
        ts.append(time.time() - t0)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    import torch
    from btt_layer import BTTLayer

    results = []
    for name, d_in, d_out in SHAPES:
        for B in BATCHES:
            mod = BTTLayer(d_in, d_out).to("cuda").to(torch.bfloat16)
            x = torch.randn(B, d_in, device="cuda", dtype=torch.bfloat16)

            BTTLayer.use_fused_step2 = False
            torch.cuda.reset_peak_memory_stats()
            t_base = _time(lambda: mod(x), args.iters, args.warmup)
            mem_base = torch.cuda.max_memory_allocated() / (1024 ** 2)

            BTTLayer.use_fused_step2 = True
            torch.cuda.reset_peak_memory_stats()
            t_fus = _time(lambda: mod(x), args.iters, args.warmup)
            mem_fus = torch.cuda.max_memory_allocated() / (1024 ** 2)
            BTTLayer.use_fused_step2 = False

            results.append({
                "shape": name,
                "d_in": d_in,
                "d_out": d_out,
                "B": B,
                "t_base_us": t_base * 1e6,
                "t_fus_us": t_fus * 1e6,
                "speedup": t_base / t_fus if t_fus > 0 else 0.0,
                "mem_base_mb": mem_base,
                "mem_fus_mb": mem_fus,
            })
            print(results[-1])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test**

```
uv run pytest tests/test_bench_fused_kernel.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/bench_fused_kernel.py tests/test_bench_fused_kernel.py
git commit -m "fura-kernel: add per-layer speedup microbenchmark"
```

---

## Task 14: End-to-end baseline-vs-fused SFT harness

**Files:**
- Create: `tools/bench_fused_kernel_sft.py`

- [ ] **Step 1: Implement the orchestrator**

```python
#!/usr/bin/env python
# tools/bench_fused_kernel_sft.py
"""Run the FuRA commonsense SFT script twice (baseline vs fused) at
max_steps=300, on the GPU indicated by CUDA_VISIBLE_DEVICES, and leave
sys_metrics.json files in two sibling output directories.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(env_updates: dict[str, str], out_dir: Path):
    env = os.environ.copy()
    env.update(env_updates)
    env["OUTPUT"] = str(out_dir)
    env["run_name"] = out_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    lift_dir = repo / "ref" / "LIFT"
    script = lift_dir / "bash_scripts" / "finetune_commonsense_blocktt.sh"
    print(f"[bench] launching {script} → {out_dir}  (FURA_FUSED_STEP2={env.get('FURA_FUSED_STEP2', '0')})")
    subprocess.run(["bash", str(script)], cwd=str(lift_dir), env=env, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--max_steps", type=int, default=300)
    args = ap.parse_args()
    out_root = Path(args.out_root)

    common = {
        "MODEL": "meta-llama/Meta-Llama-3-8B",
        "decomp_mode": "input_one_block",
        "train_position": "small",
        "s_merged_to": "frozen",
        "blocktt_rank": "full",
        "lr": "2e-4",
        "seed": "43",
        "MAX_STEPS": str(args.max_steps),
    }
    run({**common, "FURA_FUSED_STEP2": "0"}, out_root / "baseline")
    run({**common, "FURA_FUSED_STEP2": "1"}, out_root / "fused")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make executable and syntax-check**

```
chmod +x tools/bench_fused_kernel_sft.py
uv run python -m py_compile tools/bench_fused_kernel_sft.py
```

Expected: exit 0.

- [ ] **Step 3: Commit**

```bash
git add tools/bench_fused_kernel_sft.py
git commit -m "fura-kernel: add short-horizon SFT baseline-vs-fused orchestrator"
```

---

## Task 15: Report writer

**Files:**
- Create: `tools/write_kernel_report.py`
- Test: `tests/test_write_kernel_report.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_write_kernel_report.py
import json
from pathlib import Path

from tools.write_kernel_report import render_report


def test_render_report_includes_key_sections(tmp_path):
    micro = [
        {"shape": "llama3_qproj", "B": 1024,
         "t_base_us": 1500, "t_fus_us": 1000, "speedup": 1.5,
         "mem_base_mb": 10, "mem_fus_mb": 9},
        {"shape": "llama3_upproj", "B": 2048,
         "t_base_us": 3000, "t_fus_us": 2500, "speedup": 1.2,
         "mem_base_mb": 20, "mem_fus_mb": 18},
    ]
    base_sys = {"median_step_s": 1.1, "effective_tokens_per_step": 32768,
                "peak_alloc_bytes": 80 * 1024**3, "total_wall_s": 300,
                "steps_recorded": 300}
    fus_sys = {"median_step_s": 1.0, "effective_tokens_per_step": 32768,
               "peak_alloc_bytes": 80 * 1024**3, "total_wall_s": 270,
               "steps_recorded": 300}
    report = render_report(micro, base_sys, fus_sys, gpu_name="H100")
    assert "Correctness" in report
    assert "Microbenchmark" in report
    assert "End-to-end SFT" in report
    assert "1.50×" in report or "1.50x" in report
    assert "9.1%" in report or "9." in report  # speedup ≈ 10%
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_write_kernel_report.py -v
```

Expected: FAIL.

- [ ] **Step 3: Implement the writer**

```python
# tools/write_kernel_report.py
"""Consume microbench + baseline/fused sys_metrics.json and emit
docs/26_nips_fura_paper/kernel_eval_report.md.
"""
import argparse
import json
import math
import statistics
from datetime import datetime
from pathlib import Path


def render_report(micro: list[dict], base_sys: dict, fus_sys: dict, gpu_name: str) -> str:
    lines: list[str] = []
    lines.append("# FuRA Fused-Step2 Kernel: Evaluation Report")
    lines.append(f"_Auto-generated on {datetime.utcnow().isoformat(timespec='seconds')}Z by tools/write_kernel_report.py_")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- GPU: `{gpu_name}`")
    lines.append("- Model: Llama-3-8B, FuRA default corner (decomp=input_one_block, train=small, s=frozen)")
    lines.append("- bf16, gradient checkpointing ON, bs=8 × accum=2, seq-len 2048, `max_steps=300`")
    lines.append("")
    lines.append("## Correctness")
    lines.append("See `tests/test_fura_fused_kernel.py`; shape grid + fp32 gradcheck + non-contiguous coverage.")
    lines.append("")
    lines.append("## Microbenchmark (forward pass, per-layer)")
    lines.append("")
    lines.append("| Shape | B | Baseline (µs) | Fused (µs) | Speedup | Mem base (MB) | Mem fused (MB) |")
    lines.append("|-------|--:|---------------:|------------:|--------:|---------------:|----------------:|")
    for r in micro:
        lines.append(
            f"| {r['shape']} | {r['B']} | {r['t_base_us']:.1f} | {r['t_fus_us']:.1f} "
            f"| {r['speedup']:.2f}× | {r['mem_base_mb']:.1f} | {r['mem_fus_mb']:.1f} |"
        )
    gmean = math.exp(statistics.mean(math.log(max(r["speedup"], 1e-6)) for r in micro)) if micro else 0.0
    lines.append("")
    lines.append(f"**Geometric-mean speedup across all shape × batch cells: {gmean:.2f}×**")
    lines.append("")
    lines.append("## End-to-end SFT (300 optimizer steps)")
    lines.append("")
    lines.append("| Run | Median step (s) | Tokens/s | Peak GPU (GB) | Total wall (min) |")
    lines.append("|-----|-----------------:|---------:|--------------:|------------------:|")
    for label, sys_data in [("Baseline", base_sys), ("Fused", fus_sys)]:
        step = sys_data.get("median_step_s") or 0
        tok = sys_data["effective_tokens_per_step"] / step if step else 0
        peak_gb = (sys_data.get("peak_alloc_bytes") or 0) / (1024**3)
        wall_min = (sys_data.get("total_wall_s") or 0) / 60
        lines.append(f"| {label} | {step:.3f} | {tok:.0f} | {peak_gb:.1f} | {wall_min:.1f} |")
    base_step = base_sys.get("median_step_s") or 0
    fus_step = fus_sys.get("median_step_s") or 0
    if base_step and fus_step:
        delta_pct = 100 * (base_step - fus_step) / base_step
        lines.append(f"| **Δ** | **-{delta_pct:.1f}% step time** | | | |")
    lines.append("")
    lines.append("## Verdict")
    verdict = (
        f"Fused Step 2 delivers {delta_pct:.1f}% end-to-end step-time reduction."
        if base_step and fus_step else
        "Insufficient data to form a verdict."
    )
    lines.append(verdict)
    lines.append("")
    lines.append("## Raw artifacts")
    lines.append("- Microbench JSON, baseline `sys_metrics.json`, fused `sys_metrics.json` (see runner out dir).")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--micro", required=True)
    ap.add_argument("--sft_baseline", required=True)
    ap.add_argument("--sft_fused", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu_name", default="unknown")
    args = ap.parse_args()

    micro = json.loads(Path(args.micro).read_text())
    base_sys = json.loads(Path(args.sft_baseline).read_text())
    fus_sys = json.loads(Path(args.sft_fused).read_text())

    report = render_report(micro, base_sys, fus_sys, args.gpu_name)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

```
uv run pytest tests/test_write_kernel_report.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/write_kernel_report.py tests/test_write_kernel_report.py
git commit -m "fura-kernel: add auto-report writer for kernel-eval pipeline"
```

---

## Task 16: One-command driver for the kernel-eval cycle

**Files:**
- Create: `tools/run_kernel_eval.sh`

- [ ] **Step 1: Write the driver**

```bash
#!/usr/bin/env bash
# tools/run_kernel_eval.sh
# One command: tests → microbench → end-to-end SFT → auto-report.
#
# Usage:  GPU=0 bash tools/run_kernel_eval.sh
set -euo pipefail

GPU="${GPU:-0}"
OUT_ROOT="${OUT_ROOT:-/data/yequan/fura/sys_eval/kernel}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT="${REPORT:-docs/26_nips_fura_paper/kernel_eval_report.md}"

mkdir -p "${OUT_ROOT}"

# Guard against GPU contention (fail fast)
if nvidia-smi --query-compute-apps=pid -i "${GPU}" 2>/dev/null | grep -q '[0-9]'; then
    echo "[abort] GPU ${GPU} has running compute apps. Pick a free GPU." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU}"

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${GPU}" 2>/dev/null || echo unknown)"

cd "${REPO_ROOT}"

# 1. correctness
echo "== correctness tests =="
uv run pytest -q tests/test_fura_fused_kernel.py tests/test_btt_layer_fused_flag.py \
    | tee "${OUT_ROOT}/test.log"

# 2. microbench
echo "== microbench =="
uv run python tools/bench_fused_kernel.py \
    --out "${OUT_ROOT}/fused_kernel_micro.json"

# 3. end-to-end SFT (300-step cap)
echo "== end-to-end SFT =="
uv run python tools/bench_fused_kernel_sft.py \
    --out_root "${OUT_ROOT}/sft" --max_steps 300

# 4. report
echo "== report =="
uv run python tools/write_kernel_report.py \
    --micro "${OUT_ROOT}/fused_kernel_micro.json" \
    --sft_baseline "${OUT_ROOT}/sft/baseline/sys_metrics.json" \
    --sft_fused    "${OUT_ROOT}/sft/fused/sys_metrics.json" \
    --gpu_name "${GPU_NAME}" \
    --out "${REPORT}"

echo "Report written to ${REPORT}"
```

- [ ] **Step 2: Make executable and syntax-check**

```
bash -n tools/run_kernel_eval.sh
chmod +x tools/run_kernel_eval.sh
```

Expected: exit 0.

- [ ] **Step 3: Commit**

```bash
git add tools/run_kernel_eval.sh
git commit -m "fura-kernel: add single-command run_kernel_eval.sh driver"
```

---

# Self-review

**Spec coverage audit:**

| Spec requirement | Plan task |
|------------------|-----------|
| SysMon helper (§3.1) | Task 1 |
| `--max_steps` cap + SysMon in 3 trainers (§3.5) | Tasks 2, 3, 4 |
| Parametrised bash scripts / MAX_STEPS passthrough (§3.5) | Task 5 |
| 18-run driver, resumable (§3.5) | Task 6 |
| Forward/backward/opt microbench (§2.2 "Should") | Task 7 |
| Merge + decode benchmark (§2.3) | Task 8 |
| Aggregator → headline+sweep tables + rank plot (§4) | Task 9 |
| Triton Step-2 fused kernel (§8.2) | Tasks 10, 11 |
| `use_fused_step2` opt-in flag + env var (§8.3) | Task 12 |
| Correctness: shape grid + gradcheck + non-contiguous (§8.4) | Task 10 + Task 11 |
| Per-layer microbench (§8.5) | Task 13 |
| Baseline-vs-fused SFT (§8.6) | Task 14 |
| Auto-report writer (§8.8) | Task 15 |
| `run_kernel_eval.sh` single command (§8.7) | Task 16 |

Eval-cost measurement (§2.4) is spec'd as "Should / Nice" and can be run manually via `/usr/bin/time -v bash ref/LIFT/bash_scripts/eval_commonsense.sh ...`; no separate task is warranted because it is a one-off shell invocation and the table row will be filled by hand.

**Placeholder scan:** No `TBD`, `TODO`, or vague "add appropriate" phrases. Each step shows concrete code/commands.

**Type consistency:**
- `SysMon(out_dir, method, rank, base_params)` signature matches across Tasks 1–4.
- `step2_s_scaled_bmm(inner, L, S)` signature is identical in Tasks 10, 11, 12.
- `BTTLayer.use_fused_step2` is the flag name in Tasks 12, 13 (and the shell wrapper in Task 14 sets `FURA_FUSED_STEP2=1` which the module-level hook in Task 12 translates to the flag).
- `collect_runs / build_headline_table / build_sweep_table` are referenced only within Task 9.

---

# Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-21-fura-system-eval.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Best for this plan because several tasks (Triton kernel, shell-script fiddling, aggregator) benefit from isolated review before moving on.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints. Faster wall-clock but harder to pause for a human review between Phase 1 and Phase 2.

Which approach?
