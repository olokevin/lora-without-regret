# FuRA System-Performance Evaluation on LIFT Commonsense

**Date:** 2026-04-21
**Scope:** Measure the *system* cost of FuRA vs. Full FT / LoRA / DoRA / RandLoRA / LIFT on the **SFT-only** commonsense-reasoning path in `ref/LIFT/`. Accuracy is already tabulated in the paper (Table `tab:commonsense_sft`); this document is about wall-clock, memory, throughput, parameter counts, and deployment cost. RL/GRPO is **out of scope** for this document.

Paper reference: `docs/26_nips_fura_paper/neurips_2026.tex` §3.3 *Overhead analysis* (Table `tab:cost`, at $d{=}4096$ / Qwen3-1.7B / GRPO). That table reports four numbers per method: trainable %, stored-extra %, full-rank-$\Delta\mat{W}$? yes/no, wall-clock per step. We generalise that to a reproducible commonsense-SFT protocol and add a **rank sweep** so the cost story is not tied to one operating point.

---

## 1. Methods under test and run matrix

All methods already have a launch script under `ref/LIFT/bash_scripts/`.

### 1.1 Methods

| Method        | Script                                | Adapter wiring                                                                                         |
|---------------|----------------------------------------|---------------------------------------------------------------------------------------------------------|
| Full FT       | `finetune_commonsense_full.sh`         | `src/finetune_sft.py`, bs=1 / accum=16                                                                  |
| LoRA          | `finetune_commonsense_lora.sh`         | `src/finetune_lora.py`, `adapter_name=lora`, $\alpha{=}2r$                                              |
| DoRA          | (reuse LoRA script, `adapter_name=dora`) | `src/finetune_lora.py` routes `dora` → PEFT `use_dora=True` (line 444), $\alpha{=}2r$                  |
| RandLoRA      | `finetune_commonsense_randlora.sh`     | `src/finetune_lora.py`, `adapter_name=randlora`, shared frozen random projection, $\alpha$ re-tuned per rank (see §1.3) |
| LIFT          | `finetune_commonsense_lift.sh`         | `src/finetune_sft.py` + sparse tuner, top-$k$ filter, interval-based mask refresh                        |
| **FuRA**      | `finetune_commonsense_blocktt.sh`      | `src/finetune_blocktt.py`, default corner: `decomp_mode=input_one_block`, `train_position=small`, `s_merged_to=frozen` |

### 1.2 Rank sweep

Adapter-style methods (LoRA, DoRA, RandLoRA, LIFT) are run at **four ranks each**: $r \in \{16, 32, 64, 128\}$. Full FT and FuRA are run **once** (no rank to sweep — FuRA uses its default balanced parameterisation $n{=}b{=}r{=}\sqrt{d}$, which at Llama-3-8B hidden $d{=}4096$ gives $\sqrt{d}{\approx}64$).

Total runs: $4 \text{ methods} \times 4 \text{ ranks} + 2 \text{ singletons} = \mathbf{18 \text{ SFT runs}}$.

### 1.3 Per-method hyperparameter map

The LR and alpha defaults used today are already tuned (the paper's accuracy table uses these). We hold LR fixed per method and sweep only rank. Where the method's convention ties $\alpha$ to $r$, follow it.

| Method       | Ranks                 | $\alpha$ per rank               | Learning rate | Notes                                                                   |
|--------------|-----------------------|----------------------------------|---------------|-------------------------------------------------------------------------|
| LoRA         | 16 / 32 / 64 / 128    | 32 / 64 / 128 / 256 ($=2r$)      | 2e-4          | `target_modules = q_proj k_proj v_proj up_proj down_proj`                |
| DoRA         | 16 / 32 / 64 / 128    | 32 / 64 / 128 / 256 ($=2r$)      | 2e-4          | same target modules; `use_dora=True`                                     |
| RandLoRA     | 16 / 32 / 64 / 128    | 320 / 640 / 1280 / 2560 ($=20r$) | 2e-4          | current default is $r{=}32, \alpha{=}640$; scale $\alpha$ linearly with $r$ to keep effective init magnitude |
| LIFT         | 16 / 32 / 64 / 128    | n/a                              | 2e-4          | `filter_rank=rank` (same knob); `update_interval` at default              |
| Full FT      | —                     | —                                | 2e-4          | bs=1 × accum=16 (memory-bound)                                           |
| **FuRA**     | — (default corner)    | —                                | 2e-4          | $n{=}b{=}r{=}64$ implied by $\sqrt{d}$ at Llama-3-8B                      |

All 18 runs share the common envelope: Meta-Llama-3-8B, per-device bs=8 × accum=2 (except Full FT: bs=1 × accum=16), bf16, AdamW, gradient checkpointing ON, target modules `q_proj,k_proj,v_proj,up_proj,down_proj`, seq-len 2048, single H100, seed 43.

**Short-horizon system-only protocol.** Every run is capped at **300 optimizer steps** (100-step warmup + 200 measurement steps). This is *not* a full training run — it is a system microbenchmark that uses the real training loop so the measured numbers include real data loading, real optimizer state, real gradient checkpointing, and the real model shape. System metrics (step time, tokens/s, peak GPU, optimizer-state footprint) are steady-state quantities, so 200 steady-state steps give the same number a 3-epoch run would give, at ~0.5–2 H100-hours per run instead of 10–80 h. **Accuracy is not re-measured** — it is read from the paper's existing Table 1, which already used 3-epoch training under this exact protocol. The entire 18-run Phase 1 finishes in under one H100-day.

### 1.4 Headline comparison point

The paper's §3.3 story is most naturally framed at the **matched-parameter operating point**: rank-64 for LoRA/DoRA/RandLoRA, rank-32 for LIFT, vs. the single FuRA default. That selection is the headline Table in §5.1; the rest of the sweep feeds the rank-vs-cost curves in §5.5.

---

## 2. Metric shortlist

We partition system metrics into **four** groups. "Must" = required for the paper's §3.3 story; "Should" = strengthens the deployment narrative; "Nice" = extra diagnostic value.

### 2.1 Parameter footprint (static — read once per method)

| Metric                                  | Definition                                                                 | Priority |
|-----------------------------------------|----------------------------------------------------------------------------|----------|
| **Trainable params** (abs, %)           | `sum(p.numel() for p in model.parameters() if p.requires_grad)` after adapter attach. % is vs. base Llama-3-8B (~8.03B). | Must     |
| **Stored-extra params** (abs, %)        | Trainable + frozen *added* params (e.g. FuRA's frozen $\mat L$, LIFT's mask metadata, RandLoRA's frozen random projection). Computed as `total_params_after_attach − base_params`. | Must     |
| **Full-rank $\Delta \mat W$?**          | Yes/No flag per method (Full FT / LIFT / FuRA: Yes; LoRA / DoRA / RandLoRA: No). Hard-coded in the results writer. | Must     |
| Optimizer-state footprint               | For AdamW: `2 × sizeof(bf16 or fp32) × |trainable|`. Deterministic from #1. | Should   |

### 2.2 Training wall-clock and memory (measured during the real SFT run)

| Metric                                  | How to measure                                                                                                 | Priority |
|-----------------------------------------|----------------------------------------------------------------------------------------------------------------|----------|
| **Steps/sec (steady state)**            | Take the median of last 200 logged `train/step_time` values after a 100-step warmup.                           | Must     |
| **Tokens/sec**                          | `steps_per_sec × per_device_bs × grad_accum × seq_len` (effective tokens, not padded).                         | Must     |
| **Time per epoch** / **total train time** | Wall-clock from first optimizer step to last, logged from `finetune_*.py`.                                   | Must     |
| **Peak GPU memory (train)**             | `torch.cuda.max_memory_allocated()` + `torch.cuda.max_memory_reserved()` captured right before `save_model`.   | Must     |
| **Activation memory delta**             | Peak memory minus (params + optim state + KV cache estimate). Diagnostic — helps explain why FuRA is cheaper than DoRA. | Should |
| **Forward-only time per step**          | Timer around `model(**batch)` in a separate 20-step micro-benchmark before training. Useful to isolate adapter overhead from backward cost. | Should |
| **Backward time per step**              | Same as above, backward only.                                                                                  | Should   |
| **Optimizer-step time**                 | Timer around `optimizer.step()`; tiny for LoRA, medium for FuRA, largest for Full FT.                          | Nice     |

### 2.3 Inference/deployment wall-clock (post-merge)

Rationale: the paper's "Deployment" claim is that FuRA re-merges to a dense $\mat W'$ with no serving overhead. LoRA/DoRA merge likewise; LIFT/RandLoRA/S$^2$FT do not (or have trade-offs). Measure:

| Metric                                  | How to measure                                                                                                 | Priority |
|-----------------------------------------|----------------------------------------------------------------------------------------------------------------|----------|
| **Merge time**                          | Wall-clock of `model.merge_and_unload()` (PEFT) or the FuRA L·S·R re-merge loop in `src/finetune_blocktt.py` saving path. Full FT: 0. | Must     |
| **Merged checkpoint size on disk**      | `du -sb` of the saved `.safetensors`. Expect Full FT ≈ base; LoRA/DoRA/FuRA ≈ base (merged); LIFT/RandLoRA: adapter-only size. | Must     |
| **First-token latency** (batch=1)       | HuggingFace `generate(max_new_tokens=1)`, median of 50 runs after 10 warmups. Same prompt across methods.      | Should   |
| **Decode throughput (tok/s)**           | `generate(max_new_tokens=128)`, batch=8, median of 10 runs.                                                    | Should   |
| **vLLM serving throughput**             | Serve merged checkpoint with `vllm serve`, hit with 64-concurrent `benchmark_serving.py` prompts, report `completion_tokens/s`. Unmergeable methods (LIFT, RandLoRA): note "not directly servable in vLLM; requires custom kernel or merge-to-dense first". | Nice     |

### 2.4 Evaluation-time cost (commonsense suite)

The LIFT commonsense eval (`bash_scripts/eval_commonsense.sh` → `LLM-Adapters/commonsense_evaluate.py`) runs 8 tasks sequentially. Under the short-horizon protocol we do not have converged checkpoints to evaluate, so eval-time cost is measured **once per method shape** using the base Llama-3-8B model (with adapter shape attached but untrained) or a merged FuRA checkpoint if available.

| Metric                                  | How to measure                                                                                                 | Priority |
|-----------------------------------------|----------------------------------------------------------------------------------------------------------------|----------|
| **Total eval wall-clock (8 tasks)**     | Wrap `eval_commonsense.sh` in `/usr/bin/time -v`, run once per method shape.                                   | Should   |
| **Eval peak GPU memory**                | Same `cuda.max_memory_allocated()` hook as training.                                                           | Nice     |

Accuracy per task is read from paper Table 1, not re-measured. Eval-time cost is roughly method-agnostic for methods that merge cleanly (all weights end up as a dense 8B checkpoint); LIFT and RandLoRA are the interesting outliers because they evaluate with the adapter still attached.

---

## 3. Measurement procedure

### 3.1 Instrumentation (one-shot edit)

Add a small `system_metrics.py` helper imported by `src/finetune_blocktt.py`, `src/finetune_lora.py`, `src/finetune_sft.py`:

```python
# system_metrics.py
import json, time, torch
from pathlib import Path

class SysMon:
    def __init__(self, out_dir):
        self.out = Path(out_dir) / "sys_metrics.json"
        self.step_times = []
        self.start_wall = time.time()
        torch.cuda.reset_peak_memory_stats()

    def record_step(self, dt): self.step_times.append(dt)

    def dump(self, model, extra=None):
        import statistics as st
        warm = self.step_times[100:] if len(self.step_times) > 150 else self.step_times
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        data = {
            "trainable_params": trainable,
            "total_params":     total,
            "trainable_pct":    100.0 * trainable / 8.03e9,
            "stored_extra_pct": 100.0 * (total - 8.03e9) / 8.03e9,
            "steps_recorded":   len(self.step_times),
            "median_step_s":    st.median(warm) if warm else None,
            "total_wall_s":     time.time() - self.start_wall,
            "peak_alloc_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            **(extra or {}),
        }
        self.out.write_text(json.dumps(data, indent=2))
```

Call sites:
- After adapter attach / `configure_blocktt_trainability()` → `mon = SysMon(args.output_dir)`.
- Inside the train loop, wrap each optimizer step with `t0 = time.time(); ...; mon.record_step(time.time()-t0)`.
- After training (before eval) → `mon.dump(model, extra={"method": args.adapter_name or "blocktt" or "full", "effective_tokens_per_step": bs*accum*seq_len})`.

Writes one `sys_metrics.json` per run. Accuracy is not measured in the short-horizon protocol — it is read from the paper's existing Table 1 at the matched operating point.

### 3.2 Micro-benchmark for forward/backward/optimizer decomposition

A standalone script `tools/bench_fbopt.py` (new) that:
1. Loads each already-trained checkpoint (or freshly attached adapter if we only need shapes).
2. Feeds 20 dummy batches of shape `(8, 2048)` int64 tokens.
3. Times: (a) fwd-only, (b) fwd+bwd, (c) fwd+bwd+opt-step, reporting median of last 15 after 5 warmups.
4. Writes `fbopt_<method>.json`.

Run on a dedicated GPU (no training in parallel). This isolates adapter cost from data-loader noise.

### 3.3 Inference/merge benchmarks

`tools/bench_merge_and_decode.py` (new):
1. Load fine-tuned adapter + base.
2. `t0 = time.time(); merged = merge(model); merge_s = time.time() - t0`.
3. Save merged safetensors; record file size.
4. Run HF `generate` latency/throughput as defined in §2.3.
5. Optionally: serve with vLLM and run `vllm/benchmarks/benchmark_serving.py` (nice-to-have, gate on a CLI flag).

For Full FT: skip merge (identity). For LIFT/RandLoRA: record "not merge-compatible" and measure generation *with adapter still attached*.

### 3.4 Eval timing

Wrap the existing `eval_commonsense.sh` call:

```bash
/usr/bin/time -v -o $OUTPUT/eval_time.log \
  bash ./bash_scripts/eval_commonsense.sh ...
```

Parse `Elapsed (wall clock)` and `Maximum resident set size` into the report.

### 3.5 Run matrix

One seed (43), per-method LR fixed (the one already set as default in each bash script), rank swept per §1.2. **18 short-horizon runs total**: 4 ranks × {LoRA, DoRA, RandLoRA, LIFT} + Full FT + FuRA. Each run is capped at **300 optimizer steps** (100 warmup + 200 measurement), ~5–20 min per run on H100.

Implementation of the cap: the existing bash scripts already accept `--num_train_epochs` but not `--max_steps`. The driver script below injects `--max_steps 300` as an extra argument to `accelerate launch ... src/finetune_*.py` (HuggingFace `TrainingArguments` supports `max_steps`; verify once before launching the full matrix). If the trainer used in `finetune_blocktt.py` / `finetune_lora.py` / `finetune_sft.py` does not already surface this flag, add one argparse line per script (trivial, shared across the three).

Order of execution (stagger to avoid GPU contention):
1. Full FT (longest *per step*, run first to fail early on memory if so)
2. FuRA (single run, baseline for comparison)
3. LoRA r=16,32,64,128
4. DoRA r=16,32,64,128
5. RandLoRA r=16,32,64,128
6. LIFT r=16,32,64,128

Each writes to a distinct `OUTPUT_SRC_DIR/commonsense/.../sys_metrics.json` keyed by method + rank. No checkpoint is saved (`--save_interval` bumped above 300) since we measure and discard.

A driver script `tools/run_sft_matrix.sh` (new) iterates the 18 configurations sequentially on a single operator-supplied GPU, injecting `--max_steps 300` per invocation, and skipping any configuration whose output directory already contains a complete `sys_metrics.json` so the script is resumable. Expected wall-clock: under one H100-day total.

For merge-time and decode-throughput metrics (§2.3), we do *not* need trained checkpoints — the cost is shape-dependent, not weight-dependent. `tools/bench_merge_and_decode.py` constructs a freshly-initialised adapter at each method's rank-64 / rank-32 shape and measures merge + `generate` directly. Run **per method** (6 method instances: Full FT, LoRA-64, DoRA-64, RandLoRA-64, LIFT-32, FuRA). Same story for `tools/bench_fbopt.py` (forward/backward/opt-step split).

---

## 4. Results aggregation

A small script `tools/aggregate_sys_metrics.py` walks `OUTPUT_SRC_DIR`, reads every `sys_metrics.json` + `fbopt_*.json` + `merge_*.json` + `eval_time.log`, parses `method` and `rank` from the directory name (or JSON fields), and emits:

1. `docs/26_nips_fura_paper/tables/tab_system_perf_commonsense_headline.{tex,md}` — the **headline table at the matched-parameter operating point** (LoRA/DoRA/RandLoRA at $r{=}64$, LIFT at $r{=}32$, plus Full FT and FuRA). Mirrors §3.3 but with commonsense-specific columns:

| Method     | Rank | Trainable (%) | Stored extra (%) | Full-rank $\Delta\mat{W}$ | Train step (s) | Tokens/s | Peak train GPU (GB) | Merge (s) | Ckpt size (GB) | Decode tok/s | Eval time (min) |
|------------|------|---------------|-------------------|----------------------------|-----------------|-----------|----------------------|-----------|-----------------|---------------|-----------------|

2. `docs/26_nips_fura_paper/tables/tab_system_perf_commonsense_sweep.{tex,md}` — the **long-form rank-sweep table** (18 rows, one per run) keyed on `(method, rank)`.

3. `docs/26_nips_fura_paper/figs/sys_perf_vs_rank.pdf` — four line plots (step time, tokens/s, peak GPU, trainable%) with rank on the x-axis and one line per adapter method; Full FT and FuRA drawn as horizontal reference lines.

---

## 5. Concrete results template (to be filled after runs)

Numbers placeholder-marked `@R` are read from the aggregation script; no hand entry. The headline tables (§5.1–§5.4) report **one row per method** at the headline operating point: LoRA/DoRA/RandLoRA at $r{=}64$, LIFT at $r{=}32$, Full FT and FuRA as singletons. §5.5 holds the full sweep.

### 5.1 Static parameter table (headline)

```
| Method         | Rank | Trainable | Trainable % | Stored extra % | Optim state (GB) |
|----------------|-----:|-----------|-------------|-----------------|-------------------|
| Full FT        |   —  | @R        | 100         | 0               | @R                |
| LoRA           |   64 | @R        | @R          | @R              | @R                |
| DoRA           |   64 | @R        | @R          | @R              | @R                |
| RandLoRA       |   64 | @R        | @R          | @R              | @R                |
| LIFT           |   32 | @R        | @R          | @R              | @R                |
| **FuRA**       |   —  | @R        | @R          | @R              | @R                |
```

### 5.2 Training cost table (steady-state, headline)

```
| Method         | Rank | Step (s) | Tokens/s | Epoch (min) | Peak GPU (GB) | Fwd (ms) | Bwd (ms) | Opt (ms) |
|----------------|-----:|----------|----------|--------------|---------------|----------|----------|----------|
| Full FT        |   —  | @R       | @R       | @R           | @R            | @R       | @R       | @R       |
| LoRA           |   64 | @R       | @R       | @R           | @R            | @R       | @R       | @R       |
| DoRA           |   64 | @R       | @R       | @R           | @R            | @R       | @R       | @R       |
| RandLoRA       |   64 | @R       | @R       | @R           | @R            | @R       | @R       | @R       |
| LIFT           |   32 | @R       | @R       | @R           | @R            | @R       | @R       | @R       |
| **FuRA**       |   —  | @R       | @R       | @R           | @R            | @R       | @R       | @R       |
```

### 5.3 Deployment cost table (headline)

```
| Method         | Rank | Merge (s) | Merged ckpt (GB) | First-token (ms) | Decode tok/s | vLLM servable? |
|----------------|-----:|-----------|-------------------|-------------------|---------------|-----------------|
| Full FT        |   —  | @R        | @R                | @R                | @R            | Y               |
| LoRA           |   64 | @R        | @R                | @R                | @R            | Y               |
| DoRA           |   64 | @R        | @R                | @R                | @R            | Y               |
| RandLoRA       |   64 | @R        | @R                | @R                | @R            | N (shared proj) |
| LIFT           |   32 | @R        | @R                | @R                | @R            | N (sparse mask) |
| **FuRA**       |   —  | @R        | @R                | @R                | @R            | Y               |
```

### 5.4 Eval cost table (headline)

```
| Method         | Rank | 8-task eval time (min) | Eval peak GPU (GB) |
|----------------|-----:|-------------------------|---------------------|
| Full FT        |   —  | @R                      | @R                  |
| LoRA           |   64 | @R                      | @R                  |
| DoRA           |   64 | @R                      | @R                  |
| RandLoRA       |   64 | @R                      | @R                  |
| LIFT           |   32 | @R                      | @R                  |
| **FuRA**       |   —  | @R                      | @R                  |
```

### 5.5 Full rank sweep (long form)

All 18 runs, one row per `(method, rank)`. Adapter methods contribute 4 rows each; Full FT and FuRA contribute 1 row each.

```
| Method    | Rank | Trainable % | Stored extra % | Step (s) | Tokens/s | Peak GPU (GB) | Merge (s) | Ckpt size (GB) |
|-----------|-----:|-------------|-----------------|----------|----------|---------------|-----------|-----------------|
| Full FT   |   —  | 100         | 0               | @R       | @R       | @R            | 0         | @R              |
| LoRA      |   16 | @R          | @R              | @R       | @R       | @R            | @R        | @R              |
| LoRA      |   32 | @R          | @R              | @R       | @R       | @R            | @R        | @R              |
| LoRA      |   64 | @R          | @R              | @R       | @R       | @R            | @R        | @R              |
| LoRA      |  128 | @R          | @R              | @R       | @R       | @R            | @R        | @R              |
| DoRA      |   16 | @R          | @R              | @R       | @R       | @R            | @R        | @R              |
| DoRA      |   32 | @R          | @R              | @R       | @R       | @R            | @R        | @R              |
| DoRA      |   64 | @R          | @R              | @R       | @R       | @R            | @R        | @R              |
| DoRA      |  128 | @R          | @R              | @R       | @R       | @R            | @R        | @R              |
| RandLoRA  |   16 | @R          | @R              | @R       | @R       | @R            | @R        | @R              |
| RandLoRA  |   32 | @R          | @R              | @R       | @R       | @R            | @R        | @R              |
| RandLoRA  |   64 | @R          | @R              | @R       | @R       | @R            | @R        | @R              |
| RandLoRA  |  128 | @R          | @R              | @R       | @R       | @R            | @R        | @R              |
| LIFT      |   16 | @R          | @R              | @R       | @R       | @R            | @R        | @R              |
| LIFT      |   32 | @R          | @R              | @R       | @R       | @R            | @R        | @R              |
| LIFT      |   64 | @R          | @R              | @R       | @R       | @R            | @R        | @R              |
| LIFT      |  128 | @R          | @R              | @R       | @R       | @R            | @R        | @R              |
| **FuRA**  |   —  | @R          | @R              | @R       | @R       | @R            | @R        | @R              |
```

**Derived plots** (emitted as `sys_perf_vs_rank.pdf`, one sub-panel each):
- Step time (s) vs rank — one line per adapter method; Full FT / FuRA as horizontal dashed lines.
- Tokens/s vs rank — same.
- Peak GPU memory (GB) vs rank — same.
- Trainable-% vs rank — same (on log-y).

These curves answer the question "how does each method scale in $r$?" which the single-point headline table cannot. FuRA should appear as a flat dashed line that crosses each adapter's curve at a known (trainable-%, step-time) point.

---

## 6. Writeup plan

After `aggregate_sys_metrics.py` produces the tables and sweep plot, a separate document `docs/26_nips_fura_paper/system_eval_commonsense.md` is written with this structure:

1. **Setup recap** (1 paragraph) — hardware, model, dataset, matched configs, rank sweep.
2. **Static parameter footprint** (headline) — Table §5.1 at the matched point (LoRA/DoRA/RandLoRA $r{=}64$, LIFT $r{=}32$). Key observation: FuRA matches LoRA-64's trainable % but keeps the frozen $\mat L$ core as extra storage.
3. **Training cost** (headline) — Table §5.2, key comparisons: FuRA vs DoRA (expected FuRA wins on step time since DoRA adds a magnitude-vector normalization), FuRA vs LIFT (expected FuRA wins on step time since LIFT does scatter-gather).
4. **Deployment cost** (headline) — Table §5.3, highlight the merge-to-dense advantage (Full FT / LoRA / DoRA / FuRA) vs. unmergeable sparse/random (LIFT / RandLoRA).
5. **Eval cost** — Table §5.4, expect roughly equal across methods (eval runs on merged weights), with LIFT/RandLoRA slightly slower if not merged.
6. **Rank sweep** — Table §5.5 and the `sys_perf_vs_rank.pdf` panel. Shows how each adapter's step time / peak memory / trainable% scales with $r$, with FuRA drawn as a reference horizontal. Conclusion we expect to write: "FuRA matches or beats rank-64 LoRA on both accuracy (paper Table 1) and step time, with no rank knob to tune."
7. **Cost-vs-accuracy Pareto** — single plot: x = train step time (from this study), y = commonsense avg (from paper Table 1). Only the methods/ranks the paper reports contribute markers (LoRA-64, DoRA-64, S²FT-64, LIFT-32, Full FT, FuRA). Off-paper ranks appear only on the rank-sweep cost plot in §5.5, without an accuracy axis.
8. **Limitations of this system study** — SFT-only (no RL/GRPO), single seed, single H100, bf16 only, no multi-GPU, eval in-process (no serving stack).

---

## 7. How to further improve FuRA (speculative, ordered by expected payoff)

These are directions the §3.3 result (FuRA at 62s/step vs DoRA at 82s/step on Qwen3-1.7B GRPO) suggests are worth exploring. Each is written as *what to try* → *why it should help* → *how to measure the win*.

### 7.1 Kernel-level

1. **Fuse the two batched GEMMs into one CUTLASS/Triton kernel.**
   The forward in Algorithm 1 is `(L · S) · R · x`. At $d=4096$, $n=b=r=64$, the inner contraction is a grouped-GEMM with 64 groups of $64 \times 64$. A single Triton `grouped_matmul` kernel can do this in one launch instead of two, saving \~1 kernel launch per layer × 224 layers ≈ 10% of the step time at our batch size. Measure with `tools/bench_fbopt.py` after the fused kernel is in place.

2. **Persist the $\mat S$ scaling inside the grouped-GEMM.**
   $\mat S$ is a diagonal per-direction scalar ($nr$ entries). Right now it multiplies $\mat R$ eagerly. Fusing it into the GEMM epilogue (as a per-column scale) removes one elementwise pass over the activations.

3. **bf16 $\times$ bf16 → fp32 accumulate, then bf16 write**, explicitly, to prevent FlashAttention-style precision loss on long sequences. The current code relies on PyTorch defaults; a custom kernel can guarantee this.

### 7.2 Memory and optimizer

4. **8-bit AdamW on $\mat S$ and $\mat R$** (bitsandbytes).
   Both cores are small ($n r b + n r$). Quantising their optimizer state to 8-bit halves optimizer-state memory and lets us increase per-device batch size from 8 → 16, doubling tokens/s. FuRA-specific cost/benefit because LoRA already has tiny state and Full FT's 8-bit state is well-studied; FuRA is the sweet spot where this saves real memory.

5. **Paged/offloaded frozen $\mat L$.**
   $\mat L$ is read-only during forward and never updated. Offloading it to CPU with prefetch (á la QLoRA's NF4 weights) would let us drop the $\sim 1.5\%$ stored extra and fit larger batch sizes. Risk: prefetch hides latency only if bandwidth > GEMM FLOPs for $\mat L$; needs a measurement pass.

### 7.3 Algorithmic

6. **Block-wise rank search within FuRA's frozen budget.**
   Currently we set $n{=}b{=}r{=}\sqrt{d}$ uniformly. The paper's Figure 2a shows that *effective* rank varies 5× across layers. An obvious win: keep FuRA's parameterisation but allow $r_k$ to be layer-dependent, chosen either offline (from a calibration pass) or via a learnable gate with an $\ell_1$ penalty on $\mat S$. This is the single most promising accuracy-side improvement because it directly exploits the observation in §4.

7. **Momentum-preconditioned $\mat R$ updates (Muon-style).**
   FuRA's gradient on $\mat R$ is already in the pretrained basis — a natural place for orthogonalised updates. Re-using `optim/muon.py` (already in repo) on $\mat R$ only (keeping AdamW for $\mat S$) might cut step count 2×.

8. **Adaptive $\mat S$ placement (online switching between design corners).**
   The four rows of Table `tab:preconditioners` are not mutually exclusive across training. Start with principal-biased (PiSSA-like) for the first 10% of steps to quickly identify important directions, then switch to the default $\mat S$-separate configuration. Cost: free (just a flag flip). Risk: training instability at the switch; mitigate with a 100-step linear interpolation.

### 7.4 Serving

9. **Serve FuRA without merging.**
   For multi-tenant serving, a vLLM-style per-request adapter (like LoRA hot-swap) would be valuable. FuRA's factored form is already compatible with vLLM's "LoRA stacked as GEMMs" approach — the $\mat L$ core is shared per model, only $\mat S,\mat R$ vary per tenant. Prototype by adapting `vllm/lora/layers.py` to accept a $(n, r, b)$-structured adapter.

10. **Int8 quantisation of the merged checkpoint.**
    After merge, $\mat W'$ is a normal dense tensor that accepts any quantisation (GPTQ/AWQ). Confirming that the FuRA-merged model survives int8 without accuracy loss (unlike LIFT/RandLoRA which cannot merge cleanly) is a deployment-story win.

### 7.5 Benchmark-coverage

11. **Multi-GPU FSDP run.**
    The paper only reports single-GPU numbers. Adding an 8×H100 FSDP run (sharding across the block axis $n$) would establish that FuRA's "no gather/scatter" claim holds under real tensor parallelism. RandLoRA's shared projection and LIFT's sparse mask both get trickier under TP; that comparison is currently missing.

12. **Longer-context training cost.**
    Rerun the training-cost micro-benchmark at seq_len ∈ {2048, 4096, 8192, 16384} and plot step time vs seq_len. FuRA's batched GEMM structure should scale linearly; LIFT's scatter-gather can superlinearly grow. This directly supports the paper's "no gather/scatter" claim with a quantitative graph.

---

---

## 8. Kernel improvement — implementation, auto-launch, and report (Phase 2)

Phase 1 (§1–6) compares FuRA against baselines *as-implemented*. Phase 2 picks the highest-payoff kernel item from §7.1 (fuse the two batched GEMMs + fold `btt_s` into the epilogue) and drives a full cycle: implement → launch on GPU → collect metrics → auto-write the report. Everything is scripted so a single `make bench-kernel` reproduces the end-to-end pipeline unattended.

### 8.1 Current FuRA forward (what we are fusing)

The reference forward lives in `btt_layer.py:786–830`. Abstracted:

```
x:          (B, n, b)                   # reshaped input
R:          (n, b, m*r)                 # btt_r
L:          (m, r*n, a)                 # btt_l
s:          (m, n, r)  or None          # btt_s (diagonal per-block singular scale)

# Step 1 (bmm #1): project into latent
x_t    = x.transpose(0,1)               # (n, B, b)
inner  = bmm(x_t, R)                    # (n, B, m*r)
inner  = inner.view(n, B, m, r).permute(2,1,0,3)   # (m, B, n, r)

# (optional) activation or SwiGLU

# Step 2 (bmm #2): lift back
if s is not None:
    L_eff = (L.view(m, n, r, a) * s[..., None]).view(m, r*n, a)
else:
    L_eff = L
out    = bmm(inner.view(m, B, r*n), L_eff)           # (m, B, a)
out    = out.permute(1,0,2).reshape(..., out_features)
```

There are **two** opportunities:
1. The explicit `L · s` materialisation allocates a temporary $m \times n \times r \times a$ tensor each forward. Folding `s` into the GEMM epilogue removes that allocation + elementwise pass.
2. The permute between Step 1 and Step 2 is a data-movement pass; with a fused kernel that keeps `inner` in registers/shared memory across the two contractions we avoid a round-trip to HBM.

### 8.2 V1 scope (what we actually implement first)

**Target: `btt_fused_forward` — a Triton kernel that folds `btt_s` into Step 2 and removes the explicit `L_eff` materialisation. Keep Step 1 and Step 2 as two kernels, but Step 2 is custom.** This is deliberately the minimal-risk win; if it works, a follow-up V2 fuses both steps through a persistent-CTA grouped-GEMM.

Concrete deliverable:

```python
# fura_kernels/triton_btt.py  (new module)
import triton, triton.language as tl, torch

@triton.jit
def _step2_s_scaled_gemm_kernel(
    INNER_ptr,        # (m, B, r*n)      fp16/bf16
    L_ptr,            # (m, r*n, a)      fp16/bf16
    S_ptr,            # (m, n, r) or 0   fp32
    OUT_ptr,          # (m, B, a)        fp16/bf16
    M, B, RN, A, N, R,
    stride_im, stride_ib, stride_irn,
    stride_lm, stride_lrn, stride_la,
    stride_sm, stride_sn, stride_sr,
    stride_om, stride_ob, stride_oa,
    BLOCK_B: tl.constexpr, BLOCK_A: tl.constexpr, BLOCK_RN: tl.constexpr,
    HAS_S: tl.constexpr,
):
    # grid = (m, ceil_div(B, BLOCK_B), ceil_div(A, BLOCK_A))
    # inner loop over rn in BLOCK_RN chunks; if HAS_S, broadcast s_mn over a tile
    ...

def step2_s_scaled_bmm(inner, L, S_or_None):
    """inner: (m, B, r*n) | L: (m, r*n, a) | S: (m, n, r) or None → (m, B, a)"""
    ...
```

A PyTorch wrapper `BTTFusedFunction(torch.autograd.Function)` calls the kernel in forward and uses two standard `torch.bmm` calls in backward (Triton backward can come later; correctness first).

The rest of the forward (Step 1 bmm, permute, bias) stays Python/PyTorch.

### 8.3 Integration point

Add a new forward path in `BTTLayer`, toggled by a class-level flag (default OFF so baseline runs are untouched):

```python
# btt_layer.py
class BTTLayer(nn.Module):
    use_fused_step2 = False  # class-level switch

    def forward(self, x):
        ...
        if BTTLayer.use_fused_step2 and self.btt_r.dtype in (torch.bfloat16, torch.float16):
            out = step2_s_scaled_bmm(
                inner.reshape(self.m, batch_n, self.rank * self.n),
                self.btt_l,
                self.btt_s if self.btt_s is not None else None,
            )
        else:
            # existing path
            ...
```

An env var (`FURA_FUSED_STEP2=1`) sets the flag at import time. This keeps the change ~20 lines in `btt_layer.py` and fully reversible.

### 8.4 Correctness tests (must pass before any GPU benchmark runs)

File: `tests/test_fura_fused_kernel.py`

1. **Shape grid.** For `(d, n, r, b, m, a) ∈ {Llama-3-8B q_proj, o_proj, up_proj, down_proj shapes}` × `batch ∈ {1, 8, 64, 256}` × `has_s ∈ {True, False}`:
   reference = existing BTTLayer forward in fp32; candidate = fused path. Assert `torch.allclose(ref, fused, atol=1e-2, rtol=1e-2)` in bf16, `1e-4/1e-4` in fp32.
2. **Gradient check.** `torch.autograd.gradcheck` on a small size (d=128) with fp64 to catch backward bugs.
3. **Non-contiguous inputs.** Feed a permuted `x` to make sure the wrapper materialises contiguous tensors where required.

Run with `pytest -q tests/test_fura_fused_kernel.py` — a `make test-kernel` target.

### 8.5 Microbenchmark harness

File: `tools/bench_fused_kernel.py` (new).

```python
# Pseudocode
SHAPES = [
    ("llama3_qproj",  dict(d=4096, n=64, r=64, b=64, m=64, a=64)),
    ("llama3_upproj", dict(d_in=4096, d_out=14336, ...)),
    # ... all 5 target_modules
]
BATCHES = [1024, 2048, 4096, 8192]   # effective tokens = bs × seq_len
DTYPES  = [torch.bfloat16]

results = []
for name, shape in SHAPES:
  for B in BATCHES:
    layer_baseline = BTTLayer(**shape)
    layer_fused    = BTTLayer(**shape); BTTLayer.use_fused_step2 = True
    x = torch.randn(B, shape["d_in"], device="cuda", dtype=torch.bfloat16)
    t_base = benchmark(lambda: layer_baseline(x), warmup=10, iters=50)
    t_fus  = benchmark(lambda: layer_fused(x),    warmup=10, iters=50)
    peak_base = peak_mem(lambda: layer_baseline(x).sum().backward())
    peak_fus  = peak_mem(lambda: layer_fused(x).sum().backward())
    results.append(dict(shape=name, B=B, t_base_us=t_base, t_fus_us=t_fus,
                        speedup=t_base/t_fus, mem_base=peak_base, mem_fus=peak_fus))

Path("reports/fused_kernel_micro.json").write_text(json.dumps(results, indent=2))
```

Uses `torch.cuda.Event` pairs for timing and `torch.cuda.max_memory_allocated()` for memory. Writes one JSON artifact.

### 8.6 End-to-end SFT timing harness

File: `tools/bench_fused_kernel_sft.py` (new).

Runs the existing FuRA commonsense script twice on a dedicated GPU (held for ~90 min), one iteration is enough because we measure steady-state step time:

```bash
# Run A: baseline (no fused kernel)
CUDA_VISIBLE_DEVICES=$GPU FURA_FUSED_STEP2=0 \
  MODEL=meta-llama/Meta-Llama-3-8B seed=43 \
  num_train_epochs=0.05 max_steps=300 \
  bash ref/LIFT/bash_scripts/finetune_commonsense_blocktt.sh

# Run B: fused
CUDA_VISIBLE_DEVICES=$GPU FURA_FUSED_STEP2=1 \
  (same env) \
  bash ref/LIFT/bash_scripts/finetune_commonsense_blocktt.sh
```

A new `num_train_epochs` / `max_steps` escape hatch (already supported by Trainer) caps the run at ~300 optimizer steps, enough for steady-state measurement but short (~5–8 min each). The `system_metrics.py` helper from §3.1 writes `sys_metrics.json` in both output dirs; they are compared directly.

### 8.7 Auto-launch orchestration

File: `tools/run_kernel_eval.sh` (new, the single entry point).

```bash
#!/usr/bin/env bash
set -euo pipefail
GPU="${GPU:-0}"
OUT_ROOT="${OUT_ROOT:-/data/yequan/fura/sys_eval/kernel}"
mkdir -p "$OUT_ROOT"

# 1. correctness
pytest -q tests/test_fura_fused_kernel.py | tee "$OUT_ROOT/test.log"

# 2. microbench
CUDA_VISIBLE_DEVICES=$GPU uv run tools/bench_fused_kernel.py \
  --out "$OUT_ROOT/fused_kernel_micro.json"

# 3. end-to-end SFT (300-step cap)
CUDA_VISIBLE_DEVICES=$GPU bash tools/bench_fused_kernel_sft.py \
  --out_root "$OUT_ROOT/sft" --max_steps 300

# 4. aggregate + write report
uv run tools/write_kernel_report.py \
  --micro "$OUT_ROOT/fused_kernel_micro.json" \
  --sft_baseline "$OUT_ROOT/sft/baseline/sys_metrics.json" \
  --sft_fused    "$OUT_ROOT/sft/fused/sys_metrics.json" \
  --out "docs/26_nips_fura_paper/kernel_eval_report.md"

echo "Report: docs/26_nips_fura_paper/kernel_eval_report.md"
```

A single command — `GPU=0 bash tools/run_kernel_eval.sh` — runs tests, microbenchmark, end-to-end, and writes the Markdown report. CI-friendly: exits non-zero if tests fail or if the microbenchmark shows a regression (`speedup < 0.95`).

For unattended operation on a shared machine, wrap the command in:

```bash
nohup bash tools/run_kernel_eval.sh > run.log 2>&1 &
```

and detect completion by polling for the report file (or rely on `ScheduleWakeup`/hooks if the orchestrator is Claude Code).

### 8.8 Auto-report generation

File: `tools/write_kernel_report.py` (new).

Reads the three JSON artifacts, computes derived numbers, and emits `docs/26_nips_fura_paper/kernel_eval_report.md` with this structure (all `{...}` filled from the JSON, no hand editing):

```
# FuRA Fused-Step2 Kernel: Evaluation Report
Auto-generated on {ISO date} by tools/write_kernel_report.py

## Setup
- GPU: {gpu_name from nvidia-smi}
- Model: Llama-3-8B, FuRA default corner (decomp=input_one_block, train=small, s=frozen)
- bf16, gradient checkpointing ON, bs=8 × accum=2, seq-len 2048

## Correctness
- Shape-grid tests: {passed}/{total} (bf16 atol/rtol 1e-2/1e-2; fp64 gradcheck passed)

## Microbenchmark (forward pass, per-layer)

| Shape         | Batch | Baseline (µs) | Fused (µs) | Speedup | Peak mem baseline (MB) | Peak mem fused (MB) |
|---------------|------:|---------------:|------------:|--------:|-----------------------:|---------------------:|
| llama3_qproj  |   1024| ...            | ...         | ...×    | ...                    | ...                  |
| llama3_upproj |   2048| ...            | ...         | ...×    | ...                    | ...                  |
| ...           |    ...| ...            | ...         | ...     | ...                    | ...                  |

**Geometric-mean speedup over all shape×batch cells: {gmean}×.**

## End-to-end SFT (300 optimizer steps)

| Run      | Median step (s) | Tokens/s | Peak GPU (GB) | Total wall (min) |
|----------|-----------------:|---------:|--------------:|------------------:|
| Baseline | ...              | ...      | ...           | ...               |
| Fused    | ...              | ...      | ...           | ...               |
| **Δ**    | **-X.X% step**   | **+Y.Y% tok/s** | ...   | ...               |

## Verdict
Auto-generated boilerplate based on the numbers above:
- "Fused Step 2 delivers {X%} end-to-end step-time reduction at Llama-3-8B FuRA, mostly driven by {upproj/downproj}, with no accuracy change (identical init + short-horizon SFT)."
- Next candidate: {V2 full-fusion / shape with worst speedup / mem regression}.

## Raw artifacts
- Microbench: fused_kernel_micro.json
- SFT baseline: sft/baseline/sys_metrics.json
- SFT fused:    sft/fused/sys_metrics.json
```

The template lives in the script itself (Python f-strings), not in a separate file. Acceptance criteria for the report: speedup > 1.0× on at least 3 of the 5 target-module shapes **and** end-to-end SFT step time improves by ≥ 2%.

### 8.9 Deliverables summary (Phase 2)

| File (new)                                  | Purpose                                                   |
|---------------------------------------------|-----------------------------------------------------------|
| `fura_kernels/__init__.py`                  | Package init                                              |
| `fura_kernels/triton_btt.py`                | Triton kernel + autograd wrapper                          |
| `tests/test_fura_fused_kernel.py`           | Correctness (shape grid + gradcheck + non-contiguous)     |
| `tools/bench_fused_kernel.py`               | Per-layer microbenchmark → `fused_kernel_micro.json`      |
| `tools/bench_fused_kernel_sft.py`           | Orchestrate 2× short-horizon SFT runs                     |
| `tools/run_kernel_eval.sh`                  | Single-command auto-launch for the full cycle             |
| `tools/write_kernel_report.py`              | Consume JSONs → write `kernel_eval_report.md`             |

| File (modified)                             | Change                                                    |
|---------------------------------------------|-----------------------------------------------------------|
| `btt_layer.py`                              | ~20 lines: opt-in `use_fused_step2` branch in `forward`   |
| `Makefile` (create if absent)               | Targets `test-kernel`, `bench-kernel`                     |

### 8.10 Risk / fallback

- **Triton bf16 numerical divergence.** Mitigated by fp32 accumulator inside the kernel and the `1e-2` tolerance in the correctness test. If bf16 tolerances fail, downgrade to `use_fused_step2=False` and report no-go in the report.
- **Kernel compile time on first launch.** Triton JIT compile can be 20-30s per config. Accept it on first run; subsequent runs hit the Triton cache.
- **GPU contention.** The auto-launch script acquires a user-specified `CUDA_VISIBLE_DEVICES` and does not attempt any lock — the operator is responsible for supplying a free GPU. The script fails fast if any other process holds that GPU (`nvidia-smi --query-compute-apps=pid -i $GPU` nonempty).

---

## 9. Out of scope

- **RL / GRPO** — explicitly excluded; this doc covers the LIFT commonsense SFT path only. A separate RL system-eval doc can re-use the harness against Qwen3-1.7B GRPO later.
- **Arithmetic SFT** (MATH-10K) — same methods, same harness would apply, but is excluded here to keep runtime bounded.
- Re-running accuracy — already in paper Table 1 at the matched point; adapter rank-sweep accuracies are not re-measured since the story is about system cost.
- Multi-seed variance on system metrics — single run is sufficient for wall-clock (variance is small; we use the median of 200 steady-state steps within a run).
- Multi-node training.
- Energy / joules measurement — nvidia-smi power logging would be a nice addition but is not required for the §3.3 story.
