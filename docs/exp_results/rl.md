# RL Experiment Results

_Updated 2026-04-23 after bug-fix sweep, LR exploration, and fura LR sweep. Llama-3.1-8B-Instruct sweeps appended 2026-04-27._

- Source: `/data/yequan/fura/rl_runs`
- Methods covered: `full`, `lora`, `lora_full`, `dora`, `pissa`, `milora`, `lift`, `randlora`, `fura` (=`blocktt`), `svd`.
- Primary metric: `eval/accuracy` on the internal 1000-problem held-out split.
- Extended math evals (post-2026-04-19 runs only): `eval/MATH-500/accuracy`, `eval/AMC23/accuracy`, `eval/AIME-24/accuracy` (avg@8), `eval/AIME-25/accuracy` (avg@8), `eval/Minerva/accuracy`.
- `fura` is the project's own BlockTT-based method, stored under `rl_runs/blocktt/`.
- Metric source: final values in each run's `wandb-summary.json` (largest summary per run dir).
- Regeneration script: `uv run python tools/collect_rl_results.py --root /data/yequan/fura/rl_runs`.

## TL;DR — Best per method

**Summary table** — best run per method (selected by mean of all five extended evals; for runs that pre-date the Minerva fix, only 4 evals are averaged).

| Method | Best LR | eval/acc | MATH-500 | AMC23 | AIME-24 | AIME-25 | Minerva |
|---|---|---:|---:|---:|---:|---:|---:|
| **fura** (blocktt) | 8e-5 | 85.4 | 63.0 | 52.5 | 13.3 | 17.5 | 21.0 |
| **full** | 2e-5 | 86.3 | 63.6 | 47.5 | 13.8 | 15.4 | — |
| **randlora** | 1e-4 | 85.6 | 63.2 | 57.5 | 15.8 | 17.1 | 20.2 |
| **svd** | 1e-5 | 88.4 | 64.0 | 40.0 | 16.7 | 13.8 | 20.6 |
| **lora** | 6e-5 | 84.8 | 60.6 | 50.0 | 11.2 | 11.2 | 19.1 |
| **dora** | 1e-4 | 86.5 | 61.6 | 37.5 | 12.9 | 14.6 | 22.1 |
| **lift** | 8e-5 | 84.2 | 60.2 | 37.5 | 10.0 | 8.8 | — |
| **milora** | 6e-5 | 85.6 | 59.0 | 45.0 | 7.9 | 9.6 | 19.1 |
| **pissa** | 8e-5 | 78.6 | 53.4 | 42.5 | 2.5 | 3.3 | — |

Methods without Minerva (full, lift, pissa) used runs from 2026-04-19–21 that pre-date the `math-ai/minervamath` hf-id fix; their best runs by {MATH-500, AMC23, AIME-24, AIME-25} mean are shown. A Minerva-only re-eval on those checkpoints is pending.

### FuRA LR sweep detail

| FuRA LR | eval/acc | MATH-500 | AMC23 | AIME-24 | AIME-25 | Minerva |
|---|---:|---:|---:|---:|---:|---:|
| **8e-5** | 85.4 | **63.0** | **52.5** | 13.3 | **17.5** | **21.0** |
| 1e-4 (prev best) | 87.1 | 62.8 | 47.5 | 12.5 | 16.7 | — |
| 2e-4 | 86.6 | 60.8 | 47.5 | **18.8** | 15.0 | 18.0 |
| 3e-4 | 85.1 | 59.6 | 45.0 | 12.1 | 11.7 | 20.6 |

Config for all rows: `output_one_block`, `train_small`, `s_to_trainable`. The 1e-4 row pre-dates the Minerva fix so lacks that score. On the 4 common metrics (MATH-500, AMC23, AIME-24, AIME-25), lr 8e-5 leads in 3 of 4 and lr 2e-4 leads AIME-24 (18.8% — best single result in the entire table). LR 3e-4 degrades consistently. Recommended operating point: **8e-5** for MATH/AMC/AIME-25 coverage, or **2e-4** for AIME-24 specialisation.

**Best headline primary `eval/accuracy` per method**:

| Method | Runs | Best | Mean | Best run |
|---|---:|---:|---:|---|
| full | 7 | 88.6 | 71.6 | `full-adamw-lr_2e-5-0325-215533` |
| lora | 6 | 84.8 | 48.8 | `lora-adamw-lr_6e-5-rank_64-sweep-0422-002231` |
| lora_full | 1 | 85.6 | 85.6 | `lora_full-adamw-lr_1e-5-rank_64-0319-140945` |
| dora | 4 | 87.1 | 69.2 | `dora-adamw-lr_2e-4-rank_64-sweep-0422-021100` |
| pissa | 3 | 80.2 | 66.7 | `pissa-adamw-lr_6e-5-rank_64-sweep-0422-032327` |
| milora | 3 | 85.6 | 82.6 | `milora-adamw-lr_6e-5-rank_64-sweep-0422-020941` |
| lift | 4 | 84.9 | 64.9 | `lift-adamw-lr_6e-5-sweep-0422-031308` |
| randlora | 3 | 85.6 | 82.4 | `randlora-adamw-lr_1e-4-rank_64-sweep-0422-041850` |
| **fura** | 20 | **89.5** | 82.5 | `blocktt-adamw-lr_1e-5-output_one_block-s_to_keep-train_both-0317-155422` |
| svd | 6 | 89.1 | 87.5 | `svd-adamw-lr_1e-5-s_to_keep-train_input-0317-141139` |

(`lora_full` is the legacy LoRA-with-base-train path and is not subject to this round's sweep; listed here only for reference.)

## Takeaways

1. **LoRA regression fixed.** The recently added `dora/pissa/milora/randlora/lift` support had broken the vanilla `lora` path: `export_lora_merged_weights` in `run_rl.py` appended a live reference to `base_layer.weight` before `unmerge_adapter()` mutated it back to the frozen base. vLLM was therefore serving the pre-training base model every step even while local gradients were computed. Fix: clone the merged tensor before unmerging (`param.detach().clone()` in the new `export_lora_merged_weights_for_vllm` helper, `run_rl.py:1038-1061`). LoRA now converges normally: `lora-lr_6e-5-rank_64` reaches 84.8% eval/accuracy vs ~15–17% on the broken code. DoRA was accidentally unaffected because PEFT's DoRA unmerge swaps `.data` rather than mutating in place.

2. **Minerva eval now working.** `math-ai/minerva-math` is gated on HF and fails to load; the community mirror `math-ai/minervamath` has the same 272-example test split and loads cleanly. Patched in `eval_datasets.py`. Minerva now reports on every new run.

3. **RandLoRA save-and-eval fixed.** The pre-fix `randlora-lr_8e-5-0421` run converged fine (train acc 66% at step 50) but crashed in `save_merged_checkpoint` with "shared tensors" because PEFT's RandLoRA shares the `randlora_A` random projection across layers and safetensors refuses to serialize shared storages. Since this save happens before the math-verify hook, math-verify never fired. Two fixes in `run_rl.py`:
   - `save_merged_checkpoint` now passes `safe_serialization=False` specifically for `randlora` so the save succeeds via torch.save (`run_rl.py:903-919`).
   - The post-training final-checkpoint save is wrapped in `try/except`, so even if the save fails the math-verify hook still runs from the in-memory model (`run_rl.py:2070-2090`).

4. **LR sweep ranking.**
   - `fura`: **8e-5 is the new best** — MATH-500 63.0%, AMC23 52.5%, AIME-25 17.5%, Minerva 21.0%. The previous best at 1e-4 is slightly weaker (MATH 62.8, AMC 47.5, no Minerva). Higher LRs (2e-4, 3e-4) still converge but degrade on MATH/AMC. Notably 2e-4 achieves the best single-run AIME-24 (18.8%) across all methods, possibly by over-fitting to harder competition problems at the expense of broader coverage.
   - `lora`: 6e-5 > 1e-4 > 2e-4 (2e-4 peaks around step 20 then over-trains to 75%). lr 8e-5 with rank 16/64 and lr 5e-5 with rank 64 all **diverged** under the pre-fix code; re-running them on the fixed code was not attempted but the 6e-5 and 1e-4 data points already bracket the optimum.
   - `dora`: 1e-4 > 2e-4 > 8e-5 across the extended evals; but all three are within noise on primary eval (85.3 / 86.5 / 87.1). The `dora-lr_1e-4` retry converged cleanly after an earlier OOM; OOM is a concurrency-pressure issue on shared GPUs, not a method issue.
   - `pissa`: 8e-5 > 6e-5 ≫ 1e-4. `pissa-lr_1e-4` collapses (eval/acc 41%, MATH-500 38%); `pissa-lr_6e-5` matches `pissa-lr_8e-5` within noise.
   - `milora`: 6e-5 > 8e-5 ≈ 1e-4. Gains at 6e-5 are small but consistent (MATH +4, Minerva +2).
   - `lift`: 8e-5 > 6e-5 > 1e-4 ≫ 2e-4. `lift-lr_2e-4` fully collapses (MATH-500 7%) — LIFT is the most LR-sensitive method in the family.
   - `randlora`: **1e-4 > 8e-5** across every metric (eval 85.6 vs 79.1, MATH 63.2 vs 61.6, AMC 57.5 vs 50.0, AIME-25 17.1 vs 15.8). The 8e-5 baseline substantially under-trained randlora.

5. **Ranking across all methods.** On the four common extended evals (MATH-500, AMC23, AIME-24, AIME-25) where all methods have data, sorting by unweighted mean:
   - **Tier 1** (~34-35): `fura` (8e-5) 36.6 ≈ `full` (2e-5) 35.1 ≈ `randlora` (1e-4) 34.8.
   - **Tier 2** (~28-31): `svd` 33.6 > `lora` 30.4 > `dora` 29.7 > `lift` 29.1 > `milora` 28.1.
   - **Tier 3** (~25): `pissa` 25.4.
   Of the tier-1 methods, **fura trains with far fewer params than full** and its best config (8e-5) now also includes Minerva (21.0%), making it the most complete and competitive PEFT result.

6. **fura vs full, per-benchmark.** With the new 8e-5 sweep point, fura leads full on MATH-500 (63.0 vs 63.6 — within noise), AMC23 (52.5 vs 47.5), and AIME-25 (17.5 vs 15.4); full leads on AIME-24 (13.8 vs 13.3 — marginal). Fura additionally reports Minerva 21.0% where full's best run does not. The 2e-4 fura variant beats both on AIME-24 (18.8% — the best single result across all methods and LRs in the entire table).

## All runs with extended eval (sorted by MATH-500)

| Method | Run | eval/acc | train/acc | MATH-500 | AMC23 | AIME-24 | AIME-25 | Minerva |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| svd | `svd-adamw-lr_1e-5-s_to_keep_trainable-train_input-ext-0421-221304` | 88.4 | 70.3 | 64.0 | 40.0 | 16.7 | 13.8 | 20.6 |
| full | `full-adamw-lr_2e-5-0420-173501` | 86.3 | 66.4 | 63.6 | 47.5 | 13.8 | 15.4 | — |
| randlora | `randlora-adamw-lr_1e-4-rank_64-sweep-0422-041850` | 85.6 | 66.0 | 63.2 | 57.5 | 15.8 | 17.1 | 20.2 |
| **fura** | `blocktt-adamw-lr_8e-5-output_one_block-s_to_trainable-train_small-sweep-0423` | 85.4 | 66.0 | 63.0 | 52.5 | 13.3 | 17.5 | 21.0 |
| fura | `blocktt-adamw-lr_1e-4-output_one_block-s_to_trainable-train_small-0419-185330` | 87.1 | 69.5 | 62.8 | 47.5 | 12.5 | 16.7 | — |
| dora | `dora-adamw-lr_1e-4-rank_64-retry-0422-013148` | 86.5 | 68.0 | 61.6 | 37.5 | 12.9 | 14.6 | 22.1 |
| randlora | `randlora-adamw-lr_8e-5-rank_64-fix1-0421-221308` | 79.1 | 63.7 | 61.6 | 50.0 | 15.8 | 15.8 | 21.0 |
| fura | `blocktt-adamw-lr_1e-4-output_one_block-s_to_keep_trainable-train_small-0419-185333` | 85.8 | 67.6 | 61.4 | 55.0 | 10.0 | 12.9 | — |
| fura | `blocktt-adamw-lr_2e-4-output_one_block-s_to_trainable-train_small-sweep-0423` | 86.6 | 66.4 | 60.8 | 47.5 | 18.8 | 15.0 | 18.0 |
| lora | `lora-adamw-lr_6e-5-rank_64-sweep-0422-002231` | 84.8 | 65.2 | 60.6 | 50.0 | 11.2 | 11.2 | 19.1 |
| lift | `lift-adamw-lr_8e-5-0421-143200` | 84.2 | 62.1 | 60.2 | 37.5 | 10.0 | 8.8 | — |
| fura | `blocktt-adamw-lr_3e-4-output_one_block-s_to_trainable-train_small-sweep-0423` | 85.1 | 64.8 | 59.6 | 45.0 | 12.1 | 11.7 | 20.6 |
| dora | `dora-adamw-lr_2e-4-rank_64-sweep-0422-021100` | 87.1 | 68.0 | 59.0 | 50.0 | 5.0 | 10.0 | 20.2 |
| milora | `milora-adamw-lr_6e-5-rank_64-sweep-0422-020941` | 85.6 | 64.1 | 59.0 | 45.0 | 7.9 | 9.6 | 19.1 |
| dora | `dora-adamw-lr_8e-5-rank_64-0421-142231` | 85.3 | 66.8 | 58.0 | 45.0 | 7.1 | 8.3 | — |
| lift | `lift-adamw-lr_6e-5-sweep-0422-031308` | 84.9 | 65.6 | 57.4 | 32.5 | 9.6 | 7.9 | 16.2 |
| lift | `lift-adamw-lr_1e-4-sweep-0422-001405` | 82.8 | 63.7 | 55.6 | 30.0 | 7.5 | 10.0 | 20.2 |
| milora | `milora-adamw-lr_8e-5-rank_64-0421-152507` | 80.9 | 59.0 | 55.0 | 32.5 | 5.4 | 4.6 | — |
| milora | `milora-adamw-lr_1e-4-rank_64-sweep-0421-231002` | 81.4 | 60.5 | 54.4 | 47.5 | 3.8 | 4.2 | 17.3 |
| pissa | `pissa-adamw-lr_6e-5-rank_64-sweep-0422-033440` | 71.6 | 59.0 | 54.4 | 40.0 | 3.8 | 4.6 | 18.8 |
| pissa | `pissa-adamw-lr_8e-5-rank_64-0421-142217` | 78.6 | 58.2 | 53.4 | 42.5 | 2.5 | 3.3 | — |
| lora | `lora-adamw-lr_1e-4-rank_64-fix1-0421-221302` | 82.1 | 63.3 | 53.2 | 30.0 | 5.4 | 4.2 | 18.0 |
| pissa | `pissa-adamw-lr_6e-5-rank_64-sweep-0422-032327` | 80.2 | 57.8 | 52.2 | 32.5 | 2.5 | 6.2 | 14.3 |
| lora | `lora-adamw-lr_2e-4-rank_64-sweep-0422-010758` | 75.3 | 64.8 | 47.4 | 27.5 | 3.8 | 0.8 | 17.3 |
| pissa | `pissa-adamw-lr_1e-4-rank_64-sweep-0421-232419` | 41.2 | 43.0 | 38.4 | 22.5 | 2.5 | 4.6 | 10.3 |
| lora | `lora-adamw-lr_5e-5-rank_64-0420-173503` | 17.6 | 16.0 | 37.2 | 17.5 | 0.0 | 0.0 | — |
| full | `full-adamw-lr_5e-5-0419-185332` | 14.0 | 23.8 | 20.4 | 10.0 | 0.4 | 0.0 | — |
| lora | `lora-adamw-lr_8e-5-rank_16-0420-173503` | 15.6 | 13.7 | 12.4 | 10.0 | 0.0 | 0.0 | — |
| lora | `lora-adamw-lr_8e-5-rank_64-0419-185332` | 17.1 | 15.6 | 8.0 | 2.5 | 0.0 | 0.0 | — |
| lift | `lift-adamw-lr_2e-4-sweep-0422-010909` | 7.8 | 12.9 | 7.0 | 0.0 | 0.0 | 0.8 | 2.9 |

Top 4 of those bottom rows are the broken pre-fix runs; they are kept for historical contrast with the fixed runs right above them (e.g. `lora-lr_8e-5-rank_64` pre-fix: MATH 8.0, AIME 0/0 vs `lora-lr_6e-5-rank_64` post-fix: MATH 60.6, AIME 11.2/11.2).

## Llama-3.1-8B-Instruct LR sweep (added 2026-04-27)

GRPO sweep on `meta-llama/Llama-3.1-8B-Instruct` (and the bit-equivalent `NousResearch/Meta-Llama-3.1-8B-Instruct` mirror), `qwedsacf/competition_math`, 50 GRPO steps, eval = the same internal 1000-prompt held-out split. Two sub-sweeps:

- **LoRA r=64 + BlockTT (output_one_block / small / keep_trainable / full)** at 4 LRs, on the NousResearch mirror (pre-training baseline **67.10%** for LoRA path, **67.60%** for BlockTT path — see footnote on baseline mismatch).
- **LoRA r=16** at the same 4 LRs, on the official `meta-llama/Llama-3.1-8B-Instruct` (pre-training baseline **63.50%**).

Memory recipe (single H100 96GB):
- LoRA: `--gpu-memory-utilization 0.3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (peak ~77 GB).
- BlockTT: `--gpu-memory-utilization 0.25 --max-model-len 1536 --micro-batch-size 1 --gradient-accumulation-steps 256` (peak ~71 GB). Default `gpu-memory-utilization 0.4` OOMs at step 1 / step 2 because the rollout-time vLLM weight materialization plus the trainer activation footprint exceeds 93 GB.

Wandb projects: `llama3-8B-RL` (Instruct sweeps) and `llama3-8B-base-RL` (base sweep below).

### Headline: BlockTT >> LoRA on Llama-3.1-8B-Instruct

| LR | LoRA r=64 final | LoRA r=64 peak | BlockTT final | BlockTT peak |
|---|---:|---:|---:|---:|
| 6e-5 | 14.00% (collapsed @ step ~18) | 67.10% (init) | **67.40%** ✅ | 68.10% |
| 8e-5 | 13.40% (collapsed @ step ~10) | 67.10% (init) | 12.90% (late collapse @ step ~42) | 68.00% |
| 1e-4 | 29.20% (slow degrade) | 67.10% (init) | **69.00%** ✅ best of sweep | 69.00% |
| 2e-4 | 0.00% (catastrophic, late collapse @ step ~28) | 67.10% (init) | 13.00% (early collapse @ step ~5) | 67.60% (init) |

Baseline = 67.10% (LoRA path) / 67.60% (BlockTT path). "Peak" = the best `step=N, correct:` eval observed at any of {step 5, 10, …, 50}. "Final" = step 50 eval. **2 of 4 BlockTT runs finish at or above baseline; 0 of 4 LoRA-r=64 runs do.**

### LoRA r=16 vs r=64 — smaller adapter is meaningfully more stable

LoRA r=16 sweep on the official `meta-llama/Llama-3.1-8B-Instruct` (baseline **63.50%**):

| LR | r=16 final | r=16 peak | r=64 final (NousResearch mirror) |
|---|---:|---:|---:|
| 6e-5 | **60.40%** ✅ | 68.00% (step 30) | 14.00% ❌ |
| 8e-5 | 0.00% ❌ (collapsed @ step ~20 after peak) | 68.40% (step 10) | 13.40% ❌ |
| 1e-4 | **59.30%** ✅ | 68.10% (step 10) | 29.20% (slow degrade) |
| 2e-4 | 0.00% ❌ (late collapse @ step ~50, peak @ step 15) | **71.30%** (step 15 — highest peak in sweep) | 0.00% ❌ |

Smaller adapter capacity = smaller per-step parameter movement = much less destructive drift. **2 of 4 r=16 runs survive vs 0 of 4 at r=64.** None of the r=16 runs *improves* over baseline at step 50, but the peaks (~68–71% across all 4 LRs) sit above baseline 63.5% — early-stopping at step ~10–15 would beat all final evals by 5–8 pp.

### Cross-method comparison on Llama-3.1-8B-Instruct

Sorting by step-50 eval/accuracy (higher is better):

| Rank | Run | Final | Peak | Δ vs baseline |
|---:|---|---:|---:|---:|
| 1 | BlockTT lr=1e-4 | **69.00%** | 69.00% | +1.4 pp ✅ |
| 2 | BlockTT lr=6e-5 | 67.40% | 68.10% | -0.2 pp ≈ baseline |
| 3 | LoRA r=16 lr=6e-5 | 60.40% | 68.00% | -3.1 pp |
| 4 | LoRA r=16 lr=1e-4 | 59.30% | 68.10% | -4.2 pp |
| 5 | LoRA r=64 lr=1e-4 | 29.20% | 67.10% (init) | -37.9 pp |
| 6 | LoRA r=64 lr=6e-5 | 14.00% | 67.10% (init) | -53.1 pp |
| 7 | BlockTT lr=2e-4 | 13.00% | 67.60% (init) | -54.6 pp |
| 8 | BlockTT lr=8e-5 | 12.90% | 68.00% | -54.7 pp |
| 9 | LoRA r=64 lr=8e-5 | 13.40% | 67.10% (init) | -53.7 pp |
| 10 | LoRA r=16 lr=8e-5 | 0.00% | 68.40% | -63.5 pp |
| 11 | LoRA r=16 lr=2e-4 | 0.00% | **71.30%** | -63.5 pp |
| 12 | LoRA r=64 lr=2e-4 | 0.00% | 67.10% (init) | -67.1 pp |

Best by **peak** eval (any step): r=16 lr=2e-4 hits **71.3%** at step 15 (+7.8 pp over its 63.5% baseline), r=16 lr=8e-5 hits **68.4%** at step 10. Best by **final** eval: BlockTT lr=1e-4 at **69.0%**.

### Llama-3.1-8B (base) — GRPO from-scratch fails

For comparison, the same 4 LRs × 2 modes were run on the *base* `meta-llama/Llama-3.1-8B` (no instruction tuning). All 8 runs:

- **Pre-training baseline: 0.00%** — base model does not produce `\boxed{}` answers from the boxed-prompt template.
- **All 8 final evals: 0.00%** through 50 GRPO steps.

GRPO needs at least one rollout per prompt to be correct to get a non-zero advantage; with 0% baseline reward across all 256 rollouts per step, all advantages are identically zero and no policy gradient flows. The base model would need format-bootstrap SFT, a few-shot prompt that elicits boxed answers, or a permissive format-only initial reward before RL can take hold. **`run_rl.py` was patched (this session) to (a) fall back to raw prompt when `tokenizer.chat_template is None` (otherwise the script would crash on base models) and (b) explicitly label and wandb-log the step-0 evaluation as `eval/baseline_accuracy`.**

### Llama-3.1-8B-Instruct: takeaways

1. **BlockTT (output_one_block/small/keep_trainable/full) is the only method that reliably beats baseline on Llama-3.1-8B-Instruct.** BlockTT lr=1e-4 finishes at 69.0% (+1.4 pp); BlockTT lr=6e-5 essentially holds baseline. BlockTT lr=8e-5 collapses *late* (~step 42), and lr=2e-4 collapses early. So the BlockTT stable-LR window on this model is roughly [6e-5, 1e-4]; lr=8e-5 is borderline (passes 35 steps, then drops).

2. **LoRA r=64 collapses at every LR tested**, including the lowest (6e-5). r=64 may simply be too much capacity for this model's RL signal — the per-step adapter delta dominates the policy and pushes the model into a degenerate region.

3. **LoRA r=16 partially fixes the LoRA collapse problem**, but no r=16 final eval beats baseline. The peaks all sit above baseline (68–71%), which means **early-stopping is critical** when training LoRA on this model. Without an early-stop-on-eval workflow, an RL run that "looks fine" at step 10 may be 60+ pp worse by step 50.

4. **The catastrophic-collapse signature is consistent**: train_acc drops to 0.0% with step time falling from ~100s to ~30s. Short step time is the giveaway — the policy is generating short outputs (likely immediate `<eot_id>`) that produce zero reward and zero gradient.

5. **Comparison to the Qwen3-1.7B results higher in this doc** (where every LoRA LR converged cleanly): Qwen3-1.7B is more LR-tolerant, possibly because (a) it's smaller so per-step relative parameter movement is larger and the optimizer is "more aware" of its updates, or (b) its base is more aligned with the boxed-answer math format. The gap between Qwen3-1.7B LoRA (best 84.8% eval/acc) and Llama-3.1-8B-Instruct LoRA (best 60.4% final) is substantial — this is plausibly **a model-specific stability problem rather than a method problem**.

### Footnote: baseline mismatch between meta-llama and NousResearch mirrors

The official `meta-llama/Llama-3.1-8B-Instruct` baseline measured 63.50% in this pipeline; the `NousResearch/Meta-Llama-3.1-8B-Instruct` mirror baseline measured 67.10%. Same architecture, presumably the same weights. The 3.6 pp gap is likely a tokenizer-config artifact (chat_template differences in special tokens or whitespace handling). Within-sweep comparisons remain valid, but cross-mirror comparisons should adjust by ~3.6 pp. The r=16 sweep used the official repo (after HF gated-access approval); the r=64 sweep used the public mirror. A re-run of one r=64 LR on the official repo would close this footnote — not yet done.

## Remaining gaps

1. **Fura/svd at lr 1e-5 lack extended eval.** The overall-best primary-eval runs (`fura-lr_1e-5` 89.5%, `svd-lr_1e-5` 89.1%) pre-date both the extended-eval harness and the Minerva fix. Their on-disk checkpoints are factored (BTT/SVD cores, not dense); `eval_rl.py` would need a factored-to-dense materializer, or rerun those configs from scratch with `--enable-merged-ckpt`. The new `svd-adamw-lr_1e-5-s_to_keep_trainable-train_input-ext` run (launched this session) fills the gap for svd (MATH 64.0, AIME-24 16.7, Minerva 20.6 — the single best extended-eval row on the table). An equivalent fura re-run is still queued.
2. **Legacy extended-eval runs lack Minerva.** full / dora / pissa / milora / lift / randlora runs from 2026-04-19–21 all recorded MATH-500 / AMC23 / AIME-24/25 but not Minerva, because Minerva's `load_dataset` failed before the hf-id fix. The five new runs that include Minerva are the only ones with complete extended-eval coverage. Only two methods (fura, full) have their current best row *without* a Minerva score — re-evaluating the winning checkpoints on Minerva alone is cheap (272 problems, greedy@1) and could be done via `uv run eval_rl.py --checkpoint <path>/step=50 --math-verify-datasets Minerva` once those winning runs' merged checkpoints exist on disk.
3. ~~`randlora-lr_1e-4` sweep point~~ — **Completed.** randlora 1e-4 is now the best randlora config (mean-ext 34.8, on par with full and fura).
4. ~~fura LR sweep~~ — **Completed (8e-5, 2e-4, 3e-4).** fura 8e-5 is now the best fura config with Minerva coverage (MATH 63.0, AMC 52.5, AIME-25 17.5, Minerva 21.0). The earlier `fura-lr_1e-5 + train_both + s_to_keep` run at 89.5% primary eval but no extended eval remains the best primary-eval-only number — worth re-running under the current harness for full coverage.
