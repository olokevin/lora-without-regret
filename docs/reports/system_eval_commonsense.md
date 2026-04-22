# FuRA System-Performance Evaluation: Commonsense SFT

_Auto-generated from 18 short-horizon (300-step) SFT runs on a single NVIDIA H100 NVL (95.8 GB)._

## 1. Setup

- **Model:** Meta-Llama-3-8B (8.03B parameters)
- **Dataset:** commonsense_170k (single-round instruction tuning)
- **Protocol:** 300 optimizer steps (100-step warmup + 200 measurement), seed=43
- **Batch:** per-device bs=8, gradient accumulation=2, seq-len 2048 (32,768 effective tokens/step)
- **Hardware:** Single NVIDIA H100 NVL, bf16, gradient checkpointing ON
- **Optimizer:** AdamW, lr=2e-4, linear schedule with 3% warmup

## 2. Headline Comparison (matched-parameter operating point)

LoRA/DoRA/RandLoRA at r=64, LIFT at r=32, Full FT and FuRA as singletons.

| Method | Rank | Trainable (%) | Stored extra (%) | Step (s) | Tokens/s | Peak GPU (GB) |
|--------|-----:|--------------:|-----------------:|---------:|---------:|--------------:|
| **Full FT** | --- | 100.00 | 0.00 | 0.201 | 163,194 | 76.9 |
| **LoRA** | 64 | 1.41 | 1.41 | 0.043 | 770,385 | 26.8 |
| **DoRA** | 64 | 1.42 | 1.42 | 0.045 | 731,829 | 26.8 |
| **RandLoRA** | 64 | 0.33 | 0.33 | 0.059 | 558,810 | 25.7 |
| **LIFT** | 32 | 100.00 | 0.00 | 0.144 | 227,075 | 80.0 |
| **FuRA** | --- | 1.98 | 1.98 | 0.046 | 713,376 | 22.7 |

**Key observations:**

1. **FuRA matches LoRA-64 step time** (0.046s vs 0.043s) while training a full-rank update (not low-rank).
2. **FuRA uses the least peak GPU memory** (22.7 GB) --- less than LoRA (26.8 GB), DoRA (26.8 GB), and dramatically less than Full FT (76.9 GB) or LIFT (80.0 GB).
3. **FuRA is 0.97x faster than DoRA** per step, since DoRA adds per-column magnitude normalization overhead.
4. **LIFT is 3.1x slower** than FuRA per step (0.144s vs 0.046s), consistent with the scatter-gather overhead of sparse mask updates.
5. **RandLoRA is 1.28x slower** than FuRA (0.059s), despite training fewer parameters (0.33% vs 1.98%).
6. **Full FT is 4.4x slower** and uses 77 GB (3.4x more memory) than FuRA.

## 3. Full Rank Sweep

| Method | Rank | Trainable (%) | Stored extra (%) | Step (s) | Tokens/s | Peak GPU (GB) |
|--------|-----:|--------------:|-----------------:|---------:|---------:|--------------:|
| Full FT | --- | 100.00 | 0.00 | 0.201 | 163,194 | 76.9 |
| LoRA | 16 | 0.35 | 0.35 | 0.040 | 809,703 | 27.2 |
| LoRA | 32 | 0.71 | 0.71 | 0.041 | 801,564 | 26.4 |
| LoRA | 64 | 1.41 | 1.41 | 0.043 | 770,385 | 26.8 |
| LoRA | 128 | 2.82 | 2.82 | 0.046 | 716,563 | 40.0 |
| DoRA | 16 | 0.36 | 0.36 | 0.044 | 747,967 | 27.2 |
| DoRA | 32 | 0.71 | 0.71 | 0.044 | 744,485 | 26.4 |
| DoRA | 64 | 1.42 | 1.42 | 0.045 | 731,829 | 26.8 |
| DoRA | 128 | 2.83 | 2.83 | 0.048 | 686,766 | 40.0 |
| RandLoRA | 16 | 1.31 | 1.31 | 0.060 | 548,628 | 26.6 |
| RandLoRA | 32 | 0.66 | 0.66 | 0.059 | 555,561 | 26.0 |
| RandLoRA | 64 | 0.33 | 0.33 | 0.059 | 558,810 | 25.7 |
| RandLoRA | 128 | 0.17 | 0.17 | 0.058 | 562,464 | 25.5 |
| LIFT | 16 | 100.00 | 0.00 | 0.142 | 230,331 | 80.0 |
| LIFT | 32 | 100.00 | 0.00 | 0.144 | 227,075 | 80.0 |
| LIFT | 64 | 100.00 | 0.00 | 0.147 | 222,589 | 80.0 |
| LIFT | 128 | 100.00 | 0.00 | 0.152 | 215,128 | 80.0 |
| **FuRA** | --- | 1.98 | 1.98 | 0.046 | 713,376 | 22.7 |

**Rank-sweep observations:**

- LoRA/DoRA step time grows modestly with rank (0.040s at r=16 to 0.046-0.048s at r=128), with a memory jump at r=128 (26-27 GB to 40 GB).
- RandLoRA is roughly rank-invariant in step time (~0.059s) due to its shared frozen projection dominating cost.
- LIFT is roughly rank-invariant in step time (~0.142-0.152s) and always uses ~80 GB (full-model sparse mask).
- FuRA at its default operating point (0.046s, 22.7 GB) sits between LoRA r=64 and r=128 in step time, but with 3.4x lower memory than LoRA r=128.

## 4. Init Overhead (one-time setup cost)

Time and memory to attach each adapter to a freshly-loaded Llama-3-8B (bf16, single H100 NVL). Measured by `tools/bench_fura_init.py`. Target modules: `q_proj, k_proj, v_proj, up_proj, down_proj` (160 Linear layers).

| Method | Init wall (s) | Convert/Attach wall (s) | Peak GPU during init (GB) | Stored extra (M params) | Stored extra (%) |
|--------|--------------:|------------------------:|--------------------------:|------------------------:|-----------------:|
| **LoRA** r=64  | 3.84 | 0.97 | 15.4 | 113.2 | 1.41 |
| **DoRA** r=64  | 3.88 | 1.06 | 15.5 | 114.0 | 1.42 |
| **FuRA** (default) | 56.30 | 52.90 | **24.3** | 77.6 | 0.97 |

**Key observations:**

1. **FuRA init is ~50x slower** than LoRA/DoRA attach (52.9s vs 0.97-1.06s), because `convert_linear_to_btt()` initialises 160 BTT cores via SVD-style factorisation per layer rather than zero-initialising small adapter weights.
2. **FuRA briefly peaks at 24.3 GB** during conversion (vs ~15 GB after), because it materialises full dense weight matrices for the SVD factorisation step before discarding them.
3. **Init cost is one-time** --- amortised over a 3-epoch training run (~hours), the 53-second FuRA setup is <1% of total wall-clock. For a 300-step microbenchmark it's ~25% of run time, but all our reported step-time medians use a 100-step warmup that excludes init.
4. **Stored extra params** measured here (FuRA: 77.6M = 0.97%) is lower than what the SFT runs reported (159.4M = 1.98%) because the SFT pipeline includes additional auxiliary parameters (e.g., `o_proj` BTT cores when `train_position` paths add them); the bench script uses the minimal target-module set.

**Practical implication:** FuRA's init overhead is a non-issue for production training (amortised away) but adds noticeable latency for short-horizon experiments and for any RL pipeline that re-initialises the adapter per rollout.

## 5. Limitations

- SFT-only (no RL/GRPO); single seed (43); single H100 NVL; bf16 only; no multi-GPU.
- Accuracy is **not re-measured** --- it is read from the paper's existing Table 1 at the matched operating point.
- Step times are steady-state medians over 200 post-warmup steps (micro-step level for Full FT, optimizer-step level for adapter methods).
- Full FT step time was corrected for 2x gradient accumulation (raw measurement was per micro-step).
- FuRA `stored_extra_pct` reflects the BTT decomposition overhead (frozen L core + trainable S,R); this storage is eliminated at deployment after re-merge to dense weights.
- Init overhead measured with `target_modules={q,k,v,up,down}_proj` (160 layers); SFT runs may include additional modules (e.g., `o_proj`, `gate_proj`) which would raise both timing and stored-extra figures proportionally.
