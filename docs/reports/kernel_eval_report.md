# FuRA Fused-Step2 Kernel: Evaluation Report
_Auto-generated on 2026-04-21T20:49:08+00:00 by tools/write_kernel_report.py_

## Setup
- GPU: `NVIDIA H100 NVL (95830 MiB)`
- Model: Llama-3-8B, FuRA default corner (decomp=input_one_block, train=small, s=frozen)
- bf16, gradient checkpointing ON, bs=8 × accum=2, seq-len 2048, `max_steps=300`

## Correctness
See `tests/test_fura_fused_kernel.py`; shape grid + fp32 gradcheck + non-contiguous coverage.

## Microbenchmark (forward pass, per-layer)

| Shape | B | Baseline (µs) | Fused (µs) | Speedup | Mem base (MB) | Mem fused (MB) |
|-------|--:|---------------:|------------:|--------:|---------------:|----------------:|
| llama3_qproj | 1024 | 1568.3 | 1961.6 | 0.80× | 1152.0 | 1152.0 |
| llama3_qproj | 2048 | 3244.5 | 4057.4 | 0.80× | 2208.0 | 2208.0 |
| llama3_qproj | 4096 | 6545.9 | 8031.0 | 0.81× | 4320.0 | 4320.0 |
| llama3_qproj | 8192 | 13171.2 | 16157.6 | 0.81× | 8544.0 | 8544.0 |
| llama3_kproj | 1024 | 807.2 | 859.7 | 0.94× | 588.0 | 588.0 |
| llama3_kproj | 2048 | 1538.8 | 1750.5 | 0.88× | 1120.0 | 1120.0 |
| llama3_kproj | 4096 | 3122.4 | 3457.1 | 0.90× | 2184.0 | 2184.0 |
| llama3_kproj | 8192 | 6437.8 | 6887.4 | 0.94× | 4312.0 | 4312.0 |
| llama3_vproj | 1024 | 806.9 | 885.2 | 0.91× | 588.0 | 588.0 |
| llama3_vproj | 2048 | 1544.8 | 1743.0 | 0.89× | 1120.0 | 1120.0 |
| llama3_vproj | 4096 | 3226.4 | 3462.4 | 0.93× | 2184.0 | 2184.0 |
| llama3_vproj | 8192 | 6513.2 | 6913.9 | 0.94× | 4312.0 | 4312.0 |
| llama3_upproj | 1024 | 3056.9 | 4582.5 | 0.67× | 2064.0 | 2064.0 |
| llama3_upproj | 2048 | 6101.8 | 9094.0 | 0.67× | 3928.0 | 3928.0 |
| llama3_upproj | 4096 | 12186.3 | 18568.0 | 0.66× | 7656.0 | 7656.0 |

**Geometric-mean speedup across all shape × batch cells: 0.83×**

## End-to-end SFT (300 optimizer steps)

| Run | Median step (s) | Tokens/s | Peak GPU (GB) | Total wall (min) |
|-----|-----------------:|---------:|--------------:|------------------:|
| Baseline | 0.000 | 0 | 0.0 | 0.0 |
| Fused | 0.000 | 0 | 0.0 | 0.0 |

## Verdict

**V1 kernel status: NOT READY for production.**

The V1 Triton Step-2 kernel is **slower** than PyTorch's native `torch.bmm` across all tested shapes (geometric-mean speedup 0.83×). Root causes:
1. The Triton kernel does not exploit shared memory or persistent-CTA grouping — each tile reads L from HBM.
2. For large output dimensions (up_proj: 4096→14336), the kernel hits CUDA illegal-memory-access errors at B≥8192, crashing the process.
3. No memory saving observed — both paths allocate the same peak memory since Step 1 (torch.bmm) dominates.

**Next steps (V2):** Fuse both Step 1 and Step 2 into a single persistent-CTA grouped GEMM with shared-memory L tiles. The V1 kernel validated the API/autograd plumbing; the actual GEMM needs a rewrite.

**End-to-end SFT comparison was skipped** because the V1 kernel is slower per-layer and crashes on up_proj/down_proj shapes used by Llama-3-8B.

## Original verdict
Insufficient data to form a verdict.

## Raw artifacts
- Microbench JSON, baseline `sys_metrics.json`, fused `sys_metrics.json` (see runner out dir).