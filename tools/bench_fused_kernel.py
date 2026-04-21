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
