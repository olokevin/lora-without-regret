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
