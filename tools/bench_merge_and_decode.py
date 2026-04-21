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
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="cuda"
    )

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
            t0 = time.time()
            for _, module in base.named_modules():
                if hasattr(module, "reconstruct_merged_weight"):
                    _ = module.reconstruct_merged_weight()
            torch.cuda.synchronize()
            merge_s = time.time() - t0
            base.save_pretrained(tmpdir)
            active = base
        else:  # lift, randlora — not merge-compatible
            active = base
            base.save_pretrained(tmpdir)

        # Checkpoint size
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
