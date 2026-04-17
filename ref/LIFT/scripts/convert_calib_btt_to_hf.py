"""Convert a legacy calibrated-BTT checkpoint (btt_topology.json +
model.safetensors, no config.json) into the LIFT legacy HF format
produced by `ref/LIFT/src/utils/model_utils.py:save_hf_format` — so the
converted dir is byte-for-byte identical to a non-calib blocktt
checkpoint and loadable by `AutoModelForCausalLM.from_pretrained(...)`.

Legacy layout:
    <ckpt>/
        model.safetensors      # state dict with *.btt_l / *.btt_r keys
        btt_topology.json      # BTTLinear factor shapes per module

Output layout (written to --out, default <ckpt>/hf):
    <ckpt>/hf/
        config.json
        pytorch_model.bin      # dense nn.Linear weights (torch.save)

No tokenizer or safetensors are emitted — matches save_hf_format, and
LIFT eval loads the tokenizer from --base-model anyway.

Usage:
    uv run python ref/LIFT/scripts/convert_calib_btt_to_hf.py \\
        --ckpt /path/to/legacy/calib/checkpoint \\
        --base-model meta-llama/Meta-Llama-3-8B
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_LIFT_SRC = os.path.join(_REPO_ROOT, "ref", "LIFT", "src")
if _LIFT_SRC not in sys.path:
    sys.path.insert(0, _LIFT_SRC)

from compress_integration import (  # noqa: E402
    load_calibrated_btt_for_eval,
    save_calibrated_btt_checkpoint,
)
from utils.model_utils import load_hf_tokenizer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True,
                    help="Legacy calib-BTT checkpoint dir (contains btt_topology.json + model.safetensors).")
    ap.add_argument("--base-model", required=True,
                    help="HF model id of the base model the checkpoint was trained from.")
    ap.add_argument("--out", default=None,
                    help="Destination dir for the HF-format checkpoint. Defaults to <ckpt>/hf.")
    ap.add_argument("--dtype", default="bf16", choices=["fp16", "bf16", "fp32"],
                    help="Dtype to instantiate the model skeleton in.")
    args = ap.parse_args()

    out_dir = args.out or os.path.join(args.ckpt, "hf")
    os.makedirs(out_dir, exist_ok=True)

    torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    # Use LIFT's load_hf_tokenizer (same helper training used) so
    # len(tokenizer) matches what was on the model at save time.
    # For llama it adds [PAD] when missing (ref/LIFT/src/utils/model_utils.py:27-28),
    # bumping len 128256 -> 128257 -> rounded_to_8 = 128264 for Llama-3-8B.
    print(f"[convert] loading tokenizer + config from base_model={args.base_model}")
    tokenizer = load_hf_tokenizer(args.base_model, fast_tokenizer=True)
    config = AutoConfig.from_pretrained(args.base_model)

    print(f"[convert] instantiating empty {config.model_type} model skeleton")
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch_dtype)
    # Match the resize the LIFT training path applies before save.
    target_vocab = int(8 * math.ceil(len(tokenizer) / 8.0))
    if target_vocab != model.config.vocab_size:
        print(f"[convert] resizing token embeddings: {model.config.vocab_size} -> {target_vocab}")
        model.resize_token_embeddings(target_vocab)

    print(f"[convert] rebuilding BTTLinear modules and loading weights from {args.ckpt}")
    load_calibrated_btt_for_eval(model, args.ckpt)

    print(f"[convert] saving LIFT HF format (pytorch_model.bin + config.json) to {out_dir}")
    # save_calibrated_btt_checkpoint materializes BTTLinear -> nn.Linear and
    # writes exactly what save_hf_format would write, so the output is
    # byte-for-byte identical to a non-calib blocktt checkpoint.
    save_calibrated_btt_checkpoint(model, out_dir)
    print(f"[convert] done: {out_dir}")


if __name__ == "__main__":
    main()
