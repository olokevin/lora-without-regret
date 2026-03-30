import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from argparse import ArgumentParser
import subprocess
from pathlib import Path

from blocktt_utils import (
    apply_blocktt_from_metadata,
    load_blocktt_metadata,
    materialize_blocktt_to_linear_,
)


def load_checkpoint_state_dict(checkpoint_dir: Path):
    safetensors_path = checkpoint_dir / "model.safetensors"
    if safetensors_path.exists():
        try:
            from safetensors.torch import load_file as load_safetensors_file
        except ImportError as exc:
            raise ImportError(
                "Found model.safetensors but safetensors is not installed."
            ) from exc
        return load_safetensors_file(str(safetensors_path))

    bin_path = checkpoint_dir / "pytorch_model.bin"
    if bin_path.exists():
        return torch.load(bin_path, map_location="cpu")

    raise FileNotFoundError(
        f"Could not find model.safetensors or pytorch_model.bin in {checkpoint_dir}"
    )


device = "cuda:0"
parser = ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="full",
    choices=["full", "lora", "spectral", "blocktt"],
)
parser.add_argument("--checkpoint", type=int, default=100)
args = parser.parse_args()
if args.model == "full":
    print(f"TEST FULL TUNING checkpoint {args.checkpoint}")
    model = AutoModelForCausalLM.from_pretrained(
        f"llama3-full/checkpoint-{args.checkpoint}",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(f"full-{args.checkpoint}/")
    model.save_pretrained(f"full-{args.checkpoint}")
    subprocess.call(["lm_eval", "--model_args", f"pretrained=full-{args.checkpoint}", "--tasks", "gsm8k", "--device", device, "--batch_size", "8"])
elif args.model == "lora":
    print(f"TEST LORA TUNING checkpoint {args.checkpoint}")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(f"lora-{args.checkpoint}/")
    model_to_merge = PeftModel.from_pretrained(model, f"llama3-lora/checkpoint-{args.checkpoint}")
    merged_model = model_to_merge.merge_and_unload()
    merged_model.save_pretrained(f"lora-{args.checkpoint}")
    subprocess.call(["lm_eval", "--model_args", f"pretrained=lora-{args.checkpoint}", "--tasks", "gsm8k", "--device", device, "--batch_size", "8"])
elif args.model == "spectral":
    print(f"TEST SPECTRAL TUNING checkpoint {args.checkpoint}")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(f"spectral-{args.checkpoint}/")
    model_to_merge = PeftModel.from_pretrained(model, f"llama3-spectral/checkpoint-{args.checkpoint}")
    merged_model = model_to_merge.merge_and_unload()
    merged_model.save_pretrained(f"spectral-{args.checkpoint}")
    subprocess.call(["lm_eval", "--model_args", f"pretrained=spectral-{args.checkpoint}", "--tasks", "gsm8k", "--device", device, "--batch_size", "8"])
elif args.model == "blocktt":
    print(f"TEST BLOCKTT TUNING checkpoint {args.checkpoint}")
    checkpoint_dir = Path(f"llama3-blocktt/checkpoint-{args.checkpoint}")
    blocktt_metadata = load_blocktt_metadata(checkpoint_dir)
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(f"blocktt-{args.checkpoint}/")

    apply_blocktt_from_metadata(model, blocktt_metadata, set_trainability=False)
    state_dict = load_checkpoint_state_dict(checkpoint_dir)
    model.load_state_dict(state_dict, strict=True)
    dense_count = materialize_blocktt_to_linear_(model)
    print(f"Materialized {dense_count} BlockTT layers into dense Linear layers")
    model.save_pretrained(f"blocktt-{args.checkpoint}")
    subprocess.call(["lm_eval", "--model_args", f"pretrained=blocktt-{args.checkpoint}", "--tasks", "gsm8k", "--device", device, "--batch_size", "8"])
