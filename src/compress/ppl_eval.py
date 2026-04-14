"""Perplexity evaluation utilities for compressed models."""

from __future__ import annotations

import math
import random
from typing import Iterable

import torch


def _get_c4_valenc(tokenizer, seqlen: int = 2048, seed: int = 0) -> torch.Tensor:
    from datasets import load_dataset

    valdata = load_dataset(
        "allenai/c4", data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"}, split="validation"
    )
    random.seed(seed)
    valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
    return valenc.input_ids[:, : (256 * seqlen)]


def _get_wikitext2_testenc(tokenizer) -> torch.Tensor:
    from datasets import load_dataset

    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
    return testenc.input_ids


def eval_ppl(model, input_ids: torch.Tensor, seqlen: int, device: str) -> float:
    """Sliding-window perplexity over a flat token tensor of shape [1, N]."""
    input_ids = input_ids.to(device)
    nsamples = input_ids.shape[1] // seqlen
    if nsamples <= 0:
        raise ValueError(f"Not enough tokens ({input_ids.shape[1]}) for seqlen={seqlen}")

    nlls = []
    for i in range(nsamples):
        chunk = input_ids[:, i * seqlen : (i + 1) * seqlen]
        with torch.no_grad():
            loss = model(chunk, labels=chunk).loss
        nlls.append(loss.item())
    return math.exp(sum(nlls) / len(nlls))


def evaluate_model_ppl(
    model,
    tokenizer,
    *,
    seqlen: int = 2048,
    seed: int = 0,
    datasets: Iterable[str] = ("wikitext2", "c4"),
    device: str | None = None,
) -> dict[str, float]:
    """Compute PPL on selected datasets and return {dataset_name: ppl}."""
    if device is None:
        if next(model.parameters()).is_cuda:
            device = "cuda"
        else:
            device = "cpu"

    model_was_training = model.training
    model.eval()
    results: dict[str, float] = {}
    try:
        for dataset in datasets:
            if dataset == "wikitext2":
                input_ids = _get_wikitext2_testenc(tokenizer)
            elif dataset == "c4":
                input_ids = _get_c4_valenc(tokenizer, seqlen=seqlen, seed=seed)
            else:
                raise ValueError(f"Unknown PPL dataset: {dataset}. Choose from: wikitext2, c4")
            results[dataset] = eval_ppl(model, input_ids, seqlen, device)
    finally:
        if model_was_training:
            model.train()
    return results

