"""Hand-rolled fast DoRA-on-Linear4bit module.

PEFT's DoRA path on a 4-bit quantized base does:
  weight_dq = dequantize_4bit(W_q)              # full bf16 dequant
  base_result = F.linear(x, weight_dq)          # bf16 matmul (NOT fused)
  weight_norm = ||weight_dq + scaling * B@A||_c # column norm, recomputed every step
  ...

This is ~2x slower than vanilla QLoRA per step because the dequant+matmul is
not fused (PEFT calls F.linear on the dequanted bf16 tensor instead of bnb's
fused matmul_4bit kernel).

This module replaces that path with:
  base_result = bnb.matmul_4bit(x, W_q, quant_state)  # fused dequant+matmul
  weight_norm = (cached, recomputed every K steps)

The DoRA forward formula is unchanged from PEFT; only the implementation of
`base_result` and the cadence of `weight_norm` recomputation differ.

Usage
-----
After loading a 4-bit base model + calling prepare_model_for_kbit_training,
call convert_to_qdora_fast(model, target_modules, r, alpha, ...) to swap
each targeted Linear4bit for a Qdora4bitLinear. Skip PEFT's get_peft_model.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

import bitsandbytes as bnb


class Qdora4bitLinear(nn.Module):
    """DoRA on top of a frozen NF4 Linear4bit base, with fused matmul.

    Trainable parameters:
      - lora_A   : (r, in_features),  bf16
      - lora_B   : (out_features, r), bf16
      - magnitude: (out_features,),    bf16  -- per-column DoRA scale

    Frozen parameters:
      - base_layer: bnb.nn.Linear4bit (NF4 weight + quant_state)

    Cached buffers (no grad):
      - norm_cache  : (out_features,) bf16, ||W + scaling * B@A||_c
      - step_counter: int64 scalar, increments every forward in train mode

    Forward (training mode):
      base_out  = bnb.matmul_4bit(x, W_q, quant_state)
      lora_out  = (B @ A @ x) * scaling
      if step_counter % norm_cache_steps == 0:
          recompute norm_cache from dequanted base + scaling*B@A
      mag_norm_scale = (magnitude / norm_cache).view(1, -1)
      out = mag_norm_scale * (base_out + lora_out)

    Forward (eval mode):
      same, but no cache update; uses the last training-time cache.

    DoRA paper Section 4.3: column norm is treated as a constant (detached
    from the gradient graph). magnitude receives gradients via the division.
    """

    def __init__(
        self,
        base_layer: bnb.nn.Linear4bit,
        r: int,
        lora_alpha: int,
        lora_dropout: float = 0.0,
        norm_cache_steps: int = 16,
    ):
        super().__init__()
        if not isinstance(base_layer, bnb.nn.Linear4bit):
            raise TypeError(
                f"Qdora4bitLinear expects bnb.nn.Linear4bit base, got {type(base_layer).__name__}"
            )
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.norm_cache_steps = norm_cache_steps

        # Freeze the base.
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        # LoRA matrices. Use Linear modules to match PEFT's naming/conventions
        # and to make per-module replacement straightforward at merge time.
        device = self.base_layer.weight.device
        # Match PEFT's default init for LoRA: Kaiming uniform on A, zeros on B.
        # See peft.tuners.lora.layer.LoraLayer.update_layer.
        self.lora_A = nn.Linear(self.in_features, r, bias=False, device=device, dtype=torch.bfloat16)
        self.lora_B = nn.Linear(r, self.out_features, bias=False, device=device, dtype=torch.bfloat16)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self.lora_dropout = (
            nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        )

        # Magnitude: initialized from ||W||_c of the original (un-LoRA-adjusted)
        # weight. PEFT does this in DoraLinearLayer.update_layer.
        with torch.no_grad():
            weight_dq = bnb.functional.dequantize_4bit(
                self.base_layer.weight.data,
                quant_state=self.base_layer.weight.quant_state,
            ).to(torch.bfloat16)
            # weight_dq shape: (out, in). column-wise norm = norm along dim=1.
            initial_norm = torch.linalg.norm(weight_dq, dim=1).to(torch.bfloat16)
        self.magnitude = nn.Parameter(initial_norm.clone(), requires_grad=True)

        # Norm cache: same shape as magnitude. Initialized to the current norm
        # (since LoRA-B is zero, the merged norm equals the base norm).
        self.register_buffer("norm_cache", initial_norm.clone())
        self.register_buffer(
            "step_counter", torch.zeros((), dtype=torch.int64, device=device)
        )

    @torch.no_grad()
    def _recompute_norm_cache(self):
        """Refresh self.norm_cache to match ||W + scaling * B@A||_c."""
        weight_dq = bnb.functional.dequantize_4bit(
            self.base_layer.weight.data,
            quant_state=self.base_layer.weight.quant_state,
        ).to(self.lora_A.weight.dtype)
        # lora_delta shape: (out, in)
        lora_delta = self.lora_B.weight @ self.lora_A.weight
        merged = weight_dq + self.scaling * lora_delta
        self.norm_cache.copy_(torch.linalg.norm(merged, dim=1).to(self.norm_cache.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Fused dequant+matmul for the frozen base.
        base_result = bnb.matmul_4bit(
            x,
            self.base_layer.weight.t(),
            self.base_layer.weight.quant_state,
            bias=self.base_layer.bias,
        )

        # 2. Standard LoRA forward.
        x_dropped = self.lora_dropout(x)
        lora_result = self.lora_B(self.lora_A(x_dropped)) * self.scaling

        # 3. Column-norm cache: refresh every K steps in training mode.
        if self.training:
            if (self.step_counter.item() % self.norm_cache_steps) == 0:
                self._recompute_norm_cache()
            self.step_counter += 1

        # 4. DoRA scale. Note: norm_cache is a buffer (no grad). magnitude is
        #    trainable. Division gives gradient only to magnitude.
        mag_norm_scale = (self.magnitude / self.norm_cache).view(1, -1)

        # 5. DoRA combination. PEFT's full formula computes:
        #      out = base_result + (mag_norm_scale - 1)*base_result + mag_norm_scale*lora_result
        #          = mag_norm_scale * (base_result + lora_result)
        # which matches the DoRA paper Eq. 5: out = (m / ||W+B@A||_c) * (W+B@A) @ x.
        # Our `lora_result` already includes `scaling` (`* self.scaling` above).
        out = mag_norm_scale * (base_result + lora_result)
        return out

    @torch.no_grad()
    def merge_to_dense_bf16(self) -> nn.Linear:
        """Materialize a single bf16 nn.Linear equivalent to this DoRA layer.

        Used at end-of-training to produce a checkpoint loadable by the eval
        scripts (which expect a plain HF model).

        Output linear weight = mag_norm_scale * (W + scaling * B @ A), where
        mag_norm_scale = magnitude / ||W + scaling * B @ A||_c.
        """
        weight_dq = bnb.functional.dequantize_4bit(
            self.base_layer.weight.data,
            quant_state=self.base_layer.weight.quant_state,
        ).to(torch.bfloat16)
        lora_delta = self.lora_B.weight @ self.lora_A.weight  # (out, in)
        merged = weight_dq + self.scaling * lora_delta
        weight_norm = torch.linalg.norm(merged, dim=1).to(torch.bfloat16)
        scale = (self.magnitude / weight_norm).view(-1, 1)  # (out, 1)
        new_weight = scale * merged

        out = nn.Linear(
            self.in_features,
            self.out_features,
            bias=(self.base_layer.bias is not None),
            device=new_weight.device,
            dtype=torch.bfloat16,
        )
        out.weight.data.copy_(new_weight)
        if self.base_layer.bias is not None:
            out.bias.data.copy_(self.base_layer.bias.data.to(torch.bfloat16))
        return out


def convert_to_qdora_fast(
    model: nn.Module,
    target_module_names: Iterable[str],
    r: int,
    lora_alpha: int,
    lora_dropout: float = 0.0,
    norm_cache_steps: int = 16,
) -> dict:
    """Walk model, replace each Linear4bit whose leaf name matches a target
    module name with a Qdora4bitLinear. In place.

    target_module_names: e.g. ("q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj").

    Returns: {"num_converted": int, "names": [full_module_name]}.

    Pattern: replicate PEFT's behavior of only replacing modules whose leaf
    matches a name in target_module_names (the last segment of the dotted
    full path).
    """
    target_set = set(target_module_names)
    replacements = []
    for full_name, module in model.named_modules():
        if not isinstance(module, bnb.nn.Linear4bit):
            continue
        leaf = full_name.split(".")[-1]
        if leaf not in target_set:
            continue
        replacements.append((full_name, module))

    for full_name, module in replacements:
        new_module = Qdora4bitLinear(
            base_layer=module,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            norm_cache_steps=norm_cache_steps,
        )
        # Navigate to parent and replace.
        path = full_name.split(".")
        parent = model
        for key in path[:-1]:
            parent = getattr(parent, key)
        setattr(parent, path[-1], new_module)

    return {
        "num_converted": len(replacements),
        "names": [n for n, _ in replacements],
    }


@torch.no_grad()
def materialize_qdora_to_linear(model: nn.Module) -> int:
    """Replace every Qdora4bitLinear with its dense bf16 nn.Linear merge.

    Used by the merge tool at end-of-training so the saved checkpoint is a
    plain HF model.
    """
    replacements = []
    for full_name, module in model.named_modules():
        if isinstance(module, Qdora4bitLinear):
            replacements.append((full_name, module))

    for full_name, module in replacements:
        merged = module.merge_to_dense_bf16()
        path = full_name.split(".")
        parent = model
        for key in path[:-1]:
            parent = getattr(parent, key)
        setattr(parent, path[-1], merged)

    return len(replacements)
